"""Verifica della concorrenza (comando ``concorrenza``).

Domanda a cui risponde: quando piu' utenti chiedono insieme, Ollama li
serve in parallelo o in coda, e quanto peggiora la latenza di ciascuno?

Metodo:
  1. una chiamata di riscaldamento (non misurata) carica il modello;
  2. livello 1: richiesta isolata, ripetuta ``--ripetizioni`` volte: la
     mediana e' il riferimento (baseline);
  3. per ogni livello K (es. 2, 4) partono K richieste identiche nello
     stesso istante (barriera tra thread), con lo stesso prompt RAG
     realistico e lunghezza della risposta limitata (``num_predict``), cosi'
     le richieste sono confrontabili;
  4. per ogni richiesta: latenza e rapporto con la baseline.

Classificazione di ogni livello K (rapporto = latenza massima / baseline):
  - "parallelo"              rapporto <= 1,25: le richieste non si ostacolano;
  - "in coda"                rapporto >= 0,85 x K: la richiesta K-esima aspetta
                             le precedenti (latenze a gradini 1x, 2x, ... Kx);
  - "parallelo con contesa"  valori intermedi: le richieste sono servite
                             insieme ma si contendono CPU/GPU e banda di memoria.

Esempio dal progetto originale: isolata ~21,5 s; due richieste insieme
20,2 s e 33,1 s. Rapporto massimo 1,54: ne' 1 (parallelo puro) ne' 2 (coda)
-> parallelismo reale con contesa delle risorse.

Nota su ``OLLAMA_NUM_PARALLEL``: e' una variabile del SERVER Ollama (non
leggibile dall'API) che fissa quante richieste per modello vengono servite
insieme; le altre attendono in coda (lunghezza massima ``OLLAMA_MAX_QUEUE``).
Con valore 1 il risultato atteso e' "in coda". Ogni posto parallelo riserva
memoria per il proprio contesto: aumentarlo fa crescere la memoria occupata
(si vede da ``/api/ps`` prima e dopo). Il valore va deciso con questa
misura, non a intuito, prima di aprire il sistema a molti utenti.
"""

from __future__ import annotations

import statistics
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from clio import risposta as clio_risposta

from . import risultati as R
from .memoria import leggi_memoria
from .ollama import ClienteOllama, ErroreGenerazione, ErroreOllama
from .statistiche import statistiche_tempi

SOGLIA_PARALLELO = 1.25
SOGLIA_CODA = 0.85
NUM_PREDICT_DEFAULT = 256

PARALLELO = "parallelo"
IN_CODA = "in coda"
CONTESA = "parallelo con contesa"
RIFERIMENTO = "riferimento"

# Prompt fisso, costruito con il prompt di Clio su tre passaggi di fantasia
# di una procedura aziendale: lunghezza e forma realistiche per un RAG.
DOMANDA_FISSA = "Entro quanto tempo va segnalato un incidente di sicurezza e a chi?"
CHUNK_FISSI = [
    {"score": 0.71, "payload": {
        "citazione": "PROC-INC, 4.2", "titolo_sezione": "Segnalazione degli incidenti",
        "titolo_documento": "Procedura di gestione degli incidenti di sicurezza",
        "testo": ("Chiunque rilevi un evento che possa compromettere la riservatezza, l'integrita' "
                  "o la disponibilita' delle informazioni lo segnala senza ritardo, e comunque entro "
                  "quattro ore dalla rilevazione, al referente per la sicurezza delle informazioni "
                  "tramite il canale di assistenza interno. La segnalazione indica data e ora "
                  "dell'evento, sistemi coinvolti, prime azioni intraprese e recapito di chi segnala. "
                  "Fuori dall'orario di servizio si utilizza il numero di reperibilita' indicato "
                  "nell'allegato B.")}},
    {"score": 0.64, "payload": {
        "citazione": "PROC-INC, 4.3", "titolo_sezione": "Classificazione e priorita'",
        "titolo_documento": "Procedura di gestione degli incidenti di sicurezza",
        "testo": ("Il referente classifica l'evento entro un giorno lavorativo secondo la matrice "
                  "di impatto (basso, medio, alto, critico). Gli incidenti alti e critici sono "
                  "comunicati alla direzione entro due ore dalla classificazione. Se l'incidente "
                  "riguarda dati personali, il referente coinvolge il responsabile della protezione "
                  "dei dati per valutare l'obbligo di notifica all'autorita' di controllo.")}},
    {"score": 0.58, "payload": {
        "citazione": "PROC-INC, 6.1", "titolo_sezione": "Registro degli incidenti",
        "titolo_documento": "Procedura di gestione degli incidenti di sicurezza",
        "testo": ("Ogni incidente e' registrato con identificativo progressivo, descrizione, "
                  "classificazione, azioni di contenimento, causa individuata e azioni correttive. "
                  "Il registro e' riesaminato ogni trimestre per individuare tendenze e "
                  "aggiornare l'analisi dei rischi.")}},
]


def prompt_fisso() -> str:
    return clio_risposta.costruisci_prompt(DOMANDA_FISSA, CHUNK_FISSI, "A")


@dataclass
class Misura:
    livello: int
    ripetizione: int
    richiesta: int
    latenza: float | None
    timeout: int = 0
    errore: str = ""


@dataclass
class EsitoLivello:
    livello: int
    misure: list[Misura] = field(default_factory=list)
    durata_complessiva: list[float] = field(default_factory=list)
    classificazione: str = ""
    rapporto_max: float | None = None
    rapporto_medio: float | None = None

    @property
    def latenze(self) -> list[float]:
        return [m.latenza for m in self.misure if m.latenza is not None and not m.errore]


def classifica(baseline: float, latenze: list[float], k: int,
               soglia_parallelo: float = SOGLIA_PARALLELO, soglia_coda: float = SOGLIA_CODA) -> str:
    if k <= 1:
        return RIFERIMENTO
    if not latenze or not baseline:
        return "n.d."
    rapporto = max(latenze) / baseline
    if rapporto <= soglia_parallelo:
        return PARALLELO
    if rapporto >= soglia_coda * k:
        return IN_CODA
    return CONTESA


def _una_richiesta(cliente: ClienteOllama, modello: str, endpoint: str, prompt: str,
                   opzioni: dict) -> tuple[float | None, int, str]:
    try:
        if endpoint == "chat":
            g = cliente.chat(modello, [{"role": "user", "content": prompt}], opzioni)
        else:
            g = cliente.genera(modello, prompt, opzioni)
        return g.durata, g.timeout, ""
    except ErroreGenerazione as exc:
        return None, exc.timeout, str(exc)


def misura_livello(cliente: ClienteOllama, modello: str, k: int, ripetizione: int,
                   endpoint: str, prompt: str, opzioni: dict) -> tuple[list[Misura], float]:
    """K richieste che partono insieme. Restituisce misure e durata complessiva."""
    barriera = threading.Barrier(k)

    def lavoro(i: int) -> Misura:
        barriera.wait()
        latenza, scadenze, errore = _una_richiesta(cliente, modello, endpoint, prompt, opzioni)
        return Misura(k, ripetizione, i, latenza, scadenze, errore)

    inizio = cliente.orologio()
    with ThreadPoolExecutor(max_workers=k) as pool:
        misure = list(pool.map(lavoro, range(1, k + 1)))
    return misure, cliente.orologio() - inizio


def analizza(livelli: dict[int, EsitoLivello]) -> float | None:
    """Calcola baseline, rapporti e classificazione; restituisce la baseline."""
    base = livelli.get(1)
    baseline = statistics.median(base.latenze) if base and base.latenze else None
    for k, esito in livelli.items():
        esito.classificazione = classifica(baseline, esito.latenze, k) if baseline else "n.d."
        if baseline and esito.latenze:
            esito.rapporto_max = round(max(esito.latenze) / baseline, 2)
            esito.rapporto_medio = round(statistics.fmean(esito.latenze) / baseline, 2)
    return baseline


def esegui_concorrenza(cliente: ClienteOllama, modello: str, paralleli: list[int], cartella: Path,
                       ripetizioni: int = 1, endpoint: str = "generate",
                       num_predict: int = NUM_PREDICT_DEFAULT, prompt: str | None = None) -> Path:
    """Esegue la verifica e scrive report Markdown e CSV in ``cartella``."""
    if endpoint not in ("generate", "chat"):
        raise ValueError("endpoint ammessi: generate, chat")
    livelli_richiesti = sorted({1, *[k for k in paralleli if k >= 1]})
    prompt = prompt or prompt_fisso()
    opzioni = {"num_predict": num_predict}
    cartella.mkdir(parents=True, exist_ok=True)

    cliente.carica(modello)
    _una_richiesta(cliente, modello, endpoint, prompt, opzioni)   # riscaldamento, non misurato
    memoria_prima = _memoria(cliente, modello)

    livelli: dict[int, EsitoLivello] = {}
    for k in livelli_richiesti:
        esito = EsitoLivello(k)
        for rip in range(1, ripetizioni + 1):
            misure, durata = misura_livello(cliente, modello, k, rip, endpoint, prompt, opzioni)
            esito.misure += misure
            esito.durata_complessiva.append(durata)
        livelli[k] = esito
    memoria_dopo = _memoria(cliente, modello)
    baseline = analizza(livelli)

    nome = f"concorrenza_{modello.replace(':', '-').replace('/', '-')}"
    righe = [{"livello": m.livello, "ripetizione": m.ripetizione, "richiesta": m.richiesta,
              "latenza_s": "" if m.latenza is None else round(m.latenza, 2),
              "rapporto_baseline": round(m.latenza / baseline, 2) if (m.latenza and baseline) else "",
              "timeout": m.timeout, "errore": m.errore}
             for e in livelli.values() for m in e.misure]
    R.scrivi_csv(cartella / f"{nome}.csv", ["livello", "ripetizione", "richiesta", "latenza_s",
                                             "rapporto_baseline", "timeout", "errore"], righe)
    percorso = cartella / f"{nome}.md"
    percorso.write_text(_markdown(modello, endpoint, num_predict, ripetizioni, baseline, livelli,
                                  memoria_prima, memoria_dopo), encoding="utf-8")
    return percorso


def _memoria(cliente: ClienteOllama, modello: str) -> str:
    try:
        m = leggi_memoria(cliente, modello)
    except ErroreOllama:
        return "n.d."
    return f"{m.size_gb} GB (VRAM {m.size_vram_gb} GB)" if m.trovato else "n.d."


def conclusione(livelli: dict[int, EsitoLivello]) -> str:
    oltre = [e for k, e in sorted(livelli.items()) if k > 1 and e.classificazione not in ("", "n.d.")]
    if not oltre:
        return "Nessun livello oltre 1 misurato: non si può concludere sulla concorrenza."
    tipi = {e.classificazione for e in oltre}
    if tipi == {IN_CODA}:
        return ("Ollama serve le richieste IN CODA: ogni utente in più allunga l'attesa degli altri "
                "di circa un tempo di risposta. Verificare OLLAMA_NUM_PARALLEL sul server.")
    if tipi == {PARALLELO}:
        return "Ollama serve le richieste IN PARALLELO senza degrado apprezzabile ai livelli misurati."
    return ("Ollama serve le richieste in parallelo, ma con CONTESA delle risorse: le latenze "
            "crescono rispetto alla richiesta isolata senza arrivare al raddoppio tipico della coda. "
            "Dettaglio per livello nella tabella.")


def _markdown(modello, endpoint, num_predict, ripetizioni, baseline, livelli, mem_prima, mem_dopo) -> str:
    out = [
        f"# Dike – verifica della concorrenza: {modello}", "",
        f"- Data: {datetime.now().isoformat(timespec='seconds')}",
        f"- Endpoint: `/api/{endpoint}` · risposta limitata a {num_predict} token · "
        f"ripetizioni per livello: {ripetizioni}",
        f"- Prompt: fisso, in formato RAG di Clio (3 passaggi, domanda: \"{DOMANDA_FISSA}\")",
        f"- Riferimento (richiesta isolata, mediana): {_f(baseline)} s",
        f"- Memoria del modello (`/api/ps`): prima {mem_prima} · dopo {mem_dopo}", "",
        "| Richieste insieme | Latenze (s) | Media (s) | Max (s) | Rapporto max / riferimento | "
        "Durata complessiva (s) | Errori | Comportamento |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for k, e in sorted(livelli.items()):
        st = statistiche_tempi(e.latenze)
        latenze = ", ".join(f"{x:.1f}" for x in sorted(e.latenze)) or "–"
        errori = sum(bool(m.errore) for m in e.misure)
        durata = ", ".join(f"{d:.1f}" for d in e.durata_complessiva)
        out.append(f"| {k} | {latenze} | {_f(st['media'])} | {_f(st['max'])} | "
                   f"{_f(e.rapporto_max)} | {durata} | {errori} | {e.classificazione} |")
    out += [
        "", f"**Conclusione.** {conclusione(livelli)}", "",
        "## Come leggere", "",
        f"- **parallelo**: rapporto massimo ≤ {SOGLIA_PARALLELO}; **in coda**: rapporto massimo ≥ "
        f"{SOGLIA_CODA} × K (latenze a gradini); **parallelo con contesa**: valori intermedi.",
        "- `OLLAMA_NUM_PARALLEL` (variabile del server Ollama, non leggibile dall'API) stabilisce "
        "quante richieste per modello vengono servite insieme; le altre restano in coda "
        "(massimo `OLLAMA_MAX_QUEUE`). Con 1 ci si aspetta \"in coda\". Ogni posto parallelo "
        "riserva memoria per il proprio contesto: confrontare la memoria prima e dopo.",
        "- Ripetere la misura dopo ogni cambio di hardware, modello o configurazione del server, "
        "e prima di aprire il sistema a più utenti.", "",
    ]
    return "\n".join(out)


def _f(x) -> str:
    return "–" if x is None else f"{x:.1f}" if isinstance(x, float) else str(x)
