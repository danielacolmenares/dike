"""Il protocollo di confronto tra modelli (comando ``esegui``).

Ordine delle operazioni:
  1. lettura di Golden Set e Negative Set della KB (funzioni di Clio);
  2. recupero dei passaggi per ogni domanda, UNA volta sola, e costruzione
     del prompt di Clio: tutto finisce in cache (``contesto.json``);
  3. per ogni esecuzione (``--run``, default 2) e per ogni modello:
       - caricamento del modello e lettura della memoria (``/api/ps``);
       - se ``--ram-max-gb`` e' superato: "escluso: vincolo hardware";
       - generazione per ogni domanda con lo stesso prompt per tutti;
       - arresto anticipato: se dopo le prime ``--stop-dopo`` risposte la
         latenza media supera ``--stop-latenza-s``, il modello viene
         escluso ("escluso: troppo lento (test interrotto)");
       - scaricamento del modello, per misurare il successivo da solo;
  4. indicatori automatici per ogni risposta, CSV di dettaglio, file per la
     revisione umana e riepilogo.

L'intero protocollo viene ripetuto ``--run`` volte: il confronto tra le
esecuzioni misura il determinismo (temperatura 0 e seed fisso dovrebbero
dare testi identici; se non succede, va saputo prima di scegliere).

Le domande senza alcun passaggio recuperato non chiamano il modello, come
in Clio: la risposta e' la frase di rifiuto e non entra nei tempi.
"""

from __future__ import annotations

import logging
import statistics
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

from clio import config as clio_config
from clio import risposta as clio_risposta
from clio.config import KB

from . import __version__, indicatori, risultati
from .contesto import PRINCIPIO, SET_GOLDEN, CacheContesto, Domanda, VoceContesto, carica_domande
from .memoria import ESITO_VINCOLO_HARDWARE, leggi_memoria, supera_limite
from .ollama import ClienteOllama, ErroreGenerazione, ErroreOllama, Generazione

log = logging.getLogger("dike.protocollo")

ESITO_COMPLETATO = "completato"
ESITO_TROPPO_LENTO = "escluso: troppo lento (test interrotto)"


@dataclass
class Opzioni:
    run: int = 2
    top_k: int = clio_risposta.TOP_K
    versione_prompt: str = "A"
    min_score: float | None = None
    ram_max_gb: float | None = None
    stop_latenza_s: float | None = None
    stop_dopo: int = 5
    scarica_dopo: bool = True


def _riga(modello: str, run: int, domanda: Domanda, voce: VoceContesto,
          gen: Generazione | None, errore: str | None) -> dict:
    r = domanda.riga
    riga = {
        "modello": modello, "run": run, "set": domanda.set, "id": domanda.id,
        "domanda": domanda.testo,
        "documento_atteso": r.get("documento_atteso", ""), "sezione_attesa": r.get("sezione_attesa", ""),
        "risposta_attesa": r.get("risposta_attesa", ""), "nota": r.get("nota", ""),
        "contesto": risultati.unisci(voce.citazioni_contesto()),
        "impronta_contesto": voce.impronta,
        "chiamata_llm": risultati.si_no(voce.prompt is not None),
        "errore": errore or "",
    }
    if errore:
        return riga
    if voce.prompt is None:
        testo = clio_risposta.FRASE_FUORI_DOMINIO       # come Clio: nessun chunk, nessun LLM
    else:
        testo = gen.testo
        riga.update(tempo_s=round(gen.durata, 3), tentativi=gen.tentativi, timeout=gen.timeout)
    if domanda.set == SET_GOLDEN:
        ind = indicatori.indicatori_golden(r, testo, voce.fonti)
    else:
        ind = indicatori.indicatori_negative(testo, voce.fonti)
    fonti_mostrate = [] if ind["rifiuto"] else voce.fonti
    riga.update(
        risposta=testo, lunghezza_caratteri=len(testo),
        fonti=risultati.unisci([f"[{f['citazione']}]" for f in fonti_mostrate]),
        citazioni_inline=risultati.unisci(ind["citazioni_inline"]),
        citazioni_inventate=risultati.unisci(ind["citazioni_inventate"]),
        esito_automatico=ind["esito_automatico"],
    )
    for campo in ("rifiuto", "falso_negativo", "rifiuto_corretto", "possibile_allucinazione",
                  "atteso_doc_in_fonti", "atteso_sez_in_fonti", "atteso_doc_in_citazioni",
                  "atteso_sez_in_citazioni"):
        if campo in ind:
            riga[campo] = risultati.si_no(ind[campo])
    return riga


def _troppo_lento(tempi: list[float], opzioni: Opzioni) -> bool:
    return (opzioni.stop_latenza_s is not None and len(tempi) == opzioni.stop_dopo
            and statistics.fmean(tempi) > opzioni.stop_latenza_s)


def esegui_protocollo(kb: KB, modelli: list[str], cartella: Path, cliente: ClienteOllama,
                      opzioni: Opzioni | None = None) -> Path:
    """Esegue il protocollo e restituisce la cartella dei risultati."""
    opzioni = opzioni or Opzioni()
    cartella.mkdir(parents=True, exist_ok=True)
    inizio = datetime.now()

    domande = carica_domande(kb)
    if not domande:
        raise ValueError(f"La KB '{kb.nome}' non ha domande in Golden Set o Negative Set.")
    cache = CacheContesto(kb, opzioni.top_k, opzioni.versione_prompt, opzioni.min_score)
    log.info("%s", PRINCIPIO)
    for d in domande:
        cache.ottieni(d)
    cache.salva(cartella / risultati.CONTESTO)
    log.info("Contesto recuperato per %s domande (%s ricerche).", len(domande), cache.chiamate_recupero)

    esiti: dict[str, str] = {m: ESITO_COMPLETATO for m in modelli}
    note: dict[str, str] = {}
    righe: list[dict] = []
    memoria: list[dict] = []

    def salva_parziale() -> None:
        risultati.scrivi_csv(cartella / risultati.DETTAGLIO, risultati.COLONNE_DETTAGLIO, righe)
        risultati.scrivi_csv(cartella / risultati.MEMORIA, risultati.COLONNE_MEMORIA, memoria)
        risultati.scrivi_json(cartella / risultati.MODELLI,
                              {"ordine": modelli, "esito": esiti, "note": note})

    for run in range(1, opzioni.run + 1):
        for modello in modelli:
            if esiti[modello] != ESITO_COMPLETATO:
                continue
            log.info("Esecuzione %s/%s - modello %s", run, opzioni.run, modello)
            try:
                t_carico = cliente.carica(modello)
                mem = leggi_memoria(cliente, modello)
            except ErroreOllama as exc:
                esiti[modello] = f"errore: {exc}"
                log.error("%s: caricamento non riuscito: %s", modello, exc)
                salva_parziale()
                continue
            escluso = supera_limite(mem, opzioni.ram_max_gb)
            memoria.append({"modello": modello, "run": run, "trovato": risultati.si_no(mem.trovato),
                            "size_gb": mem.size_gb if mem.trovato else "",
                            "size_vram_gb": mem.size_vram_gb if mem.trovato else "",
                            "tempo_caricamento_s": round(t_carico, 2),
                            "esito": ESITO_VINCOLO_HARDWARE if escluso else ""})
            if escluso:
                esiti[modello] = ESITO_VINCOLO_HARDWARE
                note[modello] = f"occupa {mem.size_gb} GB, limite {opzioni.ram_max_gb} GB"
                log.warning("%s escluso: %s", modello, note[modello])
                _scarica(cliente, modello)
                salva_parziale()
                continue

            tempi: list[float] = []
            for d in domande:
                voce = cache.ottieni(d)
                gen, fallita = None, None
                if voce.prompt is not None:
                    try:
                        gen = cliente.genera(modello, voce.prompt)
                        tempi.append(gen.durata)
                    except ErroreGenerazione as exc:
                        fallita = exc
                        log.error("%s %s: %s", modello, d.chiave, exc)
                righe.append(_riga(modello, run, d, voce, gen, str(fallita) if fallita else None))
                if fallita:
                    righe[-1].update(tentativi=fallita.tentativi, timeout=fallita.timeout)
                if run == 1 and _troppo_lento(tempi, opzioni):
                    esiti[modello] = ESITO_TROPPO_LENTO
                    note[modello] = (f"media {statistics.fmean(tempi):.1f} s sulle prime "
                                     f"{len(tempi)} risposte (soglia {opzioni.stop_latenza_s} s)")
                    log.warning("%s interrotto: %s", modello, note[modello])
                    break
            if opzioni.scarica_dopo:
                _scarica(cliente, modello)
            salva_parziale()

    salva_parziale()
    risultati.scrivi_json(cartella / risultati.PROTOCOLLO, {
        "dike": __version__, "principio": PRINCIPIO, "kb": kb.nome, "modelli": modelli,
        "modello_embedding": clio_config.modello_embedding(),
        "domande_golden": sum(d.set == SET_GOLDEN for d in domande),
        "domande_negative": sum(d.set != SET_GOLDEN for d in domande),
        "ricerche_effettuate": cache.chiamate_recupero,
        "timeout_s": cliente.timeout, "tentativi_max": cliente.tentativi,
        "inizio": inizio.isoformat(timespec="seconds"),
        "fine": datetime.now().isoformat(timespec="seconds"),
        **asdict(opzioni),
    })
    scrivi_revisione(cartella, righe)
    from .riepilogo import genera_riepilogo   # import locale: evita cicli
    genera_riepilogo(cartella)
    return cartella


def _scarica(cliente: ClienteOllama, modello: str) -> None:
    try:
        cliente.scarica(modello)
    except ErroreOllama as exc:   # non blocca il protocollo
        log.warning("%s: scaricamento non riuscito: %s", modello, exc)


def scrivi_revisione(cartella: Path, righe: list[dict]) -> Path:
    """File per la revisione umana: solo la prima esecuzione (le altre
    servono al determinismo), con il verdetto da compilare."""
    percorso = cartella / risultati.REVISIONE
    prima = [dict(r, verdetto_umano="", note_valutatore="") for r in righe if str(r["run"]) == "1"]
    risultati.scrivi_csv(percorso, risultati.COLONNE_REVISIONE, prima)
    return percorso
