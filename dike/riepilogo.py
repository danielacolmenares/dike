"""Riepilogo del benchmark: classifica per modello e report Markdown.

Si rigenera in qualsiasi momento dalla cartella dei risultati
(``dike riepilogo <cartella>``), per esempio dopo aver importato i verdetti
umani o compilato ``comportamento.csv``.

Regole di calcolo:
  - qualita' e rifiuti si calcolano sulla PRIMA esecuzione; le successive
    servono al confronto di determinismo;
  - Qualita' (Golden) = risposte con verdetto umano OK. Finche' i verdetti
    non sono completi la colonna mostra la stima automatica, marcata
    "(auto)": risposte che citano il riferimento atteso senza citazioni
    inventate. La stima non sostituisce la revisione;
  - Rifiuti (Negative) = domande fuori dominio con la frase di rifiuto
    prevista (o verdetto umano OK, se presente);
  - Totale = Qualita' + Rifiuti, come nel benchmark originale;
  - i tempi sono quelli di generazione, su tutte le esecuzioni, compresi gli
    eventuali tentativi ripetuti;
  - classifica: prima i modelli completati, per totale decrescente e poi
    per tempo medio crescente; in coda gli esclusi.
"""

from __future__ import annotations

import math
import statistics
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from . import risultati as R
from .contesto import SET_GOLDEN, SET_NEGATIVE
from .protocollo import ESITO_COMPLETATO
from .statistiche import confronta_esecuzioni, statistiche_tempi

SOGLIA_FALSI_NEGATIVI = 0.2      # quota di falsi negativi oltre cui il modello e' "prudente"
SOGLIA_VERBOSO = 1.5             # lunghezza media oltre 1,5 volte la mediana dei modelli
SOGLIA_CITAZIONI = 0.2           # quota di risposte con citazioni inventate
SOGLIA_LENTO = 1.5               # tempo medio oltre 1,5 volte il piu' veloce


def _int(valore, default: int = 0) -> int:
    try:
        return int(float(valore))
    except (TypeError, ValueError):
        return default


def _float(valore) -> float | None:
    try:
        return float(valore)
    except (TypeError, ValueError):
        return None


def carica_verdetti(cartella: Path) -> dict[tuple, dict]:
    return {R.chiave_riga(v): v for v in R.leggi_csv(cartella / R.VERDETTI)
            if (v.get("verdetto_umano") or "").strip()}


def calcola_modello(modello: str, righe: list[dict], memoria: list[dict], esito: str,
                    verdetti: dict[tuple, dict], n_golden: int, n_negative: int) -> dict:
    prima = [r for r in righe if str(r["run"]) == "1"]
    golden = [r for r in prima if r["set"] == SET_GOLDEN]
    negative = [r for r in prima if r["set"] == SET_NEGATIVE]
    valide_g = [r for r in golden if not r["errore"]]

    def verdetto(r: dict) -> str:
        return (verdetti.get(R.chiave_riga(r), {}).get("verdetto_umano") or "").strip().upper()

    conteggio = {v: sum(verdetto(r) == v for r in golden) for v in R.VERDETTI_AMMESSI}
    da_valutare = sum(verdetto(r) == "" for r in golden)
    coerenti = sum(r["esito_automatico"] == "coerente" for r in valide_g)

    rifiuti_ok, allucinazioni = 0, []
    for r in negative:
        if r["errore"]:
            continue
        v = verdetto(r)
        corretto = (v == "OK") if v else R.vero(r["rifiuto_corretto"])
        rifiuti_ok += corretto
        if not corretto:
            allucinazioni.append(r["id"])

    completo = esito == ESITO_COMPLETATO and len(golden) == n_golden and len(negative) == n_negative
    qualita_umana = conteggio["OK"] if (golden and da_valutare == 0) else None

    per_domanda: dict[tuple, list[str]] = defaultdict(list)
    for r in righe:
        if not r["errore"]:
            per_domanda[(r["set"], r["id"])].append(r["risposta"])
    det = confronta_esecuzioni(per_domanda)

    tempi = [t for r in righe if not r["errore"] and (t := _float(r["tempo_s"])) is not None]
    risposte_date = [r for r in valide_g if not R.vero(r["rifiuto"])]
    mem_size = [s for m in memoria if (s := _float(m["size_gb"])) is not None]
    mem_vram = [s for m in memoria if (s := _float(m["size_vram_gb"])) is not None]

    return {
        "modello": modello, "esito": esito, "completo": completo,
        "n_golden": n_golden, "n_negative": n_negative,
        "golden_eseguite": len(golden), "negative_eseguite": len(negative),
        "verdetti": conteggio, "da_valutare": da_valutare,
        "qualita_umana": qualita_umana, "qualita_auto": coerenti,
        "rifiuti_corretti": rifiuti_ok, "possibili_allucinazioni": allucinazioni,
        "falsi_negativi": sum(R.vero(r["falso_negativo"]) for r in valide_g),
        "atteso_in_fonti": sum(R.vero(r["atteso_sez_in_fonti"]) for r in valide_g),
        "atteso_in_citazioni": sum(R.vero(r["atteso_sez_in_citazioni"]) for r in valide_g),
        "con_citazioni_inventate": sum(bool(r["citazioni_inventate"]) for r in prima if not r["errore"]),
        "risposte_valide": sum(not r["errore"] for r in prima),
        "determinismo": det,
        "tempi": statistiche_tempi(tempi),
        "timeout": sum(_int(r["timeout"]) for r in righe),
        "ripetizioni": sum(max(_int(r["tentativi"], 1) - 1, 0) for r in righe if r["tentativi"]),
        "errori": sum(bool(r["errore"]) for r in righe),
        "memoria_gb": max(mem_size) if mem_size else None,
        "memoria_vram_gb": max(mem_vram) if mem_vram else None,
        "lunghezza_media": round(statistics.fmean(len(r["risposta"]) for r in risposte_date))
        if risposte_date else None,
    }


def comportamento_derivato(m: dict, mediana_lunghezza: float | None, tempo_minimo: float | None) -> str:
    """Etichetta euristica: un punto di partenza per il giudizio umano."""
    if m["esito"] != ESITO_COMPLETATO:
        return ""
    tratti = []
    valide = max(m["golden_eseguite"], 1)
    if m["falsi_negativi"] and m["falsi_negativi"] / valide >= SOGLIA_FALSI_NEGATIVI:
        tratti.append(f"prudente/sotto-risponde ({m['falsi_negativi']} falsi negativi)")
    if (mediana_lunghezza and m["lunghezza_media"]
            and m["lunghezza_media"] > SOGLIA_VERBOSO * mediana_lunghezza):
        tratti.append("verboso")
    if m["risposte_valide"] and m["con_citazioni_inventate"] / m["risposte_valide"] >= SOGLIA_CITAZIONI:
        tratti.append("citazioni imprecise")
    if m["possibili_allucinazioni"]:
        tratti.append("risponde a domande fuori dominio")
    if m["determinismo"]["totale"] and m["determinismo"]["identiche"] < m["determinismo"]["totale"]:
        tratti.append("non deterministico")
    media = m["tempi"]["media"]
    if tempo_minimo and media and media > SOGLIA_LENTO * tempo_minimo:
        tratti.append("lento")
    return ", ".join(tratti) or "equilibrato"


def _aggiorna_comportamento(cartella: Path, modelli: list[dict]) -> dict[str, str]:
    """Scrive comportamento.csv con l'etichetta derivata e conserva il campo
    libero ``comportamento`` compilato a mano, che ha la precedenza."""
    percorso = cartella / R.COMPORTAMENTO
    umani = {r["modello"]: (r.get("comportamento") or "").strip() for r in R.leggi_csv(percorso)}
    R.scrivi_csv(percorso, ["modello", "comportamento_derivato", "comportamento"],
                 [{"modello": m["modello"], "comportamento_derivato": m["comportamento_derivato"],
                   "comportamento": umani.get(m["modello"], "")} for m in modelli])
    return {k: v for k, v in umani.items() if v}


def _qualita(m: dict) -> str:
    if m["golden_eseguite"] == 0:
        return "n.d."
    if m["qualita_umana"] is not None:
        return f"{m['qualita_umana']}/{m['n_golden']}"
    return f"{m['qualita_auto']}/{m['n_golden']} (auto)"


def _totale(m: dict) -> tuple[str, float]:
    if not m["completo"]:
        return "–", -1
    n = m["n_golden"] + m["n_negative"]
    if m["qualita_umana"] is not None:
        t = m["qualita_umana"] + m["rifiuti_corretti"]
        return f"{t}/{n}", t
    t = m["qualita_auto"] + m["rifiuti_corretti"]
    return f"{t}/{n} (auto)", t


def _determinismo(m: dict) -> str:
    d = m["determinismo"]
    if not d["totale"]:
        return "n.d. (1 run)"
    return ("sì" if d["identiche"] == d["totale"] else "no") + f" ({d['identiche']}/{d['totale']})"


def _s(valore, unita: str = "") -> str:
    return "–" if valore is None else f"{valore}{unita}"


def genera_riepilogo(cartella: Path) -> Path:
    cartella = Path(cartella)
    if not (cartella / R.DETTAGLIO).exists():
        raise FileNotFoundError(f"{cartella}: manca {R.DETTAGLIO}, non e' una cartella di risultati di Dike")
    righe = R.leggi_csv(cartella / R.DETTAGLIO)
    memoria = R.leggi_csv(cartella / R.MEMORIA)
    info_modelli = R.leggi_json(cartella / R.MODELLI)
    protocollo = R.leggi_json(cartella / R.PROTOCOLLO)
    verdetti = carica_verdetti(cartella)

    ordine = info_modelli.get("ordine") or list(dict.fromkeys(r["modello"] for r in righe))
    esiti = info_modelli.get("esito", {})
    note = info_modelli.get("note", {})
    n_golden = protocollo.get("domande_golden") or len({r["id"] for r in righe if r["set"] == SET_GOLDEN})
    n_negative = protocollo.get("domande_negative") or len({r["id"] for r in righe if r["set"] == SET_NEGATIVE})

    modelli = [calcola_modello(m, [r for r in righe if r["modello"] == m],
                               [x for x in memoria if x["modello"] == m],
                               esiti.get(m, ESITO_COMPLETATO), verdetti, n_golden, n_negative)
               for m in ordine]

    completati = [m for m in modelli if m["esito"] == ESITO_COMPLETATO]
    lunghezze = [m["lunghezza_media"] for m in completati if m["lunghezza_media"]]
    mediana_l = statistics.median(lunghezze) if len(lunghezze) >= 2 else None
    medie = [m["tempi"]["media"] for m in completati if m["tempi"]["media"]]
    minimo = min(medie) if len(medie) >= 2 else None
    for m in modelli:
        m["comportamento_derivato"] = comportamento_derivato(m, mediana_l, minimo)
    umani = _aggiorna_comportamento(cartella, modelli)
    for m in modelli:
        m["comportamento"] = umani.get(m["modello"]) or m["comportamento_derivato"] or note.get(m["modello"], "")

    def chiave(m: dict):
        incluso = m["esito"] == ESITO_COMPLETATO
        return (0 if incluso else 1, -_totale(m)[1] if incluso else 0,
                m["tempi"]["media"] if (incluso and m["tempi"]["media"] is not None) else math.inf)

    classifica = sorted(modelli, key=chiave)
    _scrivi_csv_riepilogo(cartella, classifica)
    percorso = cartella / R.RIEPILOGO_MD
    percorso.write_text(_markdown(classifica, protocollo, note, bool(verdetti)), encoding="utf-8")
    return percorso


def _scrivi_csv_riepilogo(cartella: Path, classifica: list[dict]) -> None:
    righe = []
    for pos, m in enumerate(classifica, start=1):
        t = m["tempi"]
        righe.append({
            "posizione": pos, "modello": m["modello"], "qualita_golden": _qualita(m),
            "rifiuti_negative": f"{m['rifiuti_corretti']}/{m['n_negative']}",
            "totale": _totale(m)[0], "determinismo": _determinismo(m),
            "tempo_medio_s": _s(t["media"]), "tempo_mediano_s": _s(t["mediana"]),
            "tempo_min_s": _s(t["min"]), "tempo_max_s": _s(t["max"]), "tempo_p90_s": _s(t["p90"]),
            "timeout": m["timeout"], "ripetizioni": m["ripetizioni"], "errori": m["errori"],
            "memoria_gb": _s(m["memoria_gb"]), "memoria_vram_gb": _s(m["memoria_vram_gb"]),
            "falsi_negativi": m["falsi_negativi"], "citazioni_inventate": m["con_citazioni_inventate"],
            "possibili_allucinazioni": len(m["possibili_allucinazioni"]),
            "comportamento": m["comportamento"], "esito": m["esito"],
        })
    colonne = list(righe[0]) if righe else ["posizione", "modello"]
    R.scrivi_csv(cartella / R.RIEPILOGO_CSV, colonne, righe)


def _markdown(classifica: list[dict], protocollo: dict, note: dict, ci_sono_verdetti: bool) -> str:
    p = protocollo
    out = [
        "# Dike – riepilogo del benchmark", "",
        f"- Knowledge Base: **{p.get('kb', 'n.d.')}** · Golden Set {p.get('domande_golden', 'n.d.')} "
        f"domande · Negative Set {p.get('domande_negative', 'n.d.')} domande",
        f"- Esecuzioni per modello: {p.get('run', 'n.d.')} · prompt {p.get('versione_prompt', 'n.d.')} · "
        f"top-k {p.get('top_k', 'n.d.')} · embedding {p.get('modello_embedding', 'n.d.')}",
        f"- Periodo: {p.get('inizio', 'n.d.')} → {p.get('fine', 'n.d.')} · "
        f"riepilogo generato il {datetime.now().isoformat(timespec='seconds')}",
        "", f"> {p.get('principio', '')}", "",
        "## Classifica", "",
        "| # | Modello | Qualità (Golden) | Rifiuti (Negative) | Totale | Determinismo | "
        "Tempo medio | Memoria | Comportamento prevalente | Esito |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for pos, m in enumerate(classifica, start=1):
        rifiuti = f"{m['rifiuti_corretti']}/{m['n_negative']}" if m["negative_eseguite"] else "n.d."
        out.append(f"| {pos} | {m['modello']} | {_qualita(m)} | {rifiuti} | {_totale(m)[0]} | "
                   f"{_determinismo(m)} | {_s(m['tempi']['media'], ' s')} | "
                   f"{_s(m['memoria_gb'], ' GB')} | {m['comportamento']} | {m['esito']} |")
    out += ["",
            "Qualità = risposte con verdetto umano OK. \"(auto)\" indica la stima automatica, usata "
            "finché i verdetti non sono completi: risposta che cita il riferimento atteso senza "
            "citazioni inventate. La stima orienta la revisione, non la sostituisce.",
            ""]
    if not ci_sono_verdetti:
        out += ["**Verdetti umani non ancora importati.** Compilare `revisione_umana.csv` "
                "(colonna `verdetto_umano`: OK / PARZIALE / ERRATA) e poi eseguire "
                "`python -m dike.cli verdetti <cartella>/revisione_umana.csv`.", ""]

    out += ["## Tempi di generazione (secondi)", "",
            "| Modello | Risposte | Media | Mediana | Min | Max | p90 | Timeout | Ripetizioni | Errori |",
            "|---|---|---|---|---|---|---|---|---|---|"]
    for m in classifica:
        t = m["tempi"]
        out.append(f"| {m['modello']} | {t['n']} | {_s(t['media'])} | {_s(t['mediana'])} | "
                   f"{_s(t['min'])} | {_s(t['max'])} | {_s(t['p90'])} | {m['timeout']} | "
                   f"{m['ripetizioni']} | {m['errori']} |")

    out += ["", "## Memoria (da `/api/ps` dopo il caricamento)", "",
            "| Modello | Occupata | di cui VRAM |", "|---|---|---|"]
    for m in classifica:
        out.append(f"| {m['modello']} | {_s(m['memoria_gb'], ' GB')} | {_s(m['memoria_vram_gb'], ' GB')} |")

    out += ["", "## Golden Set: indicatori automatici (prima esecuzione)", "",
            "| Modello | Riferimento atteso tra le fonti | Riferimento atteso nelle citazioni del modello | "
            "Falsi negativi | Risposte con citazioni inventate | Verdetti OK / PARZIALE / ERRATA / da fare |",
            "|---|---|---|---|---|---|"]
    for m in classifica:
        v = m["verdetti"]
        out.append(f"| {m['modello']} | {m['atteso_in_fonti']}/{m['golden_eseguite']} | "
                   f"{m['atteso_in_citazioni']}/{m['golden_eseguite']} | {m['falsi_negativi']} | "
                   f"{m['con_citazioni_inventate']} | {v['OK']} / {v['PARZIALE']} / {v['ERRATA']} / "
                   f"{m['da_valutare']} |")

    out += ["", "## Negative Set: rifiuti", "",
            "| Modello | Rifiuti corretti | Possibili allucinazioni (da verificare) |", "|---|---|---|"]
    for m in classifica:
        out.append(f"| {m['modello']} | {m['rifiuti_corretti']}/{m['negative_eseguite']} | "
                   f"{', '.join(m['possibili_allucinazioni']) or '–'} |")

    out += ["", "## Determinismo", "",
            "| Modello | Risposte identiche tra le esecuzioni | Domande con testo diverso |", "|---|---|---|"]
    for m in classifica:
        d = m["determinismo"]
        diverse = ", ".join(f"{s}:{i}" for s, i in d["diverse"]) or "–"
        out.append(f"| {m['modello']} | {d['identiche']}/{d['totale']} | {diverse} |")

    esclusi = [m for m in classifica if m["esito"] != ESITO_COMPLETATO]
    if esclusi:
        out += ["", "## Modelli esclusi", ""]
        for m in esclusi:
            dettaglio = note.get(m["modello"])
            out.append(f"- **{m['modello']}**: {m['esito']}" + (f" – {dettaglio}" if dettaglio else ""))

    out += ["", "## Come leggere", "",
            "- Il **comportamento prevalente** è derivato da regole semplici (falsi negativi, "
            "lunghezza delle risposte, citazioni inventate, rifiuti mancati, velocità). Il campo "
            "libero `comportamento` in `comportamento.csv`, se compilato, ha la precedenza.",
            "- **Allucinazione** e **citazione imprecisa** sono conteggi separati: una citazione "
            "non corrispondente ai passaggi recuperati va verificata, ma non è di per sé un fatto inventato.",
            "- Nessun modello si adotta senza rifare il protocollo dopo ogni modifica (modello, "
            "prompt, chunking, hardware).",
            "- Dettaglio per domanda: `dettaglio.csv`.", ""]
    return "\n".join(out)
