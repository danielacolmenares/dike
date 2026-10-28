"""Importazione dei verdetti umani (comando ``verdetti``).

Gli indicatori automatici dicono dove guardare; il giudizio sulla risposta
resta umano. Il revisore compila ``revisione_umana.csv`` (colonna
``verdetto_umano``: OK / PARZIALE / ERRATA, piu' ``note_valutatore``) e lo
reimporta: i verdetti finiscono in ``verdetti.csv`` e il riepilogo viene
ricalcolato.

Si possono importare anche file parziali o in piu' riprese: un verdetto
nuovo sostituisce quello precedente per la stessa riga (modello,
esecuzione, set, id); una cella vuota non cancella un verdetto gia'
importato. Valori non ammessi bloccano l'importazione con l'elenco delle
righe da correggere.
"""

from __future__ import annotations

from pathlib import Path

from . import risultati as R
from .riepilogo import genera_riepilogo


class ErroreVerdetti(ValueError):
    pass


def _normalizza(valore: str) -> str:
    v = (valore or "").strip().upper()
    return {"PARZIALMENTE": "PARZIALE", "ERRATO": "ERRATA", "SBAGLIATA": "ERRATA"}.get(v, v)


def importa_verdetti(csv_umano: Path, cartella: Path | None = None) -> tuple[int, Path]:
    """Restituisce (verdetti importati, percorso del riepilogo)."""
    csv_umano = Path(csv_umano)
    cartella = Path(cartella) if cartella else csv_umano.parent
    if not csv_umano.exists():
        raise ErroreVerdetti(f"File non trovato: {csv_umano}")
    righe = R.leggi_csv(csv_umano)
    mancanti = [c for c in ("modello", "run", "set", "id", "verdetto_umano") if righe and c not in righe[0]]
    if mancanti:
        raise ErroreVerdetti(f"{csv_umano.name}: colonne mancanti: {', '.join(mancanti)}")

    note_dettaglio = {R.chiave_riga(r) for r in R.leggi_csv(cartella / R.DETTAGLIO)}
    errori, nuovi = [], {}
    for n, riga in enumerate(righe, start=2):       # riga 1 = intestazione
        verdetto = _normalizza(riga.get("verdetto_umano", ""))
        if not verdetto:
            continue
        if verdetto not in R.VERDETTI_AMMESSI:
            errori.append(f"riga {n}: '{riga.get('verdetto_umano')}' non ammesso "
                          f"({' / '.join(R.VERDETTI_AMMESSI)})")
            continue
        chiave = R.chiave_riga(riga)
        if note_dettaglio and chiave not in note_dettaglio:
            errori.append(f"riga {n}: {'/'.join(chiave)} non presente in {R.DETTAGLIO}")
            continue
        nuovi[chiave] = {"modello": chiave[0], "run": chiave[1], "set": chiave[2], "id": chiave[3],
                         "verdetto_umano": verdetto,
                         "note_valutatore": (riga.get("note_valutatore") or "").strip()}
    if errori:
        raise ErroreVerdetti("Verdetti non importati:\n  " + "\n  ".join(errori))

    esistenti = {R.chiave_riga(v): v for v in R.leggi_csv(cartella / R.VERDETTI)}
    esistenti.update(nuovi)
    R.scrivi_csv(cartella / R.VERDETTI, R.COLONNE_VERDETTI, list(esistenti.values()))
    return len(nuovi), genera_riepilogo(cartella)
