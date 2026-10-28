"""Statistiche sui tempi e confronto tra esecuzioni (determinismo)."""

from __future__ import annotations

import math
import statistics


def percentile(valori: list[float], q: float) -> float | None:
    """Percentile con interpolazione lineare (q tra 0 e 100), come il
    metodo di default di NumPy."""
    if not valori:
        return None
    ordinati = sorted(valori)
    if len(ordinati) == 1:
        return ordinati[0]
    posizione = (len(ordinati) - 1) * q / 100
    basso, alto = math.floor(posizione), math.ceil(posizione)
    if basso == alto:
        return ordinati[basso]
    return ordinati[basso] + (ordinati[alto] - ordinati[basso]) * (posizione - basso)


def statistiche_tempi(valori: list[float]) -> dict:
    """Media, mediana, minimo, massimo e p90 (secondi, 1 decimale)."""
    if not valori:
        return {"n": 0, "media": None, "mediana": None, "min": None, "max": None, "p90": None}
    arrotonda = lambda x: round(x, 1)  # noqa: E731
    return {
        "n": len(valori),
        "media": arrotonda(statistics.fmean(valori)),
        "mediana": arrotonda(statistics.median(valori)),
        "min": arrotonda(min(valori)),
        "max": arrotonda(max(valori)),
        "p90": arrotonda(percentile(valori, 90)),
    }


def confronta_esecuzioni(risposte: dict[tuple, list[str]]) -> dict:
    """``risposte``: per ogni domanda, i testi ottenuti nelle varie
    esecuzioni. Una domanda e' "identica" se tutte le esecuzioni hanno dato
    lo stesso testo (spazi iniziali/finali esclusi). Le domande con una sola
    esecuzione non sono confrontabili e non entrano nel conteggio."""
    confrontabili = {k: v for k, v in risposte.items() if len(v) >= 2}
    diverse = sorted(k for k, v in confrontabili.items() if len({t.strip() for t in v}) > 1)
    return {
        "identiche": len(confrontabili) - len(diverse),
        "totale": len(confrontabili),
        "diverse": diverse,
    }
