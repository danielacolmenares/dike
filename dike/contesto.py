"""Adattatore verso Clio: domande della KB e contesto recuperato in cache.

PRINCIPIO: stessi passaggi per tutti i modelli, cambia solo il generatore.

Il recupero (embedding della domanda + ricerca su Qdrant) avviene una sola
volta per domanda, con le funzioni di Clio; il prompt viene costruito una
sola volta con il prompt di Clio. Il risultato e' salvato in
``contesto.json`` nella cartella dei risultati, insieme a un'impronta
(SHA-256) del prompt: ogni riga del dettaglio riporta l'impronta, cosi' si
puo' verificare che ogni modello abbia ricevuto esattamente lo stesso
testo. Se un modello sbaglia, non e' perche' ha visto passaggi diversi.

Dike non copia codice di Clio: importa ``clio`` come libreria.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

from clio import retrieval as clio_retrieval
from clio import risposta as clio_risposta
from clio.config import KB

PRINCIPIO = ("Stesso contesto per tutti i modelli: i passaggi vengono recuperati una sola "
             "volta per domanda e messi in cache; cambia solo il modello che genera la risposta.")

SET_GOLDEN = "golden"
SET_NEGATIVE = "negative"


@dataclass
class Domanda:
    set: str                      # golden | negative
    id: str
    testo: str
    riga: dict = field(default_factory=dict)   # riga originale del CSV

    @property
    def chiave(self) -> str:
        return f"{self.set}:{self.id}"


@dataclass
class VoceContesto:
    risultati: list[dict]         # chunk recuperati (dopo l'eventuale filtro score)
    fonti: list[dict]             # fonti deterministiche di Clio
    prompt: str | None            # None se nessun chunk: Clio non chiama l'LLM
    impronta: str                 # SHA-256 del prompt (12 caratteri)

    def citazioni_contesto(self) -> list[str]:
        return [f["citazione"] for f in self.fonti]


def carica_domande(kb: KB) -> list[Domanda]:
    """Golden Set e Negative Set della KB, letti con le funzioni di Clio.
    Un set non dichiarato o assente viene semplicemente saltato."""
    domande: list[Domanda] = []
    insiemi = ((SET_GOLDEN, kb.golden_set, clio_retrieval.COLONNE_GOLDEN[:4]),
               (SET_NEGATIVE, kb.negative_set, ["id", "domanda"]))
    for nome, percorso, colonne in insiemi:
        if percorso is None or not percorso.exists():
            continue
        for riga in clio_retrieval.leggi_set(percorso, colonne):
            domande.append(Domanda(nome, str(riga["id"]).strip(), riga["domanda"].strip(), dict(riga)))
    return domande


def _impronta(prompt: str | None) -> str:
    return hashlib.sha256((prompt or "").encode("utf-8")).hexdigest()[:12]


class CacheContesto:
    """Recupero una volta per domanda; le richieste successive (altri
    modelli, altre esecuzioni) leggono dalla cache."""

    def __init__(self, kb: KB, top_k: int = clio_risposta.TOP_K, versione_prompt: str = "A",
                 min_score: float | None = None):
        self.kb = kb
        self.top_k = top_k
        self.versione_prompt = versione_prompt.upper()
        self.min_score = min_score
        self.voci: dict[str, VoceContesto] = {}
        self.chiamate_recupero = 0

    def ottieni(self, domanda: Domanda) -> VoceContesto:
        if domanda.chiave not in self.voci:
            self.chiamate_recupero += 1
            risultati, _, _ = clio_retrieval.cerca(self.kb, domanda.testo, self.top_k)
            selezionati = clio_retrieval.filtra_per_score(risultati, self.min_score)
            prompt = (clio_risposta.costruisci_prompt(domanda.testo, selezionati, self.versione_prompt)
                      if selezionati else None)
            self.voci[domanda.chiave] = VoceContesto(
                risultati=selezionati,
                fonti=clio_risposta.fonti_da_risultati(selezionati),
                prompt=prompt,
                impronta=_impronta(prompt),
            )
        return self.voci[domanda.chiave]

    def salva(self, percorso: Path) -> None:
        dati = {
            "principio": PRINCIPIO,
            "kb": self.kb.nome, "top_k": self.top_k, "versione_prompt": self.versione_prompt,
            "min_score": self.min_score,
            "voci": {k: {"risultati": v.risultati, "fonti": v.fonti, "prompt": v.prompt,
                         "impronta": v.impronta} for k, v in self.voci.items()},
        }
        percorso.write_text(json.dumps(dati, ensure_ascii=False, indent=2), encoding="utf-8")
