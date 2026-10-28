"""Formato della cartella dei risultati (un benchmark = una cartella).

    risultati/<kb>_<timestamp>/
        protocollo.json        parametri dell'esecuzione (riproducibilita')
        contesto.json          passaggi e prompt in cache, uguali per tutti i modelli
        dettaglio.csv          una riga per modello x esecuzione x domanda
        memoria.csv            memoria letta da /api/ps dopo ogni caricamento
        modelli.json           esito per modello (completato / escluso / errore)
        revisione_umana.csv    da compilare: verdetto OK / PARZIALE / ERRATA
        verdetti.csv           verdetti importati con `dike verdetti`
        comportamento.csv      comportamento prevalente (derivato + campo libero)
        riepilogo.md           classifica e sintesi leggibile
        riepilogo_modelli.csv  la stessa classifica in CSV
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

DETTAGLIO = "dettaglio.csv"
MEMORIA = "memoria.csv"
MODELLI = "modelli.json"
PROTOCOLLO = "protocollo.json"
CONTESTO = "contesto.json"
REVISIONE = "revisione_umana.csv"
VERDETTI = "verdetti.csv"
COMPORTAMENTO = "comportamento.csv"
RIEPILOGO_MD = "riepilogo.md"
RIEPILOGO_CSV = "riepilogo_modelli.csv"

COLONNE_DETTAGLIO = [
    "modello", "run", "set", "id", "domanda", "documento_atteso", "sezione_attesa",
    "risposta_attesa", "nota", "risposta", "lunghezza_caratteri", "chiamata_llm",
    "rifiuto", "falso_negativo", "rifiuto_corretto", "possibile_allucinazione",
    "atteso_doc_in_fonti", "atteso_sez_in_fonti", "atteso_doc_in_citazioni",
    "atteso_sez_in_citazioni", "esito_automatico", "fonti", "citazioni_inline",
    "citazioni_inventate", "contesto", "impronta_contesto", "tempo_s", "tentativi",
    "timeout", "errore",
]
COLONNE_MEMORIA = ["modello", "run", "trovato", "size_gb", "size_vram_gb",
                   "tempo_caricamento_s", "esito"]
COLONNE_REVISIONE = [
    "modello", "run", "set", "id", "domanda", "risposta_attesa", "nota", "risposta", "fonti",
    "citazioni_inline", "citazioni_inventate", "esito_automatico", "verdetto_umano",
    "note_valutatore",
]
COLONNE_VERDETTI = ["modello", "run", "set", "id", "verdetto_umano", "note_valutatore"]
VERDETTI_AMMESSI = ("OK", "PARZIALE", "ERRATA")

SI, NO = "sì", "no"


def si_no(valore) -> str:
    if valore is None or valore == "":
        return ""
    return SI if valore else NO


def vero(valore) -> bool:
    return str(valore or "").strip().lower() in ("sì", "si", "true", "1", "yes")


def unisci(valori: list[str]) -> str:
    return " | ".join(valori)


def chiave_riga(riga: dict) -> tuple[str, str, str, str]:
    return (riga["modello"], str(riga["run"]), riga["set"], riga["id"])


def scrivi_csv(percorso: Path, colonne: list[str], righe: list[dict]) -> None:
    with open(percorso, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=colonne, extrasaction="ignore")
        w.writeheader()
        w.writerows(righe)


def leggi_csv(percorso: Path) -> list[dict]:
    if not percorso.exists():
        return []
    with open(percorso, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def scrivi_json(percorso: Path, dati: dict) -> None:
    percorso.write_text(json.dumps(dati, ensure_ascii=False, indent=2), encoding="utf-8")


def leggi_json(percorso: Path) -> dict:
    if not percorso.exists():
        return {}
    return json.loads(percorso.read_text(encoding="utf-8"))
