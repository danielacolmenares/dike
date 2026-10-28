"""Verifica della concorrenza: classificazione e misura con server simulato."""

from __future__ import annotations

import threading
import time

import pytest

from dike import concorrenza as C
from dike import ollama

GB = 1_000_000_000


# ----------------------------------------------------------------------
# Classificazione
# ----------------------------------------------------------------------

def test_caso_originale_parallelismo_con_contesa():
    # isolata ~21,5 s; due richieste insieme 20,2 s e 33,1 s
    assert C.classifica(21.5, [20.2, 33.1], 2) == C.CONTESA


def test_parallelo_puro():
    assert C.classifica(20.0, [20.5, 21.0, 22.0, 23.0], 4) == C.PARALLELO


def test_in_coda():
    assert C.classifica(20.0, [20.1, 40.3], 2) == C.IN_CODA
    assert C.classifica(20.0, [20.0, 40.0, 60.0, 80.0], 4) == C.IN_CODA


def test_livello_uno_e_riferimento():
    assert C.classifica(20.0, [20.0], 1) == C.RIFERIMENTO


def test_analizza_calcola_baseline_e_rapporti():
    livelli = {1: C.EsitoLivello(1, [C.Misura(1, 1, 1, 21.0), C.Misura(1, 2, 1, 22.0)]),
               2: C.EsitoLivello(2, [C.Misura(2, 1, 1, 20.2), C.Misura(2, 1, 2, 33.1)])}
    baseline = C.analizza(livelli)
    assert baseline == 21.5
    assert livelli[2].rapporto_max == 1.54 and livelli[2].classificazione == C.CONTESA
    assert "CONTESA" in C.conclusione(livelli)


def test_misure_con_errore_escluse():
    livelli = {1: C.EsitoLivello(1, [C.Misura(1, 1, 1, 20.0)]),
               2: C.EsitoLivello(2, [C.Misura(2, 1, 1, 20.0), C.Misura(2, 1, 2, None, 1, "timeout")])}
    C.analizza(livelli)
    assert livelli[2].latenze == [20.0]
    assert livelli[2].classificazione == C.PARALLELO


def test_prompt_fisso_in_formato_clio():
    p = C.prompt_fisso()
    assert "FONTI RECUPERATE" in p and "[PROC-INC, 4.2]" in p and C.DOMANDA_FISSA in p


# ----------------------------------------------------------------------
# Misura end-to-end con un Ollama simulato (thread veri, attese brevi)
# ----------------------------------------------------------------------

DURATA = 0.15


class ServerSimulato:
    def __init__(self, in_coda: bool):
        self.in_coda = in_coda
        self.lock = threading.Lock()
        self.richieste = 0

    def _lavora(self):
        time.sleep(DURATA)

    def __call__(self, metodo, url, corpo=None, timeout=60):
        if url.endswith("/api/ps"):
            return {"models": [{"name": "alfa:4b", "model": "alfa:4b", "size": 3 * GB, "size_vram": 0}]}
        if not corpo.get("prompt") and "messages" not in corpo:
            return {"response": ""}                       # caricamento
        self.richieste += 1
        if self.in_coda:
            with self.lock:
                self._lavora()
        else:
            self._lavora()
        if url.endswith("/api/chat"):
            return {"message": {"content": "Entro quattro ore [PROC-INC, 4.2]."}}
        return {"response": "Entro quattro ore [PROC-INC, 4.2]."}


@pytest.mark.parametrize("in_coda,atteso", [(True, C.IN_CODA), (False, C.PARALLELO)])
def test_misura_distingue_coda_e_parallelo(monkeypatch, tmp_path, in_coda, atteso):
    server = ServerSimulato(in_coda)
    monkeypatch.setattr(ollama, "_http", server)
    cliente = ollama.ClienteOllama(url="http://127.0.0.1:11434", tentativi=1)
    percorso = C.esegui_concorrenza(cliente, "alfa:4b", [2, 4], tmp_path)
    assert server.richieste == 1 + 1 + 2 + 4               # riscaldamento + livelli 1, 2, 4
    testo = percorso.read_text(encoding="utf-8")
    assert "| 4 |" in testo and atteso in testo
    assert "OLLAMA_NUM_PARALLEL" in testo
    assert "3.0 GB" in testo
    righe = (tmp_path / "concorrenza_alfa-4b.csv").read_text(encoding="utf-8").splitlines()
    assert len(righe) == 1 + 7


def test_endpoint_chat(monkeypatch, tmp_path):
    server = ServerSimulato(False)
    monkeypatch.setattr(ollama, "_http", server)
    cliente = ollama.ClienteOllama(url="http://127.0.0.1:11434", tentativi=1)
    C.esegui_concorrenza(cliente, "alfa:4b", [1], tmp_path, endpoint="chat")
    assert server.richieste == 2


def test_endpoint_non_valido(tmp_path):
    with pytest.raises(ValueError):
        C.esegui_concorrenza(ollama.ClienteOllama(), "alfa:4b", [2], tmp_path, endpoint="altro")
