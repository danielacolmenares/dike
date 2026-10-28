"""Statistiche sui tempi, determinismo, memoria e client Ollama."""

from __future__ import annotations

import urllib.error
import urllib.request

import pytest

from dike import ollama
from dike.memoria import MemoriaModello, leggi_memoria, supera_limite
from dike.statistiche import confronta_esecuzioni, percentile, statistiche_tempi

GB = 1_000_000_000


# ----------------------------------------------------------------------
# Tempi
# ----------------------------------------------------------------------

def test_percentile_interpolato():
    valori = list(range(1, 11))             # 1..10
    assert percentile(valori, 90) == pytest.approx(9.1)
    assert percentile(valori, 50) == pytest.approx(5.5)
    assert percentile([7.0], 90) == 7.0
    assert percentile([], 90) is None


def test_statistiche_tempi():
    st = statistiche_tempi([20.0, 30.0, 25.0, 62.1, 21.5])
    assert st == {"n": 5, "media": 31.7, "mediana": 25.0, "min": 20.0, "max": 62.1, "p90": 49.3}


def test_statistiche_tempi_vuote():
    assert statistiche_tempi([])["media"] is None


# ----------------------------------------------------------------------
# Determinismo
# ----------------------------------------------------------------------

def test_confronto_esecuzioni():
    risposte = {
        ("golden", "G1"): ["A", "A"],
        ("golden", "G2"): ["B", "B "],          # spazi finali ignorati
        ("golden", "G3"): ["C", "c"],           # maiuscole no: testo diverso
        ("negative", "N1"): ["solo una"],       # non confrontabile
    }
    esito = confronta_esecuzioni(risposte)
    assert esito == {"identiche": 2, "totale": 3, "diverse": [("golden", "G3")]}


def test_confronto_tre_esecuzioni():
    assert confronta_esecuzioni({("g", "1"): ["x", "x", "y"]})["identiche"] == 0


# ----------------------------------------------------------------------
# Memoria
# ----------------------------------------------------------------------

def test_supera_limite():
    assert supera_limite(MemoriaModello("m", True, int(9.3 * GB)), 8)
    assert not supera_limite(MemoriaModello("m", True, int(3.3 * GB)), 8)
    assert not supera_limite(MemoriaModello("m", True, int(9.3 * GB)), None)
    assert not supera_limite(MemoriaModello("m", False), 8)


def test_leggi_memoria_da_api_ps(ollama_finto, cliente):
    ollama_finto.modello("gemma3:4b", memoria_gb=3.3)
    ollama_finto.modello("llama3.2", memoria_gb=2.8)
    ollama_finto.caricati |= {"gemma3:4b"}
    mem = leggi_memoria(cliente, "gemma3:4b")
    assert mem.trovato and mem.size_gb == 3.3 and mem.size_vram_gb == 0
    assert not leggi_memoria(cliente, "llama3.2").trovato


def test_leggi_memoria_nome_senza_tag(monkeypatch, cliente):
    monkeypatch.setattr(ollama, "_http", lambda *a, **k: {"models": [
        {"name": "phi4-mini:latest", "model": "phi4-mini:latest", "size": 4 * GB, "size_vram": GB}]})
    mem = leggi_memoria(cliente, "phi4-mini")
    assert mem.trovato and mem.size_gb == 4.0 and mem.size_vram_gb == 1.0


# ----------------------------------------------------------------------
# Client Ollama
# ----------------------------------------------------------------------

def test_genera_invia_opzioni_deterministiche(ollama_finto, cliente):
    ollama_finto.modello("alfa:4b", latenza=12)
    prompt = "FONTI\nDOMANDA: Entro quanto tempo va notificata una violazione?\nRISPOSTA:"
    g = cliente.genera("alfa:4b", prompt)
    assert g.testo.startswith("Entro 72 ore") and g.durata == 12 and g.tentativi == 1
    corpo = ollama_finto.chiamate[-1][2]
    assert corpo["options"]["temperature"] == 0 and corpo["options"]["seed"] == 42
    assert corpo["stream"] is False


def test_chat(ollama_finto, cliente):
    ollama_finto.modello("alfa:4b")
    prompt = "DOMANDA: Entro quanto tempo va notificata una violazione?\n"
    g = cliente.chat("alfa:4b", [{"role": "user", "content": prompt}])
    assert "72 ore" in g.testo
    assert ollama_finto.chiamate[-1][1].endswith("/api/chat")


def test_risposta_vuota_e_ripetuta_poi_errore(monkeypatch, cliente):
    monkeypatch.setattr(ollama, "_http", lambda *a, **k: {"response": "  "})
    with pytest.raises(ollama.ErroreGenerazione) as exc:
        cliente.genera("alfa:4b", "x")
    assert exc.value.tentativi == 2 and exc.value.timeout == 0


def test_http_traduce_il_timeout(monkeypatch):
    def scade(*a, **k):
        raise urllib.error.URLError(TimeoutError("timed out"))
    monkeypatch.setattr(urllib.request, "urlopen", scade)
    with pytest.raises(ollama.ScadenzaOllama):
        ollama._http("POST", "http://127.0.0.1:11434/api/generate", {}, 1)

    def scade_in_lettura(*a, **k):
        raise TimeoutError("read timed out")
    monkeypatch.setattr(urllib.request, "urlopen", scade_in_lettura)
    with pytest.raises(ollama.ScadenzaOllama):
        ollama._http("POST", "http://127.0.0.1:11434/api/generate", {}, 1)


def test_http_senza_mock_non_esce_in_rete():
    # la guardia dei test blocca qualsiasi connessione
    with pytest.raises((ollama.ErroreOllama, RuntimeError)):
        ollama._http("GET", "http://127.0.0.1:11434/api/ps", None, 1)
