"""Comandi da riga di comando, con servizi simulati."""

from __future__ import annotations

import pytest

from dike import cli, ollama
from dike import risultati as R


@pytest.fixture
def cliente_senza_attese(monkeypatch, orologio):
    originale = ollama.ClienteOllama.__init__

    def init(self, *a, **k):
        k.setdefault("orologio", orologio)
        k.setdefault("dormi", lambda s: None)
        originale(self, *a, **k)
    monkeypatch.setattr(ollama.ClienteOllama, "__init__", init)


def test_esegui_riepilogo_verdetti(kb_prova, recupero, ollama_finto, cliente_senza_attese,
                                   tmp_path, capsys):
    ollama_finto.modello("gemma3:4b", latenza=21.5)
    ollama_finto.modello("llama3.2:3b", latenza=22.3)
    codice = cli.main(["esegui", "--kb", "prova", "--modelli", "gemma3:4b,llama3.2:3b", "--run", "2",
                       "--uscita", str(tmp_path / "ris"), "--ram-max-gb", "12"])
    assert codice == 0
    uscita = capsys.readouterr().out
    assert "# Dike – riepilogo del benchmark" in uscita
    cartelle = list((tmp_path / "ris").iterdir())
    assert len(cartelle) == 1 and cartelle[0].name.startswith("prova_")
    cartella = cartelle[0]

    assert cli.main(["riepilogo", str(cartella)]) == 0
    assert "## Classifica" in capsys.readouterr().out

    revisione = cartella / R.REVISIONE
    righe = R.leggi_csv(revisione)
    for r in righe:
        r["verdetto_umano"] = "OK"
    R.scrivi_csv(revisione, R.COLONNE_REVISIONE, righe)
    assert cli.main(["verdetti", str(revisione)]) == 0
    assert "Verdetti importati: 10" in capsys.readouterr().out
    assert R.leggi_csv(cartella / R.RIEPILOGO_CSV)[0]["totale"] == "5/5"


def test_kb_inesistente(kb_prova, capsys, tmp_path):
    assert cli.main(["esegui", "--kb", "nessuna", "--modelli", "a", "--uscita", str(tmp_path)]) == 1
    assert "non trovata" in capsys.readouterr().err


def test_clio_dir(kb_prova, recupero, ollama_finto, cliente_senza_attese, tmp_path, monkeypatch):
    monkeypatch.delenv("CLIO_BASE_DIR")
    ollama_finto.modello("gemma3:4b")
    assert cli.main(["esegui", "--kb", "prova", "--modelli", "gemma3:4b", "--run", "1",
                     "--clio-dir", str(tmp_path / "clio"), "--uscita", str(tmp_path / "ris")]) == 0


def test_concorrenza_da_cli(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(ollama, "_http", lambda m, u, c=None, t=60: (
        {"models": []} if u.endswith("/api/ps") else {"response": "ok" if c.get("prompt") else ""}))
    assert cli.main(["concorrenza", "--modello", "gemma3:4b", "--paralleli", "1,2",
                     "--uscita", str(tmp_path)]) == 0
    assert "verifica della concorrenza: gemma3:4b" in capsys.readouterr().out


def test_paralleli_non_validi():
    with pytest.raises(SystemExit):
        cli.main(["concorrenza", "--modello", "x", "--paralleli", "0,2"])


def test_verdetti_file_mancante(tmp_path, capsys):
    assert cli.main(["verdetti", str(tmp_path / "manca.csv")]) == 1
    assert "non trovato" in capsys.readouterr().err
