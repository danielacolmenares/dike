"""Riepilogo, comportamento prevalente e importazione dei verdetti umani."""

from __future__ import annotations

import pytest
from clio.risposta import FRASE_FUORI_DOMINIO

from dike import risultati as R
from dike.protocollo import Opzioni, esegui_protocollo
from dike.riepilogo import genera_riepilogo
from dike.verdetti import ErroreVerdetti, importa_verdetti

RISPOSTE_PRUDENTI = {"G1": FRASE_FUORI_DOMINIO, "G2": FRASE_FUORI_DOMINIO,
                     "G3": "Dalla direzione [GDPR, art. 1].", "N1": FRASE_FUORI_DOMINIO}
RISPOSTE_FANTASIOSE = {
    "G1": "Entro 72 ore [GDPR, art. 33]. " + "Si aggiunga che la norma prevede molte altre cose. " * 12
          + "[GDPR, art. 33, par. 7] e [GDPR, art. 95].",
    "G2": "Notifica [NIS2, art. 23]. " + "Ulteriori dettagli ripetuti a lungo. " * 15,
    "G3": "Dalla direzione [GDPR, art. 1]. " + "Nota sul documento e sul suo contesto. " * 15,
    "N1": "L'aliquota ordinaria è del 22%.",
}


@pytest.fixture
def benchmark(kb_prova, recupero, ollama_finto, cliente, tmp_path):
    ollama_finto.modello("bravo:4b", latenza=30)
    ollama_finto.modello("prudente:3b", RISPOSTE_PRUDENTI, latenza=25)
    ollama_finto.modello("fantasioso:7b", RISPOSTE_FANTASIOSE, latenza=55)
    return esegui_protocollo(kb_prova, ["prudente:3b", "fantasioso:7b", "bravo:4b"],
                             tmp_path / "ris", cliente, Opzioni(run=2))


def _riepilogo(cartella):
    return {r["modello"]: r for r in R.leggi_csv(cartella / R.RIEPILOGO_CSV)}


def test_classifica_e_indicatori_automatici(benchmark):
    righe = R.leggi_csv(benchmark / R.RIEPILOGO_CSV)
    # fantasioso e prudente pari a 3/5: a parita' vince il piu' veloce
    assert [r["modello"] for r in righe] == ["bravo:4b", "prudente:3b", "fantasioso:7b"]
    assert _riepilogo(benchmark)["fantasioso:7b"]["totale"] == "3/5 (auto)"
    r = _riepilogo(benchmark)
    assert r["bravo:4b"]["qualita_golden"] == "3/3 (auto)"
    assert r["bravo:4b"]["rifiuti_negative"] == "2/2"
    assert r["bravo:4b"]["totale"] == "5/5 (auto)"
    assert r["prudente:3b"]["falsi_negativi"] == "2"
    assert r["prudente:3b"]["totale"] == "3/5 (auto)"
    assert r["fantasioso:7b"]["rifiuti_negative"] == "1/2"
    assert r["fantasioso:7b"]["possibili_allucinazioni"] == "1"
    assert r["fantasioso:7b"]["citazioni_inventate"] == "1"   # [GDPR, art. 95] in G1
    assert r["bravo:4b"]["tempo_medio_s"] == "30.0" and r["bravo:4b"]["tempo_p90_s"] == "30.0"
    assert r["bravo:4b"]["memoria_gb"] == "3.0"


def test_comportamento_derivato(benchmark):
    r = _riepilogo(benchmark)
    assert r["bravo:4b"]["comportamento"] == "equilibrato"
    assert r["prudente:3b"]["comportamento"].startswith("prudente/sotto-risponde (2 falsi negativi)")
    fantasioso = r["fantasioso:7b"]["comportamento"]
    for tratto in ("verboso", "citazioni imprecise", "risponde a domande fuori dominio", "lento"):
        assert tratto in fantasioso


def test_comportamento_compilato_a_mano_ha_la_precedenza(benchmark):
    righe = R.leggi_csv(benchmark / R.COMPORTAMENTO)
    for r in righe:
        if r["modello"] == "bravo:4b":
            r["comportamento"] = "completo, unico corretto sui metadati"
    R.scrivi_csv(benchmark / R.COMPORTAMENTO, ["modello", "comportamento_derivato", "comportamento"], righe)
    genera_riepilogo(benchmark)
    assert _riepilogo(benchmark)["bravo:4b"]["comportamento"] == "completo, unico corretto sui metadati"
    # il campo umano sopravvive alla rigenerazione
    assert any(r["comportamento"] for r in R.leggi_csv(benchmark / R.COMPORTAMENTO))


def test_report_markdown(benchmark):
    md = (benchmark / R.RIEPILOGO_MD).read_text(encoding="utf-8")
    assert md.startswith("# Dike – riepilogo del benchmark")
    assert "Stesso contesto per tutti i modelli" in md
    assert "| 1 | bravo:4b | 3/3 (auto) | 2/2 | 5/5 (auto) | sì (5/5) | 30.0 s | 3.0 GB | equilibrato |" in md
    for sezione in ("## Tempi di generazione", "## Memoria", "## Golden Set", "## Negative Set",
                    "## Determinismo", "Verdetti umani non ancora importati"):
        assert sezione in md
    assert "| fantasioso:7b | 1/2 | N1 |" in md


def test_import_verdetti_e_ricalcolo(benchmark):
    revisione = benchmark / R.REVISIONE
    righe = R.leggi_csv(revisione)
    verdetti = {("bravo:4b", "G1"): "OK", ("bravo:4b", "G2"): "ok", ("bravo:4b", "G3"): "PARZIALE",
                ("prudente:3b", "G1"): "ERRATA", ("prudente:3b", "G2"): "ERRATA",
                ("prudente:3b", "G3"): "OK"}
    for r in righe:
        r["verdetto_umano"] = verdetti.get((r["modello"], r["id"]), "")
    R.scrivi_csv(revisione, R.COLONNE_REVISIONE, righe)

    n, md = importa_verdetti(revisione)
    assert n == 6
    r = _riepilogo(benchmark)
    assert r["bravo:4b"]["qualita_golden"] == "2/3"
    assert r["bravo:4b"]["totale"] == "4/5"
    assert r["prudente:3b"]["qualita_golden"] == "1/3"
    assert r["prudente:3b"]["totale"] == "3/5"
    # senza verdetti completi resta la stima automatica
    assert r["fantasioso:7b"]["qualita_golden"].endswith("(auto)")
    testo = md.read_text(encoding="utf-8")
    assert "Verdetti umani non ancora importati" not in testo
    assert "| bravo:4b | 3/3 | 3/3 | 0 | 0 | 2 / 1 / 0 / 0 |" in testo


def test_verdetti_in_piu_riprese_e_cella_vuota_non_cancella(benchmark):
    revisione = benchmark / R.REVISIONE
    righe = R.leggi_csv(revisione)
    for r in righe:
        r["verdetto_umano"] = "OK" if r["modello"] == "bravo:4b" and r["set"] == "golden" else ""
    R.scrivi_csv(revisione, R.COLONNE_REVISIONE, righe)
    importa_verdetti(revisione)
    for r in righe:
        r["verdetto_umano"] = ""
    righe[0]["verdetto_umano"] = "ERRATA"            # prudente:3b G1
    R.scrivi_csv(revisione, R.COLONNE_REVISIONE, righe)
    importa_verdetti(revisione)
    salvati = R.leggi_csv(benchmark / R.VERDETTI)
    assert len(salvati) == 4
    assert _riepilogo(benchmark)["bravo:4b"]["qualita_golden"] == "3/3"


def test_verdetto_umano_su_negative_prevale(benchmark):
    revisione = benchmark / R.REVISIONE
    righe = R.leggi_csv(revisione)
    for r in righe:
        if r["modello"] == "fantasioso:7b" and r["id"] == "N1":
            r["verdetto_umano"] = "OK"   # il revisore giudica accettabile la risposta
    R.scrivi_csv(revisione, R.COLONNE_REVISIONE, righe)
    importa_verdetti(revisione)
    assert _riepilogo(benchmark)["fantasioso:7b"]["rifiuti_negative"] == "2/2"


def test_verdetti_non_validi_bloccano_l_importazione(benchmark):
    revisione = benchmark / R.REVISIONE
    righe = R.leggi_csv(revisione)
    righe[0]["verdetto_umano"] = "FORSE"
    righe[1]["verdetto_umano"] = "OK"
    righe[2]["modello"], righe[2]["verdetto_umano"] = "sconosciuto:1b", "OK"
    R.scrivi_csv(revisione, R.COLONNE_REVISIONE, righe)
    with pytest.raises(ErroreVerdetti) as exc:
        importa_verdetti(revisione)
    assert "riga 2" in str(exc.value) and "FORSE" in str(exc.value)
    assert "sconosciuto:1b" in str(exc.value)
    assert not (benchmark / R.VERDETTI).exists()


def test_riepilogo_su_cartella_non_valida(tmp_path):
    with pytest.raises(FileNotFoundError):
        genera_riepilogo(tmp_path)
