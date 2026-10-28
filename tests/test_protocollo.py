"""Protocollo completo con Ollama e recupero simulati."""

from __future__ import annotations

import json

from clio.risposta import FRASE_FUORI_DOMINIO

from dike import risultati as R
from dike.memoria import ESITO_VINCOLO_HARDWARE
from dike.protocollo import ESITO_COMPLETATO, ESITO_TROPPO_LENTO, Opzioni, esegui_protocollo


def _esegui(kb, cliente, tmp_path, modelli, **opzioni):
    cartella = esegui_protocollo(kb, modelli, tmp_path / "ris", cliente, Opzioni(**opzioni))
    return cartella, R.leggi_csv(cartella / R.DETTAGLIO)


# ----------------------------------------------------------------------
# Stesso contesto per tutti i modelli
# ----------------------------------------------------------------------

def test_recupero_una_volta_per_domanda_e_contesto_identico(kb_prova, recupero, ollama_finto,
                                                            cliente, tmp_path):
    ollama_finto.modello("alfa:4b")
    ollama_finto.modello("beta:3b", latenza=5)
    cartella, righe = _esegui(kb_prova, cliente, tmp_path, ["alfa:4b", "beta:3b"], run=2)

    # 5 domande, 2 modelli, 2 esecuzioni: il recupero avviene 5 volte, non 20
    assert len(recupero.chiamate) == 5
    # stessi prompt, nello stesso ordine, per entrambi i modelli e le esecuzioni
    assert ollama_finto.prompt["alfa:4b"] == ollama_finto.prompt["beta:3b"]
    assert len(ollama_finto.prompt["alfa:4b"]) == 2 * 4        # N2 non ha chunk: niente LLM
    impronte = {}
    for r in righe:
        impronte.setdefault((r["set"], r["id"]), set()).add(r["impronta_contesto"])
    assert all(len(v) == 1 for v in impronte.values())
    contesto = json.loads((cartella / R.CONTESTO).read_text(encoding="utf-8"))
    assert "Stesso contesto" in contesto["principio"]
    assert set(contesto["voci"]) == {"golden:G1", "golden:G2", "golden:G3", "negative:N1", "negative:N2"}


def test_domanda_senza_chunk_non_chiama_il_modello(kb_prova, recupero, ollama_finto, cliente, tmp_path):
    ollama_finto.modello("alfa:4b")
    _, righe = _esegui(kb_prova, cliente, tmp_path, ["alfa:4b"], run=1)
    n2 = next(r for r in righe if r["id"] == "N2")
    assert n2["chiamata_llm"] == "no"
    assert n2["risposta"] == FRASE_FUORI_DOMINIO
    assert n2["tempo_s"] == ""
    assert n2["rifiuto_corretto"] == "sì"


def test_file_prodotti(kb_prova, recupero, ollama_finto, cliente, tmp_path):
    ollama_finto.modello("alfa:4b")
    cartella, righe = _esegui(kb_prova, cliente, tmp_path, ["alfa:4b"], run=2)
    for nome in (R.DETTAGLIO, R.MEMORIA, R.MODELLI, R.PROTOCOLLO, R.CONTESTO, R.REVISIONE,
                 R.COMPORTAMENTO, R.RIEPILOGO_MD, R.RIEPILOGO_CSV):
        assert (cartella / nome).exists(), nome
    assert len(righe) == 10
    revisione = R.leggi_csv(cartella / R.REVISIONE)
    assert {r["run"] for r in revisione} == {"1"} and len(revisione) == 5
    assert all(r["verdetto_umano"] == "" for r in revisione)
    protocollo = json.loads((cartella / R.PROTOCOLLO).read_text(encoding="utf-8"))
    assert protocollo["domande_golden"] == 3 and protocollo["domande_negative"] == 2
    assert protocollo["ricerche_effettuate"] == 5 and protocollo["run"] == 2


# ----------------------------------------------------------------------
# Memoria e vincolo hardware
# ----------------------------------------------------------------------

def test_vincolo_hardware_esclude_il_modello(kb_prova, recupero, ollama_finto, cliente, tmp_path):
    ollama_finto.modello("piccolo:4b", memoria_gb=3.3)
    ollama_finto.modello("grande:14b", memoria_gb=9.3)
    cartella, righe = _esegui(kb_prova, cliente, tmp_path, ["piccolo:4b", "grande:14b"],
                              run=2, ram_max_gb=8)
    modelli = json.loads((cartella / R.MODELLI).read_text(encoding="utf-8"))
    assert modelli["esito"] == {"piccolo:4b": ESITO_COMPLETATO, "grande:14b": ESITO_VINCOLO_HARDWARE}
    assert "9.3 GB" in modelli["note"]["grande:14b"]
    assert ollama_finto.generazioni("grande:14b") == 0
    assert all(r["modello"] == "piccolo:4b" for r in righe)
    # il modello escluso viene scaricato subito e non ricaricato nella seconda esecuzione
    assert "grande:14b" not in ollama_finto.caricati
    memoria = R.leggi_csv(cartella / R.MEMORIA)
    assert [(m["modello"], m["run"]) for m in memoria] == [("piccolo:4b", "1"), ("grande:14b", "1"),
                                                           ("piccolo:4b", "2")]
    assert memoria[1]["size_gb"] == "9.3" and memoria[1]["esito"] == ESITO_VINCOLO_HARDWARE
    md = (cartella / R.RIEPILOGO_MD).read_text(encoding="utf-8")
    assert "escluso: vincolo hardware" in md


def test_senza_limite_nessuna_esclusione(kb_prova, recupero, ollama_finto, cliente, tmp_path):
    ollama_finto.modello("grande:14b", memoria_gb=9.3)
    cartella, _ = _esegui(kb_prova, cliente, tmp_path, ["grande:14b"], run=1)
    assert json.loads((cartella / R.MODELLI).read_text())["esito"]["grande:14b"] == ESITO_COMPLETATO


def test_modello_non_disponibile_segnato_come_errore(kb_prova, recupero, ollama_finto, cliente, tmp_path):
    ollama_finto.modello("alfa:4b")
    cartella, righe = _esegui(kb_prova, cliente, tmp_path, ["alfa:4b", "inesistente:1b"], run=1)
    esito = json.loads((cartella / R.MODELLI).read_text())["esito"]
    assert esito["inesistente:1b"].startswith("errore:")
    assert {r["modello"] for r in righe} == {"alfa:4b"}


# ----------------------------------------------------------------------
# Arresto anticipato
# ----------------------------------------------------------------------

def test_arresto_anticipato_modello_lento(kb_prova, recupero, ollama_finto, cliente, tmp_path):
    ollama_finto.modello("veloce:3b", latenza=25)
    ollama_finto.modello("lento:7b", latenza=67)
    cartella, righe = _esegui(kb_prova, cliente, tmp_path, ["lento:7b", "veloce:3b"],
                              run=2, stop_latenza_s=60, stop_dopo=2)
    info = json.loads((cartella / R.MODELLI).read_text(encoding="utf-8"))
    assert info["esito"]["lento:7b"] == ESITO_TROPPO_LENTO
    assert "67.0 s" in info["note"]["lento:7b"]
    assert ollama_finto.generazioni("lento:7b") == 2          # fermato dopo 2 risposte
    assert sum(r["modello"] == "lento:7b" for r in righe) == 2
    assert info["esito"]["veloce:3b"] == ESITO_COMPLETATO
    assert ollama_finto.generazioni("veloce:3b") == 8
    riepilogo = R.leggi_csv(cartella / R.RIEPILOGO_CSV)
    assert riepilogo[-1]["modello"] == "lento:7b"
    assert riepilogo[-1]["totale"] == "–"
    assert riepilogo[-1]["tempo_medio_s"] == "67.0"


def test_arresto_non_scatta_sotto_soglia(kb_prova, recupero, ollama_finto, cliente, tmp_path):
    ollama_finto.modello("alfa:4b", latenza=30)
    cartella, _ = _esegui(kb_prova, cliente, tmp_path, ["alfa:4b"], run=1, stop_latenza_s=60, stop_dopo=2)
    assert json.loads((cartella / R.MODELLI).read_text())["esito"]["alfa:4b"] == ESITO_COMPLETATO


# ----------------------------------------------------------------------
# Determinismo, timeout e ripetizioni
# ----------------------------------------------------------------------

def test_determinismo_rileva_risposte_diverse(kb_prova, recupero, ollama_finto, cliente, tmp_path):
    ollama_finto.modello("stabile:4b")
    ollama_finto.modello("instabile:3b")
    ollama_finto.risposte_per_run["instabile:3b"] = {
        "G1": ["Entro 72 ore [GDPR, art. 33].", "Entro tre giorni [GDPR, art. 33]."]}
    cartella, _ = _esegui(kb_prova, cliente, tmp_path, ["stabile:4b", "instabile:3b"], run=2)
    riepilogo = {r["modello"]: r for r in R.leggi_csv(cartella / R.RIEPILOGO_CSV)}
    assert riepilogo["stabile:4b"]["determinismo"] == "sì (5/5)"
    assert riepilogo["instabile:3b"]["determinismo"] == "no (4/5)"
    md = (cartella / R.RIEPILOGO_MD).read_text(encoding="utf-8")
    assert "golden:G1" in md
    assert "non deterministico" in riepilogo["instabile:3b"]["comportamento"]


def test_un_solo_run_determinismo_non_disponibile(kb_prova, recupero, ollama_finto, cliente, tmp_path):
    ollama_finto.modello("alfa:4b")
    cartella, _ = _esegui(kb_prova, cliente, tmp_path, ["alfa:4b"], run=1)
    assert R.leggi_csv(cartella / R.RIEPILOGO_CSV)[0]["determinismo"] == "n.d. (1 run)"


def test_timeout_e_ripetizioni_contati(kb_prova, recupero, ollama_finto, cliente, tmp_path):
    ollama_finto.modello("alfa:4b", latenza=10)
    ollama_finto.timeout_da_dare["alfa:4b"] = 1
    cartella, righe = _esegui(kb_prova, cliente, tmp_path, ["alfa:4b"], run=1)
    prima = righe[0]
    assert prima["tentativi"] == "2" and prima["timeout"] == "1"
    assert float(prima["tempo_s"]) == 110.0          # 100 s di timeout + 10 s di risposta
    riepilogo = R.leggi_csv(cartella / R.RIEPILOGO_CSV)[0]
    assert riepilogo["timeout"] == "1" and riepilogo["ripetizioni"] == "1"
    assert riepilogo["tempo_max_s"] == "110.0"


def test_generazione_fallita_registrata_come_errore(kb_prova, recupero, ollama_finto, cliente, tmp_path):
    ollama_finto.modello("alfa:4b")
    ollama_finto.timeout_da_dare["alfa:4b"] = 2          # entrambi i tentativi scadono
    cartella, righe = _esegui(kb_prova, cliente, tmp_path, ["alfa:4b"], run=1)
    assert "generazione fallita" in righe[0]["errore"]
    assert righe[0]["timeout"] == "2"
    riepilogo = R.leggi_csv(cartella / R.RIEPILOGO_CSV)[0]
    assert riepilogo["errori"] == "1" and riepilogo["timeout"] == "2"
