"""Indicatori automatici: citazioni attese, inventate, rifiuti."""

from __future__ import annotations

from clio.risposta import FRASE_FUORI_DOMINIO, fonti_da_risultati
from conftest import chunk

from dike import indicatori

FONTI = fonti_da_risultati([chunk("GDPR", "art. 33", 0.7), chunk("GDPR", "art. 34", 0.6)])
RIGA = {"documento_atteso": "GDPR", "sezione_attesa": "art. 33"}


def test_articolo_atteso_trovato_in_fonti_e_citazioni():
    ind = indicatori.indicatori_golden(RIGA, "Entro 72 ore [GDPR, art. 33, par. 1].", FONTI)
    assert ind["atteso_doc_in_fonti"] and ind["atteso_sez_in_fonti"]
    assert ind["atteso_doc_in_citazioni"] and ind["atteso_sez_in_citazioni"]
    assert ind["citazioni_inventate"] == []
    assert not ind["falso_negativo"]
    assert ind["esito_automatico"] == indicatori.COERENTE


def test_citazione_inventata():
    ind = indicatori.indicatori_golden(RIGA, "Entro 72 ore [GDPR, art. 33] e [GDPR, art. 99].", FONTI)
    assert ind["citazioni_inventate"] == ["GDPR, art. 99"]
    assert ind["atteso_sez_in_citazioni"]
    assert ind["esito_automatico"] == indicatori.DA_VERIFICARE


def test_articolo_giusto_tra_le_fonti_ma_non_citato_dal_modello():
    ind = indicatori.indicatori_golden(RIGA, "Entro 72 ore [GDPR, art. 34].", FONTI)
    assert ind["atteso_sez_in_fonti"]
    assert ind["atteso_doc_in_citazioni"] and not ind["atteso_sez_in_citazioni"]
    assert ind["esito_automatico"] == indicatori.DA_VERIFICARE


def test_rifiuto_sbagliato_e_falso_negativo():
    ind = indicatori.indicatori_golden(RIGA, FRASE_FUORI_DOMINIO, FONTI)
    assert ind["falso_negativo"] and ind["rifiuto"]
    # come in Clio: se il modello rifiuta, le fonti non vengono mostrate
    assert not ind["atteso_doc_in_fonti"]
    assert ind["esito_automatico"] == indicatori.FALSO_NEGATIVO


def test_rifiuto_riconosciuto_anche_con_testo_attorno():
    testo = "non trovo questa informazione nei documenti!! Mi dispiace."
    assert indicatori.indicatori_golden(RIGA, testo, FONTI)["falso_negativo"]


def test_sezione_non_verificabile_controlla_solo_il_documento():
    riga = {"documento_atteso": "GDPR", "sezione_attesa": "(metadati)"}
    ind = indicatori.indicatori_golden(riga, "Dalla direzione [GDPR, art. 34].", FONTI)
    assert ind["atteso_sez_in_citazioni"] and ind["atteso_sez_in_fonti"]
    assert not indicatori.sezione_verificabile("(metadati)")
    assert indicatori.sezione_verificabile("art. 5")


def test_documento_sbagliato():
    riga = {"documento_atteso": "NIS2", "sezione_attesa": "art. 33"}
    ind = indicatori.indicatori_golden(riga, "Testo [GDPR, art. 33].", FONTI)
    assert not ind["atteso_doc_in_fonti"] and not ind["atteso_doc_in_citazioni"]


def test_negative_rifiuto_corretto():
    ind = indicatori.indicatori_negative(FRASE_FUORI_DOMINIO, FONTI)
    assert ind["rifiuto_corretto"] and not ind["possibile_allucinazione"]
    assert ind["esito_automatico"] == "rifiuto_corretto"


def test_negative_risposta_data_e_possibile_allucinazione():
    ind = indicatori.indicatori_negative("L'aliquota ordinaria è il 22% [GDPR, art. 4].", FONTI)
    assert ind["possibile_allucinazione"] and not ind["rifiuto_corretto"]
    assert ind["citazioni_inventate"] == ["GDPR, art. 4"]
