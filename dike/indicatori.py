"""Indicatori automatici per ogni risposta.

Aiutano la revisione umana, non la sostituiscono: dicono dove guardare.

Golden Set (la risposta esiste nei documenti):
  - documento/sezione attesi tra le FONTI DETERMINISTICHE di Clio (quelle
    costruite dai metadati dei chunk, mostrate solo se il modello risponde);
  - documento/sezione attesi tra le CITAZIONI INLINE scritte dal modello;
  - falso negativo: il modello dice "Non trovo questa informazione nei
    documenti." anche se la risposta c'e';
  - citazioni inventate: citazioni inline che non corrispondono a nessun
    chunk recuperato (confronto fatto con la funzione di Clio).

Negative Set (la risposta NON esiste):
  - rifiuto corretto se il modello usa la frase prevista;
  - qualsiasi altra risposta e' una possibile allucinazione da verificare.

Allucinazione e citazione imprecisa restano conteggi separati: citare
"art. 33, par. 2" invece di "art. 33" non e' inventare un fatto.
"""

from __future__ import annotations

from clio import retrieval as clio_retrieval
from clio import risposta as clio_risposta

# Esito automatico sul Golden Set (stima, da confermare con il verdetto umano)
COERENTE = "coerente"              # risponde, cita il riferimento atteso, nessuna citazione inventata
DA_VERIFICARE = "da_verificare"    # risponde ma manca la citazione attesa o ce n'e' una inventata
FALSO_NEGATIVO = "falso_negativo"  # rifiuta anche se la risposta c'e'


def sezione_verificabile(sezione_attesa: str) -> bool:
    s = (sezione_attesa or "").strip()
    return bool(s) and not s.startswith(clio_retrieval.SEZIONE_NON_VERIFICABILE_PREFIX)


def _corrisponde(documento_atteso: str, sezione_attesa: str, documento: str,
                 sezione: str) -> tuple[bool, bool]:
    doc_ok = clio_retrieval.documento_corrisponde(documento_atteso, {"documento": documento})
    if not doc_ok:
        return False, False
    if not sezione_verificabile(sezione_attesa):
        return True, True
    return True, clio_retrieval.sezione_corrisponde(sezione_attesa, sezione)


def cerca_atteso(documento_atteso: str, sezione_attesa: str,
                 riferimenti: list[tuple[str, str]]) -> tuple[bool, bool]:
    """(documento trovato, documento+sezione trovati) tra i riferimenti
    (documento, sezione)."""
    doc = sez = False
    for documento, sezione in riferimenti:
        d, s = _corrisponde(documento_atteso, sezione_attesa, documento, sezione)
        doc, sez = doc or d, sez or s
    return doc, sez


def _dividi_citazione(citazione: str) -> tuple[str, str]:
    documento, _, sezione = citazione.partition(",")
    return documento.strip(), sezione.strip()


def indicatori_golden(riga: dict, testo: str, fonti: list[dict]) -> dict:
    documento_atteso = (riga.get("documento_atteso") or "").strip()
    sezione_attesa = (riga.get("sezione_attesa") or "").strip()
    rifiuto = clio_risposta.dichiara_fuori_dominio(testo)
    # Come in Clio: se il modello rifiuta, le fonti non vengono mostrate.
    fonti_mostrate = [] if rifiuto else fonti
    citazioni = clio_risposta.estrai_citazioni(testo)
    inventate = clio_risposta.citazioni_non_recuperate(testo, fonti)

    doc_fonti, sez_fonti = cerca_atteso(documento_atteso, sezione_attesa,
                                        [(f.get("documento") or "", f.get("sezione") or "")
                                         for f in fonti_mostrate])
    doc_cit, sez_cit = cerca_atteso(documento_atteso, sezione_attesa,
                                    [_dividi_citazione(c) for c in citazioni])
    if rifiuto:
        esito = FALSO_NEGATIVO
    elif sez_cit and not inventate:
        esito = COERENTE
    else:
        esito = DA_VERIFICARE
    return {
        "rifiuto": rifiuto,
        "falso_negativo": rifiuto,
        "atteso_doc_in_fonti": doc_fonti,
        "atteso_sez_in_fonti": sez_fonti,
        "atteso_doc_in_citazioni": doc_cit,
        "atteso_sez_in_citazioni": sez_cit,
        "citazioni_inline": citazioni,
        "citazioni_inventate": inventate,
        "esito_automatico": esito,
    }


def indicatori_negative(testo: str, fonti: list[dict]) -> dict:
    rifiuto = clio_risposta.dichiara_fuori_dominio(testo)
    return {
        "rifiuto": rifiuto,
        "rifiuto_corretto": rifiuto,
        "possibile_allucinazione": not rifiuto,
        "citazioni_inline": clio_risposta.estrai_citazioni(testo),
        "citazioni_inventate": clio_risposta.citazioni_non_recuperate(testo, fonti),
        "esito_automatico": "rifiuto_corretto" if rifiuto else "possibile_allucinazione",
    }
