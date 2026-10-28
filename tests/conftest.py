"""Fixture comuni. Nessun test usa la rete: le connessioni socket sono
bloccate, Ollama e' simulato sostituendo ``dike.ollama._http`` e il
recupero di Clio sostituendo ``clio.retrieval.cerca``."""

from __future__ import annotations

import json
import re
import socket
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from clio import retrieval as clio_retrieval  # noqa: E402
from clio.risposta import FRASE_FUORI_DOMINIO  # noqa: E402

from dike import ollama  # noqa: E402

GB = 1_000_000_000


class ReteBloccata(RuntimeError):
    pass


@pytest.fixture(autouse=True)
def blocca_rete(monkeypatch):
    def vietato(*args, **kwargs):
        raise ReteBloccata("Accesso alla rete vietato nei test")

    monkeypatch.setattr(socket.socket, "connect", vietato)
    monkeypatch.setattr(socket.socket, "connect_ex", vietato)
    monkeypatch.setattr(socket, "create_connection", vietato)
    monkeypatch.setattr(socket, "getaddrinfo", vietato)


# ----------------------------------------------------------------------
# KB di prova e recupero simulato
# ----------------------------------------------------------------------

def chunk(documento: str, sezione: str, score: float, testo: str = "Testo di prova.") -> dict:
    return {"id": f"{documento}-{sezione}", "score": score, "payload": {
        "documento": documento, "sezione": sezione, "citazione": f"{documento}, {sezione}",
        "titolo_sezione": f"Titolo {sezione}", "testo": testo, "pagine": [1]}}


DOMANDE = {
    "G1": "Entro quanto tempo va notificata una violazione?",
    "G2": "Quali obblighi di segnalazione prevede la NIS2?",
    "G3": "Chi ha approvato il documento?",
    "N1": "Qual è l'aliquota IVA ordinaria?",
    "N2": "Che tempo fa domani?",
}

CONTESTO = {
    DOMANDE["G1"]: [chunk("GDPR", "art. 33", 0.71), chunk("GDPR", "art. 34", 0.62)],
    DOMANDE["G2"]: [chunk("NIS2", "art. 23", 0.68)],
    DOMANDE["G3"]: [chunk("GDPR", "art. 1", 0.55)],
    DOMANDE["N1"]: [chunk("GDPR", "art. 4", 0.51)],
    DOMANDE["N2"]: [],                     # nessun chunk: l'LLM non va chiamato
}


@pytest.fixture
def kb_prova(tmp_path, monkeypatch):
    """Progetto Clio temporaneo con una KB 'prova' (3 golden, 2 negative)."""
    monkeypatch.setenv("CLIO_BASE_DIR", str(tmp_path / "clio"))
    for var in ("OLLAMA_URL", "QDRANT_URL", "CLIO_EMBED_MODEL", "CLIO_LLM_MODEL", "CLIO_MIN_SCORE"):
        monkeypatch.delenv(var, raising=False)
    cartella = tmp_path / "clio" / "kb" / "prova"
    (cartella / "corpus").mkdir(parents=True)
    (cartella / "kb.json").write_text(json.dumps({"profilo_chunking": "legislativo_ue"}), encoding="utf-8")
    (cartella / "golden_set.csv").write_text(
        "id,domanda,documento_atteso,sezione_attesa,risposta_attesa,tipo_test,difficolta,tipo_risposta\n"
        f"G1,{DOMANDE['G1']},GDPR,art. 33,72 ore,retrieval,facile,valore\n"
        f"G2,{DOMANDE['G2']},NIS2,art. 23,Segnalazione in più fasi,retrieval,media,elenco\n"
        f"G3,{DOMANDE['G3']},GDPR,(metadati),La direzione,retrieval,difficile,valore\n",
        encoding="utf-8")
    (cartella / "negative_set.csv").write_text(
        "id,domanda,nota\n"
        f"N1,{DOMANDE['N1']},fiscalità\n"
        f"N2,{DOMANDE['N2']},meteo\n", encoding="utf-8")
    from clio import config
    return config.get_kb("prova")


class RecuperoFinto:
    def __init__(self):
        self.chiamate: list[str] = []

    def __call__(self, kb, domanda, top_k):
        self.chiamate.append(domanda)
        return [dict(c) for c in CONTESTO[domanda][:top_k]], 0.01, 0.01


@pytest.fixture
def recupero(monkeypatch):
    finto = RecuperoFinto()
    monkeypatch.setattr(clio_retrieval, "cerca", finto)
    return finto


# ----------------------------------------------------------------------
# Ollama simulato
# ----------------------------------------------------------------------

class Orologio:
    def __init__(self):
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t


RISPOSTE_BUONE = {
    "G1": "Entro 72 ore dalla scoperta [GDPR, art. 33].",
    "G2": "Notifica preliminare e relazione finale [NIS2, art. 23].",
    "G3": "Dalla direzione [GDPR, art. 1].",
    "N1": FRASE_FUORI_DOMINIO,
}


class OllamaFinto:
    """Sostituisce dike.ollama._http. Per ogni modello si configurano le
    risposte (per id di domanda), la latenza simulata e la memoria."""

    def __init__(self, orologio: Orologio):
        self.orologio = orologio
        self.risposte: dict[str, dict[str, str]] = {}
        self.latenze: dict[str, float] = {}
        self.memoria: dict[str, int] = {}
        self.timeout_da_dare: dict[str, int] = {}    # modello -> quante scadenze simulare
        self.risposte_per_run: dict[str, dict[str, list[str]]] = {}
        self.prompt: dict[str, list[str]] = {}
        self.caricati: set[str] = set()
        self.chiamate: list[tuple[str, str, dict | None]] = []
        self._conteggio: dict[tuple[str, str], int] = {}

    def modello(self, nome: str, risposte: dict | None = None, latenza: float = 10.0,
                memoria_gb: float = 3.0) -> None:
        self.risposte[nome] = dict(RISPOSTE_BUONE if risposte is None else risposte)
        self.latenze[nome] = latenza
        self.memoria[nome] = int(memoria_gb * GB)

    @staticmethod
    def id_domanda(prompt: str) -> str:
        testo = re.search(r"DOMANDA: (.*)\n", prompt).group(1)
        return next(k for k, v in DOMANDE.items() if v == testo)

    def generazioni(self, modello: str) -> int:
        return len(self.prompt.get(modello, []))

    def __call__(self, metodo, url, corpo=None, timeout=60.0):
        self.chiamate.append((metodo, url, corpo))
        if url.endswith("/api/ps"):
            return {"models": [{"name": m, "model": m, "size": self.memoria[m], "size_vram": 0}
                               for m in sorted(self.caricati)]}
        modello = corpo["model"]
        if modello not in self.risposte:
            raise ollama.ErroreOllama(f"HTTP 404 modello {modello} non trovato")
        if url.endswith("/api/generate") and not corpo.get("prompt"):
            if corpo.get("keep_alive") == 0:
                self.caricati.discard(modello)
            else:
                self.caricati.add(modello)
            return {"response": "", "done_reason": "load"}
        prompt = corpo["prompt"] if "prompt" in corpo else corpo["messages"][-1]["content"]
        if self.timeout_da_dare.get(modello):
            self.timeout_da_dare[modello] -= 1
            self.orologio.t += timeout
            raise ollama.ScadenzaOllama("timeout simulato")
        self.prompt.setdefault(modello, []).append(prompt)
        self.orologio.t += self.latenze[modello]
        ident = self.id_domanda(prompt)
        n = self._conteggio[(modello, ident)] = self._conteggio.get((modello, ident), 0) + 1
        per_run = self.risposte_per_run.get(modello, {}).get(ident)
        testo = per_run[min(n, len(per_run)) - 1] if per_run else self.risposte[modello].get(ident, "Boh.")
        if url.endswith("/api/chat"):
            return {"message": {"role": "assistant", "content": testo}}
        return {"response": testo}


@pytest.fixture
def orologio():
    return Orologio()


@pytest.fixture
def ollama_finto(monkeypatch, orologio):
    finto = OllamaFinto(orologio)
    monkeypatch.setattr(ollama, "_http", finto)
    return finto


@pytest.fixture
def cliente(orologio):
    return ollama.ClienteOllama(url="http://127.0.0.1:11434", timeout=100, tentativi=2,
                                orologio=orologio, dormi=lambda s: None)
