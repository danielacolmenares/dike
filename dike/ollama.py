"""Client HTTP minimo verso Ollama, pensato per misurare.

Rispetto al client di Clio (che serve a rispondere) questo client serve a
cronometrare: per ogni generazione restituisce durata, tentativi usati e
timeout incontrati, e permette di caricare/scaricare un modello e di
leggere la memoria occupata (``/api/ps``).

Tutte le chiamate di rete passano da ``_http``: i test la sostituiscono e
non toccano mai la rete.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Callable

from clio import config as clio_config

# Stesse opzioni usate da Clio: senza temperatura 0 e seed fisso il
# confronto tra due esecuzioni non avrebbe senso.
OPZIONI_DETERMINISTICHE = {"temperature": 0, "seed": 42}

TIMEOUT_DEFAULT = 300.0
TENTATIVI_DEFAULT = 2
KEEP_ALIVE = "30m"


class ErroreOllama(RuntimeError):
    """Ollama non raggiungibile o risposta non valida."""


class ScadenzaOllama(ErroreOllama):
    """La richiesta ha superato il timeout."""


class ErroreGenerazione(ErroreOllama):
    """Generazione fallita dopo tutti i tentativi. Conserva i conteggi."""

    def __init__(self, messaggio: str, tentativi: int, timeout: int, durata: float):
        super().__init__(messaggio)
        self.tentativi = tentativi
        self.timeout = timeout
        self.durata = durata


@dataclass
class Generazione:
    testo: str
    durata: float          # secondi, compresi gli eventuali tentativi falliti
    tentativi: int = 1     # 1 = riuscita al primo colpo
    timeout: int = 0       # quanti tentativi sono scaduti

    @property
    def ripetizioni(self) -> int:
        return self.tentativi - 1


def _http(metodo: str, url: str, corpo: dict | None = None, timeout: float = 60.0) -> dict:
    """Una sola chiamata JSON, senza retry (li gestisce il chiamante)."""
    data = json.dumps(corpo, ensure_ascii=False).encode("utf-8") if corpo is not None else None
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"},
                                 method=metodo)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            testo = resp.read().decode("utf-8")
    except TimeoutError as exc:
        raise ScadenzaOllama(f"timeout dopo {timeout:.0f} s ({metodo} {url})") from exc
    except urllib.error.HTTPError as exc:
        raise ErroreOllama(f"HTTP {exc.code} ({metodo} {url})") from exc
    except urllib.error.URLError as exc:
        if isinstance(exc.reason, TimeoutError):
            raise ScadenzaOllama(f"timeout dopo {timeout:.0f} s ({metodo} {url})") from exc
        raise ErroreOllama(f"{exc.reason} ({metodo} {url})") from exc
    try:
        return json.loads(testo) if testo.strip() else {}
    except json.JSONDecodeError as exc:
        raise ErroreOllama(f"risposta non JSON ({metodo} {url})") from exc


class ClienteOllama:
    """Client con misura dei tempi. ``orologio`` e ``dormi`` sono
    iniettabili per i test."""

    def __init__(self, url: str | None = None, timeout: float = TIMEOUT_DEFAULT,
                 tentativi: int = TENTATIVI_DEFAULT, attesa: float = 2.0,
                 orologio: Callable[[], float] = time.perf_counter,
                 dormi: Callable[[float], None] = time.sleep):
        self.url = (url or clio_config.ollama_url()).rstrip("/")
        self.timeout = timeout
        self.tentativi = max(1, tentativi)
        self.attesa = attesa
        self.orologio = orologio
        self.dormi = dormi

    # ------------------------------------------------------------------
    # Generazione
    # ------------------------------------------------------------------

    def _con_tentativi(self, percorso: str, corpo: dict, estrai: Callable[[dict], str]) -> Generazione:
        inizio = self.orologio()
        scadenze = 0
        ultimo: Exception | None = None
        for tentativo in range(1, self.tentativi + 1):
            try:
                risposta = _http("POST", f"{self.url}{percorso}", corpo, self.timeout)
                testo = (estrai(risposta) or "").strip()
                if not testo:
                    raise ErroreOllama("risposta vuota")
                return Generazione(testo, self.orologio() - inizio, tentativo, scadenze)
            except ScadenzaOllama as exc:
                scadenze += 1
                ultimo = exc
            except ErroreOllama as exc:
                ultimo = exc
            if tentativo < self.tentativi:
                self.dormi(self.attesa * tentativo)
        raise ErroreGenerazione(f"generazione fallita dopo {self.tentativi} tentativi: {ultimo}",
                                self.tentativi, scadenze, self.orologio() - inizio)

    def genera(self, modello: str, prompt: str, opzioni: dict | None = None) -> Generazione:
        corpo = {"model": modello, "prompt": prompt, "stream": False,
                 "options": {**OPZIONI_DETERMINISTICHE, **(opzioni or {})},
                 "keep_alive": KEEP_ALIVE}
        return self._con_tentativi("/api/generate", corpo, lambda r: r.get("response"))

    def chat(self, modello: str, messaggi: list[dict], opzioni: dict | None = None) -> Generazione:
        corpo = {"model": modello, "messages": messaggi, "stream": False,
                 "options": {**OPZIONI_DETERMINISTICHE, **(opzioni or {})},
                 "keep_alive": KEEP_ALIVE}
        return self._con_tentativi("/api/chat", corpo,
                                   lambda r: (r.get("message") or {}).get("content"))

    # ------------------------------------------------------------------
    # Ciclo di vita del modello e memoria
    # ------------------------------------------------------------------

    def carica(self, modello: str) -> float:
        """Carica il modello in memoria (prompt vuoto) e restituisce il
        tempo di caricamento in secondi."""
        inizio = self.orologio()
        _http("POST", f"{self.url}/api/generate",
              {"model": modello, "prompt": "", "stream": False, "keep_alive": KEEP_ALIVE},
              self.timeout)
        return self.orologio() - inizio

    def scarica(self, modello: str) -> None:
        """Libera la memoria: keep_alive 0 scarica subito il modello."""
        _http("POST", f"{self.url}/api/generate",
              {"model": modello, "prompt": "", "stream": False, "keep_alive": 0}, self.timeout)

    def modelli_caricati(self) -> list[dict]:
        """Elenco di ``/api/ps``: per ogni modello ``size`` (byte occupati
        in totale) e ``size_vram`` (la parte in memoria video)."""
        risposta = _http("GET", f"{self.url}/api/ps", None, 30)
        return list(risposta.get("models") or [])
