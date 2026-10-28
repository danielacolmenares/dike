"""Memoria occupata dai modelli e vincolo hardware.

Dopo il caricamento di ogni modello Dike legge ``/api/ps`` di Ollama e
registra ``size`` (memoria totale occupata dal modello, pesi e contesto) e
``size_vram`` (la parte su GPU; 0 su macchine solo CPU).

Con ``--ram-max-gb`` un modello che occupa piu' del limite viene escluso
con l'esito "escluso: vincolo hardware". Nel benchmark originale un modello
da 14B occupava 9,3 GB su una macchina da 16 GB e lasciava troppo poco al
resto del sistema (database vettoriale, embedding, sistema operativo).

Limite: la misura avviene DOPO il caricamento. Se il modello non entra
proprio in memoria, il caricamento stesso fallisce e il modello viene
segnato come errore, non come vincolo hardware.
"""

from __future__ import annotations

from dataclasses import dataclass

from .ollama import ClienteOllama

GB = 1_000_000_000  # Ollama riporta byte; "GB" decimali come in `ollama ps`
ESITO_VINCOLO_HARDWARE = "escluso: vincolo hardware"


@dataclass
class MemoriaModello:
    modello: str
    trovato: bool
    size: int = 0
    size_vram: int = 0

    @property
    def size_gb(self) -> float:
        return round(self.size / GB, 2)

    @property
    def size_vram_gb(self) -> float:
        return round(self.size_vram / GB, 2)


def _stesso_modello(voce: dict, modello: str) -> bool:
    nomi = {str(voce.get("name") or ""), str(voce.get("model") or "")}
    candidati = {modello} if ":" in modello else {modello, f"{modello}:latest"}
    return bool(nomi & candidati)


def leggi_memoria(cliente: ClienteOllama, modello: str) -> MemoriaModello:
    for voce in cliente.modelli_caricati():
        if _stesso_modello(voce, modello):
            return MemoriaModello(modello, True, int(voce.get("size") or 0),
                                  int(voce.get("size_vram") or 0))
    return MemoriaModello(modello, False)


def supera_limite(memoria: MemoriaModello, ram_max_gb: float | None) -> bool:
    """True se e' impostato un limite e il modello lo supera. Un modello
    non trovato in /api/ps non viene escluso (la misura manca, non e' un
    superamento)."""
    if ram_max_gb is None or not memoria.trovato:
        return False
    return memoria.size > ram_max_gb * GB
