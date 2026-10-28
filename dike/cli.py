"""Riga di comando di Dike.

    python -m dike.cli esegui --kb demo --modelli gemma3:4b,llama3.2:3b --run 2
    python -m dike.cli concorrenza --modello gemma3:4b --paralleli 1,2,4
    python -m dike.cli riepilogo risultati/demo_20261028_150000
    python -m dike.cli verdetti risultati/demo_20261028_150000/revisione_umana.csv

Dike usa Clio come libreria (``pip install -e /percorso/di/clio`` per lo
sviluppo locale, oppure la dipendenza in requirements.txt). Le Knowledge
Base sono quelle di Clio: ``--clio-dir`` indica la cartella del progetto
Clio che contiene ``kb/`` (equivale a impostare CLIO_BASE_DIR). Servizi:
OLLAMA_URL e QDRANT_URL, come in Clio.

Prerequisito di ``esegui``: la KB e' gia' caricata su Qdrant con Clio
(``python -m clio.cli tutto --kb <nome>``) e i modelli sono gia' scaricati
in Ollama.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

from clio.clienti import ErroreServizio

from . import ollama


def _elenco(valore: str) -> list[str]:
    voci = [v.strip() for v in valore.split(",") if v.strip()]
    if not voci:
        raise argparse.ArgumentTypeError("elenco vuoto")
    return list(dict.fromkeys(voci))


def _interi(valore: str) -> list[int]:
    try:
        numeri = [int(v) for v in _elenco(valore)]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"attesi numeri interi separati da virgola: '{valore}'") from exc
    if any(n < 1 for n in numeri):
        raise argparse.ArgumentTypeError("i livelli di parallelismo devono essere >= 1")
    return numeri


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m dike.cli", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="comando", required=True, metavar="comando")

    def servizio(s: argparse.ArgumentParser) -> None:
        s.add_argument("--timeout", type=float, default=ollama.TIMEOUT_DEFAULT,
                       help="timeout di ogni richiesta a Ollama, in secondi (default 300)")
        s.add_argument("--uscita", type=Path, default=Path("risultati"),
                       help="cartella in cui creare i risultati (default: ./risultati)")

    s = sub.add_parser("esegui", help="protocollo completo su una KB di Clio")
    s.add_argument("--kb", required=True, help="nome della KB di Clio (cartella kb/<nome>/)")
    s.add_argument("--modelli", required=True, type=_elenco, help="modelli Ollama separati da virgola")
    s.add_argument("--run", type=int, default=2, help="esecuzioni complete per il determinismo (default 2)")
    s.add_argument("--clio-dir", type=Path, help="cartella del progetto Clio (contiene kb/)")
    s.add_argument("--top-k", type=int, default=3, help="passaggi recuperati per domanda (default 3, come Clio)")
    s.add_argument("--prompt", default="A", choices=["A", "B", "a", "b"], help="versione del prompt di Clio")
    s.add_argument("--ram-max-gb", type=float, help="escludi i modelli che occupano piu' di questo limite")
    s.add_argument("--stop-latenza-s", type=float,
                   help="interrompi un modello se la media delle prime risposte supera questa soglia")
    s.add_argument("--stop-dopo", type=int, default=5, help="numero di risposte per l'arresto anticipato")
    s.add_argument("--tentativi", type=int, default=ollama.TENTATIVI_DEFAULT,
                   help="tentativi per ogni generazione (default 2)")
    s.add_argument("--non-scaricare", action="store_true",
                   help="non liberare la memoria dopo ogni modello")
    servizio(s)

    s = sub.add_parser("concorrenza", help="richieste parallele dirette a Ollama")
    s.add_argument("--modello", required=True)
    s.add_argument("--paralleli", type=_interi, default=[1, 2, 4], help="livelli, es. 1,2,4")
    s.add_argument("--ripetizioni", type=int, default=1)
    s.add_argument("--endpoint", choices=["generate", "chat"], default="generate")
    s.add_argument("--num-predict", type=int, default=256, help="lunghezza massima della risposta in token")
    s.add_argument("--prompt-file", type=Path, help="usa questo prompt invece di quello fisso")
    servizio(s)

    s = sub.add_parser("riepilogo", help="ricalcola il riepilogo di una cartella di risultati")
    s.add_argument("cartella", type=Path)

    s = sub.add_parser("verdetti", help="importa i verdetti umani e ricalcola il riepilogo")
    s.add_argument("csv_umano", type=Path)
    s.add_argument("--risultati", type=Path, help="cartella dei risultati (default: quella del CSV)")
    return p


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                        stream=sys.stdout)
    try:
        if args.comando == "esegui":
            return _esegui(args)
        if args.comando == "concorrenza":
            return _concorrenza(args)
        if args.comando == "riepilogo":
            from .riepilogo import genera_riepilogo
            print(genera_riepilogo(args.cartella).read_text(encoding="utf-8"))
            return 0
        if args.comando == "verdetti":
            from .verdetti import importa_verdetti
            n, percorso = importa_verdetti(args.csv_umano, args.risultati)
            print(f"Verdetti importati: {n}\nRiepilogo aggiornato: {percorso}")
            return 0
    except (ValueError, FileNotFoundError, ollama.ErroreOllama, ErroreServizio) as exc:
        print(f"Errore: {exc}", file=sys.stderr)
        return 1
    return 2


def _esegui(args) -> int:
    if args.clio_dir:
        os.environ["CLIO_BASE_DIR"] = str(args.clio_dir.expanduser().resolve())
    if args.run < 1:
        raise ValueError("--run deve essere almeno 1")
    from clio import config as clio_config

    from .protocollo import Opzioni, esegui_protocollo

    kb = clio_config.get_kb(args.kb)
    opzioni = Opzioni(run=args.run, top_k=args.top_k, versione_prompt=args.prompt.upper(),
                      min_score=clio_config.min_score(), ram_max_gb=args.ram_max_gb,
                      stop_latenza_s=args.stop_latenza_s, stop_dopo=args.stop_dopo,
                      scarica_dopo=not args.non_scaricare)
    cliente = ollama.ClienteOllama(timeout=args.timeout, tentativi=args.tentativi)
    cartella = esegui_protocollo(kb, args.modelli, args.uscita / f"{kb.nome}_{_timestamp()}",
                                 cliente, opzioni)
    print((cartella / "riepilogo.md").read_text(encoding="utf-8"))
    print(f"Risultati: {cartella}")
    return 0


def _concorrenza(args) -> int:
    from .concorrenza import esegui_concorrenza

    prompt = args.prompt_file.read_text(encoding="utf-8") if args.prompt_file else None
    cliente = ollama.ClienteOllama(timeout=args.timeout, tentativi=1)   # nessun retry: falserebbe i tempi
    nome = args.modello.replace(":", "-").replace("/", "-")
    percorso = esegui_concorrenza(cliente, args.modello, args.paralleli,
                                  args.uscita / f"concorrenza_{nome}_{_timestamp()}",
                                  args.ripetizioni, args.endpoint, args.num_predict, prompt)
    print(percorso.read_text(encoding="utf-8"))
    print(f"Risultati: {percorso.parent}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
