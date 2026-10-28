"""Dike – benchmark di LLM locali per RAG: scegliere il modello con i dati.

Dike prende una Knowledge Base di Clio e un elenco di modelli Ollama ed
esegue un protocollo riproducibile: qualita' sul Golden Set, rifiuto sul
Negative Set, determinismo su piu' esecuzioni, tempi, memoria occupata e
(con un comando separato) comportamento con richieste concorrenti.

Principio cardine: il recupero dei passaggi avviene UNA sola volta per
domanda e il risultato viene messo in cache. Tutti i modelli ricevono lo
stesso identico contesto: cambia solo il generatore.
"""

__version__ = "0.1.0"
