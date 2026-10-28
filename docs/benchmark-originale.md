# Il benchmark originale

Dike generalizza un confronto tra modelli fatto a mano in un progetto precedente: un assistente RAG su una Knowledge Base privata di documenti di un sistema di gestione aziendale (sicurezza delle informazioni, privacy, qualità). Documenti, domande e infrastruttura non sono pubblicati. Questa pagina descrive il metodo e i risultati in forma anonima, perché sono il motivo per cui Dike esiste e il riferimento con cui confrontarne gli esiti.

## Principi del metodo

1. **Stesso contesto per tutti i modelli.** I passaggi recuperati per ogni domanda sono gli stessi per ogni modello: cambia solo il generatore. Altrimenti si confronta il recupero, non il modello.
2. **Golden Set e Negative Set.** Domande con risposta nota nei documenti e domande fuori dominio, per cui l'unica risposta corretta è dichiarare che l'informazione non c'è.
3. **Doppia esecuzione per il determinismo.** Lo stesso protocollo eseguito due volte con temperatura 0 e seed fisso: le risposte devono essere identiche.
4. **Tempi e memoria misurati**, non stimati: la macchina su cui il sistema gira davvero decide quali modelli sono praticabili.
5. **Allucinazione separata dalla citazione imprecisa.** Inventare un fatto e citare una sottosezione sbagliata sono errori diversi, e si contano a parte.
6. **Revisione umana delle risposte.** Gli indicatori automatici dicono dove guardare; il giudizio sulla risposta resta umano.
7. **Nessuna adozione senza misura.** Modello, prompt o parametri si cambiano solo dopo aver rifatto il protocollo.

## Primo giro: scelta del modello

**Condizioni.** Knowledge Base di prova; Golden Set di 20 domande e Negative Set di 8 (28 in totale); server virtuale con 16 GB di RAM, solo CPU; Prompt A; determinismo verificato con due esecuzioni identiche.

| Modello | Golden | Negative | Totale /28 | Determinismo | Tempo medio | Comportamento | Allucinazioni | Esito |
|---|---|---|---|---|---|---|---|---|
| gemma3:4b | 20/20 | 8/8 | 28 | sì | 32,4 s | equilibrato e completo; unico a rispondere correttamente a una domanda sui metadati del documento; nessuna sottosezione inventata | 0 | **scelto** |
| qwen2.5:7b | 19/20 | 8/8 | 27 | sì | 55,3 s | verboso, tende a citare in eccesso; a volte cita sottosezioni inesistenti nelle domande enumerative | 0 | validato, più lento |
| llama3.2:3b | 19/20 | 8/8 | 27 | sì | 27,1 s | rapido, generalmente completo | 0 | alternativa veloce |
| phi4-mini | 18/20 | 8/8 | 26 | sì | 30,8 s | inconsistente, citazioni imprecise | 0 | validato, meno preciso |
| qwen2.5:3b | 14/20 | 8/8 | 22 | sì | 25,1 s | prudente: 6 falsi negativi (non risponde anche quando la risposta c'è) | 0 | troppi falsi negativi |
| mistral 7B | n.d. | n.d. | – | – | 66,9 s | lento, verboso, artefatti nel testo | – | escluso: troppo lento (test interrotto) |
| qwen2.5:14b | – | – | – | – | – | occupava 9,3 GB | – | escluso: vincolo hardware |

**Cosa si è imparato.**

- Su 16 GB di RAM e solo CPU il limite pratico è la classe 7B: un 14B occupa troppa memoria per convivere con il resto del sistema (database vettoriale, modello di embedding, sistema operativo).
- Il modello più accurato non era il più grande: gemma3:4b ha superato qwen2.5:7b, a poco più della metà del tempo.
- Tutti i modelli validati hanno rifiutato correttamente le 8 domande fuori dominio: la differenza l'hanno fatta le risposte sul Golden Set e i tempi.
- Un modello "prudente" non è un modello sicuro: 6 falsi negativi su 20 rendono l'assistente inutile proprio dove dovrebbe servire.
- Un modello che a 67 secondi per risposta non ha ancora finito il Golden Set si può fermare: il dato sul tempo basta a escluderlo.

In Dike questi casi corrispondono a `--ram-max-gb` (vincolo hardware), `--stop-latenza-s` / `--stop-dopo` (arresto anticipato) e all'etichetta "prudente/sotto-risponde" del comportamento prevalente.

## Secondo giro: tempi dopo l'aggiornamento dell'hardware

**Condizioni.** Server con 31 GB di RAM e 16 vCPU; Knowledge Base completa (849 chunk, 21 volte più grande di quella di prova); Golden Set di 15 domande e Negative Set di 4. Solo misura dei tempi; la qualità era già stata valutata nel primo giro.

| Modello | Tempo medio | Tempo minimo | Tempo massimo | Note |
|---|---|---|---|---|
| gemma3:4b | 21,5 s | 9,4 s | 62,1 s | determinismo confermato: 0 differenze su 19 risposte (2 esecuzioni) |
| llama3.2:3b | 22,3 s | 9,3 s | 86,8 s | 1 esecuzione |
| qwen2.5:7b | 38,2 s | 18,1 s | 132,7 s | 1 esecuzione |
| qwen2.5:14b | 83,6 s | 31,7 s | 303,6 s | 1 esecuzione; 1 timeout con nuovo tentativo |

**Osservazioni.**

- La domanda più lenta era la stessa per tutti e quattro i modelli: il costo dipendeva dalla domanda e dal contesto recuperato, non dal modello. Per questo Dike riporta, oltre alla media, minimo, massimo e p90, e conserva i tempi per domanda nel dettaglio.
- Con più memoria il 14B diventa caricabile, ma resta circa quattro volte più lento del modello scelto e ha richiesto un nuovo tentativo dopo un timeout: praticabile non vuol dire conveniente.
- **Uso reale.** Nell'uso con più utenti per dieci giorni (85 interazioni), sull'hardware precedente, il tempo medio è stato 40,5 s e la mediana 35,7 s. Dopo l'aggiornamento il benchmark ha dato circa 21,5 s su una collezione 21 volte più grande. Cautela: è una sola doppia esecuzione, con servizi già caldi; il confronto con l'uso reale è indicativo, non una misura controllata.

## Esperimento sul prompt

Il Prompt B, con regole più dettagliate, ha corretto due problemi del Prompt A: il punteggio di rilevanza che a volte finiva nel testo della risposta e un fatto specifico inventato in una risposta. Ha però introdotto problemi peggiori: note che contraddicevano la risposta stessa, etichette rinominate che comparivano nel testo e risposte sistematicamente più lente. **Il Prompt B è stato respinto e il Prompt A mantenuto.** Una variante si adotta solo se migliora i risultati misurati sull'intero protocollo, non se corregge il singolo caso che l'ha motivata.

## Soglia sul punteggio di somiglianza

I punteggi di somiglianza delle domande valide e di quelle fuori dominio si sovrapponevano (circa 0,50–0,60): nessuna soglia separava gli uni dagli altri senza scartare anche risposte buone. I rifiuti corretti sono venuti dall'istruzione nel prompt, non da una soglia. Per questo in Clio la soglia esiste ma è spenta di default, e Dike la eredita dalla configurazione di Clio (`CLIO_MIN_SCORE`) senza attivarla.

## Concorrenza

Una sola prova: una richiesta isolata impiegava circa 21,5 s; due richieste inviate insieme hanno impiegato 20,2 s e 33,1 s. Se Ollama le avesse messe in coda, la seconda avrebbe impiegato circa il doppio (circa 43 s); se le avesse servite in parallelo senza contesa, entrambe circa 21,5 s. Il valore intermedio indica **parallelismo reale con contesa delle risorse**.

La raccomandazione emersa: prima di aprire il sistema a molti utenti serve un modulo dedicato alla verifica della concorrenza, con lo stesso rigore del benchmark di qualità. In Dike è il comando `concorrenza`, che ripete la prova con 1, 2, 4… richieste simultanee, classifica ogni livello (parallelo, in coda, parallelo con contesa) e riporta la memoria occupata prima e dopo. Il comportamento dipende dalla configurazione del server Ollama (`OLLAMA_NUM_PARALLEL`), che non è leggibile dall'API: va misurato.

## Dal benchmark originale a Dike

| Nel benchmark originale | In Dike |
|---|---|
| contesto identico preparato a mano | recupero una volta per domanda, in cache, con impronta del prompt su ogni riga |
| tabelle compilate a mano | `dettaglio.csv`, `riepilogo.md`, `riepilogo_modelli.csv` |
| revisione delle risposte su foglio di calcolo | `revisione_umana.csv` + comando `verdetti` |
| doppia esecuzione lanciata due volte | `--run N` e confronto automatico dei testi |
| 14B escluso dopo averlo visto in memoria | `--ram-max-gb` con lettura di `/api/ps` |
| mistral 7B fermato a mano | `--stop-latenza-s` / `--stop-dopo` |
| una prova di concorrenza | comando `concorrenza` con livelli configurabili |
