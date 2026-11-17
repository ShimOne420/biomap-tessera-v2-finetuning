# Guida alla decisione dopo il run Frozen + Spatial CNN

L'encoder è già deciso: **GeoTESSERA V2 Medium**. La strategia è partial fine-tuning, con preferenza iniziale per gli ultimi **due blocchi per modalità**. L'aggregazione scientifica **non è stata scelta**. Nel YAML principale rimane `undecided`; learning rate, batch ed epoche restano null.

## Prima: verificare il run attuale

Sul PC, verificare lo stato finale del run `C:\BioMAP\spatial_runs\v2_spatial3_seed42`, errori, fold completati, denominatore e supporto. Non dedurre la fine da un file di metriche intermedio. Conservare config, manifest e commit; la correzione multi-UTM della base è necessaria sui dati reali. Questa consegna non ha letto output o processi live del PC.

Leggere prima le predizioni OOF e il **macro-AOI RMSE dei tre fold originali**, con RMSE/MAE per ciascuna delle 17 AOI, bias, R² e confronto Dummy. Controllare i residui, le AOI con vantaggio/perdita e la stabilità rispetto al seed. Per confronti con dense V1/V2 verificare ID, target, fold e supporto pixel; non basta avere 127.909 righe. Il handoff riporta 0,17237 per V1 XGBoost e 0,17650 per V2 XGBoost: sono riferimenti dell'utente, non risultati ricalcolati qui. Non usare i risultati Liguria per decidere architettura, sampling o iperparametri.

Se Spatial CNN mostra un vantaggio coerente sullo sviluppo, sostenuto dal confronto appaiato e non soltanto da poche AOI, `spatial_cnn` è una candidata motivata. Se il vantaggio non è chiaro o il costo della CNN non è giustificato, `dense_mean` con MLP è un'alternativa meno complessa. Nessuna delle due è scelta automaticamente. Il fine-tuning potrebbe cambiare il rapporto fra aggregazioni: questo esperimento non lo dimostra in anticipo.

Il seed richiesto per il nuovo esperimento è 20260925; il run attuale contiene seed42 nel nome. Una differenza di seed va registrata, non mascherata come unico effetto del fine-tuning. La comparabilità principale è perciò **nuovo frozen + stessa testa** contro **nuovo partial + stessa testa**, con lo stesso seed e lo stesso supporto acquisito.

## Seconda decisione: supporto e budget

Il PC non conserva Sentinel grezzi. Le medie dense e le mappe frozen non consentono di aggiornare l'encoder. Gli input devono essere riacquisiti; la cache completa delle attivazioni non entra nello spazio descritto.

`full_support` preserva tutti i contributi storici e dà il confronto più diretto, ma il budget disco disponibile è insufficiente alle scale stimate. `uniform_pixels` mantiene tutte le celle e seleziona K contributi per cella; è adatto a una media, ma rende sparse le mappe CNN. `center_windows` preserva adiacenze locali per la CNN, cambiando però il supporto e la rappresentatività all'interno della cella. Tutte le opzioni mantengono anno, target, membership e date Sentinel originali. Nessun numero K/lato definitivo viene imposto dalla consegna.

La configurazione smoke 2×2 su quattro celle è solo tecnica. Il suo eventuale successo non approva automaticamente 2×2 per tutto il benchmark. Qualunque supporto ridotto richiede controllo frozen sullo stesso nuovo dataset e distinzione esplicita dai risultati storici a supporto pieno. Se serve un disco aggiuntivo, sceglierlo prima dell'acquisizione: raw giornalieri, raw finali, checkpoint di tutte le epoche e riserva 20 GiB vanno considerati insieme.

## Terza decisione: due blocchi e risorse

L'audit reale conta 6,49 milioni di parametri trainabili con un blocco, 11,21 milioni con due e 15,92 milioni con tre, esclusa la testa. La preferenza **due** resta quella autorizzata, con checkpointing e piccoli microbatch. Il checkpoint non è un modello interno a 256 canali: D=440 cambia il budget di attenzione/attivazioni. Due blocchi rappresentano circa metà dei parametri totali, non un adattamento trascurabile.

Un blocco è il primo candidato da valutare **se** il pilot con due blocchi non lascia memoria sufficiente; sarebbe una nuova scelta esplicita, non un fallback automatico del codice. Tre aumenta stati e attivazioni e non è raccomandato senza una misura che ne giustifichi il costo. Il piccolo test CPU con checkpoint reale dimostra trainabilità, non capacità della RTX 5080 su celle intere. FP32 è il default; AMP o BF16 sono decisioni aggiuntive da registrare e verificare.

## Ordine operativo

Seguire [il runbook PowerShell](../development/FINETUNING_V2_RUNBOOK_PC.md), nel nuovo clone. Inventario e `doctor` funzionano prima della scelta. Dopo la scelta esplicita dell'aggregazione, lo smoke acquisisce pochi input reali, confronta embedding frozen, esegue backward/step e produce metriche tecniche, con deadline totale di 900 secondi. Se supera il budget, segnalarlo come test non completato: riprendere i validi o rivalutare il piano, senza ridurre date o cambiare dati silenziosamente.

Registrare tempi per tile/blocco, giorni effettivi, byte, RAM disponibile, VRAM di picco e microbatch. Da lì decidere sampling, numero di blocchi, learning rate testa/encoder, batch ed epoche massime, lasciando un margine di memoria. Fare il dry-run completo e controllare l'ammissione con cataloghi reali. Non scalare ingenuamente le 2,5 giornate della r1 per il rapporto di pixel: le richieste COG dipendono anche dalla dispersione geografica e dalle scene annuali.

Il training completo richiede contemporaneamente: dati di tutte le celle previsti dal piano; checksum validi; prova campionaria input/frozen superata con tolleranze originali; GPU e memoria verificate su supporto rappresentativo; nessun altro run pesante attivo; budget disco per raw e checkpoint; decisioni approvate; config non ambigua. Poi eseguire controllo frozen e partial con la stessa aggregazione, selezione CV, refit e soltanto dopo valutazione esterna. Se una condizione manca, lo stato resta **NOT_VERIFIED_FOR_REAL_TRAINING**.

## Decisioni ancora aperte e limiti del software

L'utente deve approvare aggregazione, supporto/K/lato, eventuale disco aggiuntivo, LR, batch, epoche ed eventuali variazioni rispetto ai due blocchi/FP32. Deve anche stabilire quanti seed scientifici eseguire dopo il pilot: la consegna fissa un seed riproducibile, non avvia una campagna multipla.

Sono implementati online, due head alternative, CV/refit/test, controllo frozen, checkpoint/resume, monitor e confronto fra nuovi run. La cache di corpus, LoRA, import automatico certificato dei vecchi risultati e ottimizzazione di un grid search non sono implementati. Non esiste ancora un risultato scientifico del nuovo fine-tuning, né una validazione dei tempi/VRAM sul PC. Non trasformare queste assenze in un'affermazione di vittoria rispetto a V1.
