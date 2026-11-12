# Preparazione esperimento V2 Dense + teste di regressione MSA

Data: 11 novembre 2026. Documento di implementazione, non report di risultati scientifici del dataset completo.

## 1. Audit e isolamento

Base condivisa verificata: main e720ad0c365ed55e29c4439c7a7e63a92f34cb54. Sviluppo nel worktree separato biomap-v2-dense-cnn, branch v2-dense-cnn. Nel checkout biomap-finetuning-dev esistono modifiche non committate relative a un altro esperimento: non sono state modificate, spostate, committate o incluse in questo branch.

Il codice condiviso campaign/extraction.py consolida somme/count in 128 medie per cella, produce v2.parquet e extraction_complete.json. campaign/probing.py applica Dummy/Ridge/HistGB/XGBoost/MLP a matrici tabellari, con tre fold AOI e Liguria separata. campaign/state.py e configs/msa_v1_liguria_cells.csv sono l'autorità per le 132.900 celle totali. L'estrazione universitaria attuale non è stata ispezionata direttamente: potrebbe usare modifiche non pubblicate. Se nomi/schema/marker differiscono, serve un adattamento dichiarato di sola lettura; non si modifica il produttore durante il run.

E2 esistente: biomap_ml/spatial.py usa feature [128,5,5], maschera e pesi dei 25 punti stratificati. Questa assunzione non è compatibile con una media 128D. E2 non è stata riutilizzata o alterata. Alcuni percorsi legacy conservano embedding per pixel, ma non sono l'input del benchmark di medie dense; non viene introdotto un requisito di nuove mappe spaziali.

Hardware locale effettivamente ispezionato: Mac arm64, Torch2.10.0, CUDA assente. GPU/driver/VRAM del PC universitario non sono stati verificati da remoto. Sono forniti setup Windows isolato, preflight CUDA e profiling; test CPU non certificano RTX5080.

## 2. Architettura e significato

Input [B,128], StandardScaler training-only, reshape [B,1,128]. Conv1D(1,8,k5) → GroupNorm → GELU → Conv1D(8,16,k3) → GELU → AdaptiveAvgPool1D(8) → flatten → dropout0,1 → Linear128→32 → GELU → Linear32→1. Output identity o sigmoid, scelto su sviluppo. Parametri: **4.625**.

MLP di controllo: Linear128→36 → GELU → dropout0,1 → Linear36→1, stessi output alternativi. Parametri: **4.681**, circa1,2% di differenza. La larghezza viene adattata automaticamente al budget CNN se la configurazione cambia.

L'asse convolutivo è quello delle componenti apprese, che non hanno adiacenza spaziale o temporale naturale. Questo è un prior sperimentale discutibile, non apprendimento della geometria interna delle celle. Non si trasformano i canali in un'immagine2D. L'MLP controlla quanto il risultato dipenda da quel prior. In futuro una permutazione fissa dei canali può diagnosticarne la sensibilità, ma non viene aggiunta una ricerca automatica in questa fase.

TESSERA non viene importata come modello, caricata o eseguita dal training. Nessun parametro del foundation model può essere aggiornato: il nuovo codice vede solo vettori cached. Una vera Conv2D richiederebbe una campagna separata con mappe spaziali effettive.

## 3. Dati e readiness

L’operatore ha confermato F:\biomap_dense_public\work come cartella di output, v2.parquet e extraction_complete.json come nomi finali, e273tile della campagna attuale (informazione fornita, non ispezione remota). Sono richiesti il parquet completo e un marker con SHA/n_cells/complete_campaign e method=dense127909-stream-legacy-exact-r1; un marker del percorso precedente senza method è rifiutato. Entrambi sono controllati prima/dopo la lettura. Non si assume che un file esistente sia completo e non vengono letti memmap parziali. Schema principale mean_000..127, cell_id globale, target unitario; supportato f000..127. Metadati geografici sono confrontati col manifest o uniti da esso se assenti. Target esterno opzionale solo se non è già nel parquet. Non si ricostruisce MSA e non si applica di nuovo MSA-area/cell-area.

Preflight: unicità ID, finitezza, target[0,1], SHA, conteggio, membership, coordinate e tre fold AOI, assenza di overlap test. La verifica richiede 127.909 sviluppo e4.991 Liguria, non un numero approssimato. Fonte e output non possono essere annidati. Scritture soltanto nell'output nuovo; lock del nuovo training, non dei dati. Il marker condiviso non contiene necessariamente SHA dell'encoder: il suo ruolo è confermare il dataset completato; la provenienza del checkpoint resta quella dichiarata dal produttore, non una verifica di nuovo inferita qui.

## 4. Protocollo di training e selezione

Tre fold originali: 89.101/38.808,83.014/44.895,83.703/44.206 (training/validation). Scaler StandardScaler stimato esclusivamente su train del fold, sia per rete sia per Ridge; HistGB usa le stesse feature senza scaler. Dummy è la media del solo train pertinente.

Neurali: AdamW, LR1e-3, weight decay0,01, MSE, batch256, massimo150epoche, patience15, seed42, gradient clipping1, caricamento deterministico senza processi secondari. Early stopping usa RMSE medio sulle AOI validation; vengono salvati trainingMSE, validationMSE e validationmacroRMSE. Configurazioni prefissate quattro: Conv1D/MLP × identity/sigmoid. Nessuna ricerca automatica aggiuntiva.

Baseline: Ridgealpha100; HistGB90iterazioni,15foglie, LR0,05,L2=1 e early stopping interno al solo training, seed storico20260925; DummyMean. Si riusano factory e metriche comuni. Le diverse famiglie hanno parametri e selezione dichiarati: il confronto non equivale a una ricerca esaustiva ottimale per ogni regressore.

Sequenza: tutti i fold di sviluppo → selezione output per famiglia sulla macro-RMSE OOF → congelamento selection.json → refit baseline sullo sviluppo e refit delle due reti selezionate per il numero di epoche mediano migliore dei fold → valutazione Liguria. Totale26fit. Nessun refit/early stopping/variante scelta guardando Liguria. Umbria resta esclusa.

OOF di sviluppo è usato per checkpoint e selezione: non è una stima finale non distorta equivalente a un test cieco. Liguria è un benchmark geografico già osservato. Questa limitazione rimane anche con split spaziali corretti. Output sigmoid può saturare, identity può uscire da[0,1]; le predizioni non vengono tagliate per migliorare le metriche.

## 5. Risultati e confronto

MAE,RMSE,MAPE,bias(pred−true),R²,Pearson, medie, conteggi, valori fuori scala e riduzioneRMSE vsDummy. Macro-AOI è distinto da pooledRMSE e metriche per fold/AOI. MAPE esclude zero indicando il denominatore; MAE×100 sono punti percentuali della scala. Predizioni per cella includono ID,AOI,fold,target/predizione,residuo,errori assoluti/percentuali,modello,esperimento e Dummy.

Excel e grafici derivano dai CSV. Con la configurazione completa si generano895.363righe OOF e24.955righe test, distribuite per modello/partizione. L'export può richiedere RAM e tempo e ha una fase separata nel monitor. Se Excel blocca il file, export riparte dai CSV senza riaddestrare.

I valori V1 storici sono caricati dal riferimento già nel repository in un file separato marcato indicative, non ricopiati come nuove misure. Confronto appaiato opzionale richiede predizioni, medesimi ID e target e scope modello/partizione esatto; se incompatibile non viene inventato un delta. Non si mescola V2 a25punti con V2 denso senza etichetta. Un buon accordo col target modellato GLOBIO non dimostra accuratezza di biodiversità sul campo.

## 6. Checkpoint, monitor e risorse

Checkpoint last/best contiene solo testa, scaler, optimizer, RNG e configurazione. Resume richiede stessa impronta dati/codice/device/split e riprende dall'ultima epoca completa; i fit già completi sono recuperati. Nessuna nuova estrazione. Protocollo registra GitSHA, hash dei moduli, versioni, dispositivo e hash dei dati.

Monitor standalone legge soltanto il progress.json atomico nel nuovo output, non stream_progress.json della campagna corrente (segnalato dall’operatore come fonte di un precedente Accesso negato): fase,modello,fold,fit/26,epoca,celle/s e stime. ETA nasce da epoche misurate; campagna estrapolata al budget massimo è a bassa confidenza, può differire fra architetture, non include accuratamente baseline CPU/export ed early stopping può accorciare. Durante un fit sklearn senza callback l'assenza di aggiornamenti non prova un blocco. Il primo test Windows ha rilevato che la condivisione DELETE da sola non garantisce il replace con handle aperto. Il lettore ora copia il testo in memoria e chiude l’handle OS prima del parsing, oltre a richiedere condivisione READ/WRITE/DELETE; retry limitati del writer impediscono che un Accesso negato temporaneo sul solo progresso termini il training. Un test mantiene aperta la snapshot del consumer durante atomic replace, verificando che il file originale sia già rilasciato. Il monitor non prenota GPU e Ctrl+C nel monitor non interrompe il training.

Matrice132.900×128float32:68.044.800byte, circa64,9MiB; DataFrame e copie aggiungono RAM. Pesi+gradienti+due momenti Adam delle teste occupano circa74–75kB, esclusi attivazioni, allocator e contesto CUDA. Questo non è una misura della VRAM totale. Non si promette una durata. Profile esegue3epoche su4096train/1024validation di un solo fold, salva secondi/epoca e piccoCUDA e restituisce una stima lineare dichiarata del budget neurale. Il profiling e il run universitario non sono stati avviati in questa preparazione.

## 7. File e workflow

- scripts/train_v2_dense_cnn.py: unico ingresso run/profile/smoke/monitor/export/compare.
- scripts/preflight_v2_dense_cnn.py: controllo di readiness, nessun avvio implicito.
- scripts/setup_dense_heads.ps1: ambiente Windows separato, nessun uso del modello/encoder.
- biomap_dense_heads/artifacts.py: schema, hash, membership e isolamento sorgente.
- models.py: Conv1D e MLP comparabile; engine.py: scaler, fit,early stopping,resume e predict.
- campaign.py: fold,baseline,selezione,refit,Liguria,profiling.
- evaluation.py: metriche/CSV/Excel/grafici e storico appaiabile.
- progress.py: stato atomico e monitor di sola lettura.
- config.yaml: budget prefissato; tests/test_dense_cnn_heads.py: verifiche CPU isolate.

```text
Estrazione esistente (invariata)
   -> marker completo + v2.parquet
   -> preflight di sola lettura
   -> 3 fold geografici, scaler per fold
   -> Ridge/HistGB/Dummy + Conv1D/MLP identity/sigmoid
   -> selezione macro-AOI su sviluppo
   -> refit e Liguria separata
   -> predizioni,metriche,grafici,Excel,report
                 |
                 +-> progress.json -> monitor nel secondo terminale
```

## 8. Verifiche e blocchi residui

Smoke completo eseguito su dati sintetici CPU:26fit,modelli salvati/caricati,predizioni/grafici/Excel e stato complete. Nessuna metrica sintetica è risultato scientifico. Test coprono forme/gradienti/budget parametri,marker incompleto,source read-only,NaN,duplicati,leakage,scaler training-only,riproducibilità,resume,confronto storico parziale e profiling. Verifica locale:108test superati,0errori,0fallimenti (58,506secondi CPU). CI Windows/Linux verificata separatamente nella consegna GitHub; non è una prova CUDA.

Resta necessario che il run attuale termini e produca gli artifact dichiarati, che gli artifact compaiano nel percorso universitario confermato e che preflight/profile GPU passino sul PC reale. Codice pronto non significa dati completati o esperimento già eseguito. Non sono stati fermati o riavviati processi, cambiate configurazioni, riscritti output o scaricati nuovi dati Sentinel. Non sono state modificate le directory dell'altro lavoro né i moduli del produttore.
