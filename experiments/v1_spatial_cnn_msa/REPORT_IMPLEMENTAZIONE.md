# Preparazione dell'esperimento V1 Spatial CNN per MSA

**Data: 11 novembre 2026. Sessione B. Ramo `experiment/v1-spatial-cnn-msa`.**

È stata implementata una pipeline isolata per verificare se una testa CNN sulle mappe spaziali GeoTESSERA V1.0 migliori la predizione GLOBIO4 MSA rispetto alle medie degli stessi embedding. Il foundation model resta congelato: vengono allenate soltanto le teste. Nessuna estrazione completa, nessun training scientifico e nessun merge con l'altra sessione sono stati eseguiti durante questa preparazione.

Il [README operativo](README.md) contiene i comandi PowerShell copiabili, nell'ordine necessario, e il comando per un secondo terminale di monitoraggio. Questo documento spiega le scelte, le verifiche e i limiti della consegna.

## A. Audit del repository e delle sorgenti effettive

L'implementazione parte da `main` al commit `e720ad0c365ed55e29c4439c7a7e63a92f34cb54`, verificato prima della creazione di un clone e di un worktree indipendenti. L'audit iniziale è conservato in [AUDIT.md](AUDIT.md). Non sono stati modificati moduli condivisi, dipendenze comuni o file dell'esperimento V2.

Il percorso già esistente `src/biomap_v2/dense/v1_support.py` legge prodotti **V1.0, variante vultr, anno 2020**, precalcolati. Ogni tile comprende:

- un NPY int8 di shape `[H,W,128]`;
- scale float32 di shape `[H,W]`;
- un TIFF landmask che fornisce CRS e trasformazione affine;
- metadati del prodotto e della versione.

La dequantizzazione è `embedding_int8.astype(float32) * scales[...,None]`. La validità segue le scale finite, coerentemente con il lettore V1 presente. Le mappe sono spaziali reali; il parquet storico di medie/std non permette di ricostruirle.

Sul Mac sono state trovate mappe su SSD per più aree, incluse dieci tile liguri. Una tile letta aveva shape `[1116,804,128]`, CRS EPSG:32632 e passo 10 m. Non è stata certificata la copertura completa delle 17 AOI. La reference portabile completa può dipendere da file iCloud non materializzati: la disponibilità sul Mac non equivale alla disponibilità sul PC universitario.

Il download implementato usa il prodotto ufficiale V1.0 di [GeoTessera](https://github.com/ucam-eo/geotessera). Le richieste HEAD dei tre file di una tile reale hanno restituito HTTP 200. Il percorso remoto NPY è `/npy/v1/2020/grid_<lon>_<lat>/...`, mentre la directory storica locale contiene anche `global_0.1_degree_representation`: il codice distingue i due percorsi. Il prodotto V1.1 non viene sostituito a V1.0.

Non serve scaricare o eseguire l'encoder né il checkpoint QAT per consumare questi embedding precalcolati. Il suo checksum locale non è stato verificato e non viene dichiarato tale. Non è stato inventato un percorso raw Sentinel alternativo.

## B. Rappresentazione spaziale e geometria

Ogni cella è rappresentata da uno o più tensori `[128,h,w]`, su griglia nativa UTM a 10 m. Le dimensioni sono variabili. Per le 4.991 celle liguri, le finestre proiettate nel CRS della tile esaminata hanno altezza 31–33 e larghezza 23–24 pixel; mediane 32 e 23. Questi numeri descrivono la geometria ligure esaminata, non una misura globale di tutte le AOI.

Un audit isolato di due celle reali ha prodotto due frammenti di 31×23, rispettivamente con 707 e 683 centri validi, senza sorgenti mancanti/incompatibili. È una verifica di accesso e geometria, non una valutazione MSA o una certificazione di copertura completa.

I centri dei pixel sono trasformati dal CRS della tile a WGS84 e assegnati alla griglia GLOBIO a 10 arcsec con confini semiaperti: ovest incluso, est escluso, sud escluso, nord incluso. Gli ID originali e l'assegnazione fisica alla cella vengono preservati. Non si fa resizing, interpolazione, ricostruzione delle medie o acquisizione di contesto fuori cella.

## C. Strategia delle patch e cache

È stata scelta la strategia B del task: un frammento per intersezione cella/tile, suddiviso ulteriormente solo quando una finestra supera il lato massimo configurabile, inizialmente 64. Una cella può quindi avere un numero variabile di frammenti. Il limite 64 è un limite di suddivisione, non una risoluzione imposta al dato.

La cache è un indice SQLite transazionale, con tabelle `meta`, `tiles`, `cells` e `patches`. Conserva finestre, maschere bit-packed, relazioni patch/cella, statistiche dei pixel, provenienza e hash. Gli array int8 e le scale rimangono nei file sorgente, letti con memmap read-only e un piccolo LRU di otto tile. Non vengono duplicati milioni di tensori float32 su disco.

Il build riusa le tile già committate. L'intera elaborazione di una tile è una transazione: un'interruzione annulla soltanto quella tile. La pubblicazione finale richiede finalizzazione e manifest SHA256. La ripresa verifica identità di configurazione, membership, target e sorgenti. Il controllo profondo può ricalcolare tutti gli hash delle sorgenti; i controlli ordinari verificano anche dimensione/mtime.

Le sorgenti esterne sono lette in sola lettura. I download mancanti finiscono in un overlay privato. I file scaricati hanno `.partial`, ripresa HTTP Range/If-Range, controllo Content-Range e lunghezza, MD5 quando l'ETag è un MD5 singolo, SHA256 e marker recuperabili. Un ETag multipart non viene presentato come MD5 verificato. Una tile assente o incompatibile produce un errore esplicito, senza imputazione.

La copertura è `n_valid/n_total` sui centri acquisiti delle tile necessarie e presenti. La soglia iniziale è 0,8. Di default una cella esclusa impedisce di pubblicare la cache come completa. L'opzione esplicita `allow_partial_cells` permette soltanto un sottoinsieme dichiarato, condiviso da tutti i modelli; non lo rende equivalente al benchmark completo. Restano richiesti tutti i tre fold e le 17 AOI per il training.

## D. CNN, maschere e loss

La CNN iniziale contiene **94.561 parametri allenabili**:

| Passaggio | Operazione |
|---|---|
| Input | Patch native a 128 canali e maschera binaria |
| Primo blocco | Convoluzione parziale 3×3, 128→64; LayerNorm sui canali di ogni pixel; GELU |
| Secondo blocco | Convoluzione parziale 3×3, 64→32; GELU |
| Pooling spaziale | Media dei soli pixel validi di ogni patch |
| Pooling alla cella | Media delle rappresentazioni delle patch pesata per il numero di pixel validi |
| Testa | Linear 32→64, GELU, dropout 0,1, Linear 64→1 |

La convoluzione parziale elimina i valori invalidi con `torch.where`, conta i vicini validi e riscalibra la risposta per `9/count`. La maschera originale viene mantenuta dopo ogni convoluzione: non si propagano feature in posizioni sconosciute o fuori cella. La LayerNorm opera sui canali di un singolo pixel, evitando statistiche dipendenti dalle dimensioni del padding. I test cambiano valori del padding, introducono NaN e aumentano il padding esterno, verificando l'invarianza delle predizioni.

Si produce **una sola predizione e un solo contributo MSE per cella**. Il numero di patch non attribuisce automaticamente maggiore peso statistico alla cella. Non si assegna il target della cella a patch trattate come osservazioni indipendenti.

Il controllo MLP usa `128→64→32→1`, GELU e dropout: **10.369 parametri**. Legge direttamente le medie già presenti nell'indice, evitando I/O spaziale superfluo. L'output iniziale è lineare senza clipping; frazione di stime fuori [0,1] e distribuzioni sono riportate. Sigmoid e pooling max sono alternative configurabili, da selezionare mediante sviluppo/CV, con un nome di run diverso.

Limiti: la riscalibrazione ai bordi modifica il comportamento rispetto a una convoluzione ordinaria; è una scelta modellistica dichiarata. I frammenti separati non scambiano vicinato attraverso i bordi delle tile. Eventuali contributi di tile sovrapposte sono mantenuti e pesati anche nei baseline appaiati: non si assume che questo coincida con un mosaico unico senza duplicazioni.

## E. File creati

Tutti i file sono sotto `experiments/v1_spatial_cnn_msa/`:

| File | Responsabilità |
|---|---|
| `AUDIT.md` | Evidenze precedenti all'implementazione |
| `README.md` | Installazione, download, avvio, ripresa e monitoraggio |
| `REPORT_IMPLEMENTAZIONE.md` | Ricostruzione tecnica e limiti della consegna |
| `RESOURCE_ESTIMATES.json`, `VALIDATION.json` | Stime e risultati verificabili della preparazione |
| `config.yaml` | Dati, sampling, storage, architettura e training |
| `runtime.py` | Config/CLI, membership congelata, budget, lock e status |
| `sources.py` | Target, inventario, acquisizione e verifica V1.0 |
| `patch_builder.py` | Target/fetch/build/estimate e indice transazionale |
| `dataset.py` | Dataset spaziale, medie, normalizzatore e collate mascherato |
| `model.py` | CNN con maschera e MLP di controllo |
| `train.py` | Profiling, CV, checkpoint, refit e inferenza esterna |
| `evaluate.py` | OOF/Liguria, confronto appaiato, diagnostica e grafici |
| `preflight.py` | Stato dati/cache/hardware e cause di blocco |
| `monitor.py` | Monitor terminale read-only, ETA e heartbeat |
| `setup_windows.ps1` | Ambiente privato Python 3.12/PyTorch CUDA 12.8 |
| `run.ps1` | Sequenza esplicita pilot/intermedio/full con stop su errore |
| `tests/conftest.py`, `tests/test_experiment.py` | Fixture e test CPU senza rete |

Ogni script Python, PowerShell e modulo ausiliario inizia con un commento che ne descrive scopo, fasi, input/output, parametri, risorse e ripresa. La logica condivisa esistente viene importata senza modificarla quando necessaria, per esempio nell'acquisizione del target.

## F. Verifiche eseguite e ancora necessarie

La suite dell'esperimento verifica caricamento spaziale, geometria, media dei pixel equivalente al baseline appaiato, maschere, patch multiple, loss alla cella, backward e cambiamento dei pesi, normalizzazione train-only, split, checkpoint, inferenza deterministica, export, metriche, copertura parziale esplicita, ripresa transazionale e HTTP. La ripresa MLP interrotta viene confrontata con il training continuo, richiedendo identità esatta dei pesi su CPU.

Un'integrazione sintetica con 17 AOI e Liguria esegue tre modelli, tre fold, refit, predizioni e grafici, verificando che Liguria non entri nel normalizzatore o nella selezione. Sono verificati anche il profiling dei batch e il monitor read-only. Le fixture sono piccole e temporanee: non costituiscono risultati scientifici.

La suite congiunta include anche i test preesistenti del repository. Conteggi, comando e stato finale sono registrati in [VALIDATION.json](VALIDATION.json). I warning di deprecazione Affine riguardano l'operatore per applicare l'affine agli array e non hanno causato errori nei test.

Non sono disponibili PowerShell e una GPU NVIDIA sul Mac. **Esecuzione PowerShell nativa, CUDA sulla RTX 5080, driver, download completo e prestazioni sul PC universitario restano da verificare.** `setup_windows.ps1 -CheckGpu` esegue un vero forward/backward della testa; va lanciato dopo la conclusione del run esistente. Il lock di dipendenze comune non è stato modificato.

## G. Risorse e tempi

| Preset | Sviluppo | Liguria | Tile nominali | Sorgenti int8+scale stimate | Indice a 4 KiB/cella |
|---|---:|---:|---:|---:|---:|
| pilot | 850 | 150 | 151 | 17,06 GiB | 0,004 GiB |
| intermediate | 8.500 | 1.500 | 155 | 17,53 GiB | 0,038 GiB |
| full | 127.909 | 4.991 | 155 | 17,53 GiB | 0,507 GiB |

Sono stime geometriche e di pianificazione, non misure di download/completamento: non certificano numero di frammenti, copertura o spazio finale. La selezione sparsa del pilot può toccare quasi tutte le tile del full. Il volume non diminuisce proporzionalmente al numero di celle. Le fonti già disponibili vengono riutilizzate.

La stima full è circa 93,87 milioni di pixel, media circa 706 per cella. Una copia float32 a 128 canali richiederebbe circa **44,76 GiB** prima di maschere e metadati: l'indice evita tale duplicazione. Il numero reale di patch e i pixel validi vengono misurati nel build e riportati in `coverage.csv`/`cache_report.json`, senza trasformare l'ipotesi nominale di una patch per cella in una misura.

La work-root ha limite iniziale 60 GiB e richiede 20 GiB liberi di riserva. Sul Mac lo spazio rilevato era circa 15 GiB: il preflight ha correttamente bloccato l'acquisizione, insieme all'assenza dei target nella root nuova. Sul PC si devono riconfermare spazio, 32 GB RAM e RTX 5080 16 GB.

`train.py --profile` prova microbatch crescenti fino al batch configurato, include lettura/dequantizzazione nel tempo di training e registra picco VRAM. Le stime di epoca e budget massimo CNN derivano dai batch osservati: tre fold equivalgono complessivamente a due passaggi sullo sviluppo, più un refit. La stima di validation forward esclude I/O; MLP/HistGB, cache fredda, eterogeneità tra aree ed early stopping richiedono misure separate. Il campione tecnico non garantisce il picco peggiore di tutte le celle.

**Non vengono inventati tempi per epoca, ore totali o picco VRAM:** questi campi sono null nelle stime statiche. La CPU può dominare build e I/O anche quando il training usa la GPU. Se il batch provato produce OOM, il lancio si ferma invece di ridurre pixel o patch di nascosto.

## H. Protocollo di confronto scientifico

Il manifest congelato contiene 132.900 celle: **127.909 sviluppo in 17 AOI e 4.991 Liguria Levante**. Il suo SHA normalizzato LF è verificato contro la reference versionata. Il target rimane GLOBIO4 2020 `msa_area/cellarea`, non viene sostituito con BII. Umbria nord rimane esclusa. Il pilot seleziona 50 celle per AOI e 150 liguri con una regola geografica deterministica indipendente dal target; l'intermedio usa 500 per AOI e 1.500 liguri.

Si riusano i tre fold geografici originali, senza assegnazioni casuali di patch:

| Fold | Celle training full | Celle validation full |
|---|---:|---:|
| 0 | 89.101 | 38.808 |
| 1 | 83.014 | 44.895 |
| 2 | 83.703 | 44.206 |

CNN, Mean MLP, Mean HistGB e Dummy consumano gli stessi ID, target, maschere e pixel eleggibili. Il normalizzatore usa soltanto i pixel delle celle di training del fold, mediante le statistiche dell'indice. Il Dummy usa la media del training del fold. HistGB ripete i parametri storici, compreso l'early stopping interno di scikit-learn; questo non sostituisce la validation geografica esterna.

Per le teste neurali: AdamW, LR 1e-3, weight decay 1e-4, MSE, massimo 150 epoche, patience 15, clipping 1, seed 42, batch iniziale 16 celle e precisione FP32. Il best checkpoint è scelto con macro-AOI RMSE di validation. La selezione del modello usa le predizioni OOF; le epoche di refit sono la mediana delle epoche migliori dei fold. Il refit utilizza tutto lo sviluppo, senza Liguria.

Si esportano MAE, RMSE pooled, MAPE con esclusione esplicita dei target zero e denominatore, R², Pearson e bias; fold, AOI e macro-AOI restano distinti. Sono prodotti scatter, istogrammi, distribuzioni e mappa dei residui, oltre a diagnostica per intervallo MSA, copertura, numero di patch ed eterogeneità degli embedding. Non si inventano etichette ambientali assenti.

La vecchia assegnazione tramite `trunc` ai bordi dei ritagli può includere supporti diversi dalla nuova assegnazione fisica tramite `floor`. I baseline vengono quindi ricalcolati sui pixel effettivamente usati dalla CNN. Quando disponibili, i conteggi storici vengono confrontati nel report di supporto. Il MAPE storico **26,451133%**, MAE 0,12206058, RMSE 0,15807774, R² 0,056603 e bias −0,0465199 sono riferimenti descrittivi, non una prova di equivalenza del supporto.

Liguria è un'area esterna già osservata in esperimenti precedenti, non un blind test mai consultato. I suoi risultati e le diagnostiche non devono essere utilizzati per scegliere architettura, attivazione o soglia. Un eventuale miglioramento appaiato riguarderebbe la testa e l'informazione spaziale, non un encoder V1 modificato.

## I. Avvio e monitoraggio

I comandi completi sono nel [README](README.md): clone separato su `C:\BioMAP`, setup privato, preflight, acquisizione esplicita target e mappe, pilot, monitor nel secondo terminale, comandi separati, intermedio e full. Non richiedono attivazione della venv o percorsi Mac.

`run.ps1` avvia soltanto quando eseguito dall'utente. Esegue eventuali download richiesti, build, preflight training, profiling, CV/refit, valutazione OOF e Liguria. Il full richiede il flag esplicito `-FullApproved`. Nessun comando costoso è stato avviato durante la consegna.

`monitor.py --follow` legge `status.json` e `pipeline.json`: fase, progresso, velocità, ETA, epoca/batch, VRAM ed errori. Rimane aperto tra le fasi della sequenza; Ctrl+C chiude solo il monitor. L'ETA è una stima osservata: inizialmente può essere assente e il costo di job differenti non è uniforme. Un heartbeat vecchio è segnalato.

## J. Sicurezza dello sviluppo parallelo e ripresa

La Sessione B è stata sviluppata in un worktree indipendente, con branch dedicato. Il commit contiene esclusivamente questa directory. Nessuna modifica a `src`, `scripts`, `configs`, lock di dipendenze o `experiments/v2_dense_cnn_msa`. Non sono stati inviati comandi a processi Windows, arrestati processi di estrazione, aperti checkpoint V2 in scrittura o aggiornati i file del run in corso. La pubblicazione riguarda il ramo dedicato, senza merge automatico in `main`.

Sul PC il README usa un clone indipendente per evitare di cambiare codice sotto il processo attuale. I dati dell'esperimento finiscono in `data/v1_spatial_msa`, separati da V2. La sequenza PowerShell ha un lock OS e ogni fase Python un FileLock; l'esistenza residua del file lock dopo un'interruzione non equivale a un processo attivo.

Il training salva `last.pt` e `best.pt` separati con optimizer, scaler e RNG Python/NumPy/Torch/CUDA. Le identità includono cache, configurazione e hash del codice. Con `--resume` si riprende dall'ultima epoca completata, conservando i fold già finalizzati e verificandone checksum. Non eseguire pull nel clone di questo esperimento mentre il suo training è attivo.

## K. Stato finale e verifiche mancanti

| Aspetto | Stato della consegna |
|---|---|
| Codice | Implementato nel ramo dedicato; test CPU eseguiti |
| Mappe spaziali | Confermate su campione Mac; sorgente download implementata; copertura completa Windows da verificare |
| Target | Acquisizione isolata preparata; disponibilità completa locale non certificata |
| Estrazione | Comandi e ripresa pronti; avvio condizionato da preflight/spazio/sorgenti |
| Training | Pipeline implementata; PowerShell/CUDA/pilot reali ancora da verificare sul PC |
| Comparabilità | Baseline appaiati implementati; equivalenza al supporto storico non presunta |
| Risultato scientifico | Nessun miglioramento misurato o dichiarato |
| Full | Non avviato; richiede esecuzione esplicita dopo il pilot |

Non mancano pesi dell'encoder per questa modalità precalcolata. I blocchi pratici restanti sono acquisizione/verifica dei dati sul PC, spazio disponibile, accessibilità delle tile V1.0 e prova CUDA/PowerShell. Il preflight espone queste condizioni senza dichiarare pronto un esperimento privo della cache necessaria.
