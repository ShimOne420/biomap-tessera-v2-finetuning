# Spatial CNN V2: audit e implementazione incrementale

Base verificata: `main` a `b7da382d4424c1ddd35a285830f95bb2d42c6436`, con integrazione A/B e infrastruttura incluse. Sviluppo su `feature/spatial-cnn-v2`; nessun accesso al PC universitario. Questo report distingue codice verificato su fixture e disponibilità dei dati reali, ancora da verificare sul PC.

## Audit iniziale e riuso

L'Esperimento B contiene già PartialConv mascherata, PixelNorm per pixel, GELU, pooling spaziale, aggregazione patch→cella pesata dai pixel validi e testa 32→64→1. Le primitive matematiche 3×3 e PixelNorm possono essere riusate senza modificare B. La baseline attuale fissa però 128 canali, ricava il numero di celle dai target e il collate alloca sempre 128 canali. Il nuovo percorso generalizza questi punti e conserva i nomi dei parametri per copiare i pesi nei test di parità.

`dataset.fit_normalizer` di B implementa correttamente i momenti pixel-weighted training-only: sarà riusato attraverso un caricamento isolato, senza rendere globali i suoi import non qualificati. Il trainer/checkpoint B rimane legacy: il nuovo engine deve identificare il feature space e funzionare senza target in inferenza. Le metriche comuni `dense.evaluation.full_metrics` sono già riutilizzabili; i tre fold verranno letti dal manifest congelato, non rigenerati dalla funzione ligure a cinque fold.

L'Esperimento A controlla marker, checksum, membership e target prima della GPU, ma legge medie `[128]`: le sue Conv1D operano sui canali. Non è una Spatial CNN. Esce dal percorso operativo principale, rimanendo disponibile per riprodurre esperimenti storici. Non vengono cancellati i suoi file o test.

## Disponibilità V2 Spatial: limite essenziale

La r1 salva `pixel_embeddings/<aoi>/<tile>.npy` soltanto con `--keep-pixel-embeddings`, disabilitato di default. Il sidecar r1 contiene `sha256`, `geometry`, `order`, `dtype`; il legame a checkpoint/protocollo è nel marker della tile aggregata. La r2-fast aggiunge `identity`, `geometry_sha256` e `grid`: non si devono scambiare i due schemi.

`v2.parquet`, `sums` e `counts` non permettono di recuperare le mappe. Inoltre una tile già completata viene riusata dalla r1 prima del ramo che conserva i pixel: aggiungere il flag su una work-root già completata non rigenera i pixel mancanti. La documentazione deve quindi distinguere verifica degli artefatti già presenti da un'eventuale nuova estrazione **in una directory separata**, solo dopo la campagna attuale. Nessun recupero automatico o cancellazione di cache.

## Protocollo e sequenza di lavoro

Il conteggio esatto è **127.909 celle di sviluppo in 17 AOI**, tre fold geografici originali, più **4.991 celle Liguria**, per **132.900 celle totali**. Non 129.000 celle di training. Liguria è già osservata, ma resta esclusa dal tuning. Il target rimane GLOBIO4 MSA 2020.

Le fasi implementative sono: contratti/adattatori/supporto; generalizzazione minima della CNN; test sintetici e parità B; indice e engine CV/refit/evaluation; documentazione Windows e audit fine-tuning. Commit separati rendono ispezionabili queste fasi. Non sono previste Residual CNN, attention, Transformer predittivi, halo o nuovi formati degli embedding.

L'indice spaziale usa il pattern SQLite già presente in B, senza duplicare le mappe. Gli originali rimangono read-only. Il reader legge soltanto i pixel della finestra richiesta e gestisce memmap con cache limitata. La membership spaziale V2 deriva dalle geometrie congelate della reference; non si corregge retroattivamente floor/trunc. Le sovrapposizioni mantengono la molteplicità legacy e vengono riportate.

## Moduli implementati

Il nuovo namespace è `src/biomap_spatial`: `contracts` per configurazioni/rappresentazioni, `sources` per Array/V1/V2 e indice, `support` per appartenenza e finestre, `normalization` per il fit indipendente, `batching` per P patch e B celle, `model` per baseline/proiezione/kernel, `engine` per CV/refit/evaluation/registry, `cli` per azioni esplicite e `utils` per import isolati/hash/I/O atomico. Un unico ingresso `scripts/spatial_cnn.py` e un unico YAML `configs/spatial_v2_msa.yaml` evitano script numerati duplicati.

Le funzioni corrette B non sono state riscritte: PartialConv 3×3, PixelNorm e fit pixel-weighted sono riusati. Per kernel 1 il blocco conserva la stessa semantica sul supporto, con vicinato di un pixel. La head mantiene GELU e dimensioni B. Diretta C=128 senza proiezione: nomi e shape dei pesi sono compatibili con la copia dello state_dict legacy; forward e gradienti sono confrontati esattamente su CPU.

`V1TileAdapter` legge int8 HWC, scale float32 HW e georeferenziazione TIFF; `V1IndexSource` usa un indice B finalizzato, sorgenti e momenti verificati, sempre read-only. `V2Source` legge mappe vere r1/r2, senza rimodellare le medie. Il runner prioritario usa V2. `ArraySource` consente training sintetico con C=32,128,384 e non contiene un encoder.

Sono modificati soltanto il README principale, AGENTS.md, il controllo import dev e `.gitignore`, oltre ai nuovi file. Non cambia il lock scientifico, la configurazione pytest o il workflow CI: la raccolta standard include automaticamente `tests/test_spatial_*.py`.

## Integrità e supporto degli input

Prima di interrogare CUDA, il preflight verifica completamento dell'estrazione FP32, checksum del parquet, reference e membership congelata, target/coordinate/fold, mappe e sidecar, marker delle tile, checkpoint d'origine, codice r1 pubblico, geometrie, ordine e dtype float32 `[N,C]`, valori finiti e griglia nativa. r1 e r2 hanno schemi distinti; un'incompatibilità interrompe il lavoro con un errore, senza fallback a dati differenti.

Durante prepare, i pixel delle finestre vengono verificati contro somme e conteggi delle tile aggregate originali. I conteggi devono coincidere esattamente; per somme float64 si usa un controllo `rtol=1e-12, atol=1e-10` per l'ordine di riduzione, senza cambiare i valori o ammettere supporto diverso. Un test altera un pixel e aggiorna il suo checksum: la discordanza con il chunk rimane rilevata. Il dato non viene accettato solo perché il suo file ha un hash internamente coerente.

Le mask escludono supporto esterno, padding e invalidi. Non si sostituiscono pixel validi NaN/Inf con dati inventati. Le geometrie V2 congelate rimangono l'autorità: il mapping non corregge retroattivamente i bordi legacy. Il confronto di parità con B riguarda identici input sintetici, non certifica un'uguaglianza geografica fra ogni vecchio indice B floor e ogni reference V2 storica trunc. Gli overlap originali mantengono lo stesso peso dei contributi; il numero di centri esattamente duplicati è riportato.

L'indice SQLite salva finestre/mask e momenti per cella, non i 50 GB di embedding. Ogni tile è una transazione. Un'interruzione lascia riutilizzabili solo le tile committate; marker e checksum dell'indice sono pubblicati alla fine. Un test interrompe dopo la prima tile e verifica la ripresa. Cambiare codice del mapping, supporto o dimensione patch invalida l'identità dell'indice: usare una root separata, non sovrascrivere vecchie cache.

Le letture usano memmap LRU e finestre selezionate tramite indici ordinati. Gli hash completi e la scansione dei valori leggono tutti i pixel: il primo prepare/preflight richiede I/O reale. Non sono disponibili benchmark di velocità o picco RAM sull'intera campagna; i tempi delle fixture non vengono estrapolati come promessa sui 99,5 milioni di pixel.

## Training, selezione e ablation

Default: PartialConv C→64, PixelNorm/GELU, PartialConv 64→32/GELU, masked mean per patch, media patch→cella pesata dai pixel, head 32→64→1. Proiezione 1×1 C→D opzionale. Inferenza guidata da cell_ids/owners, senza target. Gli invalidi sono azzerati tramite mask prima delle operazioni: non propagano NaN o gradienti non finiti.

La normalizzazione pixel-weighted riusa esattamente i momenti B; ogni fold fit soltanto su training. `cell_standard` è disponibile per il controllo tabellare, `none` è esplicita. La combinazione cell_standard+CNN viene rifiutata perché non ha una semantica spaziale definita; non è reinterpretata come normalizzazione per pixel bilanciata per cella. Nessuno scaler nascosto viene aggiunto a MLP/CNN.

CV: tre fold originali con AOI disgiunte; monitoraggio macro-AOI RMSE, massimo 50 epoche, pazienza 8. AdamW, LR 1e-3, weight decay 1e-4, clipping 1, batch iniziale quattro celle e accumulo verso 16. Le loss sommate vengono normalizzate per il numero effettivo di celle nel gruppo di accumulo, incluso l'ultimo gruppo corto. Una cella con più patch non pesa di più. FP32 di default; AMP CUDA è una variante esplicita.

Un run corrisponde a un modello/seed. La baseline Mean MLP usa le medie dei pixel dello stesso indice, il controllo pixelwise usa kernel 1, la Spatial CNN kernel 3. Parametri, kernel, dimensioni e receptive field sono registrati. I kernel 1 e 3 cambiano anche capacità: non isolano perfettamente la sola spazialità. I test con patch unica/due/quattro verificano copertura e una previsione per cella; l'invarianza del kernel 1 è verificata entro l'errore di riduzione float32, mentre per kernel 3 si registra la variazione dovuta ai bordi.

Il confronto tra run usa soltanto OOF e rifiuta differenze di source/support identity, target, celle, AOI/fold, normalizzazione o seed. Il refit usa tutte le 127.909 celle sviluppo per la mediana delle epoche migliori. `evaluate` applica il modello alla Liguria solo dopo refit verificato. Non c'è tuning su Liguria o imputazione per celle mancanti. Seed 43/44 e varianti projected richiedono run-id distinti.

## Checkpoint, registry, output e monitor

Ogni run salva configurazione effettiva, manifest con commit/hash del codice, encoder/feature-space/protocollo/supporto/device/precisione, checkpoint fold, optimizer/scaler/RNG e normalizzatore con ID training. La ripresa è dall'ultima epoca completa; l'epoca parzialmente elaborata viene ripetuta. Il test con dropout confronta i pesi di training continuo e interrotto/ripreso con uguaglianza esatta.

Il registry è solo filesystem, senza MLflow. OOF e predizioni esterne sono separati, ogni modello ha una riga per cella; Dummy usa la media del training appropriato. Metriche comuni riusate: MAE, RMSE, R², bias, Pearson e MAPE con esclusione dei soli zero e denominatore dichiarato. I target vicini a zero vengono contati e rimangono inclusi; pooled, fold, AOI e macro-AOI sono distinti. Sono salvati errori per intervalli MSA, patch, copertura ed eterogeneità, più grafici esterni. Confronti storici con supporto diverso restano descrittivi.

Il monitor mostra epoca/fold, celle, velocità ed ETA dell'epoca; dopo la prima epoca stima anche il budget residuo della fase. Early stopping e dimensioni dei fold rendono queste stime approssimative. Checkpoint e avanzamento sono indipendenti, con replace atomico/retry Windows; il monitor chiude subito i file. I dati di produzione non vengono aperti dal monitor.

`profile` misura un solo batch del predittore: tempo forward/backward e picco CUDA, senza caricare TESSERA. Non misura il massimo batch sostenibile né il costo di un futuro encoder fine-tuned. Il README indica come modificare batch/proiezione e ripetere questa prova prima del training reale.

## Validazione e limiti di prontezza

Baseline prima delle modifiche: **168 passati, 0 falliti, 1 skip Windows-only**, 47,96 s sul Mac. Nuovi test mirati: **40 passati, 0 falliti, 0 skip**, 6,42 s. Coprono reader r1/r2, ripresa indice, checksum/coorti, somme originali, geometrie irregolari/bordi/overlap/risoluzioni, dequantizzazione V1, C variabile/proiezione, forward/gradienti e parità B, normalizzatori, ripresa pesi, inferenza senza target, CV/refit/evaluation e confronti incompatibili.

Durante lo sviluppo è stato corretto un errore nella fixture SQLite V1 (numero di placeholder) e completato il controllo delle duplicazioni dell'adapter V1; i test finali conservano quelle verifiche. Nessun fallimento è stato trasformato in skip, nessun test legacy eliminato. La suite legacy esegue anche piccoli PixelStudent casuali CPU già esistenti; non carica il Medium preaddestrato. Il nuovo percorso esegue soltanto CNN/MLP su fixture.

Il codice è pronto per verifiche CPU riproducibili; i comandi Windows sono preparati. **La disponibilità dei pixel universitari e il profilo RTX 5080 restano da verificare sul PC.** Se il run corrente ha conservato soltanto medie, sarà necessaria una nuova estrazione esplicita dei pixel in una root separata; aggiungere il flag su tile già cache-hit non basta. Non dichiariamo quindi dimostrata l'immediata avviabilità del training reale, né alcun miglioramento MSA.

Il [runbook](windows_runbook.md) contiene verifica, download pesi solo se necessario, nuova estrazione opzionale, ripresa e percorso corretto al training. Il [rapporto di fattibilità](finetuning_feasibility.md) è statico: nessun fine-tuning implementato/eseguito. Restano rinviati Residual CNN, nuovi encoder, attention, Transformer predittivi, halo e nuovi formati degli embedding.

I risultati della suite completa, l'audit r1 e la CI del commit pubblicato sono registrati nel riepilogo di validazione e nella PR; gli esiti remoti non sono anticipati da questo report.


## Verifica finale prima della pubblicazione

Sul codice `a383cba`, la suite unificata finale ha dato **210 passati, 0 falliti, 1 skip Windows-only**, 1.426 warning, **53,67 secondi**. I nuovi test sono **42 passati**, nessun skip, 8,44 secondi. La verifica precedente (208/0/1) è stata estesa con i test di inizializzazione deterministica CUDA senza inizializzare GPU e di parità/configurabilità del Mean MLP. Import, sintassi e piccolo backward CPU passano; tutti i 21 casi B rimangono inclusi.

È configurato `CUBLAS_WORKSPACE_CONFIG=:4096:8` prima dell'inizializzazione CUDA, come nel trainer B, e profile usa le stesse impostazioni deterministiche del training. Questo prepara il percorso tecnico, ma non sostituisce la prova sulla GPU. I limiti del determinismo sono documentati da [PyTorch](https://docs.pytorch.org/docs/2.10/generated/torch.use_deterministic_algorithms.html) e la configurazione workspace da [NVIDIA](https://docs.nvidia.com/cuda/cublas/index.html#results-reproducibility). Non promettiamo identità fra CPU/MPS/CUDA.

Configurazione effettiva e selezione OOF hanno checksum verificati prima di ripresa/refit/confronto; il test di alterazione delle epoche scelte viene rifiutato. Gli output restano nella root privata, con guardia sui nomi dei run. La verifica raw ha confermato **5/5 SHA256 e 5/5 EOL r1** rispetto alla fixture pubblica; il diff di tutti i moduli V2 legacy, Dense heads, esperimenti A/B, workflow, `.gitattributes`, lock e configurazione pytest rispetto al main base è vuoto. I dettagli riproducibili sono in [validation.json](validation.json).

Il branch è `feature/spatial-cnn-v2`, con commit distinti per audit, contratti/reader, CNN, engine, documentazione e guardie di riproducibilità. La pubblicazione apre una PR verso main e non esegue merge. Gli esiti remoti finali saranno leggibili nei check e nella descrizione della PR; il JSON conserva lo stato locale verificato al momento del commit documentale.
