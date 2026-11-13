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

Le sezioni successive saranno aggiornate con funzionalità e verifiche realmente completate; questo audit iniziale non dichiara il run reale pronto né eseguito.
