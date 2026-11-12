# Audit precedente all'implementazione — 11 novembre 2026

Base Git `e720ad0c365ed55e29c4439c7a7e63a92f34cb54`; branch/worktree separati `experiment/v1-spatial-cnn-msa`. Modifiche consentite esclusivamente in questa directory. Nessuna lettura dei checkpoint o scrittura nei risultati della Sessione A; nessun processo V2 avviato, arrestato o modificato.

## V1 effettivo

Il lettore esistente `src/biomap_v2/dense/v1_support.py` consuma prodotti V1 già calcolati: NPY int8 `[H,W,128]`, scale float32 `[H,W]`, TIFF landmask per CRS e affine. Dequantizzazione `int8.astype(float32) * scales[...,None]`, validità determinata dalle scale finite. Non carica un encoder V1. Il parquet storico contiene medie/std, non mappe spaziali.

Su SSD è stata letta una tile reale ligure `grid_9.55_44.35`: shape `[1116,804,128]`, scale `[1116,804]`, EPSG:32632, pixel 10×10 m. Le finestre dei 4.991 rettangoli liguri in quel CRS hanno altezza 31–33 (mediana 32) e larghezza 23–24 (mediana 23). È una misura ligure, non di tutte le aree.

I metadati storici fissano **dataset V1.0 / variante vultr / anno 2020**, sorgente `https://data.source.coop/tessera/tessera/npy/v1/`. Il prodotto corrente V1.1 ha uno spazio di feature diverso: non è un sostituto automatico. Documentazione ufficiale: [GeoTessera](https://github.com/ucam-eo/geotessera), [TESSERA](https://github.com/ucam-eo/tessera), [formato e versioni](https://geotessera.readthedocs.io/en/stable/index.html).

Il checkpoint QAT originario che generò V1.0 è identificato nella documentazione ufficiale come `best_model_fsdp_20250608_220648_QAT.pt`; **non è stato trovato/verificato un suo checksum locale e non è necessario al consumo delle mappe precalcolate**. Questa consegna non inventa un percorso di inferenza raw Sentinel V1: prepara riuso/download esplicito del prodotto V1.0 congelato, senza encoder durante il training.

## Dati e protocollo

Il manifest versionato contiene 132.900 ID: 127.909 sviluppo in 17 AOI e 4.991 Liguria, con i tre fold geografici originali. `configs/msa_v1_liguria_reference.json` fissa membership/hash e risultati storici. Target MSA = msa_area/cellarea, 2020, griglia EPSG:4326 a 10 arcsec. I target storici e la reference portabile completa non sono garantiti disponibili sul PC universitario; l'audit del 10 novembre segnala file iCloud non materializzati. Il bundle Liguria su Release contiene medie: non basta per questa CNN.

Sono presenti mappe V1 raw su SSD per più aree e 10 tile liguri. Questa disponibilità sul Mac non implica disponibilità sul PC Windows. La preparazione dovrà controllare tutti i file necessari e consentire download separato, esplicito, nello spazio dell'esperimento. Non viene lanciata alcuna estrazione completa.

## Scelta architetturale

Strategia B: una o più finestre native per cella, separate per tile/CRS, con maschera rigorosa dei centri dentro la cella. Massimo lato 64 come limite di suddivisione, motivato dalle finestre liguri osservate; nessun resizing. Una cella resta una sola osservazione statistica. Pooling dei frammenti pesato per pixel validi, loss alla cella.

Per evitare una copia float32 da circa 50 GB, l'indice spaziale usa SQLite transazionale: finestre, maschere bit-packed, statistiche e provenienza, riferimenti ai file int8/scale originali. DataLoader con memmap read-only. La dimensione esatta dei frammenti è verificata nel preflight/preparazione di ogni scala.

La nuova assegnazione geometrica usa appartenenza fisica alla cella (floor e confini semiaperti). Il codice storico contiene il noto comportamento `trunc` ai bordi dei ritagli e conserva eventuali contributi sovrapposti. Per questo i baseline vengono **ricalcolati dagli stessi frammenti della CNN**; l'eventuale differenza rispetto a n_valid/mean storici viene riportata, non corretta silenziosamente. Non si promette equivalenza integrale al numero storico 26,45%.

CNN con convoluzioni parziali e maschera mantenuta sul supporto originale, normalizzazione per pixel sui canali, pooling mascherato. Evita che padding/valori invalidi contaminino feature o statistiche spaziali. La divisione in frammenti interrompe il vicinato fra tile: limite documentato, non ricostruzione inventata.

## Risorse e blocchi

Hardware Windows dichiarato: RTX 5080 16 GB, RAM 32 GB, circa 100 GB liberi da riconfermare. La CPU legge/dequantizza; la GPU allena soltanto CNN/MLP. Memmap/letture casuali, download V1 e validazione delle sorgenti sono potenziali colli di bottiglia. Tempo/VRAM richiedono pilot reale; non sono deducibili da test sintetici.

Stato al termine dell'audit: **codice da implementare; mappe spaziali confermate su campione locale; disponibilità Windows e target completi da verificare**. Nessun miglioramento scientifico dimostrato. Segue implementazione locale isolata e verifiche.
