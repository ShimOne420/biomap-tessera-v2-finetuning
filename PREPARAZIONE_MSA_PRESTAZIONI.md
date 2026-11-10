# Preparazione MSA: diagnosi, ottimizzazione e ripresa su Windows

> **Aggiornamento del 6 novembre 2026.** Questa guida descrive la prima versione del motore ottimizzato. Il profilo usa ora un lotto per area, nessuna cache di blocchi e checkpoint unici per giorno: i numeri su lotti da 16 celle, quattro worker e cache da 8 GiB nelle sezioni di diagnosi e strategia sono storici. Comandi attuali nel [README](README.md#preparazione-parallela-per-aree), dettagli nel [report delle modifiche](REPORT_ACCELERAZIONE_PREPARE_MSA.md).

Il profilo `configs/msa_v1_liguria.yaml` usa ora automaticamente il motore `optimized`. Non cambiano il comando `prepare`, le dipendenze o il formato degli input TESSERA. La GPU rimane normalmente inattiva durante l'acquisizione: viene usata nel successivo `run` per gli embedding.

## Diagnosi del percorso precedente

Analisi del codice al commit `b0783a6`, non una profilazione remota del PC universitario:

| Operazione | Costo precedente | Correzione |
|---|---|---|
| `data.Reader.values` | Riapertura di SCL + 10 bande S2 e di VV/VH S1 per scena e per cella; `ds.sample` richiede letture 1×1 per i 25 punti | Un'apertura per banda e lotto quando servono blocchi non presenti; letture di finestre native condivise |
| Cataloghi `data.query` | Le risposte erano già condivise per AOI in questo YAML, ma il JSON veniva decodificato nuovamente per ogni cella | Un caricamento per AOI/collezione nell'invocazione; stessa risposta STAC congelata su disco |
| HTTP/SAS | Nuovo reader/sessione per cella; rinnovi dei token e retry GDAL più retry esterni | Reader condiviso, sessioni per thread, token sincronizzati e retry limitati |
| `common.budget` | Scansione ricorsiva di tutta la data-root prima di molte scritture, sempre più costosa con l'aumento della cache | Inventario iniziale unico, conteggio incrementale e controllo costante dello spazio libero |
| Orchestrazione | Una cella alla volta, bande seriali, avanzamento visibile solo dopo un'intera serie annuale | Lotti vicini, quattro worker I/O, heartbeat ogni dieci secondi |

Le celle vicine richiedevano spesso **gli stessi asset e gli stessi blocchi COG**. La cache dei giorni era locale alla cella: non consentiva il riuso fra celle. GDAL poteva riusare parte dei byte in memoria, ma mancava una cache persistente condivisa dei blocchi e rimanevano aperture, campionamenti e trasformazioni ripetuti.

Quattro celle ogni trenta minuti equivalgono a otto celle/ora: estrapolando linearmente, 132.900 celle richiederebbero circa **692 giorni continuativi**. È un'estrapolazione del dato segnalato, non una durata misurata dell'intera campagna.

## Strategia implementata e rischi controllati

`preparation.py` raggruppa al massimo 16 celle vicine della stessa AOI soltanto per organizzare l'I/O. `raster_cache.py` legge finestre sulla griglia originaria, senza interpolazione, e mantiene una cache condivisa in `observation_cache/native_v1`. La chiave comprende scena, banda, versione/metadati disponibili dell'asset e URL senza firma SAS. Le firme che scadono non invalidano pixel già acquisiti. Non si riscaricano scene complete.

| Scelta | Beneficio atteso | Rischio e controllo |
|---|---|---|
| Finestre native, lato massimo 512 pixel | Riuso dei pixel fra punti e celle vicine | Lettura di pixel aggiuntivi rispetto ai soli punti; cache limitata e confronto esatto con `rasterio.sample` |
| Lotti di 16 celle | Ammortizza aperture e metadati su più celle | Primo lotto può richiedere tempo prima di completare celle; heartbeat mostra sorgente/data corrente |
| Quattro worker tra bande | Sovrappone latenza di rete | Pressione sul server e RAM: limite configurabile 1–8, niente handle GDAL condivisi |
| RAM 256 MiB, disco 8 GiB | Riuso rapido e persistente dei blocchi | La LRU elimina soltanto propri blocchi temporanei, mai `series.npz` completati |
| Cache GDAL 64 MiB | Evita un consumo predefinito eccessivo | Limite impostato una volta nel thread principale e ripristinato dopo il run |
| Inventario disco incrementale | Elimina scansioni ripetute del filesystem | Protezione OS contro prepare concorrenti nello stesso namespace; spazio libero verificato a ogni scrittura |

Il limite disco resta **50 GiB per tutta la data-root** con almeno **20 GiB liberi**. Gli 8 GiB della cache nativa sono compresi, non aggiuntivi. Non è ancora misurato lo spazio finale delle 132.900 serie. La contabilità incrementale presuppone che altri programmi non scrivano contemporaneamente nella data-root; lo spazio libero effettivo è comunque ricontrollato. Ogni invocazione ricostruisce l'inventario, senza fidarsi di un totale vecchio.

Il catalogo salvato è il riferimento del run. Se il fornitore sostituisse un raster dietro lo stesso URL senza aggiornare i metadati STAC, la cache non potrebbe rilevarlo su un hit che evita completamente la rete. Non aggiornare cataloghi o asset a metà esperimento. I metadati sono verificati quando si riapre un asset e gli hash dei cataloghi sono registrati nel report.

## Invarianti scientifiche

- Stesse **132.900 celle**: 127.909 sviluppo e 4.991 test Liguria, con gli stessi tre fold geografici.
- Stessi target **GLOBIO4 overall MSA 2020**, senza imputazioni o sostituzioni.
- Stessi **25 punti** stratificati per cella, stesso anno e stesse osservazioni temporali.
- Stesso ordine delle scene per nuvolosità/ID e stesso arresto del mosaicking **per singola cella**, anche nei lotti.
- Stesse bande, SCL, NoData, clipping, conversione S1, orbite e politica del giorno dell'anno.
- Stesse serie, maschere, normalizzazione, binning e input dell'encoder. Solo l'ordine delle operazioni I/O cambia; il CSV finale conserva l'ordine originario.
- Le celle con copertura insufficiente vengono segnalate. Il benchmark completo rifiuta conteggi ridotti: nessun sottoinsieme viene presentato come equivalente.

Il percorso originale resta disponibile con `--set acquisition.engine=legacy` come riferimento. Non è la configurazione consigliata per la campagna grande.

## Comandi sul PC universitario

### 1. Fermare il vecchio processo e ricevere la correzione

Premere **Ctrl+C** nella finestra del vecchio `prepare` e attendere che ritorni il prompt. Non fare pull mentre il processo sta ancora usando il codice. Non cancellare la cartella dati.

```powershell
Set-Location 'C:\Users\brome\Desktop\BIOMAP_TESSERA_V2\biomap-tessera-v2-finetuning'
git status --short
git pull --ff-only origin main
$Python = Join-Path $PWD '.venv\Scripts\python.exe'
$Cfg = 'configs\msa_v1_liguria.yaml'
$DataRoot = (Resolve-Path '..\data').Path
& $Python -m pip check
```

Prerequisiti: ambiente già configurato e la stessa `..\data` del comando precedente. Non servono nuove librerie. Se Git segnala modifiche locali, conservarle e risolverle prima del pull, senza usare reset distruttivi. Ridefinire le variabili quando si apre una nuova finestra PowerShell.

### 2. Riprendere direttamente tutte le celle

```powershell
& $Python scripts\probe_msa.py prepare --config $Cfg --data-root $DataRoot
```

È lo stesso comando già usato: ora il YAML seleziona il motore ottimizzato. Riusa pesi/target/cataloghi e **verifica e conserva le serie già complete**, senza riscriverle. Il download dei pesi non è necessario per `prepare`. Per i comandi storici a 25 punti vedere [la guida archiviata](docs/archive/README_RUN_MSA_V1_V2_25POINTS.md); per il probing denso corrente vedere il [README](README.md).

Output: `..\data\msa_v1_liguria\series`, `cells.csv` al completamento e report sotto `diagnostics`. A ogni heartbeat compare una riga simile a:

```text
prepare 1234/132900 | cached=1202 new=32 | 180.0/h ETA=731.48h | AOI s2 2020-06-01, batch=16
```

**La riga è un esempio di formato, non una prestazione misurata.** La velocità usa solo celle nuove elaborate, senza gonfiarla con quelle già in cache. L'ETA si aggiorna dopo il primo lotto completato e può oscillare: densità di scene, copertura, banda e cache cambiano per territorio. All'inizio può comparire `warming up`/`unknown` mentre sono comunque visibili sorgente e data.

### 3. Leggere la profilazione anche da una seconda finestra

```powershell
Set-Location 'C:\Users\brome\Desktop\BIOMAP_TESSERA_V2\biomap-tessera-v2-finetuning'
$DataRoot = (Resolve-Path '..\data').Path
$Profile = Get-ChildItem (Join-Path $DataRoot 'msa_v1_liguria\diagnostics') -Directory -Filter '*_prepare' |
    Sort-Object Name -Descending | Select-Object -First 1
Get-Content (Join-Path $Profile.FullName 'progress.json') -Raw
```

Ripetere l'ultimo comando per aggiornare. Ogni invocazione produce una cartella distinta:

| File | Contenuto |
|---|---|
| `config.json` | Configurazione effettiva e parametri I/O |
| `progress.json`, `profile.json` | Stato, celle complete/in cache/nuove/escluse, velocità, ETA, tempi e contatori; aggiornati durante il run |
| `catalog_AOI.json` | Hash dei cataloghi annuali effettivamente utilizzati |
| `assets.csv` | Identità hash, scena, banda, richieste, aperture, letture, hit RAM/disco, byte decodificati e tempi per asset; scritto alla chiusura normale o gestita con Ctrl+C |

Tempi principali: `stac_query`, `catalog_cache_load`, `sas_token`, `remote_open`, `remote_read`, `coordinate_transform`, `preprocessing`, `cache_write`, `block_cache_read`, `checksum`, `disk_inventory`, `disk_budget`. `remote_read_failures` e `remote_retries` permettono di distinguere rete instabile da cache lenta. Una query STAC comprende tutte le pagine; un caricamento di catalogo su disco non è una query remota. Il percorso legacy usa anche `catalog_query_or_load` e `legacy_point_read`.

**Interpretazione:** i secondi dei worker sono sommati e possono superare il tempo di parete. Le aperture raster non sono il numero di richieste HTTP Range interne a GDAL. `decoded_raster_bytes` misura dati e maschere decodificati, non traffico HTTP compresso. Non confrontare un run interamente in cache con un run a freddo come misura della rete.

### 4. Regolare l'I/O senza cambiare l'esperimento

Iniziare con i valori predefiniti e con `--parallel-aois 4`. Se il server restituisce molti errori o la rete è limitata, ridurre i processi o i worker:

```powershell
& $Python scripts\probe_msa.py prepare --config $Cfg --data-root $DataRoot --parallel-aois 2 --set acquisition.io_workers=4
```

Confrontare `blocchi/s` su **nuove** celle, senza dedurre vantaggi dai giorni già salvati. Questi parametri non modificano il campionamento. I limiti ammessi sono 16 `io_workers` e 16 `day_workers`; aumentarli non garantisce prestazioni migliori.

`block_cache_gib: 0` non cancella una cache esistente: `observation_cache\native_v1` resta su disco inutilizzata e occupa circa 8 GiB, conteggiati nel limite di `storage.max_gib`. Con `--aoi` e `--parallel-aois` la cache deve restare a zero. Non cambiare anno, punti, osservazioni o target per accelerare.

### 5. Errori di rete, interruzione e ripresa

I raster hanno timeout di connessione **10 s** e di operazione HTTP **30 s** e al massimo **otto tentativi** con attesa crescente e casuale fino a 60 s e rinnovo SAS; i blocchi già ottenuti non si rileggono. Il retry interno GDAL è disattivato per evitare moltiplicazioni dei tentativi. STAC e token usano al massimo due retry aggiuntivi, anche sulle POST di ricerca, per 429/500/502/503/504; viene rispettato `Retry-After`. Non è un limite di trenta secondi per un'intera banda: questa può richiedere più operazioni HTTP. I timeout sono configurabili nelle chiavi `acquisition.*` del YAML.

Con Ctrl+C attendere la chiusura dei worker: le operazioni già iniziate possono dover terminare o raggiungere il timeout. Il comando successivo è sempre lo stesso `prepare`. I giorni acquisiti restano in `batch_checkpoints` e vengono riutilizzati; le serie e i manifest vengono pubblicati atomicamente con transazioni recuperabili e checksum. Le serie del precedente motore restano compatibili. Un processo terminato forzatamente perde soltanto i giorni in corso.

Se una vecchia cache risulta corrotta o priva di manifest, il comando si ferma indicando il file: non la considera valida e non sovrascrive silenziosamente i dati. Il blocco OS rifiuta due processi sulla stessa area e un `prepare` completo mentre è attiva una qualsiasi area, e viene rilasciato anche in caso di terminazione del processo. La presenza dei piccoli file `prepare*.lock` sul disco non indica da sola un processo attivo.

Le firme SAS vengono rimosse dai messaggi rasterio; non attivare logging verbose HTTP/GDAL e non pubblicare chiavi o URL firmati.

## Evidenza di equivalenza e limiti della verifica

La suite include GeoTIFF sintetici realmente letti con rasterio: otto celle vicine, bande S2, SCL, scene con priorità diverse, nuvole, NoData, valori estremi, entrambi gli orientamenti S1 e il 31 dicembre. Una cella è deliberatamente non eleggibile in entrambi i percorsi. Il confronto verifica **uguaglianza esatta** degli array, delle celle eleggibili, degli input normalizzati/binning e degli embedding di uno student ridotto. Altri test coprono due CRS, bordi raster, NoData assente/numerico/NaN, lotti e worker differenti, cache vecchie immutate, hit persistenti senza rete, corruzione, LRU, retry, timeout STAC, interruzione e ripresa.

In questa fixture le aperture raster scendono da **362 a 70**; i caricamenti dei cataloghi da 16 a 2. Questo non equivale a una riduzione identica del tempo remoto. Un'esecuzione locale sui piccoli TIFF ha impiegato circa 0,50 s nel riferimento e 0,83 s con cache/thread: l'overhead locale è visibile quando manca la latenza di rete. La riduzione delle aperture e il riuso sono verificati; **il throughput sul Planetary Computer e il tempo finale sul PC universitario devono essere misurati con i nuovi report**. Nessun benchmark reale è stato accorciato o eseguito sul Mac come sostituto della campagna.

Per eseguire la suite nell'ambiente Windows già configurato:

```powershell
& $Python -m pytest
```

I test usano soltanto cartelle temporanee e non alterano `..\data`. La CI ripete la suite CPU su Windows e Linux; non misura il traffico remoto della rete universitaria.
