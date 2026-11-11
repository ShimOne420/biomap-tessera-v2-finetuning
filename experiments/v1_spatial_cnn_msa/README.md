# Sessione B — GeoTESSERA V1.0 spatial CNN → GLOBIO4 MSA

**Ramo:** `experiment/v1-spatial-cnn-msa`. **Codice confinato a questa directory.** Non integrare automaticamente con la Sessione A e non fare checkout/pull nel working tree usato dal processo V2 attivo.

Si allenano una CNN spaziale e i controlli mean+MLP/HistGB sullo stesso supporto V1.0 congelato. Si usano mappe int8 reali a 128 canali/10 m con scale per pixel; nessun encoder, nessuna immagine costruita da un vettore medio, nessun fine-tuning del foundation model.

**Stato:** implementazione e test CPU disponibili. La presenza delle mappe è stata verificata sul Mac, anche con un piccolo audit geometrico reale; ciò non dimostra la disponibilità sul PC Windows. Il full non è stato eseguito. `preflight.py` distingue `BLOCKED`, `READY_FOR_EXTRACTION` e `READY_FOR_TRAINING`. Leggere [AUDIT.md](AUDIT.md), [REPORT_IMPLEMENTAZIONE.md](REPORT_IMPLEMENTAZIONE.md) e [RESOURCE_ESTIMATES.json](RESOURCE_ESTIMATES.json).

## 1. Scaricare il ramo in un checkout separato

Questi comandi **non cambiano i file del checkout attualmente in esecuzione**. Creano un clone indipendente in `C:\BioMAP`, con dati locali sul disco C. Se la cartella destinazione esiste già, entrarvi e verificarla; non sovrascriverla e non forzare reset/stash.

```powershell
New-Item -ItemType Directory -Force 'C:\BioMAP' | Out-Null
git clone --single-branch --branch experiment/v1-spatial-cnn-msa https://github.com/ShimOne420/biomap-tessera-v2-finetuning.git 'C:\BioMAP\biomap-v1-spatial-cnn-msa'
Set-Location 'C:\BioMAP\biomap-v1-spatial-cnn-msa'
```

Il clone può essere creato mentre il vecchio processo continua; **avviare i comandi GPU seguenti soltanto dopo la sua conclusione**. Per aggiornare successivamente questo clone, usare `git pull --ff-only` quando il suo run è fermo. Il ramo non viene unito a `main`. Il codice della Sessione A in `experiments/v2_dense_cnn_msa/` non viene modificato.

## 2. Ambiente Windows privato

Servono Git e Python 3.12 x64 con launcher `py`. Il setup crea `experiments\v1_spatial_cnn_msa\.venv`, separato dall'ambiente V2. Installa PyTorch 2.10.0 CUDA 12.8 e il lock già presente nel repository, senza modificarlo. Non occorrono pesi V1: si consumano embedding precalcolati V1.0.

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\experiments\v1_spatial_cnn_msa\setup_windows.ps1 -CheckGpu
$Py = (Resolve-Path '.\experiments\v1_spatial_cnn_msa\.venv\Scripts\python.exe').Path
$DataRoot = [System.IO.Path]::GetFullPath('..\data')
& $Py -m pytest experiments\v1_spatial_cnn_msa\tests
```

Il controllo deve vedere la RTX 5080 e superare forward/backward della testa. Per soli test CPU il setup accetta `-CpuOnly`; per il training CPU occorre anche `--set training.device=cpu`. Il setup non modifica driver e non aggiorna l'ambiente del run esistente.

## 3. Preflight e stima prima di scaricare

```powershell
& $Py experiments\v1_spatial_cnn_msa\preflight.py --data-root $DataRoot --preset pilot
& $Py experiments\v1_spatial_cnn_msa\patch_builder.py estimate --data-root $DataRoot --preset pilot
```

Senza target in una nuova root, `BLOCKED` ed exit code 2 sono previsti: leggere le cause nel JSON, poi acquisire esplicitamente i dati. Non ignorare un errore di schema, griglia, integrità o spazio. I file sono in `$DataRoot\v1_spatial_msa\preflight` ed `estimates`.

Il pilot ha **850 celle sviluppo (50 per ognuna delle 17 AOI) e 150 Liguria**, preservando i fold originali. La scelta è geografica, deterministica e indipendente dai target. È una verifica/esplorazione ridotta; non equivale al benchmark completo.

**Disco:** la selezione sparsa del pilot può richiedere circa **17 GiB di tile** se non già presenti, quasi quanto il full. L'indice privato evita invece la copia float32 di circa 45 GiB stimata per tutte le celle. Default: massimo 60 GiB per questa work-root, riserva di almeno 20 GiB liberi. La stima non garantisce copertura o spazio finale: il controllo usa il disco reale.

## 4. Due modalità per i dati

### A. Riutilizzare V1 già disponibile

La radice V1 contiene `<aoi>\tessera\2020`, NPY int8, scale, TIFF e `tessera_metadata.json` con versione **1.0**, variante **vultr**, anno 2020. La reference opzionale è il pacchetto finalizzato da `dense_campaign.py pack-reference`, con `reference.json`, `v1.parquet` e relativi checksum. Il bundle di sole medie Liguria non è una mappa spaziale.

I segnaposto seguenti devono essere sostituiti con directory esistenti, **non con output V2 in corso di scrittura**:

```powershell
$V1Root = 'F:\PERCORSO_TILE_V1_PER_AOI'
$Reference = 'F:\dense127909_reference'
& $Py experiments\v1_spatial_cnn_msa\preflight.py --data-root $DataRoot --preset pilot --v1-root $V1Root --reference $Reference
```

Le sorgenti esterne sono lette in sola lettura. `fetch` può completare le tile mancanti in un overlay privato, senza scrivere nella sorgente originale. Come alternativa avanzata alla reference, `--targets-csv` accetta un CSV/parquet verificato con `cell_id,target`: deve contenere il rapporto GLOBIO4 2020 già validato e gli ID originali. La provenienza del CSV viene congelata; non viene inventata dal software.

### B. Scaricare personalmente target e mappe V1.0

Non richiede i dati sull'SSD del Mac o la reference completa dell'altro esperimento. I target vengono acquisiti con lo stesso servizio/formula della pipeline esistente, **per l'intero manifest già dal primo download**, evitando un ritaglio pilot insufficiente al full. Tutte le scritture restano nel namespace V1 spatial.

```powershell
& $Py experiments\v1_spatial_cnn_msa\patch_builder.py targets --data-root $DataRoot --preset pilot
& $Py experiments\v1_spatial_cnn_msa\patch_builder.py fetch --data-root $DataRoot --preset pilot
```

`fetch` scarica solo tile necessarie al preset e riusa quelle già complete. Fonte: prodotto Source Cooperative V1.0, non V1.1/Zarr. I tre endpoint di una tile reale sono stati verificati con richieste HEAD; nessun full è stato scaricato durante lo sviluppo. Un 404/copertura assente interrompe l'azione: non viene sostituito il prodotto o imputata una feature.

Download con `.partial`, Range/ETag, lunghezza, MD5 quando ETag è un MD5 singolo, SHA256 locale e pubblicazione recuperabile payload/marker. Riprendere con lo stesso comando. Non cancellare sorgenti, marker o checkpoint per forzare il completamento.

## 5. Primo run pilot completo — terminale 1

**Solo dopo la fine del run attuale.** Dalla radice del nuovo worktree:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\experiments\v1_spatial_cnn_msa\run.ps1 -DataRoot $DataRoot -Preset pilot -RunName pilot_seed42 -DownloadTargets -FetchV1 -Resume
```

Questo è un avvio esplicito: acquisizione se richiesta, indice spaziale, preflight, microbatch profiling, CV, refit, OOF e valutazione Liguria. Non è stato eseguito automaticamente qui. Con sorgenti esistenti:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\experiments\v1_spatial_cnn_msa\run.ps1 -DataRoot $DataRoot -Preset pilot -RunName pilot_seed42 -Reference $Reference -V1Root $V1Root -Resume
```

Aggiungere `-FetchV1` solo per completare eventuali tile mancanti. Omettere `-DownloadTargets` quando i target/reference sono già verificati. Un errore ferma la sequenza; nessuna fase successiva viene avviata. L'uso di `-Resume` è consentito anche al primo avvio e non modifica dati già validi.

### Monitoraggio live — terminale 2

Aprire una seconda PowerShell, ridefinendo percorsi/variabili:

```powershell
Set-Location 'C:\BioMAP\biomap-v1-spatial-cnn-msa'
$Py = (Resolve-Path '.\experiments\v1_spatial_cnn_msa\.venv\Scripts\python.exe').Path
$DataRoot = [System.IO.Path]::GetFullPath('..\data')
& $Py experiments\v1_spatial_cnn_msa\monitor.py --data-root $DataRoot --follow
```

Mostra fase, tile/file/byte, celle o job completati, epoca/batch, tempo trascorso, ETA stimato, VRAM rilevata ed errori. Continua fra le fasi dello script e conclude con lo stato della sequenza. Ctrl+C chiude soltanto il monitor.

L'ETA non è una scadenza garantita: nei download le dimensioni dei file differiscono; nel training early stopping e costo dei modelli cambiano il tempo. Il budget residuo della fase usa il massimo delle epoche; l'ETA globale si basa sui job osservati. All'inizio può restare in stima. Un heartbeat vecchio viene segnalato e non è presentato come processo ancora attivo.

## 6. Comandi separati e personalizzazione

Permettono di ispezionare gli output tra una fase e la successiva:

```powershell
& $Py experiments\v1_spatial_cnn_msa\patch_builder.py build --data-root $DataRoot --preset pilot
& $Py experiments\v1_spatial_cnn_msa\preflight.py --data-root $DataRoot --preset pilot --require-training
& $Py experiments\v1_spatial_cnn_msa\train.py --data-root $DataRoot --preset pilot --profile
& $Py experiments\v1_spatial_cnn_msa\train.py --data-root $DataRoot --preset pilot --run-name pilot_seed42 --phase cv --resume
& $Py experiments\v1_spatial_cnn_msa\train.py --data-root $DataRoot --preset pilot --run-name pilot_seed42 --phase refit --resume
& $Py experiments\v1_spatial_cnn_msa\evaluate.py oof --data-root $DataRoot --preset pilot --run-name pilot_seed42
& $Py experiments\v1_spatial_cnn_msa\evaluate.py liguria --data-root $DataRoot --preset pilot --run-name pilot_seed42
& $Py experiments\v1_spatial_cnn_msa\evaluate.py compare --data-root $DataRoot --preset pilot --run-name pilot_seed42
```

Se usati, aggiungere **gli stessi** `--reference`, `--v1-root` o `--targets-csv` a ogni comando. L'identità impedisce di mescolare configurazioni. Le metriche esterne vengono calcolate solo nell'azione `liguria`, dopo CV e refit. La selezione rimane sullo sviluppo.

I parametri sono nel `config.yaml` locale oppure `--set`. Esempio di nuovo candidato, da scegliere attraverso la CV e non dal test:

```powershell
& $Py experiments\v1_spatial_cnn_msa\train.py --data-root $DataRoot --preset pilot --run-name pilot_sigmoid --set model.output=sigmoid --phase all
& $Py experiments\v1_spatial_cnn_msa\evaluate.py liguria --data-root $DataRoot --preset pilot --run-name pilot_sigmoid --set model.output=sigmoid
```

Non confrontare ripetutamente Liguria per decidere l'attivazione. Il default è lineare, senza clipping; vengono riportate le predizioni fuori [0,1]. `pooling=max` è una variante esplicita con nuovo run-name. Cambiare geometria, soglia di copertura, target o patch richiede una nuova cache/work-root; non sovrascrivere quella storica. `--work-root` deve terminare con `v1_spatial_msa`, per esempio `D:\nuovo_protocollo\v1_spatial_msa`.

Se il profiling non sostiene il batch configurato, il training viene fermato. Ridurre `training.batch_cells` al valore misurato in tutti i comandi e usare un nuovo run-name coerente. Il codice non riduce pixel o patch per risolvere un OOM.

## 7. Intermedio e full, solo dopo il pilot

L'intermedio ha 8.500 sviluppo + 1.500 Liguria. Il full ha **127.909 sviluppo + 4.991 Liguria**. Nessun run full è stato avviato durante la preparazione del codice.

```powershell
# Intermedio: avvio esplicito dopo aver verificato i risultati tecnici del pilot.
powershell -NoProfile -ExecutionPolicy Bypass -File .\experiments\v1_spatial_cnn_msa\run.ps1 -DataRoot $DataRoot -Preset intermediate -RunName intermediate_seed42 -DownloadTargets -FetchV1 -Resume

# Full: comando da eseguire solo dopo decisione esplicita su dati, pilot, tempo e spazio.
powershell -NoProfile -ExecutionPolicy Bypass -File .\experiments\v1_spatial_cnn_msa\run.ps1 -DataRoot $DataRoot -Preset full -RunName full_seed42 -DownloadTargets -FetchV1 -Resume -FullApproved
```

La stessa finestra monitor mostra il nuovo preset. I download dei tre preset condividono soltanto le sorgenti V1 private; cache e run rimangono separati. Con sorgenti esterne ripetere i relativi parametri come nel pilot.

Per separare preparazione e training full, eseguire esplicitamente le singole azioni seguenti dopo la decisione sul full. I comandi Python singoli sono già avvii espliciti: il flag `-FullApproved` appartiene soltanto all'orchestratore PowerShell.

```powershell
& $Py experiments\v1_spatial_cnn_msa\patch_builder.py targets --data-root $DataRoot --preset full
& $Py experiments\v1_spatial_cnn_msa\patch_builder.py fetch --data-root $DataRoot --preset full
& $Py experiments\v1_spatial_cnn_msa\patch_builder.py build --data-root $DataRoot --preset full
& $Py experiments\v1_spatial_cnn_msa\preflight.py --data-root $DataRoot --preset full --require-training
& $Py experiments\v1_spatial_cnn_msa\train.py --data-root $DataRoot --preset full --profile
& $Py experiments\v1_spatial_cnn_msa\train.py --data-root $DataRoot --preset full --run-name full_seed42 --phase cv --resume
& $Py experiments\v1_spatial_cnn_msa\train.py --data-root $DataRoot --preset full --run-name full_seed42 --phase refit --resume
& $Py experiments\v1_spatial_cnn_msa\evaluate.py oof --data-root $DataRoot --preset full --run-name full_seed42
& $Py experiments\v1_spatial_cnn_msa\evaluate.py liguria --data-root $DataRoot --preset full --run-name full_seed42
& $Py experiments\v1_spatial_cnn_msa\evaluate.py compare --data-root $DataRoot --preset full --run-name full_seed42
```

## 8. Output, validità e ripresa

```text
data/v1_spatial_msa/
  sources/                    # V1.0 scaricato; overlay separato dalle sorgenti esterne
  targets/                    # GLOBIO, puntatore e acquisizioni preservate
  cache/pilot/                # spatial.sqlite3 + manifest SHA, coverage/cache_report
  cache/intermediate/
  cache/full/
  preflight/ estimates/
  profile_training.json
  status.json pipeline.json   # monitor, distinti dai risultati scientifici
  runs/<preset>/<run-name>/
    run.json selection.json refit_complete.json
    <modello>/fold_0..2/      # last/best, epoch log, OOF e checksum
    <modello>/refit/
    oof_* liguria_*           # CSV, metriche pooled/macro/AOI, distribuzioni e PNG
```

Un'interruzione nel build riparte dalla tile non committata. Training: ultimo stato optimizer/scaler/RNG; best separato, sorgenti e configurazione verificate. Ripetere il comando con `-Resume`/`--resume`. I file `experiment.lock`/`workflow.lock` possono restare sul disco: conta il lock OS, non l'esistenza del nome.

Di default una cella non eleggibile blocca la pubblicazione della cache completa. `data.allow_partial_cells=true` è un'autorizzazione esplicita a un sottoinsieme dichiarato: tutte le teste e i baseline usano gli stessi ID, e non si presenta il risultato come full equivalente. La presenza di tutti i fold e delle 17 AOI rimane un requisito di training.

I tempi e la VRAM del PC universitario vanno misurati con il profiling. I risultati sintetici non dimostrano un miglioramento MSA. I numeri storici Liguria sono descrittivi finché il supporto non è dimostrato identico. GLOBIO è un target modellato, non una misura indipendente di biodiversità sul campo.
