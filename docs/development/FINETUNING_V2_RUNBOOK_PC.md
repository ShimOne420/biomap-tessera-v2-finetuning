# Runbook PC — GeoTESSERA V2 Medium partial fine-tuning

Branch da scaricare: **`feature/finetuning-v2`**, dipendente da `fix/spatial-v2-real-parquet-schema` (`1737b1f`). Non fare pull, checkout o installazioni in `C:\BioMAP\spatial-code` né nella sua venv. **Aspettare la conclusione del run Spatial CNN prima dei probe GPU e del pilot.** Questo documento non dimostra che quel run sia terminato.

PowerShell **5.1**. I percorsi sotto sono quelli del handoff del 17/11/2026; le nuove cartelle `finetuning-code`, `biomap_ft_*` sono destinazioni separate. Nessun dataset o peso viene scaricato durante setup/doctor. Le decisioni richieste con `Read-Host` sono deliberate: non ci sono valori scientifici definitivi inventati o segnaposto da sostituire nel codice.

## 1. Clone nuovo e ambiente indipendente

Prerequisiti: Git e Python 3.12 disponibili. Comandi dopo la fine del run per le operazioni che usano risorse. `C:\BioMAP\finetuning-code` deve essere nuovo.

```powershell
$ErrorActionPreference = 'Stop'
New-Item -ItemType Directory -Force 'C:\BioMAP' | Out-Null
if (Test-Path 'C:\BioMAP\finetuning-code') { throw 'Clone gia presente: non sovrascriverlo; vedere aggiornamento sicuro in fondo' }
git clone --branch feature/finetuning-v2 https://github.com/ShimOne420/biomap-tessera-v2-finetuning.git 'C:\BioMAP\finetuning-code'
if ($LASTEXITCODE -ne 0) { throw 'Clone fallito' }
Set-Location 'C:\BioMAP\finetuning-code'
git merge-base --is-ancestor 1737b1f HEAD
if ($LASTEXITCODE -ne 0) { throw 'Base errata: non proseguire' }
git status --short
git log -1 --oneline
py -3.12 -m venv .venv-finetune
if ($LASTEXITCODE -ne 0) { throw 'Python/venv non disponibili' }
$FtPython = (Resolve-Path '.\.venv-finetune\Scripts\python.exe').Path
& $FtPython -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128
if ($LASTEXITCODE -ne 0) { throw 'Torch non installato' }
& $FtPython -m pip install -r requirements-lock.txt
if ($LASTEXITCODE -ne 0) { throw 'Dipendenze non installate' }
& $FtPython -m pip install -e . --no-deps --no-build-isolation
if ($LASTEXITCODE -ne 0) { throw 'Progetto non installato' }
& $FtPython -m pip check
if ($LASTEXITCODE -ne 0) { throw 'Dipendenze incoerenti' }
& $FtPython -m pytest -ra
if ($LASTEXITCODE -ne 0) { throw 'Test CPU falliti: non proseguire' }
```

Il setup installa pacchetti nella nuova venv; non modifica il lock e non riusa l'ambiente del run. Non avviare download di Sentinel qui. Non riscaricare il checkpoint: il file esistente è obbligatorio e viene verificato tramite SHA256.

Definire una funzione fail-fast per tutti i comandi seguenti, nello stesso terminale:

```powershell
function Invoke-FT {
    & $FtPython 'scripts\finetune_v2.py' @args
    if ($LASTEXITCODE -ne 0) { throw 'Comando fine-tuning fallito: leggere errore e progress, non proseguire automaticamente' }
}
$Checkpoint = 'C:\Users\brome\Desktop\BIOMAP_TESSERA_V2\data\weights\student_medium.pt'
$Protected = 'paths.read_only_roots=["C:/BioMAP/spatial-code","C:/BioMAP/spatial_runs","C:/BioMAP/v2_spatial_pixels_r2","F:/biomap_dense_public"]'
$Shared = @('--reference','F:\biomap_dense_public\reference','--catalog-root','F:\biomap_dense_public\work\api','--frozen-pixels','C:\BioMAP\v2_spatial_pixels_r2','--checkpoint',$Checkpoint,'--device','cuda:0','--set',$Protected,'--set','paths.frozen_work=F:/biomap_dense_public/work','--set','paths.frozen_spatial_run=C:/BioMAP/spatial_runs/v2_spatial3_seed42')
$SmokePaths = @('--raw-root','F:\biomap_ft_smoke_inputs','--output-root','F:\biomap_ft_smoke_runs')
$MainPaths = @('--raw-root','F:\biomap_ft_inputs','--output-root','F:\biomap_ft_runs')
New-Item -ItemType Directory -Force 'C:\BioMAP\finetuning-logs' | Out-Null
```

Le root di scrittura sono rifiutate se coincidono/si annidano con quelle protette o fra loro. Se F: non ha spazio, scegliere una destinazione **nuova** su un disco adeguato modificando soltanto i due array dei nuovi percorsi; non cancellare reference, r1, mappe r2 o output Spatial per far posto. Lo spazio C/F riportato nel handoff non è una misura corrente.

## 2. Doctor comune, prima della scelta dell'aggregazione

Prerequisiti: file checkpoint, reference e GPU installati. Il primo comando fa inventario; il secondo aggiunge un piccolo forward/backward sintetico del checkpoint reale, non usa Sentinel.

```powershell
nvidia-smi
Invoke-FT doctor --config configs\v2_partial_finetuning.yaml @Shared @MainPaths
Invoke-FT doctor --config configs\v2_partial_finetuning.yaml @Shared @MainPaths --probe | Tee-Object -FilePath 'C:\BioMAP\finetuning-logs\doctor.txt'
```

L'esito atteso prima dei dati è `NOT_VERIFIED_FOR_REAL_TRAINING`, con sampling/aggregazione/LR ancora da decidere. `TECHNICAL_PROBE_PASSED` riguarda solo due pixel sintetici × otto token; non misura il batch reale. Se SHA, strict load, backward, driver o spazio falliscono, fermarsi. Doctor non cambia i pesi sul disco.

## 3. Scelta esplicita e smoke reale, budget massimo 15 minuti

Leggere prima [la guida alla decisione](../finetuning/TOMORROW_DECISION_GUIDE.md). Il run attuale Frozen+Spatial CNN deve essere analizzato; questo documento non ne anticipa l'esito. Scegliere **una sola** aggregazione, anche per la prova tecnica:

```powershell
$Aggregation = Read-Host 'Aggregazione da provare: dense_mean oppure spatial_cnn'
if ($Aggregation -notin @('dense_mean','spatial_cnn')) { throw 'Scelta esplicita richiesta' }
$SmokeArgs = @('--config','configs\v2_partial_smoke.yaml') + $Shared + $SmokePaths + @('--aggregation',$Aggregation)
Invoke-FT doctor @SmokeArgs --dry-run | Tee-Object -FilePath 'C:\BioMAP\finetuning-logs\smoke-plan.txt'
Invoke-FT smoke @SmokeArgs --run-id medium_partial_smoke --limit-seconds 900 | Tee-Object -FilePath 'C:\BioMAP\finetuning-logs\smoke.txt'
```

**Questo è il primo comando che acquisisce Sentinel.** Usa tutte le date annuali delle poche posizioni scelte; 4 celle della AOI `abruzzo_gran_sasso`, finestre native 2×2, una epoca, batch una cella e LR solo tecnici. Il limite temporale comprende acquisizione, confronto con pixel frozen, backward/step, predizioni e metriche. Se la rete non conclude entro 900 secondi, il worker viene fermato e l'esito è fallimento tecnico, non “smoke passato”. I giorni/blocchi validi sono conservati. Non viene diminuito il numero di date per rispettare il limite.

Output nuovi: `F:\biomap_ft_smoke_inputs\dataset.json`, `input_validation.json`, `storage_admission.json`, blocchi/giorni e progress; `F:\biomap_ft_smoke_runs\medium_partial_smoke\pilot`, predizioni e metriche `technical_*`. Una bassa MSE sullo smoke non misura generalizzazione. Per una configurazione tecnica diversa usare una nuova raw-root/run-id: non riutilizzare una root incompatibile.

## 4. Monitor live da un secondo terminale

Non serve attivare la venv e non si apre un file in scrittura. Durante acquisizione:

```powershell
Set-Location 'C:\BioMAP\finetuning-code'
& 'C:\BioMAP\finetuning-code\.venv-finetune\Scripts\python.exe' scripts\finetune_v2.py monitor --progress 'F:\biomap_ft_smoke_inputs\progress.json' --interval 5
```

Durante il training dello smoke, in un'altra finestra oppure dopo Ctrl+C sul solo monitor:

```powershell
& 'C:\BioMAP\finetuning-code\.venv-finetune\Scripts\python.exe' scripts\finetune_v2.py monitor --progress 'F:\biomap_ft_smoke_runs\medium_partial_smoke\progress.json' --interval 5
```

Leggere `phase`, `done/total`, `rate_per_second`, `eta_seconds`, `estimated_finish_utc`, eventuale `error`. Un ETA null significa che manca ancora una velocità misurabile, non che il processo sia finito. Il training stima i passaggi ancora pianificati nella fase/fold corrente, non tutta la CV futura. Il monitor non controlla il run Spatial esistente. Interrompere il monitor non interrompe il training.

## 5. Piano completo e decisioni dell'utente

Il benchmark non eredita i valori tecnici dello smoke. Il piano comune iniziale rimane bloccato finché non si sceglie il sampling:

```powershell
Invoke-FT doctor --config configs\v2_partial_finetuning.yaml @Shared @MainPaths --aggregation $Aggregation --dry-run
```

Dopo aver esaminato tempi, VRAM e spazio del pilot, inserire le scelte effettive. Le domande non propongono parametri scientifici obbligati. Per i learning rate usare decimali con punto (per esempio `0.00001` solo se è il valore approvato), non una virgola locale.

```powershell
$Sampling = Read-Host 'Supporto approvato: full_support, uniform_pixels oppure center_windows'
if ($Sampling -notin @('full_support','uniform_pixels','center_windows')) { throw 'Sampling esplicito richiesto' }
$Decisions = @('--aggregation',$Aggregation,'--set',('sampling.method=' + $Sampling),'--set','sampling.approved=true')
if ($Sampling -eq 'uniform_pixels') {
    $PixelCount = [int](Read-Host 'Numero di contributi pixel per cella approvato')
    if ($PixelCount -lt 1) { throw 'K deve essere positivo' }
    $Decisions += @('--set',('sampling.pixels_per_cell=' + $PixelCount))
}
if ($Sampling -eq 'center_windows') {
    $WindowSide = [int](Read-Host 'Lato della finestra per cella e tile approvato')
    if ($WindowSide -lt 1) { throw 'Lato deve essere positivo' }
    $Decisions += @('--set',('sampling.window_side=' + $WindowSide))
}
$LastBlocks = [int](Read-Host 'Ultimi blocchi per modalita: 1, 2 o 3; preferenza attuale 2, da verificare sul pilot')
if ($LastBlocks -notin @(1,2,3)) { throw 'Numero blocchi non valido' }
$EncoderLR = Read-Host 'Learning rate encoder approvato, decimale con punto'
$HeadLR = Read-Host 'Learning rate testa approvato, decimale con punto'
$CellBatch = [int](Read-Host 'Batch celle approvato dopo misura GPU')
$Epochs = [int](Read-Host 'Massimo epoche approvato, considerando anche archivio checkpoint')
$Decisions += @('--set',('finetuning.last_n_blocks=' + $LastBlocks),'--set',('training.learning_rate_encoder=' + $EncoderLR),'--set',('training.learning_rate_head=' + $HeadLR),'--set',('training.batch_size=' + $CellBatch),'--set',('training.epochs=' + $Epochs))
$MainArgs = @('--config','configs\v2_partial_finetuning.yaml') + $Shared + $MainPaths + $Decisions
Invoke-FT doctor @MainArgs --dry-run | Tee-Object -FilePath 'C:\BioMAP\finetuning-logs\full-plan.txt'
```

Verificare il numero di celle **132.900 totali**, AOI **18 incluse Liguria**, pixel selezionati, tile, budget raw giornaliero/finale, archivio checkpoint, riserva minima 20 GiB. Il numero di AOI di sviluppo resta 17. Il piano temporale è ipotetico finché manca una misura; il limite a 100 Mbit/s è un lower bound payload, non un ETA. Il supporto integrale e la cache intermedia completa non entrano nei dischi attualmente dichiarati. Il dry-run non autorizza automaticamente acquisizione o training; eseguire i prossimi blocchi solo dopo approvazione delle scelte e capacità sufficiente.

## 6. Acquisizione completa sul supporto approvato e verifica

Prerequisiti: smoke riuscito, decisioni approvate, spazio per raw + giorni e successivi checkpoint, nessun altro run pesante. Download diretto nella nuova root F:, non sull'SSD storico. I cataloghi r1 vengono solo letti/cop­iati. L'acquisizione può essere molto lunga anche su pochi pixel per cella.

```powershell
Invoke-FT acquire @MainArgs | Tee-Object -FilePath 'C:\BioMAP\finetuning-logs\acquire.txt'
Invoke-FT doctor @MainArgs --verify-inputs | Tee-Object -FilePath 'C:\BioMAP\finetuning-logs\input-parity.txt'
```

Un errore di confronto non va aggirato allargando tolleranze o rimuovendo celle. `storage_admission.json` usa tutte le date effettive dei cataloghi e può rifiutare il piano **prima delle letture raster**. Conservare il report dell'errore per l'agente. I file storici non vengono alterati.

Nel terminale separato, per monitorare questa acquisizione:

```powershell
& 'C:\BioMAP\finetuning-code\.venv-finetune\Scripts\python.exe' scripts\finetune_v2.py monitor --progress 'F:\biomap_ft_inputs\progress.json' --interval 10
```

## 7. Controllo frozen, partial, refit e valutazione

Prerequisiti: raw completi, prova input valida, risorse verificate e autorizzazione ad avviare il training scientifico. Eseguire **una sola aggregazione**. Il controllo qui è nuovo: stessa testa, seed, supporto, preprocessing e protocollo del partial. Non sostituisce automaticamente il confronto con i vecchi run.

```powershell
$FrozenArgs = $MainArgs + @('--set','training.encoder_mode=frozen')
Invoke-FT train @FrozenArgs --run-id medium_frozen_control --phase cv
Invoke-FT train @MainArgs --run-id medium_partial --phase cv
```

Esaminare `oof_metrics.json`, `oof_by_aoi.csv`, `selection.json`, aggiornamenti dei pesi e risorse **prima** del test esterno. Il modello si seleziona sul macro-AOI RMSE dei tre fold, non sulla Liguria. Refit per il numero di epoche derivato dalla CV, poi test già osservato:

```powershell
Invoke-FT train @FrozenArgs --run-id medium_frozen_control --phase refit --resume
Invoke-FT train @MainArgs --run-id medium_partial --phase refit --resume
Invoke-FT evaluate @FrozenArgs --run-id medium_frozen_control
Invoke-FT evaluate @MainArgs --run-id medium_partial
Invoke-FT report @MainArgs --run-id medium_partial --compare-run medium_frozen_control
```

Output sotto `F:\biomap_ft_runs\medium_partial` e `medium_frozen_control`: manifest/config effettiva, fold/refit con checkpoint per epoca e marker, history/risorse, OOF e Liguria parquet, metriche/diagnostiche/Dummy, `paired_comparison.json`, `report.md`. Il report non inventa un confronto appaiato con i vecchi CSV quando il supporto differisce. L'archivio immutabile richiede spazio: nessun checkpoint viene cancellato automaticamente.

Monitor del partial da terminale separato:

```powershell
& 'C:\BioMAP\finetuning-code\.venv-finetune\Scripts\python.exe' scripts\finetune_v2.py monitor --progress 'F:\biomap_ft_runs\medium_partial\progress.json' --interval 10
```

## 8. Interruzione e ripresa

Non cambiare codice, sampling, aggregazione, seed o configurazione per riprendere lo stesso run. I manifest rifiutano identità differenti. I checkpoint registrano RNG/optimizer/scaler e normalizzazione. La ripresa è dal confine di epoca: i minibatch dell'epoca interrotta vengono ricalcolati; gli stati già validi non vengono persi. Durante acquisizione vengono riusati i giorni/blocchi con marker/SHA validi.

```powershell
Invoke-FT smoke @SmokeArgs --run-id medium_partial_smoke --limit-seconds 900 --resume
```

Oppure, per le rispettive fasi complete, eseguire **solo il comando che era stato interrotto**, non tutti come nuova campagna:

```powershell
Invoke-FT acquire @MainArgs --resume
Invoke-FT train @MainArgs --run-id medium_partial --phase cv --resume
Invoke-FT train @MainArgs --run-id medium_partial --phase refit --resume
```

Per un futuro aggiornamento del solo clone di sviluppo, prima fermare i suoi processi e preservare i risultati; non aggiornare il clone del run Spatial. Verificare working tree pulito, poi fast-forward:

```powershell
Set-Location 'C:\BioMAP\finetuning-code'
if (git status --porcelain) { throw 'Modifiche locali: conservarle prima del pull; nessun reset/clean' }
git pull --ff-only origin feature/finetuning-v2
if ($LASTEXITCODE -ne 0) { throw 'Aggiornamento non riuscito: non forzare' }
```

Una modifica del codice può invalidare l'identità del nuovo esperimento; non promettere ripresa scientifica dopo un pull. La prova input include il codice e va riverificata quando necessario. Gli hash e gli indici delle vecchie pipeline restano preservati dal branch.
