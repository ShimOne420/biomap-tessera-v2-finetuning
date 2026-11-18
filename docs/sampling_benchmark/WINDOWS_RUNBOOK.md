# Runbook Windows — benchmark sampling frozen

Usare un clone nuovo. Non modificare `C:\BioMAP\spatial-code`, l'ambiente del run storico, l'indice o gli output esistenti. Questo documento prepara comandi per l'agente PC: non autorizza automaticamente screening o campagna reale. Ambiente PowerShell5.1/Python3.12, niente bash/attivazione venv. Attendere risorse disponibili; non avviare nuovi lavori pesanti in parallelo al run universitario.

## 1. Clone, ambiente separato e test CPU

```powershell
$ErrorActionPreference = 'Stop'
$env:OMP_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:CUBLAS_WORKSPACE_CONFIG = ':4096:8'
if (Test-Path 'C:\BioMAP\sampling-code') { throw 'Clone gia presente: non sovrascrivere o fare checkout sotto un run' }
git -c core.autocrlf=false clone --branch feature/sampling-benchmark https://github.com/ShimOne420/biomap-tessera-v2-finetuning.git 'C:\BioMAP\sampling-code'
if ($LASTEXITCODE -ne 0) { throw 'Clone fallito' }
Set-Location 'C:\BioMAP\sampling-code'
git merge-base --is-ancestor 07dca6b HEAD
if ($LASTEXITCODE -ne 0) { throw 'Base errata' }
git status --short
git log -1 --oneline
py -3.12 -m venv .venv-sampling
if ($LASTEXITCODE -ne 0) { throw 'Creazione ambiente fallita' }
$SamplingPython = (Resolve-Path '.\.venv-sampling\Scripts\python.exe').Path
& $SamplingPython -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128
if ($LASTEXITCODE -ne 0) { throw 'Torch non installato' }
& $SamplingPython -m pip install -r requirements-lock.txt
if ($LASTEXITCODE -ne 0) { throw 'Dipendenze non installate' }
& $SamplingPython -m pip install -e . --no-deps --no-build-isolation
if ($LASTEXITCODE -ne 0) { throw 'Progetto non installato' }
& $SamplingPython -m pip check
if ($LASTEXITCODE -ne 0) { throw 'Dipendenze incoerenti' }
& $SamplingPython scripts\check_dev_environment.py
if ($LASTEXITCODE -ne 0) { throw 'Check ambiente fallito' }
& $SamplingPython -m pytest -ra
if ($LASTEXITCODE -ne 0) { throw 'Test CPU falliti' }
function Invoke-Sampling {
    & $SamplingPython 'scripts\sampling_benchmark.py' @args
    if ($LASTEXITCODE -ne 0) { throw 'Comando fallito: leggere errore e non proseguire automaticamente' }
}
```

Il clone mantiene LF sui moduli senza pin e rispetta comunque le EOL esplicite dei cinque CODE_FILES. Doctor verifica l'identità dell'indice; se non coincide, fermarsi e confrontare con il checkout storico. Non ricostruire l'indice e non riscrivere file protetti per aggirare il problema.

## 2. Percorsi reali e dry-run

```powershell
$Protected = 'paths.read_only_roots=["C:/BioMAP/spatial-code","C:/BioMAP/spatial_runs","C:/BioMAP/v2_spatial_pixels_r2","C:/Users/brome/Desktop/BIOMAP_TESSERA_V2/data","F:/biomap_dense_public"]'
$Common = @('--config','configs\spatial_sampling_benchmark.yaml','--pixel-root','C:\BioMAP\v2_spatial_pixels_r2','--reference','F:\biomap_dense_public\reference','--index-root','C:\BioMAP\spatial_runs\index','--historical-run','C:\BioMAP\spatial_runs\v2_spatial3_seed42','--cache-root','C:\BioMAP\sampling_cache_r1','--output-root','F:\biomap_sampling_runs\benchmark_r1','--device','cuda:0','--set',$Protected)
Invoke-Sampling doctor @Common --dry-run
Invoke-Sampling doctor @Common
```

Doctor è read-only e può richiedere minuti per verificare SHA di48GB. Verifica reader/indice, popolazione sviluppo e fonte canonica del riferimento0,17602, se disponibile. Le metriche storiche non sono la soglia del nuovo nested. Le vecchie serie Sentinel25punti non vengono lette o convertite. CUDA disponibile non significa che lo smoke o il benchmark siano già validati.

Il piano distingue costi stimati e misurati.12ore vale solo come budget screening. Il nested può richiedere48–96fit interni, più refit/valutazioni: non promettere il completamento in12ore. I costi di cache/hash/valutazione esterna ancora non misurati sono null, non zero. Controllare spazio separatamente su C e F.

## 3. Smoke reale, soltanto quando autorizzato

```powershell
$Smoke = $Common + @('--cache-root','C:\BioMAP\sampling_smoke_cache','--output-root','F:\biomap_sampling_runs\smoke_r1')
Invoke-Sampling smoke @Smoke --dry-run
Invoke-Sampling smoke @Smoke
```

Otto celle di una sola AOI di sviluppo, una epoca tecnica S64, nessuna selezione geografica o scientifica. Budget totale900s, inclusi lettura/verifica/cache/fit/valutazione. Se non termina, è uno smoke non completato: i validi restano e si può riprendere con `Invoke-Sampling smoke @Smoke --resume`. Non si riducono dati/date per ottenere un successo. Output: cache dedicata, `technical_predictions.parquet`, `smoke_result.json` e `progress.json`. Le metriche smoke non misurano generalizzazione.

## 4. Preparazione geometrica e cache, dopo autorizzazione del benchmark

```powershell
Invoke-Sampling prepare @Common
Invoke-Sampling cache @Common
```

Output C: piano firmato, shard geometrici e vettori float32 condivisi, SQLite nuovo, momenti per configurazione. Nessuna copia dell'intera campagna FULL e nessuna scrittura nell'indice originale. Durata reale da misurare sul PC: una passata di estrazione più serializzazione, non un tempo garantito. Resume: aggiungere `--resume` allo stesso comando. La cache completata viene verificata e riusata, non sovrascritta.

Il manifest registra anche celle insufficienti e RF reale. FULL-SUBSET deve contenere circa11.000celle globali; i fit usano solo il sottoinsieme appartenente al loro training. Le altre configurazioni mantengono127.909celle di sviluppo.

## 5. Screening interno, dopo autorizzazione

```powershell
Invoke-Sampling screen @Common --dry-run
Invoke-Sampling screen @Common
```

Sette fit: FULL42/FULL43, quattro candidati42 e FULL-SUBSET42. Outer0 non sceglie nulla: training interno gruppo1, validation interna gruppo2. Il margine completo nested richiederà anche la suddivisione interna inversa.

Budget12h con arresto controllato e resume: `Invoke-Sampling screen @Common --resume`. Un esito `BUDGET_EXHAUSTED_NOT_COMPLETE` non significa che tutti i fit siano conclusi. Checkpoint/RNG validi sono conservati; l'epoca incompleta viene ricalcolata. Tempi per epoca, picco GPU e RAM di processo sono nei fit. Lo scratch validation può essere sostituito solo all'interno del nuovo workspace posseduto.

## 6. Stima completa e approvazione separata del nested

```powershell
Invoke-Sampling run @Common --dry-run
$ApprovedPlan = Read-Host 'Inserisci il budget_id soltanto dopo approvazione separata del costo della campagna nested'
```

Non incollare automaticamente il budget_id in un comando per interpretarlo come consenso. Prima verificare fit minimi/massimi, nuove misure del pilot, dischi, tempo disponibile e limiti degli errori. Senza approvazione fermarsi qui. I blocchi successivi sono istruzioni per la futura esecuzione autorizzata.

```powershell
Invoke-Sampling run @Common --approved-plan $ApprovedPlan --phase controls --resume
Invoke-Sampling run @Common --approved-plan $ApprovedPlan --phase candidates --resume
Invoke-Sampling run @Common --approved-plan $ApprovedPlan --phase confirm --resume
Invoke-Sampling run @Common --approved-plan $ApprovedPlan --phase recommend --resume
Invoke-Sampling run @Common --approved-plan $ApprovedPlan --phase outer --resume
Invoke-Sampling report @Common
```

Questi comandi operano sui tre outer; `--outer-fold 0` limita una fase al primo. Controlli FULL prima dei candidati; decisioni sigillate prima delle metriche esterne. Refit: nuova CNN, stesso seed, mediana di due migliori epoche con mezzi arrotondati verso l'alto. FULL e candidati usano identica regola.

Se la raccomandazione globale richiede conferme aggiuntive interne, vengono eseguite nella fase `recommend`. FULL-SUBSET rimane separato dalla selezione. Non viene generata alcuna acquisizione o avviato fine-tuning. In assenza di candidati sufficientemente vicini, l'esito può essere “nessun campionamento ammissibile”.

## 7. Monitor da un secondo terminale

```powershell
Set-Location 'C:\BioMAP\sampling-code'
& 'C:\BioMAP\sampling-code\.venv-sampling\Scripts\python.exe' scripts\sampling_benchmark.py monitor --progress 'F:\biomap_sampling_runs\benchmark_r1\progress.json' --interval 5
```

Per la preparazione/cache monitorare `C:\BioMAP\sampling_cache_r1\progress.json`. Il monitor apre/chiude il file subito; Ctrl+C ferma solo il monitor. Campi: fase, fold, epoca, fatto/totale, velocità, ETA corrente. ETA null significa che manca una misura utilizzabile; l'ETA di epoca non comprende automaticamente tutta la campagna.

## 8. Verifica degli output e prossime decisioni

Leggere `report.md`, `metrics_summary.csv`, predizioni interne/esterne, decisioni per outer, `global_margin.json`, `recommendation.json`, budget e risorse. R², code di errore e AOI problematiche vanno valutati insieme al delta RMSE. Il margine da due seed è esplorativo, non una prova di non inferiorità.

Il report deve distinguere la popolazione FULL-SUBSET, il riferimento storico ottimista e il nuovo FULL nested. La valutazione esterna dei vincitori riguarda la procedura di selezione, non una configurazione globale scelta a posteriori.

Prima di qualsiasi acquisizione futura verificare il manifest esatto, lo spazio corrente per unità, il budget dei checkpoint e l'approvazione dell'utente. I raw stimati M128/S144 non entrano nei dischi dichiarati con la riserva attuale; compressione non misurata non è una soluzione dimostrata. Non usare le vecchie serie25punti e non riscaricare input a ogni epoca. Un blocco encoder è soltanto la raccomandazione iniziale per il successivo pilot, non un run autorizzato.
