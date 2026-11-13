<#
=====================================================================================
 run_e2.ps1 — Esecuzione completa dell'esperimento E2 (TESSERA v2 Medium congelato + CNN 5x5)
=====================================================================================

 COSA FA
   Esegue in sequenza, nel terminale corrente (output visibile in tempo reale), tutti i
   passaggi E2 descritti in PIANO_OPERATIVO.md, sezioni 5-8:
     1. check_environment  -> verifica device CUDA e percorsi (environment.json)
     2. prepare --stage index -> audit in sola lettura di celle, target, serie, fold
                                 (dataset\index.csv, dataset\audit.json)
     3. prepare --stage cache -> embedding puntuali Medium frozen in SQLite (1-6 h, una tantum;
                                 un rilancio riusa le celle già registrate)
     4. profile_training      -> breve profilo della CNN su cache reale (profile_E2.json)
     5. train --phase all     -> 3 fold di CV geografica + refit sulle 127.909 celle
     6. evaluate              -> benchmark Liguria Levante (una sola volta), metriche, grafici, Excel
     7. predict               -> GeoTIFF COG sulla griglia GLOBIO
   Ogni passo si ferma al primo errore (exit code != 0) indicando quale passo è fallito.

 DATI E PERCORSI
   Sorgente (SOLO LETTURA): ..\..\data\msa_v1_liguria  (cells.csv, series\, targets\)
   Checkpoint: risolto dalla CLI come <sorgente>\..\weights\student_medium.pt
               (= BIOMAP_TESSERA_V2\data\weights\student_medium.pt, Medium originale).
   Work-root (scrittura): <checkout>\data_dev\E2  (ignorata da Git). Ogni output E2 va qui;
   la sorgente del probing non viene modificata.
   Log completo: data_dev\E2\logs\run_e2_<timestamp>.log (Start-Transcript).

 COME RICHIAMARLO (dal terminale VS Code, nella root del checkout)
   powershell -ExecutionPolicy Bypass -File .\scripts\run_e2.ps1
   Dopo un'interruzione del training:
   powershell -ExecutionPolicy Bypass -File .\scripts\run_e2.ps1 -Resume
     -> i passi index/cache sono idempotenti (riusano quanto già fatto); il training riparte
        dall'ultima epoca completata con --resume.
   Parametri opzionali: -RunId (default e2_msa_seed42), -Device (default cuda:0),
                        -SkipProfile per saltare il profilo.
=====================================================================================
#>
param(
    [string]$RunId = 'e2_msa_seed42',
    [string]$Device = 'cuda:0',
    [switch]$Resume,
    [switch]$SkipProfile
)
$ErrorActionPreference = 'Stop'
$repo = Split-Path $PSScriptRoot -Parent
Set-Location $repo

$taskPython = (Resolve-Path '.\.venv\Scripts\python.exe').Path
$sourceData = (Resolve-Path (Join-Path $repo '..\data\msa_v1_liguria')).Path
$workRoot = Join-Path $repo 'data_dev\E2'
New-Item -ItemType Directory -Force (Join-Path $workRoot 'logs') | Out-Null
$e2Args = @('--config', 'configs\e2_msa.yaml', '--source-data-root', $sourceData, '--work-root', $workRoot)

$log = Join-Path $workRoot ("logs\run_e2_{0}.log" -f (Get-Date -Format 'yyyyMMdd_HHmmss'))
Start-Transcript -Path $log | Out-Null

# Esegue un passo, misura la durata e interrompe tutto se il comando Python fallisce.
function Invoke-Step([string]$Name, [string[]]$StepArgs) {
    Write-Host "`n==== [$(Get-Date -Format 'HH:mm:ss')] $Name ====" -ForegroundColor Cyan
    $t = Get-Date
    & $taskPython @StepArgs
    if ($LASTEXITCODE -ne 0) {
        Write-Host "FALLITO: $Name (exit $LASTEXITCODE). Log: $log" -ForegroundColor Red
        Stop-Transcript | Out-Null
        exit $LASTEXITCODE
    }
    Write-Host ("OK: {0} in {1:hh\:mm\:ss}" -f $Name, ((Get-Date) - $t)) -ForegroundColor Green
}

try {
    Write-Host "Sorgente (sola lettura): $sourceData"
    Write-Host "Work-root: $workRoot`nRun: $RunId  Device: $Device  Resume: $Resume"

    Invoke-Step 'Controllo ambiente' (@('-m', 'biomap_ml.tools.check_environment') + $e2Args + @('--device', $Device))
    Invoke-Step 'Audit dataset (index)' (@('-m', 'biomap_ml.prepare') + $e2Args + @('--stage', 'index', '--device', 'cpu'))
    Get-Content (Join-Path $workRoot 'dataset\audit.json') -TotalCount 40
    Invoke-Step 'Cache embedding puntuali' (@('-m', 'biomap_ml.prepare') + $e2Args + @('--stage', 'cache', '--device', $Device))
    if (-not $SkipProfile) {
        Invoke-Step 'Profilo training CNN' (@('-m', 'biomap_ml.tools.profile_training') + $e2Args + @('--steps', '20', '--device', $Device))
        Get-Content (Join-Path $workRoot 'profile_E2.json')
    }
    $trainArgs = @('-m', 'biomap_ml.train') + $e2Args + @('--run-id', $RunId, '--phase', 'all', '--device', $Device)
    if ($Resume -and (Test-Path (Join-Path $workRoot "runs\$RunId"))) { $trainArgs += '--resume' }
    Invoke-Step 'Training E2 (3 fold + refit)' $trainArgs
    Invoke-Step 'Valutazione Liguria + Excel' (@('-m', 'biomap_ml.evaluate') + $e2Args + @('--run-id', $RunId, '--device', $Device))
    Invoke-Step 'Export GeoTIFF' (@('-m', 'biomap_ml.predict') + $e2Args + @('--run-id', $RunId))
    Write-Host "`nE2 completato. Risultati: $workRoot\runs\$RunId\report.md" -ForegroundColor Green
}
finally {
    try { Stop-Transcript | Out-Null } catch {}
}
