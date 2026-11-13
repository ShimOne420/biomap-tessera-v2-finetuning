# SCOPO
#   Fermare la campagna V2 densa r1 in modo ordinato SUBITO DOPO il completamento di una tile, senza
#   che lo script di ripresa (run_v2_dense_stream.ps1) la rilanci. Pensato per sostituire il cavo
#   Ethernet o cambiare porta e poi riprendere con lo stesso comando.
# COME VIENE RICHIAMATO (PowerShell, in una finestra diversa da quella del run)
#   powershell -ExecutionPolicy Bypass -File scripts\stop_r1_after_tile.ps1            # prova: mostra cosa farebbe
#   powershell -ExecutionPolicy Bypass -File scripts\stop_r1_after_tile.ps1 -Execute   # attende la tile e ferma
#   Parametri: -WorkRoot F:\biomap_dense_public\work -PollSeconds 5 -TimeoutMinutes 90
# LOGICA IN ORDINE
#   1. Trova i processi: lo script di ripresa (powershell con run_v2_dense_stream.ps1) e i python con
#      "dense_campaign.py stream" e la work-root indicata (lanciatore .venv e processo reale).
#   2. Legge SOLO la dimensione di stream_log.jsonl (metadati, nessuna apertura del file). La r1 aggiunge
#      una riga a quel file dopo aver scritto tile.npz + tile.json, verificato la rilettura e cancellato i
#      temporanei: una crescita del file = una tile appena resa persistente.
#   3. Con -Execute: attende la crescita, poi termina SUBITO prima lo script di ripresa (così non rilancia),
#      poi i processi python. La tile successiva è appena iniziata: si perdono al massimo pochi secondi di
#      lavoro; i giorni Sentinel già salvati restano in tmp e vengono riusati.
#   4. Stampa l'ultima tile completata (lettura condivisa dell'ultima riga del log).
# INPUT E OUTPUT
#   Input: elenco dei processi, dimensione e ultima riga di stream_log.jsonl. Output: solo testo.
#   Non scrive nulla. Termina processi SOLO con -Execute.
# PARAMETRI
#   -Execute (senza: solo prova), -WorkRoot, -PollSeconds, -TimeoutMinutes.
# RISORSE
#   Trascurabili.
# RIPRESA
#   Dopo l'arresto: verify_r1_resume.py, poi lo stesso comando run_v2_dense_stream.ps1 dell'avvio.
param(
    [switch]$Execute,
    [string]$WorkRoot = 'F:\biomap_dense_public\work',
    [int]$PollSeconds = 5,
    [int]$TimeoutMinutes = 90
)
$ErrorActionPreference = 'Stop'
$log = Join-Path $WorkRoot 'stream_log.jsonl'

function Get-LastLine([string]$Path) {
    $share = [IO.FileShare]::ReadWrite -bor [IO.FileShare]::Delete
    $fs = [IO.File]::Open($Path, [IO.FileMode]::Open, [IO.FileAccess]::Read, $share)
    try {
        $len = $fs.Length; $take = [Math]::Min(65536, $len)
        $fs.Seek($len - $take, 'Begin') | Out-Null
        $buf = New-Object byte[] $take; [void]$fs.Read($buf, 0, $take)
        $text = [Text.Encoding]::UTF8.GetString($buf).TrimEnd()
        return ($text -split "`n")[-1]
    } finally { $fs.Dispose() }
}

function Get-Targets {
    $all = Get-CimInstance Win32_Process
    $wrappers = @($all | Where-Object { $_.Name -match 'powershell|pwsh' -and $_.CommandLine -like '*run_v2_dense_stream.ps1*' })
    $workEsc = [regex]::Escape($WorkRoot) + '(\s|"|$)'
    $pythons = @($all | Where-Object { $_.Name -eq 'python.exe' -and $_.CommandLine -like '*dense_campaign.py*stream*' -and $_.CommandLine -match $workEsc })
    return @{ wrappers = $wrappers; pythons = $pythons }
}

$t = Get-Targets
Write-Host ('Script di ripresa : {0}' -f (($t.wrappers | ForEach-Object { "PID $($_.ProcessId) avviato $($_.CreationDate)" }) -join '; '))
Write-Host ('Processi python   : {0}' -f (($t.pythons | ForEach-Object { "PID $($_.ProcessId)" }) -join ', '))
if ($t.pythons.Count -eq 0) { Write-Host 'Nessun processo della campagna attivo: niente da fermare.' -ForegroundColor Yellow; exit 0 }
$last = Get-LastLine $log | ConvertFrom-Json
Write-Host ('Ultima tile nel log: {0}/{1} ({2} pixel, finita {3} UTC)' -f $last.aoi_id, $last.tile, $last.coverage.pixels, $last.finished_utc)
if (-not $Execute) {
    Write-Host 'PROVA: nessun processo terminato. Con -Execute attende la prossima tile completata e ferma la campagna.' -ForegroundColor Cyan
    exit 0
}
$size0 = (Get-Item $log).Length
$deadline = (Get-Date).AddMinutes($TimeoutMinutes)
Write-Host ("In attesa della prossima tile completata (controllo ogni {0} s, massimo {1} min). Ctrl+C per annullare senza fermare nulla." -f $PollSeconds, $TimeoutMinutes)
while ((Get-Item $log).Length -eq $size0) {
    if ((Get-Date) -gt $deadline) { Write-Host 'Tempo massimo raggiunto: nessun processo terminato.' -ForegroundColor Yellow; exit 2 }
    Start-Sleep -Seconds $PollSeconds
}
$t = Get-Targets
foreach ($p in $t.wrappers) { Stop-Process -Id $p.ProcessId -Force; Write-Host "Terminato script di ripresa PID $($p.ProcessId)" }
foreach ($p in ($t.pythons | Sort-Object { $_.ParentProcessId -in $t.pythons.ProcessId } -Descending)) {
    Stop-Process -Id $p.ProcessId -Force -ErrorAction SilentlyContinue; Write-Host "Terminato python PID $($p.ProcessId)"
}
Start-Sleep -Seconds 2
$last = Get-LastLine $log | ConvertFrom-Json
Write-Host ('FERMATO alle {0}. Ultima tile resa persistente: {1}/{2} ({3} pixel).' -f (Get-Date -Format 'HH:mm:ss'), $last.aoi_id, $last.tile, $last.coverage.pixels) -ForegroundColor Green
$left = @((Get-Targets).pythons).Count
if ($left) { Write-Host "ATTENZIONE: $left processi della campagna ancora attivi" -ForegroundColor Red; exit 1 }
Write-Host 'Prossimo passo: python scripts\verify_r1_resume.py ... (vedi docs\performance\PROCEDURA_CAVO_E_RIPRESA.md)'
