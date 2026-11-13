<#
=====================================================================================
 watch_e2.ps1 - Monitor live (sola lettura) del run E2 lanciato con run_e2.ps1
=====================================================================================

 COSA FA
   Ogni N secondi pulisce lo schermo e stima quanto manca alla fine del run E2,
   passo per passo, piu' un totale:
     - AUDIT (prepare --stage index): l'audit legge ogni serie due volte (SHA256 + np.load),
       quindi il progresso e' stimato come byte letti dal processo Python (contatore I/O di
       Windows, ReadTransferCount) / (2 x dimensione totale di series\). E' un'approssimazione.
     - CACHE (prepare --stage cache): righe gia' presenti nella tabella `points` del SQLite
       (aperto in sola lettura) su 132.900 celle; velocita' calcolata tra due letture.
     - TRAINING (3 fold + refit): legge i history.csv (epoch_seconds) e stima un intervallo:
       minimo = early stopping al primo momento possibile (patience 8),
       massimo = tutte le 50 epoche E2 per fold e 50 nel refit. Il tempo per epoca e' scalato
       sul numero di celle di training di ciascun fit (fold 89.101/83.014/83.703, refit 127.909).
     - Profilo, valutazione Liguria ed export sono brevi e non sono stimati.
   Prima che un passo inizi, la sua durata non e' stimabile e viene indicata come tale.
   Non scrive, non blocca e non interrompe nulla: Ctrl+C chiude solo il monitor.

 COME RICHIAMARLO (in un secondo terminale VS Code, nella root del checkout)
   powershell -ExecutionPolicy Bypass -File .\scripts\watch_e2.ps1
   Parametro opzionale: -Seconds 30 (intervallo di aggiornamento, default 30).
=====================================================================================
#>
param([int]$Seconds = 30)
$repo = Split-Path $PSScriptRoot -Parent
$work = Join-Path $repo 'data_dev\E2'
$runDir = Join-Path $work 'runs\e2_msa_seed42'
$py = Join-Path $repo '.venv\Scripts\python.exe'
$seriesDir = 'C:\Users\brome\Desktop\BIOMAP_TESSERA_V2\data\msa_v1_liguria\series'
$totalCells = 132900
$foldCells = @(89101, 83014, 83703)
$refitCells = 127909
$warmup = 5; $maxEpochs = 50; $patience = 8

function Fmt([double]$sec) {
    if ($sec -lt 0) { $sec = 0 }
    $t = [TimeSpan]::FromSeconds($sec)
    return ('{0}h {1:00}m' -f [int][math]::Floor($t.TotalHours), $t.Minutes)
}

# Legge un history.csv: righe completate, secondi totali, epoche dall'ultimo miglioramento.
function Read-History([string]$path) {
    if (-not (Test-Path $path)) { return $null }
    $rows = @(Import-Csv $path)
    if ($rows.Count -eq 0) { return $null }
    $best = [double]::MaxValue; $bestIdx = 0; $i = 0
    foreach ($r in $rows) {
        $i++
        if ($r.validation_macro_aoi_rmse -ne '' -and [double]$r.validation_macro_aoi_rmse -lt $best - 1e-5) {
            $best = [double]$r.validation_macro_aoi_rmse; $bestIdx = $i
        }
    }
    $sec = ($rows | Measure-Object -Property epoch_seconds -Sum).Sum
    return @{ Done = $rows.Count; Seconds = $sec; Stale = $rows.Count - $bestIdx; BestEpoch = $bestIdx }
}

# Dimensione totale delle serie: calcolata una sola volta (265k file, ~1 min) e memorizzata in TEMP.
$sizeFile = Join-Path $env:TEMP 'watch_e2_series_bytes.txt'
$seriesBytes = 0
if (Test-Path $sizeFile) { $seriesBytes = [double](Get-Content $sizeFile -Raw) }
if ($seriesBytes -le 0) {
    Write-Host 'Calcolo dimensione serie (solo la prima volta, circa 1 minuto)...'
    $seriesBytes = (Get-ChildItem $seriesDir -Recurse -File | Measure-Object Length -Sum).Sum
    Set-Content -Path $sizeFile -Value $seriesBytes -Encoding ascii
}
$prevCache = $null

while ($true) {
    $now = Get-Date
    $out = @("E2 monitor - $($now.ToString('HH:mm:ss'))   (Ctrl+C chiude solo il monitor, il run continua)", '')

    $procs = @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -match 'biomap_ml\.' })
    $module = ''; $stage = ''; $elapsed = $null; $readBytes = 0
    if ($procs.Count) {
        $p = $procs | Sort-Object CreationDate | Select-Object -First 1
        $module = [regex]::Match($p.CommandLine, 'biomap_ml\.[\w\.]+').Value
        $stage = [regex]::Match($p.CommandLine, '--stage (\w+)').Groups[1].Value
        $elapsed = $now - $p.CreationDate
        $readBytes = ($procs | Measure-Object ReadTransferCount -Maximum).Maximum
        $out += "Passo in corso: $module $stage  (da $(Fmt $elapsed.TotalSeconds))"
    } else {
        $out += 'Nessun processo E2 attivo: concluso, fallito o tra due passi (vedi log sotto).'
    }
    $out += "GPU (util, VRAM usata, totale): $(nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader 2>$null)"
    $out += ''
    $known = 0.0; $knownMax = 0.0; $unknown = @()

    # 1. Audit
    if (Test-Path (Join-Path $work 'dataset\audit.json')) {
        $out += '[1] Audit:    completato'
    } elseif ($module -eq 'biomap_ml.prepare' -and $stage -eq 'index' -and $readBytes -gt 0) {
        $frac = [math]::Min(0.99, $readBytes / (2.0 * $seriesBytes))
        $rem = $elapsed.TotalSeconds * (1 - $frac) / $frac
        $known += $rem; $knownMax += $rem
        $out += ('[1] Audit:    ~{0:N0}% (stima da I/O)  residuo ~{1}' -f (100 * $frac), (Fmt $rem))
    } else { $out += '[1] Audit:    in attesa'; $unknown += 'audit' }

    # 2. Cache
    $cache = Get-ChildItem (Join-Path $work 'cache\points_*.sqlite3') -ErrorAction SilentlyContinue | Select-Object -First 1
    $done = $null
    if ($cache) {
        $uri = 'file:' + ($cache.FullName -replace '\\', '/') + '?mode=ro'
        $n = & $py -c "import sqlite3,sys;c=sqlite3.connect(sys.argv[1],uri=True,timeout=5);print(c.execute('select count(*) from points').fetchone()[0])" $uri 2>$null
        if ($n -match '^\d+$') { $done = [int]$n }
    }
    if ($null -ne $done -and $done -ge $totalCells) {
        $out += '[2] Cache:    completata'
    } elseif ($null -ne $done) {
        $line = '[2] Cache:    {0:N0}/{1:N0} celle ({2:N1}%)' -f $done, $totalCells, (100.0 * $done / $totalCells)
        if ($prevCache -and $done -gt $prevCache.Done) {
            $rate = ($done - $prevCache.Done) / ($now - $prevCache.Time).TotalSeconds
            $rem = ($totalCells - $done) / $rate
            $known += $rem; $knownMax += $rem
            $line += '  {0:N1} celle/s  residuo ~{1}' -f $rate, (Fmt $rem)
        } else { $line += '  (velocita'' al prossimo aggiornamento)'; $unknown += 'cache' }
        if (-not $prevCache -or $done -ne $prevCache.Done) { $prevCache = @{ Done = $done; Time = $now } }
        $out += $line
    } else { $out += '[2] Cache:    non iniziata (piano: 1-6 h, stimabile all''avvio)'; $unknown += 'cache' }

    # 3. Training
    $secPerCell = $null; $sumSec = 0.0; $sumCellEpochs = 0.0
    $fits = @()
    for ($f = 0; $f -lt 3; $f++) {
        foreach ($ph in 'warmup', 'E2') {
            $h = Read-History (Join-Path $runDir "fold_$f\$ph\history.csv")
            $fits += @{ Name = "fold_$f/$ph"; H = $h; Cells = $foldCells[$f]; Phase = $ph }
            if ($h) { $sumSec += $h.Seconds; $sumCellEpochs += $h.Done * $foldCells[$f] }
        }
    }
    foreach ($ph in 'warmup', 'E2') {
        $h = Read-History (Join-Path $runDir "refit_$ph\history.csv")
        $fits += @{ Name = "refit/$ph"; H = $h; Cells = $refitCells; Phase = "refit_$ph" }
        if ($h) { $sumSec += $h.Seconds; $sumCellEpochs += $h.Done * $refitCells }
    }
    if ($sumCellEpochs -gt 0) { $secPerCell = $sumSec / $sumCellEpochs }
    $selection = Test-Path (Join-Path $runDir 'selection.json')
    $final = Test-Path (Join-Path $runDir 'final.pt')
    if ($final) {
        $out += '[3] Training: completato (refit salvato)'
    } elseif ($secPerCell) {
        $minEp = 0; $maxEp = 0; $minSec = 0.0; $maxSec = 0.0; $lines = @()
        for ($k = 0; $k -lt $fits.Count; $k++) {
            $fit = $fits[$k]
            $d = if ($fit.H) { $fit.H.Done } else { 0 }
            # Un fit E2 e' concluso (early stopping) se il fit successivo e' gia' iniziato.
            $nextStarted = ($k + 1 -lt $fits.Count) -and $fits[$k + 1].H
            switch ($fit.Phase) {
                'warmup' { $lo = $warmup - $d; $hi = $lo }
                'E2' {
                    $hi = $maxEpochs - $d
                    $lo = if ($d -eq 0) { $patience + 1 } else { [math]::Max(0, $patience - $fit.H.Stale) }
                    if ($d -gt 0 -and ($fit.H.Stale -ge $patience -or $selection -or $nextStarted)) { $lo = 0; $hi = 0 }
                }
                'refit_warmup' { $lo = $warmup - $d; $hi = $lo }
                'refit_E2' { $lo = [math]::Max(0, 1 - $d); $hi = $maxEpochs - $d }
            }
            $lo = [math]::Max(0, $lo); $hi = [math]::Max(0, $hi)
            $minSec += $lo * $secPerCell * $fit.Cells; $maxSec += $hi * $secPerCell * $fit.Cells
            if ($fit.H) { $lines += ('      {0,-14} epoche fatte {1,2}  media {2:N0}s/epoca' -f $fit.Name, $d, ($fit.H.Seconds / $d)) }
        }
        $known += $minSec; $knownMax += $maxSec
        $out += ('[3] Training: residuo tra ~{0} (early stopping) e ~{1} (tutte le epoche)' -f (Fmt $minSec), (Fmt $maxSec))
        $out += $lines
    } else { $out += '[3] Training: non iniziato (stimabile dopo la prima epoca)'; $unknown += 'training' }

    $out += '[4] Valutazione Liguria, Excel, mappe: brevi, non stimate'
    $out += ''
    $u = ($unknown | Select-Object -Unique) -join ', '
    $eta = $now.AddSeconds($known); $etaMax = $now.AddSeconds($knownMax)
    if ($knownMax -gt $known) {
        $out += ('TOTALE stimato: tra ~{0} e ~{1}  (fine tra {2:HH:mm} e {3:dd/MM HH:mm})' -f (Fmt $known), (Fmt $knownMax), $eta, $etaMax)
    } else {
        $out += ('TOTALE stimato: ~{0}  (fine ~{1:dd/MM HH:mm})' -f (Fmt $known), $eta)
    }
    if ($u) { $out += "  + passi non ancora stimabili: $u" }

    $log = Get-ChildItem (Join-Path $work 'logs\*.log') -ErrorAction SilentlyContinue | Sort-Object LastWriteTime | Select-Object -Last 1
    if ($log) {
        $out += '', "Log: $($log.Name)"
        $out += Get-Content $log.FullName -Tail 10 | Where-Object { $_ -match '====|OK:|FALLITO|completato' }
    }

    Clear-Host
    $out | ForEach-Object { Write-Host $_ }
    Start-Sleep -Seconds $Seconds
}
