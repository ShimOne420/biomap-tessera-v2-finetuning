# SCOPO
#   Monitor in tempo reale della campagna V2 densa in streaming (scripts\dense_campaign.py stream):
#   mostra avanzamento, velocità, tempo rimanente stimato e ora di fine prevista, senza toccare il run.
#
# COME VIENE RICHIAMATO (PowerShell, in una finestra separata da quella del run)
#   powershell -ExecutionPolicy Bypass -File scripts\watch_v2_dense_eta.ps1
#   Parametri facoltativi: -WorkRoot F:\biomap_dense_public\work -Seconds 60 -WindowMinutes 60
#   Ctrl+C per uscire: chiude solo il monitor, non il run.
#
# LOGICA IN ORDINE (a ogni giro)
#   1. Legge stream_progress.json e stream_log.jsonl aprendoli con condivisione ReadWrite+Delete e
#      chiudendoli subito: così il run può sostituire stream_progress.json (os.replace) anche mentre
#      lo leggiamo. Il blocco "Accesso negato" dell'11/11 era causato da letture senza condivisione Delete.
#   2. Stima A (velocità recente): pixel fatti negli ultimi -WindowMinutes, letti da campioni
#      (updated_utc, pixels_done) raccolti dal monitor stesso. Serve almeno qualche minuto di dati.
#   3. Stima B (modello sulle tile finite): da stream_log.jsonl, per ogni tile completata,
#      tempo = acquire_seconds + encode_seconds. Minimi quadrati senza intercetta su
#      tempo = a * 1 tile + b * pixel  (a = costo fisso di acquisizione per tile, b = costo per pixel).
#      ETA = a * tile mancanti + b * pixel mancanti. È robusta anche subito dopo un riavvio, perché usa
#      la storia; ignora le tile riusate dalla cache (acquisizione < 30 s) che falserebbero i costi.
#   4. Controlla che esista un processo python con "dense_campaign.py" e quanto è vecchio l'ultimo
#      aggiornamento: oltre -StaleMinutes senza processo segnala FERMO.
#   5. Mostra anche il numero di errori in stream_errors.jsonl e lo spazio libero sul disco.
#
# INPUT E OUTPUT
#   Input (solo lettura): <WorkRoot>\stream_progress.json, stream_log.jsonl, stream_errors.jsonl,
#   extraction_complete.json. Output: solo testo a schermo. Non scrive e non cancella nulla.
#
# PARAMETRI
#   -WorkRoot      cartella --work-root del run (default F:\biomap_dense_public\work)
#   -Seconds       intervallo di aggiornamento, minimo 30 s (default 60)
#   -WindowMinutes finestra della stima A (default 60): le tile grandi alternano ~15 min di
#                  acquisizione senza pixel nuovi a fasi di codifica, quindi finestre corte oscillano.
#   -StaleMinutes  soglia oltre cui un progresso fermo è sospetto (default 40)
#
# RISORSE
#   Trascurabili: due letture di file piccoli al minuto e una query dei processi.
#
# RIPRESA
#   Nessuno stato su disco: si può chiudere e rilanciare quando si vuole; la stima A riparte da zero,
#   la stima B no.
param(
    [string]$WorkRoot = 'F:\biomap_dense_public\work',
    [int]$Seconds = 60,
    [int]$WindowMinutes = 60,
    [int]$StaleMinutes = 40
)
$ErrorActionPreference = 'Continue'
if ($Seconds -lt 30) { $Seconds = 30 }

function Read-Shared([string]$Path) {
    if (-not (Test-Path $Path)) { return $null }
    $share = [IO.FileShare]::ReadWrite -bor [IO.FileShare]::Delete
    $fs = $null
    try {
        $fs = [IO.File]::Open($Path, [IO.FileMode]::Open, [IO.FileAccess]::Read, $share)
        $reader = New-Object IO.StreamReader($fs)
        return $reader.ReadToEnd()
    } catch { return $null } finally { if ($fs) { $fs.Dispose() } }
}

function Format-Span([double]$s) {
    if ($s -le 0 -or [double]::IsNaN($s) -or [double]::IsInfinity($s)) { return 'n.d.' }
    $t = [TimeSpan]::FromSeconds($s)
    return ('{0} h {1:00} min' -f [int][Math]::Floor($t.TotalHours), $t.Minutes)
}

function Format-End([double]$s) {
    if ($s -le 0 -or [double]::IsNaN($s) -or [double]::IsInfinity($s)) { return 'n.d.' }
    return (Get-Date).AddSeconds($s).ToString('ddd dd/MM HH:mm')
}

$samples = New-Object System.Collections.ArrayList
while ($true) {
    $raw = Read-Shared (Join-Path $WorkRoot 'stream_progress.json')
    $p = $null
    if ($raw) { try { $p = $raw | ConvertFrom-Json } catch { $p = $null } }

    # Stima B: modello tempo = a*tile + b*pixel sulle tile completate (non riusate dalla cache).
    $Saa = 0.0; $Sab = 0.0; $Sbb = 0.0; $Sat = 0.0; $Sbt = 0.0; $nFit = 0
    $logRaw = Read-Shared (Join-Path $WorkRoot 'stream_log.jsonl')
    if ($logRaw) {
        foreach ($line in ($logRaw -split "`n")) {
            if (-not $line.Trim()) { continue }
            try { $e = $line | ConvertFrom-Json } catch { continue }
            if ($null -eq $e.acquire_seconds -or $e.acquire_seconds -lt 30) { continue }
            $px = [double]$e.coverage.pixels
            $t = [double]$e.acquire_seconds + [double]$e.encode_seconds
            $Saa += 1; $Sab += $px; $Sbb += $px * $px; $Sat += $t; $Sbt += $px * $t; $nFit++
        }
    }
    $etaB = [double]::NaN; $a = [double]::NaN; $b = [double]::NaN
    $det = $Saa * $Sbb - $Sab * $Sab
    if ($nFit -ge 5 -and $det -gt 0) {
        $a = ($Sat * $Sbb - $Sab * $Sbt) / $det
        $b = ($Saa * $Sbt - $Sab * $Sat) / $det
    }

    Clear-Host
    Write-Host ('Monitor V2 denso  -  {0}   (aggiornamento ogni {1} s, Ctrl+C per uscire: il run continua)' -f (Get-Date -Format 'dd/MM HH:mm:ss'), $Seconds)
    Write-Host ''
    $done = Read-Shared (Join-Path $WorkRoot 'extraction_complete.json')
    if ($done) {
        Write-Host 'CAMPAGNA COMPLETATA: extraction_complete.json presente' -ForegroundColor Green
        Write-Host $done
        break
    }
    if (-not $p) {
        Write-Host "stream_progress.json non leggibile in $WorkRoot (riprovo)" -ForegroundColor Yellow
        Start-Sleep -Seconds $Seconds
        continue
    }

    $updated = [DateTime]::Parse($p.updated_utc).ToUniversalTime()
    $remainingPx = [double]$p.pixels_total - [double]$p.pixels_done
    $remainingTiles = [double]$p.tiles_total - [double]$p.tiles_done
    if ($samples.Count -eq 0 -or $samples[$samples.Count - 1].t -ne $updated) {
        [void]$samples.Add([pscustomobject]@{ t = $updated; px = [double]$p.pixels_done })
    }
    while ($samples.Count -gt 2 -and ($updated - $samples[0].t).TotalMinutes -gt $WindowMinutes) { $samples.RemoveAt(0) }

    # Stima A: velocità recente osservata dal monitor.
    $etaA = [double]::NaN; $rateA = [double]::NaN
    if ($samples.Count -ge 2) {
        $dt = ($samples[$samples.Count - 1].t - $samples[0].t).TotalSeconds
        $dp = $samples[$samples.Count - 1].px - $samples[0].px
        if ($dt -ge 300 -and $dp -gt 0) { $rateA = $dp / $dt; $etaA = $remainingPx / $rateA }
    }
    if (-not [double]::IsNaN($a)) {
        # Il pixel della tile corrente già fatti sono esclusi da remainingPx; la tile corrente conta per intero (prudente).
        # 0.0 e non 0: con un intero PowerShell sceglie Max(int,int) e arrotonda b (~0,001 s/pixel) a zero.
        $etaB = [Math]::Max([double]0.0, $a) * $remainingTiles + [Math]::Max([double]0.0, $b) * $remainingPx
    }

    $pct = 100.0 * [double]$p.pixels_done / [double]$p.pixels_total
    $barLen = 40; $filled = [int][Math]::Round($barLen * $pct / 100)
    Write-Host ('[{0}{1}] {2:N1}%' -f ('#' * $filled), ('.' * ($barLen - $filled)), $pct)
    Write-Host ('Pixel   : {0:N0} / {1:N0}   (mancano {2:N0})' -f $p.pixels_done, $p.pixels_total, $remainingPx)
    Write-Host ('Tile    : {0} / {1}   (mancano {2})' -f $p.tiles_done, $p.tiles_total, $remainingTiles)
    if ($p.tile_pixels -gt 0) {
        Write-Host ('Corrente: {0}  {1:N0} / {2:N0} pixel ({3:N0}%)' -f $p.current, $p.tile_pixels_done, $p.tile_pixels, (100.0 * $p.tile_pixels_done / $p.tile_pixels))
    } else { Write-Host ('Corrente: {0}' -f $p.current) }
    Write-Host ''
    if ([double]::IsNaN($etaA)) {
        Write-Host ('Stima A (ultimi {0} min) : in raccolta dati, almeno 5 min con pixel nuovi' -f $WindowMinutes)
    } else {
        Write-Host ('Stima A (ultimi {0} min) : {1,-14} fine ~ {2}   ({3:N0} pixel/s)' -f $WindowMinutes, (Format-Span $etaA), (Format-End $etaA), $rateA)
    }
    if ([double]::IsNaN($etaB)) {
        Write-Host 'Stima B (tile finite)   : servono almeno 5 tile nel log'
    } else {
        Write-Host ('Stima B (tile finite)   : {0,-14} fine ~ {1}   ({2:N0} s fissi/tile + {3:N1} ms/pixel, {4} tile)' -f (Format-Span $etaB), (Format-End $etaB), $a, (1000 * $b), $nFit)
    }
    Write-Host ('Stima del run (sessione): {0,-14} fine ~ {1}' -f (Format-Span (3600 * [double]$p.eta_hours_linear)), (Format-End (3600 * [double]$p.eta_hours_linear)))
    Write-Host ''

    $age = ((Get-Date).ToUniversalTime() - $updated).TotalMinutes
    $proc = @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue |
              Where-Object { $_.CommandLine -like '*dense_campaign.py*stream*' })
    if ($proc.Count -eq 0) {
        Write-Host ('STATO: nessun processo dense_campaign attivo; ultimo aggiornamento {0:N0} min fa' -f $age) -ForegroundColor Red
        Write-Host '       Se lo script di ripresa è aperto sta aspettando il rilancio, altrimenti va rilanciato.' -ForegroundColor Red
    } elseif ($age -gt $StaleMinutes) {
        Write-Host ('STATO: processo attivo ma nessun progresso da {0:N0} min (acquisizione lunga o rete lenta?)' -f $age) -ForegroundColor Yellow
    } else {
        Write-Host ('STATO: in corso, ultimo aggiornamento {0:N0} min fa' -f $age) -ForegroundColor Green
    }
    $errRaw = Read-Shared (Join-Path $WorkRoot 'stream_errors.jsonl')
    $nErr = if ($errRaw) { @($errRaw -split "`n" | Where-Object { $_.Trim() }).Count } else { 0 }
    $drive = (Split-Path -Qualifier $WorkRoot).TrimEnd(':')
    $free = (Get-PSDrive $drive -ErrorAction SilentlyContinue).Free / 1GB
    Write-Host ('Errori registrati: {0}   Spazio libero {1}: {2:N0} GB' -f $nErr, $drive, $free)
    Start-Sleep -Seconds $Seconds
}
