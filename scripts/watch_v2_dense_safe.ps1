# SCOPO
#   Monitor in tempo reale della campagna V2 densa che NON PUÒ fermarla: non apre mai stream_progress.json.
#   Verificato il 12/11/2026: su Windows os.replace fallisce con "Accesso negato" se il file di destinazione
#   è aperto da un altro processo, ANCHE con condivisione Read|Write|Delete. Il monitor precedente
#   (watch_v2_dense_eta.ps1), che rileggeva quel file ogni 60 s, è la causa probabile dell'arresto del
#   12/11 alle 12:35 (gli arresti dell'11/11 sono avvenuti con letture analoghe).
# COME VIENE RICHIAMATO
#   powershell -ExecutionPolicy Bypass -File scripts\watch_v2_dense_safe.ps1
#   Parametri: -WorkRoot F:\biomap_dense_public\work -Reference F:\biomap_dense_public\reference -Seconds 60 -ChunkSize 4096
#   Ctrl+C chiude solo il monitor.
# LOGICA IN ORDINE (a ogni giro)
#   1. Pixel e tile totali da reference.json e geometry.json (letti una volta all'avvio; mai riscritti).
#   2. Tile complete = file chunks\<aoi>\<tile>\tile.json esistenti (solo elenco cartelle, nessuna apertura).
#   3. Tile in corso = cartelle tmp\<aoi>\<tile>: giorni salvati e blocchi codificati contati dall'elenco dei file.
#   4. Stima del tempo residuo dal modello tempo = a*tile + b*pixel sulle tile del log stream_log.jsonl
#      (file a sola aggiunta: leggerlo non blocca nulla) escluse quelle riprese dalla cache.
#   5. Stato dei processi della campagna e spazio libero.
# INPUT E OUTPUT
#   Input: elenchi di file, geometry.json, stream_log.jsonl. Output: solo testo a schermo.
# PARAMETRI
#   -WorkRoot, -Reference, -Seconds (minimo 30), -ChunkSize (blocchi della campagna, 4096).
# RISORSE
#   Trascurabili.
# RIPRESA
#   Nessuno stato; si chiude e si rilancia quando si vuole.
param(
    [string]$WorkRoot = 'F:\biomap_dense_public\work',
    [string]$Reference = 'F:\biomap_dense_public\reference',
    [int]$Seconds = 60,
    [int]$ChunkSize = 4096
)
$ErrorActionPreference = 'Continue'
if ($Seconds -lt 30) { $Seconds = 30 }

function Read-Shared([string]$Path) {
    $share = [IO.FileShare]::ReadWrite -bor [IO.FileShare]::Delete
    $fs = $null
    try { $fs = [IO.File]::Open($Path, 'Open', 'Read', $share); return (New-Object IO.StreamReader($fs)).ReadToEnd() }
    catch { return $null } finally { if ($fs) { $fs.Dispose() } }
}
function Format-Span([double]$s) {
    if ($s -le 0 -or [double]::IsNaN($s)) { return 'n.d.' }
    $t = [TimeSpan]::FromSeconds($s); return ('{0} h {1:00} min' -f [int][Math]::Floor($t.TotalHours), $t.Minutes)
}

$spec = (Read-Shared (Join-Path $Reference 'reference.json')) | ConvertFrom-Json
$sizes = @{}; $order = @()
foreach ($a in $spec.areas) { foreach ($t in $a.tiles) {
    $key = "$($a.aoi_id)/$t"; $order += $key
    $sizes[$key] = [double]((Read-Shared (Join-Path $Reference "$($a.aoi_id)\$t\geometry.json")) | ConvertFrom-Json).pixels } }
$total = ($sizes.Values | Measure-Object -Sum).Sum

while ($true) {
    $done = @(Get-ChildItem (Join-Path $WorkRoot 'chunks') -Recurse -Filter tile.json -ErrorAction SilentlyContinue |
              ForEach-Object { "$($_.Directory.Parent.Name)/$($_.Directory.Name)" })
    $doneSet = @{}; foreach ($d in $done) { $doneSet[$d] = $true }
    $pixDone = 0.0; foreach ($d in $done) { if ($sizes.ContainsKey($d)) { $pixDone += $sizes[$d] } }
    $current = @()
    foreach ($dir in @(Get-ChildItem (Join-Path $WorkRoot 'tmp') -Directory -ErrorAction SilentlyContinue | ForEach-Object { Get-ChildItem $_.FullName -Directory })) {
        $key = "$($dir.Parent.Name)/$($dir.Name)"
        $days = @(Get-ChildItem (Join-Path $dir.FullName 'days') -Filter *.json -ErrorAction SilentlyContinue).Count
        $enc = @(Get-ChildItem (Join-Path $dir.FullName 'encoded') -Filter *.json -ErrorAction SilentlyContinue).Count
        if ($sizes.ContainsKey($key)) { $current += [pscustomobject]@{ key = $key; days = $days; px = [Math]::Min($sizes[$key], $enc * $ChunkSize); size = $sizes[$key] } }
    }
    $pixCur = ($current | Measure-Object px -Sum).Sum; if (-not $pixCur) { $pixCur = 0 }
    $remTiles = $order.Count - $done.Count; $remPx = $total - $pixDone - $pixCur

    $Saa = 0.0; $Sab = 0.0; $Sbb = 0.0; $Sat = 0.0; $Sbt = 0.0; $n = 0
    $logRaw = Read-Shared (Join-Path $WorkRoot 'stream_log.jsonl')
    if ($logRaw) { foreach ($line in ($logRaw -split "`n")) {
        if (-not $line.Trim()) { continue }
        try { $e = $line | ConvertFrom-Json } catch { continue }
        if ($null -eq $e.acquire_seconds -or $e.acquire_seconds -lt 30) { continue }
        $px = [double]$e.coverage.pixels; $t = [double]$e.acquire_seconds + [double]$e.encode_seconds
        $Saa += 1; $Sab += $px; $Sbb += $px * $px; $Sat += $t; $Sbt += $px * $t; $n++ } }
    $eta = [double]::NaN; $det = $Saa * $Sbb - $Sab * $Sab
    if ($n -ge 5 -and $det -gt 0) {
        $a = ($Sat * $Sbb - $Sab * $Sbt) / $det; $b = ($Saa * $Sbt - $Sab * $Sat) / $det
        $eta = [Math]::Max([double]0.0, $a) * $remTiles + [Math]::Max([double]0.0, $b) * $remPx }

    Clear-Host
    Write-Host ('Monitor V2 denso (sicuro: non apre stream_progress.json)  -  {0}' -f (Get-Date -Format 'dd/MM HH:mm:ss'))
    $pct = 100.0 * ($pixDone + $pixCur) / $total; $fill = [int][Math]::Round(40 * $pct / 100)
    Write-Host ('[{0}{1}] {2:N1}%' -f ('#' * $fill), ('.' * (40 - $fill)), $pct)
    Write-Host ('Tile complete : {0} / {1}' -f $done.Count, $order.Count)
    Write-Host ('Pixel         : {0:N0} / {1:N0}' -f ($pixDone + $pixCur), $total)
    foreach ($c in $current) { Write-Host ('In corso      : {0}  giorni salvati {1}, codificati {2:N0} / {3:N0} pixel' -f $c.key, $c.days, $c.px, $c.size) }
    if ([double]::IsNaN($eta)) { Write-Host 'Tempo residuo : servono almeno 5 tile nel log' }
    else { Write-Host ('Tempo residuo : {0}   fine ~ {1}' -f (Format-Span $eta), (Get-Date).AddSeconds($eta).ToString('ddd dd/MM HH:mm')) }
    $proc = @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue | Where-Object { $_.CommandLine -like '*dense_campaign.py*stream*' })
    if ($proc.Count -eq 0) { Write-Host 'STATO: nessun processo della campagna attivo' -ForegroundColor Red }
    else { Write-Host ('STATO: in corso (processo avviato {0})' -f ($proc | Sort-Object CreationDate | Select-Object -Last 1).CreationDate) -ForegroundColor Green }
    $drive = (Split-Path -Qualifier $WorkRoot).TrimEnd(':')
    Write-Host ('Spazio libero {0}: {1:N0} GB' -f $drive, ((Get-PSDrive $drive).Free / 1GB))
    Start-Sleep -Seconds $Seconds
}
