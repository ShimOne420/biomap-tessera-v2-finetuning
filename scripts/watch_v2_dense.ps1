# SCOPO
#   Mostrare in tempo reale l'avanzamento dell'estrazione V2 densa in streaming, senza toccare il run.
#   Legge solo file della work-root: stream_progress.json, stream_log.jsonl, stream_errors.jsonl.
#
# COME SI RICHIAMA (in un secondo terminale PowerShell)
#   powershell -ExecutionPolicy Bypass -File scripts\watch_v2_dense.ps1 -WorkRoot F:\biomap_dense_public\work
#   Opzione: -Seconds 60 (intervallo di aggiornamento). Ctrl+C per uscire: il run continua.
param([Parameter(Mandatory = $true)][string]$WorkRoot, [int]$Seconds = 60)
while ($true) {
    Clear-Host
    $p = Join-Path $WorkRoot 'stream_progress.json'
    if (Test-Path $p) {
        $s = Get-Content $p -Raw | ConvertFrom-Json
        $pct = if ($s.pixels_total) { 100.0 * $s.pixels_done / $s.pixels_total } else { 0 }
        "Aggiornato:      $($s.updated_utc) (UTC)"
        "Tile corrente:   $($s.current)  ($($s.tile_pixels_done)/$($s.tile_pixels) pixel)"
        "Tile complete:   $($s.tiles_done)/$($s.tiles_total)"
        "Pixel:           {0:N0}/{1:N0}  ({2:N2}%)" -f $s.pixels_done, $s.pixels_total, $pct
        "Velocita:        {0:N0} pixel/s in questa sessione" -f $s.pixels_per_second_this_session
        if ($s.eta_hours_linear) { "ETA lineare:     {0:N1} ore (stima, non garantita)" -f $s.eta_hours_linear }
    } else { "Nessun avanzamento ancora (fase di acquisizione della prima tile)." }
    $log = Join-Path $WorkRoot 'stream_log.jsonl'
    if (Test-Path $log) {
        ''
        'Ultime tile completate:'
        Get-Content $log -Tail 3 | ForEach-Object {
            $r = $_ | ConvertFrom-Json
            "  $($r.aoi_id)/$($r.tile): acquisizione {0:N0}s, codifica {1:N0}s, S2 medi validi {2:N1}" -f $r.acquire_seconds, $r.encode_seconds, $r.coverage.s2_clear_mean
        }
    }
    $err = Join-Path $WorkRoot 'stream_errors.jsonl'
    if (Test-Path $err) { ''; 'Ultimo errore registrato:'; (Get-Content $err -Tail 1 | ConvertFrom-Json).error }
    $python = Get-Process python -ErrorAction SilentlyContinue | Where-Object { $_.Path -like '*biomap-tessera-v2-finetuning*' }
    ''
    if ($python) { "Processo attivo: PID $($python.Id -join ', ')" } else { 'ATTENZIONE: nessun processo python del repository in esecuzione' }
    Start-Sleep -Seconds $Seconds
}
