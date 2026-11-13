# SCOPO
#   Mostrare in tempo reale QUANTO MANCA alla fine dell'estrazione V2 densa in streaming
#   (comando `dense_campaign.py stream`, avviato da scripts\run_v2_dense_stream.ps1). Sola lettura:
#   non tocca il run, non scrive file, si può chiudere con Ctrl+C in qualsiasi momento.
#
# COME SI RICHIAMA (in un secondo terminale PowerShell, dalla cartella del repository)
#   powershell -ExecutionPolicy Bypass -File scripts\eta_v2_dense.ps1 -WorkRoot F:\biomap_dense_public\work
#   Opzione: -Seconds 60 (intervallo di aggiornamento).
#
# LOGICA DELLA STIMA
#   - Pixel fatti/totali da stream_progress.json (scritto dal run dopo ogni blocco codificato).
#   - Velocità REALE da stream_log.jsonl: somma dei pixel delle tile completate diviso la somma dei loro
#     tempi (acquisizione Sentinel + codifica). Include quindi la rete, che è il collo di bottiglia, ed
#     esclude le pause tra un riavvio e l'altro. Tempo mancante = pixel mancanti / velocità.
#   - Prima che si completi la prima tile si mostra solo l'ETA lineare del run (meno affidabile).
#   - Le tile hanno dimensioni e numero di scene diversi: la stima è indicativa e si stabilizza
#     dopo una decina di tile.
#   - Lo stato finale affidabile resta stream_status.json (status = complete).
param([string]$WorkRoot = 'F:\biomap_dense_public\work', [int]$Seconds = 60)
while ($true) {
    Clear-Host
    "ETA estrazione V2 densa - $(Get-Date -Format 'dd/MM HH:mm:ss')"
    $status = Join-Path $WorkRoot 'stream_status.json'
    if ((Test-Path $status) -and ((Get-Content $status -Raw | ConvertFrom-Json).status -eq 'complete')) {
        'COMPLETATO: stream_status.json = complete. Passare a probe/evaluate-liguria/compare/export.'
        break
    }
    $p = Join-Path $WorkRoot 'stream_progress.json'
    $log = Join-Path $WorkRoot 'stream_log.jsonl'
    if (-not (Test-Path $p)) {
        'Il run non ha ancora codificato nessun blocco (acquisizione della prima tile in corso o run non avviato).'
    } else {
        $s = Get-Content $p -Raw | ConvertFrom-Json
        $left = $s.pixels_total - $s.pixels_done
        "Tile complete:  $($s.tiles_done)/$($s.tiles_total)   (corrente: $($s.current))"
        "Pixel:          {0:N0} / {1:N0}  ({2:N2}%)" -f $s.pixels_done, $s.pixels_total, (100.0 * $s.pixels_done / $s.pixels_total)
        $rows = @()
        if (Test-Path $log) { $rows = @(Get-Content $log | ForEach-Object { $_ | ConvertFrom-Json }) }
        if ($rows.Count -gt 0) {
            $px = ($rows | ForEach-Object { [double]$_.coverage.pixels } | Measure-Object -Sum).Sum
            $sec = ($rows | ForEach-Object { [double]$_.acquire_seconds + [double]$_.encode_seconds } | Measure-Object -Sum).Sum
            $rate = $px / $sec
            $eta = [TimeSpan]::FromSeconds($left / $rate)
            "Velocita reale: {0:N0} pixel/s (su {1} tile completate, rete inclusa)" -f $rate, $rows.Count
            "TEMPO MANCANTE: {0}g {1}h {2:D2}m   -> fine prevista circa {3}" -f $eta.Days, $eta.Hours, $eta.Minutes, (Get-Date).Add($eta).ToString('ddd dd/MM HH:mm')
        } elseif ($s.eta_hours_linear) {
            "ETA provvisoria: {0:N1} ore (nessuna tile ancora completa: stima grezza)" -f $s.eta_hours_linear
        }
        "Ultimo aggiornamento del run: $($s.updated_utc) UTC"
    }
    $python = Get-Process python -ErrorAction SilentlyContinue | Where-Object { $_.Path -like '*biomap-tessera-v2-finetuning*' }
    if ($python) { "Processo attivo: PID $($python.Id -join ', ')" } else { 'ATTENZIONE: nessun processo python del run in esecuzione' }
    Start-Sleep -Seconds $Seconds
}
