# SCOPO
#   Misurare in SOLA LETTURA velocità del collegamento e traffico reale della scheda Ethernet: serve a
#   confrontare la campagna prima e dopo il cambio di cavo o porta.
# COME VIENE RICHIAMATO
#   powershell -ExecutionPolicy Bypass -File scripts\measure_net.ps1 [-Minutes 10] [-IntervalSeconds 10] [-Adapter Ethernet]
# LOGICA IN ORDINE
#   1. Stampa modello, velocità negoziata e duplex della scheda.
#   2. Ogni -IntervalSeconds legge i contatori di byte ricevuti (Get-NetAdapterStatistics) e stampa i Mbit/s.
#   3. Alla fine: media, 95° percentile e massimo, più errori e pacchetti scartati nel periodo.
# INPUT E OUTPUT
#   Input: contatori di Windows. Output: solo testo; nessuna modifica.
# PARAMETRI
#   -Minutes (10), -IntervalSeconds (10), -Adapter (Ethernet).
# RISORSE
#   Trascurabili. Il traffico misurato è quello di TUTTO il PC, non solo della campagna.
# RIPRESA
#   Nessuno stato.
param([int]$Minutes = 10, [int]$IntervalSeconds = 10, [string]$Adapter = 'Ethernet')
$a = Get-NetAdapter -Name $Adapter
Write-Host ('{0} | {1} | velocità negoziata {2} | full duplex {3}' -f $a.Name, $a.InterfaceDescription, $a.LinkSpeed, $a.FullDuplex)
$prev = Get-NetAdapterStatistics -Name $Adapter; $t0 = Get-Date; $start = $prev
$samples = New-Object System.Collections.Generic.List[double]
$end = (Get-Date).AddMinutes($Minutes)
while ((Get-Date) -lt $end) {
    Start-Sleep -Seconds $IntervalSeconds
    $now = Get-NetAdapterStatistics -Name $Adapter; $t1 = Get-Date
    $mbit = ($now.ReceivedBytes - $prev.ReceivedBytes) * 8 / ($t1 - $t0).TotalSeconds / 1e6
    $samples.Add($mbit)
    Write-Host ('{0}  ricezione {1,7:N1} Mbit/s' -f $t1.ToString('HH:mm:ss'), $mbit)
    $prev = $now; $t0 = $t1
}
$sorted = $samples | Sort-Object
$p95 = $sorted[[Math]::Min($sorted.Count - 1, [int][Math]::Floor(0.95 * $sorted.Count))]
Write-Host ('RIEPILOGO {0} campioni: media {1:N1} Mbit/s, 95° percentile {2:N1}, massimo {3:N1}' -f $samples.Count, ($samples | Measure-Object -Average).Average, $p95, ($samples | Measure-Object -Maximum).Maximum)
Write-Host ('Errori ricezione nel periodo: {0}, scartati: {1}' -f ($prev.ReceivedPacketErrors - $start.ReceivedPacketErrors), ($prev.ReceivedDiscardedPackets - $start.ReceivedDiscardedPackets))
