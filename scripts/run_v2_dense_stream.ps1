# SCOPO
#   Avviare (o riprendere) l'estrazione V2 densa in streaming della campagna 127.909 + Liguria e
#   rilanciarla automaticamente dopo errori transitori (rete, token SAS, timeout) senza perdere lavoro:
#   ogni giorno Sentinel, ogni blocco codificato e ogni tile completa sono checkpoint atomici.
#
# COME SI RICHIAMA (PowerShell, dalla radice del repository)
#   powershell -ExecutionPolicy Bypass -File scripts\run_v2_dense_stream.ps1 `
#     -Reference F:\biomap_dense_public\reference -WorkRoot F:\biomap_dense_public\work
#   Parametri facoltativi: -Device cuda:0 -PixelBatch 256 -IoThreads 8 -MaxRestarts 50
#
# LOGICA
#   1. Esegue `scripts\dense_campaign.py stream` con l'interprete .venv del repository.
#   2. Output su console e in <WorkRoot>\stream_console.log (append).
#   3. Exit 0 e stream_status.json completo (tutte le tile, tutte le celle) -> fine.
#   4. Errore -> attesa crescente (1, 2, 4 ... max 15 minuti) e rilancio dello stesso comando, fino a
#      -MaxRestarts. Lo stesso errore ripetuto su dati (non di rete) va letto in stream_errors.jsonl:
#      lo script lo segnala ma non lo maschera; dopo i tentativi si ferma con exit 1.
#   Lo script non cancella nulla e non modifica reference, tile V1 o risultati.
param(
    [Parameter(Mandatory = $true)][string]$Reference,
    [Parameter(Mandatory = $true)][string]$WorkRoot,
    [string]$Checkpoint = '',
    [string]$Device = 'cuda:0',
    [int]$PixelBatch = 256,
    [int]$IoThreads = 8,
    [int]$MaxRestarts = 50
)
# Continue: in PowerShell 5.1 lo stderr di python (anche semplici warning) diventa ErrorRecord e con 'Stop'
# interromperebbe il ciclo di ripresa. Gli errori veri si leggono dal codice di uscita.
$ErrorActionPreference = 'Continue'
$Repo = Split-Path -Parent $PSScriptRoot
Set-Location $Repo
$Python = Join-Path $Repo '.venv\Scripts\python.exe'
if (-not (Test-Path $Python)) { throw "Interprete non trovato: $Python" }
if (-not $Checkpoint) { $Checkpoint = (Resolve-Path (Join-Path $Repo '..\data\weights\student_medium.pt')).Path }
New-Item -ItemType Directory -Force $WorkRoot | Out-Null
$Log = Join-Path $WorkRoot 'stream_console.log'
$Arguments = @('scripts\dense_campaign.py', 'stream', '--reference', $Reference, '--work-root', $WorkRoot,
               '--checkpoint', $Checkpoint, '--device', $Device, '--pixel-batch', $PixelBatch, '--io-threads', $IoThreads)
for ($attempt = 0; $attempt -le $MaxRestarts; $attempt++) {
    $stamp = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'
    "[$stamp] avvio tentativo $attempt" | Tee-Object -FilePath $Log -Append
    & $Python -u @Arguments 2>&1 | ForEach-Object { "$_" } | Tee-Object -FilePath $Log -Append
    $code = $LASTEXITCODE
    $status = Join-Path $WorkRoot 'stream_status.json'
    if ($code -eq 0 -and (Test-Path $status)) {
        $s = Get-Content $status -Raw | ConvertFrom-Json
        if ($s.tiles_done -eq $s.tiles_total -and $s.cells_complete -eq $s.cells_eligible) {
            "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] COMPLETATO: $($s.tiles_done)/$($s.tiles_total) tile, $($s.cells_complete) celle" | Tee-Object -FilePath $Log -Append
            exit 0
        }
    }
    $wait = [Math]::Min([Math]::Pow(2, $attempt), 15) * 60
    "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] uscita $($code): ripresa tra $wait s (ultimi errori in stream_errors.jsonl)" | Tee-Object -FilePath $Log -Append
    Start-Sleep -Seconds $wait
}
"Tentativi esauriti: leggere $WorkRoot\stream_errors.jsonl" | Tee-Object -FilePath $Log -Append
exit 1
