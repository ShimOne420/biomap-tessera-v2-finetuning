# SCOPO: sequenza esplicita V1 spatial dopo il termine del run attuale, con output isolati.
# FASI: eventuali target/download autorizzati -> build -> preflight -> profile -> CV/refit -> Liguria.
# INPUT: data-root e opzionali reference/V1Root; OUTPUT: v1_spatial_msa, mai directory V2.
# PARAMETRI: -Preset pilot|intermediate|full, -RunName, -FetchV1, -DownloadTargets, -Resume, -FullApproved.
# ESEMPIO: .\experiments\v1_spatial_cnn_msa\run.ps1 -DataRoot C:\BioMAP_Data -FetchV1 -DownloadTargets.
# RISORSE: CPU/rete preparazione, GPU teste; full richiede -FullApproved e non parte durante lo sviluppo.
# RIPRESA: stesso RunName e -Resume; un errore ferma la sequenza, checkpoint validi conservati.
param(
    [string]$DataRoot='..\data',
    [ValidateSet('pilot','intermediate','full')][string]$Preset='pilot',
    [string]$RunName='pilot_seed42',
    [string]$Reference='', [string]$V1Root='', [string]$TargetsCsv='',
    [switch]$FetchV1,[switch]$DownloadTargets,[switch]$Resume,[switch]$FullApproved
)
$ErrorActionPreference='Stop'
if ($Preset -eq 'full' -and -not $FullApproved) { throw 'Full non autorizzato: valutare pilot e spazio, poi aggiungere esplicitamente -FullApproved' }
$Python=Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path $Python)) { throw 'Eseguire prima setup_windows.ps1 in questa directory' }
$DataRoot=[System.IO.Path]::GetFullPath($DataRoot)
$Work=Join-Path $DataRoot 'v1_spatial_msa'
New-Item -ItemType Directory -Force $Work | Out-Null
$WorkflowHandle=[System.IO.File]::Open((Join-Path $Work 'workflow.lock'),[System.IO.FileMode]::OpenOrCreate,[System.IO.FileAccess]::ReadWrite,[System.IO.FileShare]::None)
$PipelinePath=Join-Path $Work 'pipeline.json'
$Pipeline=@{status='running';started_utc=[DateTime]::UtcNow.ToString('o');preset=$Preset;run_name=$RunName;stage='initializing'}
function Write-Pipeline {
    $Temp=$PipelinePath+'.tmp'
    [System.IO.File]::WriteAllText($Temp,($Pipeline | ConvertTo-Json -Depth 5),[System.Text.UTF8Encoding]::new($false))
    if (Test-Path $PipelinePath) { [System.IO.File]::Replace($Temp,$PipelinePath,$null) }
    else { [System.IO.File]::Move($Temp,$PipelinePath) }
}
$Common=@('--data-root',$DataRoot,'--preset',$Preset)
if ($Reference) { $Common+=@('--reference',$Reference) }
if ($V1Root) { $Common+=@('--v1-root',$V1Root) }
if ($TargetsCsv) { $Common+=@('--targets-csv',$TargetsCsv) }
function Invoke-Step([string]$Script,[string[]]$Arguments) {
    $Pipeline.stage=$Script+' '+($Arguments -join ' ')
    Write-Pipeline
    & $Python (Join-Path $PSScriptRoot $Script) @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Passaggio fallito: $Script (exit $LASTEXITCODE). Vedere status.json; nessuna fase successiva avviata." }
}
try {
    Write-Pipeline
    if ($DownloadTargets) { Invoke-Step 'patch_builder.py' (@('targets')+$Common) }
    if ($FetchV1) { Invoke-Step 'patch_builder.py' (@('fetch')+$Common) }
    Invoke-Step 'patch_builder.py' (@('build')+$Common)
    Invoke-Step 'preflight.py' ($Common+@('--require-training'))
    Invoke-Step 'train.py' ($Common+@('--profile'))
    $Train=$Common+@('--run-name',$RunName,'--phase','all')
    if ($Resume) { $Train+=@('--resume') }
    Invoke-Step 'train.py' $Train
    Invoke-Step 'evaluate.py' (@('oof')+$Common+@('--run-name',$RunName))
    Invoke-Step 'evaluate.py' (@('liguria')+$Common+@('--run-name',$RunName))
    $Pipeline.status='complete';$Pipeline.finished_utc=[DateTime]::UtcNow.ToString('o');Write-Pipeline
    Write-Host "Run terminato e valutato: $DataRoot\v1_spatial_msa\runs\$Preset\$RunName"
} catch {
    $Pipeline.status='failed';$Pipeline.error=$_.Exception.Message;$Pipeline.finished_utc=[DateTime]::UtcNow.ToString('o');Write-Pipeline
    throw
} finally { $WorkflowHandle.Dispose() }
