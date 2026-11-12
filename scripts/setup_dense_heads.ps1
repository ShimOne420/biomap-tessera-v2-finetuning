# Scopo: ambiente Windows SEPARATO per le teste; nessun download Sentinel/encoder e nessun training.
# Usare solo nel nuovo clone/worktree. Non eseguire questo setup nel repository del run attivo.
# Esempio: powershell -ExecutionPolicy Bypass -File scripts\setup_dense_heads.ps1
param([switch]$CpuOnly)
$ErrorActionPreference = 'Stop'
Set-Location (Split-Path $PSScriptRoot -Parent)
function Invoke-Checked {
    param([string]$Program, [string[]]$Arguments)
    & $Program @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Comando fallito: $Program" }
}
Invoke-Checked 'py' @('-3.12','-c','import sys; assert sys.version_info[:2] == (3,12)')
if (-not (Test-Path '.\.venv\Scripts\python.exe')) {
    Invoke-Checked 'py' @('-3.12','-m','venv','.venv')
}
$TaskPython = Join-Path $PWD '.venv\Scripts\python.exe'
$TorchIndex = if ($CpuOnly) { 'https://download.pytorch.org/whl/cpu' } else { 'https://download.pytorch.org/whl/cu128' }
Invoke-Checked $TaskPython @('-m','pip','install','torch==2.10.0','--index-url',$TorchIndex)
Invoke-Checked $TaskPython @('-m','pip','install','-r','requirements-lock.txt')
Invoke-Checked $TaskPython @('-m','pip','install','-e','.','--no-deps','--no-build-isolation')
Invoke-Checked $TaskPython @('-m','pip','check')
Invoke-Checked $TaskPython @('scripts\train_v2_dense_cnn.py','--help')
Write-Host 'Ambiente teste pronto. Nessun encoder caricato, nessun dataset scaricato, nessun training avviato.'
