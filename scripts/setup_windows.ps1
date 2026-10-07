# Scopo: installare l'ambiente Windows Python 3.12 per TESSERA V2 e RTX 5080.
# Fasi: verifica Python, crea .venv, installa wheel ufficiali e package, esegue doctor.
# Input: requirements-lock.txt, pyproject.toml; output: .venv e doctor.json.
# Parametri: -DataRoot 'E:\BioMAP', -CpuOnly per test; GPU CUDA per esperimenti reali.
# Esempio: powershell -ExecutionPolicy Bypass -File scripts\setup_windows.ps1 -DataRoot E:\BioMAP.
# Risorse: Internet, Git/Python gia installati, driver NVIDIA recente; nessun CUDA Toolkit necessario.
# Ripresa: rieseguibile; errori dei comandi nativi interrompono immediatamente l'installazione.
param([string]$DataRoot = '.\local_data', [switch]$CpuOnly)
$ErrorActionPreference = 'Stop'
Set-Location (Split-Path $PSScriptRoot -Parent)
function Invoke-Checked {
    param([string]$Program, [string[]]$Arguments)
    & $Program @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Comando fallito: $Program (exit $LASTEXITCODE)" }
}
Invoke-Checked 'py' @('-3.12', '-c', 'import sys; assert sys.version_info[:2] == (3,12)')
Invoke-Checked 'py' @('-3.12', '-m', 'venv', '.venv')
$TaskPython = Join-Path $PWD '.venv\Scripts\python.exe'
Invoke-Checked $TaskPython @('-m', 'pip', 'install', 'pip==25.1.1')
$TorchIndex = if ($CpuOnly) { 'https://download.pytorch.org/whl/cpu' } else { 'https://download.pytorch.org/whl/cu128' }
Invoke-Checked $TaskPython @('-m', 'pip', 'install', 'torch==2.10.0', '--index-url', $TorchIndex)
Invoke-Checked $TaskPython @('-m', 'pip', 'install', '-r', 'requirements-lock.txt')
Invoke-Checked $TaskPython @('-m', 'pip', 'install', '-e', '.', '--no-deps', '--no-build-isolation')
Invoke-Checked $TaskPython @('-m', 'pip', 'check')
$TaskDevice = if ($CpuOnly) { 'cpu' } else { 'cuda:0' }
Invoke-Checked $TaskPython @('scripts\probe_msa.py', 'doctor', '--data-root', $DataRoot, '--device', $TaskDevice)
Write-Host 'Ambiente pronto. Seguire README: pesi -> prepare smoke -> probing -> fine-tuning.'
