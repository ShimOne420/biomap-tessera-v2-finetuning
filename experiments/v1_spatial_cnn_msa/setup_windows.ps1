# SCOPO: installare l'ambiente Python 3.12 privato dell'esperimento V1 spatial su Windows.
# FASI: controlla Python -> crea .venv locale -> Torch CUDA 12.8 -> lock del repository -> pip check.
# INPUT: repository su branch experiment/v1-spatial-cnn-msa; OUTPUT: solo questa directory/.venv.
# PARAMETRI: -CpuOnly per test; -CheckGpu esegue un piccolo forward/backward soltanto dopo il run V2.
# ESEMPIO: powershell -ExecutionPolicy Bypass -File experiments\v1_spatial_cnn_msa\setup_windows.ps1 -CheckGpu.
# RISORSE: rete per le wheel, alcuni GB; nessun dato/encoder/peso GeoTESSERA viene scaricato.
# RIPRESA: ripetibile; nessuna attivazione richiesta, nessuna modifica agli ambienti condivisi o V2.
param([switch]$CpuOnly,[switch]$CheckGpu)
$ErrorActionPreference='Stop'
if ($CpuOnly -and $CheckGpu) { throw '-CpuOnly e -CheckGpu non sono combinabili' }
$Repo=(Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$Python=Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path $Python)) {
    & py -3.12 -m venv (Join-Path $PSScriptRoot '.venv')
    if ($LASTEXITCODE -ne 0) { throw 'Installare Python 3.12 x64 con py launcher' }
}
$Index=if ($CpuOnly) {'https://download.pytorch.org/whl/cpu'} else {'https://download.pytorch.org/whl/cu128'}
& $Python -m pip install 'torch==2.10.0' --index-url $Index
if ($LASTEXITCODE -ne 0) { throw 'Installazione PyTorch fallita' }
& $Python -m pip install -r (Join-Path $Repo 'requirements-lock.txt')
if ($LASTEXITCODE -ne 0) { throw 'Installazione lock dipendenze fallita' }
& $Python -m pip check
if ($LASTEXITCODE -ne 0) { throw 'Dipendenze incoerenti' }
$Probe=@'
import json,sys,torch,platform
from pathlib import Path
sys.path.insert(0,sys.argv[1])
record={'python':platform.python_version(),'torch':torch.__version__,'cuda_build':torch.version.cuda,'cuda_available':torch.cuda.is_available(),'encoder_loaded':False}
if sys.argv[2]=='check':
    if not torch.cuda.is_available(): raise RuntimeError('CUDA non disponibile: verificare driver NVIDIA')
    from model import SpatialCNN
    m=SpatialCNN({'conv_channels':[64,32],'hidden':64,'dropout':0.1,'pooling':'mean','output':'linear'}).cuda()
    mask=torch.ones(2,1,32,24,device='cuda',dtype=torch.bool)
    b={'features':torch.randn(2,128,32,24,device='cuda'),'mask':mask,'owners':torch.tensor([0,1],device='cuda'),'weights':torch.tensor([768.,768.],device='cuda'),'target':torch.tensor([.4,.6],device='cuda')}
    loss=torch.nn.functional.mse_loss(m(b),b['target']);loss.backward();torch.cuda.synchronize()
    record.update(gpu=torch.cuda.get_device_name(0),forward_backward='passed',peak_vram_bytes=torch.cuda.max_memory_allocated())
Path(sys.argv[1],'.venv','environment.json').write_text(json.dumps(record,indent=2))
print(json.dumps(record,indent=2))
'@
$Mode=if ($CheckGpu) {'check'} else {'no-gpu-work'}
& $Python -c $Probe $PSScriptRoot $Mode
if ($LASTEXITCODE -ne 0) { throw 'Controllo ambiente fallito' }
Write-Host "Ambiente privato pronto: $Python"
Write-Host 'Nessun run scientifico è stato avviato.'
