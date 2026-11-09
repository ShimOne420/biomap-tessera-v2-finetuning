# SCOPO: verificare ambiente e percorsi E2 senza eseguire acquisizione o training.
# Un piccolo forward/backward verifica solo il device richiesto, non la fattibilità completa.
# Output: work-root/environment.json; il doctor legacy non viene eseguito sui dati attivi.
import platform
import shutil
from pathlib import Path
import torch
from biomap_v2.common import save_json
from biomap_ml.settings import parser, command_context


# Usare --device cpu durante il probing; CUDA soltanto dopo la sua conclusione.
def main():
    p = parser('E2: controllo ambiente e isolamento sorgente/lavoro')
    args = p.parse_args()
    with command_context(args) as (cfg, _):
        device = torch.device(cfg['runtime']['device'])
        if device.type == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('CUDA non disponibile; --device cpu solo per controlli tecnici')
        if not (Path(cfg['source'])/'cells.csv').is_file():
            raise FileNotFoundError('cells.csv assente nella source-data-root')
        tensor = torch.ones((2, 2), device=device, requires_grad=True)
        tensor.square().mean().backward()
        info = {'python': platform.python_version(), 'platform': platform.platform(), 'torch': torch.__version__,
                'device': str(device), 'cuda_available': torch.cuda.is_available(),
                'source_read_only_by_design': True, 'work_root': cfg['work'],
                'work_free_gib': shutil.disk_usage(cfg['work']).free/1024**3,
                'full_cuda_training_verified': False}
        if device.type == 'cuda':
            info.update(gpu=torch.cuda.get_device_name(device), vram_gib=torch.cuda.get_device_properties(device).total_memory/1024**3)
        save_json(Path(cfg['work'])/'environment.json', info)
        print(info)


if __name__ == '__main__':
    main()
