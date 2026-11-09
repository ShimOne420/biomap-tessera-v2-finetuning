# SCOPO: misurare piccoli batch E2 reali dopo la conclusione del probing universitario.
# INPUT: cache completa; OUTPUT: profilo tempi/picco memoria nella work-root, non un run scientifico.
# Misura la CNN e la lettura cache. Il costo encoder è separato e registrato durante prepare --stage cache.
import time
from pathlib import Path
import numpy as np
import torch
from biomap_v2.common import save_json
from biomap_ml.settings import parser, command_context
from biomap_ml.source import indexed
from biomap_ml.cache import PointDataset
from biomap_ml.engine import batch, autocast
from biomap_ml.spatial import SpatialCNN


# Campioni distribuiti nelle AOI di sviluppo; Liguria non entra nel profiling supervisionato.
def main():
    p = parser('E2: profiling breve della sola Frozen CNN, senza campagna automatica')
    p.add_argument('--steps', type=int, default=20)
    args = p.parse_args()
    if not 3 <= args.steps <= 100:
        p.error('--steps deve essere fra 3 e 100')
    with command_context(args) as (cfg, _):
        frame, audit = indexed(cfg)
        development = frame[frame.partition == 'train'].groupby('aoi_id', group_keys=False).head(32).reset_index(drop=True)
        dataset = PointDataset(cfg, development, audit)
        torch.set_num_threads(cfg['runtime']['threads'])
        device = torch.device(cfg['runtime']['device'])
        model = SpatialCNN().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['training']['learning_rate'])
        scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda' and cfg['runtime']['mixed_precision'] and not torch.cuda.is_bf16_supported())
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device)
        timings = []
        try:
            for step in range(args.steps+2):
                if device.type == 'cuda':
                    torch.cuda.synchronize(device)
                start = time.perf_counter()
                idx = (np.arange(cfg['training']['microbatch'])+step*cfg['training']['microbatch']) % len(dataset)
                x, mask, weights, target = batch(dataset, idx, device)
                optimizer.zero_grad(set_to_none=True)
                with autocast(cfg):
                    loss = (model(x, mask, weights).float()-target).square().mean()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                if device.type == 'cuda':
                    torch.cuda.synchronize(device)
                if step >= 2:
                    timings.append(time.perf_counter()-start)
            seconds_per_cell = float(np.median(timings))/cfg['training']['microbatch']
            ndev = int((frame.partition == 'train').sum())
            # Tre training fold (2N cumulati) più refit (N); non include validation, checkpoint e cache encoder.
            training_cell_epochs = 3*ndev*(cfg['training']['warmup_epochs']+cfg['training']['epochs'])
            info = {'scope': 'E2_cached_head_training_only', 'steps': args.steps,
                    'median_microbatch_seconds': float(np.median(timings)), 'seconds_per_cell': seconds_per_cell,
                    'training_only_upper_epoch_budget_hours': training_cell_epochs*seconds_per_cell/3600,
                    'estimate_excludes': ['encoder_cache_generation', 'validation', 'checkpoint_writes', 'early_stopping_effect'],
                    'trainable_parameters': sum(p.numel() for p in model.parameters()), 'device': str(device)}
            if device.type == 'cuda':
                info.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(device), peak_reserved_bytes=torch.cuda.max_memory_reserved(device))
            save_json(Path(cfg['work'])/'profile_E2.json', info)
            print(info)
        finally:
            dataset.close()


if __name__ == '__main__':
    main()
