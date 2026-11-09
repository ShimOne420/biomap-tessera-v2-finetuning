# SCOPO: addestrare solo la CNN E2 con accumulo gradienti e checkpoint riprendibili.
# INPUT: cache puntuale, celle training/validation del fold, configurazione congelata.
# OUTPUT: warmup.pt oppure best.pt/last.pt, optimizer/scaler/RNG, storia e tempi.
# Il motore non carica l'encoder: i suoi pesi restano quelli del checkpoint verificato.
# OOM: ripristina l'inizio epoca da disco e riduce il microbatch, non dati o batch effettivo.
from contextlib import nullcontext
from pathlib import Path
import random
import time
import numpy as np
import pandas as pd
import torch
from biomap_v2.common import geographic_rmse
from biomap_v2.experiments import seed_all, rng_state, restore_rng
from .cache import PointDataset
from .settings import identity, save_csv, save_torch, work_budget
from .spatial import SpatialCNN


# AMP è limitata alla CNN; la cache frozen resta FP32 e non viene rigenerata.
def autocast(cfg):
    device = torch.device(cfg['runtime']['device'])
    if device.type == 'cuda' and cfg['runtime']['mixed_precision']:
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        return torch.autocast('cuda', dtype=dtype)
    return nullcontext()


# Nessun multiprocessing: non ricarica i file della sorgente attiva e limita la RAM.
def batch(dataset, indices, device):
    records = [dataset[int(i)] for i in indices]
    return tuple(torch.stack([r[col] for r in records]).to(device) for col in range(4))


# Previsioni in ordine identico al DataFrame: necessarie per metriche per AOI e raster.
def predict(head, dataset, cfg, microbatch=None):
    head.eval()
    device = cfg['runtime']['device']
    size = microbatch or cfg['training']['microbatch']
    values = []
    with torch.no_grad():
        start = 0
        while start < len(dataset):
            try:
                x, mask, weights, _ = batch(dataset, range(start, min(start+size, len(dataset))), device)
                with autocast(cfg):
                    values.extend(head(x, mask, weights).float().cpu().tolist())
                start += size
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if size == 1:
                    raise RuntimeError('OOM in valutazione anche con una cella; nessuna predizione parziale salvata') from None
                size = max(1, size//2)
    return np.asarray(values)


# Un ordine/target/split differente rende incompatibile il resume anche a numerosità uguale.
def fit_identity(cfg, training, validation, audit, phase, epochs, initial_sha):
    cols = ['cell_id', 'series_sha256', 'target', 'aoi_id', 'cv_fold']
    payload = {'dataset_id': audit['fingerprint']['dataset_id'], 'training': cfg['training'],
               'runtime': cfg['runtime'], 'phase': phase, 'epochs': epochs, 'initial_sha': initial_sha,
               'train': training[cols].to_dict('records'),
               'validation': validation[cols].to_dict('records') if len(validation) else []}
    return identity(payload)


# Un fit è una singola fase: warmup, continuazione E2 nel fold oppure refit fisso.
def fit(cfg, training, validation, audit, out, phase, epochs, initial=None, resume=False):
    from biomap_v2.common import digest
    seed_all(cfg['training']['seed'])
    torch.set_num_threads(cfg['runtime']['threads'])
    device = torch.device(cfg['runtime']['device'])
    head = SpatialCNN().to(device)
    if initial:
        head.load_state_dict(torch.load(initial, map_location='cpu', weights_only=False)['head'])
    optimizer = torch.optim.AdamW(head.parameters(), lr=cfg['training']['learning_rate'],
                                 weight_decay=cfg['training']['weight_decay'])
    scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda' and cfg['runtime']['mixed_precision'] and not torch.cuda.is_bf16_supported())
    fingerprint = fit_identity(cfg, training, validation, audit, phase, epochs, digest(initial) if initial else None)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    history, first, stale, best, best_epoch, step = [], 0, 0, float('inf'), None, 0
    micro = cfg['training']['microbatch']
    last = out/'last.pt'
    if last.exists():
        if not resume:
            raise ValueError('Fit già esistente: usare --resume sulla stessa run-root')
        payload = torch.load(last, map_location='cpu', weights_only=False)
        if payload['fingerprint'] != fingerprint:
            raise ValueError('Resume incompatibile: configurazione, dati o partizioni cambiati')
        head.load_state_dict(payload['head'])
        optimizer.load_state_dict(payload['optimizer'])
        scaler.load_state_dict(payload['scaler'])
        restore_rng(payload['rng'])
        history, first, stale, best = payload['history'], payload['epoch']+1, payload['stale'], payload['best']
        best_epoch, micro, step = payload['best_epoch'], payload['microbatch'], payload['global_step']
        if payload['complete']:
            return out/('best.pt' if len(validation) and phase != 'warmup' else 'last.pt')
    train_set = PointDataset(cfg, training, audit)
    val_set = PointDataset(cfg, validation, audit) if len(validation) else None
    effective = cfg['training']['effective_batch']
    started = time.perf_counter()

    # Stato completo a fine epoca; E2 fa riferimento all'encoder immutato senza duplicarne 84 MB.
    def checkpoint(epoch, complete=False):
        return {'head': head.state_dict(), 'optimizer': optimizer.state_dict(), 'scaler': scaler.state_dict(),
                'rng': rng_state(), 'epoch': epoch, 'global_step': step, 'best': best, 'best_epoch': best_epoch,
                'stale': stale, 'history': history, 'microbatch': micro, 'fingerprint': fingerprint,
                'complete': complete, 'experiment': 'E2', 'encoder_changed': False,
                'encoder_sha256': audit['fingerprint']['checkpoint_sha256'], 'dataset_id': audit['fingerprint']['dataset_id']}
    try:
        for epoch in range(first, epochs):
            # Salvataggio CPU/disco, mai deepcopy dell'intero modello CUDA per gestire un OOM.
            recovery = out/'epoch_start.pt'
            work_budget(cfg, 2*1024**2)
            save_torch(recovery, checkpoint(epoch-1))
            while True:
                epoch_start = time.perf_counter()
                order = np.random.permutation(len(train_set))
                total = 0.
                head.train()
                try:
                    for start in range(0, len(order), effective):
                        window = order[start:start+effective]
                        optimizer.zero_grad(set_to_none=True)
                        for pos in range(0, len(window), micro):
                            x, mask, weights, target = batch(train_set, window[pos:pos+micro], device)
                            with autocast(cfg):
                                output = head(x, mask, weights)
                                loss = (output.float()-target).square().sum()/len(window)
                            if not torch.isfinite(loss):
                                raise RuntimeError('Loss non finita')
                            scaler.scale(loss).backward()
                            total += float(loss.detach())*len(window)
                        scaler.unscale_(optimizer)
                        grad = torch.nn.utils.clip_grad_norm_(head.parameters(), cfg['training']['clip_grad'])
                        if not torch.isfinite(grad):
                            raise RuntimeError('Gradienti non finiti')
                        scaler.step(optimizer)
                        scaler.update()
                        step += 1
                    break
                except torch.cuda.OutOfMemoryError:
                    # Ripristina aggiornamenti, optimizer e RNG anche se l'OOM avviene a metà epoca.
                    optimizer.zero_grad(set_to_none=True)
                    recovered = torch.load(recovery, map_location='cpu', weights_only=False)
                    head.load_state_dict(recovered['head'])
                    optimizer.load_state_dict(recovered['optimizer'])
                    scaler.load_state_dict(recovered['scaler'])
                    restore_rng(recovered['rng'])
                    step = recovered['global_step']
                    torch.cuda.empty_cache()
                    if micro == 1:
                        raise RuntimeError('OOM con microbatch minimo: fermare e profilare; nessun punto/cella rimosso') from None
                    micro = max(1, micro//2)
                    print(f'OOM: ripeto epoca {epoch+1} con microbatch={micro}, batch effettivo={effective}', flush=True)
            score = None
            if val_set is not None:
                prediction = predict(head, val_set, cfg, micro)
                score = geographic_rmse(validation, prediction, grouping='aoi')
            improved = score is not None and score < best-cfg['training']['min_delta']
            if improved:
                best, best_epoch, stale = score, epoch+1, 0
            elif phase != 'warmup' and score is not None:
                stale += 1
            history.append({'epoch': epoch+1, 'train_mse': total/len(train_set), 'validation_macro_aoi_rmse': score,
                            'microbatch': micro, 'effective_batch': effective, 'epoch_seconds': time.perf_counter()-epoch_start})
            done = epoch+1 == epochs or (phase != 'warmup' and val_set is not None and stale >= cfg['training']['patience'])
            payload = checkpoint(epoch, done)
            if improved:
                save_torch(out/'best.pt', payload)
            save_torch(last, payload)
            save_csv(out/'history.csv', pd.DataFrame(history))
            label = f'{score:.6f}' if score is not None else 'refit senza validation'
            print(f'{out.name}/{phase} epoca {epoch+1}/{epochs}, RMSE macro-AOI={label}', flush=True)
            if done:
                break
        if not history:
            raise ValueError('Nessuna epoca completata')
        return out/('best.pt' if val_set is not None and phase != 'warmup' else 'last.pt')
    finally:
        train_set.close()
        if val_set is not None:
            val_set.close()
