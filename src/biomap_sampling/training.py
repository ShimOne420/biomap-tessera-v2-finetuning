# Scopo: fit della sola Spatial CNN originale, early stopping interno e refit da inizializzazione nuova.
# Fasi: disgiunzione geografica -> normalizzatore training-only -> AdamW/MSE -> checkpoint -> predizioni dense.
# Input: source frozen/cache e frame espliciti; output: epoche immutabili, RNG, normalizzatore e risorse.
# Parametri: seed, batch/accumulo, 50 epoche/patience8, deadline; uso fit(...), refit_epochs(2,3)==3.
# Risorse: CPU/CUDA per la testa, worker spawn; nessun encoder o download. Ripresa al confine di epoca.
import json
from pathlib import Path
import random
import time
import uuid
import numpy as np
import torch
from torch.utils.data import DataLoader
from biomap_spatial.batching import CellDataset,collate_cells
from biomap_spatial.model import SpatialCNN
from biomap_spatial.contracts import ModelConfig
from biomap_spatial.normalization import fit_normalizer
from biomap_spatial.engine import seed_all,device_batch
from .metrics import metrics
from .state import *


def refit_epochs(first,second):
    if type(first) is not int or type(second) is not int or min(first,second)<1:raise ValueError('Epoche intere positive, numerate da uno')
    return (first+second+1)//2


def split_guard(training,validation=None,external=None):
    for f in [x for x in [training,validation,external] if x is not None]:
        if f.empty or not f.partition.eq('train').all() or f.aoi_id.eq('liguria_levante').any() or f.cell_id.duplicated().any():raise ValueError('Split non valido/Liguria')
    frames=[x for x in [training,validation,external] if x is not None]
    for i,a in enumerate(frames):
        for b in frames[i+1:]:
            if set(a.cell_id)&set(b.cell_id) or set(a.aoi_id)&set(b.aoi_id):raise ValueError('Leakage geografico')


def loader(source,frame,norm,cfg,shuffle=False):
    n=cfg['training']['num_workers']
    return DataLoader(CellDataset(source,frame,norm,True),batch_size=cfg['training']['batch_cells'],shuffle=shuffle,
        num_workers=n,collate_fn=collate_cells,**({'multiprocessing_context':'spawn','timeout':cfg['training'].get('worker_timeout_seconds',120)} if n else {}))


def predict(model,source,frame,norm,cfg,deadline=None,progress=None,phase='predict'):
    model.eval();result=[];device=torch.device(cfg['execution']['device']);started=time.perf_counter()
    with torch.no_grad():
        for batch in loader(source,frame,norm,cfg):
            check_deadline(deadline);result.extend(model(device_batch(batch,device)).float().cpu().tolist())
            if progress:
                rate=len(result)/max(time.perf_counter()-started,1e-9)
                progress.update(phase=phase,cells_done=len(result),cells_total=len(frame),cells_per_second=rate,eta_seconds=(len(frame)-len(result))/rate)
    columns=[k for k in ['cell_id','aoi_id','partition','cv_fold','target','lon','lat','reference_index','n_valid','valid_fraction'] if k in frame]
    out=frame[columns].copy();out['prediction']=result
    if not np.isfinite(result).all():raise ValueError('Predizioni non finite')
    return out


def fit(train_source,dense_source,training,validation,cfg,folder,tag,seed,resume=False,epochs=None,deadline=None,progress=None):
    split_guard(training,validation);folder=Path(folder);own_id={'identity':tag,'kind':'fit','seed':seed}
    marker=folder/'fit.json'
    if marker.exists():
        if not resume:raise ValueError('Fit esistente: --resume')
        if json.loads(marker.read_text(encoding='utf-8'))!=own_id:raise ValueError('Fit incompatibile')
    else:
        if folder.exists() and any(folder.iterdir()):raise ValueError('Cartella fit non posseduta')
        folder.mkdir(parents=True,exist_ok=True);save_json(marker,own_id)
    check_deadline(deadline);device=torch.device(cfg['execution']['device']);seed_all(seed,device)
    # Always a NEW model. A resume subsequently restores only this exact fit, never an inner checkpoint for refit.
    model=SpatialCNN(ModelConfig()).to(device);initial=signature({k:digest_tensor(v) for k,v in model.state_dict().items()})
    norm=fit_normalizer(training,'pixel_weighted_standard');t=cfg['training'];limit=epochs or t['epochs']
    optimizer=torch.optim.AdamW(model.parameters(),lr=t['learning_rate'],weight_decay=t['weight_decay'])
    history=[];start=0;best=float('inf');best_epoch=0;stale=0
    complete=sorted(folder.glob('epoch_*.json'),key=lambda p:(int(p.stem.split('_')[1]),p.stat().st_mtime_ns),reverse=True)
    if resume and complete:
        saved=load_checkpoint(complete[0].with_suffix('.pt'),tag);model.load_state_dict(saved['model'],strict=True);optimizer.load_state_dict(saved['optimizer'])
        if saved['initialization']!=initial or saved['normalizer']['training_ids_sha256']!=norm['training_ids_sha256']:raise ValueError('Inizializzazione/training IDs incompatibili')
        norm=saved['normalizer'];start=saved['epoch'];best=saved['best'];best_epoch=saved['best_epoch'];stale=saved['stale'];history=saved['history']
        random.setstate(saved['rng_python']);np.random.set_state(saved['rng_numpy']);torch.set_rng_state(saved['rng_torch'])
        if device.type=='cuda':torch.cuda.set_rng_state_all(saved['rng_cuda'])
    if not (folder/'initialization.json').exists():save_json(folder/'initialization.json',{'seed':seed,'sha':initial,'source':'fresh_seeded_initialization',
        'training_ids_sha256':norm['training_ids_sha256'],'training_cells':len(training),'validation_cells':len(validation) if validation is not None else 0})
    started=time.perf_counter()
    if device.type=='cuda':torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(start,limit):
        if validation is not None and stale>=t['patience']:break
        check_deadline(deadline);model.train();seen=0;loss_sum=0.;pending=0;tick=time.perf_counter();optimizer.zero_grad(set_to_none=True)
        batches=loader(train_source,training,norm,cfg,True)
        for step,batch in enumerate(batches,1):
            check_deadline(deadline);batch=device_batch(batch,device);n=len(batch['cell_ids']);goal=t['accumulate_cells']
            errors=(model(batch)-batch['target']).square()
            if not torch.isfinite(errors).all():raise RuntimeError('Loss non finita')
            (errors.sum()/goal).backward();seen+=n;pending+=n;loss_sum+=float(errors.detach().sum())
            if pending>=goal or step==len(batches):
                for p in model.parameters():
                    if p.grad is not None:p.grad.mul_(goal/pending)
                torch.nn.utils.clip_grad_norm_(model.parameters(),t['clip_grad'],error_if_nonfinite=True)
                optimizer.step();optimizer.zero_grad(set_to_none=True);pending=0
            if progress:
                rate=seen/max(time.perf_counter()-tick,1e-9);progress.update(phase='training',epoch=epoch+1,cells_done=seen,cells_total=len(training),
                    cells_per_second=rate,eta_current_epoch_seconds=(len(training)-seen)/rate,eta_seconds=(len(training)-seen)/rate)
        train_seconds=time.perf_counter()-tick;validation_started=time.perf_counter()
        score=metrics(predict(model,dense_source,validation,norm,cfg,deadline,progress,'internal_validation'))['macro_aoi_rmse'] if validation is not None else None
        validation_seconds=time.perf_counter()-validation_started if validation is not None else 0.
        improved=score is None or score<best
        if improved:best=score if score is not None else 0.;best_epoch=epoch+1;stale=0
        else:stale+=1
        history.append({'epoch':epoch+1,'train_mse':loss_sum/seen,'validation_macro_aoi_rmse':score,'epoch_seconds':time.perf_counter()-tick,
            'training_seconds':train_seconds,'validation_seconds':validation_seconds})
        state={'identity':tag,'epoch':epoch+1,'model':model.state_dict(),'optimizer':optimizer.state_dict(),'normalizer':norm,
            'best':best,'best_epoch':best_epoch,'stale':stale,'history':history,'initialization':initial,
            'rng_python':random.getstate(),'rng_numpy':np.random.get_state(),'rng_torch':torch.get_rng_state(),
            'rng_cuda':torch.cuda.get_rng_state_all() if device.type=='cuda' else None}
        name=f'epoch_{epoch+1:04d}.pt'
        if (folder/name).exists():name=f'epoch_{epoch+1:04d}_{uuid.uuid4().hex[:8]}.pt'
        disk_budget(folder,cfg['execution']['reserve_gib'],8*1024**2);save_checkpoint(folder/name,state)
        save_json(folder/'history.json',history)
        print(f'{folder.name}: epoch {epoch+1}, train MSE {loss_sum/seen:.6f}, internal macro RMSE {score}',flush=True)
    if not best_epoch:raise ValueError('Nessuna epoca completa')
    # The immutable epoch markers are authoritative, also after a crash before pointer publication.
    records=sorted(folder.glob(f'epoch_{best_epoch:04d}*.json'))
    best_path=records[-1].with_suffix('.pt');saved=load_checkpoint(best_path,tag);model.load_state_dict(saved['model'],strict=True)
    save_json(folder/'best.json',{'file':best_path.name,'best_epoch':best_epoch,'identity':tag,'score':best})
    resources={'seconds_this_invocation':time.perf_counter()-started,'gpu_peak_bytes':torch.cuda.max_memory_allocated(device) if device.type=='cuda' else None}
    from biomap_v2.campaign.reporting import peak_ram
    resources['process_peak_rss']=peak_ram();resources['ram_note']='process lifetime peak, not isolated incremental fit RAM; DataLoader child memory is not included'
    save_json(folder/'resources.json',resources)
    return model,norm,best_epoch


def digest_tensor(value):
    import hashlib
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
