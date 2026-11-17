# Scopo: training online parziale/frozen con loss per cella, CV geografica e Liguria solo dopo refit.
# Fasi: gate decisioni/input -> normalizzazione training-only -> optimizer gruppi -> checkpoint epoca -> report.
# Input: nuovi raw verificati e config esplicita; output: checkpoint, predizioni, metriche, tempi e progress.
# Parametri: LR encoder/head, batch/epoche, precisione, last_n, aggregazione; esempio train --phase cv.
# Risorse: CPU/CUDA, worker spawn con default zero; nessuna rete o modifica degli esperimenti esistenti.
# Ripresa: checkpoint immutabili per epoca, marker ultimo, RNG/optimizer/scaler e normalizzazione inclusi.
import json
from pathlib import Path
import random
import time
import uuid
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from biomap_spatial.engine import seed_all,metrics_report,atomic_parquet
from .encoder import load_medium,PartialMedium,FineTuneModel
from .data import CellDataset,collate_cells
from .config import require_ready
from .io import checked_roots,claim_root,signature,save_json,save_checkpoint,load_checkpoint,Progress,code_identity,git_commit,disk_budget
from .validation import require_input_proof,changed_parameters


def loader(dataset,cfg,shuffle=False):
    n=cfg['training']['num_workers']
    return DataLoader(dataset,batch_size=cfg['training']['batch_size'],shuffle=shuffle,collate_fn=collate_cells,
        num_workers=n,**({'multiprocessing_context':'spawn'} if n else {}))


def build_model(cfg):
    student,_=load_medium(cfg['encoder']['checkpoint'])
    encoder=PartialMedium(student,cfg['finetuning']['last_n_blocks'],cfg['finetuning']['train_output_projection'],
        cfg['finetuning']['train_normalization'],mode=cfg['training']['encoder_mode'],activation_checkpointing=cfg['training']['activation_checkpointing'])
    return FineTuneModel(encoder,cfg['aggregation']['method'],cfg['aggregation']['head']).to(cfg['execution']['device'])


def fit_normalization(model,dataset,cfg):
    if cfg['training']['normalization']=='none':return
    model.eval();count=0;mean=np.zeros(128,np.float64);m2=np.zeros(128,np.float64)
    with torch.no_grad():
        for batch in loader(dataset,cfg):
            for group in batch['groups']:
                x=model.encoder.encode_raw(group['raw'],cfg['training']['pixel_batch']).cpu().numpy().astype(np.float64)
                local_mean=x.mean(0);delta=local_mean-mean;total=count+len(x)
                m2+=((x-local_mean)**2).sum(0)+delta**2*count*len(x)/total
                mean+=delta*len(x)/total;count=total
    if not count:raise ValueError('Normalizzazione senza pixel training')
    std=np.sqrt(m2/count);std=np.maximum(std,1e-6)
    model.embedding_mean.copy_(torch.tensor(mean,device=model.embedding_mean.device,dtype=torch.float32))
    model.embedding_std.copy_(torch.tensor(std,device=model.embedding_std.device,dtype=torch.float32))


def predict(model,dataset,cfg):
    model.eval();values=[]
    with torch.no_grad():
        for batch in loader(dataset,cfg):values.extend(model(batch,cfg['training']['pixel_batch']).float().cpu().tolist())
    if not np.isfinite(values).all():raise ValueError('Predizioni non finite')
    out=dataset.frame.copy();out['prediction']=values;out['model']='predictor';return out


def macro_rmse(predictions):
    if predictions.aoi_id.eq('liguria_levante').any() or not predictions.partition.eq('train').all():raise ValueError('Liguria/test vietata per selezione')
    return float(np.mean([np.sqrt(np.mean((g.prediction-g.target)**2)) for _,g in predictions.groupby('aoi_id')]))


def run_identity(cfg,dataset):
    return signature({'config':cfg,'dataset':dataset.spec,'code':code_identity()})


def registry(cfg,run_id,resume=False):
    require_ready(cfg);checked_roots(cfg)
    if not run_id or Path(run_id).name!=run_id or run_id in {'.','..'}:raise ValueError('run-id deve essere un nome semplice')
    source=CellDataset(cfg['paths']['raw_root']);require_input_proof(cfg,source.spec)
    # Acquisition/config must describe exactly the same selected support, not merely equal cell counts.
    from .data import plan
    _,_,_,expected=plan(cfg)
    if expected!=source.spec['identity']:raise ValueError('Sampling/config non corrispondente al dataset acquisito')
    workspace=Path(cfg['paths']['output_root']);claim_root(workspace,signature({'namespace':'finetune-v2-r1','role':'run_workspace'}),'run_workspace')
    folder=workspace/run_id;identity=run_identity(cfg,source)
    if folder.exists() and not resume:raise ValueError('Run esistente: usare --resume')
    claim_root(folder,identity,'training');disk_budget(folder,cfg['storage']['reserve_gib'],1024**3)
    if not (folder/'effective_config.json').exists():save_json(folder/'effective_config.json',cfg)
    manifest={'identity':identity,'dataset_identity':source.spec['identity'],'support':source.spec['support'],
        'git_commit':git_commit(),'code':code_identity(),'profile':cfg['experiment']['profile'],'normalization':'fixed initial encoder, training pixels only',
        'aggregation':cfg['aggregation']['method'],'seed':cfg['experiment']['seed'],'encoder_mode':cfg['training']['encoder_mode']}
    if not (folder/'manifest.json').exists():save_json(folder/'manifest.json',manifest)
    return source,folder,identity


def fit(cfg,training,validation,folder,identity,resume=False,epoch_count=None,progress=None):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True);device=torch.device(cfg['execution']['device'])
    seed_all(cfg['experiment']['seed'],device);model=build_model(cfg)
    from biomap_v2.common import WEIGHTS_SHA256
    if not (folder/'training_manifest.json').exists():save_json(folder/'training_manifest.json',{
        'identity':identity,'initial_checkpoint_sha256':WEIGHTS_SHA256,'resolved_encoder_policy':model.encoder.policy(),
        'head':cfg['aggregation'],'training':cfg['training'],'training_cells':len(training),
        'training_ids_sha256':signature(training.frame.cell_id.tolist()),'validation_cells':len(validation) if validation is not None else 0,
        'torch':torch.__version__,'seed':cfg['experiment']['seed'],'precision':cfg['training']['precision'],
        'cublas_workspace_config':__import__('os').environ.get('CUBLAS_WORKSPACE_CONFIG')})
    groups=[{'params':[p for p in model.head.parameters() if p.requires_grad],'lr':cfg['training']['learning_rate_head'],'role':'head'}]
    enc=[p for p in model.encoder.parameters() if p.requires_grad]
    if enc:groups.append({'params':enc,'lr':cfg['training']['learning_rate_encoder'],'role':'encoder'})
    optimizer=torch.optim.AdamW(groups)
    precision=cfg['training']['precision'];amp=precision!='fp32'
    if amp and device.type!='cuda':raise ValueError('AMP solo CUDA esplicita')
    if precision=='bfloat16' and not torch.cuda.is_bf16_supported():raise ValueError('BF16 non supportato')
    scaler=torch.amp.GradScaler('cuda',enabled=precision=='float16')
    start=0;best=float('inf');best_epoch=0;stale=0;history=[];pointer=folder/'latest.json'
    complete=sorted(folder.glob('epoch_*.json'),key=lambda p:(int(p.stem.split('_')[1]),p.stat().st_mtime_ns),reverse=True)
    if resume and complete:
        latest=complete[0].with_suffix('.pt');state=load_checkpoint(latest,identity)
        model.load_state_dict(state['model'],strict=True);optimizer.load_state_dict(state['optimizer']);scaler.load_state_dict(state['scaler'])
        start=state['epoch'];best=state['best'];best_epoch=state['best_epoch'];stale=state['stale'];history=state['history']
        random.setstate(state['rng_python']);np.random.set_state(state['rng_numpy']);torch.set_rng_state(state['rng_torch'])
        if device.type=='cuda':torch.cuda.set_rng_state_all(state['rng_cuda'])
        best_files=[p for p in complete if int(p.stem.split('_')[1])==best_epoch]
        if not best_files:raise ValueError('Checkpoint migliore assente')
        best_path=best_files[0].with_suffix('.pt');load_checkpoint(best_path,identity)
        save_json(folder/'best.json',{'file':best_path.name,'epoch':best_epoch,'score':best if validation is not None else None})
        save_json(pointer,{'file':latest.name})
    else:fit_normalization(model,training,cfg)
    epochs=epoch_count or cfg['training']['epochs'];began=time.perf_counter()
    if device.type=='cuda':torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(start,epochs):
        if validation is not None and stale>=cfg['training']['early_stopping']:break
        model.train();loss_sum=0.;seen=0
        for batch in loader(training,cfg,True):
            before={n:p.detach().cpu().clone() for n,p in model.named_parameters()} if not history and seen==0 else None
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device.type,enabled=amp,dtype=torch.float16 if precision=='float16' else torch.bfloat16):
                prediction=model(batch,cfg['training']['pixel_batch']);target=torch.tensor(batch['target'],device=device)
                loss=torch.nn.functional.mse_loss(prediction,target)
            if not torch.isfinite(loss):raise RuntimeError('Loss non finita')
            scaler.scale(loss).backward();scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],cfg['training']['clip_grad'],error_if_nonfinite=True)
            scaler.step(optimizer);scaler.update()
            if before is not None:
                audit=changed_parameters(model,before)
                if cfg['training']['encoder_mode']=='partial':
                    for name in ['s2_backbone.transformer_encoder','s1_backbone.transformer_encoder']:
                        if not any(name in p for p in audit['changed_names']):raise RuntimeError('Suffisso senza aggiornamento reale: '+name)
                save_json(folder/'weight_update_audit.json',audit)
            seen+=len(target);loss_sum+=float(loss.detach())*len(target)
            if progress:
                elapsed=time.perf_counter()-began;done=(epoch-start)*len(training)+seen;remaining=(epochs-epoch-1)*len(training)+len(training)-seen
                progress.update(phase='train',epoch=epoch+1,done=done,total=(epochs-start)*len(training),unit='cell_passes',rate_per_second=done/max(elapsed,1e-9),eta_seconds=remaining*elapsed/max(done,1))
        score=macro_rmse(predict(model,validation,cfg)) if validation is not None else None
        improved=score is None or score<best
        if improved:best=score if score is not None else 0.;best_epoch=epoch+1;stale=0
        else:stale+=1
        history.append({'epoch':epoch+1,'train_mse':loss_sum/seen,'validation_macro_aoi_rmse':score,'seconds':time.perf_counter()-began})
        state={'identity':identity,'epoch':epoch+1,'model':model.state_dict(),'optimizer':optimizer.state_dict(),'scaler':scaler.state_dict(),
            'best':best,'best_epoch':best_epoch,'stale':stale,'history':history,'policy':model.encoder.policy(),
            'rng_python':random.getstate(),'rng_numpy':np.random.get_state(),'rng_torch':torch.get_rng_state(),
            'rng_cuda':torch.cuda.get_rng_state_all() if device.type=='cuda' else None}
        disk_budget(folder,cfg['storage']['reserve_gib'],512*1024**2)
        name=f'epoch_{epoch+1:04d}.pt'
        if (folder/name).exists():name=f'epoch_{epoch+1:04d}_{uuid.uuid4().hex[:8]}.pt'
        save_checkpoint(folder/name,state)
        save_json(pointer,{'file':name});save_json(folder/'history.json',history)
        if improved:save_json(folder/'best.json',{'file':name,'epoch':epoch+1,'score':score})
        print(f'{folder.name}: epoca {epoch+1}, MSE {loss_sum/seen:.6f}, validation={score}',flush=True)
        if validation is not None and stale>=cfg['training']['early_stopping']:break
    best_record=json.loads((folder/'best.json').read_text(encoding='utf-8'))
    best_state=load_checkpoint(folder/best_record['file'],identity);model.load_state_dict(best_state['model'],strict=True)
    save_json(folder/'resources.json',{'seconds_this_session':time.perf_counter()-began,'cuda_peak_bytes':torch.cuda.max_memory_allocated(device) if device.type=='cuda' else None})
    return model,best_record['epoch']


def train(cfg,run_id,phase='cv',resume=False):
    source,folder,identity=registry(cfg,run_id,resume);progress=Progress(folder/'progress.json')
    dev=source.frame[source.frame.partition=='train'];raw=cfg['paths']['raw_root']
    if cfg['experiment']['profile']=='smoke':
        if phase!='pilot':raise ValueError('Smoke consente solo phase=pilot')
    elif phase=='pilot':raise ValueError('Pilot richiede configurazione smoke, nessun subset scientifico implicito')
    try:
        if phase=='pilot':
            model,_=fit(cfg,CellDataset(raw,dev.cell_id),None,folder/'pilot',identity,resume,progress=progress)
            predictions=predict(model,CellDataset(raw,dev.cell_id),cfg);predictions['fold']=0
            dummy=predictions.copy();dummy['prediction']=float(dev.target.mean());dummy['model']='dummy_mean'
            predictions=pd.concat([predictions,dummy],ignore_index=True)
            atomic_parquet(folder/'technical_predictions.parquet',predictions);metrics_report(predictions,folder,'technical')
        elif phase=='cv':
            parts=[];epochs=[]
            for fold in [0,1,2]:
                a=CellDataset(raw,dev[dev.cv_fold!=fold].cell_id);b=CellDataset(raw,dev[dev.cv_fold==fold].cell_id)
                model,ep=fit(cfg,a,b,folder/f'fold_{fold}',identity,resume,progress=progress)
                pred=predict(model,b,cfg);pred['fold']=fold
                dummy=pred.copy();dummy['prediction']=float(a.frame.target.mean());dummy['model']='dummy_mean'
                parts.extend([pred,dummy]);epochs.append(ep)
                del model
            predictions=pd.concat(parts,ignore_index=True);atomic_parquet(folder/'oof_predictions.parquet',predictions)
            metrics_report(predictions,folder,'oof');save_json(folder/'selection.json',{'identity':identity,'criterion':'macro_AOI_RMSE_geographic_3fold',
                'score':macro_rmse(predictions[predictions.model=='predictor']),'best_epochs':epochs,'refit_epochs':int(np.median(epochs)),'liguria_used':False})
        elif phase=='refit':
            selection=json.loads((folder/'selection.json').read_text(encoding='utf-8'))
            if selection['identity']!=identity:raise ValueError('Selezione incompatibile')
            fit(cfg,CellDataset(raw,dev.cell_id),None,folder/'refit',identity,resume,selection['refit_epochs'],progress)
        else:raise ValueError('Fase non riconosciuta')
        progress.update(force=True,phase='complete',eta_seconds=0)
    except BaseException as error:
        progress.update(force=True,phase='interrupted',error=repr(error),eta_seconds=None);raise
    return {'run':run_id,'phase':phase,'status':'TECHNICAL_ONLY' if phase=='pilot' else 'COMPLETE'}


def evaluate(cfg,run_id):
    if cfg['experiment']['profile']!='benchmark':raise ValueError('Liguria disponibile solo nel benchmark completo')
    source,folder,identity=registry(cfg,run_id,True)
    record=json.loads((folder/'refit/best.json').read_text(encoding='utf-8'))
    model=build_model(cfg);model.load_state_dict(load_checkpoint(folder/'refit'/record['file'],identity)['model'])
    test=CellDataset(cfg['paths']['raw_root'],source.frame[source.frame.partition=='test'].cell_id)
    if len(test)!=4991:raise ValueError('Test Liguria incompleto')
    out=predict(model,test,cfg);out['fold']=-1
    dummy=out.copy();dummy['prediction']=float(source.frame[source.frame.partition=='train'].target.mean());dummy['model']='dummy_mean'
    out=pd.concat([out,dummy],ignore_index=True);atomic_parquet(folder/'liguria_predictions.parquet',out)
    metrics_report(out,folder,'liguria');return {'cells':len(test),'status':'EVALUATED_ALREADY_OBSERVED_EXTERNAL_TEST'}
