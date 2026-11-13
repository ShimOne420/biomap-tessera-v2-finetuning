# Scopo: CV geografica e refit riprendibili di CNN/MLP/HistGB sugli stessi pixel V1 congelati.
# Fasi: verifica cache -> scaler train-only -> tre fold AOI -> selezione macro-RMSE -> refit sviluppo.
# Input: cache finalizzata e config; output: checkpoint, log, OOF e selezione in run privato.
# Parametri: --phase cv|refit|all, --run-name, --resume, --profile; nessuna valutazione Liguria nel fit.
# Risorse: GPU solo per teste neurali, CPU HistGB; precisione FP32 di default, no encoder.
# Ripresa: last.pt con optimizer/scaler/RNG e best.pt atomici; identità di dati/config obbligatoria.
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
os.environ.setdefault('OMP_NUM_THREADS','1')
from copy import deepcopy
import csv
import json
import math
import random
from pathlib import Path
import time
import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.ensemble import HistGradientBoostingRegressor
from runtime import common_parser,configuration,cache_path,save_json,signature,exclusive,tracked,Budget,digest,code_identity
from dataset import SpatialDataset,collate_cells,cell_frame,fit_normalizer,metadata,verify_sources,MeanDataset,collate_mean
from model import make_model
from evaluate import macro_rmse,summarize


def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.use_deterministic_algorithms(True)


def run_directory(cfg,name):
    if not name or Path(name).name!=name or any(x in name for x in [':','/','\\']):raise ValueError('run-name deve essere un nome semplice')
    return Path(cfg['work_root'])/'runs'/cfg['preset']/name


def atomic_torch(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.partial')
    with tmp.open('wb') as out:torch.save(value,out);out.flush();os.fsync(out.fileno())
    os.replace(tmp,path)


def identity(cfg):return signature({'cache':metadata(cache_path(cfg))['identity'],'model':cfg['model'],'training':cfg['training'],'code':code_identity()})


def model_path(folder,name):return folder/('model.joblib' if name=='mean_histgb' else 'best.pt')


def load_run(cfg,run):
    spec=json.loads((run/'run.json').read_text())
    if spec['identity']!=identity(cfg):raise ValueError('Dati/config diversi dal run salvato')
    return spec


def loader(path,frame,norm,cfg,shuffle=False,spatial=True):
    ds=SpatialDataset(path,frame,norm,cfg['data']['source_mmap_limit']) if spatial else MeanDataset(frame,norm)
    return DataLoader(ds,batch_size=cfg['training']['batch_cells'],shuffle=shuffle,num_workers=cfg['training']['num_workers'],collate_fn=collate_cells if spatial else collate_mean)


def device_batch(batch,device):return {k:(v.to(device) if torch.is_tensor(v) else v) for k,v in batch.items()}


def infer(model,dl,device):
    model.eval();values=[];ids=[]
    with torch.no_grad():
        for batch in dl:
            out=model(device_batch(batch,device)).float().cpu().numpy()
            if not np.isfinite(out).all():raise ValueError('Predizioni non finite')
            values.extend(out.tolist());ids.extend(batch['cell_ids'])
    return pd.Series(values,index=ids,dtype=float)


def prediction_frame(frame,values,name,fold):
    result=frame[['cell_id','aoi_id','partition','cv_fold','lon','lat','target','n_valid','valid_fraction','patches']].copy()
    result['prediction']=values;result['model']=name;result['fold']=fold
    result['heterogeneity']=[float(np.mean(v)) for v in frame.variance]
    return result


def train_neural(cfg,path,train,val,name,folder,run_identity,progress,epochs,resume):
    folder.mkdir(parents=True,exist_ok=True);device=torch.device(cfg['training']['device'])
    if device.type=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA non disponibile: usare --set training.device=cpu solo per prova CPU')
    seed_all(cfg['training']['seed']);norm=fit_normalizer(train);model=make_model(name,cfg['model']).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['training']['learning_rate'],weight_decay=cfg['training']['weight_decay'])
    amp=bool(cfg['training']['mixed_precision'] and device.type=='cuda');scaler=torch.amp.GradScaler('cuda',enabled=amp)
    start=0;best=float('inf');bad=0;best_epoch=0
    last=folder/'last.pt';best_path=folder/'best.pt'
    if last.exists():
        if not resume:raise FileExistsError('Checkpoint esistente: specificare --resume')
        saved=torch.load(last,map_location=device,weights_only=False)
        if saved['identity']!=run_identity or saved['name']!=name:raise ValueError('Checkpoint incompatibile')
        model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer']);scaler.load_state_dict(saved['scaler'])
        start=saved['epoch'];best=saved['best'];bad=saved['bad'];best_epoch=saved['best_epoch']
        random.setstate(saved['python_rng']);np.random.set_state(saved['numpy_rng']);torch.set_rng_state(saved['torch_rng'].cpu())
        if device.type=='cuda':torch.cuda.set_rng_state_all([v.cpu() for v in saved['cuda_rng']])
    train_loader=loader(path,train,norm,cfg,True,name=='spatial_cnn');val_loader=loader(path,val,norm,cfg,spatial=name=='spatial_cnn') if val is not None else None
    if device.type=='cuda':torch.cuda.reset_peak_memory_stats(device)
    durations=[]
    try:
        for epoch in range(start,epochs):
            if val is not None and bad>=cfg['training']['patience']:break
            began=time.perf_counter();model.train();loss_sum=0.;count=0
            for step,batch in enumerate(train_loader,1):
                batch=device_batch(batch,device);optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type,enabled=amp,dtype=torch.float16):
                    pred=model(batch);loss=nn.functional.mse_loss(pred,batch['target'])
                if not torch.isfinite(loss):raise ValueError('Loss non finita')
                scaler.scale(loss).backward();scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['training']['clip_grad'],error_if_nonfinite=True)
                scaler.step(optimizer);scaler.update();loss_sum+=float(loss.detach())*len(pred);count+=len(pred)
                progress.update(epoch=epoch+1,epochs_max=epochs,batch=step,batches=len(train_loader),current_model=name,device=str(device))
            score=None
            if val is not None:
                values=infer(model,val_loader,device).reindex(val.cell_id).to_numpy();score=macro_rmse(val,values)
                if score<best-1e-12:best=score;bad=0;best_epoch=epoch+1
                else:bad+=1
            else:best_epoch=epoch+1
            seconds=time.perf_counter()-began;durations.append(seconds)
            state={'identity':run_identity,'name':name,'model':model.state_dict(),'optimizer':optimizer.state_dict(),'scaler':scaler.state_dict(),
                'epoch':epoch+1,'best':best,'bad':bad,'best_epoch':best_epoch,'normalizer':norm,'options':cfg['model'],
                'python_rng':random.getstate(),'numpy_rng':np.random.get_state(),'torch_rng':torch.get_rng_state(),
                'cuda_rng':torch.cuda.get_rng_state_all() if device.type=='cuda' else []}
            if val is None or bad==0:atomic_torch(best_path,state)
            atomic_torch(last,state)
            row={'epoch':epoch+1,'train_mse':loss_sum/max(count,1),'validation_macro_aoi_rmse':score,'epoch_seconds':seconds,'best_epoch':best_epoch,
                'gpu_peak_allocated_gib':torch.cuda.max_memory_allocated(device)/1024**3 if device.type=='cuda' else None,
                'gpu_peak_reserved_gib':torch.cuda.max_memory_reserved(device)/1024**3 if device.type=='cuda' else None}
            log=folder/'epochs.csv';new=not log.exists()
            with log.open('a',newline='',encoding='utf-8') as stream:
                writer=csv.DictWriter(stream,fieldnames=list(row))
                if new:writer.writeheader()
                writer.writerow(row)
            progress.update(**row,phase_eta_seconds=(epochs-epoch-1)*float(np.mean(durations)),eta_note='fase: budget massimo; early stopping può anticipare; ETA globale media per job')
            progress.flush();print(f'{name} {folder.name} epoch {epoch+1}/{epochs} train={row["train_mse"]:.6f} val_macro={score}',flush=True)
        if not best_path.exists():raise ValueError('Nessun checkpoint prodotto')
        saved=torch.load(best_path,map_location=device,weights_only=False);model.load_state_dict(saved['model'])
        return model,norm,saved['best_epoch']
    finally:
        train_loader.dataset.close()
        if val_loader:val_loader.dataset.close()


def train(cfg,run,phase,resume,progress):
    path=cache_path(cfg);verify_sources(path);frame=cell_frame(path);dev=frame[frame.partition=='train'].reset_index(drop=True)
    if set(dev.cv_fold)!={0,1,2}:raise ValueError('Servono tutti i fold originali')
    if not len(frame[frame.partition=='test']):raise ValueError('Cache senza supporto Liguria: nessun esperimento appaiato')
    run.mkdir(parents=True,exist_ok=True);ident=identity(cfg);models=list(cfg['training']['models'])
    if not set(models)<={'mean_histgb','mean_mlp','spatial_cnn'}:raise ValueError('Modelli non supportati')
    if (run/'run.json').exists():load_run(cfg,run)
    else:save_json(run/'run.json',{'identity':ident,'config':cfg,'cache':str(path),'cache_report':json.loads((path.parent/'cache_report.json').read_text()),'models':models,'code_sha256':code_identity(),'created_role':'spatial_CV_then_observed_external_Liguria','encoder':'V1.0 frozen precomputed; not loaded'})
    Budget(cfg).check(128*1024**2)
    finished=0;total=(len(models)*3 if phase in {'cv','all'} else 0)+(len(models) if phase in {'refit','all'} else 0)
    progress.update(total=total,unit='model/fold jobs',done=0)
    best_epochs={};all_predictions=[]
    if phase in {'cv','all'}:
        for name in models:
            best_epochs[name]=[]
            for fold in range(3):
                training=dev[dev.cv_fold!=fold].reset_index(drop=True);validation=dev[dev.cv_fold==fold].reset_index(drop=True)
                if set(training.aoi_id)&set(validation.aoi_id):raise ValueError('Leakage AOI')
                folder=run/name/f'fold_{fold}';folder.mkdir(parents=True,exist_ok=True)
                progress.update(phase='cv',fold=fold,current_model=name)
                cached=folder/'predictions.csv';info=folder/'complete.json'
                if cached.exists() and info.exists():
                    mark=json.loads(info.read_text())
                    if mark['identity']!=ident or digest(cached)!=mark['predictions_sha256']:raise ValueError('Predizioni CV alterate')
                    if mark.get('model_sha256')!=digest(model_path(folder,name)):raise ValueError('Pesi CV alterati')
                    preds=pd.read_csv(cached,dtype={'cell_id':str});best_epoch=mark['best_epoch']
                else:
                    norm=fit_normalizer(training)
                    if name=='mean_histgb':
                        model=HistGradientBoostingRegressor(**cfg['training']['histgb']);model.fit(np.stack(training['mean']),training.target)
                        values=model.predict(np.stack(validation['mean']));joblib.dump(model,folder/'model.joblib');best_epoch=0
                    else:
                        model,norm,best_epoch=train_neural(cfg,path,training,validation,name,folder,ident,progress,cfg['training']['epochs'],resume)
                        dl=loader(path,validation,norm,cfg,spatial=name=='spatial_cnn')
                        try:values=infer(model,dl,cfg['training']['device']).reindex(validation.cell_id).to_numpy()
                        finally:dl.dataset.close()
                    preds=prediction_frame(validation,values,name,fold);preds.to_csv(cached,index=False)
                    save_json(info,{'identity':ident,'best_epoch':best_epoch,'predictions_sha256':digest(cached),'model_sha256':digest(model_path(folder,name)),'training_cells':len(training),'validation_cells':len(validation),'training_aois':sorted(set(training.aoi_id)),'validation_aois':sorted(set(validation.aoi_id))})
                all_predictions.append(preds);best_epochs[name].append(best_epoch);finished+=1;progress.update(done=finished);progress.flush()
        # Dummy uses training-fold mean, never the held-out target mean.
        dummy=[]
        for fold in range(3):
            training=dev[dev.cv_fold!=fold];validation=dev[dev.cv_fold==fold]
            dummy.append(prediction_frame(validation,np.full(len(validation),training.target.mean()),'dummy_mean',fold))
        combined=pd.concat(all_predictions+dummy,ignore_index=True);table=summarize(combined,run,'oof')
        rank=table[table.model!='dummy_mean'].sort_values(['macro_aoi_rmse','model'])
        save_json(run/'selection.json',{'criterion':'mean RMSE of original development AOIs, OOF only','selected_model':rank.iloc[0].model,
            'refit_epochs':{name:max(1,int(np.median(vals))) for name,vals in best_epochs.items() if name!='mean_histgb'},'identity':ident,'liguria_used':False})
    if phase in {'refit','all'}:
        selection=json.loads((run/'selection.json').read_text())
        if selection['identity']!=ident:raise ValueError('Selezione di un altro esperimento')
        for name in models:
            folder=run/name/'refit';folder.mkdir(parents=True,exist_ok=True);progress.update(phase='refit',current_model=name,fold='development')
            if not (folder/'complete.json').exists():
                if name=='mean_histgb':
                    model=HistGradientBoostingRegressor(**cfg['training']['histgb']);model.fit(np.stack(dev['mean']),dev.target);joblib.dump(model,folder/'model.joblib')
                else:train_neural(cfg,path,dev,None,name,folder,ident,progress,selection['refit_epochs'][name],resume)
                save_json(folder/'complete.json',{'identity':ident,'n_development':len(dev),'liguria_used':False,'model_sha256':digest(model_path(folder,name))})
            else:
                mark=json.loads((folder/'complete.json').read_text())
                if mark['identity']!=ident or mark.get('model_sha256')!=digest(model_path(folder,name)):raise ValueError('Refit esistente alterato')
            finished+=1;progress.update(done=finished);progress.flush()
        save_json(run/'refit_complete.json',{'identity':ident,'models':models,'n_development':len(dev)})
    return run


def predict_external(cfg,run,manifest):
    path=cache_path(cfg);verify_sources(path);frame=cell_frame(path);test=frame[frame.partition=='test'].reset_index(drop=True);dev=frame[frame.partition=='train']
    ident=manifest['identity'];mark=json.loads((run/'refit_complete.json').read_text())
    if mark['identity']!=ident:raise ValueError('Refit non valido')
    result=[]
    for name in manifest['models']:
        folder=run/name/'refit'
        complete=json.loads((folder/'complete.json').read_text())
        if complete['identity']!=ident or complete.get('model_sha256')!=digest(model_path(folder,name)):raise ValueError('Modello finale diverso/alterato')
        if name=='mean_histgb':values=joblib.load(folder/'model.joblib').predict(np.stack(test['mean']))
        else:
            saved=torch.load(folder/'best.pt',map_location=cfg['training']['device'],weights_only=False)
            if saved['identity']!=ident:raise ValueError('Checkpoint finale diverso')
            model=make_model(name,saved['options']).to(cfg['training']['device']);model.load_state_dict(saved['model'])
            dl=loader(path,test,saved['normalizer'],cfg,spatial=name=='spatial_cnn')
            try:values=infer(model,dl,cfg['training']['device']).reindex(test.cell_id).to_numpy()
            finally:dl.dataset.close()
        result.append(prediction_frame(test,values,name,-1))
    result.append(prediction_frame(test,np.full(len(test),dev.target.mean()),'dummy_mean',-1))
    return summarize(pd.concat(result,ignore_index=True),run,'liguria')


def profile(cfg,path,progress):
    frame=cell_frame(path);dev=frame[frame.partition=='train'].reset_index(drop=True)
    sample=dev.head(max(2,cfg['training']['batch_cells']*2));norm=fit_normalizer(sample)
    device=torch.device(cfg['training']['device']);maximum=cfg['training']['batch_cells'];sizes=[x for x in [1,2,4,8,16,32,64,128] if x<maximum]+[maximum]
    trials=[];seed_all(cfg['training']['seed'])
    for size in sizes:
        local=deepcopy(cfg);local['training']['batch_cells']=size;dl=loader(path,sample,norm,local)
        model=make_model('spatial_cnn',cfg['model']).to(device);optimizer=torch.optim.AdamW(model.parameters(),lr=.001)
        batch=None;loss=None
        times=[];val_times=[];amp=bool(cfg['training']['mixed_precision'] and device.type=='cuda');scaler=torch.amp.GradScaler('cuda',enabled=amp)
        if device.type=='cuda':torch.cuda.reset_peak_memory_stats(device)
        try:
            iterator=iter(dl)
            for step in range(min(2,len(dl))):
                start=time.perf_counter();batch=device_batch(next(iterator),device);model.train();optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type,enabled=amp,dtype=torch.float16):loss=nn.functional.mse_loss(model(batch),batch['target'])
                scaler.scale(loss).backward();scaler.step(optimizer);scaler.update()
                if device.type=='cuda':torch.cuda.synchronize(device)
                times.append(time.perf_counter()-start);model.eval();start=time.perf_counter()
                with torch.no_grad():model(batch)
                if device.type=='cuda':torch.cuda.synchronize(device)
                val_times.append(time.perf_counter()-start)
            trials.append({'batch_cells':size,'status':'passed','train_batch_seconds':float(np.mean(times)),'validation_forward_batch_seconds':float(np.mean(val_times)),
                'peak_allocated_gib':torch.cuda.max_memory_allocated(device)/1024**3 if device.type=='cuda' else None})
        except torch.cuda.OutOfMemoryError:
            trials.append({'batch_cells':size,'status':'oom'});break
        finally:
            dl.dataset.close();del dl,model,optimizer,batch,loss
            if device.type=='cuda':torch.cuda.empty_cache()
    passed=[v for v in trials if v['status']=='passed']
    if not passed:raise RuntimeError('Neppure una cella entra nel device')
    selected=passed[-1];batches=math.ceil(len(dev)/selected['batch_cells']);train_seconds=selected['train_batch_seconds']*batches;val_seconds=selected['validation_forward_batch_seconds']*batches
    epochs=cfg['training']['epochs']
    result={'sample_cells':len(sample),'trials':trials,'largest_tested_batch_cells':selected['batch_cells'],
        'estimated_train_epoch_seconds':train_seconds,'validation_forward_only_epoch_seconds':val_seconds,
        'cnn_max_cv_and_refit_training_seconds':train_seconds*epochs*3,
        'cnn_cv_validation_forward_seconds':val_seconds*epochs,
        'note':'technical sample, not scientific metric. Max budget: 3-fold training (2 full-data equivalents) + refit (1). Includes sampled I/O/train step; validation estimate excludes I/O. Warm-cache/area heterogeneity, all patches, early stopping and MLP/HistGB costs require separate measurement.'}
    save_json(Path(cfg['work_root'])/'profile_training.json',result);print(json.dumps(result,indent=2))
    if selected['batch_cells']<maximum:raise RuntimeError('Ridurre training.batch_cells al massimo misurato e avviare un nuovo run coerente; il training non parte automaticamente')


def main():
    p=common_parser('Teste CNN/MLP V1 frozen con CV geografica originale')
    p.add_argument('--run-name',default='pilot_seed42');p.add_argument('--phase',choices=['cv','refit','all'],default='all')
    p.add_argument('--resume',action='store_true');p.add_argument('--profile',action='store_true')
    args=p.parse_args();cfg=configuration(args)
    with exclusive(cfg),tracked(cfg,'profile' if args.profile else 'training') as progress:
        from preflight import inspect
        ready=inspect(cfg)
        if ready['status']!='READY_FOR_TRAINING':
            reasons=ready['blockers']+[ready['checks'].get('training_device_blocker','Cache spaziale da costruire')]
            raise ValueError('Preflight non pronto al training: '+json.dumps(reasons,ensure_ascii=False))
        if args.profile:profile(cfg,cache_path(cfg),progress)
        else:print(train(cfg,run_directory(cfg,args.run_name),args.phase,args.resume,progress))


if __name__=='__main__':main()
