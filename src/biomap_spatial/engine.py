# Scopo: CV geografica, refit, valutazione esterna e registry della sola testa Spatial CNN/Mean MLP.
# Fasi: preflight dati completo -> device -> normalizzazione training-only -> epoche/checkpoint -> report.
# Input: ExperimentSpec e indice verificato; output: config/manifest/checkpoint/predizioni/metriche private.
# Parametri: phase cv/refit, resume, run_id e limite tecnico di epoche per invocazione; Liguria solo evaluate.
# Risorse: CPU/GPU per il predittore, nessun encoder o rete. ETA indicativa, early stopping puo anticipare.
# Ripresa: ultima epoca completa con optimizer/scaler/RNG; identita source/feature-space/protocollo obbligatoria.
from dataclasses import asdict
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import yaml

from biomap_v2.dense.evaluation import full_metrics
from .batching import CellDataset,collate_cells
from .model import make_model
from .normalization import fit_normalizer
from .sources import make_source
from .utils import signature,save_json,digest,atomic_torch,code_identity,git_commit,disk_budget,unique_tmp,replace_with_retry


def device_batch(batch,device):
    return {k:v.to(device) if torch.is_tensor(v) else v for k,v in batch.items()}


def make_loader(source,frame,norm,spec,shuffle=False):
    return DataLoader(CellDataset(source,frame,norm,spec.model.architecture=='spatial_cnn'),
        batch_size=spec.training['batch_cells'],shuffle=shuffle,num_workers=spec.training.get('num_workers',0),
        collate_fn=collate_cells,persistent_workers=False)


def check_device(spec):
    device=torch.device(spec.execution['device'])
    if device.type=='cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA richiesta ma non disponibile; verificare ambiente cu128 separato')
    if device.type=='mps' and not torch.backends.mps.is_available():
        raise ValueError('MPS esplicitamente richiesto ma non disponibile')
    if device.type not in {'cpu','cuda','mps'}:
        raise ValueError('Device non supportato')
    return device


def preflight(spec):
    out=Path(spec.execution['output_root'])
    source=make_source(spec,out/'index')
    try:
        max_h,max_w=source.db.execute('SELECT max(height),max(width) FROM patches').fetchone()
        max_patches=source.db.execute('SELECT max(n) FROM (SELECT count(*) AS n FROM patches GROUP BY cell_id)').fetchone()[0]
        input_bytes=int(spec.training['batch_cells']*max_patches*spec.model.input_channels*max_h*max_w*4)
        disk_budget(out,spec.execution.get('reserve_gib',20),64*1024**2)
        report={'status':'READY','profile':spec.protocol.profile,'scientific_result':False,
            'n_development':int((source.frame.partition=='train').sum()),'n_external':int((source.frame.partition=='test').sum()),
            'development_aois':int(source.frame[source.frame.partition=='train'].aoi_id.nunique()),
            'representation':'authentic_pixel_spatial','feature_space_id':source.feature_space_id,
            'source_fingerprint':source.fingerprint,'duplicate_contributions':int(source.frame.duplicate_contributions.sum()),
            'batch_input_upper_bound_bytes':input_bytes,'rough_training_bytes':input_bytes*12,
            'memory_note':'stima conservativa, non profiling VRAM; profile misura un batch reale',
            'source_writes':False,'encoder_loaded':False}
        # All file/shape/split/support validation happened before this device query.
        device=check_device(spec)
        report['device']=str(device)
        if device.type=='cuda':
            report.update(gpu=torch.cuda.get_device_name(device),torch_cuda=torch.version.cuda,
                gpu_total_bytes=torch.cuda.get_device_properties(device).total_memory)
            if report['rough_training_bytes']>report['gpu_total_bytes']:
                raise ValueError('Stima batch oltre VRAM: ridurre batch_cells/projection e riprovare profile')
        out.mkdir(parents=True,exist_ok=True)
        save_json(out/'preflight.json',report)
        return source,report
    except BaseException:
        source.close()
        raise


class Progress:
    def __init__(self,path):
        self.path=Path(path)
        self.start=time.perf_counter()
        self.last=0.
        self.state={}

    def update(self,force=False,**values):
        self.state.update(values)
        now=time.perf_counter()
        self.state['elapsed_session_seconds']=now-self.start
        if force or now-self.last>=2:
            try:save_json(self.path,self.state)
            except OSError as error:print('Avanzamento non scritto, checkpoint indipendente:',error,flush=True)
            self.last=now


def run_identity(spec,source):
    return signature({'spec':spec.to_dict(),'source':source.fingerprint,'index':source.index_identity,'code':code_identity()})


def run_path(spec,run_id):
    if not run_id or Path(run_id).name!=run_id or any(x in run_id for x in [':','/','\\']) or run_id in {'.','..','index','comparisons'}:
        raise ValueError('run_id deve essere un nome semplice e non riservato')
    return Path(spec.execution['output_root'])/run_id


def registry(spec,source,run_id,create=False):
    run=run_path(spec,run_id)
    identity=run_identity(spec,source)
    manifest=run/'manifest.json'
    if manifest.exists():
        saved=json.loads(manifest.read_text())
        if saved['identity']!=identity or saved['feature_space_id']!=source.feature_space_id:
            raise ValueError('Run/checkpoint di un altro feature space, protocollo, codice o dataset')
    else:
        if not create:raise FileNotFoundError('Run non inizializzato')
        run.mkdir(parents=True,exist_ok=True)
        (run/'config.yaml').write_text(yaml.safe_dump(spec.to_dict(),sort_keys=False),encoding='utf-8')
        saved={'identity':identity,'source_fingerprint':source.fingerprint,'feature_space_id':source.feature_space_id,
            'support_identity':source.index_identity,
            'encoder':getattr(source,'encoder',spec.source.get('encoder','geotessera_v2')),'adapter':type(source).__name__+'/'+source.protocol,
            'protocol':asdict(spec.protocol),'model':asdict(spec.model),'source_representation':'spatial',
            'commit':git_commit(),'code_sha256':code_identity(),'device':spec.execution['device'],
            'precision':'amp_float16' if spec.training.get('mixed_precision') else 'fp32',
            'state':'created','scientific_result':spec.protocol.profile=='benchmark','encoder_frozen':True,
            'torch':torch.__version__,'numpy':np.__version__}
        save_json(manifest,saved)
    return run,saved


def write_state(run,manifest,state,**extra):
    manifest.update(state=state,**extra)
    save_json(run/'manifest.json',manifest)


def predict(model,loader,device):
    model.eval()
    result={}
    with torch.no_grad():
        for batch in loader:
            values=model(device_batch(batch,device)).float().cpu().numpy()
            if not np.isfinite(values).all():raise ValueError('Predizioni non finite')
            result.update(zip(batch['cell_ids'],values.tolist()))
    return result


def prediction_frame(frame,values,model,fold):
    columns=[c for c in ['cell_id','aoi_id','partition','cv_fold','lon','lat','target','n_valid','valid_fraction','patches','heterogeneity','duplicate_contributions'] if c in frame]
    out=frame[columns].copy()
    out['prediction']=[values[c] for c in out.cell_id]
    out['model'],out['fold']=model,fold
    return out


def macro_score(frame,values):
    return float(np.mean([np.sqrt(np.mean((g.target.to_numpy()-np.array([values[c] for c in g.cell_id]))**2)) for _,g in frame.groupby('aoi_id')]))


def seed_all(seed,device):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if device.type=='cuda':torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.use_deterministic_algorithms(True)


def fit_one(spec,source,training,validation,folder,identity,resume=False,epoch_limit=None,progress=None):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    device=check_device(spec);seed_all(spec.protocol.seed,device)
    norm=fit_normalizer(training,spec.protocol.normalization)
    model=make_model(spec.model).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=spec.training['learning_rate'],weight_decay=spec.training['weight_decay'])
    amp=bool(spec.training.get('mixed_precision',False) and device.type=='cuda')
    scaler=torch.amp.GradScaler('cuda',enabled=amp)
    start,best,bad,best_epoch=0,float('inf'),0,0
    last,best_path=folder/'last.pt',folder/'best.pt'
    if last.exists():
        if not resume:raise FileExistsError('Checkpoint esistente: specificare --resume')
        saved=torch.load(last,map_location='cpu',weights_only=False)
        if saved['identity']!=identity or saved['feature_space_id']!=source.feature_space_id or saved['normalizer']['training_ids_sha256']!=norm['training_ids_sha256']:
            raise ValueError('Checkpoint/normalizzatore incompatibile')
        model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer']);scaler.load_state_dict(saved['scaler'])
        start,best,bad,best_epoch=saved['epoch'],saved['best'],saved['bad'],saved['best_epoch']
        random.setstate(saved['python_rng']);np.random.set_state(saved['numpy_rng']);torch.set_rng_state(saved['torch_rng'])
        if device.type=='cuda':torch.cuda.set_rng_state_all(saved['cuda_rng'])
        norm=saved['normalizer']
    loader=make_loader(source,training,norm,spec,True)
    val_loader=make_loader(source,validation,norm,spec) if validation is not None else None
    epochs=spec.training['epochs']
    stop=min(epochs,start+epoch_limit) if epoch_limit is not None else epochs
    began=time.perf_counter()
    durations=[]
    if device.type=='cuda':torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(start,stop):
        if validation is not None and bad>=spec.training['patience']:break
        tick=time.perf_counter();model.train();loss_sum=0.;seen=pending=0
        optimizer.zero_grad(set_to_none=True)
        goal=spec.training.get('accumulate_cells',spec.training['batch_cells'])
        for step,batch in enumerate(loader,1):
            batch=device_batch(batch,device);n=len(batch['cell_ids'])
            with torch.autocast(device_type=device.type,enabled=amp,dtype=torch.float16):
                losses=(model(batch)-batch['target']).square()
                loss=losses.sum()/goal
            scaler.scale(loss).backward()
            pending+=n;seen+=n;loss_sum+=float(losses.detach().sum())
            if pending>=goal or step==len(loader):
                scaler.unscale_(optimizer)
                for p in model.parameters():
                    if p.grad is not None:p.grad.mul_(goal/pending)
                torch.nn.utils.clip_grad_norm_(model.parameters(),spec.training['clip_grad'])
                scaler.step(optimizer);scaler.update();optimizer.zero_grad(set_to_none=True);pending=0
            if progress:
                elapsed=time.perf_counter()-tick;rate=seen/elapsed if elapsed else 0
                progress.update(epoch=epoch+1,epochs_budget=epochs,cells_done=seen,cells_total=len(training),
                    cells_per_second=rate,eta_current_epoch_seconds=(len(training)-seen)/rate if rate else None,
                    eta_note='stima epoca corrente; validation/refit e early stopping modificano ETA globale')
        if validation is not None:
            score=macro_score(validation,predict(model,val_loader,device))
            if score<best:best,bad,best_epoch=score,0,epoch+1
            else:bad+=1
        else:
            score=None;best_epoch=epoch+1;bad=0
        if device.type=='cuda':torch.cuda.synchronize(device)
        state={'identity':identity,'feature_space_id':source.feature_space_id,'model':model.state_dict(),
            'model_config':asdict(spec.model),'protocol':asdict(spec.protocol),'normalizer':norm,
            'optimizer':optimizer.state_dict(),'scaler':scaler.state_dict(),'epoch':epoch+1,'best':best,'bad':bad,'best_epoch':best_epoch,
            'python_rng':random.getstate(),'numpy_rng':np.random.get_state(),'torch_rng':torch.get_rng_state(),
            'cuda_rng':torch.cuda.get_rng_state_all() if device.type=='cuda' else []}
        disk_budget(folder,spec.execution.get('reserve_gib',20),32*1024**2)
        if bad==0:atomic_torch(best_path,state)
        atomic_torch(last,state)
        record={'epoch':epoch+1,'train_mse':loss_sum/seen,'validation_macro_aoi_rmse':score,
            'epoch_seconds':time.perf_counter()-tick,'train_seconds_this_invocation':time.perf_counter()-began,
            'gpu_peak_allocated_bytes':torch.cuda.max_memory_allocated(device) if device.type=='cuda' else None,
            'model':model.description()}
        durations.append(record['epoch_seconds'])
        remaining_epochs=max(epochs-epoch-1,0)
        record['eta_job_budget_seconds']=remaining_epochs*float(np.mean(durations))
        if progress:
            remaining_jobs=max(progress.state.get('jobs_total',1)-progress.state.get('jobs_done',0)-1,0)
            record['eta_phase_budget_seconds']=(remaining_epochs+remaining_jobs*epochs)*float(np.mean(durations))
            record['eta_note']='budget approssimativo fase corrente, validation inclusa; early stopping puo anticipare; fold di dimensioni diverse'
        with (folder/'epochs.jsonl').open('a',encoding='utf-8') as stream:stream.write(json.dumps(record)+'\n')
        if progress:progress.update(force=True,**record)
        print(json.dumps({k:v for k,v in record.items() if k!='model'}),flush=True)
    saved=torch.load(last,map_location='cpu',weights_only=False)
    done=saved['epoch']>=epochs or (validation is not None and saved['bad']>=spec.training['patience'])
    best_saved=torch.load(best_path,map_location='cpu',weights_only=False)
    model.load_state_dict(best_saved['model'])
    return model,norm,int(best_saved['best_epoch']),done


def atomic_parquet(path,frame):
    tmp=unique_tmp(path)
    frame.to_parquet(tmp,index=False)
    replace_with_retry(tmp,path)


def metrics_report(predictions,run,prefix):
    rows={};by_aoi=[];by_fold=[];diagnostics=[]
    for model,g in predictions.groupby('model'):
        if g.cell_id.duplicated().any():raise ValueError('Predizione ripetuta per cella')
        row=full_metrics(g.target,g.prediction)
        local=[]
        for aoi,part in g.groupby('aoi_id'):
            item={'model':model,'aoi_id':aoi,**full_metrics(part.target,part.prediction)}
            local.append(item);by_aoi.append(item)
        row.update(macro_aoi_rmse=float(np.mean([i['rmse'] for i in local])),
            macro_aoi_mae=float(np.mean([i['mae'] for i in local])),mape_near_zero_n=int((g.target.abs()<.01).sum()))
        rows[model]=row
        for fold,part in g.groupby('fold'):by_fold.append({'model':model,'fold':int(fold),**full_metrics(part.target,part.prediction)})
        for name in ['target','patches','valid_fraction','heterogeneity']:
            if name not in g:continue
            grouping=pd.cut(g[name],[-np.inf,.2,.4,.6,.8,np.inf]) if name in {'target','valid_fraction'} else pd.qcut(g[name],4,duplicates='drop')
            for label,part in g.groupby(grouping,observed=True):
                diagnostics.append({'model':model,'grouping':name,'interval':str(label),**full_metrics(part.target,part.prediction)})
    if len({frozenset(g.cell_id) for _,g in predictions.groupby('model')})!=1:
        raise ValueError('Confronto su popolazioni differenti')
    dummy=rows.get('dummy_mean')
    if dummy:
        for name,row in rows.items():
            row['rmse_delta_vs_dummy']=row['rmse']-dummy['rmse']
    pd.DataFrame(by_aoi).to_csv(run/(prefix+'_by_aoi.csv'),index=False)
    pd.DataFrame(by_fold).to_csv(run/(prefix+'_by_fold.csv'),index=False)
    pd.DataFrame(diagnostics).to_csv(run/(prefix+'_diagnostics.csv'),index=False)
    save_json(run/(prefix+'_metrics.json'),rows)
    return rows


def train(spec,source,run_id,phase,resume=False,epoch_limit=None):
    run,manifest=registry(spec,source,run_id,create=True)
    identity=manifest['identity']
    if phase=='cv' and (run/'fold_0').exists() and not resume:
        raise FileExistsError('Run CV esistente: --resume oppure nuovo run-id')
    progress=Progress(run/'progress.json')
    write_state(run,manifest,'running',phase=phase)
    dev=source.frame[source.frame.partition=='train'].reset_index(drop=True)
    try:
        if phase=='cv':
            predictions=[];best_epochs=[]
            for fold in [0,1,2]:
                training=dev[dev.cv_fold!=fold].reset_index(drop=True)
                validation=dev[dev.cv_fold==fold].reset_index(drop=True)
                if set(training.aoi_id)&set(validation.aoi_id):raise ValueError('Leakage AOI')
                folder=run/f'fold_{fold}';folder.mkdir(exist_ok=True)
                progress.update(force=True,phase='cv',fold=fold,jobs_done=fold,jobs_total=3)
                marker=folder/'complete.json';path=folder/'predictions.parquet'
                if marker.exists():
                    mark=json.loads(marker.read_text())
                    if mark['identity']!=identity or mark['sha256']!=digest(path) or mark['model_sha256']!=digest(folder/'best.pt'):
                        raise ValueError('Fold completato corrotto/incompatibile')
                    pred=pd.read_parquet(path);best_epoch=mark['best_epoch']
                else:
                    model,norm,best_epoch,done=fit_one(spec,source,training,validation,folder,identity,resume,epoch_limit,progress)
                    if not done:
                        write_state(run,manifest,'interrupted',reason='limite tecnico epoche per invocazione')
                        progress.update(force=True,status='interrupted')
                        return run
                    values=predict(model,make_loader(source,validation,norm,spec),check_device(spec))
                    pred=prediction_frame(validation,values,'predictor',fold)
                    atomic_parquet(path,pred)
                    save_json(marker,{'identity':identity,'sha256':digest(path),'model_sha256':digest(folder/'best.pt'),'best_epoch':best_epoch})
                predictions.append(pred);best_epochs.append(best_epoch)
                predictions.append(prediction_frame(validation,dict.fromkeys(validation.cell_id,float(training.target.mean())),'dummy_mean',fold))
            combined=pd.concat(predictions,ignore_index=True)
            atomic_parquet(run/'predictions.parquet',combined)
            rows=metrics_report(combined,run,'oof')
            save_json(run/'metrics.json',{'oof':rows})
            selection={'identity':identity,'criterion':'oof_macro_aoi_rmse','score':rows['predictor']['macro_aoi_rmse'],
                'refit_epochs':max(1,int(np.median(best_epochs))),'fold_best_epochs':best_epochs,'liguria_used':False,
                'oof_sha256':digest(run/'predictions.parquet')}
            save_json(run/'selection.json',selection)
            write_state(run,manifest,'cv_complete')
        elif phase=='refit':
            progress.update(force=True,phase='refit',jobs_done=0,jobs_total=1)
            selection=json.loads((run/'selection.json').read_text())
            if selection['identity']!=identity or selection['oof_sha256']!=digest(run/'predictions.parquet'):
                raise ValueError('Selezione OOF non valida')
            from dataclasses import replace
            local=replace(spec,training={**spec.training,'epochs':selection['refit_epochs']})
            model,norm,_,done=fit_one(local,source,dev,None,run/'refit',identity,resume,epoch_limit,progress)
            if not done:
                write_state(run,manifest,'interrupted',reason='limite tecnico epoche per invocazione')
                progress.update(force=True,status='interrupted');return run
            saved=torch.load(run/'refit/best.pt',map_location='cpu',weights_only=False)
            atomic_torch(run/'checkpoint.pt',saved)
            save_json(run/'refit_complete.json',{'identity':identity,'sha256':digest(run/'checkpoint.pt'),'n_development':len(dev)})
            write_state(run,manifest,'refit_complete')
        else:raise ValueError('Fase train sconosciuta')
        progress.update(force=True,status=manifest['state'],jobs_done=3 if phase=='cv' else 1,jobs_total=3 if phase=='cv' else 1)
        return run
    except BaseException as error:
        write_state(run,manifest,'interrupted' if isinstance(error,KeyboardInterrupt) else 'failed',error=type(error).__name__)
        progress.update(force=True,status=manifest['state'])
        raise


def evaluate(spec,source,run_id):
    run,manifest=registry(spec,source,run_id)
    mark=json.loads((run/'refit_complete.json').read_text())
    if mark['identity']!=manifest['identity'] or mark['sha256']!=digest(run/'checkpoint.pt'):
        raise ValueError('Checkpoint finale non valido')
    saved=torch.load(run/'checkpoint.pt',map_location='cpu',weights_only=False)
    if saved['feature_space_id']!=source.feature_space_id or saved['identity']!=manifest['identity']:
        raise ValueError('Feature space/checkpoint incompatibile')
    test=source.frame[source.frame.partition=='test'].reset_index(drop=True)
    dev=source.frame[source.frame.partition=='train']
    model=make_model(spec.model).to(check_device(spec));model.load_state_dict(saved['model'])
    values=predict(model,make_loader(source,test,saved['normalizer'],spec),check_device(spec))
    predictions=pd.concat([prediction_frame(test,values,'predictor',-1),
        prediction_frame(test,dict.fromkeys(test.cell_id,float(dev.target.mean())),'dummy_mean',-1)],ignore_index=True)
    atomic_parquet(run/'external_predictions.parquet',predictions)
    rows=metrics_report(predictions,run,'external')
    original=json.loads((run/'metrics.json').read_text());original['external']=rows
    save_json(run/'metrics.json',original)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    frame=predictions[predictions.model=='predictor']
    fig,ax=plt.subplots(1,2,figsize=(9,4))
    ax[0].scatter(frame.target,frame.prediction,s=3,alpha=.3);ax[0].plot([0,1],[0,1],'k--')
    ax[0].set(xlabel='GLOBIO4 MSA target',ylabel='Predizione',title='Liguria esterna gia osservata')
    ax[1].hist(frame.prediction-frame.target,bins=30);ax[1].set(xlabel='Predizione - target',title='Residui')
    fig.tight_layout();fig.savefig(run/'external_diagnostics.png',dpi=130);plt.close(fig)
    text=['# Spatial CNN: valutazione esterna','',f'Profilo: {spec.protocol.profile}; celle esterne: {len(test)}; sviluppo: {len(dev)}.',
        'Target GLOBIO4 MSA modellato, non biodiversita misurata sul campo. Liguria gia osservata; non usata per selezione.',
        'MAPE esclude soltanto target zero, riportando il denominatore; i target vicini a zero restano inclusi e contati.',
        'Le differenze rispetto a risultati storici non sono automaticamente appaiate sullo stesso supporto.','',json.dumps(rows,indent=2)]
    (run/'report.md').write_text('\n'.join(text),encoding='utf-8')
    write_state(run,manifest,'complete')
    save_json(run/'complete.json',{'identity':manifest['identity'],'checkpoint_sha256':digest(run/'checkpoint.pt'),
        'predictions_sha256':digest(run/'external_predictions.parquet'),'metrics_sha256':digest(run/'metrics.json')})
    return rows


def compare_runs(spec,source,names,label):
    rows=[];cohort=None;protocol=None
    for name in names:
        run=run_path(spec,name);manifest=json.loads((run/'manifest.json').read_text())
        selection=json.loads((run/'selection.json').read_text())
        if manifest['source_fingerprint']!=source.fingerprint or manifest['support_identity']!=source.index_identity or selection['identity']!=manifest['identity'] or selection['oof_sha256']!=digest(run/'predictions.parquet'):
            raise ValueError('Run non comparabili o predizioni alterate')
        preds=pd.read_parquet(run/'predictions.parquet');preds=preds[preds.model=='predictor'].sort_values('cell_id')
        current=signature(preds[['cell_id','target','aoi_id','cv_fold','n_valid']].to_dict('records'))
        if cohort is not None and (current!=cohort or manifest['protocol']!=protocol):
            raise ValueError('Target/split/supporto/normalizzazione/seed differenti')
        cohort,protocol=current,manifest['protocol']
        rows.append({'run_id':name,'macro_aoi_rmse':selection['score'],'model':manifest['model']})
    rows.sort(key=lambda r:(r['macro_aoi_rmse'],r['run_id']))
    path=run_path(spec,label)
    out=Path(spec.execution['output_root'])/'comparisons'/path.name
    out.mkdir(parents=True,exist_ok=True)
    save_json(out/'selection.json',{'selected_run_id':rows[0]['run_id'],'runs':rows,'liguria_used':False,
        'note':'1x1/3x3 cambiano anche parametri e capacita; non isolano perfettamente la spazialita'})
    return rows


def profile(spec,source):
    frame=source.frame[source.frame.partition=='train']
    # Real complete-data preflight already passed; this sample is technical, never a scientific run.
    sample=frame.head(spec.training['batch_cells']).copy()
    norm=fit_normalizer(frame,spec.protocol.normalization)
    device=check_device(spec);model=make_model(spec.model).to(device)
    batch=device_batch(next(iter(make_loader(source,sample,norm,spec))),device)
    if device.type=='cuda':torch.cuda.reset_peak_memory_stats(device)
    began=time.perf_counter();loss=(model(batch)-batch['target']).square().mean();loss.backward()
    if device.type=='cuda':torch.cuda.synchronize(device)
    report={'status':'passed','sample_cells':len(sample),'sample_patch_count':int(len(batch.get('owners',[]))),
        'forward_backward_seconds':time.perf_counter()-began,'device':str(device),'model':model.description(),
        'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated(device) if device.type=='cuda' else None,
        'scientific_result':False,'note':'solo batch configurato, non massimo sostenibile o durata intera campagna'}
    save_json(Path(spec.execution['output_root'])/'profile.json',report)
    return report
