# Scopo: confronto controllato tre fold -> selezione solo sviluppo -> refit -> Liguria.
# Input: matrice congelata; output: modelli/curve/predizioni, mai nuovi embedding.
import hashlib
import json
import platform
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import torch
from biomap_v2.common import digest, save_json
from biomap_v2.dense.evaluation import make_estimator
from biomap_v2.campaign.state import MEANS
from . import engine
from .models import make
from .evaluation import record,export

# Cache fit baseline con checksum; lo scaler della Ridge è già nella pipeline sklearn.
def baseline(name,x,y,indices,out,seed):
    path=out/(name+'.joblib');meta=path.with_suffix('.json')
    if path.exists() and meta.exists():
        if json.loads(meta.read_text())['sha256']!=digest(path):raise ValueError('Baseline corrotta')
        return joblib.load(path)
    params={'max_iter':90,'max_leaf_nodes':15,'learning_rate':.05,'l2_regularization':1.,'early_stopping':True,'random_state':20260925}
    model=make_estimator(name,params,100.,seed)
    model.fit(x[indices],y[indices]);tmp=path.with_suffix('.partial');joblib.dump(model,tmp);tmp.replace(path)
    save_json(meta,{'sha256':digest(path),'n_training':len(indices)})
    return model

# Il protocollo statico include codice/dati/config; timestamps sono registrati separatamente.
def protocol(cfg,audit,out,resume):
    out=Path(out);path=out/'protocol.json'
    import sklearn,subprocess
    payload={'git_sha':subprocess.check_output(['git','rev-parse','HEAD'],cwd=Path(__file__).resolve().parents[2],text=True).strip(),'device':audit['device'],'sklearn':sklearn.__version__,'configuration':cfg,'feature_sha256':audit['feature_sha256'],'completion_sha256':audit['completion_sha256'],
        'targets_sha256':audit['proof'].get('targets_sha256'),'source_code':{p.name:digest(p) for p in Path(__file__).parent.glob('*.py')},
        'torch':torch.__version__,'numpy':np.__version__,'platform':platform.platform(),'frozen_encoder':True,
        'representation':'128 dense means; Conv1D over learned channels, no spatial layout','smoke':audit['smoke']}
    if path.exists():
        if not resume or json.loads(path.read_text())!=payload:raise ValueError('Run già esistente o protocollo cambiato; usare --resume solo per lo stesso esperimento')
    else:save_json(path,payload)
    return hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()

# Training completo chiamato soltanto esplicitamente; nessun avvio al termine dell'estrazione.
def run(frame,cfg,audit,out,device,progress,resume=False):
    out=Path(out);fingerprint=protocol(cfg,audit,out,resume)
    x=frame[MEANS].to_numpy(dtype=np.float32);y=frame.target.to_numpy(dtype=np.float32);aois=frame.aoi_id.to_numpy()
    dev=np.flatnonzero(frame.partition.to_numpy()=='train');test=np.flatnonzero(frame.partition.to_numpy()=='test')
    predictions=[];best_epochs={};macro_scores={};fits_done=0
    candidates=[(name,variant) for name in cfg['models'] for variant in cfg['outputs']]
    fits_total=len(cfg['baselines'])*4+len(candidates)*3+len(cfg['models'])
    all_names=[*cfg['baselines'],*[f'{n}_{v}' for n,v in candidates]]
    for label in all_names:
        family=label.split('_')[0];variant=label.split('_')[1] if '_' in label else None
        part=[];best_epochs[label]=[]
        for fold in range(3):
            train=np.flatnonzero((frame.partition=='train')&(frame.cv_fold!=fold));val=np.flatnonzero((frame.partition=='train')&(frame.cv_fold==fold))
            folder=out/'models'/f'{label}_fold{fold}';folder.mkdir(parents=True,exist_ok=True)
            progress.update(state='running',phase='cross_validation',model=label,fold=fold,fits_done=fits_done,fits_total=fits_total,
                epoch=0,max_epochs=cfg['epochs'] if variant else None,eta_fit_cap_seconds=None)
            if variant:
                (head,scaler),history,best=engine.fit(x,y,aois,train,val,family,variant,cfg,folder,device,progress,fingerprint,resume=resume)
                pred=engine.predict(head,scaler.transform(x[val]).astype(np.float32),cfg['batch_size'],device);best_epochs[label].append(best)
            else:
                model=baseline(label,x,y,train,folder,cfg['seed']);pred=model.predict(x[val])
            rec=record(frame,val,pred,label,out.name);part.append(rec);predictions.append(rec);fits_done+=1
        combined=pd.concat(part);macro_scores[label]=engine.macro_rmse(combined.target.to_numpy(),combined.prediction.to_numpy(),combined.aoi_id.to_numpy())
    selected={name:min([f'{name}_{v}' for v in cfg['outputs']],key=lambda key:macro_scores[key]) for name in cfg['models']}
    selection={'criterion':'mean RMSE per held development AOI; Liguria excluded','macro_rmse':macro_scores,
        'selected_output_by_family':selected,'best_epochs':best_epochs,'refit_epochs':{},'test_role':'previously observed Liguria benchmark'}
    # Congela la scelta PRIMA del primo uso dei target liguri.
    save_json(out/'selection.json',selection)
    for label in [*cfg['baselines'],*selected.values()]:
        progress.update(phase='refit_development',model=label,fold=None,fits_done=fits_done,fits_total=fits_total,epoch=0,eta_fit_cap_seconds=None)
        folder=out/'models'/f'{label}_final';folder.mkdir(parents=True,exist_ok=True)
        if label in cfg['baselines']:
            model=baseline(label,x,y,dev,folder,cfg['seed']);pred=model.predict(x[test])
        else:
            name,variant=label.split('_');epochs=max(1,int(np.median(best_epochs[label])));selection['refit_epochs'][label]=epochs
            (head,scaler),history,_=engine.fit(x,y,aois,dev,None,name,variant,cfg,folder,device,progress,fingerprint,epochs=epochs,resume=resume)
            pred=engine.predict(head,scaler.transform(x[test]).astype(np.float32),cfg['batch_size'],device)
        fits_done+=1;predictions.append(record(frame,test,pred,label,out.name))
    progress.update(phase='export',fits_done=fits_done,fits_total=fits_total)
    result=pd.concat(predictions,ignore_index=True);result.to_csv(out/'predictions.csv',index=False)
    save_json(out/'selection.json',selection)
    from biomap_v2.common import ROOT
    reference=json.loads((ROOT/'configs/msa_v1_liguria_reference.json').read_text())
    save_json(out/'historical_reference_indicative.json',{'status':'historical scalar reference only; NOT a new measured result',
        'source':'configs/msa_v1_liguria_reference.json','reference':reference,'paired_comparison_requires_predictions':True})
    export(out,result,selection)
    progress.update(state='complete',phase='complete',fits_done=fits_done,fits_total=fits_total,eta_fit_cap_seconds=0)
    return result

# Profilo limitato di una sola testa e un solo fold; risultati non sono selezione scientifica.
def profile(frame,cfg,audit,out,device,progress,cells=4096,epochs=3):
    out=Path(out);x=frame[MEANS].to_numpy(np.float32);y=frame.target.to_numpy(np.float32);a=frame.aoi_id.to_numpy()
    train=np.flatnonzero((frame.partition=='train')&(frame.cv_fold!=0))[:cells]
    val=np.flatnonzero((frame.partition=='train')&(frame.cv_fold==0))[:max(1,cells//4)]
    if str(device).startswith('cuda'):torch.cuda.reset_peak_memory_stats()
    progress.update(state='running',phase='profile',model='Conv1D_identity',fold=0)
    (_, _),history,_=engine.fit(x,y,a,train,val,'Conv1D','identity',cfg,out/'profile_fit',device,progress,'profile_'+audit['feature_sha256'],epochs=epochs)
    per_epoch=float(np.mean([r['seconds'] for r in history]));n_dev=int((frame.partition=='train').sum())
    configs=len(cfg['models'])*len(cfg['outputs']);upper_neural=(configs*3+len(cfg['models']))*cfg['epochs']
    result={'mode':'limited profile, not scientific result','measured_epoch_seconds':per_epoch,'sample_train':len(train),'sample_validation':len(val),
        'linear_full_cv_epoch_seconds':per_epoch*n_dev/(len(train)+len(val)),
        'neural_campaign_cap_seconds_estimate':per_epoch*n_dev/(len(train)+len(val))*upper_neural,
        'limitations':'linear extrapolation; early stopping may shorten; baseline CPU fits/export excluded; hardware-dependent',
        'gpu_peak_bytes':torch.cuda.max_memory_allocated() if device.startswith('cuda') else None,
        'feature_matrix_float32_bytes':int(x.nbytes),'configurations':configs,
        'parameters':{name:sum(p.numel() for p in make(name,'identity',cfg).parameters()) for name in cfg['models']}}
    save_json(out/'profile.json',result);progress.update(state='complete',phase='profile_complete',**result)
