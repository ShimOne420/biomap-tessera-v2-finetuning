# Scopo: probing, fine-tuning progressivo e valutazione appaiata degli indicatori.
# Fasi: verifica input, training solo train, selezione validation, predizioni e report.
# Input: celle preparate, checkpoint e YAML; output: run con modelli, optimizer, CSV e PNG.
# Parametri: --stage head/partial/full/all, --resume RUN/last.pt, --smoke, --set training.*.
# Esempio: python scripts/finetune.py run --indicator bii --stage all --data-root E:\BioMAP.
# Risorse: GPU consigliata; mixed precision, clipping, accumulo e microbatch cella adattabile.
# Ripresa: salva optimizer e RNG a fine epoca; ripete solo l'epoca eventualmente interrotta.
# Controllo: partial/full hanno un controllo head-only aggiuntivo con lo stesso budget di fase.
from pathlib import Path
from contextlib import nullcontext
import copy
import json
import random
import time
import joblib
import numpy as np
import pandas as pd
import torch
from .common import budget, digest, geographic_rmse, make_run, metrics, report, root, save_json
from .data import cell_directory, prepared
from .encoder import Regression, cell_embedding, embeddings, load_encoder, trainable

FEATURES=[f'f{i:03d}' for i in range(128)]


def prediction_frame(frame,pred,model):
    result=frame[['cell_id','lon','lat','aoi_id','partition','target','valid_fraction']].copy()
    result['prediction']=np.asarray(pred);result['model']=model
    return result


def probe(cfg,args):
    from sklearn.dummy import DummyRegressor
    from sklearn.linear_model import Ridge
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    frame=prepared(cfg,args.smoke);features,checkpoint=embeddings(cfg,frame,args.checkpoint)
    f=frame.merge(features,on='cell_id',validate='one_to_one',how='left')
    if f[FEATURES].isna().any().any():raise ValueError('Feature mancanti')
    out=make_run(cfg,'probe_smoke' if args.smoke else 'probe');start=time.perf_counter()
    factories={'DummyMean':lambda:DummyRegressor(strategy='mean'),
        'Ridge':lambda:make_pipeline(StandardScaler(),Ridge(alpha=cfg['probing']['ridge_alpha'])),
        'HistGB':lambda:HistGradientBoostingRegressor(max_iter=cfg['probing']['histgb_iterations'],
            max_leaf_nodes=15,l2_regularization=1,early_stopping=False,random_state=42)}
    if 'XGBoost' in cfg['probing']['models']:
        from xgboost import XGBRegressor
        factories['XGBoost']=lambda:XGBRegressor(n_estimators=cfg['probing']['xgb_estimators'],max_depth=2,
            learning_rate=.05,random_state=42,n_jobs=cfg['probing']['workers'],objective='reg:squarederror')
    train=f[f.partition=='train'];validation=f[f.partition=='validation'];selection={};predictions=[]
    names=list(dict.fromkeys(['DummyMean',*cfg['probing']['models']]))
    for name in names:
        if name not in factories:raise ValueError(f'Regressore sconosciuto: {name}')
        model=factories[name]();model.fit(train[FEATURES],train.target)
        validation_prediction=model.predict(validation[FEATURES])
        selection[name]=geographic_rmse(validation,validation_prediction,cfg['evaluation']['block_m'],cfg['evaluation'].get('grouping','spatial_block'))
        joblib.dump(model,out/f'{name}.joblib')
        predictions.append(prediction_frame(f,model.predict(f[FEATURES]),name))
    winner=min((n for n in names if n!='DummyMean'),key=lambda n:selection[n])
    save_json(out/'selection.json',{'criterion':'validation_macro_'+cfg['evaluation'].get('grouping','spatial_block')+'_rmse','scores':selection,
        'selected':winner,'checkpoint_sha256':digest(checkpoint),'seconds':time.perf_counter()-start,
        'support':'identical prepared cells and sampled pixel positions'})
    report(out,pd.concat(predictions,ignore_index=True));print(f'Run: {out}');return out


def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)


def torch_save(path,payload):
    tmp=path.with_suffix('.tmp');torch.save(payload,tmp);tmp.replace(path)


def rng_state():
    return {'python':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state(),
        'cuda':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(rng):
    random.setstate(rng['python']);np.random.set_state(rng['numpy']);torch.set_rng_state(rng['torch'].cpu())
    if rng['cuda'] and torch.cuda.is_available():torch.cuda.set_rng_state_all([s.cpu() for s in rng['cuda']])


def amp_context(cfg):
    device=torch.device(cfg['model']['device'])
    if cfg['training']['mixed_precision'] and device.type=='cuda':
        dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        return torch.autocast('cuda',dtype=dtype)
    return nullcontext()


def predict(model,frame,cfg):
    model.eval();result=[]
    with torch.no_grad():
        for cell in frame.itertuples():
            with amp_context(cfg):p=model.forward_cell(cell_directory(cfg,cell.cell_id)/'series.npz',cfg)
            result.append(float(p.float().cpu()))
    return np.asarray(result)


def phase(cfg,frame,out,stage,seed,initial=None,resume=None):
    seed_all(seed);encoder,model_args,source=load_encoder(cfg)
    model=Regression(encoder).to(cfg['model']['device'])
    if initial:
        initial_payload=torch.load(initial,map_location='cpu',weights_only=False)
        model.encoder.load_state_dict(initial_payload['encoder']);model.head.load_state_dict(initial_payload['head'])
    names=trainable(model,stage,cfg['training']['unfreeze_blocks'])
    params=[{'params':model.head.parameters(),'lr':cfg['training']['head_lr']}]
    encparams=[p for p in model.encoder.parameters() if p.requires_grad]
    if encparams:params.append({'params':encparams,'lr':cfg['training']['encoder_lr']})
    optimizer=torch.optim.AdamW(params,weight_decay=cfg['training']['weight_decay'])
    scaler=torch.amp.GradScaler('cuda',enabled=torch.device(cfg['model']['device']).type=='cuda' and cfg['training']['mixed_precision'] and not torch.cuda.is_bf16_supported())
    train=frame[frame.partition=='train'].reset_index(drop=True);val=frame[frame.partition=='validation'].reset_index(drop=True)
    fingerprint={'cells':frame[['cell_id','series_sha256','target','partition']].to_dict('records'),
        'source_checkpoint_sha256':digest(source),'indicator':cfg['indicator'],'sampling':cfg['sampling'],
        'training':cfg['training'],'stage':stage,'seed':seed,'model':cfg['model']}
    epoch_start=0;best=float('inf');stale=0;history=[];microbatch=cfg['training']['batch_cells']
    effective=cfg['training']['effective_batch_cells'];out.mkdir(parents=True,exist_ok=True)
    save_json(out/'config.json',cfg)
    if resume:
        payload=torch.load(resume,map_location='cpu',weights_only=False)
        if payload['fingerprint']!=fingerprint:raise ValueError('Resume incompatibile: dati/configurazione/stage/seed cambiati')
        model.encoder.load_state_dict(payload['encoder']);model.head.load_state_dict(payload['head'])
        optimizer.load_state_dict(payload['optimizer']);scaler.load_state_dict(payload['scaler'])
        restore_rng(payload['rng']);epoch_start=payload['epoch']+1
        best=payload['best'];stale=payload['stale'];history=payload['history'];microbatch=payload['microbatch']
        if not (out/'best.pt').exists():raise ValueError('Resume richiede anche best.pt nella stessa cartella')
    initial_encoder={k:v.detach().cpu().clone() for k,v in model.encoder.state_dict().items()}
    start=time.perf_counter()
    for epoch in range(epoch_start,cfg['training']['epochs']):
        order=np.random.permutation(len(train));epoch_rng=rng_state()
        while True:
            # Roll back full epoch on OOM, preserving optimizer/RNG and effective batch.
            snapshot={'model':copy.deepcopy(model.state_dict()),'optimizer':copy.deepcopy(optimizer.state_dict()),'scaler':copy.deepcopy(scaler.state_dict())}
            total=0.;model.train()
            model.encoder.eval()  # Eval disables encoder dropout identically in every arm; gradients remain enabled.
            try:
                for start_batch in range(0,len(order),effective):
                    window=order[start_batch:start_batch+effective];optimizer.zero_grad(set_to_none=True)
                    for start_micro in range(0,len(window),microbatch):
                        subset=window[start_micro:start_micro+microbatch]
                        with amp_context(cfg):
                            pred=torch.stack([model.forward_cell(cell_directory(cfg,train.iloc[int(i)].cell_id)/'series.npz',cfg) for i in subset])
                            target=torch.as_tensor(train.iloc[subset].target.to_numpy(copy=True),device=pred.device,dtype=torch.float32)
                            loss=(pred.float()-target).square().sum()/len(window)
                        if not torch.isfinite(loss):raise RuntimeError('Loss non finita')
                        scaler.scale(loss).backward();total+=float(loss.detach())*len(window)
                    scaler.unscale_(optimizer)
                    grad=torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['training']['clip_grad'])
                    if not torch.isfinite(grad):raise RuntimeError('Gradienti non finiti')
                    scaler.step(optimizer);scaler.update()
                break
            except torch.cuda.OutOfMemoryError:
                model.load_state_dict(snapshot['model']);optimizer.load_state_dict(snapshot['optimizer']);scaler.load_state_dict(snapshot['scaler'])
                optimizer.zero_grad(set_to_none=True);restore_rng(epoch_rng);torch.cuda.empty_cache()
                if microbatch<=1:raise RuntimeError('OOM anche con una cella: ridurre model.pixel_batch o sampling.pixels_per_cell in un nuovo studio') from None
                microbatch=max(1,microbatch//2);print(f'OOM: ripeto epoca con batch_cells={microbatch}',flush=True)
            finally:
                del snapshot
        vp=predict(model,val,cfg);score=geographic_rmse(val,vp,cfg['evaluation']['block_m'],cfg['evaluation'].get('grouping','spatial_block'))
        improved=score<best-cfg['training']['min_delta']
        if improved:best=score;stale=0
        else:stale+=1
        history.append({'epoch':epoch,'train_mse':total/len(train),'validation_macro_rmse':score,'microbatch':microbatch})
        changed=any(not torch.equal(v.detach().cpu(),initial_encoder[k]) for k,v in model.encoder.state_dict().items())
        payload={'encoder':model.encoder.state_dict(),'head':model.head.state_dict(),'args':model_args,'optimizer':optimizer.state_dict(),
            'scaler':scaler.state_dict(),'rng':rng_state(),'epoch':epoch,'best':best,'stale':stale,'history':history,
            'fingerprint':fingerprint,'stage':stage,'seed':seed,'microbatch':microbatch,'encoder_changed':changed}
        budget(cfg,sum(p.numel()*p.element_size() for p in model.parameters())*4+1024**2)
        torch_save(out/'last.pt',payload)
        if improved:torch_save(out/'best.pt',payload)
        pd.DataFrame(history).to_csv(out/'history.csv',index=False)
        print(f'{stage} seed={seed} epoch={epoch+1}: validation RMSE={score:.6f}',flush=True)
        if stale>=cfg['training']['patience']:break
    if not (out/'best.pt').exists():raise ValueError('Nessun checkpoint valido')
    payload=torch.load(out/'best.pt',map_location='cpu',weights_only=False)
    model.encoder.load_state_dict(payload['encoder']);model.head.load_state_dict(payload['head'])
    if stage!='head' and not payload['encoder_changed']:raise RuntimeError('Encoder non modificato nel checkpoint migliore')
    if stage=='head' and payload['encoder_changed']:raise RuntimeError('Encoder modificato nel controllo frozen')
    save_json(out/'phase.json',{'stage':stage,'seed':seed,'best_validation_macro_rmse':payload['best'],
        'trainable_parameters':names,'encoder_changed':payload['encoder_changed'],'seconds':time.perf_counter()-start,
        'cuda_peak_gib':torch.cuda.max_memory_allocated()/1024**3 if torch.cuda.is_available() else None,
        'actual_microbatch':microbatch})
    return model,out/'best.pt',payload['best']


def train(cfg,args):
    frame=prepared(cfg,args.smoke)
    if args.resume:
        checkpoint=Path(args.resume).resolve();payload=torch.load(checkpoint,map_location='cpu',weights_only=False)
        out=checkpoint.parent
        model,best,score=phase(cfg,frame,out,payload['stage'],payload['seed'],resume=checkpoint)
        preds=prediction_frame(frame,predict(model,frame,cfg),f'{payload["stage"]}_seed{payload["seed"]}')
        report(out,preds);print(f'Fase ripresa: {out}; usare evaluate per il confronto');return out
    out=make_run(cfg,'finetune_smoke' if args.smoke else 'finetune');selection={};rows=[]
    from sklearn.dummy import DummyRegressor
    dummy=DummyRegressor().fit(np.zeros((sum(frame.partition=='train'),1)),frame[frame.partition=='train'].target)
    rows.append(prediction_frame(frame,dummy.predict(np.zeros((len(frame),1))),'DummyMean'))
    stages=['head','partial'] if args.stage=='all' else [args.stage]
    for seed in cfg['training']['seeds']:
        initial=None
        if any(stage!='head' for stage in stages):
            _,initial,_=phase(cfg,frame,out/f'head_warmup_seed{seed}','head',seed)
            model,path,score=phase(cfg,frame,out/f'head_control_seed{seed}','head',seed,initial=initial)
            selection[f'head_control_seed{seed}']={'score':score,'checkpoint':str(path.relative_to(out))}
            rows.append(prediction_frame(frame,predict(model,frame,cfg),f'head_control_seed{seed}'))
        for stage in stages:
            if stage=='head' and initial:continue
            model,path,score=phase(cfg,frame,out/f'{stage}_seed{seed}',stage,seed,initial=initial)
            name=f'{stage}_seed{seed}';selection[name]={'score':score,'checkpoint':str(path.relative_to(out))}
            rows.append(prediction_frame(frame,predict(model,frame,cfg),name))
    # All models/checkpoints selected by validation; test is never used here.
    stage_scores={}
    for name,item in selection.items():
        stage_name=name.rsplit('_seed',1)[0]
        stage_scores.setdefault(stage_name,[]).append(item['score'])
    stage_means={name:float(np.mean(scores)) for name,scores in stage_scores.items()}
    selected_stage=min(stage_means,key=stage_means.get)
    candidates=[name for name in selection if name.rsplit('_seed',1)[0]==selected_stage]
    median=float(np.median([selection[name]['score'] for name in candidates]))
    winner=min(candidates,key=lambda name:(abs(selection[name]['score']-median),name))
    save_json(out/'selection.json',{'criterion':'validation_macro_'+cfg['evaluation'].get('grouping','spatial_block')+'_rmse','candidates':selection,
        'selected':winner,'selected_checkpoint':selection[winner]['checkpoint'],
        'stage_mean_validation_scores':stage_means,'selected_stage':selected_stage,
        'representative_seed_policy':'closest to stage median validation score; never test',
        'control':'head_control and adapted encoder start from the same head_warmup checkpoint; same phase epoch budget'})
    predictions=pd.concat(rows,ignore_index=True);m=report(out,predictions)
    grouped=m[m.model!='DummyMean'].copy();grouped['stage']=grouped.model.str.replace(r'_seed\d+$','',regex=True)
    grouped.groupby(['stage','partition'])[['mae','rmse','bias']].agg(['mean','std']).to_csv(out/'seed_summary.csv')
    print(f'Run: {out}');return out


def evaluate(cfg,args):
    frame=prepared(cfg,args.smoke);out=Path(args.run_dir).resolve()
    if args.checkpoint:
        payload=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
        if payload['fingerprint']['indicator']!=cfg['indicator']:raise ValueError('Checkpoint di un altro indicatore')
        expected=frame[['cell_id','series_sha256','target','partition']].to_dict('records')
        if payload['fingerprint']['cells']!=expected:raise ValueError('Checkpoint e celle non appaiati')
        encoder,_,_=load_encoder(cfg,args.checkpoint);model=Regression(encoder).to(cfg['model']['device'])
        model.head.load_state_dict(payload['head']);preds=prediction_frame(frame,predict(model,frame,cfg),'evaluated_checkpoint')
        destination=make_run(cfg,'evaluation');report(destination,preds);print(destination);return destination
    predictions=pd.read_csv(out/'predictions.csv',dtype={'cell_id':str})
    expected=frame[['cell_id','partition','target']]
    for _,group in predictions.groupby('model'):
        check=group.merge(expected,on='cell_id',suffixes=('_saved','_current'),validate='one_to_one')
        if len(check)!=len(expected) or not np.allclose(check.target_saved,check.target_current) or not check.partition_saved.eq(check.partition_current).all():
            raise ValueError('Predizioni non appaiate al manifest corrente')
    report(out,predictions);print(out);return out
