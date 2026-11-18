# Scopo: screening e nested 3x2 con decisioni interne sigillate prima di leggere metriche esterne.
# Fasi: FULL42/43 -> margine preregistrato -> candidati42 -> conferma43/44 -> refit nuovo -> outer denso.
# Input: cache e indice verificati, budget approvato; output: ledger, decisioni, predizioni e raccomandazione.
# Parametri: fold screening, deadline12h, seed42/43/44; esempio Study(...).screen().
# Risorse: sola CNN; validation scratch limitato a un gruppo. Ripresa per fit/epoca, mai download/encoder.
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
from . import data,training
from .metrics import metrics,compare
from .state import *

PRIMARY=['S64','M128','S144','M64']


def nested_pairs(outer):
    if outer not in [0,1,2]:raise ValueError('Fold non valido')
    a,b=[x for x in [0,1,2] if x!=outer]
    return [(a,b),(b,a)]


def freeze_json(path,value):
    path=Path(path)
    if path.exists():
        if json.loads(path.read_text(encoding='utf-8'))!=value:raise ValueError('Decisione preregistrata diversa: nuovo run richiesto')
    else:save_json(path,value)
    seal=path.with_name(path.name+'.sealed.json')
    if seal.exists():
        if json.loads(seal.read_text(encoding='utf-8'))['sha256']!=digest(path):raise ValueError('Decisione sigillata modificata')
    else:save_json(seal,{'sha256':digest(path)})


def read_sealed(path):
    path=Path(path);seal=path.with_name(path.name+'.sealed.json')
    if not path.exists() or not seal.exists():return None
    if json.loads(seal.read_text(encoding='utf-8'))['sha256']!=digest(path):raise ValueError('Decisione sigillata modificata')
    return json.loads(path.read_text(encoding='utf-8'))


class Study:
    def __init__(self,source,cfg,resume=False):
        self.source=source;self.cfg=cfg;self.frame=data.development(source);self.resume=resume;self.sources={};self.root=Path(cfg['paths']['output_root'])
        self.full_source=data.DevelopmentView(source)
        plan=data.Plan(cfg['paths']['cache_root'])
        if plan.manifest['source']!=source.fingerprint:raise ValueError('Piano non corrispondente alla sorgente')
        if set(plan.frame.cell_id)!=set(self.frame.cell_id) or plan.manifest['sampling']!=cfg['sampling']:raise ValueError('Cache tecnica/subset o sampling diverso: preparare tutto lo sviluppo prima del nested')
        paired=plan.frame.set_index('cell_id').loc[self.frame.cell_id];canonical=self.frame.set_index('cell_id')
        for column in ['target','aoi_id','partition','cv_fold']:
            if not paired[column].equals(canonical[column]):raise ValueError('Target/fold del piano cambiati')
        identity_cfg={'sampling':cfg['sampling'],'training':{k:v for k,v in cfg['training'].items() if k!='num_workers'},'decision':cfg['decision']}
        self.identity=signature({'protocol':PROTOCOL,'source':source.fingerprint,'plan':plan.manifest['identity'],'config':identity_cfg,'code':data.code_identity()})
        data.own(self.root,self.identity,'study');self.progress=Progress(self.root/'progress.json')
        if not (self.root/'effective_config.json').exists():save_json(self.root/'effective_config.json',cfg)
        freeze_json(self.root/'protocol.json',{'identity':self.identity,'protocol':PROTOCOL,'selection':'internal_only',
            'epoch_rule':'(e1+e2+1)//2, fresh seeded refit','margin':'abs(FULL42-FULL43), frozen before candidates',
            'code':data.code_identity(),'liguria_used':False,'sampling_seed':cfg['sampling']['seed']})

    def close(self):
        for source in self.sources.values():source.close()
        self.sources.clear()

    def sampled(self,mode):
        if mode=='FULL':return self.full_source
        if mode not in self.sources:
            source=data.CachedSource(self.cfg['paths']['cache_root'],mode,self.cfg['cache']['mmap_limit'])
            expected=set(self.frame.cell_id) if mode!='FULL-SUBSET' else data.subset_ids(self.frame,self.cfg['sampling']['subset_cells'],self.cfg['sampling']['seed'])
            if set(source.frame.cell_id)!=expected:source.close();raise ValueError('Celle della cache diverse dal piano')
            self.sources[mode]=source
        return self.sources[mode]

    def location(self,outer,train_fold,val_fold,mode,seed):return self.root/f'outer_{outer}'/f'inner_{train_fold}_{val_fold}'/mode/f'seed_{seed}'

    def inner_fit(self,outer,train_fold,val_fold,mode,seed,deadline=None):
        folder=self.location(outer,train_fold,val_fold,mode,seed);tag=signature({'study':self.identity,'outer':outer,'train':train_fold,'validation':val_fold,'mode':mode,'seed':seed})
        final=folder/'result.json';pred_path=folder/'predictions.parquet'
        if final.exists():
            record=json.loads(final.read_text(encoding='utf-8'))
            if record['identity']!=tag or digest(pred_path)!=record['sha256'] or digest(folder/record['best_file'])!=record['best_sha256']:raise ValueError('Fit finalizzato corrotto')
            return pd.read_parquet(pred_path),record['best_epoch']
        check_deadline(deadline);train_source=self.sampled(mode)
        train=train_source.frame[train_source.frame.cv_fold==train_fold].copy();val=self.frame[self.frame.cv_fold==val_fold].copy()
        training.split_guard(train,val,self.frame[self.frame.cv_fold==outer])
        dense=data.full_validation_cache(self.source,val,self.cfg,deadline)
        try:
            self.progress.update(force=True,phase='inner',fold=outer,inner_train=train_fold,inner_validation=val_fold,mode=mode,seed=seed)
            model,norm,epoch=training.fit(train_source,dense,train,val,self.cfg,folder,tag,seed,self.resume,deadline=deadline,progress=self.progress)
            pred=training.predict(model,dense,val,norm,self.cfg,deadline);atomic_frame(pred_path,pred)
            best=json.loads((folder/'best.json').read_text(encoding='utf-8'))
            record={'identity':tag,'sha256':digest(pred_path),'best_epoch':epoch,'best_file':best['file'],
                'best_sha256':digest(folder/best['file']),'training_cells':len(train),'validation_cells':len(val),'metrics':metrics(pred)}
            save_json(final,record);return pred,epoch
        finally:dense.close()

    def inner_results(self,outer,mode,seed):
        parts=[];epochs=[]
        for a,b in nested_pairs(outer):
            folder=self.location(outer,a,b,mode,seed);record=json.loads((folder/'result.json').read_text(encoding='utf-8'))
            path=folder/'predictions.parquet'
            if digest(path)!=record['sha256']:raise ValueError('Predizioni interne modificate')
            parts.append(pd.read_parquet(path));epochs.append(record['best_epoch'])
        return pd.concat(parts,ignore_index=True),training.refit_epochs(*epochs)

    def controls(self,outer,deadline=None):
        for a,b in nested_pairs(outer):
            for seed in [42,43]:self.inner_fit(outer,a,b,'FULL',seed,deadline)
        first,_=self.inner_results(outer,'FULL',42);second,_=self.inner_results(outer,'FULL',43)
        value={'study':self.identity,'outer':outer,'full_42':metrics(first)['macro_aoi_rmse'],'full_43':metrics(second)['macro_aoi_rmse']}
        value['margin']=abs(value['full_42']-value['full_43']);value['interpretation']='exploratory seed difference, NOT total uncertainty or non-inferiority'
        freeze_json(self.root/f'outer_{outer}'/'margin.json',value)
        return value

    def candidates(self,outer,deadline=None):
        self.controls(outer,deadline)
        for a,b in nested_pairs(outer):
            for mode in self.cfg['sampling']['modes']:self.inner_fit(outer,a,b,mode,42,deadline)

    def screen(self,deadline=None):
        outer=self.cfg['execution']['screen_outer_fold'];a,b=nested_pairs(outer)[0]
        jobs=[('FULL',42),('FULL',43)]+[(m,42) for m in self.cfg['sampling']['modes']]
        for i,(mode,seed) in enumerate(jobs):
            self.progress.update(force=True,jobs_done=i,jobs_total=len(jobs))
            self.inner_fit(outer,a,b,mode,seed,deadline)
        save_json(self.root/'screen_complete.json',{'study':self.identity,'outer_fold_never_used_for_selection':outer,'inner_train':a,'inner_validation':b,
            'jobs':len(jobs),'status':'exploratory_internal_screen_only','global_elimination':False})

    def costs(self,mode):
        source=self.sampled(mode);return int(source.frame.n_valid.sum())*2956

    def cost_rank(self,mode,candidate,full):
        base=full.set_index('cell_id');excess=[]
        for _,g in candidate.groupby('aoi_id'):
            f=base.loc[g.cell_id];excess.append(float(np.sqrt(np.mean((g.target-g.prediction)**2))-np.sqrt(np.mean((f.target-f.prediction)**2))))
        return (self.costs(mode),float(np.quantile(excess,.9)),metrics(candidate)['macro_aoi_rmse'],mode)

    def confirm(self,outer,deadline=None):
        decision=self.root/f'outer_{outer}'/'decision.json'
        saved=read_sealed(decision)
        if saved is not None:
            if saved['study']!=self.identity:raise ValueError('Decisione di uno studio diverso')
            return saved
        margin=self.controls(outer,deadline);self.candidates(outer,deadline);full42,_=self.inner_results(outer,'FULL',42)
        base=metrics(full42)['macro_aoi_rmse'];eligible=[];trials=[]
        for mode in PRIMARY:
            p,_=self.inner_results(outer,mode,42);delta=metrics(p)['macro_aoi_rmse']-base
            if delta<=margin['margin']:eligible.append(mode)
            trials.append({'mode':mode,'initial_delta':delta,'raw_bytes_estimate':self.costs(mode)})
        chosen=None
        for mode in sorted(eligible,key=lambda m:self.cost_rank(m,self.inner_results(outer,m,42)[0],full42)):
            for a,b in nested_pairs(outer):
                self.inner_fit(outer,a,b,'FULL',44,deadline)
                for seed in [43,44]:self.inner_fit(outer,a,b,mode,seed,deadline)
            deltas=[]
            for seed in [42,43,44]:
                c,_=self.inner_results(outer,mode,seed);f,_=self.inner_results(outer,'FULL',seed)
                deltas.append(metrics(c)['macro_aoi_rmse']-metrics(f)['macro_aoi_rmse'])
            trials.append({'mode':mode,'confirmation_deltas':deltas,'mean_delta':float(np.mean(deltas))})
            if np.mean(deltas)<=margin['margin']:chosen=mode;break
        value={'study':self.identity,'outer':outer,'selected':chosen,'margin':margin,'trials':trials,'outer_targets_used':False,
            'approved_for_acquisition':False,'interpretation':'exploratory recommendation; R2 and extreme errors require human review'}
        freeze_json(decision,value);return value

    def external(self,outer,deadline=None):
        path=self.root/f'outer_{outer}'/'decision.json'
        if not path.exists():raise ValueError('Sigillare prima la decisione interna')
        decision=read_sealed(path)
        if decision is None:raise ValueError('Decisione interna senza sigillo completo')
        decision_sha=digest(path)
        if decision['study']!=self.identity or decision['outer_targets_used']:raise ValueError('Decisione incompatibile')
        modes=['FULL']+([decision['selected']] if decision['selected'] else [])
        for mode in modes:
            for seed in [42,43,44]:
                # FULL44 may be absent when no candidate was admitted. Complete it internally, never using external labels.
                for a,b in nested_pairs(outer):self.inner_fit(outer,a,b,mode,seed,deadline)
                _,epochs=self.inner_results(outer,mode,seed);sampled=self.sampled(mode)
                train=sampled.frame[(sampled.frame.partition=='train')&(sampled.frame.cv_fold!=outer)].copy();test=self.frame[self.frame.cv_fold==outer].copy();training.split_guard(train,external=test)
                folder=self.root/f'outer_{outer}'/'refit'/mode/f'seed_{seed}';tag=signature({'study':self.identity,'role':'refit','outer':outer,'mode':mode,'seed':seed,'epochs':epochs,'decision':decision_sha})
                final=folder/'external_complete.json'
                if final.exists():
                    mark=json.loads(final.read_text(encoding='utf-8'))
                    if mark['identity']!=tag or digest(folder/'external.parquet')!=mark['sha256']:raise ValueError('Valutazione esterna corrotta')
                    continue
                model,norm,_=training.fit(sampled,self.source,train,None,self.cfg,folder,tag,seed,self.resume,epochs,deadline,self.progress)
                if digest(path)!=decision_sha:raise ValueError('Decisione modificata durante refit')
                predictions=training.predict(model,self.source,test,norm,self.cfg,deadline,self.progress,'external_evaluation')
                atomic_frame(folder/'external.parquet',predictions)
                save_json(final,{'identity':tag,'sha256':digest(folder/'external.parquet'),'decision_sha256':decision_sha,'metrics':metrics(predictions),'refit_epochs':epochs})

    def global_results(self,mode,seed):
        parts=[]
        for outer in [0,1,2]:
            p,_=self.inner_results(outer,mode,seed);p['original_cell_id']=p.cell_id;p['cell_id']=str(outer)+'/'+p.cell_id.astype(str);parts.append(p)
        return pd.concat(parts,ignore_index=True)

    def recommend(self,deadline=None):
        path=self.root/'recommendation.json'
        saved=read_sealed(path)
        if saved is not None:return saved
        for outer in [0,1,2]:self.confirm(outer,deadline)
        f42=self.global_results('FULL',42);f43=self.global_results('FULL',43)
        margin=abs(metrics(f42)['macro_aoi_rmse']-metrics(f43)['macro_aoi_rmse'])
        freeze_json(self.root/'global_margin.json',{'study':self.identity,'margin':margin,'calibration_seeds':[42,43]})
        eligible=[m for m in PRIMARY if metrics(self.global_results(m,42))['macro_aoi_rmse']-metrics(f42)['macro_aoi_rmse']<=margin]
        selected=None;comparisons=[]
        for mode in sorted(eligible,key=lambda m:self.cost_rank(m,self.global_results(m,42),f42)):
            for outer in [0,1,2]:
                for a,b in nested_pairs(outer):
                    self.inner_fit(outer,a,b,'FULL',44,deadline)
                    for seed in [43,44]:self.inner_fit(outer,a,b,mode,seed,deadline)
            results=[]
            for seed in [42,43,44]:
                p=self.global_results(mode,seed);f=self.global_results('FULL',seed)
                results.append({'seed':seed,'candidate':metrics(p),'full':metrics(f),
                    'paired':compare(f,p,self.cfg['decision']['bootstrap_samples'],self.cfg['decision']['bootstrap_seed'])})
            delta=float(np.mean([r['paired']['macro_aoi_delta'] for r in results]))
            comparisons.append({'mode':mode,'mean_delta':delta,'seeds':results,'raw_bytes_estimate':self.costs(mode),
                'seed_variability':{'candidate_rmse_std':float(np.std([r['candidate']['macro_aoi_rmse'] for r in results],ddof=1)),
                    'full_rmse_std':float(np.std([r['full']['macro_aoi_rmse'] for r in results],ddof=1))}})
            if delta<=margin:selected=mode;break
        value={'study':self.identity,'selected':selected,'margin':margin,'comparisons':comparisons,'basis':'internal_only, each development cell predicted in two training contexts',
            'n_unique_development_cells':len(self.frame),'n_internal_prediction_contexts_per_seed':2,
            'raw_bytes_estimate':self.costs(selected) if selected else None,'approved_for_acquisition':False,
            'future_partial_last_n_blocks_recommended':1,'proof_of_noninferiority':False,'human_review':['R2','extreme_errors','per_volume_storage','GPU pilot']}
        freeze_json(path,value);return value
