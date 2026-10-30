# Scopo: probing MSA V2 sullo stesso sviluppo/holdout del riferimento V1 Liguria.
# Fasi: verifica membership/fold, CV geografica 3-fold, refit sviluppo, test Liguria e confronto storico.
# Input: msa_v1_liguria.yaml, manifest celle/fold, embedding frozen e riferimento V1 verificato.
# Output: run separato, OOF per cella/AOI, modelli, metriche e confronto con limiti espliciti.
# Parametri: probe_msa.py run --config configs/msa_v1_liguria.yaml --data-root ../data.
# Risorse: GPU per embedding, CPU per HistGB; la CV non aggiorna l'encoder né vede i target test.
# Ripresa: feature SQLite per cella; smoke esplicitamente non confrontabile con il risultato V1 completo.
import json
from pathlib import Path
import time
import joblib
import numpy as np
import pandas as pd
from .common import ROOT,cells,digest,make_run,metrics,report,save_json


def validate_protocol(cfg,frame,smoke=False):
    development=frame[frame.partition=='train'];test=frame[frame.partition=='test']
    if set(frame.partition)!={'train','test'} or not len(development) or not len(test):
        raise ValueError('Protocollo storico: sviluppo train e test Liguria richiesti, non validation fissa')
    if 'cv_fold' not in frame or set(development.cv_fold.astype(int))!={0,1,2}:
        raise ValueError('Servono tutti e tre i fold geografici congelati')
    if not development.groupby('aoi_id').cv_fold.nunique().eq(1).all():
        raise ValueError('Una AOI è divisa fra fold: possibile leakage')
    if set(development.aoi_id)&set(test.aoi_id):raise ValueError('AOI test presente nello sviluppo')
    if set(test.aoi_id)!={'liguria_levante'} or not test.cv_fold.eq(-1).all():
        raise ValueError('Test deve essere soltanto Liguria, fuori dalla CV')
    if not smoke:
        if frame.partition.value_counts().to_dict()!=cfg['evaluation']['expected_counts']:
            raise ValueError('Numerosità diversa dal V1: benchmark non equivalente')
        if development.aoi_id.nunique()!=17:raise ValueError('Servono le 17 AOI storiche di sviluppo')
        reference=json.loads((ROOT/cfg['evaluation']['reference']).read_text(encoding='utf-8'))
        if digest(cfg['cells'])!=reference['cells_sha256']:raise ValueError('Membership/fold del benchmark modificati')
    source=cells(cfg)
    original=source[['cell_id','partition','cv_fold','aoi_id']].copy()
    actual=frame[['cell_id','partition','cv_fold','aoi_id']].merge(original,on='cell_id',validate='one_to_one',suffixes=('_actual','_original'))
    if len(actual)!=len(frame) or not all(actual[c+'_actual'].eq(actual[c+'_original']).all() for c in ['partition','cv_fold','aoi_id']):
        raise ValueError('Membership o fold differenti dal manifest congelato')
    for fold in range(3):
        training=development[development.cv_fold!=fold];validation=development[development.cv_fold==fold]
        if not len(training) or not len(validation) or set(training.aoi_id)&set(validation.aoi_id):
            raise ValueError('Fold geografico vuoto o AOI sovrapposte')


def estimator(cfg,name):
    from sklearn.dummy import DummyRegressor
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    if name=='DummyMean':return DummyRegressor(strategy='mean')
    if name=='HistGB':return HistGradientBoostingRegressor(**cfg['probing']['histgb_params'])
    if name=='Ridge':return make_pipeline(StandardScaler(),Ridge(alpha=cfg['probing']['ridge_alpha']))
    raise ValueError('Profilo storico: usare HistGB e opzionalmente Ridge; Dummy sempre incluso')


def historical_probe(cfg,args):
    from .data import prepared
    from .encoder import embeddings
    from .experiments import prediction_frame,FEATURES
    frame=prepared(cfg,args.smoke);validate_protocol(cfg,frame,args.smoke)
    features,checkpoint=embeddings(cfg,frame,args.checkpoint)
    frame=frame.merge(features,on='cell_id',validate='one_to_one')
    if len(frame)!=len(features) or frame[FEATURES].isna().any().any():raise ValueError('Feature non appaiate')
    out=make_run(cfg,'msa_v1_liguria_smoke' if args.smoke else 'msa_v1_liguria');started=time.perf_counter()
    development=frame[frame.partition=='train'];names=['DummyMean',*cfg['probing']['models']]
    predictions=[];oof=[];scores=[];folds=[]
    for name in names:
        for fold in range(3):
            training=development[development.cv_fold!=fold];validation=development[development.cv_fold==fold]
            model=estimator(cfg,name);model.fit(training[FEATURES].to_numpy(dtype=np.float32),training.target.to_numpy(dtype=np.float32))
            pred=model.predict(validation[FEATURES].to_numpy(dtype=np.float32));record=prediction_frame(validation,pred,name)
            record['partition']='validation';record['cv_fold']=fold;oof.append(record)
            for aoi,group in record.groupby('aoi_id'):
                scores.append({'model':name,'fold':fold,'aoi_id':aoi,**metrics(group.target,group.prediction)})
            folds.append({'model':name,'fold':fold,'n_train':len(training),'n_validation':len(validation),
                          'validation_aois':sorted(validation.aoi_id.unique().tolist())})
        final=estimator(cfg,name);final.fit(development[FEATURES].to_numpy(dtype=np.float32),development.target.to_numpy(dtype=np.float32))
        joblib.dump(final,out/f'{name}.joblib')
        predictions.append(prediction_frame(frame,final.predict(frame[FEATURES].to_numpy(dtype=np.float32)),name))
    cv=pd.DataFrame(scores);macro=cv.groupby('model')[['mae','rmse']].mean()
    selected=macro.drop(index='DummyMean').rmse.idxmin()
    cv.to_csv(out/'cv_metrics_by_aoi.csv',index=False);macro.to_csv(out/'cv_macro_metrics.csv')
    pd.concat(oof,ignore_index=True).to_csv(out/'predictions_cv.csv',index=False)
    save_json(out/'folds.json',folds)
    save_json(out/'selection.json',{'criterion':'macro mean RMSE over 17 held AOIs, fixed 3-fold GroupKFold',
        'selected':selected,'scores':macro.rmse.to_dict(),'checkpoint_sha256':digest(checkpoint),
        'seconds':time.perf_counter()-started,'smoke':args.smoke})
    save_json(out/'protocol.json',{'expected_counts':cfg['evaluation']['expected_counts'],
        'actual_counts':frame.partition.value_counts().to_dict(),'validation':'rotating OOF; no independent fixed validation set',
        'n_validation_oof':len(development),'n_development_aois':development.aoi_id.nunique(),
        'cells_sha256':digest(cfg['cells']),'frozen_encoder':True,'smoke':args.smoke})
    report(out,pd.concat(predictions,ignore_index=True))
    comparison_report(cfg,out,args.smoke)
    print(f'Run: {out}',flush=True);return out


def comparison_report(cfg,out,smoke):
    reference=json.loads((ROOT/cfg['evaluation']['reference']).read_text(encoding='utf-8'))
    historic=pd.DataFrame([{'model':name,'partition':'test',**values} for name,values in reference['metrics'].items()])
    historic.to_csv(out/'historical_reference.csv',index=False)
    current=pd.read_csv(out/'metrics.csv');rows=[]
    for record in current[current.partition=='test'].to_dict('records'):
        name=record['model'];old=reference['metrics'].get(name)
        if old is None:continue
        complete=not smoke and record['n']==reference['test_cells']
        for metric in ['mae','rmse','mape_pct','bias','r2']:
            rows.append({'model':name,'metric':metric,'v1':old.get(metric),'v2':record.get(metric),
                         'delta_v2_minus_v1':record[metric]-old[metric] if complete and old.get(metric) is not None else None,
                         'n_v1':old['n'],'n_v2':record['n'],
                         'comparison':'same_cells_different_pixel_pooling' if complete else 'smoke_not_comparable'})
    pd.DataFrame(rows).to_csv(out/'comparison_v1_v2.csv',index=False)
    with (out/'report.md').open('a',encoding='utf-8') as output:
        output.write('\n## Protocollo V1 Liguria\n\n')
        output.write('Sviluppo V1: 127.909 celle; test Liguria: 4.991; validation: GroupKFold(3) per AOI. ')
        output.write('OOF e conteggi reali in predictions_cv.csv, folds.json e protocol.json. La selezione usa solo RMSE macro-AOI OOF.\n\n')
        output.write('Il riferimento storico corretto è historical_reference.csv, run '+reference['holdout_run']+'. ')
        output.write('V1 usa pooling denso; V2 usa 25 punti stratificati salvo override. Anche con stesse celle/fold, un delta non isola il solo encoder. ')
        output.write('Per affermare superiorità attribuibile al modello occorre ricalcolare V1 e V2 con lo stesso supporto di pixel e preprocessing.\n')
        if smoke:output.write('\nSMOKE: nessun confronto numerico valido con le 4.991 celle V1.\n')


def evaluate_historical(cfg,args):
    from .data import prepared
    frame=prepared(cfg,args.smoke);validate_protocol(cfg,frame,args.smoke)
    out=Path(args.run_dir).resolve();predictions=pd.read_csv(out/'predictions.csv',dtype={'cell_id':str})
    for _,group in predictions.groupby('model'):
        joined=group.merge(frame[['cell_id','partition','target']],on='cell_id',validate='one_to_one',suffixes=('_saved','_current'))
        if len(joined)!=len(frame) or not joined.partition_saved.eq(joined.partition_current).all() or not np.allclose(joined.target_saved,joined.target_current,rtol=0,atol=1e-10):
            raise ValueError('Predizioni e manifest non appaiati')
    report(out,predictions);comparison_report(cfg,out,args.smoke);return out
