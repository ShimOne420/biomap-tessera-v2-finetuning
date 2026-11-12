# Scopo: probing tabellare frozen, tre fold originali, refit e test ligure separato.
# Gli scaler e il Dummy sono stimati dentro ogni training; nessun tuning su Liguria.
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from biomap_v2.common import ROOT, digest, save_json
from biomap_v2.dense.evaluation import make_estimator, full_metrics
from .state import MEANS

NAMES=['DummyMean','Ridge','HistGB','XGBoost','MLP']

# La stessa matrice deve contenere tutte le celle: nessun inner join che ne perda alcune.
def aligned(reference, candidate):
    if candidate.cell_id.duplicated().any() or set(candidate.cell_id)!=set(reference.cell_id):
        raise ValueError('Feature incomplete o duplicate')
    candidate=candidate.set_index('cell_id').loc[reference.cell_id].reset_index()
    for c in ['aoi_id','partition','cv_fold','n_valid','n_total']:
        if not np.array_equal(reference[c],candidate[c]):raise ValueError(f'Protocollo discordante: {c}')
    if not np.allclose(reference[['lon','lat','target']],candidate[['lon','lat','target']],rtol=0,atol=1e-8):raise ValueError('Target/geometria diversi')
    if not np.isfinite(candidate[MEANS]).all().all():raise ValueError('Feature non finite')
    return candidate

# Parametri comuni congelati prima di leggere le metriche di test.
def params(seed):
    ref=json.loads((ROOT/'configs/msa_v1_liguria_reference.json').read_text())
    return {'histgb':ref['histgb_params'],'ridge_alpha':100.,'seed':seed,
        'xgb':{'n_estimators':400,'max_depth':2,'learning_rate':.05,'objective':'reg:squarederror','n_jobs':1,'random_state':seed},
        'mlp':{'hidden_layer_sizes':[256,128],'max_iter':500,'early_stopping':True,'random_state':seed}}

# Ogni fit salvato è riutilizzabile solo insieme al fingerprint dati/configurazione.
def probe(args, reference):
    work=Path(args.work_root); encoder=args.encoder
    f=reference if encoder=='v1' else aligned(reference,pd.read_parquet(work/'v2.parquet'))
    run=work/'probing'/encoder;run.mkdir(parents=True,exist_ok=True)
    conf=params(args.seed); signature={'parameters':conf,'encoder':encoder,'reference_sha256':digest(Path(args.reference)/'reference.json'),
        'feature_sha256':digest(Path(args.reference)/'v1.parquet' if encoder=='v1' else work/'v2.parquet')}
    protocol=run/'protocol.json'
    if protocol.exists() and json.loads(protocol.read_text())!=signature:raise ValueError('Run incompatibile: usare work-root nuova')
    save_json(protocol,signature); records=[]; scores=[]
    for name in NAMES:
        for fold in range(3):
            train=f[(f.partition=='train')&(f.cv_fold!=fold)]; val=f[(f.partition=='train')&(f.cv_fold==fold)]
            path=run/f'{name}_fold{fold}.joblib'
            model=fit(path,train,name,conf)
            rec=val[['cell_id','aoi_id','lon','lat','target','cv_fold']].copy()
            rec['prediction']=model.predict(val[MEANS].to_numpy(dtype=np.float32));rec['model']=name
            rec['partition']='validation';rec['encoder']=encoder;records.append(rec)
            for aoi,g in rec.groupby('aoi_id'):
                scores.append({'encoder':encoder,'model':name,'cv_fold':fold,'aoi_id':aoi,**full_metrics(g.target,g.prediction)})
        fit(run/f'{name}_final.joblib',f[f.partition=='train'],name,conf)
    oof=pd.concat(records);oof.to_csv(run/'oof_predictions.csv',index=False)
    fold_scores=[];bins=[]
    for (model,fold),g in oof.groupby(['model','cv_fold']):
        fold_scores.append({'model':model,'cv_fold':fold,**full_metrics(g.target,g.prediction)})
    for model,g in oof.groupby('model'):
        labels=pd.cut(g.target,[-.001,.01,.1,.2,.4,.6,.8,1.001])
        for label,part in g.groupby(labels,observed=True):
            bins.append({'model':model,'target_bin':str(label),**full_metrics(part.target,part.prediction)})
    pd.DataFrame(fold_scores).to_csv(run/'metrics_by_fold.csv',index=False)
    pd.DataFrame(bins).to_csv(run/'metrics_by_target_bin.csv',index=False)
    by=pd.DataFrame(scores);by.to_csv(run/'metrics_by_aoi.csv',index=False)
    macro=by.groupby('model')[['mae','rmse','mape_pct','bias','r2','pearson_r']].mean()
    macro.to_csv(run/'macro_metrics.csv')
    selected=macro.drop(index='DummyMean').rmse.idxmin()
    save_json(run/'selection.json',{'selected':selected,'criterion':'macro mean RMSE over 17 held AOIs','n_development':127909})
    return run

# L'esistenza del joblib da sola non è sufficiente per riutilizzare un fit interrotto.
def fit(path, train, name, conf):
    meta=path.with_suffix('.json')
    if path.exists() and meta.exists():
        if json.loads(meta.read_text())['sha256']!=digest(path):raise ValueError('Modello corrotto')
        return joblib.load(path)
    model=make_estimator(name,conf['histgb'],conf['ridge_alpha'],conf['seed'],conf['mlp'],conf['xgb'])
    model.fit(train[MEANS].to_numpy(dtype=np.float32),train.target.to_numpy(dtype=np.float32))
    tmp=path.with_suffix('.partial');joblib.dump(model,tmp);tmp.replace(path)
    save_json(meta,{'sha256':digest(path),'n_training':len(train)})
    return model

# Nessun rifit o early stopping: tutti i modelli finali sono quelli congelati dalla CV.
def liguria(args, reference):
    work=Path(args.work_root);f=reference if args.encoder=='v1' else aligned(reference,pd.read_parquet(work/'v2.parquet'))
    f=f[f.partition=='test'];run=work/'probing'/args.encoder;records=[]
    if not (run/'selection.json').exists():raise ValueError('Eseguire prima la CV e il refit')
    for name in NAMES:
        path=run/f'{name}_final.joblib';meta=json.loads(path.with_suffix('.json').read_text())
        if digest(path)!=meta['sha256']:raise ValueError('Modello corrotto')
        model=joblib.load(path);r=f[['cell_id','aoi_id','lon','lat','target','cv_fold']].copy()
        r['prediction']=model.predict(f[MEANS].to_numpy(dtype=np.float32));r['model']=name
        r['encoder']=args.encoder;r['partition']='liguria_benchmark';records.append(r)
    pd.concat(records).to_csv(run/'liguria_predictions.csv',index=False)

# Confronto solo su righe appaiate e parametri identici; i risultati storici non sono inventati.
def compare(work):
    work=Path(work);protocol=[]
    for e in ['v1','v2']:protocol.append(json.loads((work/'probing'/e/'protocol.json').read_text()))
    if protocol[0]['reference_sha256']!=protocol[1]['reference_sha256'] or protocol[0]['parameters']!=protocol[1]['parameters']:raise ValueError('Iperparametri diversi')
    rows=[]
    for filename in ['oof_predictions.csv','liguria_predictions.csv']:
        files=[work/'probing'/e/filename for e in ['v1','v2']]
        if not all(p.exists() for p in files):continue
        first,second=[pd.read_csv(p,dtype={'cell_id':str}) for p in files]
        joined=first.merge(second,on=['cell_id','model','partition','aoi_id','cv_fold'],validate='one_to_one',suffixes=('_v1','_v2'))
        if len(joined)!=len(first) or len(joined)!=len(second) or not np.array_equal(joined.target_v1,joined.target_v2):raise ValueError('Predizioni non appaiate')
        for (model,part,aoi),g in joined.groupby(['model','partition','aoi_id']):
            a=full_metrics(g.target_v1,g.prediction_v1);b=full_metrics(g.target_v2,g.prediction_v2)
            for metric in ['mae','rmse','mape_pct','bias','r2','pearson_r']:
                x,y=a[metric],b[metric];rows.append({'model':model,'partition':part,'aoi_id':aoi,'metric':metric,'v1':x,'v2':y,'delta':None if x is None or y is None else y-x,'delta_pct':None if not x or y is None else 100*(y-x)/abs(x)})
    pd.DataFrame(rows).to_csv(work/'comparison_v1_v2_dense_127909.csv',index=False)
    # Intervalli bootstrap appaiati (AOI / blocchi 2 km) e riproduzione del V1 storico: vedi uncertainty.py.
    from .uncertainty import paired_uncertainty, historical_v1_check
    paired_uncertainty(work);historical_v1_check(work)
