# Scopo: metriche MSA e confronto appaiato, distinguendo sempre macro-AOI da pooled.
# Fasi: verifica ID/target -> metriche e code estreme -> bootstrap cluster -> report senza non-inferiorita.
# Input: predizioni dense delle sole AOI di sviluppo; output: JSON/CSV e intervalli descrittivi.
# Parametri: bootstrap 2000, seed 20260925; esempio compare(full,candidate).
# Risorse: CPU sui vettori, nessun modello/rete. Ripresa: risultati deterministici da predizioni immutabili.
import numpy as np
import pandas as pd
from biomap_v2.dense.evaluation import full_metrics,bootstrap_delta


def metrics(frame):
    if not len(frame) or frame.cell_id.duplicated().any():raise ValueError('Popolazione vuota/duplicata')
    if frame.aoi_id.eq('liguria_levante').any() or not frame.partition.eq('train').all():raise ValueError('Liguria vietata')
    if not np.isfinite(frame[['target','prediction']]).all().all():raise ValueError('Target/predizioni non finiti')
    y=frame.target.to_numpy();p=frame.prediction.to_numpy();e=np.abs(y-p)
    local=[full_metrics(g.target,g.prediction) for _,g in frame.groupby('aoi_id')]
    result=full_metrics(y,p)
    result.update(n_cells=len(frame),n_aois=frame.aoi_id.nunique(),macro_aoi_rmse=float(np.mean([g['rmse'] for g in local])),
        macro_aoi_mae=float(np.mean([g['mae'] for g in local])),error_quantiles=dict(zip(['p50','p90','p95','p99'],map(float,np.quantile(e,[.5,.9,.95,.99])))),
        error_gt_03_fraction=float(np.mean(e>.3)),error_gt_04_fraction=float(np.mean(e>.4)),
        pearson=float(np.corrcoef(y,p)[0,1]) if len(y)>1 and np.std(y)>0 and np.std(p)>0 else None)
    return result


def compare(full,candidate,n_boot=2000,seed=20260925):
    if full.cell_id.duplicated().any() or candidate.cell_id.duplicated().any() or set(full.cell_id)!=set(candidate.cell_id):raise ValueError('Popolazioni diverse: nessun inner join')
    a=full.set_index('cell_id');b=candidate.set_index('cell_id').loc[a.index]
    for column in ['target','aoi_id','cv_fold','partition']:
        if not a[column].equals(b[column]):raise ValueError('Target/geografia diversi: '+column)
    rows=[]
    for aoi,g in a.groupby('aoi_id'):
        h=b.loc[g.index];rows.append(float(np.sqrt(np.mean((h.target-h.prediction)**2))-np.sqrt(np.mean((g.target-g.prediction)**2))))
    rng=np.random.default_rng(seed);values=np.asarray(rows)
    samples=rng.choice(values,size=(n_boot,len(values)),replace=True).mean(1)
    pooled=bootstrap_delta(a.target,a.prediction,b.prediction,a.aoi_id,n_boot=n_boot,seed=seed)
    return {'macro_aoi_delta':float(values.mean()),'macro_aoi_cluster_ci95':np.quantile(samples,[.025,.975]).tolist(),
        'pooled_bootstrap':pooled,'n_boot':n_boot,'seed':seed,'interpretation':'exploratory, conditional on fitted models; NOT a proof of non-inferiority'}


def tables(frame):
    by_aoi=[{'aoi_id':a,**metrics(g)} for a,g in frame.groupby('aoi_id')]
    cuts=[(0,.1),(.1,.2),(.2,.4),(.4,.6),(.6,.8),(.8,1.0000000001)];bins=[]
    for low,high in cuts:
        g=frame[(frame.target>=low)&(frame.target<high)]
        if len(g):bins.append({'lower':low,'upper':min(high,1.),**metrics(g)})
    return pd.DataFrame(by_aoi),pd.DataFrame(bins)
