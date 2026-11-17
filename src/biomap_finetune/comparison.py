# Scopo: confronto principale frozen/partial realmente appaiato; nessuna intersezione favorevole.
# Fasi: verifica manifest/head/seed/supporto -> ID/target/fold uguali -> delta macro-AOI -> bootstrap AOI.
# Input: due nuovi run completi sullo stesso dataset; output: JSON/MD nel run richiedente, mai nel controllo.
# Parametri: run-id del controllo e 2000 repliche seed fisso; uso report --compare-run frozen_control.
# Risorse: CPU/predizioni per cella; nessun accesso a Sentinel. Ripresa: report rigenerabile, dati immutati.
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .io import separate_output,save_json


def compare(folder,control):
    folder,control=Path(folder),Path(control)
    left=json.loads((folder/'manifest.json').read_text(encoding='utf-8'));right=json.loads((control/'manifest.json').read_text(encoding='utf-8'))
    a=json.loads((folder/'effective_config.json').read_text(encoding='utf-8'));b=json.loads((control/'effective_config.json').read_text(encoding='utf-8'))
    for key in ['dataset_identity','aggregation','seed','profile','code']:
        if left[key]!=right[key]:raise ValueError('Confronto non appaiabile: '+key)
    if left['encoder_mode']!='partial' or right['encoder_mode']!='frozen':raise ValueError('Confronto richiede partial contro frozen')
    if a['aggregation']!=b['aggregation'] or a['training']['normalization']!=b['training']['normalization']:
        raise ValueError('Testa/normalizzazione differenti')
    for key in ['learning_rate_head','batch_size','epochs','precision','clip_grad','early_stopping']:
        if a['training'][key]!=b['training'][key]:raise ValueError('Budget/protocollo della testa differente: '+key)
    result={'support_identity':left['dataset_identity'],'comparison':'partial minus matched frozen','historical_comparisons':'NOT_VERIFIED; equal cells do not establish equal pixel support'}
    for name in ['oof','liguria']:
        path=folder/(name+'_predictions.parquet');baseline=control/(name+'_predictions.parquet')
        if not path.exists() or not baseline.exists():continue
        x=pd.read_parquet(path);y=pd.read_parquet(baseline)
        x=x[x.model=='predictor'].set_index('cell_id');y=y[y.model=='predictor'].set_index('cell_id')
        if x.index.has_duplicates or y.index.has_duplicates or set(x.index)!=set(y.index):raise ValueError('ID diversi: nessun inner join selettivo')
        y=y.loc[x.index]
        for key in ['target','aoi_id','cv_fold','partition','fold']:
            if not x[key].equals(y[key]):raise ValueError('Celle/target/fold diversi: '+key)
        rows=[]
        for aoi,g in x.groupby('aoi_id'):
            h=y.loc[g.index];rows.append({'aoi':aoi,'cells':len(g),
                'rmse_partial':float(np.sqrt(np.mean((g.target-g.prediction)**2))),
                'rmse_frozen':float(np.sqrt(np.mean((h.target-h.prediction)**2)))})
        delta=np.array([r['rmse_partial']-r['rmse_frozen'] for r in rows]);rng=np.random.default_rng(20260925)
        interval=np.quantile(np.mean(rng.choice(delta,(2000,len(delta))),axis=1),[.025,.975]).tolist() if len(delta)>1 else None
        result[name]={'cells':len(x),'aois':len(rows),'delta_macro_aoi_rmse':float(delta.mean()),'paired_aoi_bootstrap_95':interval,
            'aoi_rows':rows,'uncertainty_note':'AOI-cluster resampling of paired per-AOI RMSE differences; one AOI cannot estimate this interval'}
    if not any(k in result for k in ['oof','liguria']):raise ValueError('Nessuna coppia di predizioni completata')
    save_json(folder/'paired_comparison.json',result);return result
