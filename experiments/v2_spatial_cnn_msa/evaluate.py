# Scopo: metriche, OOF, confronto appaiato e valutazione Liguria dei modelli su embedding V2 per pixel (stesse funzioni dell'Esp. B).
# Fasi: predizioni a cella -> conteggi/metriche fold/AOI -> macro e pooled distinti -> grafici/report.
# Input: run-dir privato e cache; output: CSV/PNG/JSON; nessuna selezione usando Liguria.
# Parametri: oof|liguria|compare, --run-name, --preset; Liguria è un'azione esplicita.
# Risorse: CPU per metriche, device configurato per inferenza CNN; nessun training o encoder.
# Ripresa: idempotente sulle predizioni; verifiche di identità prima del caricamento dei checkpoint.
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from runtime import common_parser,configuration,cache_path,save_json,exclusive,tracked


def metrics(y,pred):
    y=np.asarray(y,dtype=np.float64);pred=np.asarray(pred,dtype=np.float64)
    if not len(y) or y.shape!=pred.shape or not np.isfinite(y).all() or not np.isfinite(pred).all():raise ValueError('Metriche: input non valido')
    nonzero=y!=0
    return {'n':len(y),'mae':float(mean_absolute_error(y,pred)),'rmse':float(np.sqrt(mean_squared_error(y,pred))),
        'mape_pct':float(np.mean(np.abs((pred[nonzero]-y[nonzero])/y[nonzero]))*100) if nonzero.any() else None,
        'mape_n':int(nonzero.sum()),'mape_excluded_zero':int((~nonzero).sum()),'r2':float(r2_score(y,pred)) if len(y)>1 and np.std(y)>0 else None,
        'pearson':float(np.corrcoef(y,pred)[0,1]) if len(y)>1 and np.std(y)>0 and np.std(pred)>0 else None,
        'bias':float(np.mean(pred-y)),'prediction_below_zero':float((pred<0).mean()),'prediction_above_one':float((pred>1).mean())}


def macro_rmse(frame,pred):
    return float(np.mean([np.sqrt(np.mean((g.target.to_numpy()-np.asarray(pred)[g.index])**2)) for _,g in frame.reset_index(drop=True).groupby('aoi_id')]))


def summarize(predictions,out,prefix):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);predictions=predictions.copy()
    keys=['model','cell_id']
    if predictions.duplicated(keys).any():raise ValueError('Predizioni ripetute per cella/modello')
    # No comparison between different eligible populations.
    sets={name:set(g.cell_id) for name,g in predictions.groupby('model')}
    if len({frozenset(x) for x in sets.values()})!=1:raise ValueError('Modelli valutati su celle diverse')
    rows=[];by_aoi=[];by_fold=[];diagnostics=[]
    for model,g in predictions.groupby('model'):
        row={'model':model,**metrics(g.target,g.prediction)}
        local=[]
        for aoi,part in g.groupby('aoi_id'):
            m={'model':model,'aoi_id':aoi,**metrics(part.target,part.prediction)};by_aoi.append(m);local.append(m)
        row['macro_aoi_rmse']=float(np.mean([x['rmse'] for x in local]));row['macro_aoi_mae']=float(np.mean([x['mae'] for x in local]));rows.append(row)
        if 'fold' in g:
            for fold,part in g.groupby('fold'):by_fold.append({'model':model,'fold':fold,**metrics(part.target,part.prediction)})
        for column in ['target','valid_fraction','patches','heterogeneity']:
            if column not in g:continue
            groups=pd.cut(g[column],[-np.inf,.2,.4,.6,.8,np.inf]) if column in {'target','valid_fraction'} else pd.qcut(g[column],4,duplicates='drop')
            for label,part in g.groupby(groups,observed=True):
                if len(part):diagnostics.append({'model':model,'grouping':column,'interval':str(label),**metrics(part.target,part.prediction)})
    predictions.to_csv(out/f'{prefix}_predictions.csv',index=False)
    pd.DataFrame(rows).to_csv(out/f'{prefix}_metrics.csv',index=False);pd.DataFrame(by_aoi).to_csv(out/f'{prefix}_by_aoi.csv',index=False)
    pd.DataFrame(by_fold).to_csv(out/f'{prefix}_by_fold.csv',index=False);pd.DataFrame(diagnostics).to_csv(out/f'{prefix}_diagnostics.csv',index=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for model,g in predictions.groupby('model'):
        fig,axes=plt.subplots(1,4,figsize=(17,4))
        axes[0].scatter(g.target,g.prediction,s=4,alpha=.3);axes[0].plot([0,1],[0,1],'k--');axes[0].set(xlabel='MSA target',ylabel='Predizione',title=model)
        residual=g.prediction-g.target;axes[1].hist(residual,bins=40);axes[1].set(xlabel='Predizione - target',title='Residui')
        if {'lon','lat'}<=set(g):
            v=axes[2].scatter(g.lon,g.lat,c=residual,s=3,cmap='coolwarm');fig.colorbar(v,ax=axes[2]);axes[2].set(xlabel='Longitudine',ylabel='Latitudine',title='Residui per cella')
        axes[3].hist(g.target,bins=30,alpha=.5,label='target');axes[3].hist(g.prediction,bins=30,alpha=.5,label='predizione');axes[3].legend();axes[3].set(title='Distribuzioni',xlabel='MSA')
        fig.tight_layout();fig.savefig(out/f'{prefix}_{model}.png',dpi=150);plt.close(fig)
    return pd.DataFrame(rows)


def compare(out,prefix='liguria'):
    out=Path(out);p=out/f'{prefix}_predictions.csv'
    f=pd.read_csv(p,dtype={'cell_id':str});table=summarize(f,out,prefix)
    baseline=table[table.model=='mean_histgb']
    if len(baseline):
        for metric in ['mae','rmse','mape_pct','r2','bias']:table['delta_vs_matched_histgb_'+metric]=table[metric]-baseline.iloc[0][metric]
    table.to_csv(out/f'{prefix}_comparison.csv',index=False)
    historical={'mae':.12206058082648942,'rmse':.15807773627994595,'mape_pct':26.451133059127336,'r2':.05660319323004637,'bias':-.046519923809239416,'n':4991}
    save_json(out/'historical_reference.json',{'values':historical,'role':'descriptive; observed Liguria; historical support can differ; no automatic paired deltas'})
    run_info=json.loads((out/'run.json').read_text()) if (out/'run.json').exists() else {}
    preset=run_info.get('config',{}).get('preset','not declared')
    text=['# V2 spatial CNN: confronto a cella','',f'Preset: {preset}. Partizione: {prefix}; {len(f.model.unique())} modelli, {f.cell_id.nunique()} celle identiche.','',
        'Embedding TESSERA V2 (r2-fast) precalcolati e congelati; patch e maschere dell Esperimento B. Il confronto controllato usa medie e CNN dagli stessi pixel, target e split. Nessun resizing o vettore medio convertito in immagine.',
        'OOF pooled e media delle RMSE AOI sono metriche differenti. MAPE esclude i target zero e ne riporta il denominatore.',
        'Liguria è un holdout già osservato, mai usato per fit/early stopping/scelta. Le diagnosi per intervallo sono descrittive.',
        'Il numero storico 26,45% (V1) non costituisce un confronto appaiato se supporto o copertura differiscono.', '',table.to_string(index=False)]
    if preset!='full':text.insert(4,'Scala ridotta: verifica/esplorazione, non risultato del benchmark completo.')
    (out/f'{prefix}_report.md').write_text('\n'.join(text),encoding='utf-8')
    return table


def main():
    p=common_parser('Valutazione e confronto controllato V2 spatial')
    p.add_argument('action',choices=['oof','liguria','compare']);p.add_argument('--run-name',required=True)
    args=p.parse_args();cfg=configuration(args)
    from train import run_directory,load_run,predict_external
    run=run_directory(cfg,args.run_name);manifest=load_run(cfg,run)
    with exclusive(cfg),tracked(cfg,'evaluate_'+args.action):
        if args.action=='liguria':predict_external(cfg,run,manifest)
        print(compare(run,'oof' if args.action=='oof' else 'liguria').to_string(index=False))


if __name__=='__main__':main()
