# Scopo: tabelle/figure da predizioni effettive; storico indicativo separato dal confronto appaiato.
from pathlib import Path
import numpy as np
import pandas as pd
from biomap_v2.common import save_json
from biomap_v2.dense.evaluation import full_metrics

# Cella e ruolo della valutazione rimangono espliciti, anche per il refit e Liguria.
def record(frame,indices,prediction,model,experiment_id):
    result=frame.iloc[indices][['cell_id','aoi_id','cv_fold','lon','lat','target','partition']].copy()
    result['prediction']=prediction;result['model_name']=model;result['experiment_id']=experiment_id
    result['partition']=np.where(result.partition=='test','liguria_benchmark','validation')
    result['y_true']=result.target;result['y_pred']=result.prediction;result['residual']=result.prediction-result.target
    result['absolute_error']=result.residual.abs()
    result['absolute_percentage_error']=np.where(result.target!=0,100*result.absolute_error/result.target.abs(),np.nan)
    return result

# Metriche pooled/fold/AOI e macro distinte; MAPE esclude zero secondo definizione comune.
def export(out,predictions,selection):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out=Path(out)
    if 'dummy_prediction' not in predictions:
        dummy=predictions[predictions.model_name=='DummyMean'][['cell_id','partition','prediction']].rename(columns={'prediction':'dummy_prediction'})
        predictions=predictions.merge(dummy,on=['cell_id','partition'],how='left',validate='many_to_one')
    predictions.to_csv(out/'predictions.csv',index=False);rows=[]
    for (model,part),f in predictions.groupby(['model_name','partition']):
        rows.append({'model':model,'partition':part,'aggregation':'pooled','group':'ALL',**full_metrics(f.target,f.prediction)})
        groups=[]
        for aoi,g in f.groupby('aoi_id'):
            m=full_metrics(g.target,g.prediction);groups.append(m)
            rows.append({'model':model,'partition':part,'aggregation':'AOI','group':aoi,**m})
        if part=='validation':
            names=['mae','rmse','mape_pct','bias','r2','pearson_r']
            avg=pd.DataFrame(groups)[names].mean().to_dict()
            rows.append({'model':model,'partition':part,'aggregation':'macro_AOI','group':f'{f.aoi_id.nunique()} AOI',**avg})
            for fold,g in f.groupby('cv_fold'):rows.append({'model':model,'partition':part,'aggregation':'fold','group':str(fold),**full_metrics(g.target,g.prediction)})
        fig,axes=plt.subplots(1,2,figsize=(9,4));axes[0].scatter(f.target,f.prediction,s=1,alpha=.2)
        axes[0].plot([0,1],[0,1],'k--');axes[0].set(xlabel='MSA osservato',ylabel='MSA predetto')
        axes[1].hist(f.residual,bins=50);axes[1].set(xlabel='Residuo predetto − osservato',ylabel='Celle')
        fig.tight_layout();fig.savefig(out/f'{model}_{part}.png');plt.close(fig)
    metrics=pd.DataFrame(rows)
    # Dummy corrispondente alla stessa partizione e aggregazione, non una media test.
    dummy=metrics[metrics.model=='DummyMean'][['partition','aggregation','group','rmse']].rename(columns={'rmse':'dummy_rmse'})
    metrics=metrics.merge(dummy,on=['partition','aggregation','group'],how='left',validate='many_to_one')
    metrics['rmse_reduction_vs_dummy_pct']=100*(metrics.dummy_rmse-metrics.rmse)/metrics.dummy_rmse.replace(0,np.nan)
    metrics.to_csv(out/'metrics.csv',index=False)
    temp=out/'results.partial.xlsx'
    with pd.ExcelWriter(temp,engine='openpyxl') as writer:
        metrics.to_excel(writer,sheet_name='Risultati',index=False)
        for (model,part),g in predictions.groupby(['model_name','partition']):g.to_excel(writer,sheet_name=f'{model}_{part}'[:31],index=False)
    try:temp.replace(out/'results.xlsx')
    except PermissionError:raise RuntimeError('Chiudere results.xlsx e usare export; nessun training va ripetuto') from None
    save_json(out/'selection.json',selection)
    text='# Teste frozen su V2 Dense — risultati del run\n\n'
    text+='Conv1D opera sui canali medi, non sulla geometria dei pixel. Selezione su RMSE macro-AOI di sviluppo. '
    text+='La validation OOF è usata per early stopping/scelta, quindi non è una stima finale non distorta. '
    text+='Liguria è un benchmark già osservato, non un test cieco.\n\n'
    text+=metrics[metrics.aggregation.isin(['pooled','macro_AOI'])].to_csv(index=False)
    text+='\nMAPE relativo e MAE×100 (punti della scala) non sono intercambiabili. Migliore riproduzione di GLOBIO non significa validazione di biodiversità sul campo.\n'
    (out/'report.md').write_text(text,encoding='utf-8')
    return metrics

# Storico importato solo quando celle e target sono presenti: altrimenti nessun delta numerico.
def compare_external(current,path,out):
    old=pd.read_csv(path,dtype={'cell_id':str});keys=['cell_id','model_name','partition']
    required=set(keys+['target','prediction'])
    if not required<=set(old):raise ValueError('Storico: servono cell_id/model_name/partition/target/prediction; metriche sole non appaiabili')
    scope=current[current.model_name.isin(old.model_name)&current.partition.isin(old.partition)]
    join=scope.merge(old,on=keys,validate='one_to_one',suffixes=('_current','_historical'))
    if len(join)!=len(scope) or len(join)!=len(old) or not np.allclose(join.target_current,join.target_historical,rtol=0,atol=1e-8):
        raise ValueError('Confronto storico non appaiato; non viene pubblicato un delta')
    join.to_csv(Path(out)/'paired_historical_predictions.csv',index=False)
    rows=[]
    for (model,part),g in join.groupby(['model_name','partition']):
        a=full_metrics(g.target_current,g.prediction_historical);b=full_metrics(g.target_current,g.prediction_current)
        for metric in ['mae','rmse','mape_pct','bias','r2','pearson_r']:
            rows.append({'model':model,'partition':part,'metric':metric,'historical':a[metric],'current':b[metric],
                'delta':b[metric]-a[metric] if a[metric] is not None and b[metric] is not None else None,'n_cells':len(g)})
    pd.DataFrame(rows).to_csv(Path(out)/'paired_historical_metrics.csv',index=False)
