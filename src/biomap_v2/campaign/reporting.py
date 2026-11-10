# Scopo: metriche, grafici, Excel ricostruibile e report descrittivo dai CSV effettivi.
# Nessuna nuova regressione viene eseguita durante l'esportazione.
import os
import json
from pathlib import Path
import numpy as np
import pandas as pd
from biomap_v2.dense.evaluation import full_metrics
from biomap_v2.common import save_json

# Restituisce il picco RAM dove disponibile; null esplicito invece di una misura inventata.
def peak_ram():
    try:
        import resource
        return {'value':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'unit':'bytes' if os.sys.platform=='darwin' else 'KiB'}
    except ImportError:
        try:
            import ctypes
            from ctypes import wintypes
            class Counters(ctypes.Structure):
                _fields_=[('cb',wintypes.DWORD),('PageFaultCount',wintypes.DWORD)]+[(k,ctypes.c_size_t) for k in ['PeakWorkingSetSize','WorkingSetSize','QuotaPeakPagedPoolUsage','QuotaPagedPoolUsage','QuotaPeakNonPagedPoolUsage','QuotaNonPagedPoolUsage','PagefileUsage','PeakPagefileUsage']]
            c=Counters();c.cb=ctypes.sizeof(c)
            kernel=ctypes.WinDLL('kernel32');kernel.GetCurrentProcess.restype=wintypes.HANDLE
            psapi=ctypes.WinDLL('psapi');psapi.GetProcessMemoryInfo.argtypes=[wintypes.HANDLE,ctypes.POINTER(Counters),wintypes.DWORD]
            if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(),ctypes.byref(c),c.cb):raise OSError('GetProcessMemoryInfo failed')
            return {'value':int(c.PeakWorkingSetSize),'unit':'bytes'}
        except (AttributeError,OSError):return {'value':None,'unit':None,'reason':'OS memory API unavailable'}

# L'Excel viene scritto atomicamente; un lock di Excel non genera copie principali alternative.
def export(work):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    work=Path(work);files=list((work/'probing').glob('*/*predictions.csv'))
    if not files:raise ValueError('Nessuna predizione disponibile')
    frames=[pd.read_csv(p,dtype={'cell_id':str}) for p in files];allrows=[];detail=[]
    for f in frames:
        dummy=f[f.model=='DummyMean'][['cell_id','prediction']].rename(columns={'prediction':'dummy'})
        f=f.merge(dummy,on='cell_id',validate='many_to_one')
        f['error']=f.prediction-f.target;f['absolute_error']=f.error.abs()
        f['absolute_percentage_error']=np.where(f.target!=0,100*f.absolute_error/f.target.abs(),np.nan)
        detail.append(f)
        for (encoder,model,part),g in f.groupby(['encoder','model','partition']):
            m=full_metrics(g.target,g.prediction);d=full_metrics(g.target,g.dummy)
            allrows.append({'encoder':encoder,'model':model,'partition':part,**m,
                'rmse_improvement_vs_dummy_pct':100*(d['rmse']-m['rmse'])/d['rmse'] if d['rmse'] else None,
                'mae_scale_percentage_points':100*m['mae']})
            fig,axes=plt.subplots(1,2,figsize=(9,4));axes[0].scatter(g.target,g.prediction,s=1,alpha=.15)
            axes[0].plot([0,1],[0,1],'k--');axes[0].set(xlabel='MSA GLOBIO',ylabel='MSA predetto')
            axes[1].hist(g.error,bins=60);axes[1].set(xlabel='Predetto − osservato',ylabel='Celle')
            folder=work/'plots';folder.mkdir(exist_ok=True);fig.tight_layout();fig.savefig(folder/f'{encoder}_{model}_{part}.png');plt.close(fig)
    summary=pd.DataFrame(allrows);summary.to_csv(work/'metrics.csv',index=False)
    pd.concat(detail).to_csv(work/'cell_predictions.csv',index=False)
    temp=work/'results.partial.xlsx';dest=work/'results.xlsx'
    with pd.ExcelWriter(temp,engine='openpyxl') as writer:
        pd.DataFrame({'Definizione':['CV = 17 AOI, tre fold originali; Liguria separata.','MAPE esclude target zero; MAE ×100 = punti percentuali della scala.','CSV/JSON sono la fonte di verità; nessuna validazione della biodiversità sul campo.']}).to_excel(writer,sheet_name='Summary',index=False)
        summary.to_excel(writer,sheet_name='Risultati',index=False)
        for f in detail:
            for (encoder,model,part),g in f.groupby(['encoder','model','partition']):
                for i,low in enumerate(range(0,len(g),1000000)):
                    name=f'{encoder}_{model}_{"cv" if part=="validation" else "lig"}_{i}'
                    g.iloc[low:low+1000000].to_excel(writer,sheet_name=name[:31],index=False)
    try:temp.replace(dest)
    except PermissionError:raise RuntimeError('Chiudere results.xlsx in Excel e rilanciare export; nessun retraining necessario') from None
    text='# Probing denso multi-AOI — risultati effettivi\n\n'
    text+='Validation geografica sulle 17 AOI e benchmark ligure sono popolazioni distinte. '
    text+='Le metriche sottostanti derivano dai CSV; la scelta è in selection.json e usa RMSE macro-AOI.\n\n'
    text+=summary.to_csv(index=False)+'\n\nLimiti: target GLOBIO modellato; Liguria già osservata; precisione V1 int8/V2 FP32 e preprocessing upstream non necessariamente identici.\n'
    (work/'report.md').write_text(text,encoding='utf-8')
    save_json(work/'export_complete.json',{'n_summary_rows':len(summary),'n_detail_rows':sum(len(f) for f in detail)})
