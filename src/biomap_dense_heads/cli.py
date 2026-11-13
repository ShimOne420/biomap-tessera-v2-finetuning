# Scopo: preflight/run/profile/smoke/monitor/export. Tutte le scritture sono nell'output nuovo.
# Non modifica la pipeline di estrazione; il training richiede un marker completo verificato.
import argparse
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from filelock import FileLock
from biomap_v2.common import ROOT,digest,save_json
from biomap_v2.campaign.state import MEANS
from .artifacts import load
from .progress import Progress,monitor
from .campaign import run,profile
from .evaluation import export,compare_external

# Interfaccia stretta; nessun riconoscimento implicito di file parziali in cartelle ricorsive.
def parser():
    p=argparse.ArgumentParser(description='V2 dense frozen heads: Conv1D canali + MLP + Ridge/HistGB')
    p.add_argument('command',choices=['preflight','run','profile','smoke','monitor','export','compare'])
    p.add_argument('--source-root');p.add_argument('--features');p.add_argument('--completion');p.add_argument('--targets')
    p.add_argument('--out',required=True);p.add_argument('--device',default='cpu');p.add_argument('--resume',action='store_true')
    p.add_argument('--config',default=str(ROOT/'experiments/v2_dense_cnn_msa/config.yaml'))
    p.add_argument('--interval',type=int,default=5);p.add_argument('--once',action='store_true');p.add_argument('--historical')
    p.add_argument('--profile-cells',type=int,default=4096);p.add_argument('--profile-epochs',type=int,default=3)
    return p

# Configurazione validata prima del training; nessuna ricerca implicita di iperparametri.
def config(path):
    cfg=yaml.safe_load(Path(path).read_text())
    for key in ['epochs','patience','batch_size','threads','pool_size','dense_width']:
        if not isinstance(cfg[key],int) or cfg[key]<1:raise ValueError(f'{key} deve essere intero positivo')
    if not set(cfg['models'])<={'Conv1D','MLP'} or not cfg['models']:raise ValueError('Modelli neurali non validi')
    for key in ['models','outputs','baselines']:
        if len(cfg[key])!=len(set(cfg[key])):raise ValueError('Configurazioni duplicate')
    if not set(cfg['outputs'])<={'identity','sigmoid'} or not cfg['outputs']:raise ValueError('Output non validi')
    if not set(cfg['baselines'])<={'DummyMean','Ridge','HistGB'} or 'DummyMean' not in cfg['baselines']:raise ValueError('Dummy richiesto; baseline non valida')
    if not 0<=cfg['dropout']<1 or cfg['learning_rate']<=0 or cfg['weight_decay']<0:raise ValueError('Iperparametri non validi')
    return cfg

# Fixture separata, etichettata smoke; mai presentata come risultato MSA del dataset reale.
def synthetic(out):
    source=Path(out)/'synthetic_source';source.mkdir(parents=True,exist_ok=True);rng=np.random.default_rng(42)
    n=96;x=rng.normal(size=(n,128));target=1/(1+np.exp(-x[:,0]*.4))
    f=pd.DataFrame(x,columns=MEANS);f['cell_id']=[f'smoke_{i}' for i in range(n)]
    f['aoi_id']=[f'area_{i//12}' for i in range(n)];f['cv_fold']=np.repeat([0,1,2,-1],[24,24,24,24])
    f['partition']=np.where(f.cv_fold==-1,'test','train');f['target']=target;f['lon']=10+np.arange(n)*.001;f['lat']=44.
    path=source/'v2.parquet';f.to_parquet(path,index=False)
    save_json(source/'extraction_complete.json',{'sha256':digest(path),'n_cells':n,'complete_campaign':False,'smoke':True})
    return source

# Main non viene mai lanciato da un import o da un monitor del processo di estrazione.
def main(argv=None):
    args=parser().parse_args(argv);out=Path(args.out).resolve()
    if args.command=='monitor':return monitor(out,args.interval,args.once)
    if args.command in ['export','compare']:
        if not out.exists():raise ValueError('Output inesistente')
        with FileLock(str(out/'head_training.lock'),timeout=0):
            predictions=pd.read_csv(out/'predictions.csv',dtype={'cell_id':str})
            if args.command=='compare':
                if not args.historical:raise ValueError('--historical richiesto; metriche aggregate sole non appaiabili')
                compare_external(predictions,args.historical,out)
            else:export(out,predictions,json.loads((out/'selection.json').read_text()))
        return 0
    if args.command=='profile' and not (1<=args.profile_cells<=8192 and 1<=args.profile_epochs<=5):raise ValueError('Profiling limitato a 1..8192 celle e 1..5 epoche')
    cfg=config(args.config);smoke=args.command=='smoke'
    if smoke:
        if out.exists() and any(out.iterdir()):raise ValueError('Usare output smoke nuovo e vuoto')
        source=synthetic(out);destination=out/'smoke_results';cfg.update(epochs=2,patience=2,batch_size=16,threads=1)
    else:
        if not args.source_root:raise ValueError('--source-root richiesto: cartella estrazione completata')
        source=Path(args.source_root);destination=out
    # CUBLAS richiede questa variabile per algoritmi deterministici CUDA.
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    frame,audit=load(source,args.device,destination,args.features,args.completion,args.targets,smoke)
    if args.command=='preflight':print(json.dumps(audit,indent=2));return 0
    progress=Progress(destination)
    with FileLock(str(destination/'head_training.lock'),timeout=0):
        try:
            if args.command=='profile':profile(frame,cfg,audit,destination,args.device,progress,args.profile_cells,args.profile_epochs)
            else:run(frame,cfg,audit,destination,args.device,progress,resume=args.resume)
        except BaseException as error:
            progress.update(state='interrupted' if isinstance(error,KeyboardInterrupt) else 'failed',error=str(error))
            raise
    return 0
