# Scopo: unica CLI del benchmark frozen, senza acquisizioni Sentinel o caricamento dell'encoder.
# Fasi: audit/dry-run -> prepare -> cache -> smoke/screen -> nested approvato -> report/monitor.
# Input: config e percorsi espliciti; output: root nuove. Esempio sampling_benchmark.py doctor --help.
# Parametri: --resume, --approved-plan, budget; monitor legge/chiude subito progress.json.
# Risorse: CPU per audit, GPU solo testa se richiesta. Smoke supervisionato entro900s, screening riprendibile.
from copy import deepcopy
import argparse
import json
import multiprocessing
from pathlib import Path
import time
import traceback
from filelock import FileLock
from .state import load_config,save_json,Progress,BudgetExpired
from . import data,reporting


def monitor(path,interval=5,once=False):
    if interval<=0:raise ValueError('Intervallo positivo')
    while True:
        try:print(Path(path).read_text(encoding='utf-8'),flush=True)
        except (FileNotFoundError,PermissionError) as error:print(str(error),flush=True)
        if once:return
        time.sleep(interval)


def smoke_worker(cfg,resume,queue):
    source=None
    try:
        from .training import fit,predict
        from .metrics import metrics
        from .state import signature,atomic_frame
        source=data.open_source(cfg);frame=data.development(source)
        frame=frame[frame.aoi_id==cfg['execution']['smoke_aoi']].head(cfg['execution']['smoke_cells']).copy()
        if len(frame)!=cfg['execution']['smoke_cells']:raise ValueError('Celle smoke non disponibili')
        plan=data.prepare(source,cfg,resume,frame=frame);data.cache(source,cfg,resume)
        sampled=data.CachedSource(cfg['paths']['cache_root'],'S64');out=Path(cfg['paths']['output_root'])
        tag=signature({'plan':plan['identity'],'smoke':True});data.own(out,tag,'technical_smoke');progress=Progress(out/'progress.json')
        try:
            save_json(out/'effective_config.json',cfg)
            model,norm,_=fit(sampled,source,sampled.frame,None,cfg,out/'fit',tag,42,resume,epochs=1,progress=progress)
            predictions=predict(model,source,frame,norm,cfg);atomic_frame(out/'technical_predictions.parquet',predictions)
            result={'status':'TECHNICAL_ONLY','scientific_result':False,'cells':len(frame),'metrics':metrics(predictions)}
            save_json(out/'smoke_result.json',result);progress.update(force=True,phase='complete',eta_seconds=0)
            queue.put({'ok':True,'result':result})
        finally:sampled.close()
    except BaseException as error:traceback.print_exc();queue.put({'ok':False,'error':repr(error)})
    finally:
        if source is not None:source.close()


def smoke(cfg,resume=False):
    cfg=deepcopy(cfg);cfg['training']['num_workers']=0;cfg['training']['epochs']=1
    ctx=multiprocessing.get_context('spawn');queue=ctx.Queue();worker=ctx.Process(target=smoke_worker,args=(cfg,resume,queue))
    worker.start();start=time.monotonic();limit=cfg['execution']['smoke_seconds']
    while worker.is_alive() and time.monotonic()-start<limit:worker.join(timeout=1)
    if worker.is_alive():
        worker.terminate();worker.join(timeout=10)
        if worker.is_alive():worker.kill();worker.join()
        raise BudgetExpired('Smoke non concluso entro900s; NON passato. Dati validi conservati, --resume')
    try:result=queue.get(timeout=2)
    except Exception as error:raise RuntimeError('Worker smoke senza esito, exit='+str(worker.exitcode)) from error
    if not result['ok']:raise RuntimeError(result['error'])
    return result['result']


def parser():
    p=argparse.ArgumentParser(description='Benchmark frozen sampling; nessun download o fine-tuning encoder')
    p.add_argument('action',choices=['doctor','prepare','cache','smoke','screen','run','report','monitor'])
    p.add_argument('--config',default='configs/spatial_sampling_benchmark.yaml')
    for name in ['pixel-root','reference','index-root','cache-root','output-root','historical-run']:p.add_argument('--'+name)
    p.add_argument('--device');p.add_argument('--set',action='append',default=[]);p.add_argument('--dry-run',action='store_true');p.add_argument('--resume',action='store_true')
    p.add_argument('--phase',choices=['controls','candidates','confirm','outer','recommend','all'],default='all')
    p.add_argument('--outer-fold',type=int,choices=[0,1,2]);p.add_argument('--approved-plan',help='budget_id esaminato e approvato separatamente dal proprietario della campagna')
    p.add_argument('--progress');p.add_argument('--interval',type=float,default=5);p.add_argument('--once',action='store_true')
    return p


def main(argv=None):
    args=parser().parse_args(argv)
    if args.action=='monitor':
        if not args.progress:raise ValueError('--progress richiesto')
        return monitor(args.progress,args.interval,args.once)
    overrides=list(args.set)
    for name in ['pixel_root','reference','index_root','cache_root','output_root','historical_run']:
        if getattr(args,name) is not None:overrides.append('paths.'+name+'='+json.dumps(getattr(args,name)))
    if args.device:overrides.append('execution.device='+json.dumps(args.device))
    cfg=load_config(args.config,overrides)
    if args.dry_run:
        result=reporting.forecast(cfg);print(json.dumps(result,indent=2,default=str));return result
    if args.action=='doctor':
        import torch,platform
        result={'status':'NOT_VERIFIED_ON_PC','budget':reporting.forecast(cfg),'cuda_available':torch.cuda.is_available(),
            'torch':torch.__version__,'python':platform.python_version(),'platform':platform.platform(),'encoder_loaded':False,'historical':reporting.historical(cfg)}
        if torch.cuda.is_available():result['gpu']={'name':torch.cuda.get_device_name(),'vram_bytes':torch.cuda.get_device_properties(0).total_memory,'torch_cuda':torch.version.cuda}
        if all(cfg['paths'].get(k) for k in ['pixel_root','reference','index_root']):
            data.roots(cfg);source=data.open_source(cfg)
            try:result.update(source_fingerprint=source.fingerprint,index_identity=source.index_identity,development_cells=len(data.development(source)),development_aois=data.development(source).aoi_id.nunique(),historical=reporting.historical(cfg,data.development(source).cell_id))
            finally:source.close()
        print(json.dumps(result,indent=2,default=str));return result
    cache_root,out=data.roots(cfg);cache_root.parent.mkdir(parents=True,exist_ok=True);out.parent.mkdir(parents=True,exist_ok=True)
    with FileLock(str(cache_root)+'.sampling.lock',timeout=0),FileLock(str(out)+'.sampling.lock',timeout=0):
        if args.action=='smoke':result=smoke(cfg,args.resume)
        elif args.action=='report':result=reporting.report(cfg)
        else:
            source=data.open_source(cfg)
            try:
                if args.action=='prepare':result=data.prepare(source,cfg,args.resume)
                elif args.action=='cache':result=data.cache(source,cfg,args.resume)
                else:
                    from .study import Study
                    budget=reporting.forecast(cfg)
                    if args.action=='run' and args.approved_plan!=budget['budget_id']:
                        raise ValueError('Campagna NON autorizzata: esaminare --dry-run e passare --approved-plan budget_id solo dopo approvazione separata')
                    study=Study(source,cfg,args.resume);save_json(out/'effective_config.json',cfg);save_json(out/'budget.json',budget)
                    deadline=time.monotonic()+cfg['execution']['screen_budget_hours']*3600 if args.action=='screen' else None
                    try:
                        if args.action=='screen':study.screen(deadline);result={'status':'SCREEN_COMPLETE','outer_fold_used':False}
                        else:
                            folds=[args.outer_fold] if args.outer_fold is not None else [0,1,2]
                            for fold in folds:
                                if args.phase=='controls':study.controls(fold)
                                elif args.phase=='candidates':study.candidates(fold)
                                elif args.phase in ['confirm','all']:study.confirm(fold)
                                elif args.phase=='outer':study.external(fold)
                            if args.phase in ['recommend','all']:study.recommend()
                            if args.phase=='all':
                                for fold in folds:study.external(fold)
                            result={'status':'PHASE_COMPLETE','phase':args.phase}
                        study_progress=Progress(out/'progress.json');study_progress.update(force=True,phase='complete',eta_seconds=0)
                    except BudgetExpired as error:
                        Progress(out/'progress.json').update(force=True,phase='interrupted',reason=str(error),eta_seconds=None)
                        result={'status':'BUDGET_EXHAUSTED_NOT_COMPLETE','resume_required':True}
                    finally:study.close()
            finally:source.close()
    print(json.dumps(result,indent=2,default=str));return result
