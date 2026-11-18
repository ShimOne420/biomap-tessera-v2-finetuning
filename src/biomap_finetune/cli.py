# Scopo: unico comando isolato per doctor, piano, acquisizione, smoke, train, evaluate, report e monitor.
# Fasi: config+override -> gate -> azione esplicita -> JSON/log; nessun download in doctor o dry-run.
# Input: percorsi CLI/YAML; output: soltanto root nuove. Esempio: finetune_v2.py doctor --config ... .
# Parametri: aggregation obbligatoria per train/smoke, --set sezione.chiave=valore, --resume e --phase.
# Risorse: doctor metadati o piccolo probe esplicito; smoke ha limite massimo 900s con processo spawn.
# Ripresa: stesso comando --resume; monitor chiude subito il file, non importa modelli o apre cache.
import argparse
from datetime import datetime,timezone
import json
import multiprocessing
import os
from pathlib import Path
import platform
import shutil
import sys
import time
import traceback

from .config import load,blockers,require_ready
from .io import checked_roots,save_json,signature,digest


def doctor(cfg,probe=False,inputs=False):
    import torch
    from biomap_v2.common import WEIGHTS_SHA256
    checked_roots(cfg)
    result={'python':platform.python_version(),'platform':platform.platform(),'torch':torch.__version__,
        'cuda_available':torch.cuda.is_available(),'device_requested':cfg['execution']['device'],
        'status':'NOT_VERIFIED_FOR_REAL_TRAINING','blockers':blockers(cfg),'cache_mode_resolved':'online',
        'paths':{k:{'configured':bool(v),'exists':Path(v).exists() if v else False} for k,v in cfg['paths'].items() if k!='read_only_roots'}}
    if torch.cuda.is_available():
        result['gpu']={'name':torch.cuda.get_device_name(),'vram_bytes':torch.cuda.get_device_properties(0).total_memory,
            'torch_cuda':torch.version.cuda,'bf16':torch.cuda.is_bf16_supported()}
    try:
        import psutil
        result['ram']={'total_bytes':psutil.virtual_memory().total,'available_bytes':psutil.virtual_memory().available}
    except ImportError:result['ram']='psutil not installed; consult OS; no dependency installed automatically'
    checkpoint=cfg['encoder']['checkpoint']
    result['checkpoint_verified']=bool(checkpoint and Path(checkpoint).is_file() and digest(checkpoint)==WEIGHTS_SHA256)
    result['legacy_series_policy']='INCOMPATIBLE_READ_ONLY_NOT_USED: nearest S2, 10log10 S1 and capped DOY cannot reproduce dense historical inputs'
    for name in ['raw_root','output_root']:
        value=cfg['paths'][name]
        if value:
            p=Path(value).resolve()
            while not p.exists():p=p.parent
            result[name+'_free_bytes']=shutil.disk_usage(p).free
    if probe:
        from .validation import architecture_probe
        result['technical_probe']=architecture_probe(cfg)
    if inputs:
        from .validation import verify_inputs
        result['input_parity']=verify_inputs(cfg)
    result['readiness_note']='Probe sintetico/parita campionaria non misurano microbatch reale o training scientifico; nessuna READY automatica'
    return result


def smoke_worker(cfg,run_id,resume,queue):
    try:
        from .data import acquire
        from .validation import verify_inputs
        from .training import train
        acquire(cfg,resume);verify_inputs(cfg);result=train(cfg,run_id,'pilot',resume)
        queue.put({'ok':True,'result':result})
    except BaseException as error:
        traceback.print_exc();queue.put({'ok':False,'error':repr(error)})


def smoke(cfg,run_id,resume=False,limit=900):
    if cfg['experiment']['profile']!='smoke':raise ValueError('Smoke richiede profilo smoke')
    require_ready(cfg)
    if cfg['training']['num_workers']!=0:raise ValueError('Smoke supervisionato richiede num_workers=0 per non lasciare worker figli dopo timeout')
    if limit<1 or limit>900:raise ValueError('Smoke budget richiesto: 1..900 secondi')
    ctx=multiprocessing.get_context('spawn');queue=ctx.Queue();process=ctx.Process(target=smoke_worker,args=(cfg,run_id,resume,queue))
    started=time.perf_counter();process.start()
    # Poll <=1s; parent supervises the entire acquisition+validation+training, not just epochs.
    while process.is_alive() and time.perf_counter()-started<limit:process.join(timeout=1)
    if process.is_alive():
        process.terminate();process.join(timeout=10)
        if process.is_alive():process.kill();process.join()
        raw_root=Path(cfg['paths']['raw_root'])
        if (raw_root/'ownership.json').is_file():
            save_json(raw_root/'smoke_budget.json',{'status':'BUDGET_EXCEEDED','elapsed_seconds':time.perf_counter()-started,'limit_seconds':limit,'resume':'explicit --resume; valid blocks preserved'})
        raise TimeoutError('Smoke oltre budget; marker incompleti non validi. Riprendere con --resume; nessuna promessa di durata rete')
    try:result=queue.get(timeout=2)
    except Exception as error:raise RuntimeError('Worker smoke terminato senza esito; controllare log/exitcode '+str(process.exitcode)) from error
    if not result['ok']:raise RuntimeError(result['error'])
    result.update(seconds=time.perf_counter()-started,limit_seconds=limit)
    save_json(Path(cfg['paths']['output_root'])/run_id/'smoke_result.json',result)
    return result


def monitor(path,interval=5,once=False):
    if interval<=0:raise ValueError('Intervallo positivo richiesto')
    path=Path(path)
    while True:
        try:
            # read_text opens and closes before sleeping: no Windows handle retained.
            record=json.loads(path.read_text(encoding='utf-8'))
            print(json.dumps(record,ensure_ascii=False),flush=True)
        except (FileNotFoundError,json.JSONDecodeError,PermissionError) as error:print(str(error),flush=True)
        if once:return
        time.sleep(interval)


def report(cfg,run_id,compare_run=None):
    from .io import PROTOCOL,code_identity
    from .training import registry
    source,folder,identity=registry(cfg,run_id,True)
    if compare_run:
        if Path(compare_run).name!=compare_run or compare_run in {'.','..'}:raise ValueError('compare-run deve essere un nome semplice')
        from .comparison import compare
        compare(folder,Path(cfg['paths']['output_root'])/compare_run)
    lines=['# Report fine-tuning V2', '',f'Protocollo `{PROTOCOL}`, identità `{identity}`.',
        f"Profilo: **{cfg['experiment']['profile']}**, aggregazione `{cfg['aggregation']['method']}`, encoder `{cfg['training']['encoder_mode']}`.",
        f"Celle dataset: {len(source.frame)}. Supporto: `{source.spec['support']}`.",
        'I risultati tecnici smoke non costituiscono validazione MSA. Liguria non seleziona parametri.',
        'Confronto principale: nuovo controllo frozen sullo stesso dataset/head/seed. Le baseline storiche non sono automaticamente comparabili sul supporto pixel.', '']
    for path in sorted(folder.rglob('*.json')):
        if path.name in {'selection.json','resources.json','history.json','paired_comparison.json'} or path.name.endswith('_metrics.json'):
            lines.extend(['## '+str(path.relative_to(folder)), '```json',path.read_text(encoding='utf-8'),'```',''])
    (folder/'report.md').write_text('\n'.join(lines),encoding='utf-8')
    return {'report':str(folder/'report.md'),'identity':identity}


def parser():
    p=argparse.ArgumentParser(description='GeoTESSERA V2 Medium partial FT, namespace isolato; nessuna decisione scientifica implicita')
    sub=p.add_subparsers(dest='action',required=True)
    for name in ['doctor','acquire','train','evaluate','report','smoke','dense-eval']:
        q=sub.add_parser(name,help=name+' nel nuovo namespace finetune-v2-r1')
        q.add_argument('--config',default='configs/v2_partial_finetuning.yaml')
        q.add_argument('--set',action='append',default=[],metavar='SEZIONE.CHIAVE=VALORE')
        for flag in ['reference','raw-root','output-root','catalog-root','frozen-pixels','checkpoint','device','aggregation']:
            q.add_argument('--'+flag)
        q.add_argument('--dry-run',action='store_true',help='Piano soltanto: nessuna acquisizione/training/scrittura cache')
        q.add_argument('--resume',action='store_true')
        q.add_argument('--run-id',default='partial_v2')
        if name=='doctor':
            q.add_argument('--probe',action='store_true',help='Forward/backward sul checkpoint locale, input sintetici, niente Sentinel')
            q.add_argument('--verify-inputs',action='store_true',help='Confronta raw acquisiti con pixel frozen, copertura esplicita')
        if name=='train':q.add_argument('--phase',choices=['pilot','cv','refit'],default='cv')
        if name=='smoke':q.add_argument('--limit-seconds',type=int,default=900)
        if name=='dense-eval':q.add_argument('--dense-pixel-batch',type=int,help='Batch inferenza densa FP32 esplicito, non il batch celle del training')
        if name=='report':q.add_argument('--compare-run',help='Controllo nuovo frozen con stessa testa/seed/supporto, OOF e/o Liguria')
    q=sub.add_parser('monitor',help='Monitor JSON in sola lettura, anche da terminale separato')
    q.add_argument('--progress',required=True);q.add_argument('--interval',type=float,default=5);q.add_argument('--once',action='store_true')
    return p


def main(argv=None):
    args=parser().parse_args(argv)
    if args.action=='monitor':return monitor(args.progress,args.interval,args.once)
    overrides=list(args.set)
    mapping={'reference':'paths.reference','raw_root':'paths.raw_root','output_root':'paths.output_root','catalog_root':'paths.catalog_root',
        'frozen_pixels':'paths.frozen_pixels','checkpoint':'encoder.checkpoint','device':'execution.device','aggregation':'aggregation.method'}
    for name,key in mapping.items():
        value=getattr(args,name)
        if value is not None:overrides.append(key+'='+json.dumps(value))
    cfg=load(args.config,overrides);checked_roots(cfg)
    if cfg['execution']['device'].startswith('cuda'):os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    if cfg['experiment']['profile']=='synthetic':raise ValueError('Profilo synthetic riservato alle fixture pytest, non alla CLI scientifica')
    if args.dry_run:
        if args.action=='dense-eval':
            from .dense_evaluation import dense_plan
            plan=dense_plan(cfg)
        else:
            from .data import estimates
            plan=estimates(cfg)
        result={'doctor':doctor(cfg),'plan':plan,'effective_config':cfg}
    elif args.action=='doctor':result=doctor(cfg,args.probe,args.verify_inputs)
    elif args.action=='acquire':
        from .data import acquire
        result=acquire(cfg,args.resume)
    elif args.action=='smoke':result=smoke(cfg,args.run_id,args.resume,args.limit_seconds)
    elif args.action=='train':
        from .training import train
        result=train(cfg,args.run_id,args.phase,args.resume)
    elif args.action=='evaluate':
        from .training import evaluate
        result=evaluate(cfg,args.run_id)
    elif args.action=='dense-eval':
        from .dense_evaluation import dense_evaluate
        result=dense_evaluate(cfg,args.run_id,args.dense_pixel_batch,args.resume)
    else:result=report(cfg,args.run_id,args.compare_run)
    print(json.dumps(result,ensure_ascii=False,indent=2,default=str),flush=True)
    return result
