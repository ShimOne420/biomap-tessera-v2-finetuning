# Scopo: unica CLI operativa Spatial CNN V2, con azioni esplicite e override YAML/CLI verificati.
# Fasi: config -> doctor oppure preflight -> prepare/profile/CV/refit/evaluate/compare; monitor read-only.
# Input: YAML e root locali; output: registry privato e avanzamento; mai acquisizioni implicite.
# Parametri/esempio: spatial_cnn.py doctor --config configs/spatial_v2_msa.yaml --source-root ROOT --reference REF.
# Risorse: doctor/monitor CPU minima; le altre azioni usano dati precalcolati e soltanto il predittore.
# Ripresa: prepare per tile, train --resume da epoca completata. Liguria richiede evaluate separato.
import argparse
import json
from pathlib import Path
import time

from filelock import FileLock
import yaml

from .contracts import ModelConfig,ExperimentProtocol,ExperimentSpec
from .sources import make_source,prepare_index,inventory
from .utils import separate_output


def parser():
    p=argparse.ArgumentParser(description='Spatial CNN encoder-agnostic: V2 pixel autentici, niente encoder live')
    p.add_argument('action',choices=['doctor','prepare','preflight','profile','train','evaluate','compare','monitor'])
    p.add_argument('--config',default='configs/spatial_v2_msa.yaml')
    p.add_argument('--source-root');p.add_argument('--reference');p.add_argument('--output-root');p.add_argument('--device')
    p.add_argument('--run-id');p.add_argument('--phase',choices=['cv','refit'],default='cv')
    p.add_argument('--resume',action='store_true')
    p.add_argument('--max-epochs-this-invocation',type=int)
    p.add_argument('--architecture',choices=['spatial_cnn','mean_mlp'])
    p.add_argument('--kernel',type=int,choices=[1,3]);p.add_argument('--projection-channels',type=int)
    p.add_argument('--seed',type=int);p.add_argument('--batch-cells',type=int);p.add_argument('--epochs',type=int)
    p.add_argument('--normalization',choices=['pixel_weighted_standard','cell_standard','none'])
    p.add_argument('--set',action='append',default=[],metavar='section.key=VALUE')
    p.add_argument('--compare-runs',nargs='+');p.add_argument('--comparison-name',default='ablation')
    p.add_argument('--watch',action='store_true');p.add_argument('--interval',type=float,default=5.)
    return p


def configuration(args):
    cfg=yaml.safe_load(Path(args.config).read_text(encoding='utf-8'))
    for item in args.set:
        key,sep,value=item.partition('=')
        if not sep:raise ValueError('--set richiede sezione.chiave=valore')
        keys=key.split('.');target=cfg
        for part in keys[:-1]:
            if part not in target or not isinstance(target[part],dict):raise ValueError('Sezione sconosciuta: '+key)
            target=target[part]
        if keys[-1] not in target:raise ValueError('Parametro sconosciuto: '+key)
        target[keys[-1]]=yaml.safe_load(value)
    direct={'source_root':('source','root'),'reference':('source','reference'),'output_root':('execution','output_root'),
        'device':('execution','device'),'architecture':('model','architecture'),'kernel':('model','kernel'),
        'projection_channels':('model','projection_channels'),'seed':('protocol','seed'),'batch_cells':('training','batch_cells'),
        'epochs':('training','epochs'),'normalization':('protocol','normalization')}
    for name,(section,key) in direct.items():
        if getattr(args,name) is not None:cfg[section][key]=getattr(args,name)
    spec=ExperimentSpec(cfg['source'],ModelConfig(**cfg['model']),ExperimentProtocol(**cfg['protocol']),cfg['training'],cfg['execution'])
    t=spec.training
    if min(t['epochs'],t['batch_cells'],t['patience'])<1 or t['learning_rate']<=0 or t['weight_decay']<0 or t['clip_grad']<=0:
        raise ValueError('Parametri training fuori intervallo')
    if t.get('num_workers',0) not in range(5) or t.get('accumulate_cells',t['batch_cells'])<t['batch_cells']:
        raise ValueError('num_workers 0..4; accumulate_cells >= batch_cells')
    if spec.source.get('mmap_limit',4)<1 or spec.source.get('max_patch_side',64)<1 or spec.execution.get('reserve_gib',20)<0:
        raise ValueError('Risorse/patch non valide')
    if t.get('mixed_precision') and not spec.execution['device'].startswith('cuda'):
        raise ValueError('AMP e esplicita e inizialmente supportata soltanto su CUDA')
    separate_output(spec.execution['output_root'],[spec.source.get('root'),spec.source.get('reference')])
    return spec


def monitor(spec,run_id,watch,interval):
    from .engine import run_path
    if interval<1:raise ValueError('Intervallo monitor almeno un secondo')
    path=(run_path(spec,run_id)/'progress.json') if run_id else Path(spec.execution['output_root'])/'prepare_progress.json'
    while True:
        try:
            state=json.loads(path.read_text(encoding='utf-8'))
            print(json.dumps(state,indent=2),flush=True)
        except FileNotFoundError:
            state={};print('Avanzamento non ancora presente',flush=True)
        if not watch or state.get('status') in {'complete','cv_complete','refit_complete','failed','interrupted','prepare_complete'}:return 0
        time.sleep(interval)


def main(argv=None):
    args=parser().parse_args(argv)
    spec=configuration(args)
    if args.max_epochs_this_invocation is not None and args.max_epochs_this_invocation<1:
        raise ValueError('Limite tecnico epoche positivo')
    if args.action=='doctor':
        report=inventory(spec);print(json.dumps(report,indent=2))
        return 2 if report['status']=='BLOCKED' else 0
    if args.action=='monitor':return monitor(spec,args.run_id,args.watch,args.interval)
    if args.action in {'train','evaluate'} and not args.run_id:raise ValueError('--run-id obbligatorio')
    if args.action=='compare' and not args.compare_runs:raise ValueError('--compare-runs obbligatorio')
    from .engine import preflight,Progress,train,evaluate,profile,compare_runs
    out=Path(spec.execution['output_root']);out.mkdir(parents=True,exist_ok=True)
    with FileLock(str(out/'.framework.lock'),timeout=0):
        source=None
        try:
            if args.action=='prepare':
                source=make_source(spec,out/'index',require_index=False)
                progress=Progress(out/'prepare_progress.json')
                progress.update(force=True,status='running',phase='prepare')
                frame=source.frame if spec.source.get('kind')=='v1_index' else prepare_index(source,progress.update)
                progress.update(force=True,status='prepare_complete',cells_completed=len(frame),cells_total=len(frame))
                print(json.dumps({'prepared_cells':len(frame),'index_identity':source.index_identity}))
                return 0
            source,report=preflight(spec)
            if args.action=='preflight':print(json.dumps(report,indent=2))
            elif args.action=='profile':print(json.dumps(profile(spec,source),indent=2))
            elif args.action=='train':print(train(spec,source,args.run_id,args.phase,args.resume,args.max_epochs_this_invocation))
            elif args.action=='evaluate':print(json.dumps(evaluate(spec,source,args.run_id),indent=2))
            elif args.action=='compare':print(json.dumps(compare_runs(spec,source,args.compare_runs,args.comparison_name),indent=2))
            return 0
        finally:
            if source is not None:source.close()
