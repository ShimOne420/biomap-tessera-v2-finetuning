# Scopo: costo preregistrato per volume, audit storico e report senza inventare risultati scientifici.
# Fasi: piano/config -> fit min/max e scenari tempo -> metriche disponibili -> Markdown/CSV/JSON.
# Input: manifest e risultati locali di nuove root; output: report e budget nel nuovo run, mai sorgenti.
# Parametri: 2956 byte/raw per pixel dichiarati dal PC, 2000 bootstrap; uso forecast(cfg), report(cfg).
# Risorse: CPU/metadati; ripresa: report ricostruibile, missing evidence marcata NOT_VERIFIED.
import json
from pathlib import Path
import shutil
import numpy as np
import pandas as pd
from .state import *
from .data import Plan
from .metrics import metrics,tables,compare


def available(path):
    if not path:return None
    p=Path(path).resolve()
    while not p.exists():
        if p.parent==p:raise FileNotFoundError('Volume non disponibile: '+str(p))
        p=p.parent
    usage=shutil.disk_usage(p)
    return {'existing_parent':str(p),'free_bytes':usage.free,'total_bytes':usage.total}


def forecast(cfg):
    path=cfg['paths'].get('cache_root');plan=Plan(path) if path and (Path(path)/'plan.json').exists() else None
    science={'sampling':cfg['sampling'],'training':{k:v for k,v in cfg['training'].items() if k!='num_workers'},'decision':cfg['decision']}
    from .data import code_identity
    budget_id=signature({'protocol':PROTOCOL,'config':science,'plan_identity':plan.manifest['identity'] if plan else None,'code':code_identity()})
    rows=[]
    for mode,nominal in [('S64',64),('M128',128),('S144',144),('M64',64),('FULL-SUBSET',None),('FULL',None)]:
        measured=plan.manifest['summary'].get(mode) if plan and mode!='FULL' else None
        pixels=measured['pixels'] if measured else (95981208 if mode=='FULL' else (min(cfg['sampling']['subset_cells'],127909)*713 if nominal is None else 127909*nominal))
        rows.append({'mode':mode,'pixels':pixels,'count_basis':'manifest' if measured else 'PC reference/nominal estimate',
            'cells':measured['cells'] if measured else (min(cfg['sampling']['subset_cells'],127909) if mode=='FULL-SUBSET' else 127909),
            'raw_bytes_estimate':pixels*2956,'embedding_bytes_upper':pixels*512})
    timings=[];train_times=[];validation_times=[];external_times=[];output=cfg['paths'].get('output_root')
    if output:
        for history in Path(output).rglob('history.json'):
            history=json.loads(history.read_text(encoding='utf-8'))
            timings.extend(x['epoch_seconds'] for x in history)
            train_times.extend(x['training_seconds'] for x in history if 'training_seconds' in x)
            validation_times.extend(x['validation_seconds'] for x in history if 'validation_seconds' in x)
        for marker in Path(output).rglob('external_complete.json'):
            marker=json.loads(marker.read_text(encoding='utf-8'))
            if 'dense_evaluation_seconds' in marker:external_times.append(marker['dense_evaluation_seconds'])
    cache_marker=Path(path)/'cache.json' if path else None
    cache_info=json.loads(cache_marker.read_text(encoding='utf-8')) if cache_marker and cache_marker.exists() else {}
    cache_bytes=sum(p.stat().st_size for p in Path(path).rglob('*') if p.is_file()) if path and Path(path).exists() else None
    # Seven fits/inner split initially: FULL x2, four primary candidates and one diagnostic subset.
    initial=3*2*7;inner_min=initial+3*2*3;inner_max=3*2*(3+4*3+1)
    refit_min=3*3;refit_max=3*2*3
    epoch=float(np.median(timings)) if timings else 555.
    return {'protocol':PROTOCOL,'budget_id':budget_id,'plan_present':plan is not None,'candidates':rows,
        'jobs':{'screen_fits':7,'initial_inner_fits':initial,'total_inner_fits_min_no_candidate':initial+6,'total_inner_fits_min_with_one_confirmed_candidate':inner_min,
            'total_inner_fits_max':inner_max,'outer_refits_min_full_only':refit_min,'outer_refits_max_full_and_selected':refit_max},
        'time':{'epoch_reference_seconds':epoch,'basis':'measured new fit history' if timings else 'PC reported historical 520-590s; NOT a benchmark of new caching',
            'screen_upper_seconds_at_max_epochs':7*cfg['training']['epochs']*epoch,
            'nested_upper_seconds_at_max_epochs':(inner_max+refit_max)*cfg['training']['epochs']*epoch,
            'training_epoch_median_seconds':float(np.median(train_times)) if train_times else None,
            'validation_epoch_median_seconds':float(np.median(validation_times)) if validation_times else None,
            'cache_seconds':cache_info.get('seconds_finalizing_invocation'),
            'cache_time_note':'last completing invocation only; earlier interrupted invocations excluded',
            'dense_outer_seconds':float(np.median(external_times)) if external_times else None,
            'timing_note':'mixed modes/folds/early-stop lengths: measured medians are pilot descriptors, not per-candidate guarantees',
            'missing_costs':'unmeasured cache/validation preprocessing, hash audits and final dense passes require PC pilot measurements'},
        'storage':{'cache_volume':available(cfg['paths'].get('cache_root')),'output_volume':available(output),'reserve_gib_per_volume':cfg['execution']['reserve_gib'],
            'candidate_embeddings_sum_upper_bytes':sum(r['embedding_bytes_upper'] for r in rows if r['mode']!='FULL'),
            'cache_current_bytes_measured':cache_bytes,
            'validation_scratch_limit_bytes':cfg['cache']['validation_max_gib']*1024**3,
            'instruction':'Do not add disk capacities. Budget includes manifests, staging, scratch and checkpoint archive. Compression not measured is NOT admission.'},
        'scientific_campaign_authorized':False,'sentinel_acquisition_authorized':False,'note':'12h is an interruptible screening budget, not a completion guarantee'}


def historical(cfg,expected_ids=None):
    root=cfg['paths'].get('historical_run')
    if not root:return {'status':'NOT_VERIFIED','reason':'historical-run not configured','reported_overall_macro':.17602}
    root=Path(root);p=root/'predictions.parquet'
    needed=[p,root/'selection.json',root/'cv_complete.json']
    if not all(x.is_file() for x in needed):return {'status':'NOT_VERIFIED','reason':'canonical artifacts missing','reported_overall_macro':.17602}
    selection=json.loads(needed[1].read_text(encoding='utf-8'));complete=json.loads(needed[2].read_text(encoding='utf-8'))
    if digest(p)!=selection['oof_sha256'] or complete['oof_sha256']!=digest(p) or complete['selection_sha256']!=digest(needed[1]):raise ValueError('Artefatti storici incoerenti')
    frame=pd.read_parquet(p);frame=frame[frame.model=='predictor'].copy()
    if expected_ids is not None and set(frame.cell_id)!=set(expected_ids):raise ValueError('Popolazione storica diversa')
    result=metrics(frame)
    if not np.isclose(result['macro_aoi_rmse'],selection['score'],rtol=0,atol=1e-10):raise ValueError('Score storico non ricostruibile')
    return {'status':'CANONICAL_PREDICTIONS_VERIFIED','sha256':digest(p),'metrics':result,
        'reported_overall_macro':.17602,'rounded_value_matches':f"{result['macro_aoi_rmse']:.5f}"=='0.17602',
        'per_fold':{str(k):metrics(g)['macro_aoi_rmse'] for k,g in frame.groupby('cv_fold')},
        'interpretation':'historical epoch selection used held-out fold; NOT equivalent to new nested FULL'}


def report(cfg):
    root=Path(cfg['paths']['output_root'])
    if not (root/'ownership.json').is_file():raise ValueError('Creare prima un nuovo workspace benchmark con screen/run')
    budget=forecast(cfg);save_json(root/'budget.json',budget)
    lines=['# Benchmark sampling V2 — report','',
        'Solo CNN frozen. Nessuna prova di non inferiorita; nessuna approvazione automatica di acquisizione o fine-tuning.',
        'Le metriche interne sono di selezione. Gli outer valutano la procedura con decisioni interne sigillate.',
        'Le maschere temporali Sentinel non sono contenute negli embedding e restano da verificare in una futura fase autorizzata.','']
    records=[];outer={};resource_rows=[]
    for path in sorted(root.rglob('resources.json')):
        resource_rows.append({'fit':path.parent.relative_to(root).as_posix(),**json.loads(path.read_text(encoding='utf-8'))})
    if resource_rows:
        pd.json_normalize(resource_rows,sep='_').to_csv(root/'resources_summary.csv',index=False)
        lines.extend(['Risorse per fit: `resources_summary.csv`; dettagli e tempi separati training/validation nelle history.',
            'Il picco RAM e del processo per tutta la sua vita, senza memoria dei worker; non e il consumo incrementale del singolo fit.',''])
    for path in sorted(root.rglob('result.json')):
        value=json.loads(path.read_text(encoding='utf-8'));pred=path.parent/'predictions.parquet'
        if digest(pred)!=value['sha256']:raise ValueError('Predizioni interne modificate')
        frame=pd.read_parquet(pred);row={'fit':path.parent.relative_to(root).as_posix(),**metrics(frame)};records.append(row)
        a,b=tables(frame);a.to_csv(path.parent/'by_aoi.csv',index=False);b.to_csv(path.parent/'by_msa.csv',index=False)
    for path in sorted(root.rglob('external_complete.json')):
        value=json.loads(path.read_text(encoding='utf-8'));p=path.parent/'external.parquet'
        if digest(p)!=value['sha256']:raise ValueError('Risultato esterno modificato')
        frame=pd.read_parquet(p);a,b=tables(frame);a.to_csv(path.parent/'by_aoi.csv',index=False);b.to_csv(path.parent/'by_msa.csv',index=False)
        records.append({'fit':path.parent.relative_to(root).as_posix(),**metrics(frame)})
        mode=path.parent.parent.name;seed=path.parent.name;kind='FULL' if mode=='FULL' else 'nested_selection_procedure'
        outer.setdefault((kind,seed),[]).append(frame)
    completed={}
    for (kind,seed),parts in outer.items():
        frame=pd.concat(parts,ignore_index=True)
        if set(frame.cv_fold)!={0,1,2}:continue
        expected=set(Plan(cfg['paths']['cache_root']).frame.cell_id)
        if set(frame.cell_id)!=expected or frame.cell_id.duplicated().any():raise ValueError('OOF esterno incompleto/duplicato')
        atomic_frame(root/(kind+'_'+seed+'_oof.parquet'),frame);save_json(root/(kind+'_'+seed+'_metrics.json'),metrics(frame))
        completed[(kind,seed)]=frame
    for seed in ['seed_42','seed_43','seed_44']:
        if ('FULL',seed) in completed and ('nested_selection_procedure',seed) in completed:
            save_json(root/('paired_outer_'+seed+'.json'),compare(completed[('FULL',seed)],completed[('nested_selection_procedure',seed)],cfg['decision']['bootstrap_samples'],cfg['decision']['bootstrap_seed']))
    if records:
        pd.json_normalize(records,sep='_').to_csv(root/'metrics_summary.csv',index=False)
        lines+=['| Fit | Celle | RMSE macro | RMSE pooled | R² | Errori >0,4 |','|---|---:|---:|---:|---:|---:|']
        for r in records:
            r2='n/d' if r['r2'] is None else f"{r['r2']:.5f}"
            lines.append(f"| {r['fit']} | {r['n_cells']} | {r['macro_aoi_rmse']:.5f} | {r['rmse']:.5f} | {r2} | {r['error_gt_04_fraction']:.3%} |")
        lines.append('')
    lines.extend([f'Fit completati con metriche: {len(records)}.', ''])
    for path in [root/'global_margin.json',root/'recommendation.json']+sorted(root.glob('outer_*/decision.json')):
        if path.exists():lines.extend(['## '+path.relative_to(root).as_posix(),'```json',path.read_text(encoding='utf-8'),'```',''])
    if cfg['paths'].get('cache_root') and (Path(cfg['paths']['cache_root'])/'plan.json').exists():
        plan=Plan(cfg['paths']['cache_root']);geometry=[]
        for _,cells in plan.blocks():
            for cell in cells:
                for mode,item in cell['modes'].items():geometry.append({'cell_id':cell['cell_id'],'aoi_id':cell['aoi_id'],'mode':mode,**item['diagnostics']})
        pd.json_normalize(geometry,sep='_').to_csv(root/'sampling_diagnostics.csv',index=False)
        if (root/'recommendation.json').exists():
            recommendation=json.loads((root/'recommendation.json').read_text(encoding='utf-8'))
            proposal={'sampling_plan_identity':plan.manifest['identity'],'sampling_plan_sha256':digest(plan.root/'plan.json'),
                'mode':recommendation['selected'],'sampling_seed':cfg['sampling']['seed'],'source_fingerprint':plan.manifest['source'],
                'checkpoint_sha256':plan.manifest['reference_checkpoint'],'approved':False,'future_last_n_blocks_proposed':1,
                'temporal_sentinel_parity':'NOT_VERIFIED','instruction':'Use exact manifest patch boundaries, tile positions, spatial masks and weights. No Sentinel acquisition authorized.'}
            save_json(root/'acquisition_proposal.json',proposal)
    lines.extend(['## Costi e autorizzazioni','```json',json.dumps(budget,indent=2,ensure_ascii=False),'```',
        'La variabilita fra due seed e un indicatore esplorativo, non una stima completa dell’incertezza.',
        'FULL-SUBSET cambia popolazione di training: non partecipa alla selezione primaria.',
        'R2, code di errore e fattibilita per volume richiedono revisione umana prima di approvare qualsiasi supporto.'])
    (root/'report.md').write_text('\n'.join(lines),encoding='utf-8')
    return {'report':str(root/'report.md'),'completed_fits':len(records)}
