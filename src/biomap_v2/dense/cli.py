# SCOPO: unica CLI del probing denso V1/V2 su MSA; entrypoint scripts/dense_msa.py.
# FLUSSO: pack su Mac -> copia ZIP -> all --bundle su Windows -> CV e report del run.
# Conserva check-v1/check-v2/check-encoder; non ricampiona a 25 punti o aggiorna l'encoder.
# Ogni run usa un output nuovo; report e modelli storici non vengono sovrascritti.
import json
import math
import time
from pathlib import Path
import pandas as pd
from filelock import FileLock
from biomap_v2.common import save_json, digest, WEIGHTS_SHA256
from .cli_args import build_parser, paths, require, timestamp
from .bundle import strict_equivalence, load_bundle, verify_bundle, pack


# Permette feature/equivalenze preparate prima, ma rifiuta un output di training già usato.
def fresh_run(out):
    out = Path(out)
    for name in ['protocol.json', 'run_state.json', 'predictions.csv', 'metrics.csv']:
        if (out/name).exists():
            raise FileExistsError(f'{out} contiene già un run: usare un nuovo --out')
    out.mkdir(parents=True, exist_ok=True)
    return out


# Trunc riproduce l'artefatto storico; floor è una diagnostica distinta dei bordi.
def check_v1(args):
    from .grid import load_target
    from .v1_support import aggregate_v1, compare_v1
    inputs = paths(args)
    require(inputs, ['v1_tiles', 'target', 'v1_parquet'])
    out = fresh_run(inputs['out'])
    grid = load_target(inputs['target'])
    reference = pd.read_parquet(inputs['v1_parquet'])
    result = {}
    for op in ['trunc', 'floor']:
        frame = aggregate_v1(inputs['v1_tiles'], grid, edge_op=op, min_valid_fraction=args.min_valid_fraction,
                              progress=lambda i, n, tile: print(f'V1 {op} {i}/{n}: {tile}', flush=True))
        result[op] = compare_v1(frame, reference)
        if op == 'trunc':
            frame.to_parquet(out/'v1_recomputed.parquet')
    save_json(out/'v1_equivalence.json', result)
    strict_equivalence(result['trunc'], 1e-6, 'V1', require_total=True)
    return result


# Riaggregazione V2 per chunk con lo stesso denominatore di copertura del V1.
def check_v2(args):
    from .v2_support import aggregate_v2, compare_v2
    inputs = paths(args)
    require(inputs, ['v2_tiles', 'v1_parquet'])
    out = fresh_run(inputs['out'])
    v1 = pd.read_parquet(inputs['v1_parquet'])
    frame = aggregate_v2(inputs['v2_tiles'], n_total=v1.n_total,
                         min_valid_fraction=args.min_valid_fraction, include_std=True)
    frame.to_parquet(out/'v2_recomputed.parquet')
    result = {'n_recomputed': len(frame), 'reference_available': False}
    if inputs['v2_parquet']:
        result = compare_v2(frame, pd.read_parquet(inputs['v2_parquet']))
        strict_equivalence(result, 1e-5, 'V2')
    save_json(out/'v2_equivalence.json', result)
    return result


# Non decomprime nuovi NPY nelle tile sorgenti: estrae soltanto un piccolo campione.
def check_encoder(args):
    from .sample import create_sample, verify_sample
    inputs = paths(args)
    require(inputs, ['v2_tiles', 'checkpoint'])
    out = fresh_run(inputs['out'])
    create_sample(inputs['v2_tiles'], out/'encoder_sample.npz', args.pixels)
    result = verify_sample(out/'encoder_sample.npz', inputs['checkpoint'], args.device, args.pixel_batch, args.tolerance)
    save_json(out/'encoder_equivalence.json', result)
    return result


# Parametri congelati prima della valutazione; nessuna ricerca sulle celle OOF.
def parameters(args):
    from .evaluation import make_estimator
    hist = {'max_iter': 90, 'max_leaf_nodes': 15, 'learning_rate': .05,
            'l2_regularization': 1., 'early_stopping': True, 'random_state': args.seed}
    result = {'histgb': hist, 'mlp': {}, 'xgb': {}}
    for key in result:
        filename = getattr(args, key+'_params', None)
        if filename:
            result[key] = json.loads(Path(filename).read_text(encoding='utf-8'))
    for key, name in [('histgb', 'HistGB'), ('mlp', 'MLP'), ('xgb', 'XGBoost')]:
        if name not in args.models:
            continue
        model = make_estimator(name, result['histgb'], args.ridge_alpha, args.seed, result['mlp'], result['xgb'])
        if hasattr(model, 'steps'):
            model = model.steps[-1][1]
        # XGBoost espone missing=NaN: non serializzare un JSON non standard né passare None al posto di NaN.
        result[key] = {k: v for k, v in model.get_params(deep=False).items()
                       if not (isinstance(v, float) and not math.isfinite(v))}
    return result


# Stesse righe/fold/target per entrambi gli encoder. Questa CV non è il test esterno 17 AOI.
def run(args, equivalence=None):
    import platform
    import subprocess
    from importlib.metadata import version
    from .paired import pair_features
    from .evaluation import benchmark
    from .report import write_report
    from .exports import export_excel
    inputs = paths(args)
    original_inputs = dict(inputs)
    out = fresh_run(inputs['out'])
    report = out/'REPORT_V1_V2_DENSE_MSA.md' if not args.report else Path(args.report)
    if report.resolve().parent != out.resolve():
        raise ValueError('--report deve essere nella cartella del nuovo run')
    if args.bundle:
        frame, manifest = load_bundle(args.bundle)
        inputs = {'bundle': str(Path(args.bundle).resolve()), 'out': str(out), 'aoi_id': manifest['aoi_id']}
        if manifest['folds'] != args.folds:
            raise ValueError('Il pacchetto contiene fold congelati diversi: non cambiarli nel run')
    else:
        require(inputs, ['v1_parquet', 'v2_tiles'])
        from .v2_support import aggregate_v2
        v1 = pd.read_parquet(args.v1_features or inputs['v1_parquet'])
        v2 = pd.read_parquet(args.v2_features) if args.v2_features else aggregate_v2(inputs['v2_tiles'], n_total=v1.n_total)
        frame = pair_features(v1, v2)
        frame['aoi_id'] = args.aoi_id
    config = parameters(args)
    models = tuple(dict.fromkeys(['DummyMean', *args.models]))
    protocol = {'created': timestamp(), 'paths': inputs, 'n_cells': len(frame), 'folds': args.folds,
                'report_file': report.name,
                'seed': args.seed, 'bootstrap': args.bootstrap, 'block_m': args.block_m,
                'histgb_params': config['histgb'], 'mlp_params': config['mlp'], 'xgb_params': config['xgb'],
                'ridge_alpha': args.ridge_alpha, 'models': models, 'pooling': 'dense_arithmetic_mean_128',
                'checkpoint_sha256': WEIGHTS_SHA256, 'fine_tuning': False,
                'bundle_manifest_sha256': digest(Path(args.bundle)/'bundle.json') if args.bundle else None,
                'software': {k: version(k) for k in ['numpy', 'pandas', 'scikit-learn', 'xgboost', 'torch']},
                'python': platform.python_version(), 'xgboost_missing_value_policy': 'NaN_default',
                'equivalence': equivalence or {}, 'evaluation': 'within_AOI_spatial_OOF_not_external_transfer'}
    code = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=Path(__file__).resolve().parents[3],
                          capture_output=True, text=True, timeout=10)
    protocol['code_revision'] = code.stdout.strip() if code.returncode == 0 else 'unavailable'
    protocol['code_files_sha256'] = {p.name: digest(p) for p in sorted(Path(__file__).parent.glob('*.py'))}
    started = time.perf_counter()
    with FileLock(str(out/'.run.lock'), timeout=0):
        fresh_run(out)
        frame.to_parquet(out/'paired_features.parquet')
        protocol['paired_features_sha256'] = digest(out/'paired_features.parquet')
        save_json(out/'run_state.json', {'status': 'running', 'created': protocol['created']})
        save_json(out/'protocol.json', protocol)
        try:
            result = benchmark(frame, config['histgb'], out, n_splits=args.folds, seed=args.seed,
                               ridge_alpha=args.ridge_alpha, n_boot=args.bootstrap, block_m=args.block_m,
                               models=models, mlp_params=config['mlp'], xgb_params=config['xgb'])
            from .historical import evaluate_reference
            historical = evaluate_reference(frame, original_inputs['v1_model'], original_inputs['reference'])
            if historical:
                save_json(out/'historical_v1.json', historical)
            write_report(report, {**result, 'equivalence': equivalence or {}, 'paths': inputs,
                                  'protocol': protocol, 'historical': historical})
            export_excel(out)
            save_json(out/'run_state.json', {'status': 'complete', 'seconds': time.perf_counter()-started})
        except BaseException:
            save_json(out/'run_state.json', {'status': 'failed_or_interrupted', 'seconds': time.perf_counter()-started})
            raise
    print(f'Run completo: {out}', flush=True)
    return result, protocol


# I controlli terminano prima di ogni fit; il bundle evita tutte le dipendenze dai percorsi Mac.
def all_steps(args):
    if args.bundle:
        if not args.skip_encoder_check and not paths(args)['checkpoint']:
            raise ValueError('all --bundle richiede --checkpoint; verify-bundle può controllare i soli dati')
        proof = verify_bundle(args.bundle, None if args.skip_encoder_check else paths(args)['checkpoint'],
                              args.device, args.pixel_batch, args.tolerance)
        equivalence = {'bundle': proof, 'encoder': proof.get('encoder')}
    else:
        from .support import verify_support
        from .grid import load_target
        equivalence = {'v1': check_v1(args), 'v2': check_v2(args)}
        inputs = paths(args)
        equivalence['support'] = verify_support(inputs['v1_tiles'], inputs['v2_tiles'], load_target(inputs['target']),
                                                pd.read_parquet(inputs['v1_parquet']))
        equivalence['encoder'] = None if args.skip_encoder_check else check_encoder(args)
        args.v1_features = str(Path(args.out)/'v1_recomputed.parquet')
        args.v2_features = str(Path(args.out)/'v2_recomputed.parquet')
    return run(args, equivalence)


# Gli argomenti vengono validati prima di letture estese o scritture di risultati.
def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if not 0 < args.min_valid_fraction <= 1 or args.pixels < 1 or args.pixel_batch < 1 or args.tolerance <= 0 or args.folds < 2:
        parser.error('Copertura, batch, tolleranza o fold non validi')
    if hasattr(args, 'bootstrap') and (args.bootstrap < 1 or args.block_m < 1):
        parser.error('Bootstrap e block-m devono essere positivi')
    if args.command == 'pack':
        inputs = paths(args)
        require(inputs, ['v1_tiles', 'v2_tiles', 'target', 'v1_parquet'])
        pack(inputs, args.out, args.min_valid_fraction, args.pixels, args.folds)
    elif args.command == 'fetch-bundle':
        from .download import fetch_bundle
        fetch_bundle(args.release_spec, args.out)
    elif args.command == 'verify-bundle':
        if not args.bundle:
            parser.error('verify-bundle richiede --bundle')
        print(verify_bundle(args.bundle, paths(args)['checkpoint'], args.device, args.pixel_batch, args.tolerance))
    elif args.command == 'export':
        from .exports import export_run
        print(export_run(Path(args.out)))
    else:
        {'check-v1': check_v1, 'check-v2': check_v2, 'check-encoder': check_encoder,
         'run': run, 'all': all_steps}[args.command](args)
    return 0
