# Scopo: CLI del benchmark denso V1/V2 (equivalenze, appaiamento, CV, report).
# Fasi: check-v1 -> check-v2 -> check-encoder -> run -> report; 'all' le esegue in sequenza.
# Input: tile V1, tile V2, raster target, parquet di riferimento, modello V1 congelato.
# Output: feature ricalcolate, JSON di equivalenza, predizioni OOF, metriche e report.
# Parametri: percorsi via flag o variabili BIOMAP_*; vedere --help.
# Risorse: CPU; GPU/MPS solo se un tile V2 va completato.
# Ripresa: gli embedding V2 usano memmap + progresso; i comandi sono idempotenti sui file.
import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd


def _env(name, default=None):
    value = os.environ.get(name)
    return value if value else default


def _paths(args):
    v1_tiles = args.v1_tiles or _env('BIOMAP_V1_TILES')
    v2_tiles = args.v2_tiles or _env('BIOMAP_V2_TILES')
    target = args.target or _env('BIOMAP_MSA_TARGET')
    return {'v1_tiles': v1_tiles, 'v2_tiles': v2_tiles, 'target': target,
            'v1_parquet': args.v1_parquet or _env('BIOMAP_V1_PARQUET'),
            'v2_parquet': args.v2_parquet or _env('BIOMAP_V2_PARQUET'),
            'checkpoint': args.checkpoint or _env('BIOMAP_STUDENT_CKPT'),
            'v1_model': args.v1_model or _env('BIOMAP_V1_MODEL'),
            'reference': args.reference or _env('BIOMAP_V1_REFERENCE'),
            'out': args.out}


def _require(paths, keys):
    missing = [key for key in keys if not paths.get(key)]
    if missing:
        raise SystemExit(f'Percorsi mancanti: {", ".join(missing)}. Usare i flag o le variabili BIOMAP_*.')


def _read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str) + '\n', encoding='utf-8')
    temporary.replace(path)


def _timestamp():
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')


def check_v1(args):
    from .grid import load_target
    from .v1_support import aggregate_v1, compare_v1
    paths = _paths(args)
    _require(paths, ('v1_tiles', 'target', 'v1_parquet'))
    out = Path(paths['out'])
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    target = load_target(paths['target'])
    reference = pd.read_parquet(paths['v1_parquet'])
    results = {}
    frames = {}
    for op in ('trunc', 'floor'):
        frame = aggregate_v1(paths['v1_tiles'], target, edge_op=op,
                             min_valid_fraction=args.min_valid_fraction)
        results[op] = compare_v1(frame, reference)
        frames[op] = frame
        print(f"V1 {op}: {results[op]}", flush=True)
    frames['trunc'].to_parquet(out / 'v1_recomputed.parquet')
    result = {'reference': paths['v1_parquet'], 'seconds': time.time() - started, **results}
    _write_json(out / 'v1_equivalence.json', result)
    strict = results['trunc']
    if strict['n_common'] == 0 or strict.get('max_abs_mean', 1.0) > 1e-6 \
            or strict.get('n_valid_mismatch', 1) != 0 or strict.get('n_total_mismatch', 1) != 0 \
            or strict.get('missing_in_recomputed', 1) != 0:
        raise SystemExit('Equivalenza V1 (trunc) fallita: confronto non valido')
    return result


def check_v2(args):
    from .v2_support import aggregate_v2, compare_v2
    paths = _paths(args)
    _require(paths, ('v2_tiles', 'v1_parquet'))
    out = Path(paths['out'])
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    v1 = pd.read_parquet(paths['v1_parquet'])
    frame = aggregate_v2(paths['v2_tiles'], n_total=v1['n_total'],
                         min_valid_fraction=args.min_valid_fraction, include_std=True)
    frame.to_parquet(out / 'v2_recomputed.parquet')
    result = {'tiles': paths['v2_tiles'], 'seconds': time.time() - started}
    if paths.get('v2_parquet') and Path(paths['v2_parquet']).is_file():
        reference = pd.read_parquet(paths['v2_parquet'])
        result['comparison'] = compare_v2(frame, reference)
        print(f"V2: {result['comparison']}", flush=True)
        strict = result['comparison']
        if strict['n_common'] == 0 or strict.get('max_abs_mean', 1.0) > 1e-5 \
                or strict.get('n_valid_mismatch', 1) != 0:
            raise SystemExit('Equivalenza V2 fallita: riaggregazione non coerente')
    _write_json(out / 'v2_equivalence.json', result)
    return result


def check_encoder(args):
    from .v2_support import discover_tiles, _doys, _load_array
    from .. import encoder
    from ..reference import encode_pixels
    import torch

    paths = _paths(args)
    _require(paths, ('v2_tiles', 'checkpoint'))
    tiles = [tile for tile in discover_tiles(paths['v2_tiles'], require_complete=True) if tile['complete']]
    if not tiles:
        raise SystemExit('Nessun tile V2 completo per il controllo encoder')
    tile = tiles[0]
    pixels = int(args.pixels)
    directory = tile['directory']
    geometry = np.load(tile['geometry'])
    total = len(geometry['flat_indices'])
    pixels = min(pixels, total)
    manifest = _read_json(directory / 'inputs_manifest.json')
    doys = _doys(manifest)
    inputs = {key: _load_array(directory, key) for key in ('s2', 'mask', 'asc', 'desc')}
    s2 = np.ascontiguousarray(inputs['s2'][:, :pixels].transpose(1, 0, 2))
    mask = np.ascontiguousarray(inputs['mask'][:, :pixels].T)
    asc = np.ascontiguousarray(inputs['asc'][:, :pixels].transpose(1, 0, 2))
    desc = np.ascontiguousarray(inputs['desc'][:, :pixels].transpose(1, 0, 2))
    cfg = {'indicator': 'msa', 'model': {'device': args.device, 'checkpoint': paths['checkpoint']}}
    model, _, _ = encoder.load_encoder(cfg, override=paths['checkpoint'])
    with torch.no_grad():
        computed = encode_pixels(model, s2, doys['s2'], s1_asc_bands=asc, s1_asc_doys=doys['asc'],
                                 s1_desc_bands=desc, s1_desc_doys=doys['desc'], s2_masks=mask,
                                 batch_pixels=int(args.pixel_batch), device=args.device)
    stored = np.load(directory / 'embeddings_fp32.npy', mmap_mode='r')[:pixels]
    maximum = float(np.max(np.abs(computed - stored)))
    tolerance = float(args.tolerance)
    result = {'tile': tile['name'], 'pixels': pixels, 'max_abs': maximum,
              'tolerance': tolerance, 'reference': str(directory / 'embeddings_fp32.npy')}
    print(f"Encoder: {result}", flush=True)
    out = Path(paths['out'])
    out.mkdir(parents=True, exist_ok=True)
    _write_json(out / 'encoder_equivalence.json', result)
    if maximum > tolerance:
        raise SystemExit('Equivalenza encoder V2 fallita')
    return result


def _historical(v1, paths):
    if not paths.get('v1_model') or not Path(paths['v1_model']).is_file():
        return None
    try:
        import joblib
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            payload = joblib.load(paths['v1_model'])
        model = payload['model'] if isinstance(payload, dict) else payload
        names = [f'mean_{i:03d}' for i in range(128)]
        prediction = model.predict(v1[names].to_numpy(np.float32))
        from .evaluation import full_metrics
        metrics = full_metrics(v1.msa_overall.to_numpy(), prediction)
        result = {'metrics': metrics, 'model': str(paths['v1_model'])}
        if paths.get('reference') and Path(paths['reference']).is_file():
            reference = _read_json(paths['reference']).get('metrics', {}).get('HistGB', {})
            deltas = [abs(metrics[key] - reference[key]) for key in ('mae', 'rmse', 'mape_pct', 'r2', 'bias')
                      if key in reference and metrics.get(key) is not None]
            result['max_abs_reference_delta'] = float(max(deltas)) if deltas else None
        return result
    except Exception as error:
        return {'error': f'{type(error).__name__}: {error}'}


def run(args, equivalence=None):
    from .paired import pair_features
    from .evaluation import benchmark
    from .report import write_report

    paths = _paths(args)
    _require(paths, ('v1_parquet', 'v2_tiles'))
    out = Path(paths['out'])
    out.mkdir(parents=True, exist_ok=True)
    v1_path = args.v1_features or paths['v1_parquet']
    v1 = pd.read_parquet(v1_path)
    if args.v2_features:
        v2 = pd.read_parquet(args.v2_features)
    else:
        from .v2_support import aggregate_v2
        v2 = aggregate_v2(paths['v2_tiles'], n_total=v1['n_total'],
                          min_valid_fraction=args.min_valid_fraction, include_std=False)
    paired = pair_features(v1, v2)
    paired.to_parquet(out / 'paired_features.parquet')
    print(f'Celle appaiate: {len(paired)}', flush=True)

    histgb_params = None
    if args.histgb_params:
        histgb_params = _read_json(args.histgb_params)
    if histgb_params is None:
        histgb_params = {'max_iter': 90, 'max_leaf_nodes': 15, 'learning_rate': 0.05,
                         'l2_regularization': 1.0, 'early_stopping': True, 'random_state': 20260925}
    models = tuple(args.models)
    result = benchmark(paired, histgb_params, out, n_splits=args.folds, seed=args.seed,
                       ridge_alpha=args.ridge_alpha, n_boot=args.bootstrap,
                       block_m=args.block_m, models=models)
    historical = _historical(v1, paths)
    if historical:
        _write_json(out / 'historical_v1.json', historical)
    protocol = {'created': _timestamp(), 'paths': paths, 'n_cells': len(paired),
                'folds': args.folds, 'seed': args.seed, 'bootstrap': args.bootstrap,
                'block_m': args.block_m, 'histgb_params': histgb_params,
                'ridge_alpha': args.ridge_alpha, 'models': list(models),
                'min_valid_fraction': args.min_valid_fraction,
                'v1_features': str(v1_path),
                'v2_features': str(args.v2_features) if args.v2_features else 'recomputed_from_tiles',
                'equivalence': equivalence or {}}
    _write_json(out / 'protocol.json', protocol)
    if args.report:
        context = {'metrics': result['metrics'], 'paired_delta': result['paired_delta'],
                   'equivalence': (equivalence or {}), 'historical': historical or {},
                   'paths': {key: str(value) for key, value in paths.items() if value}}
        write_report(args.report, context)
        print(f'Report: {args.report}', flush=True)
    return result, protocol


def all_steps(args):
    equivalence = {}
    equivalence['v1'] = check_v1(args)
    equivalence['v2'] = check_v2(args)
    if args.skip_encoder_check:
        equivalence['encoder'] = None
    else:
        try:
            equivalence['encoder'] = check_encoder(args)
        except Exception as error:
            equivalence['encoder'] = {'error': f'{type(error).__name__}: {error}'}
    args.v1_features = str(Path(args.out) / 'v1_recomputed.parquet')
    args.v2_features = str(Path(args.out) / 'v2_recomputed.parquet')
    return run(args, equivalence)


def build_parser():
    parser = argparse.ArgumentParser(description='Benchmark denso V1/V2 TESSERA GLOBIO MSA')
    subparsers = parser.add_subparsers(dest='command', required=True)

    def common(sub):
        sub.add_argument('--v1-tiles', help='radice tessera/2020 V1 (env BIOMAP_V1_TILES)')
        sub.add_argument('--v2-tiles', help='radice tile V2 Medium (env BIOMAP_V2_TILES)')
        sub.add_argument('--target', help='raster MSA unitless (env BIOMAP_MSA_TARGET)')
        sub.add_argument('--v1-parquet', help='parquet V1 denso di riferimento (env BIOMAP_V1_PARQUET)')
        sub.add_argument('--v2-parquet', help='parquet V2 denso di riferimento (env BIOMAP_V2_PARQUET)')
        sub.add_argument('--checkpoint', help='checkpoint student_medium.pt (env BIOMAP_STUDENT_CKPT)')
        sub.add_argument('--v1-model', help='modello V1 congelato .joblib (env BIOMAP_V1_MODEL)')
        sub.add_argument('--reference', help='JSON metriche storiche V1 (env BIOMAP_V1_REFERENCE)')
        sub.add_argument('--out', default=os.environ.get('BIOMAP_DENSE_OUT', 'outputs/dense_benchmark'))
        sub.add_argument('--min-valid-fraction', type=float, default=0.8)

    check1 = subparsers.add_parser('check-v1', help='ricalcola V1 e confronta con il parquet storico')
    common(check1)
    check2 = subparsers.add_parser('check-v2', help='riaggrega V2 e confronta con il parquet storico')
    common(check2)
    encoder = subparsers.add_parser('check-encoder', help='verifica encoder attivo vs embedding storici')
    common(encoder)
    encoder.add_argument('--pixels', type=int, default=256)
    encoder.add_argument('--device', default='cpu')
    encoder.add_argument('--pixel-batch', type=int, default=64)
    encoder.add_argument('--tolerance', type=float, default=1e-4)

    runner = subparsers.add_parser('run', help='CV spaziale appaiata V1/V2 e report')
    common(runner)
    runner.add_argument('--v1-features', help='parquet V1 ricalcolato (default: --v1-parquet)')
    runner.add_argument('--v2-features', help='parquet V2 ricalcolato (default: aggrega --v2-tiles)')
    runner.add_argument('--folds', type=int, default=5)
    runner.add_argument('--seed', type=int, default=20260925)
    runner.add_argument('--bootstrap', type=int, default=2000)
    runner.add_argument('--block-m', type=int, default=2000)
    runner.add_argument('--ridge-alpha', type=float, default=100.0)
    runner.add_argument('--models', nargs='+', default=['DummyMean', 'Ridge', 'HistGB'])
    runner.add_argument('--histgb-params', help='JSON con iperparametri HistGB congelati')
    runner.add_argument('--report', default='reports/REPORT_V1_V2_DENSE_MSA.md')

    everyone = subparsers.add_parser('all', help='check-v1 + check-v2 + check-encoder + run')
    common(everyone)
    everyone.add_argument('--skip-encoder-check', action='store_true')
    everyone.add_argument('--folds', type=int, default=5)
    everyone.add_argument('--seed', type=int, default=20260925)
    everyone.add_argument('--bootstrap', type=int, default=2000)
    everyone.add_argument('--block-m', type=int, default=2000)
    everyone.add_argument('--ridge-alpha', type=float, default=100.0)
    everyone.add_argument('--models', nargs='+', default=['DummyMean', 'Ridge', 'HistGB'])
    everyone.add_argument('--histgb-params')
    everyone.add_argument('--device', default='cpu')
    everyone.add_argument('--pixels', type=int, default=256)
    everyone.add_argument('--pixel-batch', type=int, default=64)
    everyone.add_argument('--tolerance', type=float, default=1e-4)
    everyone.add_argument('--report', default='reports/REPORT_V1_V2_DENSE_MSA.md')
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == 'check-v1':
        check_v1(args)
    elif args.command == 'check-v2':
        check_v2(args)
    elif args.command == 'check-encoder':
        check_encoder(args)
    elif args.command == 'run':
        run(args)
    elif args.command == 'all':
        all_steps(args)
    else:
        parser.error('comando sconosciuto')
    return 0


if __name__ == '__main__':
    sys.exit(main())
