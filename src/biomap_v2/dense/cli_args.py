# SCOPO: parser unico del probing denso, con percorsi raw oppure pacchetto portabile.
# pack prepara i dati; verify-bundle li verifica; all controlla prima del training.
import argparse
import os
from datetime import datetime, timezone


# Timestamp UTC: un nuovo run non riutilizza l'output del precedente.
def timestamp():
    return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')


# Nessun percorso Mac è obbligatorio sul PC Windows: i flag prevalgono sull'ambiente.
def paths(args):
    names = {'v1_tiles': 'BIOMAP_V1_TILES', 'v2_tiles': 'BIOMAP_V2_TILES', 'target': 'BIOMAP_MSA_TARGET',
             'v1_parquet': 'BIOMAP_V1_PARQUET', 'v2_parquet': 'BIOMAP_V2_PARQUET',
             'checkpoint': 'BIOMAP_STUDENT_CKPT', 'v1_model': 'BIOMAP_V1_MODEL', 'reference': 'BIOMAP_V1_REFERENCE'}
    result = {key: getattr(args, key, None) or os.environ.get(env) for key, env in names.items()}
    result.update(out=args.out, aoi_id=args.aoi_id)
    return result


# Mancanza dati esplicita: nessuna acquisizione o sostituzione con i 25 punti.
def require(inputs, keys):
    missing = [key for key in keys if not inputs.get(key)]
    if missing:
        raise ValueError('Percorsi mancanti: '+', '.join(missing))


# Modelli e iperparametri sono dichiarati prima del run, senza ottimizzazione sul test.
def build_parser():
    parser = argparse.ArgumentParser(description='Probing denso appaiato TESSERA V1/V2 — MSA')
    commands = parser.add_subparsers(dest='command', required=True)
    for name in ['pack', 'fetch-bundle', 'verify-bundle', 'check-v1', 'check-v2', 'check-encoder', 'run', 'all', 'export']:
        sub = commands.add_parser(name)
        for option in ['v1-tiles', 'v2-tiles', 'target', 'v1-parquet', 'v2-parquet', 'checkpoint', 'v1-model', 'reference']:
            sub.add_argument('--'+option)
        sub.add_argument('--bundle', help='Cartella estratta contenente bundle.json')
        if name == 'fetch-bundle':
            from biomap_v2.common import ROOT
            sub.add_argument('--release-spec', default=str(ROOT/'configs/dense_bundle_release.json'))
        sub.add_argument('--out', default=os.environ.get('BIOMAP_DENSE_OUT', 'outputs/dense_benchmark/'+timestamp()))
        sub.add_argument('--aoi-id', default='liguria_levante')
        sub.add_argument('--min-valid-fraction', type=float, default=.8)
        sub.add_argument('--device', default='cpu')
        sub.add_argument('--pixels', type=int, default=32)
        sub.add_argument('--pixel-batch', type=int, default=16)
        sub.add_argument('--tolerance', type=float, default=1e-4)
        sub.add_argument('--folds', type=int, default=5)
        if name in {'all', 'run'}:
            sub.add_argument('--v1-features')
            sub.add_argument('--v2-features')
            sub.add_argument('--seed', type=int, default=20260925)
            sub.add_argument('--bootstrap', type=int, default=2000)
            sub.add_argument('--block-m', type=int, default=2000)
            sub.add_argument('--ridge-alpha', type=float, default=100.)
            sub.add_argument('--models', nargs='+', choices=['DummyMean', 'Ridge', 'XGBoost', 'HistGB', 'MLP'],
                             default=['DummyMean', 'Ridge', 'XGBoost', 'HistGB', 'MLP'])
            sub.add_argument('--histgb-params')
            sub.add_argument('--mlp-params')
            sub.add_argument('--xgb-params')
            sub.add_argument('--report', help='Destinazione nella cartella del nuovo run')
        if name == 'all':
            sub.add_argument('--skip-encoder-check', action='store_true', help='Esplicito, registrato come controllo non eseguito')
    return parser
