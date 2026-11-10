# SCOPO: preparare sul Mac un pacchetto denso verificato da copiare sul PC universitario.
# Contiene feature appaiate, campione encoder, target e prove/hash; non serve copiare tutti i Sentinel.
# Non cambia l'aggregazione V1, non usa i 25 punti e non inventa le altre 17 AOI mancanti.
# Il pacchetto non è un risultato di training e non viene caricato automaticamente su GitHub.
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import time
import zipfile
import numpy as np
import pandas as pd
from biomap_v2.common import digest, save_json, WEIGHTS_SHA256
from .grid import load_target
from .paired import pair_features
from .v1_support import aggregate_v1, compare_v1
from .v2_support import aggregate_v2, compare_v2
from .support import verify_support
from .sample import create_sample, verify_sample

FORMAT = 'biomap_dense_paired_bundle_v1'


# Un nuovo pacchetto non deve sovrascrivere prove o dati precedenti.
def pack(paths, destination, min_valid_fraction=.8, pixels=32, folds=5):
    destination = Path(destination)
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError('Cartella pacchetto non vuota: usare una nuova destinazione')
    destination.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    grid = load_target(paths['target'])
    if not np.allclose([grid.transform.a, grid.transform.e], [1/360, -1/360], atol=1e-10, rtol=0):
        raise ValueError('Il pacchetto operativo richiede il target MSA nativo 10 arcsec')
    reference = pd.read_parquet(paths['v1_parquet'])
    print('Controllo del supporto fisico V1/V2', flush=True)
    support = verify_support(paths['v1_tiles'], paths['v2_tiles'], grid, reference)
    print('Ricalcolo delle medie V1 da tile: nessun training', flush=True)
    v1 = aggregate_v1(paths['v1_tiles'], grid, edge_op='trunc', min_valid_fraction=min_valid_fraction,
                      progress=lambda i, n, name: print(f'Aggregazione V1 {i}/{n}: {name}', flush=True))
    v1_check = compare_v1(v1, reference)
    strict_equivalence(v1_check, 1e-6, 'V1', require_total=True)
    print('Aggregazione V2 di tutti i pixel densi già calcolati', flush=True)
    v2 = aggregate_v2(paths['v2_tiles'], n_total=v1.n_total, min_valid_fraction=min_valid_fraction)
    v2_check = {'reference_available': False}
    if paths.get('v2_parquet'):
        v2_check = compare_v2(v2, pd.read_parquet(paths['v2_parquet']))
        strict_equivalence(v2_check, 1e-5, 'V2')
    paired = pair_features(v1, v2)
    paired['aoi_id'] = paths.get('aoi_id', 'liguria_levante')
    from .evaluation import spatial_folds
    paired['predefined_fold'] = spatial_folds(paired.lon, paired.lat, n_splits=folds)
    paired.to_parquet(destination/'paired_features.parquet', compression='zstd')
    shutil.copy2(paths['target'], destination/'msa_2020_unitless.tif')
    save_json(destination/'support_verification.json', support)
    save_json(destination/'aggregation_equivalence.json', {'v1': v1_check, 'v2': v2_check})
    sample = create_sample(paths['v2_tiles'], destination/'encoder_sample.npz', pixels)
    names = ['paired_features.parquet', 'msa_2020_unitless.tif', 'support_verification.json',
             'aggregation_equivalence.json', 'encoder_sample.npz']
    sources = ('# Fonti del pacchetto\n\n'
               'TESSERA V1: University of Cambridge, ucam-eo/geotessera; embedding CC0.\n'
               'https://github.com/ucam-eo/tessera#license\n'
               'TESSERA V2 Medium: checkpoint 3823be7d..., revisione 8c80b926f67cb0b44c2cd41d0226e61741503b9a.\n'
               'https://huggingface.co/geotessera/TESSERA-V-2.0-2B-M\n'
               'Target: PBL / GLOBIO4, MSA overall 2020 da GLOBIOweb; ritaglio Liguria e normalizzazione MSA-area/cell-area.\n'
               'https://www.globio.info/globioweb\nhttps://www.globio.info/globio-data-downloads\n'
               'Conservare attribuzione e termini dei dati; la licenza MIT del software non sostituisce quelli degli input.\n'
               'Campione Sentinel-1/2: dati Copernicus distribuiti tramite Microsoft Planetary Computer.\n'
               'https://planetarycomputer.microsoft.com/dataset/sentinel-2-l2a\n'
               'Il pacchetto contiene un benchmark derivato; non è un dataset di osservazioni di biodiversità in campo.\n')
    (destination/'SOURCES.md').write_text(sources, encoding='utf-8')
    names.append('SOURCES.md')
    manifest = {'format': FORMAT, 'created_utc': datetime.now(timezone.utc).isoformat(),
                'aoi_id': paths.get('aoi_id', 'liguria_levante'), 'n_cells': len(paired), 'folds': folds,
                'n_pixels': int(paired.n_valid.sum()), 'checkpoint_sha256': WEIGHTS_SHA256,
                'method': 'dense_all_valid_V1_pixel_centers_arithmetic_mean_128', 'year': 2020,
                'edge_op': 'trunc_historical', 'min_valid_fraction': min_valid_fraction,
                'encoder_sample': sample, 'seconds': time.perf_counter()-started,
                'files': {name: {'sha256': digest(destination/name), 'bytes': (destination/name).stat().st_size} for name in names},
                'source_hashes': {'v1_parquet': digest(paths['v1_parquet']), 'target': digest(paths['target'])}}
    save_json(destination/'bundle.json', manifest)
    archive = destination.with_suffix('.zip')
    if archive.exists():
        raise FileExistsError('ZIP già presente: non sovrascritto')
    with zipfile.ZipFile(archive, 'x', zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for path in sorted(destination.iterdir()):
            z.write(path, path.name)
    save_json(archive.with_suffix('.zip.sha256.json'), {'file': archive.name, 'sha256': digest(archive), 'bytes': archive.stat().st_size})
    print(f'Pacchetto: {archive} ({archive.stat().st_size/1024**2:.2f} MiB)', flush=True)
    return manifest


# La riproduzione deve coincidere in coordinate, target, conteggi, membership e feature.
def strict_equivalence(result, tolerance, name, require_total=False):
    invalid = result.get('n_common', 0) == 0 or result.get('max_abs_mean', np.inf) > tolerance
    invalid |= any(result.get(key, 1) != 0 for key in ['missing_in_recomputed', 'extra_in_recomputed', 'n_valid_mismatch'])
    if require_total:
        invalid |= result.get('n_total_mismatch', 1) != 0
        invalid |= any(result.get(key, np.inf) > 1e-8 for key in ['max_abs_lon', 'max_abs_lat', 'max_abs_target'])
    if invalid:
        raise ValueError(f'Equivalenza {name} fallita: {result}')


# Verifica i file piccoli sul PC ricevente: le prove sui raw sono state prodotte alla creazione.
def load_bundle(directory):
    directory = Path(directory).resolve()
    manifest = json.loads((directory/'bundle.json').read_text(encoding='utf-8'))
    if manifest.get('format') != FORMAT or manifest.get('checkpoint_sha256') != WEIGHTS_SHA256:
        raise ValueError('Formato/checkpoint pacchetto non compatibile')
    if not {'paired_features.parquet', 'msa_2020_unitless.tif', 'support_verification.json',
            'aggregation_equivalence.json', 'encoder_sample.npz'} <= set(manifest.get('files', {})):
        raise ValueError('Manifest incompleto: mancano file obbligatori e relativi hash')
    for name, entry in manifest['files'].items():
        path = directory/name
        if path.resolve().parent != directory or digest(path) != entry['sha256']:
            raise ValueError('File pacchetto corrotto o percorso invalido: '+name)
    paired = pd.read_parquet(directory/'paired_features.parquet')
    required = ['target', 'lon', 'lat', 'n_valid', 'n_total', 'valid_fraction']
    required += [f'{version}_mean_{i:03d}' for version in ['v1', 'v2'] for i in range(128)]
    if not set(required) <= set(paired) or paired.index.duplicated().any():
        raise ValueError('Colonne mancanti o celle duplicate nel pacchetto')
    if len(paired) != manifest['n_cells'] or not np.isfinite(paired[required].to_numpy()).all():
        raise ValueError('Numerosità/feature del pacchetto invalide')
    if not paired.target.between(0, 1).all() or (paired.n_valid <= 0).any() or (paired.n_valid > paired.n_total).any():
        raise ValueError('Target o copertura del pacchetto invalida')
    if 'predefined_fold' not in paired or set(paired.predefined_fold) != set(range(manifest['folds'])):
        raise ValueError('Fold congelati mancanti o incompatibili')
    if not np.allclose(paired.valid_fraction, paired.n_valid/paired.n_total, atol=1e-8, rtol=0):
        raise ValueError('Copertura non coerente con i conteggi')
    proof = json.loads((directory/'support_verification.json').read_text(encoding='utf-8'))
    if proof.get('status') != 'passed' or proof.get('n_cells') != len(paired) or proof.get('pixels_in_reference_cells') != int(paired.n_valid.sum()):
        raise ValueError('Prova di supporto fisico incompatibile con le feature')
    return paired, manifest


# All sul pacchetto ricontrolla anche il Medium locale sul campione delle serie originali.
def verify_bundle(directory, checkpoint=None, device='cpu', pixel_batch=16, tolerance=1e-4):
    frame, manifest = load_bundle(directory)
    result = {'status': 'passed', 'n_cells': len(frame), 'n_pixels': int(frame.n_valid.sum()),
              'support': 'source_audit_imported_and_bundle_hashes_verified'}
    if checkpoint:
        result['encoder'] = verify_sample(Path(directory)/'encoder_sample.npz', checkpoint, device, pixel_batch, tolerance)
    else:
        result['encoder'] = {'status': 'not_recomputed_no_checkpoint'}
    return result
