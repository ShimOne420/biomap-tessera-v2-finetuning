# SCOPO: audit in sola lettura dei dati del probing e nuovo indice E2 separato.
# INPUT: cells.csv/json, manifest.json per cella, serie Sentinel e target GLOBIO esistenti.
# CONTROLLI: checksum, protocollo, fold, unità, griglia, target, punti, quattro eccezioni.
# OUTPUT: index.csv e audit.json SOLO nella work-root; nessun prepare/download automatico.
# Ogni cella rimane nel suo fold: un problema interrompe il controllo, non cambia il campione.
import json
from pathlib import Path
import numpy as np
import pandas as pd
from biomap_v2.common import WEIGHTS_SHA256, digest, save_json
from biomap_v2.benchmark import manifest_digest, validate_protocol
from biomap_v2.data import PROTOCOL as SERIES_PROTOCOL, BANDS
from biomap_v2.target_data import locate_target, is_ratio, check_grid, sample_raster
from . import PROTOCOL
from .settings import identity, save_csv, work_budget
from .spatial import lattice, validate_raw


# Il nome su disco è derivato dal cell_id verificato nel manifest ufficiale.
def series_path(cfg, cell_id):
    if '/' in cell_id or '\\' in cell_id or '..' in cell_id:
        raise ValueError('cell_id non valido per il percorso serie')
    path = Path(cfg['source'])/'series'/cell_id.replace(':', '_')/'series.npz'
    if Path(cfg['source']).resolve() not in path.resolve().parents:
        raise ValueError('Serie symlink esterna alla sorgente')
    return path


# Confronta i valori indicizzati con il raster nativo, dividendo MSA-area una volta sola.
def verify_targets(legacy, frame, source_info):
    path = locate_target(legacy)
    denominator = path.with_name('cellarea_km2_10sec.tif') if is_ratio(legacy, path) else None
    check_grid(legacy, path, denominator)
    expected = source_info.get('target_source', {})
    actual_hash = digest(path)
    if expected.get('sha256') != actual_hash:
        raise ValueError('Target diverso da quello registrato dal prepare')
    values, inside, nodata, aligned = sample_raster(path, frame)
    hashes = {'raster': str(path), 'sha256': actual_hash, 'formula': 'already unitless'}
    if denominator:
        den_hash = digest(denominator)
        if expected.get('cellarea_sha256') != den_hash:
            raise ValueError('Denominatore diverso da quello registrato')
        den, dinside, dnodata, daligned = sample_raster(denominator, frame)
        if not (dinside & ~dnodata & daligned & np.isfinite(den) & (den > 0)).all():
            raise ValueError('Denominatore invalido o disallineato')
        values = values/den
        hashes.update(cellarea_sha256=den_hash, formula='MSA-area / cell-area')
    elif legacy['target']['units'] != 'unitless':
        raise ValueError('E2 accetta MSA normalizzato oppure il rapporto MSA-area/cell-area')
    if not (inside & ~nodata & aligned & np.isfinite(values) & (values >= 0) & (values <= 1)).all():
        raise ValueError('Target invalido, fuori copertura o disallineato')
    if not np.allclose(values, frame.target, atol=1e-6, rtol=0):
        raise ValueError('Target indicizzato diverso dal raster originale')
    return hashes


# Verifica il checkpoint locale senza scaricarlo e senza scrivere provenance nella sorgente.
def verify_checkpoint(cfg):
    path = Path(cfg['checkpoint'])
    if not path.is_file():
        raise FileNotFoundError(f'Medium assente: usare --checkpoint con il file già scaricato: {path}')
    value = digest(path)
    if value != WEIGHTS_SHA256:
        raise ValueError('SHA256 checkpoint Medium discordante')
    return value


# L'audit può essere ripetuto; l'identità vincola dati, griglia e precisione della cache.
def build_index(cfg, legacy):
    source = Path(cfg['source'])
    table = source/'cells.csv'
    info = json.loads(table.with_suffix('.json').read_text(encoding='utf-8'))
    if digest(table) != info['eligible_sha256']:
        raise ValueError('cells.csv sorgente modificato dopo prepare')
    if info['source_cells_sha256'] not in {digest(legacy['cells']), manifest_digest(legacy['cells'])}:
        raise ValueError('Manifest geografico diverso dal prepare sorgente')
    if info.get('year') != 2020 or info.get('protocol') != SERIES_PROTOCOL or info['config_sampling'] != legacy['sampling']:
        raise ValueError('Anno, protocollo o sampling sorgente incompatibile')
    frame = pd.read_csv(table, dtype={'cell_id': str})
    validate_protocol(legacy, frame)
    source_frame = pd.read_csv(legacy['cells'], dtype={'cell_id': str}).set_index('cell_id')
    cols = ['lon', 'lat', 'west', 'south', 'east', 'north']
    if not np.allclose(frame[cols], source_frame.loc[frame.cell_id, cols], atol=1e-10, rtol=0):
        raise ValueError('Geometrie sorgente diverse dal manifest ufficiale')
    if not np.allclose(frame.east-frame.west, 1/360, atol=1e-10, rtol=0) or not np.allclose(frame.north-frame.south, 1/360, atol=1e-10, rtol=0):
        raise ValueError('Supporto diverso dalla cella nativa di 10 secondi di arco')
    target = verify_targets(legacy, frame, info)
    checkpoint_hash = verify_checkpoint(cfg)
    counts, waivers = [], []
    for i, cell in enumerate(frame.itertuples()):
        path = series_path(cfg, cell.cell_id)
        meta = json.loads(path.with_name('manifest.json').read_text(encoding='utf-8'))
        if digest(path) != cell.series_sha256 or meta['sha256'] != cell.series_sha256:
            raise ValueError(f'Serie modificata: {cell.cell_id}')
        if meta.get('protocol') != SERIES_PROTOCOL or meta.get('year') != 2020 or meta.get('sampling') != legacy['sampling'] or meta.get('band_order') != BANDS:
            raise ValueError(f'Metadati serie incompatibili: {cell.cell_id}')
        if not np.allclose(meta.get('bounds', []), [cell.west, cell.south, cell.east, cell.north], atol=1e-10, rtol=0):
            raise ValueError('Bounds del manifest serie incompatibili')
        with np.load(path, allow_pickle=False) as z:
            raw = {key: z[key] for key in z.files}
        validate_raw(raw, cell)
        good = raw['mask'].sum(axis=1) >= legacy['sampling']['min_s2_observations']
        good &= (np.any(raw['asc'] != 0, axis=-1).sum(axis=1)+np.any(raw['desc'] != 0, axis=-1).sum(axis=1)) >= legacy['sampling']['min_s1_observations']
        if not good.all():
            raise ValueError('Punti salvati sotto le soglie di osservazioni Sentinel')
        slots = lattice(raw['points'], cell)
        fraction = len(slots)/25
        if not np.isclose(fraction, cell.valid_fraction) or not np.isclose(fraction, meta['valid_fraction']):
            raise ValueError('Copertura discordante fra serie e indice')
        waived = fraction < legacy['sampling']['min_valid_fraction']
        if waived and (cell.cell_id not in legacy['acquisition']['low_coverage_cells'] or not meta.get('low_coverage_accepted')):
            raise ValueError('Cella a bassa copertura non autorizzata')
        counts.append(len(slots))
        if waived:
            waivers.append({'cell_id': cell.cell_id, 'valid_points': len(slots), 'fraction': fraction})
        if (i+1) % 1000 == 0:
            print(f'Audit sorgente: {i+1}/{len(frame)} celle', flush=True)
    frame['valid_points'] = counts
    frame['quality_flag'] = np.where(frame.valid_fraction < legacy['sampling']['min_valid_fraction'], 'explicit_low_coverage', 'standard')
    fingerprint = {'protocol': PROTOCOL, 'source_cells_sha256': digest(table),
                   'checkpoint_sha256': checkpoint_hash, 'sampling': legacy['sampling'],
                   'target': target, 'cache_precision': 'float32', 'cnn_grid': 'north_up_5x5'}
    fingerprint['dataset_id'] = identity(fingerprint)
    audit = {'fingerprint': fingerprint, 'counts': frame.partition.value_counts().to_dict(),
             'n_development_aois': frame[frame.partition == 'train'].aoi_id.nunique(), 'low_coverage': waivers,
             'folds': [{'fold': fold, 'train': int(((frame.partition == 'train') & (frame.cv_fold != fold)).sum()),
                        'validation': int(((frame.partition == 'train') & (frame.cv_fold == fold)).sum())} for fold in range(3)]}
    work_budget(cfg, len(frame)*1000)
    destination = Path(cfg['work'])/'dataset'
    save_csv(destination/'index.csv', frame)
    audit['index_sha256'] = digest(destination/'index.csv')
    save_json(destination/'audit.json', audit)
    print(f'Audit completo: {len(frame)} celle; indice scritto in {destination}', flush=True)
    return frame, audit


# Training e valutazione leggono un indice già verificato: non rilanciano prepare/download.
def indexed(cfg):
    directory = Path(cfg['work'])/'dataset'
    audit = json.loads((directory/'audit.json').read_text(encoding='utf-8'))
    if digest(directory/'index.csv') != audit['index_sha256']:
        raise ValueError('Indice E2 modificato')
    if digest(Path(cfg['source'])/'cells.csv') != audit['fingerprint']['source_cells_sha256']:
        raise ValueError('cells.csv sorgente cambiato: ripetere audit prima di proseguire')
    if verify_checkpoint(cfg) != audit['fingerprint']['checkpoint_sha256']:
        raise ValueError('Checkpoint diverso dall’audit')
    return pd.read_csv(directory/'index.csv', dtype={'cell_id': str}), audit
