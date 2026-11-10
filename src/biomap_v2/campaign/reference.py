# Scopo: pacchetto portabile delle feature V1, target, supporto e fold delle 17 AOI + Liguria.
# Il pack legge gli artefatti storici: non estrae nuovi embedding, non modifica i sorgenti.
import json
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
from biomap_v2.common import digest, save_json
from biomap_v2.dense.grid import load_target
from .state import membership, MEANS, writable

# Coordinate quantizzate solo per il join; successivo controllo numerico obbligatorio.
def keys(frame):
    return list(zip(frame.lon.round(7), frame.lat.round(7)))

# Congela riferimenti globali e locali; controlla tutti i conteggi geometry prima del pack.
def pack(project, legacy, out, v1_root):
    project, legacy = Path(project), Path(legacy)
    out = writable(out, [project, legacy, v1_root])
    if (out/'reference.json').exists(): raise FileExistsError('Reference già presente')
    history = project/'outputs/runs/20260928T111520077028Z'
    report = json.loads((history/'training_dataset_report.json').read_text())
    rows = membership(); all_features = []; areas = []; evidence = []
    development = pd.read_parquet(history/'multi_aoi_training_dataset.parquet')
    for aoi, cells in rows.groupby('aoi_id', sort=True):
        folder = out/aoi; folder.mkdir(exist_ok=True)
        if aoi == 'liguria_levante':
            frame = pd.read_parquet(project/f'data/areas/{aoi}/processed/tessera_globio_msa.parquet')
            target = project/f'data/areas/{aoi}/raw/globio/msa_2020_unitless.tif'
        else:
            frame = development[development.aoi_id == aoi].copy()
            spec = next(x for x in report['per_aoi'] if x['aoi_id'] == aoi)
            target = project/spec['target_path']
        if getattr(target.stat(), 'st_flags', 0) & 0x40000000:
            raise FileNotFoundError(f'Target solo iCloud, scaricarlo prima del pack: {target}')
        grid = load_target(target)
        if not np.allclose([grid.transform.a,-grid.transform.e],[1/360,1/360],rtol=0,atol=1e-9):
            raise ValueError('Target non nativo 10 arcsec')
        idx = dict(zip(keys(frame), range(len(frame))))
        if len(idx) != len(frame): raise ValueError('Coordinate V1 duplicate')
        take = [idx.get(k, -1) for k in keys(cells)]
        if min(take) < 0: raise ValueError(f'Feature V1 mancanti: {aoi}')
        matched = frame.iloc[take].reset_index(drop=True)
        record = cells.reset_index(drop=True).copy()
        if not np.allclose(record[['lon','lat']], matched[['lon','lat']], atol=1e-8, rtol=0):
            raise ValueError('Coordinate fuori tolleranza')
        record['reference_index'] = cells.index.to_numpy()
        record['target'] = matched.msa_overall.to_numpy()
        for col in ['n_valid','n_total','valid_fraction',*MEANS]: record[col] = matched[col].to_numpy()
        local = dict(zip(zip(np.round(grid.lon,7), np.round(grid.lat,7)), range(grid.n_cells)))
        local_ids = [local[k] for k in keys(cells)]
        lookup = np.full(grid.n_cells, -1, dtype=np.int64)
        lookup[local_ids] = cells.index.to_numpy()
        if not np.allclose(record.target, grid.values[grid.rows[local_ids],grid.cols[local_ids]], atol=1e-7, rtol=0):
            raise ValueError('Target V1/raster discordanti')
        count = np.zeros(grid.n_cells, dtype=np.int64); tiles = []
        for tile in sorted((legacy/aoi).glob('grid_*')):
            src = tile/'geometry.npz'; spec = json.loads((tile/'geometry.json').read_text())
            if digest(src) != spec['geometry_sha256']: raise ValueError('Geometry corrotta')
            with np.load(src) as z:
                ids = z['cell_ids']; flat = z['flat_indices']
                if len(np.unique(flat)) != len(flat): raise ValueError('Pixel duplicati nella tile')
                count += np.bincount(ids, minlength=grid.n_cells)
            dest = folder/tile.name; dest.mkdir(exist_ok=True)
            shutil.copy2(src, dest/'geometry.npz'); save_json(dest/'geometry.json',spec)
            tiles.append(tile.name)
        if not np.array_equal(count[local_ids], record.n_valid.to_numpy()):
            raise ValueError(f'Supporto storico non riprodotto: {aoi}; nessun probing consentito')
        from .physical import verify
        proof=verify(Path(v1_root)/aoi/'tessera/2020', legacy/aoi, grid, local_ids, record)
        save_json(folder/'physical_support.json',proof)
        shutil.copy2(target, folder/'target.tif')
        np.save(folder/'lookup.npy', lookup)
        evidence.append({'aoi_id':aoi,'n_cells':len(record),'pixel_contributions':int(record.n_valid.sum()),
                         'support':'legacy geometry matches V1 counts; cross-tile overlap audit required'})
        all_features.append(record); areas.append({'aoi_id':aoi,'tiles':tiles})
    pd.concat(all_features).sort_values('reference_index').to_parquet(out/'v1.parquet',index=False)
    save_json(out/'support_audit.json',evidence)
    save_json(out/'reference.json',{'protocol':'dense127909-r1','areas':areas,'n_development':127909,
        'n_liguria':4991,'edge_policy':'historical geometry; no silent floor correction',
        'files':{str(p.relative_to(out)):digest(p) for p in sorted(out.rglob('*')) if p.is_file()}})
    return out

# Valida tutto il pacchetto e il join col manifest, prima di usare dati per il probing.
def load(root):
    root = Path(root); spec = json.loads((root/'reference.json').read_text())
    for name, sha in spec['files'].items():
        p = (root/name).resolve()
        if root.resolve() not in p.parents or digest(p) != sha: raise ValueError('Reference corrotta')
    frame = pd.read_parquet(root/'v1.parquet'); expected = membership()
    if len(frame) != len(expected) or set(frame.cell_id) != set(expected.cell_id): raise ValueError('Celle reference differenti')
    check = frame.merge(expected,on='cell_id',validate='one_to_one',suffixes=('','_original'))
    for col in ['aoi_id','partition','cv_fold']:
        if not check[col].eq(check[col+'_original']).all(): raise ValueError('Partizioni reference modificate')
    if not np.isfinite(frame[MEANS+['target']]).all().all(): raise ValueError('Feature/target non finiti')
    return spec, frame
