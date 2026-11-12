# SCOPO
#   Costruire sul PC, solo da dati pubblici, il "reference" della campagna densa 127.909 che
#   reference.pack() ricava dagli artefatti storici del Mac. Produce la stessa struttura letta da
#   reference.load(), extraction, streaming e probing: v1.parquet, <aoi>/lookup.npy, <aoi>/target.tif,
#   <aoi>/grid_*/geometry.npz|json, prove per AOI e reference.json con SHA256 di ogni file.
#
# COME SI RICHIAMA
#   python scripts\dense_campaign.py build-reference --reference <dir_nuova> --work-root <dir_lavoro>
#     --v1-root <tile V1 di scripts/download_v1_tiles.py> --msa-area <msa_area_10sec.tif>
#     --cellarea <cellarea_km2_10sec.tif> [--liguria-bundle <pacchetto denso ligure>]
#
# LOGICA, PER OGNI AOI DEL MANIFEST configs/msa_v1_liguria_cells.csv (hash verificato da membership())
#   1. Target: MSA-area / cell-area GLOBIO 2020 in float32, ritagliato al box AOI del manifest. Sulla
#      Liguria coincide al bit con il target storico (verificato): stessa griglia nativa 10".
#   2. V1: dense/v1_support.aggregate_v1, cioè il codice storico invariato: centri pixel della tile UTM,
#      CRS tile -> EPSG:4326, assegnazione a riga/colonna GLOBIO, int8*scale, scale non finite escluse,
#      somme float64 / n_valid, copertura >= 0,8. Nessun campionamento e nessun cambio di risoluzione.
#   3. Bordo: si prova prima 'trunc' (protocollo storico), poi 'floor'. Si accetta la regola il cui
#      insieme di celle eleggibili contiene tutte le celle del manifest; la scelta è registrata per AOI.
#   4. Geometria V2: per ogni tile, i pixel a scala finita assegnati (stessa funzione, stessa regola di
#      bordo) a una cella del manifest. Ordine riga-maggiore. Il conteggio per cella deve essere
#      identico a n_valid V1, altrimenti il comando si ferma: V2 sarà calcolato sugli stessi pixel.
#   5. Le duplicazioni tra tile adiacenti restano come nel V1 storico (stessa molteplicità per V1 e V2).
#
# RIPRESA
#   Ogni AOI completata scrive <aoi>/area.json per ultimo; un rilancio riusa le AOI complete (con hash)
#   e rifà solo le altre. reference.json è scritto alla fine. Nessun file sorgente viene modificato.
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer
from rasterio.transform import Affine, array_bounds
from rasterio.warp import transform_bounds
from rasterio.windows import Window

from biomap_v2.common import digest, save_json
from biomap_v2.dense.grid import load_target, assign
from biomap_v2.dense.v1_support import aggregate_v1, discover_tiles, tile_centroids
from .state import membership, MEANS, writable

PROTOCOL = 'dense127909-public-r1'
RESOLUTION = 1 / 360


# Numeratore e denominatore devono essere co-registrati; nodata -> -999 come nel target storico.
def crop_target(msa_area, cellarea, box, destination):
    with rasterio.open(msa_area) as a, rasterio.open(cellarea) as c:
        for ds in (a, c):
            if ds.crs != rasterio.crs.CRS.from_epsg(4326) or not np.allclose(ds.res, [RESOLUTION] * 2, rtol=0, atol=1e-10):
                raise ValueError(f'Target GLOBIO non nativo 10" EPSG:4326: {ds.name}')
        if a.shape != c.shape or not np.allclose(tuple(a.transform), tuple(c.transform), rtol=0, atol=1e-10):
            raise ValueError('MSA-area e cell-area non co-registrati')
        col0 = round((box[0] - a.transform.c) / RESOLUTION)
        row0 = round((a.transform.f - box[3]) / RESOLUTION)
        width = round((box[2] - box[0]) / RESOLUTION)
        height = round((box[3] - box[1]) / RESOLUTION)
        if col0 < 0 or row0 < 0 or col0 + width > a.width or row0 + height > a.height:
            raise ValueError(f'Box AOI fuori dal raster GLOBIO: {box}')
        window = Window(col0, row0, width, height)
        numerator, denominator = a.read(1, window=window), c.read(1, window=window)
        bad = ~np.isfinite(numerator) | ~np.isfinite(denominator) | (denominator <= 0)
        if a.nodata is not None:
            bad |= numerator == a.nodata
        with np.errstate(divide='ignore', invalid='ignore'):
            values = (numerator / denominator).astype(np.float32)
        values[bad] = -999.0
        t = a.window_transform(window)
        transform = Affine(t.a, 0.0, round(t.c, 10), 0.0, t.e, round(t.f, 10))
    profile = {'driver': 'GTiff', 'height': height, 'width': width, 'count': 1, 'dtype': 'float32',
               'crs': 'EPSG:4326', 'transform': transform, 'nodata': -999.0, 'compress': 'deflate'}
    with rasterio.open(destination, 'w', **profile) as out:
        out.write(values, 1)
    return {'window': [col0, row0, width, height], 'transform': list(transform)[:6]}


# Stessi passi di aggregate_v1 per una tile: cella locale di ogni pixel (-1 fuori) e scala finita.
def tile_pixels(tile, grid, edge_op):
    with rasterio.open(tile['landmask']) as landmask:
        transform, crs = landmask.transform, landmask.crs
    scales = np.load(tile['scales'], mmap_mode='r')
    embedding = np.load(tile['embeddings'], mmap_mode='r')
    height, width = embedding.shape[:2]
    if scales.shape != (height, width):
        raise ValueError(f"Tile {tile['stem']}: shape incoerente")
    x, y = tile_centroids((height, width), transform)
    lon, lat = Transformer.from_crs(crs, 'EPSG:4326', always_xy=True).transform(*np.meshgrid(x, y))
    rows, cols = assign(grid.transform, lon.ravel(), lat.ravel(), edge_op)
    rows, cols = rows.reshape(height, width), cols.reshape(height, width)
    shape = grid.shape
    in_bounds = (rows >= 0) & (rows < shape[0]) & (cols >= 0) & (cols < shape[1])
    inside = in_bounds & grid.valid[np.clip(rows, 0, shape[0] - 1), np.clip(cols, 0, shape[1] - 1)]
    cell = np.full((height, width), -1, dtype=np.int64)
    cell[inside] = grid.cell_index[rows[inside], cols[inside]]
    return cell, np.isfinite(scales), transform, crs, (height, width)


# Celle del manifest -> indice locale del target, per coordinate arrotondate e poi controllate.
def local_ids(grid, cells):
    table = dict(zip(zip(np.round(grid.lon, 7), np.round(grid.lat, 7)), range(grid.n_cells)))
    ids = np.array([table.get(k, -1) for k in zip(cells.lon.round(7), cells.lat.round(7))], dtype=np.int64)
    if (ids < 0).any():
        raise ValueError(f'{int((ids < 0).sum())} celle del manifest senza target GLOBIO valido')
    if not (np.allclose(grid.lon[ids], cells.lon, atol=1e-8, rtol=0) and np.allclose(grid.lat[ids], cells.lat, atol=1e-8, rtol=0)):
        raise ValueError('Coordinate manifest/target fuori tolleranza')
    return ids


def build_area(aoi, cells, tiles_root, target, folder, log=print):
    grid = load_target(target)
    ids = local_ids(grid, cells)
    attempts = {}
    for edge_op in ['trunc', 'floor']:
        frame = aggregate_v1(tiles_root, grid, edge_op=edge_op, min_valid_fraction=.8)
        eligible = set(frame.index.tolist())
        missing = int(sum(i not in eligible for i in ids))
        attempts[edge_op] = {'eligible': len(eligible), 'manifest': len(ids), 'manifest_missing': missing,
                             'eligible_not_in_manifest': len(eligible - set(ids.tolist()))}
        log(f'  {aoi} {edge_op}: {attempts[edge_op]}')
        if missing == 0:
            break
    else:
        raise ValueError(f'{aoi}: nessuna regola di bordo riproduce le celle del manifest: {attempts}')
    v1 = frame.loc[ids]
    wanted = np.zeros(grid.n_cells, dtype=bool)
    wanted[ids] = True
    counts = np.zeros(grid.n_cells, dtype=np.int64)
    tiles = []
    for tile in discover_tiles(tiles_root):
        cell, finite, transform, crs, (height, width) = tile_pixels(tile, grid, edge_op)
        keep = finite & (cell >= 0)
        keep[keep] = wanted[cell[keep]]
        flat = np.flatnonzero(keep.ravel()).astype(np.int64)
        if not len(flat):
            continue
        cell_ids = cell.ravel()[flat]
        counts += np.bincount(cell_ids, minlength=grid.n_cells)
        destination = folder / tile['stem']
        destination.mkdir(parents=True, exist_ok=True)
        np.savez(destination / 'geometry.npz', flat_indices=flat, cell_ids=cell_ids)
        bounds = transform_bounds(crs, 'EPSG:4326', *array_bounds(height, width, transform))
        spec = {'tile': tile['stem'], 'pixels': int(len(flat)), 'height': int(height), 'width': int(width),
                'transform': list(transform)[:6], 'crs': str(crs), 'bbox': [float(v) for v in bounds],
                'edge_op': edge_op, 'order': 'row_major_flat_index',
                'geometry_sha256': digest(destination / 'geometry.npz'), 'v1_scales_sha256': digest(tile['scales']),
                'landmask_sha256': digest(tile['landmask']), 'v1_embeddings_sha256': digest(tile['embeddings'])}
        save_json(destination / 'geometry.json', spec)
        tiles.append(spec)
    if not np.array_equal(counts[ids], v1.n_valid.to_numpy(dtype=np.int64)):
        raise ValueError(f'{aoi}: geometria V2 diversa dal supporto V1 (n_valid); nessun reference pubblicato')
    lookup = np.full(grid.n_cells, -1, dtype=np.int64)
    lookup[ids] = cells.index.to_numpy()
    np.save(folder / 'lookup.npy', lookup)
    values = {'reference_index': cells.index.to_numpy(), 'target': v1.msa_overall.to_numpy()}
    values.update({column: v1[column].to_numpy() for column in ['n_valid', 'n_total', 'valid_fraction', *MEANS]})
    record = pd.concat([cells.reset_index(drop=True), pd.DataFrame(values)], axis=1)
    record.index = cells.index
    return record, {'aoi_id': aoi, 'edge_op': edge_op, 'attempts': attempts, 'n_cells': len(ids),
                    'pixel_contributions': int(v1.n_valid.sum()), 'tiles': [t['tile'] for t in tiles],
                    'tile_pixels': {t['tile']: t['pixels'] for t in tiles}}


# Verifica opzionale contro il pacchetto ligure storico: medie, conteggi e pixel per tile.
def check_liguria(record, evidence, bundle):
    bundle = Path(bundle)
    paired = pd.read_parquet(bundle / 'paired_features.parquet')
    support = json.loads((bundle / 'support_verification.json').read_text(encoding='utf-8'))
    left = record.sort_values(['lon', 'lat']).reset_index(drop=True)
    right = paired.sort_values(['lon', 'lat']).reset_index(drop=True)
    if len(left) != len(right) or not np.allclose(left[['lon', 'lat']], right[['lon', 'lat']], atol=1e-8, rtol=0):
        raise ValueError('Liguria: celle diverse dal pacchetto storico')
    result = {'max_abs_mean': float(np.abs(left[MEANS].to_numpy() - right[['v1_' + m for m in MEANS]].to_numpy()).max()),
              'n_valid_equal': bool((left.n_valid.to_numpy() == right.n_valid.to_numpy()).all()),
              'n_total_equal': bool((left.n_total.to_numpy() == right.n_total.to_numpy()).all()),
              'max_abs_target': float(np.abs(left.target.to_numpy() - right.target.to_numpy()).max()),
              'tile_pixels_equal': {t['tile']: evidence['tile_pixels'].get(t['tile']) == t['pixels'] for t in support['tiles']}}
    if result['max_abs_mean'] != 0 or not result['n_valid_equal'] or not result['n_total_equal'] or not all(result['tile_pixels_equal'].values()):
        raise ValueError(f'Liguria non riproduce il V1 storico: {result}')
    return result


def build(args):
    out = writable(args.reference, [args.v1_root])
    if (out / 'reference.json').exists():
        raise FileExistsError('Reference già completo: usare una nuova cartella')
    rows = membership()
    features, areas, evidence = [], [], []
    started = time.perf_counter()
    for aoi, cells in rows.groupby('aoi_id', sort=True):
        if args.aoi and aoi not in args.aoi:
            continue
        folder = out / aoi
        marker = folder / 'area.json'
        if marker.exists():
            info = json.loads(marker.read_text(encoding='utf-8'))
            if digest(folder / 'v1_part.parquet') != info['v1_part_sha256']:
                raise ValueError(f'{aoi}: parte V1 corrotta, cancellare la cartella AOI e rilanciare')
            print(f'{aoi}: già completata, riuso', flush=True)
            features.append(pd.read_parquet(folder / 'v1_part.parquet'))
            areas.append({'aoi_id': aoi, 'tiles': info['evidence']['tiles']})
            evidence.append(info['evidence'])
            continue
        t = time.perf_counter()
        folder.mkdir(parents=True, exist_ok=True)
        box = (float(cells.west.min()), float(cells.south.min()), float(cells.east.max()), float(cells.north.max()))
        crop = crop_target(args.msa_area, args.cellarea, box, folder / 'target.tif')
        tiles_root = Path(args.v1_root) / aoi / 'tessera' / '2020'
        print(f'{aoi}: box {box}, {len(cells)} celle', flush=True)
        record, info = build_area(aoi, cells, tiles_root, folder / 'target.tif', folder,
                                  log=lambda m: print(m, flush=True))
        info.update(box=box, crop=crop, target_sha256=digest(folder / 'target.tif'), seconds=time.perf_counter() - t)
        download = Path(args.v1_root) / aoi / 'v1_download_manifest.json'
        if download.exists():
            meta = json.loads(download.read_text(encoding='utf-8'))
            info['v1_source'] = {k: meta[k] for k in ['geotessera_version', 'dataset_version', 'dataset_variant', 'dataset_path', 'source']}
        if aoi == 'liguria_levante' and args.liguria_bundle:
            info['liguria_historical_check'] = check_liguria(record, info, args.liguria_bundle)
        save_json(folder / 'physical_support.json', info)
        record.to_parquet(folder / 'v1_part.parquet', index=False)
        save_json(marker, {'v1_part_sha256': digest(folder / 'v1_part.parquet'), 'evidence': info})
        print(f'{aoi}: {info["n_cells"]} celle, {info["pixel_contributions"]:,} pixel, bordo {info["edge_op"]}, '
              f'{len(info["tiles"])} tile, {info["seconds"]:.0f}s', flush=True)
        features.append(record)
        areas.append({'aoi_id': aoi, 'tiles': info['tiles']})
        evidence.append(info)
    if args.aoi:
        print('Sottoinsieme AOI costruito: reference.json non scritto (serve la campagna completa)', flush=True)
        return out
    frame = pd.concat(features).sort_values('reference_index')
    if len(frame) != len(rows) or not np.isfinite(frame[MEANS + ['target']].to_numpy()).all():
        raise ValueError('Reference incompleto o non finito')
    frame.to_parquet(out / 'v1.parquet', index=False)
    save_json(out / 'support_audit.json', evidence)
    files = {str(p.relative_to(out)).replace('\\', '/'): digest(p) for p in sorted(out.rglob('*'))
             if p.is_file() and p.name not in {'reference.json'}}
    save_json(out / 'reference.json', {
        'protocol': PROTOCOL, 'areas': areas, 'n_development': 127909, 'n_liguria': 4991,
        'edge_policy': 'trunc storico se riproduce le celle del manifest, altrimenti floor; registrato per AOI',
        'v1_source': 'GeoTESSERA 1.0/vultr (npy/v1), solo per V1 e supporto pixel; V2 = Medium locale',
        'target': {'msa_area': str(args.msa_area), 'msa_area_sha256': digest(args.msa_area),
                   'cellarea': str(args.cellarea), 'cellarea_sha256': digest(args.cellarea),
                   'formula': 'float32(MSA-area / cell-area), nodata -999'},
        'seconds': time.perf_counter() - started, 'files': files})
    print(f'Reference completo: {out} ({len(frame)} celle)', flush=True)
    return out
