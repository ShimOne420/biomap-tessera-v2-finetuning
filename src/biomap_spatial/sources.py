# Scopo: adapter read-only V1, V2 per pixel r1/r2 e array sintetici; indice SQLite riprendibile.
# Fasi: reference/marker/hash -> geometria congelata -> finestre native -> momenti e indice per cella.
# Input: root V2 completata e reference, oppure piccoli array; output soltanto nella nuova output-root.
# Parametri: max_patch_side, mmap_limit, reserve_gib; prepare/doctor dalla CLI spatial_cnn.py.
# Risorse: hash sequenziale di circa 50 GB per V2 completo, memmap LRU e finestre limitate; nessuna rete/GPU.
# Ripresa: transazione per tile; indice completo firmato per ultimo. Mai ricostruire mappe da medie.
from collections import OrderedDict
import json
from pathlib import Path
import sqlite3
import time

import numpy as np
import pandas as pd

from .contracts import SpatialSample
from .support import Grid, windows, overlap_report
from .utils import digest, signature, save_json, disk_budget, separate_output, unique_tmp, replace_with_retry

R1 = 'dense127909-stream-legacy-exact-r1'
R2 = 'dense127909-stream-legacy-exact-r2-fast'


def local_path(root, name):
    root = Path(root).resolve()
    path = (root / name).resolve()
    if root not in path.parents:
        raise ValueError('Percorso fuori dalla sorgente dichiarata')
    return path


def validate_frame(frame, profile):
    required = {'cell_id', 'aoi_id', 'partition', 'cv_fold', 'target', 'reference_index', 'n_valid'}
    if not required <= set(frame) or frame.cell_id.duplicated().any() or frame.reference_index.duplicated().any():
        raise ValueError('Reference con colonne mancanti o ID duplicati')
    if not np.isfinite(frame.target).all() or not frame.target.between(0, 1).all() or (frame.n_valid <= 0).any():
        raise ValueError('Target/supporto non valido; nessuna imputazione')
    dev, test = frame[frame.partition == 'train'], frame[frame.partition == 'test']
    if len(dev) + len(test) != len(frame) or set(dev.cv_fold) != {0, 1, 2} or not test.cv_fold.eq(-1).all() or not len(test):
        raise ValueError('Servono tre fold sviluppo e test esterno separato')
    if not dev.groupby('aoi_id').cv_fold.nunique().eq(1).all() or set(dev.aoi_id) & set(test.aoi_id):
        raise ValueError('Leakage geografico AOI/fold')
    if profile == 'benchmark':
        from biomap_v2.campaign.state import membership
        expected = membership().set_index('cell_id')
        if set(frame.cell_id) != set(expected.index) or len(dev) != 127909 or len(test) != 4991 or dev.aoi_id.nunique() != 17:
            raise ValueError('Benchmark richiede 127909 sviluppo, 17 AOI e 4991 Liguria')
        actual = frame.set_index('cell_id').loc[expected.index]
        for name in ['aoi_id', 'partition', 'cv_fold', 'lon', 'lat']:
            same = np.allclose(actual[name], expected[name], rtol=0, atol=1e-8) if name in {'lon', 'lat'} else actual[name].equals(expected[name])
            if not same:
                raise ValueError('Membership/coordinate congelate modificate: ' + name)


def inventory(spec):
    source = spec.source
    if not source.get('root') or not source.get('reference'):
        return {'status': 'BLOCKED', 'reason': 'Impostare --source-root e --reference', 'integrity_verified': False}
    root, ref = Path(source['root']), Path(source['reference'])
    marker = ref / 'reference.json'
    if not marker.is_file():
        return {'status': 'BLOCKED', 'reason': 'Reference mancante', 'integrity_verified': False}
    plan = json.loads(marker.read_text())
    missing = []
    tiles = 0
    for area in plan['areas']:
        for tile in area['tiles']:
            tiles += 1
            p = root / 'pixel_embeddings' / area['aoi_id'] / (tile + '.npy')
            if not p.is_file() or not p.with_suffix('.json').is_file():
                missing.append(area['aoi_id'] + '/' + tile)
    return {'status': 'BLOCKED' if missing or not (root / 'extraction_complete.json').is_file() else 'PRESENT_NOT_VERIFIED',
        'tiles': tiles, 'missing_pixel_tiles': len(missing), 'missing_examples': missing[:20],
        'integrity_verified': False, 'note': 'Le medie non sono mappe. prepare verifica tutti gli hash prima di dichiarare READY.'}


class ArraySource:
    def __init__(self, samples, feature_space_id):
        self.samples = samples
        self.feature_space_id = feature_space_id
        channels = {p.features.shape[0] for patches in samples.values() for p in patches}
        if len(channels) != 1 or any(p.metadata['feature_space_id'] != feature_space_id for v in samples.values() for p in v):
            raise ValueError('Canali/feature space incompatibili')
        self.channels = channels.pop()
        self.fingerprint = signature({'feature_space_id': feature_space_id, 'cells': sorted(samples),
            'arrays': [signature({'x': p.features.tolist(), 'mask': p.mask.tolist()}) for k in sorted(samples) for p in samples[k]]})

    def patches(self, cell_id):
        return self.samples[cell_id]

    def close(self):
        pass


class V1TileAdapter:
    def __init__(self, embedding, scales, geotiff, feature_space_id):
        import rasterio
        self.embedding = np.load(embedding, mmap_mode='r', allow_pickle=False)
        self.scales = np.load(scales, mmap_mode='r', allow_pickle=False)
        with rasterio.open(geotiff) as ds:
            self.grid = Grid(ds.height, ds.width, tuple(ds.transform), str(ds.crs))
        if self.embedding.dtype != np.int8 or self.embedding.ndim != 3 or self.scales.shape != self.embedding.shape[:2]:
            raise ValueError('V1 richiede int8 HWC e scale HW')
        if self.embedding.shape[:2] != (self.grid.height, self.grid.width):
            raise ValueError('Georeferenziazione V1 incompatibile')
        self.feature_space_id = feature_space_id

    def patch(self, row, col, mask):
        h, w = mask.shape
        scale = self.scales[row:row+h, col:col+w]
        valid = mask & np.isfinite(scale)
        raw = self.embedding[row:row+h, col:col+w].astype(np.float32)
        data = raw * np.where(valid, scale, 0)[..., None]
        return SpatialSample(data.transpose(2, 0, 1), valid,
            {'feature_space_id': self.feature_space_id, 'grid': self.grid, 'row': row, 'col': col, 'quantization': 'int8 * scale; zero_point=0'})


class V2Source:
    def __init__(self, spec, output, require_index=True):
        self.spec = spec
        cfg = spec.source
        if cfg.get('kind', 'v2_pixels') != 'v2_pixels':
            raise ValueError('Runner operativo V2 richiede kind=v2_pixels; V1TileAdapter/ArraySource sono API separate')
        if not cfg.get('root') or not cfg.get('reference'):
            raise ValueError('Specificare source.root e source.reference')
        self.root, self.reference = Path(cfg['root']).resolve(), Path(cfg['reference']).resolve()
        self.output = separate_output(output, [self.root, self.reference])
        self.channels, self.feature_space_id = spec.model.input_channels, cfg['feature_space_id']
        self.maps = OrderedDict()
        self.db = None
        self.protocol = cfg.get('protocol', R1)
        if self.protocol not in {R1, R2} or (spec.protocol.profile == 'benchmark' and self.channels != 128):
            raise ValueError('Protocollo/canali V2 Medium incompatibili')
        marker = self.root / 'extraction_complete.json'
        if not marker.is_file():
            raise ValueError('Estrazione non completa: extraction_complete.json assente')
        self.completion = json.loads(marker.read_text())
        features = self.root / 'v2.parquet'
        if self.completion.get('complete_campaign') is not True or self.completion.get('method') != self.protocol or self.completion.get('amp') is not False:
            raise ValueError('Richiesto marker completo FP32 del protocollo selezionato')
        if not features.is_file() or digest(features) != self.completion['sha256']:
            raise ValueError('Parquet finale assente/corrotto; non si procede alla GPU')
        reference_marker = self.reference / 'reference.json'
        plan = json.loads(reference_marker.read_text())
        if plan.get('protocol') not in {'dense127909-r1', 'dense127909-public-r1'}:
            raise ValueError('Protocollo reference non supportato')
        for name, sha in plan['files'].items():
            if digest(local_path(self.reference, name)) != sha:
                raise ValueError('Reference corrotta: ' + name)
        self.frame = pd.read_parquet(self.reference / 'v1.parquet')
        self.frame['cell_id'] = self.frame.cell_id.astype(str)
        validate_frame(self.frame, spec.protocol.profile)
        final_frame = pd.read_parquet(features)
        final_frame['cell_id'] = final_frame.cell_id.astype(str)
        if final_frame.cell_id.duplicated().any() or set(final_frame.cell_id) != set(self.frame.cell_id) or len(self.frame) != self.completion['n_cells']:
            raise ValueError('Celle finali diverse dal reference')
        paired = self.frame.merge(final_frame, on='cell_id', validate='one_to_one', suffixes=('_ref', '_v2'))
        for key in ['target', 'aoi_id', 'partition', 'cv_fold', 'n_valid']:
            if key not in final_frame or not np.array_equal(paired[key+'_ref'], paired[key+'_v2']):
                raise ValueError('Target/split/supporto finali discordanti: ' + key)
        self.frame = self.frame[[c for c in self.frame if not c.startswith('mean_')]].copy()
        self.ref_to_cell = dict(zip(self.frame.reference_index.astype(int), self.frame.cell_id))
        self.tiles = {}
        identities = []
        from biomap_v2.common import WEIGHTS_SHA256
        for area in plan['areas']:
            for tile_name in area['tiles']:
                key = area['aoi_id'] + '/' + tile_name
                tile = local_path(self.reference, key)
                geometry_path = tile / 'geometry.npz'
                info = json.loads((tile / 'geometry.json').read_text())
                grid = Grid(int(info['height']), int(info['width']), tuple(info['transform']), info['crs'])
                resolution = cfg.get('resolution', [10, 10])
                if not np.allclose([grid.transform[0], -grid.transform[4]], resolution, rtol=0, atol=1e-9):
                    raise ValueError('Risoluzione inattesa; nessun resampling')
                path = local_path(self.root, 'pixel_embeddings/' + key + '.npy')
                if not path.is_file() or not path.with_suffix('.json').is_file():
                    raise ValueError('V2 Spatial mancante: ' + key + '. Le medie non permettono di ricostruirlo; vedere docs/framework/windows_runbook.md')
                meta = json.loads(path.with_suffix('.json').read_text())
                geometry_sha = digest(geometry_path)
                if meta.get('dtype') != 'float32' or meta.get('order') != 'geometry.npz flat_indices' or meta.get('sha256') != digest(path):
                    raise ValueError('Embedding pixel corrotti o schema non supportato: ' + key)
                chunk = local_path(self.root, 'chunks/' + key + '/tile.npz')
                chunk_marker = json.loads(chunk.with_suffix('.json').read_text())
                identity = chunk_marker['identity']
                if chunk_marker['sha256'] != digest(chunk) or identity.get('geometry') != geometry_sha or identity.get('protocol') != self.protocol or identity.get('year') != 2020 or identity.get('checkpoint') != WEIGHTS_SHA256 or identity.get('precision') != 'fp32':
                    raise ValueError('Provenienza tile/checkpoint non valida: ' + key)
                if self.protocol == R1:
                    from .utils import ROOT
                    contract = json.loads((ROOT / 'tests/fixtures/r1_contract_1cdc762.json').read_text())
                    if identity.get('code') != {k:v['sha256_worktree'] for k,v in contract['files'].items()} or meta.get('geometry') != geometry_sha:
                        raise ValueError('Identita r1 diversa dal contratto pubblico')
                elif meta.get('identity') != identity or meta.get('geometry_sha256') != geometry_sha or meta.get('grid') != {'width':grid.width,'height':grid.height,'transform':info['transform'],'crs':info['crs']}:
                    raise ValueError('Sidecar r2 non corrispondente')
                with np.load(geometry_path, allow_pickle=False) as z:
                    flat, cell_ids = z['flat_indices'], z['cell_ids']
                if flat.ndim != 1 or len(flat) != info['pixels'] or len(flat) == 0 or len(flat) != len(cell_ids) or flat.dtype.kind not in 'iu' or (np.diff(flat) <= 0).any() or flat[0] < 0 or flat[-1] >= grid.height * grid.width:
                    raise ValueError('Ordine/indici geometria non validi')
                lookup_path = local_path(self.reference, area['aoi_id'] + '/lookup.npy')
                lookup = np.load(lookup_path, allow_pickle=False)
                if cell_ids.dtype.kind not in 'iu' or cell_ids.min() < 0 or cell_ids.max() >= len(lookup) or not set(lookup[cell_ids]) <= set(self.ref_to_cell):
                    raise ValueError('Membership pixel/celle non valida')
                emb = np.load(path, mmap_mode='r', allow_pickle=False)
                if emb.dtype != np.float32 or emb.shape != (len(flat), self.channels):
                    raise ValueError('V2 Spatial richiede float32 [N,C] coerente con la geometria')
                for low in range(0, len(emb), 16384):
                    if not np.isfinite(emb[low:low+16384]).all():
                        raise ValueError('Embedding validi NaN/Inf: nessuna imputazione')
                del emb
                self.tiles[key] = {'path':path,'geometry':geometry_path,'lookup':lookup_path,'grid':grid,
                    'stat':(path.stat().st_size,path.stat().st_mtime_ns),'sha256':meta['sha256']}
                identities.append((key, meta['sha256'], geometry_sha, signature(identity)))
        self.fingerprint = signature({'reference':digest(reference_marker), 'completion':digest(marker),
            'tiles':identities, 'feature_space_id':self.feature_space_id})
        self.index_identity = signature({'source':self.fingerprint,'side':cfg.get('max_patch_side',64),
            'support':spec.protocol.support,'adapter':digest(Path(__file__)),'support_code':digest(Path(__file__).with_name('support.py'))})
        self.index = self.output / 'spatial_index.sqlite3'
        if require_index:
            self._load_index()

    def _load_index(self):
        marker = self.index.with_suffix('.json')
        if not marker.is_file() or not self.index.is_file():
            raise ValueError('Indice assente: eseguire prepare')
        info = json.loads(marker.read_text())
        if info['identity'] != self.index_identity or info['sha256'] != digest(self.index) or info['frame_sha256'] != digest(self.output / 'cells.parquet'):
            raise ValueError('Indice/config/sorgente incompatibili: usare output-root separata')
        self.frame = pd.read_parquet(self.output / 'cells.parquet')
        self.db = sqlite3.connect(self.index.resolve().as_uri() + '?mode=ro', uri=True)

    def tile(self, key):
        spec = self.tiles[key]
        if (spec['path'].stat().st_size, spec['path'].stat().st_mtime_ns) != spec['stat']:
            raise ValueError('Sorgente modificata durante il run')
        if key not in self.maps:
            with np.load(spec['geometry'], allow_pickle=False) as z:
                flat = z['flat_indices'].copy()
            self.maps[key] = (np.load(spec['path'], mmap_mode='r', allow_pickle=False), flat)
            while len(self.maps) > self.spec.source.get('mmap_limit', 4):
                self.maps.popitem(last=False)
        self.maps.move_to_end(key)
        return self.maps[key]

    def patches(self, cell_id):
        result = []
        for key, row, col, h, w, packed in self.db.execute('SELECT tile,row0,col0,height,width,mask FROM patches WHERE cell_id=? ORDER BY id', (cell_id,)):
            mask = np.unpackbits(np.frombuffer(packed,dtype=np.uint8), count=h*w).reshape(h,w).astype(bool)
            rr, cc = np.where(mask)
            emb, flat = self.tile(key)
            positions = (rr + row) * self.tiles[key]['grid'].width + cc + col
            at = np.searchsorted(flat, positions)
            if (at >= len(flat)).any() or not np.array_equal(flat[at], positions):
                raise ValueError('Patch fuori dalla geometria verificata')
            values = np.zeros((self.channels,h,w), dtype=np.float32)
            values[:,rr,cc] = emb[at].T
            result.append(SpatialSample(values, mask, {'feature_space_id':self.feature_space_id,
                'grid':self.tiles[key]['grid'],'row':row,'col':col,'tile':key,'encoder':'geotessera_v2',
                'year':2020,'quantization':'float32','provenance':self.protocol,'sha256':self.tiles[key]['sha256']}))
        if not result:
            raise ValueError('Cella senza supporto nell indice')
        return result

    def close(self):
        if self.db is not None:
            self.db.close()
            self.db = None
        self.maps.clear()

    def __getstate__(self):
        state = dict(self.__dict__)
        state['db'], state['maps'] = None, OrderedDict()
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.db = sqlite3.connect(self.index.resolve().as_uri() + '?mode=ro', uri=True)


def prepare_index(source, progress=lambda **kw: None):
    out = source.output
    disk_budget(out, source.spec.execution.get('reserve_gib',20), 512*1024**2)
    out.mkdir(parents=True, exist_ok=True)
    if source.index.with_suffix('.json').exists():
        source._load_index()
        return source.frame
    db = sqlite3.connect(source.index)
    db.executescript('''CREATE TABLE IF NOT EXISTS meta(identity TEXT);
        CREATE TABLE IF NOT EXISTS tiles(tile TEXT PRIMARY KEY);
        CREATE TABLE IF NOT EXISTS moments(cell_id TEXT PRIMARY KEY,n INTEGER,sums BLOB,squares BLOB);
        CREATE TABLE IF NOT EXISTS patches(id INTEGER PRIMARY KEY,cell_id TEXT,tile TEXT,row0 INTEGER,col0 INTEGER,height INTEGER,width INTEGER,mask BLOB);
        CREATE INDEX IF NOT EXISTS patches_cell ON patches(cell_id);''')
    old = db.execute('SELECT identity FROM meta').fetchone()
    if old and old[0] != source.index_identity:
        db.close()
        raise ValueError('Indice parziale incompatibile: nuova output-root necessaria')
    if not old:
        with db:
            db.execute('INSERT INTO meta VALUES(?)', (source.index_identity,))
    complete = {r[0] for r in db.execute('SELECT tile FROM tiles')}
    began, initial = time.perf_counter(), len(complete)
    try:
        for key, tile in source.tiles.items():
            if key in complete:
                continue
            disk_budget(out, source.spec.execution.get('reserve_gib',20), 32*1024**2)
            emb, flat = source.tile(key)
            with np.load(tile['geometry'], allow_pickle=False) as z:
                ids = np.load(tile['lookup'],allow_pickle=False)[z['cell_ids']]
            order = np.argsort(ids, kind='stable')
            sorted_ids = ids[order]
            starts = np.r_[0,np.flatnonzero(np.diff(sorted_ids))+1,len(ids)]
            with db:
                computed_cells, computed_sums, computed_counts = [], [], []
                for low, high in zip(starts[:-1],starts[1:]):
                    cell = source.ref_to_cell[int(sorted_ids[low])]
                    take = order[low:high]
                    rr, cc = flat[take] // tile['grid'].width, flat[take] % tile['grid'].width
                    row, col = int(rr.min()), int(cc.min())
                    mask = np.zeros((int(rr.max())-row+1,int(cc.max())-col+1),dtype=bool)
                    mask[rr-row,cc-col] = True
                    for dr, dc, part in windows(mask,source.spec.source.get('max_patch_side',64)):
                        h,w = part.shape
                        db.execute('INSERT INTO patches(cell_id,tile,row0,col0,height,width,mask) VALUES(?,?,?,?,?,?,?)',
                            (cell,key,row+dr,col+dc,h,w,np.packbits(part).tobytes()))
                    values = emb[take].astype(np.float64)
                    sums, squares = values.sum(0), (values**2).sum(0)
                    computed_cells.append(int(sorted_ids[low]))
                    computed_sums.append(sums.copy())
                    computed_counts.append(len(take))
                    saved = db.execute('SELECT n,sums,squares FROM moments WHERE cell_id=?',(cell,)).fetchone()
                    n = len(take)
                    if saved:
                        n += saved[0]
                        sums += np.frombuffer(saved[1],dtype=np.float64)
                        squares += np.frombuffer(saved[2],dtype=np.float64)
                    db.execute('INSERT OR REPLACE INTO moments VALUES(?,?,?,?)',(cell,n,sums.tobytes(),squares.tobytes()))
                with np.load(source.root / 'chunks' / key / 'tile.npz', allow_pickle=False) as original:
                    if (not np.array_equal(original['cell'],computed_cells)
                        or not np.array_equal(original['counts'],computed_counts)
                        or not np.allclose(original['sums'],computed_sums,rtol=1e-12,atol=1e-10)):
                        raise ValueError('Pixel V2 non coerenti con somme/conteggi della tile completata')
                db.execute('INSERT INTO tiles VALUES(?)',(key,))
            complete.add(key)
            elapsed=time.perf_counter()-began
            rate=(len(complete)-initial)/elapsed if elapsed else 0
            progress(phase='prepare',done=len(complete),total=len(source.tiles),unit='tiles',
                rate_per_second=rate,eta_seconds=(len(source.tiles)-len(complete))/rate if rate else None,current=key)
        rows=[]
        for record in source.frame.to_dict('records'):
            cell=record['cell_id']
            saved=db.execute('SELECT n,sums,squares FROM moments WHERE cell_id=?',(cell,)).fetchone()
            if not saved or saved[0]!=record['n_valid']:
                raise ValueError('Conteggi del supporto congelato discordanti: '+cell)
            mean=np.frombuffer(saved[1],dtype=np.float64)/saved[0]
            variance=np.maximum(np.frombuffer(saved[2],dtype=np.float64)/saved[0]-mean**2,0)
            parts=[]
            for key,row,col,h,w,packed in db.execute('SELECT tile,row0,col0,height,width,mask FROM patches WHERE cell_id=?',(cell,)):
                mask=np.unpackbits(np.frombuffer(packed,dtype=np.uint8),count=h*w).reshape(h,w).astype(bool)
                parts.append((source.tiles[key]['grid'],row,col,mask))
            overlap=overlap_report(parts)
            record={k:v for k,v in record.items() if not k.startswith('mean_')}
            record.update(mean=mean.astype(np.float32),variance=variance.astype(np.float32),patches=len(parts),
                heterogeneity=float(variance.mean()),duplicate_contributions=overlap['duplicate_contributions'])
            record.setdefault('valid_fraction',1.)
            rows.append(record)
        frame=pd.DataFrame(rows)
        tmp=unique_tmp(out/'cells.parquet')
        frame.to_parquet(tmp,index=False)
        replace_with_retry(tmp,out/'cells.parquet')
    finally:
        db.close()
    save_json(source.index.with_suffix('.json'),{'identity':source.index_identity,'sha256':digest(source.index),
        'frame_sha256':digest(out/'cells.parquet'),'source':source.fingerprint,'complete':True})
    source._load_index()
    return source.frame
