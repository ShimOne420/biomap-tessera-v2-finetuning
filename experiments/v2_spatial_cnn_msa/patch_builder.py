# SCOPO
#   Creare l'indice spaziale dell'esperimento V2 e le medie per i baseline dagli stessi pixel, senza duplicare
#   le mappe. Finestre, centri nativi e maschere sono quelli dell'Esperimento B (fragments() è invariata, la
#   maschera è la validità della scala V1); i valori sono gli embedding V2 per pixel di r2-fast.
# COME VIENE RICHIAMATO
#   python patch_builder.py build --work-root <...>\v2_spatial_msa --v1-root <v1_tiles> --v2-root <r2 work-root>
#          --reference <reference> --preset full
# LOGICA IN ORDINE
#   1. target (reference/CSV) e inventario V1 (griglie e maschere), inventario V2 (identità, griglia, righe);
#   2. per tile, in una transazione SQLite: fragments() -> per ogni patch le righe V2 dei pixel validi
#      (ordine row-major della maschera) -> somme/quadrati per cella -> patch con maschera e righe;
#   3. finalizzazione: media/varianza per cella, eleggibilità (min_valid_fraction), report e manifest SHA256.
# INPUT E OUTPUT
#   Input: membership congelata, tile V1 (scale+TIFF), pixel_embeddings V2, geometrie, target GLOBIO.
#   Output: <work-root>/cache/<preset>/spatial.sqlite3 + manifest, coverage.csv, cache_report.json.
# PARAMETRI
#   targets|build|estimate, --preset, --v1-root, --v2-root, --reference/--targets-csv, --set.
# RISORSE
#   CPU e memmap; nessun encoder/GPU/rete. Ogni file V2 viene verificato con SHA256 una volta.
# RIPRESA
#   Transazione per tile, identità/hash, lock OS; rilanciare lo stesso comando. Nessuna scrittura in V1/V2.
import json
import sqlite3
from pathlib import Path
import time
import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer
from runtime import common_parser,configuration,membership,cache_path,data_identity,save_json,signature,Budget,exclusive,tracked
from sources import inventory,target_values,fetch_targets
from v2_sources import v2_spec,inspect_v2,row_lookup


def cell_codes(frame):
    col=np.floor((frame.lon.to_numpy()+180)*360).astype(np.int64)
    row=np.floor((90-frame.lat.to_numpy())*360).astype(np.int64)
    code=row*129600+col
    if len(np.unique(code))!=len(code):raise ValueError('Celle geograficamente sovrapposte nello stesso AOI')
    return code


def map_centers(lon,lat,sorted_codes,order):
    code=np.floor((90-np.asarray(lat))*360).astype(np.int64)*129600+np.floor((np.asarray(lon)+180)*360).astype(np.int64)
    at=np.searchsorted(sorted_codes,code);safe=np.minimum(at,len(sorted_codes)-1)
    return np.where((at<len(sorted_codes))&(sorted_codes[safe]==code),order[safe],-1)


def fragments(tile,frame,side):
    """Yield cell index, window, valid mask and total centers, in original native geometry."""
    codes=cell_codes(frame);order=np.argsort(codes);codes=codes[order]
    scale=np.load(tile['scales'],mmap_mode='r',allow_pickle=False)
    affine=rasterio.Affine(*tile['transform']);project=Transformer.from_crs(tile['crs'],4326,always_xy=True)
    hits={}
    for first in range(0,tile['height'],256):
        rr,cc=np.meshgrid(np.arange(first,min(first+256,tile['height'])),np.arange(tile['width']),indexing='ij')
        x,y=affine*(cc+.5,rr+.5);lon,lat=project.transform(x,y)
        assigned=map_centers(lon,lat,codes,order)
        for index in np.unique(assigned[assigned>=0]):
            chosen=assigned==index;rows=rr[chosen].astype(np.int32);cols=cc[chosen].astype(np.int32)
            hits.setdefault(int(index),[]).append((rows,cols))
    for index,pieces in hits.items():
        rr=np.concatenate([p[0] for p in pieces]);cc=np.concatenate([p[1] for p in pieces])
        r0,r1=int(rr.min()),int(rr.max())+1;c0,c1=int(cc.min()),int(cc.max())+1
        for row in range(r0,r1,side):
            for col in range(c0,c1,side):
                height=min(side,r1-row);width=min(side,c1-col)
                chosen=(rr>=row)&(rr<row+height)&(cc>=col)&(cc<col+width)
                total=int(chosen.sum())
                if not total:continue
                r=rr[chosen];c=cc[chosen];mask=np.zeros((height,width),dtype=bool)
                mask[r-row,c-col]=np.isfinite(scale[r,c])
                yield index,(row,col,height,width),mask,total


def initialise(db,identity,frame,proof):
    db.executescript('''
    CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY,value TEXT NOT NULL);
    CREATE TABLE IF NOT EXISTS tiles(tile_id TEXT PRIMARY KEY,spec TEXT NOT NULL,complete INTEGER NOT NULL);
    CREATE TABLE IF NOT EXISTS cells(cell_id TEXT PRIMARY KEY,aoi_id TEXT,partition TEXT,cv_fold INTEGER,lon REAL,lat REAL,target REAL,n_total INTEGER,n_valid INTEGER,sums BLOB,squares BLOB,mean BLOB,variance BLOB,valid_fraction REAL,eligible INTEGER);
    CREATE TABLE IF NOT EXISTS patches(patch_id INTEGER PRIMARY KEY,cell_id TEXT,tile_id TEXT,row0 INTEGER,col0 INTEGER,height INTEGER,width INTEGER,mask BLOB,n_valid INTEGER,rows BLOB);
    CREATE INDEX IF NOT EXISTS patches_cell ON patches(cell_id);
    ''')
    prior=db.execute("SELECT value FROM meta WHERE key='identity'").fetchone()
    if prior:
        if json.loads(prior[0])!=identity:raise ValueError('Cache di un esperimento diverso: nuova work-root necessaria')
        return
    zero=np.zeros(128,dtype=np.float64).tobytes()
    with db:
        for r in frame.itertuples():db.execute('INSERT INTO cells VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)',(r.cell_id,r.aoi_id,r.partition,int(r.cv_fold),r.lon,r.lat,float(r.target),0,0,zero,zero,None,None,0.,0))
        for key,value in [('identity',identity),('target_provenance',proof),('finalized',False)]:db.execute('INSERT INTO meta VALUES(?,?)',(key,json.dumps(value)))


def attach_v2(cfg,tiles,deep=False):
    # Ogni tile V1 riceve la sua sorgente V2; una tile senza file V2 resta utilizzabile solo se non porta pixel.
    for tile in tiles:
        spec=v2_spec(cfg,tile['aoi_id'],tile['name'])
        tile['v2']=inspect_v2(spec,tile,deep) if Path(spec['embedding']).is_file() else None
    return tiles


def build_identity(cfg,frame,proof,tiles):
    return signature({'data':data_identity(cfg,frame),'targets':proof,'tiles':[(t['aoi_id'],t['name'],t['fingerprint'],t['v2']['fingerprint'] if t.get('v2') else None) for t in tiles]})


def build(cfg,frame,progress):
    frame,proof=target_values(cfg,frame)
    tiles,missing,bad=inventory(cfg,frame,deep=False)
    if missing or bad:raise ValueError(f'Tile mancanti/non conformi: {len(missing)}/{len(bad)}; eseguire preflight/fetch')
    tiles=attach_v2(cfg,tiles)
    path=cache_path(cfg);path.parent.mkdir(parents=True,exist_ok=True);db=sqlite3.connect(path)
    identity=build_identity(cfg,frame,proof,tiles)
    try:initialise(db,identity,frame,proof)
    except BaseException:db.close();raise
    prior=db.execute("SELECT value FROM meta WHERE key='finalized'").fetchone()
    if prior and json.loads(prior[0]):
        excluded=db.execute('SELECT count(*) FROM cells WHERE eligible=0').fetchone()[0]
        db.close()
        if excluded and not cfg['data']['allow_partial_cells']:raise ValueError('Cache parziale non autorizzata in questa configurazione')
        from runtime import digest
        marker=path.with_suffix('.manifest.json')
        if not marker.exists():save_json(marker,{'identity':identity,'sha256':digest(path)})
        from dataset import metadata
        metadata(path)
        return path
    progress.update(total=len(tiles),unit='tiles');budget=Budget(cfg)
    completed={row[0] for row in db.execute('SELECT tile_id FROM tiles WHERE complete=1')}
    progress.update(done=len(completed));progress.reset_rate(len(completed))
    try:
        for index,tile in enumerate(tiles):
            tile_id=tile['aoi_id']+'/'+tile['name'];progress.update(tile=tile_id)
            found=db.execute('SELECT spec FROM tiles WHERE tile_id=?',(tile_id,)).fetchone()
            if found:
                old=json.loads(found[0])
                if old['fingerprint']!=tile['fingerprint']:raise ValueError('Tile sorgente modificata: '+tile_id)
                if (old.get('v2') or {}).get('fingerprint')!=(tile['v2'] or {}).get('fingerprint'):raise ValueError('Embedding V2 modificati: '+tile_id)
                continue
            budget.check(32*1024**2)
            # Hash original bytes once per committed tile; stat identity additionally catches changes on resume.
            from runtime import digest
            # V1: solo scale e landmask definiscono griglia/maschera; i valori V1 non sono letti.
            tile['sha256']={k:digest(tile[k]) for k in ['scales','landmask']}
            local=frame[frame.aoi_id==tile['aoi_id']].reset_index(drop=True)
            if tile['v2'] is not None:
                tile['v2']=inspect_v2(tile['v2'],tile,deep=True)
                embedding=np.load(tile['v2']['embedding'],mmap_mode='r',allow_pickle=False);lookup=row_lookup(tile['v2'],tile['height'],tile['width'])
            else:embedding=lookup=None
            db.execute('BEGIN IMMEDIATE')
            try:
                db.execute('INSERT INTO tiles VALUES(?,?,0)',(tile_id,json.dumps(tile)))
                for ci,(row,col,height,width),mask,total in fragments(tile,local,cfg['data']['max_patch_side']):
                    cell=local.iloc[ci];valid=int(mask.sum())
                    current=db.execute('SELECT n_total,n_valid,sums,squares FROM cells WHERE cell_id=?',(cell.cell_id,)).fetchone()
                    sums=np.frombuffer(current[2],dtype=np.float64).copy();squares=np.frombuffer(current[3],dtype=np.float64).copy()
                    if valid:
                        if embedding is None:raise ValueError(f'Pixel validi senza embedding V2: {tile_id}')
                        r,c=np.nonzero(mask);rows=lookup[(row+r)*tile['width']+(col+c)]
                        if (rows<0).any():raise ValueError(f'{int((rows<0).sum())} pixel del supporto non codificati in V2: {tile_id}')
                        values=np.asarray(embedding[rows],dtype=np.float32)
                        if not np.isfinite(values).all():raise ValueError('Embedding V2 non finiti')
                        sums+=values.sum(axis=0,dtype=np.float64);squares+=np.square(values.astype(np.float64)).sum(axis=0)
                        db.execute('INSERT INTO patches(cell_id,tile_id,row0,col0,height,width,mask,n_valid,rows) VALUES(?,?,?,?,?,?,?,?,?)',
                            (cell.cell_id,tile_id,row,col,height,width,np.packbits(mask.ravel()).tobytes(),valid,rows.astype(np.int32).tobytes()))
                    db.execute('UPDATE cells SET n_total=?,n_valid=?,sums=?,squares=? WHERE cell_id=?',(current[0]+total,current[1]+valid,sums.tobytes(),squares.tobytes(),cell.cell_id))
                db.execute('UPDATE tiles SET complete=1 WHERE tile_id=?',(tile_id,));db.commit()
            except BaseException:db.rollback();raise
            finally:embedding=None
            progress.update(done=db.execute('SELECT count(*) FROM tiles WHERE complete=1').fetchone()[0])
        # Finalize shared support once; all regressors consume these exact eligible IDs.
        records=[]
        progress.update(phase='build_finalize',done=0,total=len(frame),unit='cells');progress.reset_rate()
        with db:
            for row in db.execute('SELECT cell_id,aoi_id,partition,cv_fold,n_total,n_valid,sums,squares FROM cells').fetchall():
                cid,aoi,part,fold,total,valid,sums,squares=row
                fraction=valid/total if total else 0.;eligible=valid>0 and fraction>=cfg['data']['min_valid_fraction']
                mean=np.frombuffer(sums,dtype=np.float64)/max(valid,1)
                variance=np.maximum(np.frombuffer(squares,dtype=np.float64)/max(valid,1)-mean**2,0)
                db.execute('UPDATE cells SET mean=?,variance=?,valid_fraction=?,eligible=? WHERE cell_id=?',(mean.astype(np.float32).tobytes(),variance.astype(np.float32).tobytes(),fraction,int(eligible),cid))
                records.append({'cell_id':cid,'aoi_id':aoi,'partition':part,'cv_fold':fold,'n_total':total,'n_valid':valid,'valid_fraction':fraction,'eligible':eligible,
                    'patches':db.execute('SELECT count(*) FROM patches WHERE cell_id=?',(cid,)).fetchone()[0]})
                if len(records)%512==0:progress.update(done=len(records))
        progress.update(done=len(records))
        report=pd.DataFrame(records);report.to_csv(path.parent/'coverage.csv',index=False)
        if {'n_valid','n_total'}<=set(frame):
            comparison=report[['cell_id','n_valid','n_total']].merge(frame[['cell_id','n_valid','n_total']],on='cell_id',suffixes=('_spatial','_historical'))
            comparison['same_n_valid']=comparison.n_valid_spatial==comparison.n_valid_historical
            comparison['same_n_total']=comparison.n_total_spatial==comparison.n_total_historical
            comparison.to_csv(path.parent/'historical_support_comparison.csv',index=False)
            save_json(path.parent/'historical_support_summary.json',{'n':len(comparison),'count_matches':int(comparison.same_n_valid.sum()),'definition':'strict native cell support vs historical crop/trunc and multiplicity; baselines use spatial support regardless'})
        excluded=int((~report.eligible).sum())
        if excluded and not cfg['data']['allow_partial_cells']:raise ValueError(f'{excluded} celle non eleggibili: vedere coverage.csv; nessuna riduzione silenziosa. Override esplicito data.allow_partial_cells=true per confronto su sottoinsieme dichiarato.')
        with db:
            db.execute("INSERT OR REPLACE INTO meta VALUES('finalized','true')")
            db.execute("INSERT OR REPLACE INTO meta VALUES('config',?)",(json.dumps(cfg),))
            db.execute("INSERT OR REPLACE INTO meta VALUES('sources',?)",(json.dumps(tiles),))
        report_fields={'cells':len(report),'eligible':int(report.eligible.sum()),'excluded':excluded,'patches':int(report.patches.sum()),'pixels':int(report.n_valid.sum()),'shape':'variable native H/W','protocol':cfg['protocol']}
        save_json(path.parent/'cache_report.json',report_fields)
        return path
    finally:
        finalized=db.execute("SELECT value FROM meta WHERE key='finalized'").fetchone()
        db.close()
        if finalized and json.loads(finalized[0]):
            from runtime import digest
            marker=path.with_suffix('.manifest.json')
            value={'identity':identity,'sha256':digest(path)}
            if marker.exists() and json.loads(marker.read_text())!=value:raise ValueError('Indice spaziale alterato dopo la finalizzazione')
            save_json(marker,value)


def estimate(cfg,frame):
    # Geometry estimate only: no time/GPU promise before the local training/extraction pilot.
    geodetic_pixels=(111320/360/10)**2*np.cos(np.deg2rad(frame.lat.to_numpy()))
    pixels=float(geodetic_pixels.sum())
    names=__import__('sources').expected_tiles(frame)
    source_bytes=sum((11132/10)**2*np.cos(np.deg2rad(float(stem.split('_')[2])))*132 for group in names.values() for stem in group)
    return {'cells':len(frame),'development':int((frame.partition=='train').sum()),'liguria':int((frame.partition=='test').sum()),
        'estimated_pixels':round(pixels),'estimated_float32_duplicate_gib':pixels*128*4/1024**3,
        'nominal_patch_planning_assumption_per_cell':1.0,'nominal_total_patches_lower_estimate':len(frame),
        'estimated_index_gib_at_4kib_per_cell':len(frame)*4096/1024**3,
        'chosen_storage':'index-only SQLite plus original int8/scale memmaps; no float32 map duplication',
        'initial_patches_per_cell':'one per intersecting tile, split only if native window exceeds 64; measured in coverage.csv',
        'gpu_peak_gib':None,'training_seconds_per_epoch':None,'extraction_seconds':None,
        'timing_note':'measure train.py --profile and build throughput on this PC; estimates do not certify full coverage',
        'expected_tiles':sum(len(x) for x in names.values()),'estimated_source_int8_plus_scales_gib':source_bytes/1024**3,
        'download_note':'tile-level footprint, not proportional to sampled cells; pilot spread across 17 AOIs can require most full tiles; existing sources are reused'}


def main():
    p=common_parser('Indice spaziale V2 (pixel r2-fast su patch/maschere V1): nessuna esecuzione automatica')
    p.add_argument('action',choices=['targets','build','estimate'])
    args=p.parse_args();cfg=configuration(args);frame=membership(cfg)
    if args.action=='estimate':
        value=estimate(cfg,frame);save_json(Path(cfg['work_root'])/'estimates'/f"{cfg['preset']}.json",value);print(json.dumps(value,indent=2));return
    with exclusive(cfg),tracked(cfg,args.action) as progress:
        if args.action=='targets':print(fetch_targets(cfg,membership({**cfg,'preset':'full'})))
        else:print(build(cfg,frame,progress))


if __name__=='__main__':main()
