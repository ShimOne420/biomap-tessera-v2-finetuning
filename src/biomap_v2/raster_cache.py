# SCOPO
#   Legge i valori dei raster Sentinel nei punti delle celle, una finestra nativa alla volta e senza
#   interpolare, restituendo gli stessi float e maschere di rasterio.sample.
#
# COME VIENE RICHIAMATO
#   Non si esegue da solo. preparation.BatchAcquirer.read_bands -> data.Reader.values ->
#   NativeReader.values, una volta per ogni banda di ogni scena. Il Reader crea BlockStore e NativeReader
#   quando acquisition.engine=optimized.
#
# LOGICA, IN ORDINE
#   1. key_for calcola l'identità dell'asset senza firma SAS, quindi una firma scaduta non la cambia.
#   2. values cerca i metadati; se mancano, _remote apre il raster e li ricava.
#   3. _plan converte i punti in riga/colonna e li raggruppa per finestra. _project riusa la conversione
#      PROJ di ogni cella tra scene con lo stesso CRS: è la stessa chiamata, quindi lo stesso risultato.
#   4. Per ogni finestra si cerca il blocco in BlockStore; quelli assenti si leggono in _remote.
#   5. _assign copia nei risultati i valori dei punti e applica la maschera.
#   6. Se una lettura fallisce, _remote rinnova il token e riprova con attesa crescente e casuale, senza
#      rileggere le finestre già assegnate (dizionario fetched).
#
# CACHE (BlockStore)
#   block_cache_gib=0: disco disattivato. Un indice esistente non viene aperto, ridotto o cancellato.
#   block_cache_gib>0: cache LRU persistente in observation_cache/native_v1, con indice SQLite e checksum.
#   memory_cache_mib=0: niente cache in RAM. Con un lotto per area ogni blocco serve una sola volta.
#
# INPUT E OUTPUT
#   Input: item STAC, nome banda, punti WGS84. Output: array float64 con NaN per i valori mascherati.
#
# PARAMETRI (sezione acquisition del YAML)
#   memory_cache_mib, block_cache_gib, read_attempts, connect_timeout_seconds, read_timeout_seconds.
#   Costanti: MAX_PROJECTED_CELLS e MAX_PLANNED_POINTS limitano la memoria dei calcoli di coordinate.
#
# RISORSE
#   Rete verso Planetary Computer; ogni dataset GDAL resta nel thread che lo ha aperto.
#
# RIPRESA
#   Con la cache su disco: NPZ atomici e checksum, un blocco corrotto ferma il comando. Si eliminano solo
#   blocchi temporanei di questa cache, mai serie di celle.
from collections import OrderedDict
import hashlib
import json
import logging
import random
import re
from pathlib import Path
import sqlite3
import threading
import time
import numpy as np
import rasterio
from rasterio.enums import MaskFlags
from rasterio.transform import Affine,rowcol
from rasterio.warp import transform as transform_coords
from rasterio.windows import Window
import requests
from .common import digest

MAX_PROJECTED_CELLS=40000
MAX_PLANNED_POINTS=2000000


class SafeRasterMessages(logging.Filter):
    def filter(self,record):
        message=record.getMessage()
        record.msg=re.sub(r"https?://[^\s<>\"']+",lambda match:match.group(0).split('?')[0]+'?<redacted>' if '?' in match.group(0) else match.group(0),message)
        record.args=()
        return True


def install_safe_logging():
    for name in ['rasterio._err','rasterio._io','rasterio._env','rasterio.env']:
        logger=logging.getLogger(name)
        if not any(isinstance(value,SafeRasterMessages) for value in logger.filters):logger.addFilter(SafeRasterMessages())


def key_for(item,band):
    asset=dict(item['assets'][band]);asset['href']=asset['href'].split('?')[0]
    identity=[item['id'],item.get('collection'),item['properties'].get('datetime') if 'properties' in item else None,
              item.get('properties',{}).get('updated'),band,asset]
    return hashlib.sha256(json.dumps(identity,sort_keys=True,separators=(',',':')).encode()).hexdigest()


class BlockStore:
    def __init__(self,directory,options,profile,guard):
        self.directory=Path(directory)
        self.profile=profile;self.guard=guard;self.lock=threading.RLock();self.memory_lock=threading.RLock()
        self.memory=OrderedDict();self.memory_bytes=0;self.meta=OrderedDict()
        self.memory_limit=int(options['memory_cache_mib']*1024**2);self.disk_limit=int(options['block_cache_gib']*1024**3)
        # block_cache_gib=0 leaves an existing cache untouched: never opened, trimmed or deleted.
        self.db=None;self.disk_bytes=0
        if self.disk_limit==0:return
        self.directory.mkdir(parents=True,exist_ok=True)
        self.database=self.directory/'index.sqlite3';self.last_db_size=self._db_size()
        self.db=sqlite3.connect(self.database,check_same_thread=False,timeout=10)
        self.db.execute('PRAGMA journal_mode=WAL');self.db.execute('PRAGMA busy_timeout=10000')
        self.db.execute('CREATE TABLE IF NOT EXISTS assets (key TEXT PRIMARY KEY, metadata TEXT NOT NULL)')
        self.db.execute('CREATE TABLE IF NOT EXISTS blocks (key TEXT PRIMARY KEY, size INTEGER NOT NULL, sha TEXT NOT NULL, touched REAL NOT NULL)')
        self.db.execute('CREATE INDEX IF NOT EXISTS blocks_lru ON blocks(touched)');self.db.commit()
        self.disk_bytes=int(self.db.execute('SELECT COALESCE(SUM(size),0) FROM blocks').fetchone()[0]);self._account_db()
        self._trim(0)

    def _db_size(self):
        return sum(p.stat().st_size for p in [self.database,Path(str(self.database)+'-wal'),Path(str(self.database)+'-shm')] if p.exists())

    def _account_db(self):
        size=self._db_size();self.guard.account(size-self.last_db_size);self.last_db_size=size

    def _path(self,key):return self.directory/'blocks'/key[:2]/(key+'.npz')

    def _trim(self,additional):
        if self.db is None:return
        with self.lock:
            while self.disk_bytes+additional>self.disk_limit:
                row=self.db.execute('SELECT key,size FROM blocks ORDER BY touched LIMIT 1').fetchone()
                if row is None:break
                self.guard.remove_owned(self._path(row[0]));self.db.execute('DELETE FROM blocks WHERE key=?',(row[0],))
                self.disk_bytes-=row[1];self.profile.count('block_evictions')
            self.db.commit();self._account_db()

    def metadata(self,key):
        with self.lock:
            if key in self.meta:
                value=self.meta.pop(key);self.meta[key]=value;return value
            if self.db is None:return None
            row=self.db.execute('SELECT metadata FROM assets WHERE key=?',(key,)).fetchone()
            if row is None:return None
            value=json.loads(row[0]);self._remember_meta(key,value);return value

    def _remember_meta(self,key,value):
        self.meta[key]=value
        while len(self.meta)>4096:self.meta.popitem(last=False)

    def put_metadata(self,key,value):
        with self.lock:
            if self.db is None:
                self._remember_meta(key,value);return
            self.guard.check(65536)
            self.db.execute('INSERT OR REPLACE INTO assets VALUES (?,?)',(key,json.dumps(value,sort_keys=True)))
            self.db.commit();self._account_db();self._remember_meta(key,value)

    def _remember(self,key,data,mask):
        size=data.nbytes+mask.nbytes
        data.setflags(write=False);mask.setflags(write=False)
        if size>self.memory_limit:return
        with self.memory_lock:
            if key in self.memory:
                old=self.memory.pop(key);self.memory_bytes-=old[0].nbytes+old[1].nbytes
            while self.memory and self.memory_bytes+size>self.memory_limit:
                _,old=self.memory.popitem(last=False);self.memory_bytes-=old[0].nbytes+old[1].nbytes
            self.memory[key]=(data,mask);self.memory_bytes+=size

    def get(self,key,asset):
        with self.memory_lock:
            if key in self.memory:
                result=self.memory.pop(key);self.memory[key]=result
                self.profile.count('block_memory_hits',asset=asset);return result
        if self.db is None:
            self.profile.count('block_misses',asset=asset);return None
        with self.lock:
            row=self.db.execute('SELECT sha FROM blocks WHERE key=?',(key,)).fetchone()
            if row is None:self.profile.count('block_misses',asset=asset);return None
            path=self._path(key)
            with self.profile.span('block_cache_read',asset):
                if not path.exists() or digest(path)!=row[0]:
                    raise ValueError(f'Blocco cache corrotto: {path}; nessuna serie completata modificata')
                with np.load(path,allow_pickle=False) as arrays:data=arrays['data'];mask=arrays['mask']
                if data.shape!=mask.shape:raise ValueError(f'Blocco cache non valido: {path}')
            self.db.execute('UPDATE blocks SET touched=? WHERE key=?',(time.time(),key));self.db.commit();self._account_db()
            self._remember(key,data,mask);self.profile.count('block_disk_hits',asset=asset)
            return data,mask

    def put(self,key,data,mask,asset):
        if self.db is not None and data.nbytes+mask.nbytes+8192<=self.disk_limit:
            with self.lock:
                estimate=int((data.nbytes+mask.nbytes)*1.01)+8192;self._trim(estimate)
                path=self._path(key)
                self.guard.npz(path,data=data,mask=mask)
                size=path.stat().st_size
                prior=self.db.execute('SELECT size FROM blocks WHERE key=?',(key,)).fetchone()
                self.db.execute('INSERT OR REPLACE INTO blocks VALUES (?,?,?,?)',(key,size,digest(path),time.time()))
                self.disk_bytes+=size-(prior[0] if prior else 0);self.db.commit();self._account_db()
                self.profile.count('block_disk_writes',asset=asset)
        self._remember(key,data,mask)

    def close(self):
        if self.db is None:return
        with self.lock:
            self.db.commit();self.db.execute('PRAGMA wal_checkpoint(TRUNCATE)');self.db.close();self._account_db()


def metadata_from(ds):
    flags=[set(enums) for enums in ds.mask_flag_enums]
    dataset_masked=any({MaskFlags.alpha,MaskFlags.per_dataset,MaskFlags.nodata}&enums for enums in flags)
    outside_masked=not (dataset_masked and flags[0]=={MaskFlags.all_valid})
    bh,bw=ds.block_shapes[0]
    return {'crs':ds.crs.to_wkt(),'transform':list(ds.transform)[:6],'height':ds.height,'width':ds.width,
        'dtype':ds.dtypes[0],'block_height':min(bh,512),'block_width':min(bw,512),
        'nodata':repr(ds.nodata),'outside':repr(float(ds.nodata or 0)),'outside_masked':outside_masked}


class NativeReader:
    def __init__(self,owner,store,options,profile):
        self.owner=owner;self.store=store;self.options=options;self.profile=profile
        self.locks={};self.lock=threading.RLock();self.coordinates=OrderedDict();self.coordinate_points=0
        self.projected=OrderedDict()

    def _project(self,crs,selected):
        # The result of one PROJ call depends only on CRS and points: reuse it across scenes.
        key=(crs,selected.tobytes())
        with self.lock:
            cached=self.projected.get(key)
        if cached is not None:return cached
        x,y=transform_coords(4326,crs,selected[:,0].tolist(),selected[:,1].tolist())
        with self.lock:
            self.projected[key]=(x,y)
            while len(self.projected)>MAX_PROJECTED_CELLS:self.projected.popitem(last=False)
        return x,y

    def _plan(self,meta,points):
        identity=(meta['crs'],tuple(meta['transform']),meta['height'],meta['width'],meta['block_height'],meta['block_width'],hashlib.blake2b(points.tobytes(),digest_size=16).digest())
        with self.lock:
            cached=self.coordinates.pop(identity,None)
            if cached is not None:self.coordinates[identity]=cached
        if cached is not None:
            self.profile.count('coordinate_cache_hits');r,c,groups=cached
        else:
            with self.profile.span('coordinate_transform'):
                rows=[];cols=[];stride=self.owner.point_stride or len(points)
                # Preserve the original per-cell PROJ calls and sample_gen's 256-point affine chunks.
                for start in range(0,len(points),stride):
                    selected=points[start:start+stride]
                    x,y=self._project(meta['crs'],selected)
                    for offset in range(0,len(selected),256):
                        rr,cc=rowcol(Affine(*meta['transform']),x[offset:offset+256],y[offset:offset+256])
                        rows.extend(rr);cols.extend(cc)
                r=np.asarray(rows);c=np.asarray(cols);groups={}
                inside=np.flatnonzero((r>=0)&(r<meta['height'])&(c>=0)&(c<meta['width']))
                if len(inside):
                    block_rows=r[inside]//meta['block_height'];block_cols=c[inside]//meta['block_width']
                    keys=np.stack([block_rows,block_cols],axis=1)
                    unique,inverse=np.unique(keys,axis=0,return_inverse=True)
                    order=np.argsort(inverse.ravel(),kind='stable')
                    counts=np.bincount(inverse.ravel(),minlength=len(unique))
                    for position,indices in zip(unique,np.split(inside[order],np.cumsum(counts)[:-1])):
                        groups[(int(position[0]),int(position[1]))]=indices
            with self.lock:
                self.coordinates[identity]=(r,c,groups);self.coordinate_points+=len(points)
                while self.coordinate_points>MAX_PLANNED_POINTS and len(self.coordinates)>1:
                    _,oldest=self.coordinates.popitem(last=False);self.coordinate_points-=len(oldest[0])
        out=np.full(len(points),np.nan if meta['outside_masked'] else float(meta['outside']),dtype=np.float64)
        return out,r,c,groups

    def _key(self,asset,meta,position):
        value=[asset,meta,position]
        return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()

    def _assign(self,out,r,c,indices,position,meta,block):
        data,mask=block
        rr=r[indices]-position[0]*meta['block_height'];cc=c[indices]-position[1]*meta['block_width']
        values=data[rr,cc].astype(np.float64);values[mask[rr,cc]]=np.nan;out[indices]=values

    def values(self,item,band,points):
        asset=key_for(item,band);self.profile.asset(asset,item,band)
        with self.lock:asset_lock=self.locks.setdefault(asset,threading.RLock())
        with asset_lock:
            self.profile.count('asset_requests',asset=asset)
            meta=self.store.metadata(asset)
            if meta is None:return self._remote(item,band,points,asset,None)
            out,r,c,groups=self._plan(meta,points);missing={}
            for position,indices in groups.items():
                block=self.store.get(self._key(asset,meta,position),asset)
                if block is None:missing[position]=indices
                else:self._assign(out,r,c,indices,position,meta,block)
            if missing:return self._remote(item,band,points,asset,meta,(out,r,c,missing))
            return out

    def _remote(self,item,band,points,asset,meta,plan=None):
        options=self.options;fetched={}
        for attempt in range(options['read_attempts']):
            try:
                href=self.owner.signed(item['assets'][band]['href'])
                with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN='EMPTY_DIR',CPL_VSIL_CURL_ALLOWED_EXTENSIONS='.tif,.tiff',
                    GDAL_HTTP_CONNECTTIMEOUT=str(options['connect_timeout_seconds']),GDAL_HTTP_TIMEOUT=str(options['read_timeout_seconds']),
                    GDAL_HTTP_MAX_RETRY='0',CPL_CURL_VERBOSE=False,CPL_DEBUG=False):
                    with self.profile.span('remote_open',asset):
                        dataset=rasterio.open(href)
                    self.profile.count('remote_asset_opens',asset=asset)
                    with dataset as ds:
                        actual=metadata_from(ds)
                        if meta is not None and actual!=meta:raise ValueError('Metadati asset cambiati rispetto alla cache congelata')
                        if meta is None:
                            meta=actual;self.store.put_metadata(asset,meta);plan=self._plan(meta,points)
                        out,r,c,groups=plan
                        for position,indices in groups.items():
                            # Blocks assigned before a failed attempt are final and are not read twice.
                            if position in fetched:continue
                            key=self._key(asset,meta,position);block=self.store.get(key,asset)
                            if block is None:
                                row=position[0]*meta['block_height'];col=position[1]*meta['block_width']
                                window=Window(col,row,min(meta['block_width'],meta['width']-col),min(meta['block_height'],meta['height']-row))
                                with self.profile.span('remote_read',asset):values=ds.read(1,window=window,masked=True)
                                block=(np.ma.getdata(values).copy(),np.ma.getmaskarray(values).copy())
                                self.profile.count('remote_blocks_read',asset=asset)
                                self.profile.count('decoded_raster_bytes',block[0].nbytes+block[1].nbytes,asset)
                                self.store.put(key,*block,asset)
                            self._assign(out,r,c,indices,position,meta,block);fetched[position]=True
                        return out
            except (rasterio.errors.RasterioError,requests.exceptions.RequestException):
                self.owner.clear_tokens();self.profile.count('remote_read_failures',asset=asset)
                if attempt+1==options['read_attempts']:
                    raise RuntimeError(f'Lettura raster fallita: scena {item["id"]}, banda {band}, tentativi {attempt+1}; cache valida conservata') from None
                self.profile.count('remote_retries',asset=asset)
                # Jitter keeps parallel workers from retrying a throttled server at the same instant.
                with self.profile.span('retry_backoff',asset):time.sleep(min(2**attempt,60)*random.uniform(.5,1))
