# Scopo: leggere finestre V1 reali da cache indicizzata e creare batch di celle con patch variabili.
# Fasi: verifica cache/source -> SQLite read-only -> memmap int8/scale -> dequantizzazione -> padding.
# Input: spatial.sqlite3 finalizzato; output: patch C,H,W, mask, pesi e un solo target per cella.
# Parametri: batch_cells, source_mmap_limit; nessun resizing o ricostruzione da vettori mean.
# Risorse: RAM limitata da finestre/batch e LRU memmap; nessun encoder V1 viene caricato.
# Ripresa: cache immutabile, statistiche normalizzazione stimate esclusivamente sui training ID.
from collections import OrderedDict
import json
import sqlite3
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def connect(path):return sqlite3.connect(Path(path).resolve().as_uri()+'?mode=ro',uri=True)


def metadata(path):
    from runtime import digest
    marker=Path(path).with_suffix('.manifest.json')
    if not marker.exists() or json.loads(marker.read_text())['sha256']!=digest(path):raise ValueError('Checksum indice assente/discordante: completare build o verificare corruzione')
    with connect(path) as db:return {key:json.loads(value) for key,value in db.execute('SELECT key,value FROM meta')}


def cell_frame(path):
    metadata(path)
    with connect(path) as db:
        info={key:json.loads(value) for key,value in db.execute('SELECT key,value FROM meta')}
        if not info.get('finalized'):raise ValueError('Cache non finalizzata: completare build')
        rows=db.execute('SELECT cell_id,aoi_id,partition,cv_fold,lon,lat,target,n_total,n_valid,valid_fraction,mean,variance,(SELECT count(*) FROM patches p WHERE p.cell_id=cells.cell_id) FROM cells WHERE eligible=1 ORDER BY cell_id').fetchall()
    frame=pd.DataFrame(rows,columns=['cell_id','aoi_id','partition','cv_fold','lon','lat','target','n_total','n_valid','valid_fraction','mean_bytes','var_bytes','patches'])
    if not len(frame):raise ValueError('Cache senza celle eleggibili')
    frame['mean']=[np.frombuffer(x,dtype=np.float32).copy() for x in frame.pop('mean_bytes')]
    frame['variance']=[np.frombuffer(x,dtype=np.float32).copy() for x in frame.pop('var_bytes')]
    return frame


def verify_sources(path,deep=False):
    from runtime import digest
    with connect(path) as db:specs=[json.loads(x[0]) for x in db.execute('SELECT spec FROM tiles WHERE complete=1')]
    for tile in specs:
        for key in ['embedding','scales','landmask']:
            p=Path(tile[key]);s=p.stat();stored=tile['files'][key]
            if s.st_size!=stored['size'] or s.st_mtime_ns!=stored['mtime_ns']:raise ValueError('Sorgente modificata o spostata: '+str(p))
            if deep and digest(p)!=tile['sha256'][key]:raise ValueError('SHA sorgente discordante: '+str(p))
    return len(specs)


def fit_normalizer(frame):
    if not len(frame) or set(frame.partition)!={'train'}:raise ValueError('Normalizzazione: soltanto celle training')
    count=frame.n_valid.to_numpy(dtype=np.float64);means=np.stack(frame['mean']).astype(np.float64);var=np.stack(frame.variance).astype(np.float64)
    center=(means*count[:,None]).sum(axis=0)/count.sum()
    spread=np.sqrt(np.maximum(((var+means**2)*count[:,None]).sum(axis=0)/count.sum()-center**2,0))
    return {'mean':center.astype(np.float32),'std':np.maximum(spread,1e-6).astype(np.float32),'training_ids':frame.cell_id.tolist()}


class MeanDataset(Dataset):
    def __init__(self,frame,norm):self.frame=frame.reset_index(drop=True);self.norm=norm
    def __len__(self):return len(self.frame)
    def __getitem__(self,index):
        cell=self.frame.iloc[index]
        return {'cell_id':cell.cell_id,'mean':(cell['mean']-self.norm['mean'])/self.norm['std'],'target':float(cell.target)}
    def close(self):pass


def collate_mean(items):
    return {'mean':torch.from_numpy(np.stack([x['mean'] for x in items]).astype(np.float32)),
        'target':torch.tensor([x['target'] for x in items],dtype=torch.float32),'cell_ids':[x['cell_id'] for x in items]}


class SpatialDataset(Dataset):
    def __init__(self,path,frame,normalizer,source_mmap_limit=8):
        self.path=str(path);self.frame=frame.reset_index(drop=True);self.norm=normalizer;self.limit=source_mmap_limit;self.db=None;self.maps=OrderedDict()
    def __len__(self):return len(self.frame)
    def __getstate__(self):
        result=dict(self.__dict__);result['db']=None;result['maps']=OrderedDict();return result
    def _tile(self,tile_id):
        if tile_id in self.maps:
            value=self.maps.pop(tile_id);self.maps[tile_id]=value;return value
        spec=json.loads(self.db.execute('SELECT spec FROM tiles WHERE tile_id=?',(tile_id,)).fetchone()[0])
        value=(np.load(spec['embedding'],mmap_mode='r',allow_pickle=False),np.load(spec['scales'],mmap_mode='r',allow_pickle=False))
        self.maps[tile_id]=value
        while len(self.maps)>self.limit:self.maps.popitem(last=False)
        return value
    def __getitem__(self,index):
        if self.db is None:self.db=connect(self.path)
        cell=self.frame.iloc[index];patches=[]
        for tile_id,row,col,height,width,packed,n in self.db.execute('SELECT tile_id,row0,col0,height,width,mask,n_valid FROM patches WHERE cell_id=? ORDER BY patch_id',(cell.cell_id,)):
            emb,sc=self._tile(tile_id);mask=np.unpackbits(np.frombuffer(packed,dtype=np.uint8),count=height*width).reshape(height,width).astype(bool)
            if int(mask.sum())!=n or not n:raise ValueError('Maschera di patch incoerente')
            scale=sc[row:row+height,col:col+width];raw=emb[row:row+height,col:col+width].astype(np.float32)
            values=raw*np.where(mask,scale,0)[...,None]
            values=np.where(mask[...,None],(values-self.norm['mean'])/self.norm['std'],0)
            if not np.isfinite(values).all():raise ValueError('Feature di patch non finite')
            patches.append((np.ascontiguousarray(values.transpose(2,0,1)),mask[None],n))
        if not patches:raise ValueError('Cella eleggibile senza patch')
        return {'cell_id':cell.cell_id,'patches':patches,'target':float(cell.target),'mean':(cell['mean']-self.norm['mean'])/self.norm['std']}
    def close(self):
        if self.db:self.db.close();self.db=None
        self.maps.clear()


def collate_cells(items):
    flat=[(i,p) for i,item in enumerate(items) for p in item['patches']]
    h=max(p[0].shape[-2] for _,p in flat);w=max(p[0].shape[-1] for _,p in flat)
    x=np.zeros((len(flat),128,h,w),dtype=np.float32);m=np.zeros((len(flat),1,h,w),dtype=bool)
    owners=[];weights=[]
    for index,(owner,(features,mask,count)) in enumerate(flat):
        ph,pw=features.shape[-2:];x[index,:,:ph,:pw]=features;m[index,:,:ph,:pw]=mask;owners.append(owner);weights.append(count)
    return {'features':torch.from_numpy(x),'mask':torch.from_numpy(m),'owners':torch.tensor(owners,dtype=torch.long),
        'weights':torch.tensor(weights,dtype=torch.float32),'target':torch.tensor([v['target'] for v in items],dtype=torch.float32),
        'mean':torch.from_numpy(np.stack([v['mean'] for v in items]).astype(np.float32)),'cell_ids':[v['cell_id'] for v in items]}
