# SCOPO
#   Fixture CPU per l'esperimento V2 spatial: tile V1 sintetiche georiferite (come i test dell'Esp. B) più
#   geometrie r1/r2 e embedding V2 per pixel sintetici con JSON di identità r2-fast.
# COME VIENE RICHIAMATO
#   pytest experiments/v2_spatial_cnn_msa/tests (nessuna rete, nessun encoder, nessun dato della campagna).
# LOGICA IN ORDINE
#   1. 17 AOI di sviluppo + Liguria, una tile V1 int8/scale/TIFF ciascuna, 2 celle per AOI;
#   2. reference/<aoi>/<tile>/geometry.npz (tutti i pixel della tile) + geometry.json con la griglia V1;
#   3. pixel_embeddings V2 = valori V1 dequantizzati (int8 * scala), così il risultato deve coincidere con B;
#      i pixel con scala NaN ricevono un valore finito arbitrario (fuori maschera, non devono essere letti).
# INPUT E OUTPUT
#   Solo cartelle temporanee di pytest.
# PARAMETRI
#   Nessuno.
# RISORSE
#   Pochi MB, un thread Torch.
# RIPRESA
#   Ogni test ha la sua root privata.
from pathlib import Path
import sys
import json
import hashlib
import numpy as np
import pandas as pd
import pytest
import rasterio
from rasterio.transform import from_origin
from pyproj import Transformer
import torch
import yaml

EXPERIMENT=Path(__file__).resolve().parents[1]
B_EXPERIMENT=EXPERIMENT.parent/'v1_spatial_cnn_msa'
sys.path.insert(0,str(EXPERIMENT))
from runtime import Progress, V2_PIXEL_PROTOCOL, PROTOCOL
from biomap_v2.common import WEIGHTS_SHA256

torch.set_num_threads(1)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_v2(tmp_path,aoi,name,values,scales,transform,crs='EPSG:32632',protocol=V2_PIXEL_PROTOCOL):
    h,w=scales.shape;ref=tmp_path/'reference'/aoi/name;ref.mkdir(parents=True,exist_ok=True)
    flat=np.arange(h*w,dtype=np.int64)
    np.savez(ref/'geometry.npz',flat_indices=flat,cell_ids=np.zeros(h*w,dtype=np.int64))
    (ref/'geometry.json').write_text(json.dumps({'tile':name,'pixels':h*w,'height':h,'width':w,'transform':list(transform)[:6],'crs':crs}))
    emb=(values.astype(np.float32)*scales[...,None]).reshape(h*w,128)
    emb[~np.isfinite(emb)]=7.0
    out=tmp_path/'v2_r2'/'pixel_embeddings'/aoi;out.mkdir(parents=True,exist_ok=True)
    np.save(out/(name+'.npy'),emb.astype(np.float32))
    geometry=sha(ref/'geometry.npz')
    identity={'protocol':protocol,'year':2020,'checkpoint':WEIGHTS_SHA256,'precision':'fp32','code':{},'geometry':geometry,'tile':name}
    (out/(name+'.json')).write_text(json.dumps({'sha256':sha(out/(name+'.npy')),'identity':identity,'order':'geometry.npz flat_indices','dtype':'float32',
        'geometry_sha256':geometry,'grid':{'width':w,'height':h,'transform':list(transform)[:6],'crs':crs}}))


@pytest.fixture
def synthetic(tmp_path):
    cfg=yaml.safe_load((EXPERIMENT/'config.yaml').read_text(encoding='utf-8'))
    cfg.update(work_root=str(tmp_path/'v2_spatial_msa'),preset='pilot',protocol=PROTOCOL,status_name='status.json',lock_name='experiment.lock')
    cfg['data'].update(v1_root=str(tmp_path/'v1_sources'),targets_csv=str(tmp_path/'targets.csv'),v2_root=str(tmp_path/'v2_r2'),reference=str(tmp_path/'reference'))
    cfg['storage'].update(max_gib=1,reserve_gib=0)
    cfg['training'].update(device='cpu',epochs=1,patience=1,batch_cells=8,num_workers=0,pin_memory=False)
    cfg['model'].update(conv_channels=[8,4],hidden=8,dropout=0.0)
    frame=[];project=Transformer.from_crs(4326,32632,always_xy=True);step=1/360
    for a in range(18):
        aoi='aoi_%02d'%a if a<17 else 'liguria_levante';west=8.15+a*.1;south=44.05;north=south+step
        name=f'grid_{west:.2f}_{south:.2f}';root=Path(cfg['data']['v1_root'])/aoi/'tessera/2020'
        dest=root/'global_0.1_degree_representation/2020'/name;dest.mkdir(parents=True)
        rng=np.random.default_rng(a);values=rng.integers(-90,90,(48,70,128),dtype=np.int8);scales=rng.uniform(.1,1,(48,70)).astype(np.float32);scales[8:10,8:10]=np.nan
        np.save(dest/(name+'.npy'),values);np.save(dest/(name+'_scales.npy'),scales)
        x,y=project.transform(west,north);tif=root/'global_0.1_degree_tiff_all'/(name+'.tiff');tif.parent.mkdir()
        transform=from_origin(x-20,y+20,10,10)
        with rasterio.open(tif,'w',driver='GTiff',height=48,width=70,count=1,dtype='uint8',crs=32632,transform=transform) as ds:ds.write(np.ones((48,70),dtype=np.uint8),1)
        (root/'tessera_metadata.json').write_text(json.dumps({'dataset_version':'1.0','dataset_variant':'vultr','year':2020}))
        write_v2(tmp_path,aoi,name,values,scales,transform)
        for i in range(2):frame.append({'cell_id':f'{aoi}:{i}','aoi_id':aoi,'partition':'train' if a<17 else 'test','cv_fold':a%3 if a<17 else -1,
            'west':west+i*step,'east':west+(i+1)*step,'south':south,'north':north,'lon':west+(i+.5)*step,'lat':south+step/2,'target':.15+a*.02+i*.1})
    frame=pd.DataFrame(frame);frame[['cell_id','target']].to_csv(cfg['data']['targets_csv'],index=False)
    return cfg,frame.drop(columns='target')


@pytest.fixture
def cache(synthetic):
    from patch_builder import build
    cfg,frame=synthetic;p=Progress(cfg,'test_build')
    try:path=build(cfg,frame,p)
    finally:p.close()
    return cfg,frame,path
