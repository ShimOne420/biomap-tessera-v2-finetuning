# Scopo: fixture CPU isolate di vere tile georiferite sintetiche per V1 spatial.
# Fasi: 17 AOI sviluppo + Liguria -> int8/scale/UTM -> target CSV -> indice transazionale.
# Input: tmp_path; output: soltanto cartelle pytest temporanee, mai dati della campagna.
# Parametri: pytest experiments/v1_spatial_cnn_msa/tests; nessuna rete o encoder.
# Risorse: pochi MB e un thread Torch; metriche sintetiche non scientifiche.
# Ripresa: ogni test ha una root privata; i sorgenti possono essere verificati per immutabilità.
from pathlib import Path
import sys
import json
import importlib
import numpy as np
import pandas as pd
import pytest
import rasterio
from rasterio.transform import from_origin
from pyproj import Transformer
import torch
import yaml

EXPERIMENT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(EXPERIMENT))
from runtime import Progress

torch.set_num_threads(1)


@pytest.fixture
def synthetic(tmp_path):
    cfg=yaml.safe_load((EXPERIMENT/'config.yaml').read_text())
    cfg.update(work_root=str(tmp_path/'v1_spatial_msa'),preset='pilot',protocol='v1.0-spatial-native-centers-cell-loss-r1')
    cfg['data'].update(v1_root=str(tmp_path/'v1_sources'),targets_csv=str(tmp_path/'targets.csv'))
    cfg['storage'].update(max_gib=1,reserve_gib=0)
    cfg['training'].update(device='cpu',epochs=1,patience=1,batch_cells=8,num_workers=0)
    cfg['model'].update(conv_channels=[8,4],hidden=8,dropout=0.0)
    frame=[];project=Transformer.from_crs(4326,32632,always_xy=True);step=1/360
    for a in range(18):
        aoi='aoi_%02d'%a if a<17 else 'liguria_levante';west=8.15+a*.1;south=44.05;north=south+step
        name=f'grid_{west:.2f}_{south:.2f}';root=Path(cfg['data']['v1_root'])/aoi/'tessera/2020'
        dest=root/'global_0.1_degree_representation/2020'/name;dest.mkdir(parents=True)
        rng=np.random.default_rng(a);values=rng.integers(-90,90,(48,70,128),dtype=np.int8);scales=rng.uniform(.1,1,(48,70)).astype(np.float32);scales[8:10,8:10]=np.nan
        np.save(dest/(name+'.npy'),values);np.save(dest/(name+'_scales.npy'),scales)
        x,y=project.transform(west,north);tif=root/'global_0.1_degree_tiff_all'/(name+'.tiff');tif.parent.mkdir()
        with rasterio.open(tif,'w',driver='GTiff',height=48,width=70,count=1,dtype='uint8',crs=32632,transform=from_origin(x-20,y+20,10,10)) as ds:ds.write(np.ones((48,70),dtype=np.uint8),1)
        (root/'tessera_metadata.json').write_text(json.dumps({'dataset_version':'1.0','dataset_variant':'vultr','year':2020}))
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
