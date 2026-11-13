# Scopo: contratti e reader spatial su fixture r1/r2, senza rete, pesi, GPU o dati reali.
# Fasi: dati sintetici firmati -> verifiche negative -> indice atomico/ripresa -> finestre e momenti.
# Input: tmp_path; output: assert esatte su celle, pixel, CRS, dequantizzazione e integrita.
# Uso: python -m pytest tests/test_spatial_sources.py. Risorse: kilobyte; ripresa testata per tile.
from dataclasses import replace
import json
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from biomap_spatial.contracts import ModelConfig, ExperimentSpec
from biomap_spatial.sources import V2Source,V1TileAdapter,prepare_index,inventory,R1,R2
from biomap_spatial.support import Grid,cell_mask,windows,overlap_report
from biomap_spatial.utils import digest,save_json
from spatial_fixtures import make_artifacts


@pytest.mark.parametrize('protocol',[R1,R2])
def test_pixel_schemas_index_and_resume(tmp_path,protocol):
    spec,ref,work,out=make_artifacts(tmp_path,protocol)
    source=V2Source(spec,out,False)
    events=[]
    def interrupted(**kw):
        events.append(kw)
        if len(events)==1:raise KeyboardInterrupt()
    with pytest.raises(KeyboardInterrupt):prepare_index(source,interrupted)
    assert not source.index.with_suffix('.json').exists()
    frame=prepare_index(source)
    assert len(frame)==8 and frame.n_valid.eq(4).all()
    assert frame.patches.eq(1).all() and frame.duplicate_contributions.eq(0).all()
    patch=source.patches('c0')[0]
    original=np.load(work/'pixel_embeddings/aoi_0/grid_test.npy')[[0,1,4,5]]
    np.testing.assert_array_equal(patch.features[:,patch.mask].T,original)
    np.testing.assert_allclose(frame.iloc[0]['mean'],original.mean(0,dtype=np.float64),rtol=1e-7,atol=1e-7)
    sha=digest(source.index)
    source.close()
    again=V2Source(spec,out)
    assert digest(again.index)==sha
    again.close()


@pytest.mark.parametrize('failure',['missing_pixels','corrupt','mean_only','wrong_order','wrong_geometry','wrong_split','nan'])
def test_spatial_preflight_rejects_bad_inputs(tmp_path,failure):
    spec,ref,work,out=make_artifacts(tmp_path)
    p=work/'pixel_embeddings/aoi_0/grid_test.npy'
    if failure in {'missing_pixels','mean_only'}:p.unlink()
    elif failure=='corrupt':p.write_bytes(p.read_bytes()+b'corruption')
    elif failure=='nan':
        x=np.load(p);x[0,0]=np.nan;np.save(p,x)
        meta=json.loads(p.with_suffix('.json').read_text());meta['sha256']=digest(p);save_json(p.with_suffix('.json'),meta)
    elif failure=='wrong_split':
        import pandas as pd
        x=pd.read_parquet(work/'v2.parquet');x.loc[0,'cv_fold']=2;x.to_parquet(work/'v2.parquet')
        meta=json.loads((work/'extraction_complete.json').read_text());meta['sha256']=digest(work/'v2.parquet');save_json(work/'extraction_complete.json',meta)
    else:
        meta=json.loads(p.with_suffix('.json').read_text());meta['order' if failure=='wrong_order' else 'geometry']='wrong';save_json(p.with_suffix('.json'),meta)
    with pytest.raises((ValueError,FileNotFoundError)):V2Source(spec,out,False)


def test_native_grid_masks_borders_padding_overlap_and_empty():
    grid=Grid(4,4,(10,0,0,0,-10,40),'EPSG:32632')
    polygon={'type':'Polygon','coordinates':[[[0,40],[40,40],[40,0],[0,0],[0,40]],[[10,30],[30,30],[30,10],[10,10],[10,30]]]}
    mask=cell_mask(grid,polygon,'EPSG:32632')
    assert mask.sum()==12 and not mask[1:3,1:3].any()
    pieces=list(windows(mask,2));assert sum(m.sum() for _,_,m in pieces)==12
    audit=overlap_report([(grid,r,c,m) for r,c,m in pieces])
    assert audit['duplicate_contributions']==0
    audit=overlap_report([(grid,0,0,mask),(grid,0,0,mask)])
    assert audit['duplicate_contributions']==12
    with pytest.raises(ValueError,match='CRS'):cell_mask(grid,polygon,'EPSG:4326')
    with pytest.raises(ValueError,match='vuoto'):list(windows(np.zeros((2,2),bool),2))
    with pytest.raises(ValueError,match='risoluzioni'):overlap_report([(grid,0,0,mask),(Grid(4,4,(20,0,0,0,-20,80),'EPSG:32632'),0,0,mask)])


def test_v1_adapter_preserves_int8_scale_semantics(tmp_path):
    q=np.arange(24,dtype=np.int8).reshape(2,3,4);sc=np.array([[.5,1,np.nan],[2,3,4]],np.float32)
    np.save(tmp_path/'q.npy',q);np.save(tmp_path/'s.npy',sc)
    with rasterio.open(tmp_path/'grid.tif','w',driver='GTiff',height=2,width=3,count=1,dtype='uint8',crs='EPSG:32632',transform=from_origin(500000,4800000,10,10)) as ds:ds.write(np.ones((1,2,3),np.uint8))
    adapter=V1TileAdapter(tmp_path/'q.npy',tmp_path/'s.npy',tmp_path/'grid.tif','v1-fixture')
    patch=adapter.patch(0,0,np.ones((2,3),bool))
    expected=q.astype(np.float32)*np.where(np.isfinite(sc),sc,0)[...,None]
    np.testing.assert_array_equal(patch.features,expected.transpose(2,0,1))
    assert patch.mask.sum()==5


def test_mean_representation_and_cell_standard_on_maps_are_rejected(tmp_path):
    spec,_,_,_=make_artifacts(tmp_path)
    with pytest.raises(ValueError,match='cell_standard'):replace(spec,protocol=replace(spec.protocol,normalization='cell_standard'))
    with pytest.raises(ValueError,match='mappe'):replace(spec,source={**spec.source,'representation':'mean'})
    assert inventory(spec)['status']=='PRESENT_NOT_VERIFIED'
    assert inventory(replace(spec,source={**spec.source,'root':None}))['status']=='BLOCKED'


def test_pixel_values_must_match_frozen_chunk_sums(tmp_path):
    spec,ref,work,out=make_artifacts(tmp_path)
    p=work/'pixel_embeddings/aoi_0/grid_test.npy'
    x=np.load(p);x[0,0]+=1;np.save(p,x)
    info=json.loads(p.with_suffix('.json').read_text());info['sha256']=digest(p);save_json(p.with_suffix('.json'),info)
    source=V2Source(spec,out,False)
    try:
        with pytest.raises(ValueError,match='somme/conteggi'):prepare_index(source)
        assert not source.index.with_suffix('.json').exists()
    finally:source.close()


def test_duplicate_native_centers_keep_legacy_multiplicity(tmp_path):
    import pandas as pd
    import shutil
    spec,ref,work,out=make_artifacts(tmp_path)
    shutil.copytree(ref/'aoi_0/grid_test',ref/'aoi_0/grid_duplicate')
    shutil.copytree(work/'chunks/aoi_0/grid_test',work/'chunks/aoi_0/grid_duplicate')
    for ext in ['.npy','.json']:shutil.copyfile(work/('pixel_embeddings/aoi_0/grid_test'+ext),work/('pixel_embeddings/aoi_0/grid_duplicate'+ext))
    for path in [ref/'v1.parquet',work/'v2.parquet']:
        frame=pd.read_parquet(path);frame.loc[frame.aoi_id=='aoi_0',['n_valid','n_total']]=8;frame.to_parquet(path,index=False)
    plan=json.loads((ref/'reference.json').read_text());plan['areas'][0]['tiles'].append('grid_duplicate')
    plan['files']={str(p.relative_to(ref)):digest(p) for p in ref.rglob('*') if p.is_file() and p.name!='reference.json'}
    save_json(ref/'reference.json',plan)
    proof=json.loads((work/'extraction_complete.json').read_text());proof['sha256']=digest(work/'v2.parquet');save_json(work/'extraction_complete.json',proof)
    source=V2Source(spec,out,False)
    try:
        frame=prepare_index(source);part=frame[frame.aoi_id=='aoi_0']
        assert part.n_valid.eq(8).all() and part.duplicate_contributions.eq(4).all() and part.patches.eq(2).all()
    finally:source.close()


def test_disk_reserve_blocks_prepare_without_losing_source(tmp_path,monkeypatch):
    from collections import namedtuple
    import biomap_spatial.utils as util
    spec,ref,work,out=make_artifacts(tmp_path)
    spec=replace(spec,execution={**spec.execution,'reserve_gib':20})
    usage=namedtuple('Usage','total used free')
    monkeypatch.setattr(util.shutil,'disk_usage',lambda p:usage(100*1024**3,90*1024**3,10*1024**3))
    source=V2Source(spec,out,False)
    try:
        with pytest.raises(OSError,match='Spazio'):prepare_index(source)
        assert not source.index.exists() and (work/'pixel_embeddings/aoi_0/grid_test.npy').exists()
    finally:source.close()


def test_v1_existing_index_adapter_is_read_only(tmp_path):
    import pandas as pd
    import sqlite3
    from biomap_spatial.sources import V1IndexSource
    from biomap_spatial.engine import preflight
    spec,ref,work,out=make_artifacts(tmp_path)
    root=tmp_path/'v1';root.mkdir();path=root/'spatial.sqlite3'
    q=np.ones((2,4,128),np.int8);scale=np.ones((2,4),np.float32)
    np.save(root/'q.npy',q);np.save(root/'s.npy',scale)
    with rasterio.open(root/'land.tif','w',driver='GTiff',height=2,width=4,count=1,dtype='uint8',crs='EPSG:32632',transform=from_origin(500000,4800000,10,10)) as ds:ds.write(np.ones((1,2,4),np.uint8))
    tile={'embedding':str(root/'q.npy'),'scales':str(root/'s.npy'),'landmask':str(root/'land.tif')}
    tile['sha256']={k:digest(v) for k,v in tile.items()}
    with sqlite3.connect(path) as db:
        db.executescript('CREATE TABLE meta(key TEXT,value TEXT); CREATE TABLE tiles(tile_id TEXT,spec TEXT,complete INTEGER); CREATE TABLE cells(cell_id TEXT,aoi_id TEXT,partition TEXT,cv_fold INTEGER,lon REAL,lat REAL,target REAL,n_total INTEGER,n_valid INTEGER,valid_fraction REAL,mean BLOB,variance BLOB,eligible INTEGER); CREATE TABLE patches(patch_id INTEGER,cell_id TEXT,tile_id TEXT,row0 INTEGER,col0 INTEGER,height INTEGER,width INTEGER,mask BLOB,n_valid INTEGER);')
        db.execute('INSERT INTO meta VALUES(?,?)',('finalized','true'))
        db.execute('INSERT INTO tiles VALUES(?,?,?)',('tile',json.dumps(tile),1))
        for j,r in pd.read_parquet(ref/'v1.parquet').iterrows():
            db.execute('INSERT INTO cells VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)',(r.cell_id,r.aoi_id,r.partition,int(r.cv_fold),r.lon,r.lat,r.target,4,4,1.,np.ones(128,np.float32).tobytes(),np.zeros(128,np.float32).tobytes(),1))
            db.execute('INSERT INTO patches VALUES(?,?,?,?,?,?,?,?,?)',(j,r.cell_id,'tile',0,0,2,2,np.packbits(np.ones((2,2),bool)).tobytes(),4))
    save_json(path.with_suffix('.manifest.json'),{'sha256':digest(path)})
    cfg=replace(spec,source={**spec.source,'kind':'v1_index','root':str(path),'reference':None,'feature_space_id':'v1-fixture'})
    before=digest(path);source,report=preflight(cfg)
    try:
        np.testing.assert_array_equal(source.patches('c0')[0].features,np.ones((128,2,2),np.float32))
        assert report['status']=='READY'
    finally:source.close()
    assert digest(path)==before
