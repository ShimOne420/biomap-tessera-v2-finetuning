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
