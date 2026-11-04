# Scopo: provare parità numerica e riduzione delle aperture raster senza rete o dati inventati nel benchmark reale.
# Fasi: GeoTIFF sintetici -> riferimento seriale -> lotti/cache/thread -> confronto array e ripresa/interruzione.
# Input: fixture note con nuvole, NoData, valori estremi e orbite; output: report/test temporanei.
# Parametri: python -m pytest tests/test_preparation_performance.py; cache/worker variati nei test.
# Risorse: CPU e disco temporaneo; la riduzione di richieste è misurata, non è una promessa di throughput Windows.
# Ripresa: test delle transazioni pending, cache vecchie, cache nativa calda e dati validi non riscritti.
import argparse
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import time
import numpy as np
import pandas as pd
import pytest
import rasterio
from rasterio.transform import from_origin
from biomap_v2 import data
from biomap_v2.common import config,digest,root
from biomap_v2.preparation import BatchAcquirer,atomic_pair,preparation_lock,recover_pair
from biomap_v2.preparation_profile import PreparationProfile,SpaceGuard

STAC_QUERY=data.query


def make_cfg(tmp_path,engine='optimized'):
    cfg=config('msa',argparse.Namespace(config=None,data_root=str(tmp_path/'data'),set=['storage.reserve_gib=0']))
    cfg['acquisition'].update(engine=engine,shared_catalogs=True,batch_cells=8,io_workers=4,progress_seconds=120,
        block_cache_gib=.05,memory_cache_mib=4)
    cfg['sampling'].update(min_s2_observations=2,min_s1_observations=1)
    return cfg


def tif(path,values,transform,crs=4326,nodata=-9999):
    path.parent.mkdir(parents=True,exist_ok=True)
    with rasterio.open(path,'w',driver='GTiff',height=values.shape[-2],width=values.shape[-1],count=1,
        dtype=str(values.dtype),crs=f'EPSG:{crs}',transform=transform,nodata=nodata,tiled=True,blockxsize=16,blockysize=16) as ds:
        ds.write(values,1)
    return path


@pytest.fixture
def observations(tmp_path,monkeypatch):
    ncols,nrows=4,2;step=1/360;west,north=8.,44.;resolution=step/25
    yy,xx=np.indices((50,100));transform=from_origin(west,north,resolution,resolution)
    frame=[]
    for i in range(8):
        col=i%4;row=i//4
        frame.append({'cell_id':str(i),'aoi_id':'fixture_aoi','partition':'train' if i<4 else ('validation' if i<6 else 'test'),
            'west':west+col*step,'east':west+(col+1)*step,'south':north-(row+1)*step,'north':north-row*step,
            'lon':west+(col+.5)*step,'lat':north-(row+.5)*step})
    frame=pd.DataFrame(frame);csv=tmp_path/'cells.csv';frame.to_csv(csv,index=False)
    target=tif(tmp_path/'target.tif',np.arange(8,dtype=np.float32).reshape(2,4)/10,from_origin(west,north,step,step))
    items={'sentinel-2-l2a':[],'sentinel-1-rtc':[]}
    for day in ['2020-06-01','2020-12-31']:
        for priority in range(3):
            assets={};scene=f's2_{day}_{priority}'
            scl=np.full((50,100),4,dtype=np.uint8)
            if priority==0:scl[(xx<50)&(xx%7<3)]=8
            # Last cell is intentionally missing S2; original and optimized must make the same exclusion.
            scl[(yy>=25)&(xx>=75)]=8
            assets['SCL']={'href':str(tif(tmp_path/scene/'SCL.tiff',scl,transform,nodata=0))}
            for index,band in enumerate(data.BANDS):
                arr=(1000+priority*4000+index*50+xx+yy).astype(np.float32)
                if band=='B02':arr[(xx==17)&(yy==12)]=-9999
                if band=='B04':arr[(xx==7)&(yy==7)]=80000
                assets[band]={'href':str(tif(tmp_path/scene/(band+'.tif'),arr,transform))}
            items['sentinel-2-l2a'].append({'id':scene,'properties':{'datetime':day+'T10:00:00Z','eo:cloud_cover':priority*10},'assets':assets})
    for orbit in ['ascending','descending']:
        assets={};scene='s1_'+orbit
        for index,band in enumerate(['vv','vh']):
            values=(.01+xx*.0001+yy*.00001+index*.001).astype(np.float32)
            values[(xx==12)&(yy==12)]=0
            assets[band]={'href':str(tif(tmp_path/scene/(band+'.tif'),values,transform))}
        items['sentinel-1-rtc'].append({'id':scene,'properties':{'datetime':'2020-06-01T06:00:00Z','sat:orbit_state':orbit},'assets':assets})
    calls=[]
    def query(cfg,cell,collection,cache):calls.append((cell.cell_id,collection));return items[collection]
    monkeypatch.setattr(data,'query',query);monkeypatch.setattr(data.Reader,'signed',lambda self,href:href)
    args=argparse.Namespace(limit=None,target_raster=str(target),import_cache=None)
    return frame,csv,args,items,calls


def prepared_arrays(cfg,frame):
    result={}
    for cid in frame.cell_id:
        path=data.cell_directory(cfg,cid)/'series.npz'
        if path.exists():
            with np.load(path,allow_pickle=False) as values:result[cid]={key:values[key] for key in values.files}
    return result


def test_batched_pipeline_equals_original_and_reduces_opens(tmp_path,observations):
    frame,csv,args,items,calls=observations
    legacy=make_cfg(tmp_path/'legacy','legacy');legacy['cells']=str(csv)
    optimized=make_cfg(tmp_path/'optimized');optimized['cells']=str(csv)
    start=time.perf_counter();old=data.prepare(legacy,args);old_seconds=time.perf_counter()-start
    baseline=prepared_arrays(legacy,frame);old_calls=len(calls);calls.clear()
    start=time.perf_counter();new=data.prepare(optimized,args);new_seconds=time.perf_counter()-start
    actual=prepared_arrays(optimized,frame)
    assert set(actual)==set(baseline)=={str(i) for i in range(7)}
    for cid in baseline:
        assert set(actual[cid])==set(baseline[cid])
        for key in baseline[cid]:np.testing.assert_array_equal(actual[cid][key],baseline[cid][key],err_msg=f'{cid}/{key}')
    from biomap_v2.encoder import groups,pixel_embeddings
    from biomap_v2.student import PixelStudent
    model=PixelStudent(latent_dim=8,num_layers=1,nhead=4,dim_feedforward=32,dropout=0).eval()
    for (old_idx,old_s2,old_s1),(new_idx,new_s2,new_s1) in zip(groups(baseline['0']),groups(actual['0'])):
        np.testing.assert_array_equal(old_idx,new_idx);np.testing.assert_array_equal(old_s2,new_s2);np.testing.assert_array_equal(old_s1,new_s1)
    np.testing.assert_array_equal(pixel_embeddings(model,baseline['0']).detach().numpy(),pixel_embeddings(model,actual['0']).detach().numpy())
    pd.testing.assert_frame_equal(pd.read_csv(old).drop(columns='series_sha256'),pd.read_csv(new).drop(columns='series_sha256'))
    assert old_calls==16 and len(calls)==2
    old_profile=json.loads(next((root(legacy)/'diagnostics').glob('*_prepare/profile.json')).read_text())
    new_profile=json.loads(next((root(optimized)/'diagnostics').glob('*_prepare/profile.json')).read_text())
    assert new_profile['counters']['remote_asset_opens']*4<old_profile['counters']['remote_asset_opens']
    assert new_profile['timings']['disk_inventory']['calls']==1
    assert new_profile['completed_cells']==8 and new_profile['excluded_cells']==1
    assert new_profile['new_cells_per_hour']>0 and new_profile['eta_seconds']==0
    evidence={'legacy_opens':old_profile['counters']['remote_asset_opens'],'optimized_opens':new_profile['counters']['remote_asset_opens'],
        'legacy_seconds':old_seconds,'optimized_seconds':new_seconds,'data':'local synthetic GeoTIFF; not Windows network throughput'}
    print('PERFORMANCE_EVIDENCE',json.dumps(evidence))


def test_completed_legacy_cache_is_not_rewritten(tmp_path,observations,monkeypatch):
    frame,csv,args,items,calls=observations
    cfg=make_cfg(tmp_path,'legacy');cfg['cells']=str(csv)
    data.prepare(cfg,args)
    paths=[data.cell_directory(cfg,str(i))/'series.npz' for i in range(7)]
    hashes=[digest(path) for path in paths];mtimes=[path.stat().st_mtime_ns for path in paths]
    cfg['acquisition']['engine']='optimized'
    # The one excluded cell is recomputed from its retained per-day arrays, with no remote reads.
    monkeypatch.setattr(data.Reader,'values',lambda *args:pytest.fail('Unexpected raster read on resume'))
    data.prepare(cfg,args)
    assert [digest(path) for path in paths]==hashes and [path.stat().st_mtime_ns for path in paths]==mtimes


def test_warm_native_cache_survives_new_reader(tmp_path,observations,monkeypatch):
    frame,csv,args,items,calls=observations
    cfg=make_cfg(tmp_path);cfg['cells']=str(csv)
    points=data.sample_points(next(frame.itertuples()),25);item=items['sentinel-2-l2a'][0]
    p=PreparationProfile(cfg,1);guard=SpaceGuard(cfg,p);reader=data.Reader(cfg,p,guard)
    expected=reader.values(item,'B04',points);reader.close();p.finish('complete')
    p2=PreparationProfile(cfg,1);g2=SpaceGuard(cfg,p2);reader2=data.Reader(cfg,p2,g2)
    monkeypatch.setattr(reader2,'signed',lambda href:pytest.fail('Network should not be accessed for warm blocks'))
    actual=reader2.values(item,'B04',points)
    np.testing.assert_array_equal(actual,expected)
    assert p2.counters['block_disk_hits']>0
    reader2.close();p2.finish('complete')


@pytest.mark.parametrize('crs',[4326,32632])
@pytest.mark.parametrize('nodata',[None,-9999,np.nan])
def test_native_samples_match_rasterio_at_masks_edges_and_grid(tmp_path,monkeypatch,crs,nodata):
    cfg=make_cfg(tmp_path);p=PreparationProfile(cfg,1);g=SpaceGuard(cfg,p)
    transform=from_origin(8 if crs==4326 else 420000,44 if crs==4326 else 4900000,.001 if crs==4326 else 10,.001 if crs==4326 else 10)
    values=np.arange(32*32,dtype=np.float32).reshape(32,32)
    if nodata is not None:values[5:8,5:8]=nodata
    path=tif(tmp_path/'native.tif',values,transform,crs,nodata)
    x=[];y=[]
    for rr,cc in [(-1,0),(0,0),(5.5,5.5),(16,16),(31.99999,31.99999),(32,32),(2,7)]:
        a,b=transform*(cc,rr);x.append(a);y.append(b)
    from rasterio.warp import transform as warp
    lon,lat=warp(crs,4326,x,y);points=np.column_stack([lon,lat])
    item={'id':'edge-scene','assets':{'band':{'href':str(path)}}}
    monkeypatch.setattr(data.Reader,'signed',lambda self,href:href)
    baseline=data.Reader().values(item,'band',points)
    reader=data.Reader(cfg,p,g);actual=reader.values(item,'band',points)
    np.testing.assert_array_equal(actual,baseline)
    reader.close();p.finish('complete')


def test_pending_transaction_recovered_and_process_lock(tmp_path,monkeypatch):
    cfg=make_cfg(tmp_path);p=PreparationProfile(cfg,1);guard=SpaceGuard(cfg,p)
    path=root(cfg)/'series/test.npz';meta=path.with_suffix('.json');original=guard.replace
    def interrupted(source,destination):
        if destination==meta:raise KeyboardInterrupt()
        return original(source,destination)
    monkeypatch.setattr(guard,'replace',interrupted)
    with pytest.raises(KeyboardInterrupt):atomic_pair(path,meta,{'array':np.arange(10)},{'scene_ids':['a']},guard,p)
    assert path.exists() and not meta.exists()
    monkeypatch.setattr(guard,'replace',original);recover_pair(path,meta,guard,p)
    assert json.loads(meta.read_text())['sha256']==digest(path)
    with preparation_lock(cfg):
        with pytest.raises(RuntimeError,match='altro prepare'):
            with preparation_lock(cfg):pass
    with preparation_lock(cfg):pass
    p.finish('complete')


def test_interrupted_batch_resumes_verified_days(tmp_path,observations,monkeypatch):
    frame,csv,args,items,calls=observations
    cfg=make_cfg(tmp_path);cfg['cells']=str(csv)
    original=BatchAcquirer.read_bands
    def interrupt_second_date(self,item,bands,points):
        if '2020-12-31' in item['id']:raise KeyboardInterrupt()
        return original(self,item,bands,points)
    monkeypatch.setattr(BatchAcquirer,'read_bands',interrupt_second_date)
    with pytest.raises(KeyboardInterrupt):data.prepare(cfg,args)
    day_files=list((root(cfg)/'series').glob('*/s2_2020-06-01.npz'))
    assert len(day_files)==8
    hashes={str(path):digest(path) for path in day_files}
    seen=[]
    def resumed(self,item,bands,points):seen.append(item['id']);return original(self,item,bands,points)
    monkeypatch.setattr(BatchAcquirer,'read_bands',resumed)
    data.prepare(cfg,args)
    assert not any('2020-06-01' in scene and scene.startswith('s2_') for scene in seen)
    assert len(prepared_arrays(cfg,frame))==7
    profiles=[json.loads(path.read_text()) for path in (root(cfg)/'diagnostics').glob('*_prepare/profile.json')]
    assert {p['status'] for p in profiles}=={'interrupted','complete'}


def test_parallel_band_reads_are_bounded_and_deterministic(tmp_path,observations,monkeypatch):
    import threading
    frame,csv,args,items,calls=observations
    cfg=make_cfg(tmp_path);cfg['cells']=str(csv);profile=PreparationProfile(cfg,1);guard=SpaceGuard(cfg,profile)
    reader=BatchAcquirer(cfg,profile,guard);active=0;peak=0;lock=threading.Lock()
    def fake_values(item,band,points):
        nonlocal active,peak
        with lock:active+=1;peak=max(peak,active)
        try:time.sleep(.01);return np.full(len(points),data.BANDS.index(band))
        finally:
            with lock:active-=1
    monkeypatch.setattr(reader.reader,'values',fake_values)
    result=reader.read_bands(items['sentinel-2-l2a'][0],data.BANDS,np.zeros((25,2)))
    assert 1<peak<=cfg['acquisition']['io_workers']
    assert [int(arr[0]) for arr in result]==list(range(10))
    reader.close();profile.finish('complete')


def test_cache_eviction_only_removes_owned_blocks_and_detects_corruption(tmp_path,observations,monkeypatch):
    frame,csv,args,items,calls=observations
    cfg=make_cfg(tmp_path);cfg['cells']=str(csv);p=PreparationProfile(cfg,1);g=SpaceGuard(cfg,p)
    reader=data.Reader(cfg,p,g);item=items['sentinel-2-l2a'][0];points=data.sample_points(next(frame.itertuples()),25)
    untouched=root(cfg)/'series/existing/series.npz';untouched.parent.mkdir(parents=True);untouched.write_bytes(b'existing series must stay')
    reader.values(item,'B04',points)
    block_key=reader.store.db.execute('SELECT key FROM blocks LIMIT 1').fetchone()[0]
    block_path=reader.store._path(block_key);reader.close();p.finish('complete')
    block_path.write_bytes(b'corrupt')
    p2=PreparationProfile(cfg,1);g2=SpaceGuard(cfg,p2);reader2=data.Reader(cfg,p2,g2)
    with pytest.raises(ValueError,match='corrotto'):reader2.values(item,'B04',points)
    reader2.store.disk_limit=0;reader2.store._trim(0)
    assert reader2.store.disk_bytes==0 and untouched.read_bytes()==b'existing series must stay'
    reader2.close();p2.finish('complete')


def test_retry_refreshes_signed_address_and_preserves_bounds(tmp_path,observations,monkeypatch):
    import biomap_v2.raster_cache as cache
    frame,csv,args,items,calls=observations
    cfg=make_cfg(tmp_path);cfg['cells']=str(csv);p=PreparationProfile(cfg,1);g=SpaceGuard(cfg,p)
    reader=data.Reader(cfg,p,g);item=items['sentinel-2-l2a'][0];points=data.sample_points(next(frame.itertuples()),25)
    attempts=[]
    def signed(href):
        attempts.append(href)
        return str(tmp_path/'missing.tif') if len(attempts)==1 else href
    monkeypatch.setattr(reader,'signed',signed);monkeypatch.setattr(cache.time,'sleep',lambda delay:None)
    result=reader.values(item,'B04',points)
    assert len(attempts)==2 and p.counters['remote_retries']==1
    np.testing.assert_array_equal(result,data.Reader().values(item,'B04',points))
    reader.close();p.finish('complete')


def test_space_guard_inventories_once_for_many_writes(tmp_path):
    cfg=make_cfg(tmp_path);p=PreparationProfile(cfg,1);guard=SpaceGuard(cfg,p)
    for i in range(40):guard.json(root(cfg)/f'fixture/{i}.json',{'i':i})
    assert p.timings['disk_inventory']['calls']==1
    assert p.timings['disk_budget']['calls']>=40
    p.finish('complete')


def test_inventory_tolerates_profile_temp_file_replaced_during_scan(tmp_path,monkeypatch):
    cfg=make_cfg(tmp_path);profile=PreparationProfile(cfg,1)
    class ReplacedTempFile:
        def is_file(self):return True
        def stat(self):raise FileNotFoundError('heartbeat replaced this temporary file')
    with monkeypatch.context() as patch:
        patch.setattr(Path,'rglob',lambda self,pattern:iter([ReplacedTempFile()]))
        guard=SpaceGuard(cfg,profile)
    assert guard.used==32*1024**2
    profile.finish('complete')



def test_cache_identity_ignores_sas_but_tracks_source_version(tmp_path,caplog):
    from biomap_v2.raster_cache import key_for,install_safe_logging
    import logging
    item={'id':'scene','properties':{'datetime':'2020-01-01T00:00:00Z'},'assets':{'B04':{'href':'https://example.test/a.tif?sig=ONE','file:checksum':'abc'}}}
    alternate=copy.deepcopy(item);alternate['assets']['B04']['href']='https://example.test/a.tif?sig=TWO'
    assert key_for(item,'B04')==key_for(alternate,'B04')
    alternate['assets']['B04']['file:checksum']='changed'
    assert key_for(item,'B04')!=key_for(alternate,'B04')
    install_safe_logging()
    with caplog.at_level(logging.ERROR):logging.getLogger('rasterio._err').error('Failed https://example.test/a.tif?sig=SECRET')
    assert 'SECRET' not in caplog.text and '<redacted>' in caplog.text


@pytest.mark.parametrize('batch,workers',[(1,1),(3,2)])
def test_batch_boundaries_and_worker_counts_do_not_change_arrays(tmp_path,observations,batch,workers):
    frame,csv,args,items,calls=observations
    legacy=make_cfg(tmp_path/'legacy','legacy');legacy['cells']=str(csv)
    fast=make_cfg(tmp_path/'fast');fast['cells']=str(csv);fast['acquisition'].update(batch_cells=batch,io_workers=workers)
    data.prepare(legacy,args);data.prepare(fast,args)
    a=prepared_arrays(legacy,frame);b=prepared_arrays(fast,frame)
    assert a.keys()==b.keys()
    for cell in a:
        for key in a[cell]:np.testing.assert_array_equal(a[cell][key],b[cell][key])


def test_stac_timeouts_pagination_and_post_retry_policy(tmp_path,observations,monkeypatch):
    frame,csv,args,items,calls=observations
    # Exercise the actual STAC function independently of the fixture's raster catalog stub.
    query=STAC_QUERY
    cfg=make_cfg(tmp_path);cfg['cells']=str(csv);requested=[];closed=[]
    class Session:
        def request(self,method,url,**kwargs):
            requested.append((method,kwargs))
            page={'features':[{'id':str(len(requested))}],'links':[]}
            if len(requested)==1:page['links']=[{'rel':'next','href':data.STAC+'?page=2','method':'GET'}]
            return SimpleNamespace(raise_for_status=lambda:None,json=lambda:page)
        def close(self):closed.append(True)
    original=data.session
    monkeypatch.setattr(data,'session',lambda **kwargs:Session())
    first=query(cfg,next(frame.itertuples()),'sentinel-2-l2a',tmp_path/'unused.json')
    second=query(cfg,next(frame.itertuples()),'sentinel-2-l2a',tmp_path/'unused.json')
    assert first==second and len(requested)==2 and closed==[True]
    assert [call[0] for call in requested]==['POST','GET']
    assert all(call[1]['timeout']==(10,30) for call in requested)
    http=original(retries=2,post_retries=True)
    policy=http.get_adapter('https://').max_retries
    assert policy.total==2 and 'POST' in policy.allowed_methods and policy.is_retry('POST',503)
    http.close()
