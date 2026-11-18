# Scopo: contratto CPU/offline di geometria, cache e nested, senza encoder o dataset universitari.
# Fasi: griglie sintetiche/indice r2 fixture -> manifest/cache -> fit/refit/decisioni -> report e guardie.
# Input: tmp_path e piccoli array; output: assert e artefatti privati pytest, nessun benchmark scientifico.
# Parametri: quattro geometrie, cross-UTM, seed/epoche/budget; uso python -m pytest -ra.
# Risorse: CPU singolo thread, worker spawn verificato; ripresa esatta anche dopo interruzioni simulate.
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
import pytest
import torch
from pyproj import Transformer
from biomap_spatial.sources import V2Source,prepare_index,R2
from biomap_spatial.batching import collate_cells,CellDataset
from biomap_spatial.normalization import fit_normalizer
from biomap_spatial.model import SpatialCNN
from biomap_spatial.contracts import ModelConfig
from biomap_sampling import geometry as geo,data,training,reporting
from biomap_sampling.state import load_config,ROOT,digest,load_checkpoint,BudgetExpired
from biomap_sampling.study import Study,nested_pairs
from biomap_sampling.metrics import metrics,compare
from spatial_fixtures import make_artifacts


def patch(h=31,w=23,tile='a/grid',crs='EPSG:32632',x=500000.,y=4800000.):
    return {'tile':tile,'row':0,'col':0,'mask':np.ones((h,w),bool),
        'grid':{'height':h,'width':w,'transform':[10.,0.,x,0.,-10.,y],'crs':crs}}


@pytest.fixture
def ready(tmp_path):
    torch.set_num_threads(1);spec,ref,work,out=make_artifacts(tmp_path,protocol=R2)
    source=V2Source(spec,out/'index',False);prepare_index(source)
    cfg=load_config(ROOT/'configs/spatial_sampling_benchmark.yaml')
    cfg['paths'].update(pixel_root=str(work),reference=str(ref),index_root=str(out/'index'),cache_root=str(tmp_path/'cache'),output_root=str(tmp_path/'study'))
    cfg['sampling']['subset_cells']=3;cfg['cache']['cells_per_shard']=2
    cfg['training'].update(epochs=1,patience=1,num_workers=0,batch_cells=2,accumulate_cells=4)
    cfg['execution']['reserve_gib']=0;cfg['decision']['bootstrap_samples']=20
    try:yield source,cfg
    finally:source.close()


@pytest.mark.parametrize('mode,count',[('S64',64),('M128',128),('S144',144),('M64',64)])
def test_sampler_deterministic_native_and_no_target(mode,count):
    p=patch();lon,lat=Transformer.from_crs(p['grid']['crs'],4326,always_xy=True).transform(500110,4799850)
    a=geo.select([p],mode,'cell',lon,lat);b=geo.select([p],mode,'cell',lon,lat)
    assert a==b and a['diagnostics']['selected_pixels']==count
    locations=[]
    for item in a['patches']:
        rr,cc=np.where(geo.unpack(item));rr+=item['row'];cc+=item['col']
        assert ((rr>=0)&(rr<31)&(cc>=0)&(cc<23)).all()
        locations.extend(zip(rr,cc))
    assert len(set(locations))==count


@pytest.mark.parametrize('side,n',[(8,16),(12,64),(4,0)])
def test_actual_rf(side,n):
    assert geo.rf_positions(np.ones((side,side),bool))==n
    mask=np.ones((side,side),bool);mask[side//2,side//2]=False
    assert geo.rf_positions(mask)<=n


def test_overlapping_tiles_do_not_double_sampling():
    p=patch();q=deepcopy(p);q['tile']='a/overlap'
    result=geo.select([p,q],'M128','cell',9,43)
    positions=[]
    for part in result['patches']:
        rr,cc=np.where(geo.unpack(part));x,y=geo.xy(part['grid'],rr+part['row'],cc+part['col']);positions.extend(zip(x,y))
    assert len(positions)==128 and len(set(positions))==128
    assert result['diagnostics']['historical_contributions']==1426


def test_small_irregular_cell_is_retained_and_empty_fails():
    p=patch(1,1);r=geo.select([p],'M128','tiny',9,43)
    assert r['diagnostics']['selected_pixels']==1 and r['diagnostics']['insufficient']
    assert r['diagnostics']['partial_mask'] and len(r['patches'])==1
    p['mask'][:]=False
    with pytest.raises(ValueError,match='senza supporto'):geo.select([p],'S64','empty',9,43)


def test_multi_utm_footprints_not_resampled_or_duplicated():
    p=patch(12,12,x=738000,y=5100000);x,y=Transformer.from_crs('EPSG:32632','EPSG:32633',always_xy=True).transform(738000,5100000)
    q=patch(12,12,tile='a/other',crs='EPSG:32633',x=x,y=y)
    result=geo.select([p,q],'M128','cross',12,46)
    assert result['diagnostics']['multi_utm'] and result['diagnostics']['selected_pixels']<=128
    # Every selected native pixel has positive area disjoint from previously selected footprint polygons.
    occupied=[]
    for item in result['patches']:
        dense={'mask':geo.unpack(item),'row':item['row'],'col':item['col'],'grid':item['grid'],'tile':item['tile']}
        assert np.array_equal(geo.available(dense,occupied),dense['mask'])
        occupied.extend(geo.footprints(item))
    assert p['grid']['crs']=='EPSG:32632' and q['grid']['crs']=='EPSG:32633'


def test_manifest_cache_roundtrip_and_source_integrity(ready):
    source,cfg=ready;before={str(p):digest(p) for r in [source.root,source.reference,source.output] for p in r.rglob('*') if p.is_file()}
    manifest=data.prepare(source,cfg);data.cache(source,cfg)
    assert manifest['cells']==6 and manifest['summary']['FULL-SUBSET']['cells']==3
    records=[cell for _,cells in data.Plan(cfg['paths']['cache_root']).blocks() for cell in cells]
    from biomap_finetune.data import spatial_sampling_requests
    cached=data.CachedSource(cfg['paths']['cache_root'],'S64')
    try:
        for cell in records:
            request=spatial_sampling_requests(cell,'S64');assert not request['approved_for_acquisition']
            assert data.verify_requests(request,source.reference)
            original=source.patches(cell['cell_id']);actual=cached.patches(cell['cell_id'])
            for p,x in zip(cell['modes']['S64']['patches'],actual):
                rr,cc=np.where(geo.unpack(p));base=original[p['source_patch']]
                expected=base.features[:,rr+p['row']-base.metadata['row'],cc+p['col']-base.metadata['col']]
                assert np.array_equal(x.features[:,x.mask],expected)
            for p in request['patches']:
                positions=np.asarray(request['groups'][p['group']]['positions'])[p['take']]
                assert np.array_equal(positions,p['positions']) and int(geo.unpack(p).sum())==p['weight']
        assert data.prepare(source,cfg,resume=True)['identity']==manifest['identity']
        data.cache(source,cfg,resume=True)
    finally:cached.close()
    assert before=={name:digest(Path(name)) for name in before}


def test_actual_original_cnn_cache_pooling_and_no_targets(ready):
    source,cfg=ready;data.prepare(source,cfg);data.cache(source,cfg);sample=data.CachedSource(cfg['paths']['cache_root'],'FULL-SUBSET')
    try:
        norm=fit_normalizer(sample.frame,'pixel_weighted_standard');m=SpatialCNN(ModelConfig()).eval()
        batch=collate_cells([CellDataset(sample,sample.frame,norm)[0]]);batch.pop('target')
        with torch.no_grad():assert m(batch).shape==(1,)
        cell=sample.frame.iloc[:1]
        with torch.no_grad():
            a=m(next(iter(training.loader(sample,cell,norm,cfg))))
            b=m(next(iter(training.loader(source,cell,norm,cfg))))
            torch.testing.assert_close(a,b,rtol=0,atol=0)
    finally:sample.close()


def test_spawn_worker_uses_read_only_cache(ready):
    source,cfg=ready;data.prepare(source,cfg);data.cache(source,cfg);cached=data.CachedSource(cfg['paths']['cache_root'],'S64')
    cfg['training']['num_workers']=1
    try:
        norm=fit_normalizer(cached.frame,'pixel_weighted_standard')
        assert sum(len(b['cell_ids']) for b in training.loader(cached,cached.frame,norm,cfg))==6
    finally:cached.close()


@pytest.mark.parametrize('a,b,result',[(2,3,3),(2,8,5),(1,2,2),(1,1,1)])
def test_half_up_epochs(a,b,result):assert training.refit_epochs(a,b)==result


def test_fresh_refit_and_epoch_resume_are_exact(ready,tmp_path):
    source,cfg=ready;frame=data.development(source);train=frame[frame.cv_fold==0];val=frame[frame.cv_fold==1]
    cfg['training']['epochs']=2
    a,_,_=training.fit(source,source,train,None,cfg,tmp_path/'whole','id',42,epochs=2)
    training.fit(source,source,train,None,cfg,tmp_path/'resumed','id',42,epochs=1)
    b,_,_=training.fit(source,source,train,None,cfg,tmp_path/'resumed','id',42,True,epochs=2)
    for key,v in a.state_dict().items():torch.testing.assert_close(v,b.state_dict()[key],rtol=0,atol=0)
    _,_,epoch=training.fit(source,source,train,val,cfg,tmp_path/'inner','inner',42)
    training.fit(source,source,frame[frame.cv_fold!=2],None,cfg,tmp_path/'refit','refit',42,epochs=training.refit_epochs(epoch,epoch))
    first=json.loads((tmp_path/'inner/initialization.json').read_text());second=json.loads((tmp_path/'refit/initialization.json').read_text())
    assert first['sha']==second['sha'] and first['source']=='fresh_seeded_initialization'
    assert first['training_ids_sha256']!=second['training_ids_sha256']


def test_budget_interrupt_preserves_epoch_and_resume(ready,tmp_path):
    source,cfg=ready;frame=data.development(source);train=frame[frame.cv_fold==0]
    with pytest.raises(BudgetExpired):training.fit(source,source,train,None,cfg,tmp_path/'fit','id',42,deadline=time.monotonic()-1)
    model,_,_=training.fit(source,source,train,None,cfg,tmp_path/'fit','id',42,True)
    assert (tmp_path/'fit/best.json').exists()


def test_nested_splits_and_no_leakage(ready):
    source,cfg=ready;frame=data.development(source)
    for outer in range(3):
        for a,b in nested_pairs(outer):
            training.split_guard(frame[frame.cv_fold==a],frame[frame.cv_fold==b],frame[frame.cv_fold==outer])
    with pytest.raises(ValueError,match='Leakage'):training.split_guard(frame,frame)
    with pytest.raises(ValueError):training.split_guard(source.frame[source.frame.partition=='test'])


def test_macro_bootstrap_differs_from_pooled_and_exact_population():
    frame=pd.DataFrame({'cell_id':['a','b','c','d'],'aoi_id':['one','two','two','two'],'partition':['train']*4,
        'cv_fold':[0,1,1,1],'target':[.5]*4,'prediction':[.5,.9,.9,.9]})
    m=metrics(frame);assert m['macro_aoi_rmse']!=m['rmse']
    candidate=frame.copy();candidate.loc[0,'prediction']=.7
    result=compare(frame,candidate,40)
    assert result['macro_aoi_delta']!=result['pooled_bootstrap']['rmse_delta']
    assert result==compare(frame,candidate,40)
    with pytest.raises(ValueError,match='Popolazioni'):compare(frame,candidate.iloc[:2])


def test_nested_tiny_end_to_end_no_encoder_no_outer_selection(ready,monkeypatch):
    source,cfg=ready;data.prepare(source,cfg);data.cache(source,cfg)
    # Any accidental pretrained encoder use is fatal even though no checkpoint exists in the fixture.
    import biomap_finetune.encoder as encoder
    monkeypatch.setattr(encoder,'load_medium',lambda *a:pytest.fail('encoder forbidden'))
    study=Study(source,cfg)
    try:
        study.screen();decision=study.confirm(0)
        assert decision['outer_targets_used'] is False and decision['approved_for_acquisition'] is False
        original=(Path(cfg['paths']['output_root'])/'outer_0/decision.json').read_bytes()
        study.external(0)
        assert (Path(cfg['paths']['output_root'])/'outer_0/decision.json').read_bytes()==original
        for p in Path(cfg['paths']['output_root']).glob('outer_0/refit*'):assert p.is_dir()
        for fold in [1,2]:study.confirm(fold)
        recommended=study.recommend()
        assert recommended['approved_for_acquisition'] is False and not recommended['proof_of_noninferiority']
        for fold in [1,2]:study.external(fold)
        result=reporting.report(cfg);assert Path(result['report']).exists()
        budget=reporting.forecast(cfg)
        assert budget['time']['training_epoch_median_seconds']>0
        assert budget['time']['validation_epoch_median_seconds']>0
        assert budget['time']['dense_outer_seconds']>0
        assert budget['storage']['cache_current_bytes_measured']>0
        assert (Path(cfg['paths']['output_root'])/'resources_summary.csv').exists()
    finally:study.close()


def test_cli_default_dry_run_and_campaign_gate(tmp_path,capsys):
    from biomap_sampling.cli import main
    result=main(['doctor','--config',str(ROOT/'configs/spatial_sampling_benchmark.yaml'),'--dry-run'])
    assert result['jobs']['screen_fits']==7 and not result['scientific_campaign_authorized']
    assert not result['sentinel_acquisition_authorized']


def test_sampler_cache_corruption_and_roots(ready):
    source,cfg=ready;data.prepare(source,cfg);data.cache(source,cfg)
    info=json.loads((Path(cfg['paths']['cache_root'])/'cache.json').read_text())
    p=Path(cfg['paths']['cache_root'])/'features'/info['shards'][0]['key']/'features.npy'
    with p.open('ab') as f:f.write(b'corruption')
    with pytest.raises(ValueError,match='Checksum'):data.CachedSource(cfg['paths']['cache_root'],'S64')
    cfg['paths']['cache_root']=str(source.root)
    with pytest.raises(ValueError):data.roots(cfg)


def test_outer_label_changes_do_not_change_internal_selection(ready):
    source,cfg=ready;data.prepare(source,cfg);data.cache(source,cfg);first=Study(source,cfg)
    try:expected=first.confirm(0)
    finally:first.close()
    altered=deepcopy(cfg);altered['paths']['output_root']=str(Path(cfg['paths']['output_root']).with_name('altered_outer_labels'))
    second=Study(source,altered)
    try:
        second.frame.loc[second.frame.cv_fold==0,'target']=.99
        assert second.confirm(0)==expected
    finally:second.close()


def test_missing_outer_seal_and_changed_decision_are_rejected(ready):
    source,cfg=ready;data.prepare(source,cfg);data.cache(source,cfg);study=Study(source,cfg)
    try:
        with pytest.raises(ValueError,match='Sigillare'):study.external(0)
        study.confirm(0);path=Path(cfg['paths']['output_root'])/'outer_0/decision.json'
        record=json.loads(path.read_text());record['selected']='S144';record['outer_targets_used']=True;path.write_text(json.dumps(record))
        with pytest.raises(ValueError,match='sigillata'):study.external(0)
    finally:study.close()


def test_campaign_requires_separate_budget_approval(ready,monkeypatch,tmp_path):
    from biomap_sampling.cli import main
    import yaml
    source,cfg=ready;data.prepare(source,cfg);data.cache(source,cfg)
    path=tmp_path/'config.yaml';path.write_text(yaml.safe_dump(cfg),encoding='utf-8')
    monkeypatch.setattr(data,'open_source',lambda c:source)
    with pytest.raises(ValueError,match='NON autorizzata'):main(['run','--config',str(path)])
    assert not (Path(cfg['paths']['output_root'])/'ownership.json').exists()


def test_extreme_error_thresholds_are_strict_and_constant_r2_is_null():
    frame=pd.DataFrame({'cell_id':['a','b','c'],'aoi_id':['one']*3,'partition':['train']*3,'cv_fold':[0]*3,
        'target':[0.,0.,0.],'prediction':[.3,.4,.5]})
    result=metrics(frame)
    assert result['error_gt_03_fraction']==2/3 and result['error_gt_04_fraction']==1/3
    assert result['r2'] is None and result['pearson'] is None


def test_geometry_hash_is_shared_and_verified_plan_bridge_is_exact(ready,monkeypatch):
    source,cfg=ready;original=data.digest;calls=[]
    def counted(path):
        if Path(path).name=='geometry.npz':calls.append(str(path))
        return original(path)
    monkeypatch.setattr(data,'digest',counted)
    data.prepare(source,cfg)
    assert calls and len(calls)==len(set(calls))
    from biomap_finetune.data import spatial_sampling_requests_from_plan
    plan=data.Plan(cfg['paths']['cache_root']);cell=plan.frame.cell_id.iloc[0]
    result=spatial_sampling_requests_from_plan(plan,cell,'M128',source.reference)
    assert result['plan_identity']==plan.manifest['identity']
    assert len({p['patch_id'] for p in result['patches']})==len(result['patches'])
    assert all(isinstance(p['source_index_patch_id'],int) for p in result['patches'])
