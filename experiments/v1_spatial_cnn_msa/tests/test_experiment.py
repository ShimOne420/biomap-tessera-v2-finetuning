# Scopo: verifiche significative di geometria, maschere, gradienti, ripresa e CV V1 spatial.
# Fasi: tile sintetiche reali -> indice -> DataLoader -> CNN/MLP/HistGB -> valutazione e checkpoint.
# Input: fixture temporanee; output: test e CSV/checkpoint temporanei, nessuna metrica scientifica.
# Parametri: python -m pytest experiments/v1_spatial_cnn_msa/tests; CPU, no download.
# Risorse: 17 AOI sintetiche + 2 celle liguri, pochi MB; encoder e processi V2 assenti.
# Ripresa: interruzioni simulate verificano rollback e riuso; hash sorgenti controllati.
from copy import deepcopy
import json
from pathlib import Path
import sqlite3
import numpy as np
import pandas as pd
import pytest
import torch
from runtime import Progress,cache_path,membership,configuration,common_parser,digest,exclusive
from sources import inventory,expected_tiles,inspect_tile
from patch_builder import build,fragments
from dataset import cell_frame,metadata,verify_sources,fit_normalizer,SpatialDataset,collate_cells
from model import SpatialCNN,MeanMLP
from evaluate import metrics,macro_rmse,compare


def test_geometry_index_matches_native_pixels_and_means(cache):
    cfg,frame,path=cache;f=cell_frame(path);verify_sources(path,True)
    ds=SpatialDataset(path,f,{'mean':np.zeros(128,dtype=np.float32),'std':np.ones(128,dtype=np.float32)})
    try:
        assert len(f)==36 and set(f[f.partition=='train'].cv_fold)=={0,1,2}
        for i in [0,1,20,35]:
            item=ds[i];values=np.concatenate([x.transpose(1,2,0)[m[0]] for x,m,n in item['patches']])
            np.testing.assert_allclose(values.mean(0,dtype=np.float64),f.iloc[i]['mean'],atol=1e-6)
            assert sum(n for _,_,n in item['patches'])==f.iloc[i].n_valid
            assert item['target']==f.iloc[i].target
    finally:ds.close()


def test_masks_have_correct_cell_membership_and_georeferencing(synthetic):
    from rasterio import Affine
    from pyproj import Transformer
    cfg,frame=synthetic;tiles,missing,bad=inventory(cfg,frame)
    assert not missing and not bad
    tile=tiles[0];part=frame[frame.aoi_id==tile['aoi_id']].reset_index(drop=True)
    for ci,(r,c,h,w),mask,total in fragments(tile,part,64):
        rr,cc=np.nonzero(mask);x,y=Affine(*tile['transform'])*(cc+c+.5,rr+r+.5);lon,lat=Transformer.from_crs(tile['crs'],4326,always_xy=True).transform(x,y)
        cell=part.iloc[ci]
        assert ((lon>=cell.west)&(lon<cell.east)&(lat>cell.south)&(lat<=cell.north)).all()
        assert total>=int(mask.sum())


def test_padding_values_and_padding_extent_do_not_change_predictions():
    options={'conv_channels':[8,4],'hidden':8,'dropout':0.,'pooling':'mean','output':'linear'}
    torch.manual_seed(4);model=SpatialCNN(options).eval()
    x=torch.randn(1,128,5,7);mask=torch.ones(1,1,5,7,dtype=torch.bool);mask[:,:,2,3]=False
    a={'features':x,'mask':mask,'owners':torch.tensor([0]),'weights':mask.sum().reshape(1).float(),'target':torch.tensor([.5])}
    b={**a,'features':x.masked_fill(~mask,float('nan'))}
    torch.testing.assert_close(model(a),model(b),atol=0,rtol=0)
    xp=torch.full((1,128,11,13),1e20);mp=torch.zeros(1,1,11,13,dtype=torch.bool);xp[:,:,:5,:7]=x;mp[:,:,:5,:7]=mask
    c={**a,'features':xp,'mask':mp}
    torch.testing.assert_close(model(a),model(c),atol=1e-6,rtol=0)


def test_variable_patch_counts_cell_loss_and_backward(cache):
    cfg,frame,path=cache;f=cell_frame(path);norm=fit_normalizer(f[f.partition=='train'])
    ds=SpatialDataset(path,f.head(2),norm)
    a,b=ds[0],ds[1];b['patches']=b['patches']*2
    batch=collate_cells([a,b]);model=SpatialCNN(cfg['model']);pred=model(batch)
    assert pred.shape==(2,) and len(batch['owners'])==3
    model.eval()
    one=collate_cells([a,{**b,'patches':b['patches'][:1]}])
    torch.testing.assert_close(model(batch),model(one),rtol=1e-5,atol=1e-6)
    model.train()
    before=model.first.conv.weight.detach().clone();opt=torch.optim.AdamW(model.parameters(),lr=.001)
    loss=torch.nn.functional.mse_loss(pred,batch['target']);loss.backward()
    assert torch.isfinite(model.first.conv.weight.grad).all();opt.step()
    assert not torch.equal(before,model.first.conv.weight)
    ds.close()


def test_invalid_patch_rejected():
    model=SpatialCNN({'conv_channels':[8,4],'hidden':8,'dropout':0.,'pooling':'mean','output':'linear'})
    with pytest.raises(ValueError,match='vuota'):model({'features':torch.zeros(1,128,3,3),'mask':torch.zeros(1,1,3,3,dtype=torch.bool),'owners':torch.tensor([0]),'weights':torch.tensor([0.]),'target':torch.tensor([.5])})


def test_norm_never_fits_external_targets(cache):
    cfg,frame,path=cache;f=cell_frame(path)
    with pytest.raises(ValueError,match='training'):fit_normalizer(f)
    train=f[(f.partition=='train')&(f.cv_fold!=0)];norm=fit_normalizer(train)
    assert set(norm['training_ids'])==set(train.cell_id)
    assert not set(norm['training_ids'])&set(f[f.partition=='test'].cell_id)


def test_original_manifest_counts_and_geographic_folds(tmp_path):
    p=common_parser('test');args=p.parse_args(['--data-root',str(tmp_path),'--preset','full']);cfg=configuration(args)
    f=membership(cfg);assert len(f)==132900 and (f.partition=='train').sum()==127909
    for fold in range(3):
        d=f[f.partition=='train'];assert not set(d[d.cv_fold==fold].aoi_id)&set(d[d.cv_fold!=fold].aoi_id)
    args.preset='pilot';assert len(membership(configuration(args)))==1000
    args.preset='intermediate';assert len(membership(configuration(args)))==10000


def test_source_and_index_immutability(cache):
    cfg,frame,path=cache;tile=next(Path(cfg['data']['v1_root']).glob('*/tessera/2020/global_0.1_degree_representation/2020/*/*_scales.npy'))
    before=digest(tile);p=Progress(cfg,'resume')
    try:assert build(cfg,frame,p)==path
    finally:p.close()
    assert digest(tile)==before
    with path.open('ab') as stream:stream.write(b'changed')
    with pytest.raises(ValueError,match='Checksum'):metadata(path)


def test_tile_transaction_rollback_and_resume(synthetic,monkeypatch):
    import patch_builder
    cfg,frame=synthetic;original=patch_builder.fragments;seen=[]
    def interrupted(tile,*args):
        for value in original(tile,*args):
            seen.append(tile['name']);yield value
            if len(seen)==1:raise KeyboardInterrupt()
    monkeypatch.setattr(patch_builder,'fragments',interrupted);p=Progress(cfg,'interrupt')
    try:
        with pytest.raises(KeyboardInterrupt):build(cfg,frame,p)
    finally:p.close('interrupted')
    with sqlite3.connect(cache_path(cfg)) as db:assert db.execute('SELECT count(*) FROM patches').fetchone()[0]==0
    monkeypatch.setattr(patch_builder,'fragments',original);p=Progress(cfg,'resume')
    try:build(cfg,frame,p)
    finally:p.close()
    assert len(cell_frame(cache_path(cfg)))==36


def test_configuration_changes_invalidate_cache(cache):
    cfg,frame,path=cache;cfg=deepcopy(cfg);cfg['data']['max_patch_side']=32;p=Progress(cfg,'changed')
    try:
        with pytest.raises(ValueError,match='diverso'):build(cfg,frame,p)
    finally:p.close('failed')


def test_preflight_ready_states_and_no_acquisition(cache,monkeypatch):
    import preflight
    cfg,frame,path=cache;monkeypatch.setattr(preflight,'membership',lambda c:frame)
    ready=preflight.inspect(cfg);assert ready['status']=='READY_FOR_TRAINING'
    fresh=deepcopy(cfg);fresh['preset']='intermediate'
    assert preflight.inspect(fresh)['status']=='READY_FOR_EXTRACTION'
    fresh['data']['targets_csv']='missing.csv';assert preflight.inspect(fresh)['status']=='BLOCKED'


def test_metrics_zero_targets_and_macro_distinction():
    m=metrics([0,.5,1],[.1,.4,.8]);assert m['mape_n']==2 and m['mape_excluded_zero']==1
    f=pd.DataFrame({'aoi_id':['a','b','b','b'],'target':[0.,0.,0.,0.]})
    pred=np.array([1.,0.,0.,0.]);assert macro_rmse(f,pred)==.5
    assert metrics(f.target,pred)['rmse']==.5
    # Different group errors make macro and pooled RMSE differ.
    pred=np.array([1.,.2,.2,.2]);assert not np.isclose(macro_rmse(f,pred),metrics(f.target,pred)['rmse'])


def test_complete_cv_refit_predict_and_resume(cache):
    from train import train,run_directory,predict_external,load_run
    cfg,frame,path=cache;run=run_directory(cfg,'synthetic');p=Progress(cfg,'training')
    try:train(cfg,run,'all',False,p)
    finally:p.close()
    selection=json.loads((run/'selection.json').read_text());assert selection['liguria_used'] is False
    oof=pd.read_csv(run/'oof_predictions.csv',dtype={'cell_id':str})
    assert set(oof.partition)=={'train'} and not oof.duplicated(['cell_id','model']).any()
    assert len(oof)==34*4
    p=Progress(cfg,'resume')
    try:train(cfg,run,'all',True,p)
    finally:p.close()
    result=predict_external(cfg,run,load_run(cfg,run));assert len(result)==4
    external=pd.read_csv(run/'liguria_predictions.csv');assert len(external)==2*4
    compare(run);assert (run/'liguria_comparison.csv').exists()
    checkpoint=run/'spatial_cnn/refit/best.pt';saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
    assert not any('liguria' in cid for cid in saved['normalizer']['training_ids'])


def test_source_version_mismatch_blocks(synthetic):
    cfg,frame=synthetic;p=next(Path(cfg['data']['v1_root']).glob('*/tessera/2020/tessera_metadata.json'));p.write_text(json.dumps({'dataset_version':'1.1','dataset_variant':'cambridge','year':2020}))
    tiles,missing,bad=inventory(cfg,frame);assert bad


def test_experiment_does_not_write_session_a_or_shared_code():
    import subprocess
    from runtime import REPO
    result=subprocess.run(['git','diff','--name-only','--','src','scripts','configs','experiments/v2_dense_cnn_msa'],cwd=REPO,capture_output=True,text=True,check=True)
    assert not result.stdout.strip()


def test_neural_resume_matches_uninterrupted_weights(cache,tmp_path):
    from train import train_neural
    cfg,frame,path=cache;cfg=deepcopy(cfg);cfg['training']['epochs']=2
    f=cell_frame(path);development=f[f.partition=='train'].reset_index(drop=True)
    p=Progress(cfg,'resume_exact')
    try:
        train_neural(cfg,path,development,None,'mean_mlp',tmp_path/'uninterrupted','same-identity',p,2,False)
        train_neural(cfg,path,development,None,'mean_mlp',tmp_path/'interrupted','same-identity',p,1,False)
        train_neural(cfg,path,development,None,'mean_mlp',tmp_path/'interrupted','same-identity',p,2,True)
    finally:p.close()
    a=torch.load(tmp_path/'uninterrupted/last.pt',weights_only=False);b=torch.load(tmp_path/'interrupted/last.pt',weights_only=False)
    for key in a['model']:torch.testing.assert_close(a['model'][key],b['model'][key],rtol=0,atol=0)


def test_save_load_and_deterministic_inference(cache,tmp_path):
    from train import seed_all,atomic_torch
    cfg,frame,path=cache;f=cell_frame(path);norm=fit_normalizer(f[f.partition=='train']);ds=SpatialDataset(path,f.head(2),norm)
    batch=collate_cells([ds[0],ds[1]]);seed_all(42);model=SpatialCNN(cfg['model']).eval()
    expected=model(batch);atomic_torch(tmp_path/'model.pt',{'weights':model.state_dict()})
    other=SpatialCNN(cfg['model']).eval();other.load_state_dict(torch.load(tmp_path/'model.pt',weights_only=False)['weights'])
    torch.testing.assert_close(other(batch),expected,rtol=0,atol=0);torch.testing.assert_close(model(batch),expected,rtol=0,atol=0);ds.close()


def test_partial_coverage_is_explicit_and_baselines_share_it(synthetic):
    cfg,frame=synthetic;scale=next(Path(cfg['data']['v1_root']).glob('aoi_00/tessera/2020/global_0.1_degree_representation/2020/*/*_scales.npy'))
    values=np.load(scale);values[:]=np.nan;np.save(scale,values);p=Progress(cfg,'coverage')
    try:
        with pytest.raises(ValueError,match='non eleggibili'):build(cfg,frame,p)
    finally:p.close('failed')
    cfg['data']['allow_partial_cells']=True;p=Progress(cfg,'partial_explicit')
    try:path=build(cfg,frame,p)
    finally:p.close()
    f=cell_frame(path);assert len(f)==34 and 'aoi_00' not in set(f.aoi_id)
    report=json.loads((path.parent/'cache_report.json').read_text());assert report['excluded']==2


def test_http_range_resume_and_content_hash(synthetic,monkeypatch,tmp_path):
    import hashlib
    import sources
    from runtime import Budget
    cfg,frame=synthetic;blob=b'V1-test-content-123456789';etag='"'+hashlib.md5(blob).hexdigest()+'"'
    dest=Path(cfg['work_root'])/'sources/test.npy';dest.parent.mkdir(parents=True)
    partial=dest.with_suffix('.npy.partial');partial.write_bytes(blob[:7]);partial.with_suffix('.partial.json').write_text(json.dumps({'etag':etag,'bytes':len(blob)}))
    class Response:
        status_code=206;headers={'Content-Range':f'bytes 7-{len(blob)-1}/{len(blob)}'}
        def raise_for_status(self):pass
        def iter_content(self,n):yield blob[7:]
        def __enter__(self):return self
        def __exit__(self,*args):pass
    class Http:
        def mount(self,*args):pass
        def head(self,*args,**kwargs):
            class Head:
                headers={'Content-Length':str(len(blob)),'ETag':etag}
                def raise_for_status(self):pass
            return Head()
        def get(self,*args,**kwargs):assert kwargs['headers']['Range']=='bytes=7-';return Response()
        def close(self):pass
    monkeypatch.setattr(sources,'Session',Http);p=Progress(cfg,'download')
    try:sources.download('https://example.test/tile',dest,Budget(cfg),p)
    finally:p.close()
    assert dest.read_bytes()==blob and json.loads(dest.with_suffix('.npy.download.json').read_text())['sha256']==hashlib.sha256(blob).hexdigest()


def test_profile_measures_batches_and_writes_budget(cache):
    from train import profile
    cfg,frame,path=cache;p=Progress(cfg,'profile_test')
    try:profile(cfg,path,p)
    finally:p.close()
    result=json.loads((Path(cfg['work_root'])/'profile_training.json').read_text())
    assert [r['batch_cells'] for r in result['trials']]==[1,2,4,8]
    assert result['largest_tested_batch_cells']==8
    assert result['estimated_train_epoch_seconds']>0
    assert all(r['status']=='passed' and r['peak_allocated_gib'] is None for r in result['trials'])


def test_monitor_is_read_only_and_reports_eta(tmp_path):
    import subprocess
    import sys
    from datetime import datetime,timezone
    from runtime import HERE
    root=tmp_path/'v1_spatial_msa';root.mkdir()
    now=datetime.now(timezone.utc).isoformat()
    state={'started_utc':now,'updated_utc':now,'status':'running','phase':'training','preset':'pilot',
        'done':2,'total':10,'unit':'jobs','eta_seconds':120,'elapsed_seconds':30,'rate_per_second':.1}
    path=root/'status.json';path.write_text(json.dumps(state));before=path.read_bytes()
    result=subprocess.run([sys.executable,str(HERE/'monitor.py'),'--work-root',str(root)],capture_output=True,text=True,check=True)
    assert '00:02:00' in result.stdout and '2/10 jobs' in result.stdout and '0.100 jobs/s' in result.stdout
    assert path.read_bytes()==before and list(root.iterdir())==[path]
