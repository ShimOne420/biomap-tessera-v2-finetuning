# Scopo: verificare scientificamente e operativamente la pipeline senza download/GPU.
# Fasi: fixture sintetiche esplicite -> parità ufficiale -> pooling/gradienti -> resume/probing.
# Input: piccoli tensori e manifest temporanei; output: risultati pytest e run nella tmpdir.
# Parametri: python -m pytest; modello ridotto solo nei test, mai benchmark scientifico.
# Risorse: CPU, nessuna rete o credenziale; Torch usa un thread per tempi riproducibili.
# Ripresa: simula un'interruzione dopo aggiornamenti non salvati e confronta il run ripreso.
import argparse
import copy
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import torch
from biomap_v2.common import ROOT, budget, cells, config, digest, metrics, save_json
from biomap_v2.data import PROTOCOL, cell_directory, power_to_storage, prepared, sample_points
from biomap_v2.encoder import Regression, groups, pixel_embeddings, trainable
from biomap_v2.reference import encode_pixels
from biomap_v2.student import PixelStudent
from biomap_v2 import experiments

torch.set_num_threads(1)


def raw_fixture(n=4):
    rng=np.random.default_rng(3)
    return {'s2':rng.integers(100,8000,(n,13,10)).astype(np.uint16),
        'mask':np.array([[1]*8+[0]*5,[1]*13,[1]*10+[0]*3,[1]*9+[0]*4][:n],dtype=bool),
        's2_doys':np.arange(13,dtype=np.int16)*20+1,
        'asc':rng.integers(3000,8000,(n,7,2)).astype(np.int16),'asc_doys':np.arange(7,dtype=np.int16)*30+1,
        'desc':rng.integers(2000,7000,(n,10,2)).astype(np.int16),'desc_doys':np.arange(10,dtype=np.int16)*30+1}


def tiny_model():
    return PixelStudent(latent_dim=8,num_layers=1,nhead=4,dim_feedforward=32,dropout=0)


def test_official_normalization_binning_and_pooling():
    model=tiny_model().eval();raw=raw_fixture()
    raw['asc'][0,0]=0  # validity must be computed before normalizing
    expected=encode_pixels(model,raw['s2'],raw['s2_doys'],raw['asc'],raw['asc_doys'],raw['desc'],raw['desc_doys'],raw['mask'],device=torch.device('cpu'))
    actual=pixel_embeddings(model,raw,pixel_batch=2)
    np.testing.assert_allclose(actual.detach().numpy(),expected,rtol=2e-5,atol=2e-5)
    np.testing.assert_allclose(actual.mean(0).detach().numpy(),expected.mean(0),rtol=2e-5,atol=2e-5)


def test_radar_unit_conversion():
    amplitude=np.array([0.,np.nan,.01,.1,1.,10.],dtype=np.float32)
    expected=np.zeros_like(amplitude,dtype=np.int16);valid=np.isfinite(amplitude)&(amplitude>0)
    expected[valid]=np.clip((20*np.log10(amplitude[valid])+50)*200,0,32767).astype(np.int16)
    np.testing.assert_array_equal(power_to_storage(amplitude**2),expected)


def test_gradients_and_frozen_control():
    raw=raw_fixture();model=Regression(tiny_model())
    for stage in ['head','partial','full']:
        names=trainable(model,stage,1)
        before={k:v.clone() for k,v in model.encoder.state_dict().items()}
        optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.001)
        optimizer.zero_grad();embedding=pixel_embeddings(model.encoder,raw).mean(0)
        (model.head(embedding).squeeze()-.25).square().backward()
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        optimizer.step();changed=any(not torch.equal(v,before[k]) for k,v in model.encoder.state_dict().items())
        assert changed == (stage!='head')
        assert names


def test_missing_and_doy_rejected():
    raw=raw_fixture();raw['s2_doys'][0]=366
    with pytest.raises(ValueError,match='DOY'):list(groups(raw))
    raw=raw_fixture();raw['mask'][:]=False;raw['asc'][:]=0;raw['desc'][:]=0
    with pytest.raises(ValueError,match='entrambe'):list(groups(raw))


def test_metrics_zero_and_constant():
    m=metrics([0,.5,1],[0,.4,.9])
    assert m['n']==3 and m['mape_n']==2 and m['mape_excluded_zero']==1
    assert metrics([.2,.2],[.3,.3])['pearson'] is None


def test_geometries_and_storage(tmp_path):
    args=argparse.Namespace(config=None,data_root=str(tmp_path),set=[])
    cfg=config('msa',args);f=cells(cfg)
    assert f.partition.value_counts().to_dict()=={'train':1000,'test':245,'validation':122}
    points=sample_points(next(f.itertuples()),25)
    assert points.shape==(25,2)
    cfg['storage']['max_gib']=0
    with pytest.raises(RuntimeError,match='Budget'):budget(cfg,1)
    duplicate=pd.concat([f.head(1),f.head(1)])
    duplicate.to_csv(tmp_path/'bad.csv',index=False);cfg['cells']=str(tmp_path/'bad.csv')
    with pytest.raises(ValueError,match='duplicate'):cells(cfg)


@pytest.fixture
def synthetic(tmp_path):
    args=argparse.Namespace(config=None,data_root=str(tmp_path/'data'),set=['model.device=cpu'])
    cfg=config('msa',args);cfg['storage']['reserve_gib']=0;cfg['training'].update(epochs=2,patience=8,seeds=[42],mixed_precision=False,activation_checkpointing=False,batch_cells=2,effective_batch_cells=4)
    cfg['model']['pixel_batch']=2
    model=tiny_model();checkpoint=tmp_path/'synthetic.pt'
    model_args={'repr_dim':128,'latent_dim':8,'num_layers':1,'nhead':4,'dim_feedforward':32,'enable_qk_norm':False}
    torch.save({'model':model.state_dict(),'args':model_args},checkpoint);cfg['model']['checkpoint']=str(checkpoint)
    rows=[]
    for i in range(9):
        part=['train','validation','test'][i//3];cid=str(i)
        row={'cell_id':cid,'lon':8.1+i*.2,'lat':44.,'west':8.1+i*.2-.001,'east':8.1+i*.2+.001,
             'south':43.999,'north':44.001,'partition':part,'aoi_id':part,'target':.15+i*.07,'valid_fraction':1.}
        d=cell_directory(cfg,cid);d.mkdir(parents=True)
        raw=raw_fixture();raw['s2']=raw['s2']+i*100
        np.savez_compressed(d/'series.npz',**raw)
        row['series_sha256']=digest(d/'series.npz');rows.append(row)
    f=pd.DataFrame(rows);source=tmp_path/'source.csv';f.drop(columns=['target','valid_fraction','series_sha256']).to_csv(source,index=False);cfg['cells']=str(source)
    path=Path(cfg['data_root'])/'msa/cells.csv';f.to_csv(path,index=False)
    save_json(path.with_suffix('.json'),{'source_cells_sha256':digest(source),'eligible_sha256':digest(path),
        'config_sampling':cfg['sampling'],'year':cfg['year']})
    return cfg,f


def test_prepare_fingerprint(synthetic):
    cfg,f=synthetic;assert len(prepared(cfg))==9
    cfg=copy.deepcopy(cfg);cfg['sampling']['pixels_per_cell']=49
    with pytest.raises(ValueError,match='Sampling'):prepared(cfg)


def test_probe_and_progressive_training(synthetic):
    cfg,f=synthetic
    args=argparse.Namespace(smoke=False,checkpoint=None,resume=None,stage='all')
    probe=experiments.probe(cfg,args)
    preds=pd.read_csv(probe/'predictions.csv')
    assert set(preds.model)=={'DummyMean','Ridge','HistGB','XGBoost'}
    assert preds.groupby('model').cell_id.nunique().eq(9).all()
    out=experiments.train(cfg,args)
    selection=json.loads((out/'selection.json').read_text())
    assert 'head_control_seed42' in selection['candidates'] and 'partial_seed42' in selection['candidates']
    frozen=torch.load(out/'head_control_seed42/best.pt',weights_only=False)
    adapted=torch.load(out/'partial_seed42/best.pt',weights_only=False)
    assert frozen['encoder_changed'] is False and adapted['encoder_changed'] is True
    args.run_dir=str(out);experiments.evaluate(cfg,args)


def test_resume_matches_uninterrupted(synthetic,tmp_path,monkeypatch):
    cfg,f=synthetic
    model,_,_=experiments.phase(cfg,f,tmp_path/'full','head',42)
    expected={k:v.clone() for k,v in model.state_dict().items()}
    original=experiments.predict;calls=0
    def interrupted(model,frame,cfg):
        nonlocal calls
        calls+=1
        if calls==2:raise RuntimeError('simulated interruption during epoch 2')
        return original(model,frame,cfg)
    monkeypatch.setattr(experiments,'predict',interrupted)
    with pytest.raises(RuntimeError,match='simulated'):experiments.phase(cfg,f,tmp_path/'broken','head',42)
    monkeypatch.setattr(experiments,'predict',original)
    resumed,_,_=experiments.phase(cfg,f,tmp_path/'broken','head',42,resume=tmp_path/'broken/last.pt')
    for k,v in resumed.state_dict().items():torch.testing.assert_close(v,expected[k],rtol=0,atol=0)
    cfg=copy.deepcopy(cfg);cfg['training']['encoder_lr']=.01
    with pytest.raises(ValueError,match='Resume incompatibile'):
        experiments.phase(cfg,f,tmp_path/'broken','head',42,resume=tmp_path/'broken/last.pt')


def test_prepare_resume_and_target_alignment(tmp_path,monkeypatch):
    import rasterio
    from rasterio.transform import from_origin
    from biomap_v2 import data
    args=argparse.Namespace(config=None,data_root=str(tmp_path/'data'),set=['storage.reserve_gib=0'],limit=None,target_raster=str(tmp_path/'msa.tif'),import_cache=None)
    cfg=config('msa',args);res=1/360
    frame=pd.DataFrame([{'cell_id':str(i),'lon':8+(i+.5)*res,'lat':44-res/2,
        'west':8+i*res,'east':8+(i+1)*res,'south':44-res,'north':44,
        'partition':part,'aoi_id':part} for i,part in enumerate(['train','validation','test'])])
    source=tmp_path/'geometry.csv';frame.to_csv(source,index=False);cfg['cells']=str(source)
    with rasterio.open(args.target_raster,'w',driver='GTiff',height=1,width=3,count=1,dtype='float32',crs='EPSG:4326',transform=from_origin(8,44,res,res),nodata=-9999) as ds:
        ds.write(np.array([[.2,.5,.8]],dtype=np.float32),1)
    cfg['sampling']['min_s2_observations']=1
    def fake_query(cfg,cell,collection,cache):
        items=[{'id':'test-'+collection,'properties':{'datetime':'2020-06-01T00:00:00Z','sat:orbit_state':'ascending'},'assets':{}}]
        save_json(cache,items);return items
    def fake_values(self,item,band,points):return np.full(len(points),4 if band=='SCL' else (.1 if band in ['vv','vh'] else 3000))
    monkeypatch.setattr(data,'query',fake_query);monkeypatch.setattr(data.Reader,'values',fake_values)
    path=data.prepare(cfg,args);f=prepared(cfg)
    assert len(f)==3 and np.allclose(f.target,[.2,.5,.8])
    # Resume must not read a single COG again.
    def fail_read(*args):raise AssertionError('Unexpected network read on resume')
    monkeypatch.setattr(data.Reader,'values',fail_read)
    assert data.prepare(cfg,args)==path
    # A corrupt complete cell must fail, not silently choose a different cell.
    cell_directory(cfg,'0').joinpath('series.npz').write_bytes(b'corrupt')
    with pytest.raises(ValueError,match='corrotta'):data.prepare(cfg,args)


def test_oom_retries_epoch_with_same_effective_batch(synthetic,tmp_path,monkeypatch):
    cfg,f=synthetic;cfg['training']['epochs']=1
    model,_,_=experiments.phase(cfg,f,tmp_path/'baseline','head',42)
    expected={k:v.clone() for k,v in model.state_dict().items()}
    original=Regression.forward_cell;counter=0
    def fail_once(self,path,cfg):
        nonlocal counter
        counter+=1
        if counter==3:raise torch.cuda.OutOfMemoryError('simulated CUDA OOM')
        return original(self,path,cfg)
    monkeypatch.setattr(Regression,'forward_cell',fail_once)
    recovered,_,_=experiments.phase(cfg,f,tmp_path/'oom','head',42)
    for key,value in recovered.state_dict().items():torch.testing.assert_close(value,expected[key],rtol=2e-5,atol=2e-6)
    saved=torch.load(tmp_path/'oom/last.pt',weights_only=False)
    assert saved['microbatch']==1
    assert saved['fingerprint']['training']['effective_batch_cells']==4
