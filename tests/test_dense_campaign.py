# Verifiche sintetiche offline: transazioni, supporto, fold, OOM, leakage e interfacce.
import json
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
from biomap_v2.campaign.state import atomic_npz,cached,membership,writable,MEANS
from biomap_v2.campaign.probing import aligned,params
from biomap_v2.campaign.extraction import infer

# Il manifest reale piccolo è versionato; nessun download satellitare.
def test_real_membership():
    f=membership();assert len(f)==132900
    d=f[f.partition=='train'];assert len(d)==127909 and d.aoi_id.nunique()==17
    assert [int((d.cv_fold==i).sum()) for i in range(3)]==[38808,44895,44206]
    assert not set(d.aoi_id)&set(f[f.partition=='test'].aoi_id)

# Un marker mancante rende il payload orfano incompleto, non valido.
def test_atomic_resume_and_corruption(tmp_path):
    p=tmp_path/'chunk.npz';identity={'geometry':'a','start':0}
    atomic_npz(p,identity,x=np.arange(12));assert cached(p,identity)
    with pytest.raises(ValueError):cached(p,{'geometry':'b'})
    p.with_suffix('.json').unlink();assert not cached(p,identity)
    atomic_npz(p,identity,x=np.arange(12));p.write_bytes(b'broken')
    with pytest.raises(ValueError):cached(p,identity)

def test_source_root_protected(tmp_path):
    with pytest.raises(ValueError):writable(tmp_path/'source'/'output',[tmp_path/'source'])
    assert writable(tmp_path/'work',[tmp_path/'source']).exists()

# Allineamento per identificativi, non per ordine accidentale dei file.
def frame():
    f=pd.DataFrame({'cell_id':['a','b'],'aoi_id':['x','y'],'partition':['train','test'],
      'cv_fold':[0,-1],'n_valid':[2,3],'n_total':[2,3],'lon':[10.,11.],'lat':[44.,45.],'target':[.2,.5]})
    return pd.concat([f,pd.DataFrame(np.zeros((2,128)),columns=MEANS)],axis=1)

def test_alignment():
    f=frame();assert aligned(f,f.iloc[::-1]).cell_id.tolist()==['a','b']
    for column,value in [('n_valid',999),('target',.9),('cv_fold',2)]:
        bad=f.copy();bad.loc[0,column]=value
        with pytest.raises(ValueError):aligned(f,bad)
    with pytest.raises(ValueError):aligned(f,f.iloc[:1])

def test_chunk_sums_equal_monolithic():
    rng=np.random.default_rng(42);x=rng.normal(size=(100,128));ids=rng.integers(0,8,100)
    def calc(chunks):
        sums=np.zeros((8,128));count=np.zeros(8,int)
        for low,high in chunks:
            np.add.at(sums,ids[low:high],x[low:high]);count+=np.bincount(ids[low:high],minlength=8)
        return sums/count[:,None]
    assert np.allclose(calc([(0,100)]),calc([(i,min(i+7,100)) for i in range(0,100,7)]),atol=1e-14)

# OOM modifica solo il batch; gli input restano gli stessi.
def test_oom_retry(monkeypatch):
    import torch
    import biomap_v2.campaign.extraction as m
    batches=[]
    def fake(model,*a,**k):
        batches.append(k['batch_pixels'])
        if k['batch_pixels']>2:raise torch.cuda.OutOfMemoryError('simulated')
        return np.ones((3,128))
    monkeypatch.setattr(m,'encode_pixels',fake)
    raw={k:np.ones((3,1,2)) for k in ['s2','asc','desc','mask']}
    raw.update({k+'_doys':np.array([1]) for k in ['s2','asc','desc']})
    assert infer(None,raw,'cpu',8).shape==(3,128);assert batches==[8,4,2]

def test_regressor_configuration():
    p=params(20260925);assert p['ridge_alpha']==100
    assert p['histgb']['max_iter']==90 and p['xgb']['max_depth']==2

# La CLI non avvia implicitamente un download passando all.
def test_cli_contract():
    from biomap_v2.campaign.cli import parser
    a=parser().parse_args(['all','--reference','R','--work-root','W'])
    assert a.chunk_size==4096 and a.pixel_batch==32 and not a.amp
    with pytest.raises(SystemExit):parser().parse_args(['unknown'])

# Verifica integrazione su tutti i cinque nomi, senza un training scientifico reale.
def test_probe_refit_and_export(tmp_path,monkeypatch):
    import biomap_v2.campaign.probing as p
    from sklearn.dummy import DummyRegressor
    from biomap_v2.campaign.reporting import export
    f=pd.DataFrame({'cell_id':[str(i) for i in range(24)],'aoi_id':[f'a{i//3}' for i in range(24)],
      'partition':['train']*18+['test']*6,'cv_fold':([0]*6+[1]*6+[2]*6+[-1]*6),
      'n_valid':[3]*24,'n_total':[3]*24,'lon':np.linspace(10,12,24),'lat':[44.]*24,'target':np.linspace(.1,.8,24)})
    f=pd.concat([f,pd.DataFrame(np.random.default_rng(42).normal(size=(24,128)),columns=MEANS)],axis=1)
    ref=tmp_path/'ref';ref.mkdir();f.to_parquet(ref/'v1.parquet');(ref/'reference.json').write_text('{}')
    work=tmp_path/'work';work.mkdir();f.to_parquet(work/'v2.parquet')
    args=SimpleNamespace(reference=str(ref),work_root=str(work),encoder='v1',seed=20260925)
    monkeypatch.setattr(p,'make_estimator',lambda *a,**k:DummyRegressor())
    p.probe(args,f);p.liguria(args,f)
    args.encoder='v2';p.probe(args,f);p.liguria(args,f);p.compare(work)
    export(work)
    assert len(pd.read_csv(work/'metrics.csv'))==20
    assert len(pd.read_csv(work/'probing/v2/oof_predictions.csv'))==90
    assert len(list((work/'probing/v2').glob('*.joblib')))==20
    assert (work/'results.xlsx').exists()
    # La seconda chiamata deve recuperare i modelli già verificati.
    p.probe(args,f)

from test_dense_portable import dense_raw

# Il controllo fisico rifiuta una permutazione dei pixel anche a conteggi identici.
def test_physical_support(dense_raw):
    from pathlib import Path
    from biomap_v2.campaign.physical import verify
    from biomap_v2.dense.grid import load_target
    f=pd.read_parquet(dense_raw['v1_parquet']);grid=load_target(dense_raw['target'])
    args=(dense_raw['v1_tiles'],dense_raw['v2_tiles'],grid,f.index.to_numpy(),f)
    assert verify(*args)['duplicate_contributions_approx_1cm']==0
    p=Path(dense_raw['v2_tiles'])/'grid_test/geometry.npz'
    with np.load(p) as z:flat=z['flat_indices'];ids=z['cell_ids']
    ids[[0,2]]=ids[[2,0]];np.savez(p,flat_indices=flat,cell_ids=ids)
    with pytest.raises(ValueError):verify(*args)

def test_allocated_arrays_not_complete(tmp_path):
    from biomap_v2.campaign.inputs import legacy_ready
    for k in ['s2','mask','asc','desc']:np.save(tmp_path/f'{k}.npy',np.zeros((1,1)))
    p=tmp_path/'inputs_manifest.json';p.write_text(json.dumps({'scene_groups':{k:[{'date':'2020-01-01'}] for k in ['s2','asc','desc']},'complete_days':{}}))
    assert not legacy_ready(tmp_path)
    p.write_text(json.dumps({'scene_groups':{k:[{'date':'2020-01-01'}] for k in ['s2','asc','desc']},'complete_days':{k:{'2020-01-01':'sha'} for k in ['s2','asc','desc']}}))
    assert legacy_ready(tmp_path)
