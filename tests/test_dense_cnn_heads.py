# Test CPU isolati: nessuna serie Sentinel, nessun download, nessun training scientifico.
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import torch
from biomap_dense_heads.models import make
from biomap_dense_heads.cli import config,synthetic,main
from biomap_dense_heads.artifacts import load
from biomap_dense_heads.progress import Progress,monitor
from biomap_dense_heads.engine import fit,predict
from biomap_v2.common import ROOT,digest,save_json
from biomap_v2.campaign.state import MEANS

@pytest.fixture
def cfg():
    c=config(ROOT/'experiments/v2_dense_cnn_msa/config.yaml')
    c.update(epochs=2,patience=2,batch_size=16,threads=1)
    return c

# Le due reti hanno dimensioni confrontabili e gradiente, senza encoder/spazio 2D.
def test_shapes_and_gradient(cfg):
    counts=[]
    for name in ['Conv1D','MLP']:
        for output in ['identity','sigmoid']:
            model=make(name,output,cfg);x=torch.randn(7,128);y=model(x)
            assert y.shape==(7,) and torch.isfinite(y).all()
            y.square().mean().backward();assert all(p.grad is not None for p in model.parameters())
            if output=='sigmoid':assert ((y>=0)&(y<=1)).all()
        counts.append(sum(p.numel() for p in model.parameters()))
    assert abs(counts[0]-counts[1])/counts[0]<.05
    with pytest.raises(ValueError):make('Conv1D','identity',cfg)(torch.randn(2,128,5,5))

def test_completion_and_read_only(tmp_path):
    source=synthetic(tmp_path/'fixture');before={p.name:digest(p) for p in source.iterdir() if p.is_file()}
    with pytest.raises(ValueError,match='parziale'):load(source,'cpu',tmp_path/'real_out')
    frame,audit=load(source,'cpu',tmp_path/'smoke_out',smoke=True)
    assert len(frame)==96 and audit['shape']==[96,128]
    assert before=={p.name:digest(p) for p in source.iterdir() if p.is_file()}
    with pytest.raises(ValueError,match='separate'):load(source,'cpu',source/'heads',smoke=True)
    marker=source/'extraction_complete.json';marker.unlink()
    with pytest.raises(ValueError,match='manca'):load(source,'cpu',tmp_path/'bad',smoke=True)

# NaN/duplicati e leakage sono errori, non celle eliminate di nascosto.
@pytest.mark.parametrize('corruption',['nan','duplicate','overlap'])
def test_invalid_artifacts(tmp_path,corruption):
    source=synthetic(tmp_path/'fixture');p=source/'v2.parquet';f=pd.read_parquet(p)
    if corruption=='nan':f.loc[0,MEANS[0]]=np.nan
    elif corruption=='duplicate':f.loc[1,'cell_id']=f.loc[0,'cell_id']
    else:f.loc[f.partition=='test','aoi_id']=f.loc[0,'aoi_id']
    f.to_parquet(p,index=False);save_json(source/'extraction_complete.json',{'sha256':digest(p),'n_cells':len(f),'complete_campaign':False})
    with pytest.raises(ValueError):load(source,'cpu',tmp_path/'out',smoke=True)

# La validation estrema non può influenzare lo scaler; resume non riaddestra fit completi.
def test_training_scaler_reproducibility_resume(cfg,tmp_path):
    rng=np.random.default_rng(1);x=rng.normal(size=(40,128)).astype(np.float32);x[30:]+=100
    y=np.linspace(.1,.8,40).astype(np.float32);a=np.array(['a']*30+['v']*10);train=np.arange(30);val=np.arange(30,40)
    outputs=[]
    for run in ['one','two']:
        out=tmp_path/run;out.mkdir();progress=Progress(out)
        (head,scaler),history,epoch=fit(x,y,a,train,val,'Conv1D','identity',cfg,out,'cpu',progress,'fixed')
        assert np.allclose(scaler.mean_,x[train].mean(axis=0),atol=1e-6)
        assert len(history)==2 and 1<=epoch<=2
        p=predict(head,scaler.transform(x[val]).astype(np.float32),16,'cpu');outputs.append(p)
        (restored,s),h,e=fit(x,y,a,train,val,'Conv1D','identity',cfg,out,'cpu',progress,'fixed',resume=True)
        assert np.array_equal(p,predict(restored,s.transform(x[val]).astype(np.float32),16,'cpu'))
        with pytest.raises(ValueError,match='incompatibile'):fit(x,y,a,train,val,'Conv1D','identity',cfg,out,'cpu',progress,'changed',resume=True)
    assert np.array_equal(outputs[0],outputs[1])

# End-to-end 26 fit piccoli: CSV, Excel, modelli, stato completo, stessa popolazione test.
def test_smoke_cli(tmp_path):
    out=tmp_path/'smoke';assert main(['smoke','--out',str(out)])==0
    results=out/'smoke_results';state=json.loads((results/'progress.json').read_text())
    assert state['state']=='complete' and state['fits_done']==state['fits_total']==26
    f=pd.read_csv(results/'predictions.csv');test=f[f.partition=='liguria_benchmark']
    assert test.groupby('model_name').size().eq(24).all()
    assert len(test.model_name.unique())==5
    assert (results/'results.xlsx').exists() and (results/'report.md').exists()
    assert monitor(results,once=True)==0
    assert main(['export','--out',str(results)])==0

# Confronto storico parziale per modello/ruolo: tutte le celle dello scope devono coincidere.
def test_compare_historical_subset(tmp_path):
    from biomap_dense_heads.evaluation import compare_external
    current=pd.DataFrame({'cell_id':['a','b','a','b'],'model_name':['HistGB','HistGB','Conv1D_identity','Conv1D_identity'],
        'partition':['liguria_benchmark']*4,'target':[.2,.4,.2,.4],'prediction':[.21,.39,.2,.4]})
    old=current.iloc[:2].copy();old.prediction=[.25,.35];path=tmp_path/'old.csv';old.to_csv(path,index=False)
    compare_external(current,path,tmp_path)
    assert len(pd.read_csv(tmp_path/'paired_historical_metrics.csv'))==6
    old.loc[0,'target']=.8;old.to_csv(path,index=False)
    with pytest.raises(ValueError,match='non appaiato'):compare_external(current,path,tmp_path)

# Profiling misura una sola testa e dichiara esplicitamente il carattere non scientifico.
def test_profile_smoke(cfg,tmp_path):
    from biomap_dense_heads.campaign import profile
    source=synthetic(tmp_path/'fixture');out=tmp_path/'profile'
    f,audit=load(source,'cpu',out,smoke=True)
    profile(f,cfg,audit,out,'cpu',Progress(out),cells=16,epochs=1)
    result=json.loads((out/'profile.json').read_text())
    assert result['measured_epoch_seconds']>0 and result['gpu_peak_bytes'] is None
    assert result['parameters']=={'Conv1D':4625,'MLP':4681}

# Un marker dell'altro produttore non abilita il training, anche se hash e flag coincidono.
def test_marker_method_required(tmp_path):
    source=synthetic(tmp_path/'fixture');marker=source/'extraction_complete.json'
    proof=json.loads(marker.read_text());proof['complete_campaign']=True;save_json(marker,proof)
    with pytest.raises(ValueError,match='method'):load(source,'cpu',tmp_path/'out')
    proof['method']='different-producer';save_json(marker,proof)
    with pytest.raises(ValueError,match='method'):load(source,'cpu',tmp_path/'out2')

# Il consumer mantiene aperta la snapshot durante replace; l'handle OS è già chiuso.
def test_progress_reader_allows_replace(tmp_path):
    from biomap_dense_heads.progress import shared_open
    p=Progress(tmp_path);p.update(state='running')
    with shared_open(p.path) as reader:
        p.update(state='complete')
        assert json.load(reader)['state']=='running'
    assert json.loads(p.path.read_text())['state']=='complete'

# Se estrazione incompleta, nemmeno una query CUDA può interferire con la GPU attiva.
def test_incomplete_does_not_initialize_cuda(tmp_path,monkeypatch):
    source=tmp_path/'active';source.mkdir()
    def forbidden():raise AssertionError('GPU queried before completion')
    monkeypatch.setattr(torch.cuda,'is_available',forbidden)
    with pytest.raises(ValueError,match='manca'):load(source,'cuda:0',tmp_path/'out')
