# Scopo: contratto CPU/offline del nuovo fine-tuning, senza checkpoint esterni o dati dell'universita.
# Fasi: student piccolo -> preprocessing ufficiale -> freezing/step -> aggregazioni -> cache/resume/CV.
# Input: fixture sintetiche locali, target/fold separati; output: assert e temporanei pytest.
# Parametri: ultimi 1/2/3 blocchi, due aggregazioni, policy norm/fusione; uso python -m pytest -ra.
# Risorse: CPU singolo thread, nessuna rete/GPU; ripresa verificata su optimizer/RNG/checkpoint atomici.
from copy import deepcopy
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import torch

from biomap_v2.student import PixelStudent
from biomap_v2 import reference as official
from biomap_v2.common import ROOT,digest
from biomap_finetune import config
from biomap_finetune.encoder import PartialMedium,FineTuneModel,token_groups,save_prefix_prototype,cached_suffix
from biomap_finetune.io import commit_arrays,load_arrays,save_checkpoint,load_checkpoint,claim_root,checked_roots,save_json
from biomap_finetune.data import select_aoi,CellDataset,collate_cells


@pytest.fixture
def tiny():
    torch.set_num_threads(1);torch.manual_seed(19)
    return PixelStudent(latent_dim=4,num_layers=4,nhead=4,dim_feedforward=32,dropout=0)


def raw(n=4):
    rng=np.random.default_rng(99)
    return {'s2':rng.integers(50,4000,(n,12,10),dtype=np.uint16),'mask':np.ones((n,12),bool),
        's2_doys':np.array([1,20,40,60,80,100,120,200,240,300,330,366],np.int16),
        'asc':rng.integers(2500,6000,(n,9,2),dtype=np.int16),'asc_doys':np.arange(1,10,dtype=np.int16),
        'desc':rng.integers(2500,6000,(n,8,2),dtype=np.int16),'desc_doys':np.arange(100,108,dtype=np.int16)}


def cfg(tmp_path):
    c=config.load(ROOT/'configs/v2_partial_finetuning.yaml')
    c['experiment']['profile']='synthetic';c['paths'].update(reference=str(tmp_path/'reference'),raw_root=str(tmp_path/'raw'),output_root=str(tmp_path/'runs'))
    c['encoder']['checkpoint']='synthetic_fixture_not_a_public_checkpoint'
    c['sampling'].update(method='uniform_pixels',pixels_per_cell=32,approved=True)
    c['aggregation']['method']='dense_mean';c['training'].update(learning_rate_encoder=1e-4,learning_rate_head=1e-3,batch_size=1,epochs=2)
    c['storage']['reserve_gib']=0;return c


@pytest.mark.parametrize('last_n',[1,2,3])
def test_real_architecture_boundary_gradient_optimizer(tiny,last_n):
    model=PartialMedium(tiny,last_n).eval();a,b=next(token_groups(raw(2)))[1:]
    a=torch.from_numpy(a);b=torch.from_numpy(b)
    with torch.no_grad():torch.testing.assert_close(model(a,b),tiny(a,b),rtol=2e-5,atol=2e-5)
    before={n:p.detach().clone() for n,p in model.named_parameters()}
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-3)
    model.train();loss=(model(a,b)*torch.linspace(-1,1,128)).sum();loss.backward()
    for name,p in model.named_parameters():
        if not p.requires_grad:assert p.grad is None
    assert any(p.grad is not None and p.grad.abs().sum()>0 for n,p in model.named_parameters() if 'layers.3.' in n)
    ids={id(p) for g in optimizer.param_groups for p in g['params']}
    assert all((id(p) in ids)==p.requires_grad for p in model.parameters())
    optimizer.step();changed=[n for n,p in model.named_parameters() if not torch.equal(before[n],p)]
    assert changed and all(dict(model.named_parameters())[n].requires_grad for n in changed)
    for name in ['s2_backbone','s1_backbone']:
        assert all(not x.training for x in list(getattr(tiny,name).transformer_encoder.layers)[:-last_n])
        assert any(name+'.transformer_encoder' in n for n in changed)


@pytest.mark.parametrize('missing',['none','s2','s1','mixed'])
def test_preprocessing_matches_official(tiny,missing):
    r=raw()
    if missing=='s2':r['mask'][:]=False
    if missing=='s1':r['asc'][:]=0;r['desc'][:]=0
    if missing=='mixed':r['mask'][0]=False;r['asc'][1]=0;r['desc'][1]=0;r['mask'][2,1:]=False
    tiny.eval();adapter=PartialMedium(tiny,2).eval()
    expected=official.encode_pixels(tiny,r['s2'],r['s2_doys'],s2_masks=r['mask'],
        s1_asc_bands=r['asc'],s1_asc_doys=r['asc_doys'],s1_desc_bands=r['desc'],s1_desc_doys=r['desc_doys'],device=torch.device('cpu'),batch_pixels=2)
    with torch.no_grad():actual=adapter.encode_raw(r,2).numpy()
    np.testing.assert_allclose(actual,expected,rtol=2e-5,atol=2e-5)


def test_no_arbitrary_imputation(tiny):
    r=raw();r['mask'][:]=False;r['asc'][:]=0;r['desc'][:]=0
    with pytest.raises(ValueError,match='entrambe'):list(token_groups(r))


@pytest.mark.parametrize('method',['dense_mean','spatial_cnn'])
def test_cell_prediction_and_differentiable_heads(tiny,method):
    encoder=PartialMedium(tiny,2);model=FineTuneModel(encoder,method,{'dropout':0.0})
    batch={'cell_ids':['a','b'],'groups':[{'raw':raw(2),'owner':0},{'raw':raw(3),'owner':1},{'raw':raw(1),'owner':0}],
        'patches':[{'group':0,'owner':0,'height':2,'width':2,'flat_local':np.array([0,3]),'take':np.arange(2)},
        {'group':1,'owner':1,'height':2,'width':2,'flat_local':np.array([0,1,3]),'take':np.arange(3)},
        {'group':2,'owner':0,'height':1,'width':1,'flat_local':np.array([0]),'take':np.arange(1)}]}
    model.train();out=model(batch,2);assert out.shape==(2,) and torch.isfinite(out).all()
    torch.nn.functional.mse_loss(out,torch.tensor([.2,.8])).backward()
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in model.head.parameters())
    assert any(p.grad is not None and p.grad.abs().sum()>0 for n,p in tiny.named_parameters() if 'layers.3.' in n)
    assert all(p.grad is None for n,p in tiny.named_parameters() if 'layers.0.' in n)
    model.eval()
    # Targets are deliberately absent from inference input.
    with torch.no_grad():
        actual=model(batch,2);assert actual.shape==(2,)
        from biomap_finetune.dense_evaluation import predict_cell
        positions=[np.array([0,3]),np.array([0,1,3]),np.array([0])]
        for owner in range(2):
            pieces=[{'embeddings':model.encoder.encode_raw(g['raw'],2).numpy(),'flat':positions[i],'width':2 if i<2 else 1}
                for i,g in enumerate(batch['groups']) if g['owner']==owner]
            np.testing.assert_allclose(predict_cell(model,pieces,batch['cell_ids'][owner],'cpu'),float(actual[owner]),rtol=2e-5,atol=2e-5)


def test_dense_pooling_is_weighted_over_all_contributions(tiny):
    model=FineTuneModel(PartialMedium(tiny,2),'dense_mean').eval()
    batch={'cell_ids':['one'],'groups':[{'raw':raw(1),'owner':0},{'raw':raw(3),'owner':0}]}
    with torch.no_grad():
        embeddings=torch.cat([model.encoder.encode_raw(g['raw']) for g in batch['groups']])
        expected=model.head({'mean':embeddings.mean(0,keepdim=True)})
        torch.testing.assert_close(model(batch),expected)


def test_frozen_control_and_norm_projection_policy(tiny):
    model=PartialMedium(tiny,2,train_projection=False,train_normalization=False)
    assert not any(p.requires_grad for p in tiny.dim_reducer.parameters())
    assert all(not p.requires_grad for n,p in tiny.named_parameters() if '.norm' in n)
    model=PartialMedium(tiny,2,mode='frozen').train()
    assert not any(p.requires_grad for p in model.parameters())
    assert not model.student.training


def test_prefix_cache_parity_identity_and_gradients(tiny,tmp_path):
    model=PartialMedium(tiny,2).eval();_,a,b=next(token_groups(raw(2)));a=torch.tensor(a);b=torch.tensor(b)
    tag=save_prefix_prototype(tmp_path/'prefix',model,a,b,'input-A')
    torch.testing.assert_close(cached_suffix(tmp_path/'prefix',model,tag),model(a,b),rtol=2e-5,atol=2e-5)
    model.train();torch.manual_seed(55);online=model(a,b)
    torch.manual_seed(55);cached=cached_suffix(tmp_path/'prefix',model,tag)
    torch.testing.assert_close(cached,online,rtol=2e-5,atol=2e-5)
    cached_suffix(tmp_path/'prefix',model,tag).sum().backward()
    assert any(p.grad is not None for p in model.parameters() if p.requires_grad)
    with torch.no_grad():next(model.student.s2_backbone.embedding.parameters()).add_(.1)
    with pytest.raises(ValueError,match='diversi'):cached_suffix(tmp_path/'prefix',model,tag)


def test_frozen_projection_still_propagates_suffix_gradient(tiny):
    adapter=PartialMedium(tiny,2,train_projection=False).train()
    out=adapter.encode_raw(raw(2));(out*torch.linspace(-1,1,128)).sum().backward()
    assert any(p.grad is not None and p.grad.abs().sum()>0 for n,p in tiny.named_parameters() if '.layers.3.' in n)
    assert all(p.grad is None for p in tiny.dim_reducer.parameters())


def test_atomic_arrays_checksum_partial_and_checkpoint(tmp_path):
    p=tmp_path/'arrays';commit_arrays(p,{'a':np.arange(4,dtype=np.int16)},'id')
    assert np.array_equal(load_arrays(p,'id')[0]['a'],np.arange(4))
    with pytest.raises(ValueError):load_arrays(p,'other')
    with (p/'a.npy').open('ab') as f:f.write(b'corrupt')
    with pytest.raises(ValueError,match='Checksum'):load_arrays(p,'id')
    partial=tmp_path/'incomplete';partial.mkdir()
    with pytest.raises(ValueError,match='marker'):load_arrays(partial)
    save_checkpoint(tmp_path/'epoch.pt',{'identity':'id','epoch':1,'weight':torch.ones(2)})
    assert load_checkpoint(tmp_path/'epoch.pt','id')['epoch']==1
    with pytest.raises(ValueError):load_checkpoint(tmp_path/'epoch.pt','other')


def test_config_decisions_gate_and_paths(tmp_path):
    base=config.load(ROOT/'configs/v2_partial_finetuning.yaml');assert base['aggregation']['method']=='undecided'
    assert base['training']['epochs'] is None and base['finetuning']['last_n_blocks']==2
    with pytest.raises(ValueError,match='undecided'):config.require_ready(base)
    c=cfg(tmp_path);config.require_ready(c);c['cache']['mode']='cached'
    with pytest.raises(ValueError,match='prototipo'):config.require_ready(c)
    c=cfg(tmp_path);c['paths']['raw_root']=c['paths']['reference']
    with pytest.raises(ValueError):checked_roots(c)
    claim_root(tmp_path/'owned','one','raw')
    with pytest.raises(ValueError):claim_root(tmp_path/'owned','two','raw')
    with pytest.raises(ValueError):config.load(ROOT/'configs/v2_partial_finetuning.yaml',['training.unknown=1'])


def test_uniform_sampling_is_per_cell_not_per_tile_and_target_free():
    entries={'tile_A':(np.array([1,1,2,2]),np.arange(4),2),'tile_B':(np.array([1,1,2,2]),np.arange(4),2)}
    s={'method':'uniform_pixels','pixels_per_cell':3}
    selected=select_aoi(entries,s,42);counts={1:0,2:0}
    for tile,pos in selected.items():
        for i in pos:counts[int(entries[tile][0][i])]+=1
    assert counts=={1:3,2:3}
    for tile,pos in selected.items():assert np.array_equal(pos,select_aoi(dict(reversed(list(entries.items()))),s,42)[tile])


def synthetic_store(tmp_path):
    root=tmp_path/'raw';root.mkdir();rows=[];blocks=[]
    for i in range(4):
        aoi='liguria_levante' if i==3 else f'aoi_{i}';key=aoi+'/grid_1/0000';r=raw(2)
        arrays={**r,'positions':np.arange(2),'flat':np.array([0,3]),'cell_ref':np.array([i,i])}
        commit_arrays(root/'blocks'/key,arrays,'block-'+str(i))
        blocks.append({'key':key,'identity':'block-'+str(i),'marker_sha256':digest(root/'blocks'/key/'complete.json'),'aoi':aoi,'tile':'grid_1'})
        p=root/'geometries'/aoi/'grid_1.json';save_json(p,{'width':2,'height':2,'crs':'EPSG:32632' if i%2 else 'EPSG:32633','transform':[10,0,300000,0,-10,4500000]})
        rows.append({'reference_index':i,'cell_id':f'cell_{i}','aoi_id':aoi,'cv_fold':-1 if i==3 else i,'partition':'test' if i==3 else 'train','target':.15+i*.2,'n_valid':2})
    frame=pd.DataFrame(rows);frame.to_parquet(root/'cells.parquet',index=False)
    save_json(root/'dataset.json',{'identity':'synthetic-data','blocks':blocks,'counts':{str(i):2 for i in range(4)},
        'cells_sha256':digest(root/'cells.parquet'),'profile':'synthetic','support':'fixture',
        'geometries':{p.relative_to(root).as_posix():digest(p) for p in (root/'geometries').rglob('*.json')}})
    return root


def test_dataloader_spawn_contract_and_resume_exact(tiny,tmp_path,monkeypatch):
    from biomap_finetune import training
    root=synthetic_store(tmp_path);c=cfg(tmp_path)
    monkeypatch.setattr(training,'load_medium',lambda p:(deepcopy(tiny),{}))
    dataset=CellDataset(root,['cell_0','cell_1']);sample=dataset[0]
    assert sample['aoi_ids']==['aoi_0'] and sample['folds']==[0]
    batch=collate_cells([sample,dataset[1]]);assert len(batch['cell_ids'])==2
    c['training']['num_workers']=1
    # Test actual spawn worker, not just pickle.dumps: CPU, offline local fixture.
    assert sum(len(b['cell_ids']) for b in training.loader(dataset,c))==2
    c['training']['num_workers']=0
    first,_=training.fit(c,dataset,None,tmp_path/'continuous','identity',epoch_count=2)
    training.fit(c,dataset,None,tmp_path/'resume','identity',epoch_count=1)
    # Simulate a crash after a checkpoint marker, before latest/best pointer publication.
    (tmp_path/'resume/latest.json').unlink();(tmp_path/'resume/best.json').unlink()
    # Also preserve an unmarked orphan for the next epoch; do not overwrite it.
    (tmp_path/'resume/epoch_0002.pt').write_bytes(b'incomplete unmarked orphan')
    resumed,_=training.fit(c,dataset,None,tmp_path/'resume','identity',resume=True,epoch_count=2)
    for k,v in first.state_dict().items():torch.testing.assert_close(v,resumed.state_dict()[k],rtol=0,atol=0)
    assert (tmp_path/'resume/epoch_0001.pt').is_file() and (tmp_path/'resume/epoch_0002.pt').read_bytes()==b'incomplete unmarked orphan'
    assert len(list((tmp_path/'resume').glob('epoch_0002_*.pt')))==1


def test_cv_never_selects_liguria(tmp_path):
    from biomap_finetune.training import macro_rmse
    frame=pd.DataFrame({'aoi_id':['liguria_levante'],'partition':['test'],'target':[.5],'prediction':[.4]})
    with pytest.raises(ValueError,match='vietata'):macro_rmse(frame)


@pytest.mark.parametrize('aggregation',['dense_mean','spatial_cnn'])
@pytest.mark.parametrize('encoder_mode',['frozen','partial'])
def test_cpu_cv_refit_frozen_control_end_to_end(tiny,tmp_path,monkeypatch,aggregation,encoder_mode):
    from biomap_finetune import training,data
    root=synthetic_store(tmp_path);c=cfg(tmp_path);c['aggregation']['method']=aggregation
    c['training'].update(epochs=1,encoder_mode=encoder_mode)
    monkeypatch.setattr(training,'load_medium',lambda p:(deepcopy(tiny),{}))
    monkeypatch.setattr(data,'plan',lambda c:(None,None,None,'synthetic-data'))
    assert training.train(c,'control','cv')['status']=='COMPLETE'
    training.train(c,'control','refit',resume=True)
    frame=pd.read_parquet(tmp_path/'runs/control/oof_predictions.parquet')
    assert set(frame.cell_id)=={'cell_0','cell_1','cell_2'} and not frame.aoi_id.eq('liguria_levante').any()
    assert frame.groupby('model').size().to_dict()=={'dummy_mean':3,'predictor':3}
    assert (tmp_path/'runs/control/refit/best.json').is_file()


def test_comparison_requires_exact_population_and_support(tmp_path):
    from biomap_finetune.comparison import compare
    root=tmp_path/'paired';a=root/'partial';b=root/'frozen'
    c=cfg(tmp_path)
    for p,mode in [(a,'partial'),(b,'frozen')]:
        p.mkdir(parents=True);save_json(p/'manifest.json',{'dataset_identity':'same','aggregation':'dense_mean','seed':42,'profile':'synthetic','code':{'same':'sha'},'encoder_mode':mode})
        save_json(p/'effective_config.json',c)
        pd.DataFrame({'cell_id':['a','b'],'aoi_id':['x','y'],'cv_fold':[0,1],'partition':['train','train'],'fold':[0,1],
            'target':[.4,.8],'prediction':[.45,.75] if mode=='partial' else [.5,.7],'model':['predictor','predictor']}).to_parquet(p/'oof_predictions.parquet',index=False)
    assert compare(a,b)['oof']['delta_macro_aoi_rmse']<0
    frame=pd.read_parquet(b/'oof_predictions.parquet');frame.iloc[:1].to_parquet(b/'oof_predictions.parquet',index=False)
    with pytest.raises(ValueError,match='ID diversi'):compare(a,b)


def test_budget_and_tolerance_cannot_be_silently_changed(tmp_path):
    c=cfg(tmp_path);c['validation']['rtol']=.1
    with pytest.raises(ValueError,match='tolleranze'):config.validate(c)
    c=cfg(tmp_path);c['aggregation']['method']='undecided'
    from biomap_finetune.cli import smoke
    c['experiment']['profile']='smoke'
    with pytest.raises(ValueError,match='undecided'):smoke(c,'not_started')


@pytest.mark.parametrize('aggregation',['dense_mean','spatial_cnn'])
def test_dense_eval_full_support_and_resume_without_legacy_writes(tiny,tmp_path,monkeypatch,aggregation):
    from biomap_finetune import dense_evaluation as dense,training
    root=synthetic_store(tmp_path);source=CellDataset(root);c=cfg(tmp_path);c['experiment']['profile']='benchmark';c['aggregation']['method']=aggregation
    run=tmp_path/'run';run.mkdir();ref=Path(c['paths']['reference']);areas=[]
    for row in source.frame.itertuples():
        tile=ref/row.aoi_id/'grid_1';tile.mkdir(parents=True)
        np.savez(tile/'geometry.npz',flat_indices=np.array([0,3]),cell_ids=np.array([0,0]))
        np.save(tile.parent/'lookup.npy',np.array([row.reference_index]))
        save_json(tile/'geometry.json',{'width':2,'height':2,'crs':'EPSG:32632','transform':[10,0,300000,0,-10,4500000],'pixels':2})
        areas.append({'aoi_id':row.aoi_id,'tiles':['grid_1']})
    monkeypatch.setattr(training,'load_medium',lambda p:(deepcopy(tiny),{}))
    for fold in ['fold_0','fold_1','fold_2','refit']:
        path=run/fold/'epoch_0001.pt';save_checkpoint(path,{'identity':'run-id','epoch':1,'model':training.build_model(c).state_dict()})
        save_json(path.parent/'best.json',{'file':path.name,'epoch':1})
    before={str(p):digest(p) for p in ref.rglob('*') if p.is_file()}
    monkeypatch.setattr(dense,'registry',lambda *args:(source,run,'run-id'))
    monkeypatch.setattr(dense,'reference',lambda c:({'areas':areas},source.frame))
    streams={'s2':{'2020-01-01':[]},'asc':{'2020-01-01':[]},'desc':{}}
    monkeypatch.setattr(dense,'catalogs',lambda *args:(streams,[0,0,1,1],{'fixture':'catalog'}))
    monkeypatch.setattr(dense.historical,'reader',lambda *args:type('Source',(),{'close':lambda self:None})())
    calls=[]
    def acquire(*args):calls.append(True);return raw(2)
    monkeypatch.setattr(dense.historical,'acquire_points',acquire)
    result=dense.dense_evaluate(c,'run',2)
    assert result['cells']==4 and len(calls)==4
    predictions=pd.read_parquet(run/'dense-eval/oof_dense_predictions.parquet')
    assert len(predictions[predictions.model=='predictor'])==3
    assert not predictions.aoi_id.eq('liguria_levante').any()
    assert not list((run/'dense-eval/tmp').rglob('features.npy'))
    dense.dense_evaluate(c,'run',2,resume=True);assert len(calls)==4
    assert before=={str(p):digest(p) for p in ref.rglob('*') if p.is_file()}


def test_acquisition_resume_and_historical_raw_equivalence(tmp_path,monkeypatch):
    from biomap_finetune import data
    from biomap_v2.campaign import streaming as historical
    c=cfg(tmp_path);c['acquisition']['io_workers']=1
    tile=tmp_path/'reference/aoi/grid_1';tile.mkdir(parents=True)
    np.savez(tile/'geometry.npz',flat_indices=np.array([0,3]),cell_ids=np.array([0,0]))
    save_json(tile/'geometry.json',{'width':2,'height':2,'crs':'EPSG:32632','transform':[10,0,300000,0,-10,4500000],'pixels':2})
    frame=pd.DataFrame({'reference_index':[0],'cell_id':['one'],'aoi_id':['aoi'],'cv_fold':[0],'partition':['train'],'target':[.3],'n_valid':[2]})
    entry={'aoi':'aoi','tile':'grid_1','positions':np.arange(2),'ids':np.array([0,0]),'flat':np.array([0,3]),'width':2}
    monkeypatch.setattr(data,'plan',lambda c:(frame,[entry],{0:2},'raw-id'))
    streams={k:{'2020-12-31':[{'id':k,'properties':{}}]} for k in ['s2','asc','desc']}
    monkeypatch.setattr(data,'catalogs',lambda *args:(streams,[0,0,1,1],{'fixture':'catalog'}))
    source=type('Source',(),{'close':lambda self:None})()
    monkeypatch.setattr(historical,'reader',lambda *args:source)
    calls={};fail=[True]
    def compose(name,items,pixels,source):
        calls[name]=calls.get(name,0)+1
        if name=='desc' and fail[0]:fail[0]=False;raise OSError('synthetic interrupted network')
        return np.full((len(pixels),10 if name=='s2' else 2),1000 if name=='s2' else 4000,dtype=np.uint16 if name=='s2' else np.int16),np.ones(len(pixels),bool)
    monkeypatch.setattr(historical,'compose_day',compose)
    with pytest.raises(OSError,match='interrupted'):data.acquire(c)
    assert not (tmp_path/'raw/dataset.json').exists()
    result=data.acquire(c,resume=True)
    assert calls=={'s2':1,'asc':1,'desc':2}
    block=result['blocks'][0];actual,_=load_arrays(tmp_path/'raw/blocks'/block['key'],block['identity'])
    expected=historical.acquire_points(historical.Pixels(tile),streams,tmp_path/'original_days',{'fixture':'original'},source)
    for key in data.RAW_KEYS:assert np.array_equal(actual[key],expected[key])
    assert actual['s2_doys'][0]==366
    dataset=CellDataset(tmp_path/'raw');assert len(dataset)==1 and len(dataset[0]['groups'][0]['raw']['s2'])==2
    # Re-run only verifies checksums; no new remote composition.
    before=dict(calls);data.acquire(c,resume=True);assert calls==before


def test_new_command_doctor_undecided_and_monitor(tmp_path,capsys):
    from biomap_finetune.cli import main
    result=main(['doctor','--config',str(ROOT/'configs/v2_partial_finetuning.yaml')])
    assert result['status']=='NOT_VERIFIED_FOR_REAL_TRAINING' and result['blockers']
    save_json(tmp_path/'progress.json',{'phase':'complete','eta_seconds':0})
    main(['monitor','--progress',str(tmp_path/'progress.json'),'--once'])
    assert 'complete' in capsys.readouterr().out


def test_space_guard_and_source_identity_protection(tmp_path,monkeypatch):
    from biomap_spatial import utils
    from biomap_finetune.io import disk_budget
    monkeypatch.setattr(utils.shutil,'disk_usage',lambda p:type('Disk',(),{'free':10})())
    with pytest.raises(OSError):disk_budget(tmp_path,20,1)
    import hashlib
    protected=json.loads((ROOT/'tests/fixtures/finetuning_protected_1737b1f.json').read_text(encoding='utf-8'))
    for name,item in protected['files'].items():
        actual=(ROOT/name).read_bytes()
        if item['comparison']=='normalized_lf':actual=actual.replace(b'\r\n',b'\n')
        assert hashlib.sha256(actual).hexdigest()==item['sha256_checked_bytes']
