# Scopo: CNN modulare CPU, parita B, normalizzazione training-only e ablation sui bordi.
# Fasi: fixture a C variabile -> forward/backward senza target -> copia pesi B -> frammentazione.
# Input: array sintetici 8x8; output: assert e diagnostica patch locale, nessun risultato MSA reale.
# Parametri: C=32/128/384, kernel 1/3, proiezione direct/64. Uso: pytest tests/test_spatial_model.py.
# Risorse: CPU, pochi MB; checkpoint temporanei. Ripresa/ripristino in processi del test, niente encoder.
from dataclasses import replace
import numpy as np
import pandas as pd
import pytest
import torch

from biomap_spatial.batching import collate_cells
from biomap_spatial.contracts import ModelConfig
from biomap_spatial.model import SpatialCNN,MeanMLP
from biomap_spatial.normalization import fit_normalizer,transform
from biomap_spatial.utils import legacy,save_json


def batch_for(channels=128):
    rng=np.random.default_rng(4)
    mask=np.ones((5,6),bool);mask[0,0]=False
    x=rng.normal(size=(channels,5,6)).astype(np.float32);x[:,~mask]=np.nan
    return collate_cells([{'cell_id':'a','mean':np.nanmean(np.where(mask[None],x,np.nan),axis=(1,2)),
        'patches':[(x,mask)],'target':.3},{'cell_id':'b','mean':np.zeros(channels,np.float32),
        'patches':[(np.ones((channels,3,2),np.float32),np.ones((3,2),bool)),
                   (np.ones((channels,2,2),np.float32),np.ones((2,2),bool))],'target':.5}])


@pytest.mark.parametrize('channels',[32,128,384])
@pytest.mark.parametrize('projection',[None,64])
@pytest.mark.parametrize('kernel',[1,3])
def test_channels_projection_kernel_gradients_and_inference(channels,projection,kernel,tmp_path):
    config=ModelConfig(input_channels=channels,projection_channels=projection,kernel=kernel,dropout=0)
    model=SpatialCNN(config)
    batch=batch_for(channels);batch.pop('target')
    batch['features'].requires_grad_(True)
    # Restore invalid NaN after collate to exercise the backbone guard as well.
    with torch.no_grad():batch['features'][~batch['mask'].expand_as(batch['features'])]=float('nan')
    before={k:v.clone() for k,v in model.named_parameters()}
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001)
    out=model(batch)
    assert out.shape==(2,) and torch.isfinite(out).all()
    (out-torch.tensor([.3,.5])).square().mean().backward()
    assert torch.isfinite(batch['features'].grad).all()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    optimizer.step()
    assert any(not torch.equal(before[k],v) for k,v in model.named_parameters())
    path=tmp_path/'head.pt';torch.save(model.state_dict(),path)
    other=SpatialCNN(config);other.load_state_dict(torch.load(path,weights_only=True))
    model.eval();other.eval();torch.testing.assert_close(model(batch),other(batch),rtol=0,atol=0)
    if projection is not None:assert any(k.startswith('projection.') for k in model.state_dict())


def test_direct_128_matches_legacy_forward_and_gradients_exactly():
    torch.manual_seed(42)
    options={'conv_channels':[64,32],'pooling':'mean','hidden':64,'dropout':0.,'output':'linear'}
    old=legacy('model').SpatialCNN(options)
    new=SpatialCNN(ModelConfig(dropout=0))
    new.load_state_dict(old.state_dict(),strict=True)
    batch=batch_for();a=old(batch);b=new(batch)
    torch.testing.assert_close(a,b,rtol=0,atol=0)
    a.square().mean().backward();b.square().mean().backward()
    for (an,ap),(bn,bp) in zip(old.named_parameters(),new.named_parameters()):
        assert an==bn
        torch.testing.assert_close(ap.grad,bp.grad,rtol=0,atol=0)
    inference=dict(batch);inference.pop('target')
    torch.testing.assert_close(new(inference),a.detach(),rtol=0,atol=0)


def test_mean_mlp_default_parity_and_custom_width_are_honored():
    config=ModelConfig(architecture='mean_mlp',dropout=0)
    old=legacy('model').MeanMLP({'dropout':0.,'output':'linear'})
    model=MeanMLP(config);model.load_state_dict(old.state_dict(),strict=True)
    batch=batch_for()
    torch.testing.assert_close(model(batch),old(batch),rtol=0,atol=0)
    custom=MeanMLP(replace(config,hidden=48,conv_channels=(64,24)))
    assert custom.description()['hidden_dimensions']==[48,24]
    assert custom.description()['trainable_parameters']!=model.description()['trainable_parameters']
    assert custom(batch).shape==(2,)


def moments_frame():
    return pd.DataFrame({'cell_id':['a','b'],'partition':['train','train'],'n_valid':[2,6],
        'mean':[np.array([1.,2.],np.float32),np.array([3.,4.],np.float32)],
        'variance':[np.array([.5,1.],np.float32),np.array([2.,3.],np.float32)]})


def test_training_only_normalizers_preserve_legacy_moments():
    frame=moments_frame()
    old=legacy('dataset').fit_normalizer(frame)
    state=fit_normalizer(frame,'pixel_weighted_standard')
    np.testing.assert_array_equal(old['mean'],state['mean']);np.testing.assert_array_equal(old['std'],state['std'])
    assert state['training_ids']==['a','b']
    cell=fit_normalizer(frame,'cell_standard');np.testing.assert_array_equal(cell['mean'],[2.,3.])
    none=fit_normalizer(frame,'none');np.testing.assert_array_equal(none['std'],[1.,1.])
    val=frame.copy();val.loc[0,'partition']='test'
    with pytest.raises(ValueError,match='training'):fit_normalizer(val,'pixel_weighted_standard')
    with pytest.raises(ValueError,match='vettori'):transform(np.zeros((2,2,2)),np.ones((2,2),bool),cell)
    x=np.ones((2,2,2),np.float32);mask=np.ones((2,2),bool);mask[0,0]=False;x[:,0,0]=np.nan
    assert np.isfinite(transform(x,mask,state)).all()


@pytest.mark.parametrize('kernel',[1,3])
def test_patch_fragmentation_records_context_sensitivity(kernel,tmp_path):
    torch.manual_seed(7)
    x=np.random.default_rng(3).normal(size=(32,8,8)).astype(np.float32)
    model=SpatialCNN(ModelConfig(input_channels=32,kernel=kernel,dropout=0)).eval()
    partitions=[[(0,0,8,8)],[(0,0,8,4),(0,4,8,4)],[(r,c,4,4) for r in [0,4] for c in [0,4]]]
    results=[]
    for windows in partitions:
        seen=np.zeros((8,8),np.int32);patches=[]
        for r,c,h,w in windows:
            seen[r:r+h,c:c+w]+=1
            patches.append((x[:,r:r+h,c:c+w],np.ones((h,w),bool)))
        assert np.array_equal(seen,np.ones((8,8)))
        batch=collate_cells([{'cell_id':'cell','mean':x.mean((1,2)),'patches':patches}])
        pred=model(batch);assert pred.shape==(1,) and int(batch['weights'].sum())==64
        results.append(float(pred.detach()[0]))
    delta=[v-results[0] for v in results]
    if kernel==1:
        np.testing.assert_allclose(results,results[0],rtol=0,atol=1e-6)
    else:
        assert max(abs(v) for v in delta)>1e-6  # this controlled fixture exercises changed neighborhoods
    save_json(tmp_path/'patch_sensitivity.json',{'kernel':kernel,'patch_counts':[1,2,4],'predictions':results,'delta':delta,'scientific_result':False})


def test_empty_patch_invalid_weights_and_nan_on_support_fail():
    model=SpatialCNN(ModelConfig(input_channels=32))
    batch=batch_for(32);batch['weights'][0]+=1
    with pytest.raises(ValueError,match='Pesi'):model(batch)
    batch=batch_for(32);batch['features'][0,0,1,1]=float('nan')
    with pytest.raises(ValueError,match='NaN'):model(batch)
    batch=batch_for(32);batch['mask'][0]=False
    with pytest.raises(ValueError,match='vuota'):model(batch)
