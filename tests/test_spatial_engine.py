# Scopo: verificare end-to-end la nuova testa su artefatti sintetici r1 e proteggere selezione/ripresa.
# Fasi: prepare -> preflight -> CV -> refit -> evaluate -> registry; confronto checkpoint e guardie GPU.
# Input: otto celle/32 pixel firmati; output: report temporanei e assert, nessun esperimento scientifico.
# Parametri: due epoche CPU e dropout controllato. Uso: pytest tests/test_spatial_engine.py.
# Risorse: pochi MB; nessun encoder, dataset o rete. Ripresa deterministica da epoca completata.
from dataclasses import replace
from pathlib import Path
import json
import pickle
import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from biomap_spatial.sources import V2Source,prepare_index
from biomap_spatial.engine import train,evaluate,fit_one,preflight,registry,compare_runs,predict,make_loader
from biomap_spatial.normalization import fit_normalizer
from biomap_spatial.cli import main,configuration,parser
from biomap_spatial.utils import digest
from spatial_fixtures import make_artifacts


def ready(tmp_path):
    spec,ref,work,out=make_artifacts(tmp_path)
    source=V2Source(spec,out/'index',False);prepare_index(source)
    return spec,source


def test_cv_refit_external_registry_and_inference_without_labels(tmp_path):
    spec,source=ready(tmp_path)
    from pathlib import Path
    before={str(p):digest(p) for root in [source.root,source.reference] for p in root.rglob('*') if p.is_file()}
    try:
        run=train(spec,source,'cnn','cv')
        selection=json.loads((run/'selection.json').read_text())
        assert selection['liguria_used'] is False
        assert len(pd.read_parquet(run/'predictions.parquet'))==12  # six dev, predictor + Dummy
        original=(run/'selection.json').read_bytes()
        changed=dict(selection,refit_epochs=selection['refit_epochs']+1)
        (run/'selection.json').write_text(json.dumps(changed))
        with pytest.raises(ValueError,match='Selezione'):train(spec,source,'cnn','refit')
        (run/'selection.json').write_bytes(original)
        train(spec,source,'cnn','refit')
        saved=torch.load(run/'checkpoint.pt',weights_only=False)
        assert set(saved['normalizer']['training_ids'])==set(source.frame[source.frame.partition=='train'].cell_id)
        assert not set(saved['normalizer']['training_ids'])&set(source.frame[source.frame.partition=='test'].cell_id)
        rows=evaluate(spec,source,'cnn')
        assert rows['predictor']['n']==2 and (run/'complete.json').exists()
        assert json.loads((run/'manifest.json').read_text())['state']=='complete'
        old=digest(source.index)
        clone=pickle.loads(pickle.dumps(source))
        assert clone.patches('c0')[0].features.shape==(128,2,2)
        clone.close();assert digest(source.index)==old
        from biomap_spatial.model import make_model
        model=make_model(spec.model);model.load_state_dict(saved['model'])
        frame=source.frame.drop(columns='target').iloc[:2]
        results=predict(model,make_loader(source,frame,saved['normalizer'],spec),'cpu')
        assert set(results)==set(frame.cell_id)
    finally:source.close()
    assert before=={name:digest(Path(name)) for name in before}


def test_determinism_environment_is_set_without_cuda_initialization():
    import subprocess,sys
    code="import os; os.environ.pop('CUBLAS_WORKSPACE_CONFIG',None); import biomap_spatial.engine; import torch; assert os.environ['CUBLAS_WORKSPACE_CONFIG']==':4096:8'; assert not torch.cuda.is_initialized()"
    subprocess.run([sys.executable,'-c',code],check=True)


def test_epoch_resume_is_exact_and_feature_space_change_is_rejected(tmp_path):
    spec,source=ready(tmp_path)
    spec=replace(spec,model=replace(spec.model,dropout=.2))
    dev=source.frame[source.frame.partition=='train'];tr=dev[dev.cv_fold!=0];val=dev[dev.cv_fold==0]
    try:
        fit_one(spec,source,tr,val,tmp_path/'whole','same')
        _,_,_,done=fit_one(spec,source,tr,val,tmp_path/'part','same',epoch_limit=1)
        assert done is False
        fit_one(spec,source,tr,val,tmp_path/'part','same',resume=True)
        a=torch.load(tmp_path/'whole/last.pt',weights_only=False);b=torch.load(tmp_path/'part/last.pt',weights_only=False)
        for key in a['model']:torch.testing.assert_close(a['model'][key],b['model'][key],rtol=0,atol=0)
        with pytest.raises(ValueError,match='incompatibile'):fit_one(spec,source,tr,val,tmp_path/'part','other',resume=True)
        registry(spec,source,'guard',True)
        original=source.feature_space_id;source.feature_space_id='other-encoder'
        with pytest.raises(ValueError,match='feature space'):registry(spec,source,'guard')
        source.feature_space_id=original
    finally:source.close()


def test_cli_prepare_preflight_profile_and_negative_device_gate(tmp_path,monkeypatch):
    spec,ref,work,out=make_artifacts(tmp_path)
    config=tmp_path/'config.yaml';config.write_text(yaml.safe_dump(spec.to_dict()))
    args=['--config',str(config)]
    assert main(['doctor',*args])==0
    assert main(['prepare',*args])==0
    assert main(['preflight',*args])==0
    assert main(['profile',*args])==0
    assert json.loads((out/'profile.json').read_text())['scientific_result'] is False
    assert main(['monitor',*args])==0
    parsed=parser().parse_args(['doctor',*args,'--kernel','1','--seed','43'])
    changed=configuration(parsed);assert changed.model.kernel==1 and changed.protocol.seed==43
    with pytest.raises(ValueError,match='sconosciuto'):configuration(parser().parse_args(['doctor',*args,'--set','model.typo=1']))
    (work/'pixel_embeddings/aoi_0/grid_test.npy').unlink()
    def forbidden():pytest.fail('GPU queried before invalid artifacts were rejected')
    monkeypatch.setattr(torch.cuda,'is_available',forbidden)
    with pytest.raises(ValueError,match='mancante'):preflight(replace(spec,execution={**spec.execution,'device':'cuda:0'}))


def test_kernel_ablation_comparison_requires_same_seed_and_cohort(tmp_path):
    spec,source=ready(tmp_path)
    try:
        train(spec,source,'spatial3','cv')
        pixel=replace(spec,model=replace(spec.model,kernel=1))
        train(pixel,source,'pixel1','cv')
        mean=replace(spec,model=replace(spec.model,architecture='mean_mlp'))
        train(mean,source,'mean','cv')
        rows=compare_runs(spec,source,['spatial3','pixel1','mean'],'ablation')
        assert len(rows)==3 and rows[0]['macro_aoi_rmse']<=rows[-1]['macro_aoi_rmse']
        manifest_path=Path(spec.execution['output_root'])/'mean/manifest.json'
        mark=json.loads(manifest_path.read_text());mark['protocol']['seed']=44;manifest_path.write_text(json.dumps(mark))
        with pytest.raises(ValueError,match='seed'):compare_runs(spec,source,['spatial3','mean'],'bad')
    finally:source.close()


@pytest.mark.parametrize('channels',[32,128,384])
def test_array_source_training_uses_no_encoder_and_variable_channels(tmp_path,channels):
    from biomap_spatial.sources import ArraySource
    from biomap_spatial.contracts import SpatialSample
    spec,_,_,_=make_artifacts(tmp_path,channels=channels)
    spec=replace(spec,training={**spec.training,'epochs':1})
    rng=np.random.default_rng(1);samples={};records=[]
    for i in range(2):
        x=rng.normal(size=(channels,3,3)).astype(np.float32);mask=np.ones((3,3),bool)
        samples[str(i)]=[SpatialSample(x,mask,{'feature_space_id':'array-fixture'})]
        records.append({'cell_id':str(i),'partition':'train','target':.2+i*.2,'n_valid':9,'mean':x.mean((1,2)),'variance':x.var((1,2))})
    source=ArraySource(samples,'array-fixture');frame=pd.DataFrame(records)
    model,norm,_,done=fit_one(spec,source,frame,None,tmp_path/'array_run','array-identity')
    assert done and len(predict(model,make_loader(source,frame.drop(columns='target'),norm,spec),'cpu'))==2


