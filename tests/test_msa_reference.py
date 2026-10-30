# Scopo: regressioni per target smoke/completo, NoData e replica del benchmark V1 Liguria.
# Fasi: raster sintetici noti -> diagnosi per causa -> recupero copertura senza cancellazioni -> fold storici.
# Input: fixture locali e manifest pubblici; output: report temporanei e assert pytest.
# Parametri/esempio: python -m pytest tests/test_msa_reference.py; nessuna rete o GPU richiesta.
# Risorse: NumPy, pandas, rasterio e sklearn; i dati sintetici non costituiscono risultati MSA reali.
# Ripresa: verifica che raster vecchi restino identici e nuove acquisizioni siano riusabili.
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import zipfile
import numpy as np
import pandas as pd
import pytest
import rasterio
from rasterio.transform import from_origin
from biomap_v2.common import ROOT,cells,config,digest,root
from biomap_v2.target_data import diagnose,ensure_target,validated_targets


def fixture(tmp_path,values,denominators=None,offset=0):
    cfg=config('msa',argparse.Namespace(config=None,data_root=str(tmp_path/'data'),set=['storage.reserve_gib=0']))
    directory=root(cfg)/'targets';directory.mkdir(parents=True,exist_ok=True)
    frame=pd.DataFrame([{'cell_id':str(i),'lon':8+(i+.5)/360,'lat':44-.5/360,
        'west':8+i/360,'east':8+(i+1)/360,'south':44-1/360,'north':44,
        'partition':['train','validation','test'][i%3],'aoi_id':'aoi_'+str(i)} for i in range(len(values))])
    path=directory/'msa_area_10sec.tif'
    for name,array,x in [(path,values,0),(path.with_name('cellarea_km2_10sec.tif'),denominators if denominators is not None else np.ones(len(values)),offset)]:
        with rasterio.open(name,'w',driver='GTiff',height=1,width=len(array),count=1,dtype='float32',crs='EPSG:4326',transform=from_origin(8+x,44,1/360,1/360),nodata=-9999) as ds:
            ds.write(np.asarray([array],dtype=np.float32),1)
    return cfg,frame,path


def test_target_masks_zero_and_outside_are_diagnosed(tmp_path):
    cfg,frame,path=fixture(tmp_path,[.2,-9999,.8,.5],[1,1,0,-9999])
    extra=frame.iloc[-1:].copy();extra['cell_id']='outside';extra['lon']+=10;extra['west']+=10;extra['east']+=10
    frame=pd.concat([frame,extra],ignore_index=True)
    result,directory=diagnose(cfg,frame,path)
    assert result.valid_target.tolist()==[True,False,False,False,False]
    assert result.loc[1,'numerator_nodata'] and result.loc[2,'denominator_zero']
    assert result.loc[3,'denominator_nodata'] and result.loc[4,'numerator_outside']
    assert result.loc[4,'denominator_outside']
    assert result.target.isna().sum()==4
    summary=json.loads((directory/'summary.json').read_text())
    assert summary['invalid_targets']==4 and summary['valid_targets']==1
    assert len(result)==len(frame) and result.cell_id.tolist()==frame.cell_id.tolist()
    with pytest.raises(ValueError,match='nessuna cella'):validated_targets(cfg,frame,path)


def test_misaligned_numerator_denominator_rejected(tmp_path):
    cfg,frame,path=fixture(tmp_path,[.2,.5,.8],offset=1/720)
    with pytest.raises(ValueError,match='co-registrati'):diagnose(cfg,frame,path)


def test_smoke_crop_is_replaced_by_new_export_without_overwrite(tmp_path,monkeypatch):
    from biomap_v2 import data
    cfg,full,path=fixture(tmp_path,[.2,.5,.8])
    full_area=path.read_bytes();full_den=path.with_name('cellarea_km2_10sec.tif').read_bytes()
    # Simulate the old smoke-only raster with only the first cell.
    _,_,path=fixture(tmp_path,[.2]);before={p.name:digest(p) for p in path.parent.glob('*.tif')}
    state={'zip':False,'starts':0}
    class Response:
        def raise_for_status(self):pass
        def json(self):return {'info':{'status':'zip ready' if state['zip'] else 'summary ready'}}
    class Session:
        def get(self,url,params,timeout):
            if params['action']=='StartSummary':state['starts']+=1
            if params['action']=='StartZip':state['zip']=True
            return Response()
    def fake_download(url,archive,cfg,params=None):
        with zipfile.ZipFile(archive,'w') as z:
            z.writestr('msa_area_10sec.tif',full_area);z.writestr('cellarea_km2_10sec.tif',full_den)
        return archive
    monkeypatch.setattr(data,'session',lambda:Session());monkeypatch.setattr(data,'download',fake_download)
    new=ensure_target(cfg,full)
    assert new!=path and 'acquisitions' in new.parts
    assert all(digest(path.parent/name)==sha for name,sha in before.items())
    np.testing.assert_allclose(validated_targets(cfg,full,new).target,[.2,.5,.8],rtol=1e-6)
    assert ensure_target(cfg,full)==new and state['starts']==1


def test_reference_manifest_exact_cells_folds_and_no_fixed_validation():
    from biomap_v2.benchmark import validate_protocol
    cfg=config('msa',argparse.Namespace(config=str(ROOT/'configs/msa_v1_liguria.yaml'),data_root=None,set=[]))
    frame=cells(cfg);validate_protocol(cfg,frame)
    assert frame.partition.value_counts().to_dict()=={'train':127909,'test':4991}
    reference=json.loads((ROOT/cfg['evaluation']['reference']).read_text())
    for record in reference['fold_counts']:
        development=frame[frame.partition=='train']
        assert int((development.cv_fold==record['cv_fold']).sum())==record['n_validation']
        assert int((development.cv_fold!=record['cv_fold']).sum())==record['n_train']
    assert root(cfg).name=='msa_v1_liguria'
    assert cfg['probing']['histgb_params']['learning_rate']==.05
    broken=frame.copy();broken.loc[broken.partition=='test','cv_fold']=0
    with pytest.raises(ValueError,match='Liguria'):validate_protocol(cfg,broken)


def test_legacy_smoke_geometry_explains_441_outside_cells():
    frame=pd.read_csv(ROOT/'configs/msa_cells.csv')
    small=pd.concat([frame[frame.partition==p].head(4) for p in ['train','validation','test']])
    inside=(frame.lon>=small.west.min())&(frame.lon<small.east.max())&(frame.lat>small.south.min())&(frame.lat<=small.north.max())
    assert int((~inside).sum())==441


def test_reader_retries_and_hides_signed_url(tmp_path,monkeypatch):
    from biomap_v2 import data
    monkeypatch.setattr(data.time,'sleep',lambda seconds:None)
    reader=data.Reader();monkeypatch.setattr(reader,'signed',lambda href:'https://example.invalid/a.tiff?secret=SAS_TOKEN')
    attempts=[]
    def fail_open(url):
        attempts.append(url);raise rasterio.errors.RasterioIOError('secret=SAS_TOKEN')
    monkeypatch.setattr(data.rasterio,'open',fail_open)
    with pytest.raises(RuntimeError) as error:reader.values({'id':'test-scene','assets':{'vv':{'href':'test'}}},'vv',np.array([[8,44]]))
    assert len(attempts)==3 and 'SAS_TOKEN' not in str(error.value)
