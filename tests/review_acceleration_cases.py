# Scopo: riprodurre criticità della revisione 2e24f97 senza cambiare codice o dati di produzione.
# Fasi: fixture locali -> interruzione/ripresa -> quote concorrenti -> avanzamento -> letture NPZ.
# Input: checkout fissato da BIOMAP_AUDIT_REPO; output: pytest e righe AUDIT_EVIDENCE in stdout.
# Parametri: python -m pytest -s questo_file.py; richiede le dipendenze del repository.
# Risorse: CPU, GeoTIFF sintetici e file sparsi temporanei; nessuna rete o GPU.
# Ripresa: ogni caso usa tmp_path; nessuna cache o run universitario viene letto o modificato.
# Uso diagnostico: le asserzioni documentano difetti presenti in 2e24f97, non criteri di accettazione.
# Raccolta: il nome review_* esclude questo file dalla suite ordinaria; eseguirlo solo esplicitamente.
# Dopo le correzioni questi casi possono fallire: aggiornarli come vere regressioni prima di inserirli in CI.
import sys
import os
from pathlib import Path
from contextlib import nullcontext
from collections import Counter
import json
import time
import numpy as np
import pytest

REPO=Path(os.environ.get('BIOMAP_AUDIT_REPO',str(Path(__file__).resolve().parents[1])))
sys.path[:0]=[str(REPO/'src'),str(REPO/'tests')]
import test_preparation_performance as fixtures
from biomap_v2 import data,parallel_prepare
from biomap_v2.common import root
from biomap_v2.preparation import BatchAcquirer
from biomap_v2.preparation_profile import SpaceGuard

observations=fixtures.observations


def evidence(name,**values):print('AUDIT_EVIDENCE',json.dumps({'case':name,**values}))


class Profile:
    def span(self,*args):return nullcontext()
    def cell(self,**kwargs):pass
    def flush(self):pass


def test_document_checkpoint_accepts_wrong_dtype(tmp_path,observations,monkeypatch):
    frame,csv,args,items,calls=observations
    cfg=fixtures.make_cfg(tmp_path);cfg['cells']=str(csv)
    cfg['acquisition'].update(block_cache_gib=0,memory_cache_mib=0,day_workers=1)
    original=BatchAcquirer.read_bands
    def interrupted(self,item,bands,points):
        if '2020-12-31' in item['id']:raise KeyboardInterrupt()
        return original(self,item,bands,points)
    monkeypatch.setattr(BatchAcquirer,'read_bands',interrupted)
    with pytest.raises(KeyboardInterrupt):data.prepare(cfg,args)
    path=next((root(cfg)/'batch_checkpoints').glob('*/*/s2_2020-06-01.npz'))
    with np.load(path,allow_pickle=False) as saved:values={key:saved[key] for key in saved.files}
    original_dtype=str(values['bands'].dtype)
    values['bands']=values['bands'].astype(np.float32)
    values['bands'][0,0,0]+=100
    np.savez_compressed(path,**values)
    monkeypatch.setattr(BatchAcquirer,'read_bands',original)
    result=data.prepare(cfg,args)
    with np.load(data.cell_directory(cfg,'0')/'series.npz',allow_pickle=False) as saved:final_dtype=str(saved['s2'].dtype)
    assert final_dtype=='float32' and result.exists()
    evidence('checkpoint_dtype',original=original_dtype,modified='float32',final=final_dtype,prepare_accepted=True)


def test_document_reused_slot_can_exceed_global_quota(tmp_path):
    cfg=fixtures.make_cfg(tmp_path)
    cfg['storage'].update(max_gib=200/1024,max_new_gib=100/1024,reserve_gib=0)
    a=SpaceGuard(cfg,Profile());b=SpaceGuard(cfg,Profile())
    def write(guard,name,mib):
        size=mib*1024**2;guard.check(size)
        with (Path(cfg['data_root'])/name).open('wb') as stream:stream.truncate(size)
        guard.account(size)
    write(a,'area_a',90)
    c=SpaceGuard(cfg,Profile())
    write(c,'area_c',75);write(b,'area_b',100)
    used=sum(path.stat().st_size for path in Path(cfg['data_root']).iterdir() if path.is_file())/1024**2
    assert used==265 and all(g.used<=g.cap for g in [a,b,c])
    evidence('global_quota',cap_mib=200,observed_mib=used,all_local_checks_passed=True)


def test_document_restart_progress_discards_previous_worker_totals(tmp_path):
    first=tmp_path/'20261107T000001000000Z_area_prepare';first.mkdir()
    (first/'progress.json').write_text(json.dumps({'new_cells':200,'cached_cells':0,'batch_days_total':10,'batch_days_done':10,'batch_cells':200,'counters':{'remote_blocks_read':1000}}))
    second=tmp_path/'20261107T000002000000Z_area_prepare';second.mkdir()
    (second/'progress.json').write_text(json.dumps({'new_cells':800,'cached_cells':200,'batch_days_total':10,'batch_days_done':10,'batch_cells':800,'status':'complete','counters':{'remote_blocks_read':2000}}))
    state=parallel_prepare.summary(tmp_path,['area'],{'area':1000},time.monotonic()-1000,{},{'area':1},(300,0),'20261107T000000000000Z')
    assert state['saved_cells']==800 and state['estimated_cells']==800 and state['remote_blocks_read']==2000
    evidence('restart_progress',actual_saved=1000,reported_saved=state['saved_cells'],actual_blocks=3000,reported_blocks=state['remote_blocks_read'],eta_hours=state['eta_hours'])


def test_document_consolidation_redecodes_npz_members(tmp_path,monkeypatch):
    cfg=fixtures.make_cfg(tmp_path);n=cfg['sampling']['pixels_per_cell'];count=513
    from types import SimpleNamespace
    cells=[SimpleNamespace(cell_id=str(i),west=8.,south=43.9,east=8.001,north=43.901) for i in range(count)]
    checkpoint=tmp_path/'checkpoint';checkpoint.mkdir()
    np.savez_compressed(checkpoint/'s2_2020-06-01.npz',bands=np.zeros((count,n,10),dtype=np.uint16),valid=np.ones((count,n),dtype=bool),scene_ids=np.asarray(['scene']))
    reader=object.__new__(BatchAcquirer);reader.cfg=cfg;reader.profile=Profile()
    reader.finish_cell=lambda *args:{'fraction':1.,'sha256':'fixture','cached':False}
    original=np.lib.npyio.NpzFile.__getitem__;reads=Counter()
    def getitem(self,key):reads[key]+=1;return original(self,key)
    monkeypatch.setattr(np.lib.npyio.NpzFile,'__getitem__',getitem)
    result=reader.consolidate(cells,[tmp_path/str(i) for i in range(count)],checkpoint,[str(i) for i in range(count)],{'s2':{'2020-06-01':[{'id':'scene'}]}})
    assert len(result)==count and reads['bands']==4 and reads['valid']==4
    evidence('consolidation_decode',cells=count,chunks=2,bands_full_decodes=reads['bands'],valid_full_decodes=reads['valid'])


def test_document_complete_status_precedes_final_validation(tmp_path,observations,monkeypatch):
    from types import SimpleNamespace
    from biomap_v2 import target_data
    frame,csv,args,items,calls=observations
    cfg=fixtures.make_cfg(tmp_path);cfg['cells']=str(csv);cfg['acquisition']['block_cache_gib']=0
    Path(cfg['data_root']).mkdir(parents=True,exist_ok=True)
    monkeypatch.setattr(target_data,'ensure_target',lambda *a:None)
    monkeypatch.setattr(parallel_prepare,'pending_cells',lambda *a:{})
    def failed_validation(*args):raise ValueError('audit: final validation failed')
    monkeypatch.setattr(data,'prepare',failed_validation)
    options=SimpleNamespace(aois=None,target_raster=None,parallel_aois=4,stop_after_minutes=None,aoi=None)
    with pytest.raises(ValueError,match='final validation failed'):parallel_prepare.run_parallel(cfg,options)
    report=next((root(cfg)/'diagnostics').glob('*_parallel/parallel_progress.json'))
    state=json.loads(report.read_text())
    assert state['status']=='complete'
    evidence('final_status',final_validation='failed',reported_status=state['status'])
