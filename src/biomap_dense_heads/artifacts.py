# Scopo: preflight read-only della matrice densa completata e del manifest geografico.
# Non importa encoder, non crea lock/cache nella sorgente, non avvia acquisizione.
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from biomap_v2.common import ROOT, digest, save_json
from biomap_v2.campaign.state import membership, MEANS

# Hash manifest/marker prima e dopo lettura: un file in scrittura non è un dato pronto.
def load(source,device,out,features=None,completion=None,targets=None,smoke=False):
    source=Path(source).resolve();out=Path(out).resolve()
    if out==source or source in out.parents or out in source.parents:
        raise ValueError('Output e sorgente devono essere cartelle separate, non annidate')
    if device.startswith('cuda') and not torch.cuda.is_available():raise ValueError('CUDA non disponibile')
    if device.startswith('mps') and not torch.backends.mps.is_available():raise ValueError('MPS non disponibile')
    torch.device(device) # sintassi verificata prima di creare output
    if device.startswith('cuda'):
        probe=torch.ones((4,4),device=device);assert float((probe@probe)[0,0])==4.;del probe
    path=Path(features).resolve() if features else source/'v2.parquet'
    marker=Path(completion).resolve() if completion else source/'extraction_complete.json'
    if not marker.is_file():raise ValueError(f'Estrazione non confermata completa: manca {marker}')
    marker_sha=digest(marker);proof=json.loads(marker.read_text())
    if proof.get('complete_campaign') is not True and not smoke:raise ValueError('Estrazione parziale: complete_campaign non true')
    if not smoke and proof.get('method')!='dense127909-stream-legacy-exact-r1':
        raise ValueError('method del marker non valido: richiesto dense127909-stream-legacy-exact-r1')
    if not path.is_file() or proof.get('sha256')!=digest(path):raise ValueError('Feature assenti, mutate o checksum non coincidente col marker')
    feature_sha=proof['sha256'];frame=pd.read_parquet(path)
    if frame.cell_id.duplicated().any():raise ValueError('cell_id duplicati')
    frame['cell_id']=frame.cell_id.astype(str)
    if frame.cell_id.duplicated().any():raise ValueError('Identificativi duplicati dopo normalizzazione')
    if targets:
        tpath=Path(targets).resolve()
        extra=pd.read_parquet(tpath) if tpath.suffix=='.parquet' else pd.read_csv(tpath,dtype={'cell_id':str})
        if 'target' in frame:raise ValueError('Target già presenti: non vengono sostituiti')
        n=len(frame);frame=frame.merge(extra[['cell_id','target']],on='cell_id',validate='one_to_one')
        if len(frame)!=n:raise ValueError('Target non completi')
        proof['targets_sha256']=digest(tpath)
    if not set(MEANS)<=set(frame):
        fcols=[f'f{i:03d}' for i in range(128)]
        if not set(fcols)<=set(frame):raise ValueError('Attese 128 mean_000..127 oppure f000..127')
        frame=frame.rename(columns=dict(zip(fcols,MEANS)))
    if 'target' not in frame:raise ValueError('Target assenti: fornire --targets senza ricostruire MSA')
    if not smoke:
        authoritative=membership();indexed=frame.set_index('cell_id')
        if set(frame.cell_id)!=set(authoritative.cell_id):raise ValueError('Membership differente da 127.909+4.991 celle congelate')
        frame=indexed.loc[authoritative.cell_id].reset_index()
        for col in ['aoi_id','partition','cv_fold','lon','lat']:
            if col not in frame:frame[col]=authoritative[col].to_numpy()
            elif col in ['lon','lat']:
                if not np.allclose(frame[col],authoritative[col],rtol=0,atol=1e-8):raise ValueError('Coordinate diverse dal manifest')
            elif not np.array_equal(frame[col].to_numpy(),authoritative[col].to_numpy()):raise ValueError(f'Split/AOI modificati: {col}')
    if not np.isfinite(frame[MEANS+['target']].to_numpy()).all():raise ValueError('Feature/target NaN o infiniti')
    if not frame.target.between(0,1).all():raise ValueError('MSA fuori [0,1]')
    if not {'train','test'}==set(frame.partition):raise ValueError('Servono sviluppo train e test separato')
    dev=frame[frame.partition=='train'];test=frame[frame.partition=='test']
    if set(dev.cv_fold)!={0,1,2} or not dev.groupby('aoi_id').cv_fold.nunique().eq(1).all():raise ValueError('Fold AOI non validi')
    if set(dev.aoi_id)&set(test.aoi_id) or not test.cv_fold.eq(-1).all():raise ValueError('Leakage sviluppo/test')
    if len(frame)!=proof.get('n_cells') or digest(path)!=feature_sha or digest(marker)!=marker_sha:raise ValueError('Artifact incompleto o modificato durante il preflight')
    out.mkdir(parents=True,exist_ok=True)
    report={'status':'ready_smoke' if smoke else 'ready','n_development':len(dev),'n_test':len(test),
        'development_aois':dev.aoi_id.nunique(),'shape':[len(frame),128],'representation':'mean channels; no spatial layout',
        'feature_sha256':feature_sha,'completion_sha256':marker_sha,'proof':proof,'device':device,
        'source':str(source),'features':str(path),'output':str(out),'smoke':smoke,'source_writes':False}
    save_json(out/'preflight.json',report)
    return frame,report
