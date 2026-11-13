# Scopo: fixture minuscole degli schemi r1/r2 reali, senza encoder, acquisizioni o dati universitari.
# Fasi: tre AOI sviluppo + test -> geometrie/lookup -> embedding e chunk -> reference/marker hash.
# Input: tmp_path, protocollo e C; output: radici e specifica synthetic con otto celle e 32 pixel.
# Uso: make_artifacts(tmp_path). Risorse: kilobyte CPU; ripresa: ogni test usa una directory privata.
import json
from pathlib import Path
import numpy as np
import pandas as pd
from biomap_v2.common import WEIGHTS_SHA256
from biomap_spatial.contracts import ExperimentSpec, ModelConfig, ExperimentProtocol
from biomap_spatial.sources import R1, R2
from biomap_spatial.utils import ROOT, digest, save_json


def make_artifacts(tmp_path, protocol=R1, channels=128):
    ref, work, out = tmp_path/'reference',tmp_path/'source',tmp_path/'output'
    ref.mkdir();work.mkdir()
    contract=json.loads((ROOT/'tests/fixtures/r1_contract_1cdc762.json').read_text())
    code={k:v['sha256_worktree'] for k,v in contract['files'].items()}
    rng=np.random.default_rng(42)
    areas=[];records=[];means=[]
    for area in range(4):
        aoi=f'aoi_{area}';name='grid_test';tile=ref/aoi/name;tile.mkdir(parents=True)
        flat=np.arange(8,dtype=np.int64);ids=np.array([0,0,1,1,0,0,1,1],dtype=np.int64)
        np.savez(tile/'geometry.npz',flat_indices=flat,cell_ids=ids)
        info={'height':2,'width':4,'pixels':8,'transform':[10.,0.,500000.+area*1000,0.,-10.,4800000.],'crs':'EPSG:32632','edge_op':'trunc'}
        save_json(tile/'geometry.json',info)
        np.save(ref/aoi/'lookup.npy',np.array([area*2,area*2+1],dtype=np.int64))
        values=rng.normal(size=(8,channels)).astype(np.float32)
        dest=work/'pixel_embeddings'/aoi/(name+'.npy');dest.parent.mkdir(parents=True)
        np.save(dest,values)
        identity={'protocol':protocol,'geometry':digest(tile/'geometry.npz'),'year':2020,
            'checkpoint':WEIGHTS_SHA256,'precision':'fp32','code':code if protocol==R1 else {'fixture_r2':'verified_only_in_fixture'}}
        meta={'sha256':digest(dest),'order':'geometry.npz flat_indices','dtype':'float32'}
        if protocol==R1:meta['geometry']=identity['geometry']
        else:meta.update(identity=identity,geometry_sha256=identity['geometry'],grid={k:info[k] for k in ['width','height','transform','crs']})
        save_json(dest.with_suffix('.json'),meta)
        sums=np.stack([values[ids==i].astype(np.float64).sum(0) for i in range(2)])
        chunk=work/'chunks'/aoi/name/'tile.npz';chunk.parent.mkdir(parents=True)
        np.savez(chunk,cell=np.array([area*2,area*2+1]),sums=sums,counts=np.array([4,4]))
        save_json(chunk.with_suffix('.json'),{'identity':identity,'sha256':digest(chunk)})
        areas.append({'aoi_id':aoi,'tiles':[name]})
        for i in range(2):
            record={'cell_id':f'c{area*2+i}','reference_index':area*2+i,'aoi_id':aoi,'partition':'train' if area<3 else 'test',
                'cv_fold':area if area<3 else -1,'target':.2+.1*i,'n_valid':4,'n_total':4,'valid_fraction':1.,'lon':9.+area*.1+i*.001,'lat':44.}
            records.append(record);means.append(sums[i]/4)
    frame=pd.DataFrame(records)
    frame.to_parquet(ref/'v1.parquet',index=False)
    v2=pd.concat([frame,pd.DataFrame(np.stack(means),columns=[f'mean_{k:03d}' for k in range(channels)])],axis=1)
    v2.to_parquet(work/'v2.parquet',index=False)
    save_json(ref/'reference.json',{'protocol':'dense127909-public-r1','areas':areas,
        'files':{str(p.relative_to(ref)):digest(p) for p in ref.rglob('*') if p.is_file()}})
    save_json(work/'extraction_complete.json',{'sha256':digest(work/'v2.parquet'),'amp':False,'n_cells':8,'complete_campaign':True,'method':protocol})
    spec=ExperimentSpec({'kind':'v2_pixels','root':str(work),'reference':str(ref),'feature_space_id':'synthetic-v2',
        'protocol':protocol,'max_patch_side':64,'mmap_limit':2,'resolution':[10,10]},ModelConfig(input_channels=channels,dropout=0),
        ExperimentProtocol(profile='synthetic'),
        {'epochs':2,'patience':2,'batch_cells':2,'learning_rate':.001,'weight_decay':.0001,'clip_grad':1.,'num_workers':0,'mixed_precision':False},
        {'device':'cpu','output_root':str(out),'reserve_gib':0})
    return spec,ref,work,out
