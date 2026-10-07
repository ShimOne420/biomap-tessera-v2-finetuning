# Scopo: caricare pesi verificati e usare un forward V2 differenziabile per cella.
# Fasi: normalizzazione ufficiale, binning temporale, encoder pixel, media alla cella.
# Input: series.npz e checkpoint Medium; output: tensore 128D, gradienti o feature CSV.
# Parametri: model.device, model.pixel_batch, model.checkpoint e training.unfreeze_blocks.
# Esempio: finetune.py export-embeddings --indicator msa --checkpoint RUN/best.pt.
# Risorse: CPU per test, CUDA per campagna; microbatch pixel e checkpointing delle attivazioni.
# Ripresa: pesi scaricati con revision/SHA256 fissati; feature identificate dal checkpoint.
from pathlib import Path
import json
import os
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint as activation_checkpoint
from . import student
from .reference import _build_source_indices, _vec_get_bin_size
from .common import WEIGHTS_REVISION, WEIGHTS_SHA256, budget, digest, root, save_json
from .data import cell_directory


def weights(cfg):
    directory=Path(cfg['data_root'])/'weights';directory.mkdir(parents=True,exist_ok=True)
    path=directory/'student_medium.pt'
    if not path.exists():
        from huggingface_hub import hf_hub_download
        budget(cfg,100*1024**2)
        hf_hub_download('geotessera/TESSERA-V-2.0-2B-M','ckpt/student_medium.pt',
                        revision=WEIGHTS_REVISION,local_dir=directory)
        os.replace(directory/'ckpt/student_medium.pt',path)
    if digest(path)!=WEIGHTS_SHA256:raise ValueError('SHA256 dei pesi Medium discordante')
    save_json(directory/'provenance.json',{'repository':'geotessera/TESSERA-V-2.0-2B-M',
              'revision':WEIGHTS_REVISION,'sha256':WEIGHTS_SHA256})
    return path


def load_encoder(cfg,override=None):
    path=Path(override or cfg['model']['checkpoint']) if (override or cfg['model']['checkpoint']) else weights(cfg)
    if path.name=='student_medium.pt' and digest(path)!=WEIGHTS_SHA256:
        raise ValueError('Checkpoint Medium non verificato')
    payload=torch.load(path,map_location='cpu',weights_only=False)
    if 'fingerprint' in payload and payload['fingerprint']['indicator']!=cfg['indicator']:
        raise ValueError('Checkpoint adattato per un altro indicatore: usare la relativa configurazione')
    args=payload.get('args',{})
    model=student.PixelStudent(repr_dim=int(args.get('repr_dim',128)),latent_dim=int(args.get('latent_dim',64)),
        num_layers=int(args.get('num_layers',4)),nhead=int(args.get('nhead',4)),
        dim_feedforward=int(args.get('dim_feedforward',1024)),dropout=0,
        max_seq_len=int(args.get('max_seq_len',256)),enable_qk_norm=bool(args.get('enable_qk_norm',False)))
    model.load_state_dict(payload.get('encoder',payload.get('model',payload.get('model_state_dict',payload.get('state_dict',{})))),strict=True)
    model.to(cfg['model']['device']).eval()
    return model,dict(args),path


def groups(raw):
    s2=raw['s2'];mask=raw['mask'].astype(bool);n=len(s2)
    if s2.ndim!=3 or s2.shape[-1]!=10 or mask.shape!=s2.shape[:2]:raise ValueError('S2 shape/mask non valida')
    s2d=np.broadcast_to(raw['s2_doys'][None,:],s2.shape[:2])
    normalized=[];days=[];valid=[]
    for name,mean,std in [('asc',student.S1A_BAND_MEAN,student.S1A_BAND_STD),('desc',student.S1D_BAND_MEAN,student.S1D_BAND_STD)]:
        arr=raw[name].astype(np.float32)
        if arr.ndim!=3 or arr.shape[0]!=n or arr.shape[-1]!=2:raise ValueError('S1 shape non valida')
        valid.append(np.any(arr!=0,axis=-1))
        normalized.append((arr-mean)/(std+1e-9))
        days.append(np.broadcast_to(raw[name+'_doys'][None,:],arr.shape[:2]))
    s1=np.concatenate(normalized,axis=1);s1d=np.concatenate(days,axis=1);s1v=np.concatenate(valid,axis=1)
    if not np.isfinite(s2).all() or not np.isfinite(s1).all():raise ValueError('Serie non finite')
    for doy in [raw['s2_doys'],raw['asc_doys'],raw['desc_doys']]:
        if ((doy<1)|(doy>365)).any():raise ValueError('DOY fuori [1,365]')
    b2=_vec_get_bin_size(mask.sum(axis=1));b1=_vec_get_bin_size(s1v.sum(axis=1))
    if ((b2==0)&(b1==0)).any():raise ValueError('Pixel privo di entrambe le modalità')
    keys=b2*1000+b1
    for key in np.unique(keys):
        idx=np.where(keys==key)[0];k2=int(key//1000);k1=int(key%1000)
        x2=np.zeros((len(idx),max(k2,1),11),dtype=np.float32);x1=np.zeros((len(idx),max(k1,1),3),dtype=np.float32)
        if k2:
            source=_build_source_indices(mask[idx],k2)
            x2[:,:,:10]=(np.take_along_axis(s2[idx],source[:,:,None],axis=1)-student.S2_BAND_MEAN)/(student.S2_BAND_STD+1e-9)
            x2[:,:,10]=np.take_along_axis(s2d[idx],source,axis=1)
        if k1:
            source=_build_source_indices(s1v[idx],k1)
            x1[:,:,:2]=np.take_along_axis(s1[idx],source[:,:,None],axis=1)
            x1[:,:,2]=np.take_along_axis(s1d[idx],source,axis=1)
        yield idx,x2,x1


def pixel_embeddings(model,raw,pixel_batch=128,checkpointing=False):
    device=next(model.parameters()).device;parts=[];order=[]
    for idx,x2,x1 in groups(raw):
        for start in range(0,len(idx),pixel_batch):
            a=torch.from_numpy(x2[start:start+pixel_batch]).to(device)
            b=torch.from_numpy(x1[start:start+pixel_batch]).to(device)
            emb=activation_checkpoint(model,a,b,use_reentrant=False) if checkpointing and torch.is_grad_enabled() and any(p.requires_grad for p in model.parameters()) else model(a,b)
            parts.append(emb);order.extend(idx[start:start+pixel_batch])
    return torch.cat(parts,dim=0)[np.argsort(order)]


def cell_embedding(model,path,cfg):
    with np.load(path) as z:raw={k:z[k] for k in z.files}
    emb=pixel_embeddings(model,raw,cfg['model']['pixel_batch'],cfg['training']['activation_checkpointing'])
    return emb.mean(dim=0)


class Regression(nn.Module):
    def __init__(self,encoder):
        super().__init__();self.encoder=encoder
        self.head=nn.Sequential(nn.Linear(128,64),nn.ReLU(),nn.Linear(64,1),nn.Sigmoid())
    def forward_cell(self,path,cfg):
        return self.head(cell_embedding(self.encoder,path,cfg)).squeeze(-1)


def trainable(model,stage,last_blocks=1):
    if stage not in {'head','partial','full'}:raise ValueError('Stage non valido')
    for p in model.encoder.parameters():p.requires_grad_(stage=='full')
    for p in model.head.parameters():p.requires_grad_(True)
    if stage=='partial':
        for backbone in [model.encoder.s1_backbone,model.encoder.s2_backbone]:
            layers=backbone.transformer_encoder
            layers=layers if isinstance(layers,nn.ModuleList) else layers.layers
            if last_blocks<1 or last_blocks>len(layers):raise ValueError('unfreeze_blocks fuori intervallo')
            for layer in list(layers)[-last_blocks:]:
                for p in layer.parameters():p.requires_grad_(True)
            for p in backbone.attn_pool.parameters():p.requires_grad_(True)
        for p in model.encoder.dim_reducer.parameters():p.requires_grad_(True)
    return [name for name,p in model.named_parameters() if p.requires_grad]


def embeddings(cfg,frame,checkpoint=None):
    encoder,args,path=load_encoder(cfg,checkpoint)
    encoder.eval()
    filename=root(cfg)/f'embeddings_{digest(path)[:16]}.csv'
    sidecar=filename.with_suffix('.json')
    fingerprint={'checkpoint_sha256':digest(path),'cells':frame[['cell_id','series_sha256']].to_dict('records')}
    if filename.exists() and sidecar.exists():
        cached=json.loads(sidecar.read_text())
        if cached['fingerprint']==fingerprint and cached['sha256']==digest(filename):
            return pd.read_csv(filename,dtype={'cell_id':str}),path
    rows=[]
    with torch.no_grad():
        for i,cell in enumerate(frame.itertuples()):
            emb=cell_embedding(encoder,cell_directory(cfg,cell.cell_id)/'series.npz',cfg).float().cpu().numpy()
            if not np.isfinite(emb).all():raise ValueError('Embedding non finiti')
            rows.append({'cell_id':cell.cell_id,**{f'f{k:03d}':float(v) for k,v in enumerate(emb)}})
            if (i+1)%25==0:print(f'embeddings {i+1}/{len(frame)}',flush=True)
    features=pd.DataFrame(rows);features.to_csv(filename,index=False)
    save_json(sidecar,{'fingerprint':fingerprint,'sha256':digest(filename)})
    return features,path
