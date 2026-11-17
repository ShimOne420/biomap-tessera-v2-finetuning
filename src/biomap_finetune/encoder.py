# Scopo: forward differenziabile Medium con prefisso congelato e ultimi 1-3 blocchi per modalita trainabili.
# Fasi: raw storico -> token ufficiali -> prefisso eval/no_grad -> suffisso -> fusione -> head configurata.
# Input: serie annuali o due attivazioni di prefisso FP32; output: embedding e una predizione per cella.
# Parametri: last_n_blocks, policy norm/proiezione, dense_mean/spatial_cnn; nessuna scelta implicita.
# Risorse: microbatch pixel e checkpointing suffisso; solo checkpoint locale verificato, zero download.
# Ripresa: state_dict strict, policy/nome parametri e fingerprint salvati; cache prefisso solo prototipo.
from contextlib import nullcontext
import hashlib
import numpy as np
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from biomap_v2 import student,reference
from biomap_v2.common import WEIGHTS_SHA256,digest
from biomap_spatial.contracts import ModelConfig
from biomap_spatial.model import SpatialCNN,MeanMLP
from .io import signature,commit_arrays,load_arrays


def token_groups(raw):
    s2=np.asarray(raw['s2']);n=len(s2);mask=np.asarray(raw['mask'],dtype=bool)
    if s2.ndim!=3 or s2.shape[-1]!=10 or mask.shape!=s2.shape[:2]:raise ValueError('S2/mask incompatibili')
    values=[];days=[];valid=[]
    for key,mean,std in [('asc',student.S1A_BAND_MEAN,student.S1A_BAND_STD),('desc',student.S1D_BAND_MEAN,student.S1D_BAND_STD)]:
        x=np.asarray(raw[key],dtype=np.float32)
        if x.ndim!=3 or x.shape[0]!=n or x.shape[-1]!=2:raise ValueError('S1 shape non valida')
        valid.append(np.any(x!=0,axis=-1));values.append((x-mean)/(std+1e-9))
        d=np.asarray(raw[key+'_doys']);days.append(np.broadcast_to(d[None],x.shape[:2]) if d.ndim==1 else d)
    s1=np.concatenate(values,axis=1);s1d=np.concatenate(days,axis=1);s1valid=np.concatenate(valid,axis=1)
    d2=np.asarray(raw['s2_doys']);d2=np.broadcast_to(d2[None],s2.shape[:2]) if d2.ndim==1 else d2
    for d in [d2,*days]:
        if ((d<1)|(d>366)).any():raise ValueError('DOY fuori anno 2020: nessuna riduzione delle date')
    if not np.isfinite(s2).all() or not np.isfinite(s1).all():raise ValueError('Input non finiti')
    b2=reference._vec_get_bin_size(mask.sum(1));b1=reference._vec_get_bin_size(s1valid.sum(1))
    if ((b2==0)&(b1==0)).any():raise ValueError('Pixel privo di entrambe le modalita')
    keys=b2*1000+b1
    for key in np.unique(keys):
        idx=np.flatnonzero(keys==key);k2,k1=int(key//1000),int(key%1000)
        a=np.zeros((len(idx),max(k2,1),11),np.float32);b=np.zeros((len(idx),max(k1,1),3),np.float32)
        if k2:
            take=reference._build_source_indices(mask[idx],k2)
            gathered=np.take_along_axis(s2[idx],np.repeat(take[:,:,None],10,axis=2),axis=1)
            a[:,:,:10]=(gathered-student.S2_BAND_MEAN)/(student.S2_BAND_STD+1e-9)
            a[:,:,10]=np.take_along_axis(d2[idx],take,axis=1)
        if k1:
            take=reference._build_source_indices(s1valid[idx],k1)
            b[:,:,:2]=np.take_along_axis(s1[idx],np.repeat(take[:,:,None],2,axis=2),axis=1)
            b[:,:,2]=np.take_along_axis(s1d[idx],take,axis=1)
        yield idx,a,b


def load_medium(path):
    from pathlib import Path
    path=Path(path)
    if not path.is_file() or digest(path)!=WEIGHTS_SHA256:raise ValueError('Checkpoint locale Medium assente/SHA256 discordante; nessun download automatico')
    from biomap_v2.encoder import load_encoder
    model,args,_=load_encoder({'indicator':'msa','model':{'checkpoint':str(path),'device':'cpu'}},override=path)
    return model,args


def blocks(backbone):
    container=backbone.transformer_encoder
    if isinstance(container,nn.TransformerEncoder):return list(container.layers)
    if isinstance(container,nn.ModuleList):return list(container)
    raise ValueError('Struttura transformer non verificata: rifiutata')


class PartialMedium(nn.Module):
    def __init__(self,model,last_n_blocks=2,train_projection='auto',train_normalization='auto',mode='partial',activation_checkpointing=True):
        super().__init__();self.student=model;self.last_n=last_n_blocks;self.mode=mode;self.activation_checkpointing=activation_checkpointing
        if mode not in {'partial','frozen'} or last_n_blocks not in {1,2,3}:raise ValueError('Policy partial non valida')
        self.train_projection=train_projection is not False
        self.train_norm=train_normalization is not False
        if train_projection not in {'auto',True,False} or train_normalization not in {'auto',True,False}:raise ValueError('Policy auto/bool non valida')
        for name in ['s2_backbone','s1_backbone']:
            if len(blocks(getattr(model,name)))<last_n_blocks:raise ValueError('Numero blocchi non sufficiente')
        model.requires_grad_(False)
        if mode=='partial':
            for name in ['s2_backbone','s1_backbone']:
                b=getattr(model,name)
                for layer in blocks(b)[-last_n_blocks:]:layer.requires_grad_(True)
                b.attn_pool.requires_grad_(True)
                norm=getattr(b.transformer_encoder,'norm',None)
                if norm is not None:norm.requires_grad_(True)
            if self.train_projection:model.dim_reducer.requires_grad_(True)
            if not self.train_norm:
                for module in model.modules():
                    if isinstance(module,(nn.LayerNorm,nn.RMSNorm)):module.requires_grad_(False)
        self.train(False)

    def train(self,mode=True):
        super().train(mode);self.student.eval()
        if mode and self.mode=='partial':
            for name in ['s2_backbone','s1_backbone']:
                b=getattr(self.student,name)
                for layer in blocks(b)[-self.last_n:]:layer.train(True)
                b.attn_pool.train(True)
            if self.train_projection:self.student.dim_reducer.train(True)
        return self

    def prefix(self,s2,s1):
        result={}
        with torch.no_grad():
            for name,x in [('s2_backbone',s2),('s1_backbone',s1)]:
                b=getattr(self.student,name)
                value=b.embedding(x[:,:,:-1])+b.temporal_encoder(x[:,:,-1])
                for layer in blocks(b)[:-self.last_n]:value=layer(value)
                result[name]=value.detach()
        return result

    def suffix(self,activations):
        vectors=[]
        for name in ['s2_backbone','s1_backbone']:
            b=getattr(self.student,name);value=activations[name]
            for layer in blocks(b)[-self.last_n:]:
                enabled=self.activation_checkpointing and self.training and torch.is_grad_enabled() and any(p.requires_grad for p in layer.parameters())
                value=checkpoint(layer,value,use_reentrant=False) if enabled else layer(value)
            norm=getattr(b.transformer_encoder,'norm',None)
            if norm is not None:value=norm(value)
            vectors.append(b.attn_pool(value))
        return self.student.dim_reducer(torch.cat(vectors,dim=-1))

    def forward(self,s2,s1):return self.suffix(self.prefix(s2,s1))

    def encode_raw(self,raw,pixel_batch=8):
        device=next(self.parameters()).device;parts=[];order=[]
        for idx,a,b in token_groups(raw):
            for low in range(0,len(idx),pixel_batch):
                parts.append(self(torch.from_numpy(a[low:low+pixel_batch]).to(device),torch.from_numpy(b[low:low+pixel_batch]).to(device)))
                order.extend(idx[low:low+pixel_batch].tolist())
        return torch.cat(parts,0)[np.argsort(order)]

    def policy(self):
        trainable=[n for n,p in self.student.named_parameters() if p.requires_grad]
        return {'last_n_blocks_per_modality':self.last_n,'encoder_mode':self.mode,'output_projection_trainable':self.mode=='partial' and self.train_projection,
            'normalization_policy':'affine norms only within enabled suffix/fusion' if self.train_norm else 'all affine norms frozen',
            'trainable_names':trainable,'trainable_parameters':sum(p.numel() for p in self.student.parameters() if p.requires_grad),
            'frozen_parameters':sum(p.numel() for p in self.student.parameters() if not p.requires_grad),
            'blocks':{n:[f'{n}.transformer_encoder.'+('layers.' if isinstance(getattr(self.student,n).transformer_encoder,nn.TransformerEncoder) else '')+str(i) for i in range(len(blocks(getattr(self.student,n)))-self.last_n,len(blocks(getattr(self.student,n))))] for n in ['s2_backbone','s1_backbone']},
            'prefix_mode':'eval/no_grad','fusion_dropout_enabled_in_train':self.mode=='partial' and self.train_projection,
            'fusion_dropout_training':self.training and self.mode=='partial' and self.train_projection}


class FineTuneModel(nn.Module):
    def __init__(self,encoder,aggregation,head_options=None):
        super().__init__();self.encoder=encoder;self.aggregation=aggregation
        if aggregation not in {'dense_mean','spatial_cnn'}:raise ValueError('aggregation=undecided: selezione esplicita richiesta')
        options=dict(head_options or {});options['input_channels']=128
        options['architecture']='spatial_cnn' if aggregation=='spatial_cnn' else 'mean_mlp'
        config=ModelConfig(**options)
        self.head=SpatialCNN(config) if aggregation=='spatial_cnn' else MeanMLP(config)
        self.register_buffer('embedding_mean',torch.zeros(128));self.register_buffer('embedding_std',torch.ones(128))

    def forward(self,batch,pixel_batch=8):
        outputs=[self.encoder.encode_raw(g['raw'],pixel_batch) for g in batch['groups']]
        device=self.embedding_mean.device;n=len(batch['cell_ids'])
        if not n:raise ValueError('Batch senza celle')
        if self.aggregation=='dense_mean':
            sums=torch.zeros(n,128,device=device);counts=torch.zeros(n,device=device)
            for group,emb in zip(batch['groups'],outputs):
                owner=group['owner'];sums[owner]=sums[owner]+emb.sum(0);counts[owner]+=len(emb)
            if not (counts>0).all():raise ValueError('Cella senza supporto: nessuna imputazione')
            value=(sums/counts[:,None]-self.embedding_mean)/self.embedding_std
            return self.head({'mean':value,'cell_ids':batch['cell_ids']})
        patches=[];masks=[];owners=[]
        max_h=max(p['height'] for p in batch['patches']);max_w=max(p['width'] for p in batch['patches'])
        for p in batch['patches']:
            emb=(outputs[p['group']]-self.embedding_mean)/self.embedding_std
            take=torch.as_tensor(p['take'],dtype=torch.long,device=device)
            flat=torch.as_tensor(p['flat_local'],dtype=torch.long,device=device)
            value=torch.zeros(p['height']*p['width'],128,device=device).index_copy(0,flat,emb[take])
            value=value.T.reshape(128,p['height'],p['width'])
            value=nn.functional.pad(value,(0,max_w-p['width'],0,max_h-p['height']))
            mask=torch.zeros(max_h*max_w,dtype=torch.bool,device=device)
            rr=flat//p['width'];cc=flat%p['width'];mask[rr*max_w+cc]=True
            patches.append(value);masks.append(mask.reshape(1,max_h,max_w));owners.append(p['owner'])
        mask=torch.stack(masks)
        return self.head({'features':torch.stack(patches),'mask':mask,'owners':torch.tensor(owners,dtype=torch.long,device=device),
            'weights':mask.sum((1,2,3)).to(torch.float32),'cell_ids':batch['cell_ids']})


def frozen_fingerprint(encoder):
    import hashlib
    frozen=hashlib.sha256()
    for name,value in encoder.student.named_parameters():
        if not value.requires_grad:
            frozen.update(name.encode());frozen.update(value.detach().cpu().contiguous().numpy().tobytes())
    return frozen.hexdigest()


def save_prefix_prototype(path,encoder,s2,s1,identity):
    if encoder.training:raise ValueError('Prefisso cached richiede eval deterministico')
    values=encoder.prefix(s2,s1)
    arrays={k:v.detach().float().cpu().numpy() for k,v in values.items()}
    tag=signature({'input_identity':identity,'policy':encoder.policy(),'frozen_weights':frozen_fingerprint(encoder),
        'tokens':[hashlib.sha256(v.detach().cpu().contiguous().numpy().tobytes()).hexdigest() for v in [s2,s1]],'dtype':'float32'})
    commit_arrays(path,arrays,tag,{'role':'prototype_only','last_n_blocks':encoder.last_n,'frozen_weights':frozen_fingerprint(encoder)})
    return tag


def cached_suffix(path,encoder,identity):
    arrays,info=load_arrays(path,identity)
    if info['metadata']['last_n_blocks']!=encoder.last_n or info['metadata']['frozen_weights']!=frozen_fingerprint(encoder):raise ValueError('Prefisso cached di pesi/confine diversi')
    device=next(encoder.parameters()).device
    return encoder.suffix({k:torch.from_numpy(v.copy()).to(device) for k,v in arrays.items()})
