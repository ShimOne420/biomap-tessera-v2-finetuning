# Scopo: testa CNN spaziale su V1 frozen e controllo MLP sulle medie degli stessi pixel.
# Fasi: convoluzione parziale -> LayerNorm per pixel -> GELU -> pooling mascherato -> regressione.
# Input: patch N,128,H,W + mask, patch-to-cell, pixel validi; output: una stima MSA per cella.
# Parametri: canali 64/32, hidden 64, dropout, pooling mean/max, output linear/sigmoid.
# Risorse: PyTorch CPU/CUDA; nessun encoder, padding ignorato anche nelle normalizzazioni.
# Ripresa: state_dict standard; verifiche forward/backward e invarianza del padding nei test.
import torch
from torch import nn
import torch.nn.functional as F


class PartialConv(nn.Module):
    def __init__(self,cin,cout):
        super().__init__();self.conv=nn.Conv2d(cin,cout,3,padding=1,bias=False);self.bias=nn.Parameter(torch.zeros(cout))
        self.register_buffer('ones',torch.ones(1,1,3,3))
    def forward(self,x,mask):
        x=torch.where(mask,x,torch.zeros((),dtype=x.dtype,device=x.device))
        count=F.conv2d(mask.to(x.dtype),self.ones.to(x.dtype),padding=1)
        value=self.conv(x)*(9/count.clamp_min(1))+self.bias[None,:,None,None]
        # Keep original support: never create features on unknown/outside-cell positions.
        return torch.where(mask,value,torch.zeros((),dtype=value.dtype,device=value.device))


class PixelNorm(nn.Module):
    def __init__(self,channels):super().__init__();self.norm=nn.LayerNorm(channels)
    def forward(self,x,mask):
        value=self.norm(x.permute(0,2,3,1)).permute(0,3,1,2)
        return torch.where(mask,value,torch.zeros((),dtype=value.dtype,device=value.device))


class SpatialCNN(nn.Module):
    def __init__(self,options):
        super().__init__();a,b=options['conv_channels'];self.pooling=options['pooling']
        self.first=PartialConv(128,a);self.norm=PixelNorm(a);self.second=PartialConv(a,b)
        self.head=nn.Sequential(nn.Linear(b,options['hidden']),nn.GELU(),nn.Dropout(options['dropout']),nn.Linear(options['hidden'],1))
        self.output=options['output']
        if self.pooling not in {'mean','max'} or self.output not in {'linear','sigmoid'}:raise ValueError('Pooling/output non supportato')
    def forward(self,batch):
        mask=batch['mask'].bool();owners=batch['owners'];n=len(batch['target'])
        if not mask.flatten(1).any(dim=1).all():raise ValueError('Patch vuota')
        x=F.gelu(self.norm(self.first(batch['features'],mask),mask));x=F.gelu(self.second(x,mask))
        if self.pooling=='mean':
            patch=(x*mask).sum((2,3))/mask.sum((2,3)).clamp_min(1)
            weight=batch['weights'].to(x.dtype)
            cells=torch.zeros(n,x.shape[1],dtype=x.dtype,device=x.device).index_add(0,owners,patch*weight[:,None])
            denom=torch.zeros(n,dtype=x.dtype,device=x.device).index_add(0,owners,weight)
            if not (denom>0).all():raise ValueError('Cella senza supporto')
            cells=cells/denom[:,None]
        else:
            patch=x.masked_fill(~mask,float('-inf')).amax((2,3))
            cells=torch.stack([patch[owners==i].amax(0) for i in range(n)])
        out=self.head(cells).squeeze(-1)
        return torch.sigmoid(out) if self.output=='sigmoid' else out


class MeanMLP(nn.Module):
    def __init__(self,options):
        super().__init__();self.layers=nn.Sequential(nn.Linear(128,64),nn.GELU(),nn.Dropout(options['dropout']),nn.Linear(64,32),nn.GELU(),nn.Linear(32,1));self.output=options['output']
    def forward(self,batch):
        value=self.layers(batch['mean']).squeeze(-1)
        return torch.sigmoid(value) if self.output=='sigmoid' else value


def make_model(name,options):
    if name=='spatial_cnn':return SpatialCNN(options)
    if name=='mean_mlp':return MeanMLP(options)
    raise ValueError('Modello neurale sconosciuto: '+name)
