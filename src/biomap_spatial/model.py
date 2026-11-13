# Scopo: baseline B generalizzata a C variabile, senza encoder/Rasterio o target richiesti al forward.
# Fasi: mask -> proiezione 1x1 opzionale -> PartialConv/PixelNorm/GELU -> pooling -> cella -> head.
# Input: batch [P,C,H,W], mask [P,1,H,W], owners/pesi/cell_ids; output [B] stime MSA.
# Parametri: kernel 1/3, C, D, 64/32, hidden/dropout/output. Uso: SpatialCNN(ModelConfig()).
# Risorse: PyTorch CPU/CUDA/MPS per la sola testa; nessun Foundation Model o download.
# Ripresa: state_dict include la proiezione; l'engine verifica feature space e protocollo del checkpoint.
import torch
from torch import nn
import torch.nn.functional as F

from .contracts import ModelConfig
from .utils import legacy

PartialConv = legacy('model').PartialConv
PixelNorm = legacy('model').PixelNorm


class PixelwiseConv(PartialConv):
    def __init__(self, cin, cout):
        super().__init__(cin, cout)
        self.conv = nn.Conv2d(cin, cout, 1, bias=False)
        self.ones = torch.ones(1, 1, 1, 1)

    def forward(self, x, mask):
        # k=1: il conteggio valido e uno; stessa semantica di bias e supporto della PartialConv.
        x = torch.where(mask, x, torch.zeros((), dtype=x.dtype, device=x.device))
        return torch.where(mask, self.conv(x) + self.bias[None, :, None, None],
                           torch.zeros((), dtype=x.dtype, device=x.device))


def validate_batch(batch, channels):
    x, mask, owners = batch['features'], batch['mask'], batch['owners']
    n = len(batch['cell_ids'])
    if x.ndim != 4 or x.shape[1] != channels or mask.shape != (len(x), 1, *x.shape[2:]) or mask.dtype != torch.bool:
        raise ValueError('Shape/C/mask del batch incompatibili')
    if owners.dtype != torch.long or owners.shape != (len(x),) or n < 1 or len(set(batch['cell_ids'])) != n:
        raise ValueError('Owners/celle del batch non validi')
    if (owners < 0).any() or (owners >= n).any() or not mask.flatten(1).any(1).all():
        raise ValueError('Owner fuori range o patch vuota')
    if not torch.isfinite(torch.where(mask, x, 0)).all():
        raise ValueError('NaN/Inf sul supporto valido')
    weights = batch['weights']
    if weights.shape != (len(x),) or not torch.equal(weights.to(x.dtype), mask.sum((1, 2, 3)).to(x.dtype)):
        raise ValueError('Pesi patch diversi dai contributi pixel validi')
    return n


def aggregate(x, mask, owners, weights, n):
    patch = (x * mask).sum((2, 3)) / mask.sum((2, 3)).clamp_min(1)
    weights = weights.to(x.dtype)
    cells = torch.zeros(n, x.shape[1], dtype=x.dtype, device=x.device).index_add(0, owners, patch * weights[:, None])
    denom = torch.zeros(n, dtype=x.dtype, device=x.device).index_add(0, owners, weights)
    if not (denom > 0).all():
        raise ValueError('Cella senza supporto')
    return cells / denom[:, None]


class SpatialCNN(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        a, b = config.conv_channels
        self.projection = nn.Identity() if config.projection_channels is None else nn.Conv2d(config.input_channels, config.projection_channels, 1)
        cin = config.projection_channels or config.input_channels
        block = PartialConv if config.kernel == 3 else PixelwiseConv
        self.first, self.norm, self.second = block(cin, a), PixelNorm(a), block(a, b)
        self.head = nn.Sequential(nn.Linear(b, config.hidden), nn.GELU(), nn.Dropout(config.dropout), nn.Linear(config.hidden, 1))

    def forward(self, batch):
        n = validate_batch(batch, self.config.input_channels)
        mask = batch['mask']
        x = torch.where(mask, batch['features'], 0)
        x = torch.where(mask, self.projection(x), 0)
        x = F.gelu(self.norm(self.first(x, mask), mask))
        x = F.gelu(self.second(x, mask))
        cells = aggregate(x, mask, batch['owners'], batch['weights'], n)
        out = self.head(cells).squeeze(-1)
        return torch.sigmoid(out) if self.config.output == 'sigmoid' else out

    def description(self):
        return {'trainable_parameters':sum(p.numel() for p in self.parameters() if p.requires_grad),
                'kernel':self.config.kernel,'receptive_field':1+2*(self.config.kernel-1),
                'input_channels':self.config.input_channels,'projection_channels':self.config.projection_channels,
                'conv_channels':list(self.config.conv_channels)}


class MeanMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        width=config.conv_channels[-1]
        self.layers = nn.Sequential(nn.Linear(config.input_channels, config.hidden), nn.GELU(), nn.Dropout(config.dropout),
            nn.Linear(config.hidden, width), nn.GELU(), nn.Linear(width, 1))

    def forward(self, batch):
        out = self.layers(batch['mean']).squeeze(-1)
        return torch.sigmoid(out) if self.config.output == 'sigmoid' else out

    def description(self):
        return {'trainable_parameters':sum(p.numel() for p in self.parameters()),'representation':'same_support_mean',
                'input_channels':self.config.input_channels,'hidden_dimensions':[self.config.hidden,self.config.conv_channels[-1]],'receptive_field':None}


def make_model(config):
    return SpatialCNN(config) if config.architecture == 'spatial_cnn' else MeanMLP(config)
