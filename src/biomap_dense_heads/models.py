# Scopo: teste standalone sui 128 canali medi. Conv1D NON è una CNN spaziale.
# Input [B,128] -> [B,1,128]; nessuna trasformazione artificiale in immagine 2D.
from torch import nn

# Identica scelta di output per entrambe le teste; selezionata solo su validation.
def output_layer(kind):
    if kind=='identity':return nn.Identity()
    if kind=='sigmoid':return nn.Sigmoid()
    raise ValueError('Output non supportato')

class ConvHead(nn.Module):
    # Pooling a 8 posizioni conserva informazione sull'ordine artificiale dei canali.
    def __init__(self,cfg,output):
        super().__init__();a,b=cfg['channels'];k1,k2=cfg['kernels']
        if min(a,b,k1,k2)<1 or k1%2==0 or k2%2==0:raise ValueError('Canali positivi, kernel dispari richiesti')
        self.layers=nn.Sequential(nn.Conv1d(1,a,k1,padding=k1//2),nn.GroupNorm(1,a),nn.GELU(),
            nn.Conv1d(a,b,k2,padding=k2//2),nn.GELU(),nn.AdaptiveAvgPool1d(cfg['pool_size']),nn.Flatten(),
            nn.Dropout(cfg['dropout']),nn.Linear(b*cfg['pool_size'],cfg['dense_width']),nn.GELU(),
            nn.Linear(cfg['dense_width'],1),output_layer(output))
    def forward(self,x):
        if x.ndim!=2 or x.shape[1]!=128:raise ValueError('Atteso [batch,128]')
        return self.layers(x[:,None,:]).squeeze(-1)

# Parametri della MLP vicini a quelli della CNN, invece di una rete molto più grande.
def make(name,output,cfg):
    if name=='Conv1D':return ConvHead(cfg,output)
    if name=='MLP':
        budget=sum(p.numel() for p in ConvHead(cfg,output).parameters())
        h=cfg['mlp_width'] or max(1,round((budget-1)/130))
        return nn.Sequential(nn.Linear(128,h),nn.GELU(),nn.Dropout(cfg['dropout']),nn.Linear(h,1),output_layer(output),nn.Flatten(0))
    raise ValueError('Testa sconosciuta')
