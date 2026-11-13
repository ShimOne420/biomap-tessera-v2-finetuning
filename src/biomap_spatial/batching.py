# Scopo: batch di celle, ciascuna con patch multiple, e inferenza senza target.
# Fasi: reader -> normalizzatore congelato -> padding zero/mask -> owners e conteggi validi.
# Input: EmbeddingSource e frame; output: features PCHW, mask P1HW, una riga target opzionale per cella.
# Parametri: batch_size/canali provengono dai campioni; nessun C=128 globale. Uso: collate_cells(items).
# Risorse: RAM limitata al batch e memmap del reader; ripresa gestita dal DataLoader dell'engine.
import numpy as np
import torch
from torch.utils.data import Dataset
from .normalization import transform


class CellDataset(Dataset):
    def __init__(self, source, frame, normalizer, spatial=True):
        self.source, self.frame, self.normalizer, self.spatial = source, frame.reset_index(drop=True), normalizer, spatial

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        item = {'cell_id':row.cell_id,'mean':(np.asarray(row['mean'],dtype=np.float32)-self.normalizer['mean'])/self.normalizer['std']}
        if 'target' in row:
            item['target'] = float(row.target)
        if self.spatial:
            item['patches'] = [(transform(p.features,p.mask,self.normalizer),p.mask) for p in self.source.patches(row.cell_id)]
        return item


def collate_cells(items):
    ids = [i['cell_id'] for i in items]
    if not items or len(set(ids)) != len(ids):
        raise ValueError('Batch vuoto o celle ripetute')
    batch = {'cell_ids':ids,'mean':torch.from_numpy(np.stack([i['mean'] for i in items]).astype(np.float32))}
    labels = ['target' in i for i in items]
    if any(labels) and not all(labels):
        raise ValueError('Target parziali nel batch')
    if all(labels):
        batch['target'] = torch.tensor([i['target'] for i in items],dtype=torch.float32)
    if 'patches' not in items[0]:
        return batch
    flat = [(owner,p) for owner,item in enumerate(items) for p in item['patches']]
    if not flat:
        raise ValueError('Nessuna patch')
    channels = {p[0].shape[0] for _,p in flat}
    if len(channels) != 1:
        raise ValueError('Canali diversi nello stesso batch')
    c = channels.pop()
    h,w = max(p[0].shape[1] for _,p in flat),max(p[0].shape[2] for _,p in flat)
    features = np.zeros((len(flat),c,h,w),dtype=np.float32)
    mask = np.zeros((len(flat),1,h,w),dtype=bool)
    owners,weights = [],[]
    for j,(owner,(x,m)) in enumerate(flat):
        ph,pw = m.shape
        if x.shape != (c,ph,pw) or not m.any():
            raise ValueError('Patch/mask incompatibile o vuota')
        features[j,:,:ph,:pw] = np.where(m[None],x,0)
        mask[j,0,:ph,:pw] = m
        owners.append(owner)
        weights.append(int(m.sum()))
    batch.update(features=torch.from_numpy(features),mask=torch.from_numpy(mask),
        owners=torch.tensor(owners,dtype=torch.long),weights=torch.tensor(weights,dtype=torch.float32))
    return batch
