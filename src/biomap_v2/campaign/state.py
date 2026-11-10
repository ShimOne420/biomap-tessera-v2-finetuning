# Scopo: transazioni atomiche, identità delle cache e vincoli geografici del benchmark.
# Input: manifest congelato e file della work-root. Nessuna scrittura nelle sorgenti.
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
from biomap_v2.common import ROOT, digest, save_json

MEANS = [f'mean_{i:03d}' for i in range(128)]

# Il manifest originale resta l'autorità per numerosità, identificativi e fold.
def membership():
    p = ROOT/'configs/msa_v1_liguria_cells.csv'
    f = pd.read_csv(p, dtype={'cell_id': str})
    import hashlib
    ref = json.loads((ROOT/'configs/msa_v1_liguria_reference.json').read_text())
    if hashlib.sha256(p.read_bytes().replace(b'\r\n', b'\n')).hexdigest() != ref['cells_sha256']:
        raise ValueError('Manifest modificato')
    if f.cell_id.duplicated().any() or f.partition.value_counts().to_dict() != {'train':127909, 'test':4991}:
        raise ValueError('Membership non valida')
    d = f[f.partition == 'train']
    if d.aoi_id.nunique() != 17 or set(f[f.partition == 'test'].aoi_id) != {'liguria_levante'}:
        raise ValueError('AOI non valide')
    for row in ref['fold_counts']:
        k = row['cv_fold']
        if (d.cv_fold == k).sum() != row['n_validation'] or (d.cv_fold != k).sum() != row['n_train']:
            raise ValueError('Fold modificati')
    return f

# Scrive prima il payload, poi il marker verificato; un payload orfano non è completato.
def atomic_npz(path, identity, **arrays):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.partial')
    with tmp.open('wb') as out:
        np.savez_compressed(out, **arrays); out.flush(); os.fsync(out.fileno())
    with np.load(tmp, allow_pickle=False) as z:
        if set(z.files) != set(arrays): raise ValueError('Transazione NPZ incompleta')
    os.replace(tmp, path)
    save_json(path.with_suffix('.json'), {'identity':identity, 'sha256':digest(path)})

# Un checksum discordante o un'identità diversa sono errori, mai cache da sovrascrivere.
def cached(path, identity):
    path = Path(path); meta = path.with_suffix('.json')
    if not meta.exists(): return False
    info = json.loads(meta.read_text())
    if info['identity'] != identity or not path.exists() or digest(path) != info['sha256']:
        raise ValueError(f'Cache incompatibile/corrotta: {path}')
    return True

# Vietato collocare output sotto una sorgente, anche attraverso link simbolici.
def writable(work, sources):
    work = Path(work).resolve()
    for source in sources:
        if source and (work == Path(source).resolve() or Path(source).resolve() in work.parents):
            raise ValueError('Work-root deve essere esterna alle sorgenti')
    work.mkdir(parents=True, exist_ok=True)
    return work
