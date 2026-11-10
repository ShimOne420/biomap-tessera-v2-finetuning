# Scopo: appaiare le feature V1 e V2 sulle stesse celle e sullo stesso supporto pixel.
# Fasi: verifica conteggi V1 -> merge per cell_id nativo -> matrice feature per encoder.
# Input: DataFrame V1 (mean_*/std_*), DataFrame V2 (mean_*), target MSA.
# Output: tabella appaiata con metadati e 128 medie per ciascun encoder.
# Parametri: which='v1'|'v2'; std=False usa le sole 128 medie storiche.
# Risorse: CPU; nessuna copia dei tile originali.
# Ripresa: funzione pura, nessuno stato su disco.
import numpy as np
import pandas as pd

from .v1_support import DIMS


def pair_features(v1, v2):
    if v1.index.duplicated().any() or v2.index.duplicated().any():
        raise ValueError('Identificativi cella duplicati')
    if 'n_valid' not in v1:
        raise ValueError('V1 priva di n_valid')
    v2_count = 'n_valid_v2' if 'n_valid_v2' in v2 else ('n_valid' if 'n_valid' in v2 else None)
    if v2_count is None:
        raise ValueError('V2 priva di conteggio pixel')
    common = v1.index.intersection(v2.index)
    if not len(common):
        raise ValueError('Nessuna cella comune fra V1 e V2')
    if len(common) != len(v1) or len(common) != len(v2):
        raise ValueError('Copertura V1/V2 diversa: nessuna riduzione silenziosa delle celle')
    left = v1.loc[common]
    right = v2.loc[common]
    for column in ['lon', 'lat', 'msa_overall']:
        if column in right and not np.allclose(left[column], right[column], atol=1e-8, rtol=0):
            raise ValueError('Geometria o target V1/V2 diversi: '+column)
    mismatch = left.n_valid.to_numpy(dtype=np.int64) != right[v2_count].to_numpy(dtype=np.int64)
    if mismatch.any():
        raise ValueError(f'{int(mismatch.sum())} celle con supporto pixel V1/V2 diverso')
    records = {
        'cell_id': common.to_numpy(dtype=np.int64),
        'lon': left.lon.to_numpy(),
        'lat': left.lat.to_numpy(),
        'target': left.msa_overall.to_numpy(),
        'valid_fraction': left.valid_fraction.to_numpy(),
        'n_valid': left.n_valid.to_numpy(dtype=np.int64),
        'n_total': left.n_total.to_numpy(dtype=np.int64),
    }
    for band in range(DIMS):
        column = f'mean_{band:03d}'
        records[f'v1_{column}'] = left[column].to_numpy()
        records[f'v2_{column}'] = right[column].to_numpy()
    return pd.DataFrame(records).set_index('cell_id')


def features(frame, which, std=False):
    if which not in {'v1', 'v2'}:
        raise ValueError("which: 'v1' oppure 'v2'")
    names = [f'{which}_mean_{band:03d}' for band in range(DIMS)]
    if std:
        names += [f'{which}_std_{band:03d}' for band in range(DIMS)]
        missing = [name for name in names if name not in frame]
        if missing:
            raise ValueError(f'Feature std assenti: {missing[:3]}')
    return frame[names].to_numpy(dtype=np.float32)
