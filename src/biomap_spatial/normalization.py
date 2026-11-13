# Scopo: normalizzatori indipendenti dall'adapter e stimati soltanto sulle celle training del fold.
# Fasi: verifica training -> momenti B riusati o statistiche cell-balanced -> stato checkpointabile.
# Input: frame con mean/variance/n_valid e partition; output: mean/std/ID fit e trasformazioni float32.
# Parametri: pixel_weighted_standard, cell_standard (solo vettori), none. Nessun doppio scaler.
# Risorse: CPU sui momenti per cella, niente nuove letture Sentinel. Ripresa: stato congelato nel checkpoint.
import numpy as np
from .utils import legacy, signature


def fit_normalizer(frame, strategy):
    if not len(frame) or set(frame.partition) != {'train'} or frame.cell_id.duplicated().any():
        raise ValueError('Fit normalizzazione richiede soltanto celle training uniche')
    means = np.stack(frame['mean']).astype(np.float64)
    if strategy == 'pixel_weighted_standard':
        result = legacy('dataset').fit_normalizer(frame)
    elif strategy == 'cell_standard':
        result = {'mean':means.mean(0).astype(np.float32), 'std':np.maximum(means.std(0), 1e-6).astype(np.float32)}
    elif strategy == 'none':
        result = {'mean':np.zeros(means.shape[1], dtype=np.float32), 'std':np.ones(means.shape[1], dtype=np.float32)}
    else:
        raise ValueError('Normalizzatore sconosciuto')
    result.update(strategy=strategy, training_ids=frame.cell_id.tolist(),
        training_ids_sha256=signature(sorted(frame.cell_id.tolist())), std_convention='population_ddof0_floor_1e-6')
    return result


def transform(features, mask, state):
    if state['strategy'] == 'cell_standard':
        raise ValueError('cell_standard e definita sui vettori, non sulle mappe')
    x = np.where(mask[None], features, 0).astype(np.float32)
    x = (x - np.asarray(state['mean'])[:, None, None]) / np.asarray(state['std'])[:, None, None]
    x = np.where(mask[None], x, 0)
    if not np.isfinite(x).all():
        raise ValueError('Feature normalizzate non finite')
    return x
