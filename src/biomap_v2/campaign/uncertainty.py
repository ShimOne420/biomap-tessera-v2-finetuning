# SCOPO
#   Incertezza del confronto V2-V1 della campagna densa 127.909 e controllo di riproduzione del V1 storico.
#   probing.compare produce solo differenze puntuali; qui si aggiungono intervalli bootstrap APPAIATI
#   (stesse celle, stessi ricampionamenti per V1 e V2), con unità di ricampionamento coerenti con la
#   dipendenza spaziale:
#     - sviluppo (OOF sulle 17 AOI): si ricampionano le AOI intere (17 cluster). Metrica principale:
#       RMSE/MAE macro (media per AOI), la stessa usata per la selezione del modello. Anche la micro.
#     - Liguria (benchmark separato, 4.991 celle): celle e blocchi di 2 km in EPSG:3035.
#   delta = metrica(V2) - metrica(V1): negativo = V2 sbaglia meno. "p_v2_better" = frazione dei
#   ricampionamenti con delta < 0. Nessuna correzione per confronti multipli (5 regressori): dichiararlo.
#
# COME SI RICHIAMA
#   Automaticamente alla fine di `dense_campaign.py compare` (probing.compare), oppure:
#   python -c "from biomap_v2.campaign.uncertainty import paired_uncertainty; paired_uncertainty('<work>')"
#
# OUTPUT (nella work-root)
#   paired_uncertainty_v1_v2_dense_127909.csv; historical_v1_reproduction.json (V1 HistGB ricostruito
#   contro i valori storici di configs/msa_v1_liguria_reference.json: macro RMSE/MAE CV e Liguria).
import json
from pathlib import Path

import numpy as np
import pandas as pd

from biomap_v2.common import ROOT, save_json
from biomap_v2.dense.evaluation import bootstrap_delta, spatial_blocks, full_metrics


def _errors(y, p):
    e = np.asarray(p, dtype=np.float64) - np.asarray(y, dtype=np.float64)
    return e * e, np.abs(e)


# Media per AOI di RMSE e MAE; gli indici possono ripetere AOI (bootstrap a cluster).
def _macro(groups, sq1, ab1, sq2, ab2, chosen):
    r1, m1, r2, m2 = [], [], [], []
    for key in chosen:
        idx = groups[key]
        r1.append(np.sqrt(sq1[idx].mean())); m1.append(ab1[idx].mean())
        r2.append(np.sqrt(sq2[idx].mean())); m2.append(ab2[idx].mean())
    return np.mean(r1), np.mean(m1), np.mean(r2), np.mean(m2)


def _row(scope, model, n, clusters, v1, v2, samples, n_boot):
    row = {'scope': scope, 'model': model, 'n': int(n), 'n_clusters': clusters, 'n_boot': int(n_boot)}
    for i, metric in enumerate(['rmse', 'mae']):
        delta = np.asarray(samples[metric])
        row.update({f'{metric}_v1': v1[i], f'{metric}_v2': v2[i], f'{metric}_delta': v2[i] - v1[i],
                    f'{metric}_ci_low': float(np.quantile(delta, .025)), f'{metric}_ci_high': float(np.quantile(delta, .975)),
                    f'{metric}_p_v2_better': float(np.mean(delta < 0))})
    return row


def paired_uncertainty(work, n_boot=2000, seed=20260925, block_m=2000):
    work = Path(work)
    rows = []
    for filename, kind in [('oof_predictions.csv', 'development'), ('liguria_predictions.csv', 'liguria')]:
        files = [work / 'probing' / e / filename for e in ['v1', 'v2']]
        if not all(p.exists() for p in files):
            continue
        first, second = [pd.read_csv(p, dtype={'cell_id': str}) for p in files]
        joined = first.merge(second, on=['cell_id', 'model'], suffixes=('_v1', '_v2'), validate='one_to_one')
        if len(joined) != len(first) or len(joined) != len(second) or not np.allclose(joined.target_v1, joined.target_v2, rtol=0, atol=0):
            raise ValueError(f'{filename}: predizioni V1/V2 non appaiate')
        for model, g in joined.groupby('model', sort=True):
            y = g.target_v1.to_numpy()
            sq1, ab1 = _errors(y, g.prediction_v1)
            sq2, ab2 = _errors(y, g.prediction_v2)
            if kind == 'development':
                codes, names = pd.factorize(g.aoi_id_v1)
                groups = {k: np.flatnonzero(codes == k) for k in range(len(names))}
                observed = _macro(groups, sq1, ab1, sq2, ab2, list(groups))
                rng = np.random.default_rng(seed)
                samples = {'rmse': [], 'mae': []}
                for _ in range(n_boot):
                    m = _macro(groups, sq1, ab1, sq2, ab2, rng.integers(0, len(groups), len(groups)))
                    samples['rmse'].append(m[2] - m[0]); samples['mae'].append(m[3] - m[1])
                rows.append(_row('development_macro_17aoi', model, len(g), len(groups), observed[0:2], observed[2:4], samples, n_boot))
                result = bootstrap_delta(y, g.prediction_v1, g.prediction_v2, g.aoi_id_v1.to_numpy(), n_boot, seed)
                rows.append(_flat(result, 'development_micro', model, len(g), len(groups), y, g, 'blocks', n_boot))
            else:
                blocks = spatial_blocks(g.lon_v1.to_numpy(), g.lat_v1.to_numpy(), block_m).to_numpy()
                result = bootstrap_delta(y, g.prediction_v1, g.prediction_v2, blocks, n_boot, seed)
                rows.append(_flat(result, 'liguria_cells', model, len(g), None, y, g, 'cells', n_boot))
                rows.append(_flat(result, 'liguria_blocks_2km', model, len(g), int(len(set(blocks))), y, g, 'blocks', n_boot))
    out = pd.DataFrame(rows)
    out.to_csv(work / 'paired_uncertainty_v1_v2_dense_127909.csv', index=False)
    return out


def _flat(result, scope, model, n, clusters, y, g, unit, n_boot):
    row = {'scope': scope, 'model': model, 'n': int(n), 'n_clusters': clusters, 'n_boot': int(n_boot)}
    for metric in ['rmse', 'mae']:
        a = full_metrics(y, g.prediction_v1)[metric]
        b = full_metrics(y, g.prediction_v2)[metric]
        row.update({f'{metric}_v1': a, f'{metric}_v2': b, f'{metric}_delta': result[f'{metric}_delta'],
                    f'{metric}_ci_low': result[f'{metric}_{unit}_ci_low'], f'{metric}_ci_high': result[f'{metric}_{unit}_ci_high'],
                    f'{metric}_p_v2_better': result[f'{metric}_{unit}_p_second_better']})
    return row


# Il V1 ricostruito da tile pubbliche deve riprodurre i numeri storici con gli stessi parametri e fold.
def historical_v1_check(work):
    work = Path(work)
    reference = json.loads((ROOT / 'configs/msa_v1_liguria_reference.json').read_text(encoding='utf-8'))
    result = {'role': 'V1 ricostruito (GeoTESSERA 1.0/vultr) vs run storico 20260928T111520077028Z'}
    macro = work / 'probing' / 'v1' / 'macro_metrics.csv'
    if macro.exists():
        m = pd.read_csv(macro, index_col=0).loc['HistGB']
        result['cv_macro'] = {'rmse': float(m.rmse), 'mae': float(m.mae),
                              'historical_rmse': reference['cv_macro_rmse_histgb'], 'historical_mae': reference['cv_macro_mae_histgb'],
                              'abs_diff_rmse': abs(float(m.rmse) - reference['cv_macro_rmse_histgb']),
                              'abs_diff_mae': abs(float(m.mae) - reference['cv_macro_mae_histgb'])}
    lig = work / 'probing' / 'v1' / 'liguria_predictions.csv'
    if lig.exists():
        p = pd.read_csv(lig)
        p = p[p.model == 'HistGB']
        ours = full_metrics(p.target, p.prediction)
        hist = reference['metrics']['HistGB']
        result['liguria'] = {k: {'reconstructed': ours[k], 'historical': hist[k], 'abs_diff': abs(ours[k] - hist[k])}
                             for k in ['mae', 'rmse', 'mape_pct', 'bias', 'r2'] if hist.get(k) is not None}
    save_json(work / 'historical_v1_reproduction.json', result)
    return result
