# Scopo: confronto appaiato V1/V2 con fold spaziali sulle stesse celle liguri.
# Fasi: bande spaziali -> OOF per Dummy/Ridge/HistGB -> metriche -> bootstrap a blocchi.
# Input: tabella appaiata (paired.py) e iperparametri storici congelati.
# Output: predictions.csv, metrics*.csv, paired_delta.csv, fold e figure.
# Parametri: n_splits, seed, n_boot; HistGB/Ridge identici per V1 e V2.
# Risorse: CPU; HistGB su ~5.000 celle x 128 feature.
# Ripresa: funzione deterministica; i file di output vengono riscritti per run.
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Transformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .paired import features

METRICS = ('mae', 'rmse', 'mape_pct', 'r2', 'pearson_r', 'bias')


def full_metrics(y, prediction):
    y = np.asarray(y, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    if len(y) == 0 or not np.isfinite(y).all() or not np.isfinite(prediction).all():
        raise ValueError('Metriche: campione vuoto o non finito')
    nonzero = y != 0
    observed_mean = float(np.mean(y))
    observed_std = float(np.std(y, ddof=1)) if len(y) > 1 else 0.0
    mae = float(mean_absolute_error(y, prediction))
    rmse = float(np.sqrt(mean_squared_error(y, prediction)))
    return {
        'n': int(len(y)),
        'observed_mean': observed_mean,
        'observed_std': observed_std,
        'predicted_mean': float(np.mean(prediction)),
        'mae': mae,
        'rmse': rmse,
        'bias': float(np.mean(prediction - y)),
        'rrmse_mean': rmse / observed_mean if observed_mean else None,
        'mae_pct_of_observed_mean': 100 * mae / observed_mean if observed_mean else None,
        'rmse_pct_of_observed_mean': 100 * rmse / observed_mean if observed_mean else None,
        'mape_pct': float(np.mean(np.abs((prediction[nonzero] - y[nonzero]) / y[nonzero])) * 100) if nonzero.any() else None,
        'mape_n': int(nonzero.sum()),
        'r2': float(r2_score(y, prediction)) if len(y) > 1 and np.std(y) > 0 else None,
        'pearson_r': float(np.corrcoef(y, prediction)[0, 1]) if len(y) > 1 and np.std(y) > 0 and np.std(prediction) > 0 else None,
        'negative_prediction_fraction': float(np.mean(prediction < 0)),
        'above_one_fraction': float(np.mean(prediction > 1)),
    }


def spatial_folds(lon, lat, n_splits=5, seed=20260925):
    """Bande contigue lungo la prima componente principale in EPSG:3035."""
    if n_splits < 2 or n_splits > len(lon):
        raise ValueError('n_splits non valido')
    transformer = Transformer.from_crs(4326, 3035, always_xy=True)
    x, y = transformer.transform(np.asarray(lon), np.asarray(lat))
    coordinates = np.column_stack([x, y])
    centered = coordinates - coordinates.mean(axis=0)
    _, _, vectors = np.linalg.svd(centered, full_matrices=False)
    score = centered @ vectors[0]
    order = np.argsort(score, kind='stable')
    folds = np.empty(len(lon), dtype=np.int64)
    for fold, index in enumerate(np.array_split(order, n_splits)):
        folds[index] = fold
    if len(np.unique(folds)) != n_splits:
        raise ValueError('Fold spaziale vuoto')
    return folds


def spatial_blocks(lon, lat, block_m=2000):
    transformer = Transformer.from_crs(4326, 3035, always_xy=True)
    x, y = transformer.transform(np.asarray(lon), np.asarray(lat))
    return pd.Series([f'{int(a // block_m)}:{int(b // block_m)}' for a, b in zip(x, y)])


def make_estimator(name, histgb_params, ridge_alpha=100.0, seed=20260925):
    if name == 'DummyMean':
        return DummyRegressor(strategy='mean')
    if name == 'Ridge':
        return make_pipeline(StandardScaler(), Ridge(alpha=ridge_alpha))
    if name == 'HistGB':
        params = dict(histgb_params or {})
        params.setdefault('random_state', seed)
        return HistGradientBoostingRegressor(**params)
    raise ValueError(f'Modello non supportato: {name}')


def out_of_fold(frame, which, model_name, folds, histgb_params, ridge_alpha, seed):
    target = frame.target.to_numpy(dtype=np.float32)
    matrix = features(frame, which)
    prediction = np.full(len(frame), np.nan, dtype=np.float64)
    for fold in np.unique(folds):
        training = folds != fold
        validation = ~training
        model = make_estimator(model_name, histgb_params, ridge_alpha, seed)
        model.fit(matrix[training], target[training])
        prediction[validation] = model.predict(matrix[validation])
    if not np.isfinite(prediction).all():
        raise ValueError('Predizioni OOF non complete')
    return prediction


def bootstrap_delta(y, first, second, block_ids, n_boot=2000, seed=20260925):
    """Delta = metrica(second) - metrica(first). Intervalli su celle e su blocchi."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y, dtype=np.float64)
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)

    def difference(index):
        errors_first = np.abs(first[index] - y[index])
        errors_second = np.abs(second[index] - y[index])
        squared_first = (first[index] - y[index]) ** 2
        squared_second = (second[index] - y[index]) ** 2
        return {'mae': float(errors_second.mean() - errors_first.mean()),
                'rmse': float(np.sqrt(squared_second.mean()) - np.sqrt(squared_first.mean()))}

    result = {}
    observed = difference(np.arange(len(y)))
    for unit, sampler in (('cells', None), ('blocks', block_ids)):
        samples = []
        if sampler is None:
            for _ in range(n_boot):
                samples.append(difference(rng.integers(0, len(y), len(y))))
        else:
            codes, _ = pd.factorize(pd.Series(sampler))
            members = [np.flatnonzero(codes == code) for code in range(codes.max() + 1)]
            for _ in range(n_boot):
                chosen = rng.integers(0, len(members), len(members))
                index = np.concatenate([members[position] for position in chosen])
                samples.append(difference(index))
        for metric in ('mae', 'rmse'):
            values = np.array([sample[metric] for sample in samples])
            result[f'{metric}_delta'] = observed[metric]
            result[f'{metric}_{unit}_ci_low'] = float(np.quantile(values, 0.025))
            result[f'{metric}_{unit}_ci_high'] = float(np.quantile(values, 0.975))
            result[f'{metric}_{unit}_p_second_better'] = float(np.mean(values < 0))
            result[f'{metric}_{unit}_n_boot'] = int(n_boot)
    return result


def decile_table(y, prediction, deciles=10):
    frame = pd.DataFrame({'target': np.asarray(y), 'prediction': np.asarray(prediction)})
    frame['bin'] = pd.qcut(frame.prediction, deciles, labels=False, duplicates='drop')
    rows = []
    for label, part in frame.groupby('bin'):
        rows.append({'bin': int(label), 'n': int(len(part)),
                     'target_mean': float(part.target.mean()),
                     'prediction_mean': float(part.prediction.mean()),
                     'bias': float(np.mean(part.prediction - part.target))})
    return pd.DataFrame(rows)


def benchmark(frame, histgb_params, out, n_splits=5, seed=20260925, ridge_alpha=100.0,
              n_boot=2000, block_m=2000, models=('DummyMean', 'Ridge', 'HistGB')):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    target = frame.target.to_numpy(dtype=np.float32)
    folds = spatial_folds(frame.lon.to_numpy(), frame.lat.to_numpy(), n_splits, seed)
    frame = frame.copy()
    frame['fold'] = folds
    frame[['fold', 'lon', 'lat', 'target']].to_csv(out / 'folds.csv', index=False)

    predictions = []
    overall_rows = []
    fold_rows = []
    for which in ('v1', 'v2'):
        for model_name in models:
            prediction = out_of_fold(frame, which, model_name, folds, histgb_params, ridge_alpha, seed)
            record = frame[['lon', 'lat', 'target', 'fold']].copy()
            record['encoder'] = which
            record['model'] = model_name
            record['prediction'] = prediction
            predictions.append(record)
            metrics = full_metrics(target, prediction)
            overall_rows.append({'encoder': which, 'model': model_name, **metrics})
            fold_scores = []
            for fold in np.unique(folds):
                part = folds == fold
                fold_metrics = full_metrics(target[part], prediction[part])
                fold_rows.append({'encoder': which, 'model': model_name, 'fold': int(fold), **fold_metrics})
                fold_scores.append(fold_metrics['rmse'])
            overall_rows[-1]['macro_fold_rmse'] = float(np.mean(fold_scores))
    prediction_frame = pd.concat(predictions, ignore_index=True)
    prediction_frame.to_csv(out / 'predictions.csv', index=False)
    metrics_frame = pd.DataFrame(overall_rows)
    dummy = metrics_frame[metrics_frame.model == 'DummyMean'].set_index('encoder')
    metrics_frame['improvement_vs_dummy_mae'] = [
        1 - row.mae / dummy.loc[row.encoder, 'mae'] if dummy.loc[row.encoder, 'mae'] else None
        for row in metrics_frame.itertuples()]
    metrics_frame['improvement_vs_dummy_rmse'] = [
        1 - row.rmse / dummy.loc[row.encoder, 'rmse'] if dummy.loc[row.encoder, 'rmse'] else None
        for row in metrics_frame.itertuples()]
    metrics_frame.to_csv(out / 'metrics.csv', index=False)
    fold_frame = pd.DataFrame(fold_rows)
    fold_frame.to_csv(out / 'metrics_by_fold.csv', index=False)

    deltas = []
    for model_name in models:
        first = prediction_frame[(prediction_frame.encoder == 'v1') & (prediction_frame.model == model_name)]
        second = prediction_frame[(prediction_frame.encoder == 'v2') & (prediction_frame.model == model_name)]
        if not len(first) or len(first) != len(second):
            continue
        left = first.sort_values(['lon', 'lat']).reset_index(drop=True)
        right = second.sort_values(['lon', 'lat']).reset_index(drop=True)
        if not np.allclose(left.target, right.target):
            raise ValueError('Target V1/V2 non appaiati')
        block_ids = spatial_blocks(left.lon.to_numpy(), left.lat.to_numpy(), block_m).to_numpy()
        result = bootstrap_delta(left.target.to_numpy(), left.prediction.to_numpy(),
                                 right.prediction.to_numpy(), block_ids, n_boot, seed)
        deltas.append({'model': model_name, **result})
        decile_table(left.target.to_numpy(), left.prediction.to_numpy()).assign(encoder='v1', model=model_name).to_csv(
            out / f'deciles_v1_{model_name}.csv', index=False)
        decile_table(right.target.to_numpy(), right.prediction.to_numpy()).assign(encoder='v2', model=model_name).to_csv(
            out / f'deciles_v2_{model_name}.csv', index=False)
    delta_frame = pd.DataFrame(deltas)
    delta_frame.to_csv(out / 'paired_delta.csv', index=False)

    _plots(prediction_frame, out)
    return {'metrics': metrics_frame, 'folds': fold_frame, 'predictions': prediction_frame,
            'paired_delta': delta_frame, 'fold_assignment': folds}


def _plots(predictions, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for (encoder, model), part in predictions.groupby(['encoder', 'model']):
        figure, axis = plt.subplots(figsize=(5, 5))
        axis.scatter(part.target, part.prediction, s=6, alpha=0.35)
        axis.plot([0, 1], [0, 1], 'k--', linewidth=1)
        axis.set(xlabel='GLOBIO MSA osservato', ylabel='MSA predetto',
                 title=f'{encoder.upper()} / {model}, n={len(part)}')
        figure.tight_layout()
        figure.savefig(Path(out) / f'scatter_{encoder}_{model}.png', dpi=140)
        plt.close(figure)
