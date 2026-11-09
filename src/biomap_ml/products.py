# SCOPO: produrre metriche, grafici, registro Excel e mappe E2 dai CSV verificati.
# CSV/JSON restano la fonte di verità; l'Excel separato è ricostruibile e idempotente.
# Ogni foglio di dettaglio conserva celle, coordinate, target, CNN, Dummy ed errori.
# L'MSA resta a 10 secondi d'arco; NoData non è zero e non viene interpolato.
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
from biomap_v2.common import metrics, save_json
from .settings import save_csv, work_budget


# Una riga per modello/ruolo/AOI, con confronto Dummy sulla stessa popolazione.
def metric_rows(table, run_id):
    rows = []
    groups = [('all', 'all', table), *[('aoi', key, f) for key, f in table.groupby('aoi_id', sort=True)]]
    if table.evaluation_fold.nunique() > 1:
        groups.extend(('fold', 'all', f) for _, f in table.groupby('evaluation_fold'))
    for scope, aoi, frame in groups:
        dummy = metrics(frame.target, frame.dummy_prediction)
        for name, values in [('E2_FrozenCNN', frame.prediction), ('DummyMean', frame.dummy_prediction)]:
            result = metrics(frame.target, values)
            rows.append({'run_id': run_id, 'model': name, 'role': frame.evaluation_role.iloc[0], 'scope': scope,
                         'fold': int(frame.evaluation_fold.iloc[0]) if frame.evaluation_fold.nunique() == 1 else -2,
                         'aoi_id': aoi, 'n_aois': frame.aoi_id.nunique(),
                         'observed_mean': float(frame.target.mean()), 'predicted_mean': float(np.mean(values)),
                         'coverage_mean': float(frame.valid_fraction.mean()),
                         'near_zero_target_n': int(((frame.target > 0) & (frame.target < .05)).sum()),
                         'west': float(frame.west.min()), 'south': float(frame.south.min()),
                         'east': float(frame.east.max()), 'north': float(frame.north.max()),
                         **result, 'mae_scale_percentage_points': result['mae']*100,
                         'rmse_gain_vs_dummy_pct': 100*(1-result['rmse']/dummy['rmse']) if dummy['rmse'] > 0 else None})
    return rows


# Quintili separati per ruolo: diagnostica degli estremi, non selezione post-test.
def quintiles(table):
    result = table.copy()
    result['target_quintile'] = pd.qcut(result.target, 5, duplicates='drop')
    rows = []
    for group, frame in result.groupby('target_quintile', observed=True):
        rows.append({'target_interval': str(group), **metrics(frame.target, frame.prediction)})
    return pd.DataFrame(rows)


# Intervalli appaiati per blocchi 40 km; meno di tre blocchi non dà una stima affidabile.
def block_bootstrap(table, draws=500):
    from pyproj import Transformer
    x, y = Transformer.from_crs(4326, 3035, always_xy=True).transform(table.lon.to_numpy(), table.lat.to_numpy())
    keys = [f'{a}:{int(px//40000)}:{int(py//40000)}' for a, px, py in zip(table.aoi_id, x, y)]
    blocks = [np.asarray(indices) for indices in pd.Series(np.arange(len(table))).groupby(keys).apply(list)]
    if len(blocks) < 3:
        return {'n_blocks': len(blocks), 'draws': 0, 'status': 'insufficient_blocks_for_interval'}
    rng = np.random.default_rng(42)
    y, p, d = table.target.to_numpy(), table.prediction.to_numpy(), table.dummy_prediction.to_numpy()
    values = []
    for _ in range(draws):
        idx = np.concatenate([blocks[i] for i in rng.integers(0, len(blocks), len(blocks))])
        rmse = np.sqrt(np.mean((p[idx]-y[idx])**2))
        drmse = np.sqrt(np.mean((d[idx]-y[idx])**2))
        values.append([np.mean(np.abs(p[idx]-y[idx])), rmse, 100*(1-rmse/drmse) if drmse else np.nan])
    arr = np.asarray(values)
    return {'n_blocks': len(blocks), 'draws': draws, 'status': 'exploratory_spatial_block_bootstrap',
            'interval_95': {name: np.nanquantile(arr[:, j], [.025, .975]).tolist()
                            for j, name in enumerate(['mae', 'rmse', 'rmse_gain_vs_dummy_pct'])}}


# Quattro letture complementari: accordo, residui, distribuzione e localizzazione geografica.
def plots(cfg, table, directory, label):
    os.environ.setdefault('MPLCONFIGDIR', str(Path(cfg['work'])/'matplotlib_cache'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    directory.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    axes[0, 0].hexbin(table.target, table.prediction, gridsize=45, mincnt=1, cmap='viridis')
    axes[0, 0].plot([0, 1], [0, 1], 'k--')
    axes[0, 0].set(xlabel='MSA GLOBIO osservato', ylabel='MSA CNN predetto', xlim=(0, 1), ylim=(0, 1))
    axes[0, 1].scatter(table.target, table.error, s=3, alpha=.2)
    axes[0, 1].axhline(0, color='black', linestyle='--')
    axes[0, 1].set(xlabel='MSA osservato', ylabel='Predetto − osservato')
    axes[1, 0].hist([table.target, table.prediction], bins=np.linspace(0, 1, 31), label=['GLOBIO', 'CNN'])
    axes[1, 0].set(xlabel='MSA', ylabel='Celle')
    axes[1, 0].legend()
    limit = max(float(table.error.abs().quantile(.99)), 1e-5)
    scatter = axes[1, 1].scatter(table.lon, table.lat, c=table.error, s=3, cmap='RdBu_r', vmin=-limit, vmax=limit)
    axes[1, 1].set(xlabel='Longitudine', ylabel='Latitudine')
    fig.colorbar(scatter, ax=axes[1, 1], label='Errore con segno')
    fig.suptitle(f'E2 / {label} / {len(table):,} celle')
    fig.tight_layout()
    fig.savefig(directory/f'{label}.png', dpi=160)
    plt.close(fig)


# Ricostruzione da tutte le run: nessuna riga storica viene duplicata al rilancio.
def workbook(cfg):
    from openpyxl import Workbook
    from openpyxl.styles import Font, PatternFill, Alignment
    path = Path(cfg['work'])/'E2_experiment_registry.xlsx'
    if path.with_name('~$'+path.name).exists():
        raise PermissionError('Excel aperto: chiudere E2_experiment_registry.xlsx e rilanciare evaluate')
    book = Workbook()
    summary = book.active
    summary.title = 'Summary'
    for line in ['E2 — TESSERA v2 Medium frozen + CNN',
                 'Risultati: righe per run, modello, ruolo e AOI. Dettagli: una pagina per run/partizione.',
                 'OOF = validation geografica sulle 17 AOI. Liguria = benchmark già osservato.',
                 'MSA: target modellato GLOBIO 2020, scala 0–1, celle native 10 secondi di arco.',
                 'MAE × 100 = punti della scala percentuale. MAPE = errore relativo %, target zero esclusi.',
                 'Fonte di verità: CSV/JSON nelle run di questa work-root. Nessun workbook del probing modificato.']:
        summary.append([line])
    results = book.create_sheet('Risultati')
    header_written = False
    for number, run in enumerate(sorted((Path(cfg['work'])/'runs').glob('*')), start=1):
        metric_path = run/'metrics.csv'
        if not metric_path.is_file():
            continue
        frame = pd.read_csv(metric_path)
        if not header_written:
            results.append(frame.columns.tolist())
            header_written = True
        for row in frame.itertuples(index=False, name=None):
            results.append([None if pd.isna(v) else v for v in row])
        for label, filename in [('OOF', 'oof_predictions.csv'), ('Test', 'test_predictions.csv')]:
            if not (run/filename).is_file():
                continue
            table = pd.read_csv(run/filename)
            table['run_id'] = run.name
            cols = ['run_id', 'cell_id', 'aoi_id', 'lon', 'lat', 'target', 'prediction', 'dummy_prediction', 'error',
                    'absolute_error', 'absolute_percentage_error_pct', 'evaluation_fold', 'quality_flag']
            sheet = book.create_sheet(f'R{number:03d}_{label}')
            sheet.append(cols)
            for row in table[cols].itertuples(index=False, name=None):
                sheet.append([None if pd.isna(v) else v for v in row])
    for sheet in book:
        sheet.freeze_panes = 'A2'
        sheet.auto_filter.ref = sheet.dimensions
        for cell in sheet[1]:
            cell.font = Font(bold=True, color='FFFFFF')
            cell.fill = PatternFill('solid', fgColor='185B43')
            cell.alignment = Alignment(wrap_text=True, vertical='center')
        sheet.row_dimensions[1].height = 32
        for col in sheet.columns:
            letter = col[0].column_letter
            sheet.column_dimensions[letter].width = min(42, max(16, len(str(col[0].value or ''))+2))
    summary.column_dimensions['A'].width = 115
    temporary = path.with_suffix('.tmp.xlsx')
    book.save(temporary)
    os.replace(temporary, path)  # Windows segnala il file aperto, senza copie _latest spacciate per canoniche.
    return path


# Esporta metriche/figure solo da predizioni salvate dopo il congelamento del checkpoint.
def export_results(cfg, out):
    rows = []
    for label, filename in [('validation', 'oof_predictions.csv'), ('liguria_test', 'test_predictions.csv')]:
        path = out/filename
        if not path.is_file():
            continue
        table = pd.read_csv(path, dtype={'cell_id': str})
        rows.extend(metric_rows(table, out.name))
        save_csv(out/f'{label}_quintiles.csv', quintiles(table))
        near_zero = table[(table.target > 0) & (table.target < .05)]
        if len(near_zero):
            save_csv(out/f'{label}_near_zero.csv', pd.DataFrame(metric_rows(near_zero, out.name)))
        plots(cfg, table, out/'plots', label)
        if label == 'liguria_test':
            save_json(out/'test_bootstrap.json', block_bootstrap(table))
    metrics_frame = pd.DataFrame(rows)
    audit = json.loads((out/'audit.json').read_text(encoding='utf-8'))
    fold_training = {record['fold']: record['train'] for record in audit['folds']}
    metrics_frame['n_training'] = [fold_training.get(row.fold, audit['counts']['train'] if row.fold == -1 else None) for row in metrics_frame.itertuples()]
    metrics_frame['tessera_version'] = 'v2_Medium'
    metrics_frame['created_utc'] = json.loads((out/'run.json').read_text(encoding='utf-8'))['created_utc']
    save_csv(out/'metrics.csv', metrics_frame)
    selection = json.loads((out/'selection.json').read_text(encoding='utf-8'))
    lines = ['# Report E2 — Frozen CNN MSA', '',
             'Encoder Medium invariato. Testa CNN 5×5 su punti Sentinel stratificati; target GLOBIO modellato.',
             'Selezione solo sulle AOI di sviluppo. Liguria è un benchmark già osservato; Umbria esclusa.', '',
             f"RMSE macro-AOI OOF: {selection['macro_aoi_rmse']:.6f}; epoche refit dopo warmup: {selection['refit_epochs']}.", '',
             '| Ruolo | Modello | Celle | MAE | RMSE | MAPE % | Guadagno RMSE vs Dummy % |',
             '|---|---|---:|---:|---:|---:|---:|']
    for row in metrics_frame[metrics_frame.scope == 'all'].itertuples():
        lines.append(f'| {row.role} | {row.model} | {row.n} | {row.mae:.6f} | {row.rmse:.6f} | {row.mape_pct:.3f} | {row.rmse_gain_vs_dummy_pct:.2f} |')
    lines += ['', 'Gli intervalli spaziali (se stimabili) sono in test_bootstrap.json. Pochi blocchi non consentono una stima robusta.',
              'MAE × 100 misura punti della scala; MAPE è relativo. Il confronto con GLOBIO non valida biodiversità in campo.']
    (out/'report.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    work_budget(cfg, 20*1024**2)
    workbook(cfg)


# Le mappe sono separate per AOI; il supporto nativo deriva dai bounds, non da interpolazione.
def export_rasters(cfg, out, table):
    import rasterio
    from rasterio.transform import from_origin, rowcol
    destination = out/'geotiff'
    destination.mkdir(parents=True, exist_ok=True)
    for aoi, frame in table.groupby('aoi_id'):
        west, east, south, north = frame.west.min(), frame.east.max(), frame.south.min(), frame.north.max()
        res = 1/360
        width, height = round((east-west)/res), round((north-south)/res)
        transform = from_origin(west, north, res, res)
        rr, cc = rowcol(transform, frame.lon.to_numpy(), frame.lat.to_numpy())
        if len(set(zip(rr, cc))) != len(frame):
            raise ValueError('Più celle assegnate allo stesso pixel del GeoTIFF')
        columns = {'observed': 'target', 'predicted_E2': 'prediction', 'dummy': 'dummy_prediction',
                   'signed_error_E2': 'error', 'absolute_error_E2': 'absolute_error', 'coverage': 'valid_fraction'}
        for label, column in columns.items():
            values = np.full((height, width), -9999, dtype=np.float32)
            values[rr, cc] = frame[column].to_numpy(dtype=np.float32)
            path = destination/f'{aoi}_{label}.tif'
            with rasterio.open(path, 'w', driver='COG', height=height, width=width, count=1,
                               dtype='float32', crs='EPSG:4326', transform=transform, nodata=-9999, compress='DEFLATE') as ds:
                ds.write(values, 1)
                ds.update_tags(indicator='MSA', year='2020', role=str(frame.evaluation_role.iloc[0]),
                               support='native_10arcsec_stratified_25_points', model='E2_FrozenCNN')
    save_json(destination/'metadata.json', {'run_id': out.name, 'role': 'liguria_previously_observed_benchmark',
              'crs': 'EPSG:4326', 'resolution_degrees': 1/360, 'nodata': -9999, 'uncertainty_calibrated': False})
    return destination
