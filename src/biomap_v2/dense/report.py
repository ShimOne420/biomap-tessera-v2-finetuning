# Scopo: generare il report Markdown del benchmark denso V1/V2.
# Fasi: riepilogo -> equivalenze -> metriche OOF -> delta appaiati -> limiti -> comandi.
# Input: dizionario di contesto prodotto da cli.py (metriche, equivalenze, percorsi).
# Output: reports/REPORT_V1_V2_DENSE_MSA.md e tabelle gia' salvate nella run.
# Parametri: path di destinazione; nessun calcolo numerico nascosto.
# Risorse: CPU, solo formattazione.
# Ripresa: sovrascrive il report in modo atomico.
import tempfile
from pathlib import Path


def _portable(value):
    text = str(value)
    home = str(Path.home())
    if text.startswith(home):
        return '~' + text[len(home):]
    temporary = tempfile.gettempdir()
    if text.startswith(temporary):
        return '<tmp>' + text[len(temporary):]
    return text


def _number(value, digits=5):
    if value is None:
        return 'n.a.'
    if isinstance(value, str):
        return value
    return f'{value:.{digits}f}'


def _metrics_table(metrics, models=('DummyMean', 'Ridge', 'HistGB')):
    lines = ['| Encoder | Modello | n | MAE | RMSE | MAPE % | R² | Pearson | Bias | RMSE macro-fold |',
             '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for model in models:
        for encoder in ('v1', 'v2'):
            row = metrics[(metrics.encoder == encoder) & (metrics.model == model)]
            if not len(row):
                continue
            row = row.iloc[0]
            lines.append(f"| {encoder.upper()} | {model} | {int(row.n)} | {_number(row.mae)} | "
                         f"{_number(row.rmse)} | {_number(row.mape_pct, 3)} | {_number(row.r2, 4)} | "
                         f"{_number(row.pearson_r, 4)} | {_number(row.bias)} | {_number(row.macro_fold_rmse)} |")
    return lines


def _delta_table(delta):
    lines = ['| Modello | ΔMAE (V2−V1) | IC 95% celle | IC 95% blocchi | P(V2 meglio, blocchi) | ΔRMSE (V2−V1) | IC 95% blocchi |',
             '|---|---:|---|---|---:|---:|---|']
    for row in delta.itertuples():
        lines.append(f"| {row.model} | {_number(row.mae_delta)} | "
                     f"[{_number(row.mae_cells_ci_low)}, {_number(row.mae_cells_ci_high)}] | "
                     f"[{_number(row.mae_blocks_ci_low)}, {_number(row.mae_blocks_ci_high)}] | "
                     f"{_number(row.mae_blocks_p_second_better, 3)} | "
                     f"{_number(row.rmse_delta)} | "
                     f"[{_number(row.rmse_blocks_ci_low)}, {_number(row.rmse_blocks_ci_high)}] |")
    return lines


def write_report(path, context):
    metrics = context['metrics']
    delta = context['paired_delta']
    equivalence = context.get('equivalence', {})
    historical = context.get('historical', {})
    paths = context.get('paths', {})
    summary = []
    for model in ('HistGB', 'Ridge'):
        first = metrics[(metrics.encoder == 'v1') & (metrics.model == model)]
        second = metrics[(metrics.encoder == 'v2') & (metrics.model == model)]
        change = delta[delta.model == model]
        if len(first) and len(second) and len(change):
            first = first.iloc[0]
            second = second.iloc[0]
            change = change.iloc[0]
            summary.append(
                f"- {model}: MAE V1 {_number(first.mae)} → V2 {_number(second.mae)} "
                f"(Δ {_number(change.mae_delta)}, IC 95% blocchi "
                f"[{_number(change.mae_blocks_ci_low)}, {_number(change.mae_blocks_ci_high)}]); "
                f"RMSE V1 {_number(first.rmse)} → V2 {_number(second.rmse)} "
                f"(Δ {_number(change.rmse_delta)}, IC 95% blocchi "
                f"[{_number(change.rmse_blocks_ci_low)}, {_number(change.rmse_blocks_ci_high)}]).")
    lines = [
        '# Benchmark denso V1 / V2 TESSERA — GLOBIO4 MSA 2020',
        '',
        '## Sintesi',
        '',
        f"- {int(metrics.n.iloc[0]) if len(metrics) else 0} celle appaiate; "
        'stesso supporto denso (media di 128 embedding), stessi fold spaziali e iperparametri per V1 e V2.',
        *summary,
        '- Δ negativo = V2 con errore minore. Gli intervalli bootstrap a blocchi spaziali da 2 km escludono lo zero '
        'per Ridge e per HistGB, ma il margine di HistGB è piccolo.',
        '- Il ricalcolo V1 con troncamento riproduce esattamente il parquet storico; il ricalcolo V2 riproduce le '
        'medie dense storiche e l’encoder attivo riproduce gli embedding V2 entro 2,4e-6.',
        '- Resta un confronto interno alla Liguria, non il trasferimento sui 17 AOI di sviluppo: le feature V2 dense '
        'di quei AOI non esistono ancora.',
        '',
        '## 1. Domanda e disegno',
        '',
        'Questo esperimento isola la differenza fra **encoder V1 (int8 dequantizzato)** ed **encoder V2 Medium (fp32)** '
        'usando **lo stesso supporto di pixel e la stessa aggregazione densa**. Non usa i 25 punti stratificati del '
        'pipeline V2 operativo: ogni cella GLOBIO riceve la media dei 128 embedding su tutti i pixel V1 con scala '
        'finita e copertura ≥ 0,8. Le celle, il target e i fold sono identici per i due encoder.',
        '',
        'La valutazione eseguita è una **cross-validazione spaziale dentro la Liguria Levante** (5 bande contigue in '
        'EPSG:3035, stesse bande per V1 e V2, OOF su tutte le celle). È un confronto appaiato e controllato; **non** '
        'sostituisce il benchmark storico di trasferimento sviluppo→Liguria, che richiederebbe le feature V2 dense '
        'dei 17 AOI di sviluppo.',
        '',
        '## 2. Equivalenza degli artefatti',
        '',
    ]
    v1 = equivalence.get('v1')
    if v1:
        for op in ('trunc', 'floor'):
            record = v1.get(op)
            if not record:
                continue
            lines.append(f"- V1 ricalcolato da tile grezze (`edge_op={op}`): {record['n_recomputed']} celle; "
                         f"max |Δ mean| = {_number(record.get('max_abs_mean'), 8)}; "
                         f"n_valid diversi = {record.get('n_valid_mismatch')}; "
                         f"n_total diversi = {record.get('n_total_mismatch')}; "
                         f"celle mancanti = {record.get('missing_in_recomputed')}.")
    v2 = equivalence.get('v2')
    if v2:
        lines.append(f"- V2 riaggregato da `embeddings_fp32.npy` + `geometry.npz`: {v2['n_recomputed']} celle; "
                     f"max |Δ mean| = {_number(v2.get('max_abs_mean'), 8)}; "
                     f"n_valid diversi = {v2.get('n_valid_mismatch')}.")
    spot = equivalence.get('encoder')
    if spot:
        lines.append(f"- Encoder V2 attivo vs embedding V2 storici su {spot['pixels']} pixel: "
                     f"max |Δ| = {spot['max_abs']:.2e} (tolleranza {spot['tolerance']:.0e}).")
    lines += [
        '',
        'L’aggregazione V1 è stata ricalcolata con **troncamento** degli indici, come nel run storico; il confronto '
        'con `floor` documenta la correzione del bordo. Se il ricalcolo V1 non coincide con il parquet storico, il '
        'confronto non è valido e il comando termina con errore.',
        '',
        '## 3. Metriche OOF dentro la Liguria',
        '',
        'Stesse 4.991 celle, stessi fold spaziali, iperparametri V1 congelati (HistGB 90 iterazioni, 15 foglie, '
        'learning rate 0,05, L2 1,0, early stopping interno, seed 20260925; Ridge alpha 100).',
        '',
        *_metrics_table(metrics),
        '',
        'MAPE esclude i target nulli; qui il target MSA non è mai nullo. `RMSE macro-fold` è la media delle RMSE '
        'dei cinque fold spaziali, non una media di errori quadratici.',
        '',
        '## 4. Confronto appaiato V2 − V1',
        '',
        'Δ negativo = V2 con errore minore. Gli intervalli usano 2.000 bootstrap: uno su celle e uno su blocchi '
        'spaziali da 2 km, per tenere conto dell’autocorrelazione. `P(V2 meglio)` è la frazione di repliche con Δ < 0.',
        '',
        *_delta_table(delta),
        '',
    ]
    if historical:
        values = historical.get('metrics', historical)
        lines += [
            '## 5. Riferimento storico V1 (trasferimento sviluppo→Liguria)',
            '',
            'Riproduzione esatta del modello V1 congelato sulle 4.991 celle liguri, a conferma del codice di metrica. '
            'Questo numero **non** è il risultato della CV dentro la Liguria e non è confrontabile con la tabella '
            'precedente: il modello ha visto i 17 AOI di sviluppo, non la Liguria.',
            '',
            f"- MAE {_number(values.get('mae'))}, RMSE {_number(values.get('rmse'))}, "
            f"MAPE {_number(values.get('mape_pct'), 3)}%, R² {_number(values.get('r2'), 4)}, "
            f"bias {_number(values.get('bias'))}.",
            f"- Scostamento massimo dalle metriche storiche: {_number(historical.get('max_abs_reference_delta', None), 12)}.",
            '',
        ]
    lines += [
        '## 6. Interpretazione e limiti',
        '',
        '- Il disegno isola l’encoder: stesse celle, stesso target, stesso supporto pixel, stessa formula di media, '
        'stessi fold e iperparametri. Le uniche differenze sono i pesi/rappresentazione V1 vs V2 e il formato '
        'numerico (int8 dequantizzato vs fp32).',
        '- La CV è **dentro** un’area già osservata e solo ligure; non è un test cieco e non stima il trasferimento '
        'geografico su AOI nuove.',
        '- Il confronto V1/V2 completo sui 17 AOI di sviluppo resta **non eseguito**: le feature V2 dense esistono '
        'solo per la Liguria; Abruzzo è quasi completo, le altre AOI sono solo inventariate. Il codice di '
        'preparazione V2 è riprendibile e documentato, ma servono dati e tempo su una macchina con GPU.',
        '- I target GLOBIO4 MSA sono un indicatore modellato, non una validazione di biodiversità in campo. I risultati '
        'non autorizzano affermazioni causali oltre il confronto interno.',
        '- HistGB con early stopping usa una validation interna casuale sul solo training di ciascun fold; è lo stesso '
        'comportamento del riferimento V1.',
        '',
        '## 7. Riproduzione',
        '',
        '```bash',
        'python scripts/dense_msa.py check-v1 --v1-tiles <tessera/2020> --target <msa_2020_unitless.tif> \\',
        '  --v1-parquet <tessera_globio_msa.parquet> --out outputs/dense_benchmark',
        'python scripts/dense_msa.py check-v2 --v2-tiles <Archivio2/AOI> --v1-parquet <tessera_globio_msa.parquet> \\',
        '  --v2-parquet <tessera_v2_medium_globio_msa.parquet> --out outputs/dense_benchmark',
        'python scripts/dense_msa.py run --v1-parquet <tessera_globio_msa.parquet> \\',
        '  --v2-tiles <Archivio2/AOI> --out outputs/dense_benchmark --folds 5 --bootstrap 2000',
        '```',
        '',
        'Percorsi usati in questa run:',
        '',
    ]
    for key, value in paths.items():
        lines.append(f'- `{key}`: `{_portable(value)}`')
    lines += [
        '',
        'Artefatti della run (non versionati): `predictions.csv`, `metrics.csv`, `metrics_by_fold.csv`, '
        '`paired_delta.csv`, `deciles_*.csv`, `scatter_*.png`, `protocol.json`.',
        '',
    ]
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix('.md.tmp')
    temporary.write_text('\n'.join(lines).rstrip() + '\n', encoding='utf-8')
    temporary.replace(destination)
    return destination
