# SCOPO: redigere un report del singolo run usando esclusivamente numeri e controlli salvati.
# Non contiene conclusioni fisse, risultati del Mac o affermazioni di equivalenza non verificate.
# INPUT: metriche/paired delta/protocollo. OUTPUT: Markdown dentro la cartella del run.
from pathlib import Path
import numpy as np


# Dati mancanti sono dichiarati: non diventano zeri, successi o percentuali inesistenti.
def number(value, digits=5):
    return 'n.d.' if value is None or not np.isfinite(float(value)) else f'{value:.{digits}f}'


# Un intervallo osservato può indicare vantaggio, svantaggio o differenza non risolta.
def interval_interpretation(low, high):
    if low is None or high is None:
        return 'intervallo non disponibile'
    if high < 0:
        return 'intervallo interamente sotto zero: minore errore V2 in questo benchmark'
    if low > 0:
        return 'intervallo interamente sopra zero: maggiore errore V2 in questo benchmark'
    return 'intervallo comprende zero: differenza non risolta dal bootstrap'


# Ogni riga appartiene a un encoder/regressore e alla popolazione OOF del nuovo run.
def metrics_table(metrics):
    lines = ['| Encoder | Regressore | Celle OOF | MAE | RMSE | MAPE % | Bias | R² | Pearson | RMSE macro-fold | Riduzione RMSE vs Dummy % |',
             '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for row in metrics.itertuples():
        lines.append(f'| {row.encoder} | {row.model} | {row.n} | {number(row.mae)} | {number(row.rmse)} | '
                     f'{number(row.mape_pct, 3)} | {number(row.bias)} | {number(row.r2, 4)} | '
                     f'{number(row.pearson_r, 4)} | {number(row.macro_fold_rmse)} | {number(row.rmse_gain_vs_dummy_pct, 2)} |')
    return lines


# La verifica geometrica importata dal bundle è distinta dalla ricodifica del campione locale.
def equivalence_lines(equivalence):
    lines = []
    bundle = equivalence.get('bundle')
    if bundle:
        lines.append(f"- Pacchetto: stato `{bundle.get('status')}`, {bundle.get('n_cells')} celle e {bundle.get('n_pixels')} pixel.")
        lines.append('- Supporto fisico: audit sui raw effettuato alla creazione; hash dei file portabili verificati su questo PC.')
    if equivalence.get('support'):
        lines.append('- Supporto fisico verificato direttamente su coordinate pixel, scale e griglia originali.')
    for version in ['v1', 'v2']:
        record = equivalence.get(version)
        if record:
            record = record.get('trunc', record)
            lines.append(f'- Riaggregazione {version.upper()}: max differenza delle medie {number(record.get("max_abs_mean"), 9)}; '
                         f'conteggi discordanti {record.get("n_valid_mismatch", "non confrontati")}.')
    encoder = equivalence.get('encoder')
    if encoder and encoder.get('status') == 'passed':
        lines.append(f'- Encoder locale: {encoder["pixels"]} pixel, max |Δ|={encoder["max_abs"]:.3g}, tolleranza={encoder["tolerance"]:.3g}.')
        lines.append('- Questa verifica puntuale non è una ricodifica di tutti i pixel del dataset.')
    else:
        lines.append('- Ricodifica di controllo encoder: non eseguita in questa invocazione.')
    if not equivalence:
        lines.append('- Run su feature fornite: nessuna equivalenza raw/encoder verificata in questa invocazione.')
    return lines


# Il report è scritto atomicamente nella run, preservando quello storico versionato.
def write_report(path, context):
    metrics, delta = context['metrics'], context['paired_delta']
    protocol = context.get('protocol', {})
    n = int(metrics.n.iloc[0])
    aoi = protocol.get('paths', {}).get('aoi_id', 'non registrata')
    lines = ['# Probing denso TESSERA V1/V2 — GLOBIO MSA 2020', '',
             '## Procedimento e popolazione', '',
             f'Run: `{protocol.get("created", "non registrato")}`. AOI: `{aoi}`. Celle appaiate OOF: **{n:,}**.',
             '1. Riutilizzo di embedding congelati dei due encoder e target GLOBIO unitario sulla griglia nativa.',
             '2. Media aritmetica delle 128 componenti su tutti i pixel V1 validi per cella, senza campionare 25 punti.',
             '3. Stesse celle, target e partizioni per ogni coppia V1/V2. Scaler/Dummy stimati solo nel training del fold.',
             f'4. CV geografica interna alla AOI, con {protocol.get("folds", "n.d.")} bande contigue; ogni cella è esclusa dal proprio training.',
             '5. Iperparametri fissati prima del run. Nessuna ricerca o scelta sull’OOF riportato.',
             '6. Salvataggio predizioni, modelli dei fold, metriche, grafici e Excel del run.', '',
             '**Questo non è il trasferimento dalle 17 AOI alla Liguria**, né un nuovo test cieco. '
             'Le metriche non sostituiscono quelle storiche ottenute con 127.909 celle di sviluppo.', '',
             '## Controlli effettivamente eseguiti', '',
             *equivalence_lines(context.get('equivalence', {})), '',
             'Il confronto storico conserva il troncamento degli indici ai bordi (`trunc`). '
             'Il codice V1 attuale contiene la correzione `floor`: non viene applicata solo a uno dei due encoder.', '',
             '## Metriche OOF', '', *metrics_table(metrics), '',
             'MAE e RMSE sono in unità MSA [0,1]. MAE × 100 indica punti della scala percentuale; '
             'MAPE è l’errore relativo medio e scarta soltanto target pari a zero. '
             'La riduzione RMSE positiva indica vantaggio rispetto al Dummy dello stesso fold.', '']
    for row in metrics.itertuples():
        lines.append(f'- {row.encoder}/{row.model}: MAPE su {row.mape_n}/{row.n} celle; target nulli esclusi: {row.mape_excluded_zero}.')
    lines += ['', '## Differenze appaiate e interpretazione', '',
              f'Δ = V2 − V1. Bootstrap: {protocol.get("bootstrap", "n.d.")} repliche; blocchi di {protocol.get("block_m", "n.d.")} m. '
              'Gli intervalli sui blocchi sono preferibili a quelli sulle singole celle per l’autocorrelazione spaziale.', '']
    for row in delta.itertuples():
        lines.append(f'- **{row.model}**: ΔMAE={number(row.mae_delta)}, ΔRMSE={number(row.rmse_delta)}; '
                     f'IC95% ΔRMSE a blocchi [{number(row.rmse_blocks_ci_low)}, {number(row.rmse_blocks_ci_high)}]. '
                     f'{interval_interpretation(row.rmse_blocks_ci_low, row.rmse_blocks_ci_high)}.')
    historical = context.get('historical')
    if historical:
        values = historical['metrics']
        lines += ['', '## Controllo opzionale del modello storico V1', '',
                  'Questo modello è già addestrato sullo sviluppo storico: i suoi numeri non sono la CV interna riportata sopra.',
                  f'MAE {number(values["mae"])}, RMSE {number(values["rmse"])}, MAPE {number(values["mape_pct"], 3)}%. '
                  f'Hash del modello: `{historical["model_sha256"]}`. Dettagli in historical_v1.json.']
    lines += ['', '## Limiti e prossima decisione', '',
              '- Differenze osservate su una AOI non dimostrano generalizzazione su territori nuovi.',
              '- I cinque fold interni non sono i tre fold geografici della campagna storica sulle 17 AOI.',
              '- La rappresentazione V1 int8 dequantizzata e quella V2 FP32 hanno precisione numerica diversa.',
              '- HistGB/MLP usano early stopping interno al solo training del fold, se configurato; non è un’ulteriore AOI indipendente.',
              '- La CNN E2 richiede feature spaziali: non viene applicata a un vettore medio spacciato per immagine.',
              '- Nessun fine-tuning dell’encoder è stato eseguito. L’accordo con GLOBIO misura riproduzione di un target modellato, non biodiversità di campo.',
              '- La campagna densa sulle 17 AOI richiede ancora embedding densi completi e verificati su tutte quelle aree.', '',
              '## Artefatti', '',
              '`protocol.json`, `run_state.json`, `paired_features.parquet`, `folds.csv`, `predictions.csv`, '
              '`metrics.csv`, `metrics_by_fold.csv`, `paired_delta.csv`, `models/`, `scatter_*.png`, `risultati.xlsx`.', '']
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text('\n'.join(lines)+'\n', encoding='utf-8')
    temporary.replace(path)
    return path
