# SCOPO: salvare Excel riproducibili dai CSV, senza modificare workbook o risultati storici.
# Un file contiene il run e un registro comune raccoglie le metriche di tutti i run densi.
# Il registro deriva dai CSV: ricostruirlo non duplica righe o ricalcola regressioni.
import json
import os
from pathlib import Path
import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment
from biomap_v2.common import save_json


# Numericità e NoData preservati: le celle non disponibili restano vuote, non zero.
def add_table(book, title, frame):
    sheet = book.create_sheet(title)
    sheet.append(frame.columns.tolist())
    for row in frame.itertuples(index=False, name=None):
        sheet.append([None if pd.isna(value) else value for value in row])
    sheet.freeze_panes = 'A2'
    sheet.auto_filter.ref = sheet.dimensions
    sheet.row_dimensions[1].height = 32
    for cell in sheet[1]:
        cell.font = Font(bold=True, color='FFFFFF')
        cell.fill = PatternFill('solid', fgColor='185B43')
        cell.alignment = Alignment(wrap_text=True)
        sheet.column_dimensions[cell.column_letter].width = min(35, max(16, len(str(cell.value))+2))
    return sheet


# Excel aperto non viene sostituito con un finto _latest: il comando segnala il blocco.
def save_book(book, path):
    path = Path(path)
    if path.with_name('~$'+path.name).exists():
        raise PermissionError('Chiudere Excel prima di aggiornare: '+str(path))
    temporary = path.with_suffix('.tmp.xlsx')
    book.save(temporary)
    os.replace(temporary, path)


# Esportazione chiamata dopo il training: i CSV del run sono sempre la fonte di verità.
def export_excel(out):
    out = Path(out)
    book = Workbook()
    summary = book.active
    summary.title = 'Summary'
    for text in ['Probing denso TESSERA V1/V2 — GLOBIO MSA 2020',
                 'CV geografica interna alla stessa AOI; non trasferimento dalle 17 AOI.',
                 'Predetto−target = bias; MAE/RMSE in unità MSA 0–1.',
                 'MAPE: errore relativo %, con target zero esclusi. MAE×100: punti della scala.',
                 'Risultati: metriche OOF; Fold: metriche e numerosità; Celle: dettaglio per encoder/modello.',
                 'Fonte: protocol.json, metrics.csv e predictions.csv di questo run. Encoder congelati.']:
        summary.append([text])
    summary.column_dimensions['A'].width = 115
    for name, filename in [('Risultati', 'metrics.csv'), ('Fold', 'metrics_by_fold.csv'), ('Celle', 'predictions.csv')]:
        add_table(book, name, pd.read_csv(out/filename))
    save_book(book, out/'risultati.xlsx')
    rows = []
    for directory in sorted(out.parent.iterdir()):
        if not (directory/'metrics.csv').is_file() or not (directory/'protocol.json').is_file():
            continue
        frame = pd.read_csv(directory/'metrics.csv')
        protocol = json.loads((directory/'protocol.json').read_text(encoding='utf-8'))
        frame.insert(0, 'run_id', directory.name)
        frame.insert(1, 'created_utc', protocol['created'])
        frame['aoi_id'] = protocol['paths'].get('aoi_id', 'not_recorded')
        frame['evaluation'] = protocol.get('evaluation', 'historical_not_recorded')
        rows.append(frame)
    register = Workbook()
    register.remove(register.active)
    if rows:
        all_rows = pd.concat(rows, ignore_index=True)
        all_rows.to_csv(out.parent/'dense_experiment_registry.csv', index=False)
        add_table(register, 'Risultati', all_rows)
        save_book(register, out.parent/'DENSE_MSA_experiment_registry.xlsx')
    return out/'risultati.xlsx'


# Recupera un errore finale di Excel/report dai CSV completi, senza ripetere i regressori.
def export_run(out):
    from .report import write_report
    out = Path(out)
    protocol = json.loads((out/'protocol.json').read_text(encoding='utf-8'))
    history = out/'historical_v1.json'
    write_report(out/Path(protocol.get('report_file', 'REPORT_V1_V2_DENSE_MSA.md')).name,
                 {'metrics': pd.read_csv(out/'metrics.csv'), 'paired_delta': pd.read_csv(out/'paired_delta.csv'),
                  'protocol': protocol, 'equivalence': protocol.get('equivalence', {}),
                  'historical': json.loads(history.read_text()) if history.is_file() else None})
    result = export_excel(out)
    state = json.loads((out/'run_state.json').read_text())
    state.update(status='complete', exports_rebuilt_from_csv=True)
    save_json(out/'run_state.json', state)
    return result
