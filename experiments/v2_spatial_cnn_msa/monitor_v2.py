# SCOPO
#   Monitor live, in sola lettura, dell'intero run V2 spatial: estrazione r2-fast, verifica, indice, training
#   in parallelo, refit e valutazione, con una stima di quanto manca al completamento.
# COME VIENE RICHIAMATO
#   C:\BioMAP\venv-v2spatial\Scripts\python.exe experiments\v2_spatial_cnn_msa\monitor_v2.py
#       --work-root C:\BioMAP\v2_spatial_msa --v2-root C:\BioMAP\v2_spatial_work\dense_r2 --follow [--interval 15]
# LOGICA IN ORDINE
#   1. legge pipeline.json (fase corrente, rilanci), stream_progress.json di r2, status_*.json dei job,
#      epochs.csv dei fold e profile_training.json; ogni file è letto in un colpo solo e chiuso subito;
#   2. calcola per fase il residuo: estrazione dalla velocità misurata, indice dal suo ritmo, training dalla
#      durata media delle epoche con early stopping (patience) e, tra parentesi, dal budget massimo;
#   3. somma le fasi rimanenti (i fold in parallelo contano come il più lento) e stampa l'ora di fine stimata.
# INPUT E OUTPUT
#   Input: i file sopra. Output: solo testo sul terminale; nessuna scrittura, nessun uso di GPU.
# PARAMETRI
#   --follow (aggiorna fino alla fine del run), --interval secondi (minimo 5), --epochs-guess (40):
#   epoche per fold ipotizzate finché il training non è iniziato.
# RISORSE
#   Trascurabili. Ctrl+C chiude solo il monitor, mai il run.
# RIPRESA
#   Si può chiudere e riaprire in qualsiasi momento.
import argparse
import csv
import json
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

ORDER = ['extract', 'verify_r1', 'build', 'preflight', 'profile', 'cv', 'select', 'refit', 'finalize_refit', 'evaluate']
LABEL = {'extract': 'Estrazione embedding V2 per pixel (r2-fast)', 'verify_r1': 'Verifica medie r2 = r1',
         'build': 'Indice spaziale (patch/maschere B, valori V2)', 'preflight': 'Preflight', 'profile': 'Profilo GPU',
         'cv': 'Training CV (3 fold CNN + MLP/HistGB, in parallelo)', 'select': 'Selezione (solo OOF)',
         'refit': 'Refit sviluppo (in parallelo)', 'finalize_refit': 'Chiusura refit', 'evaluate': 'Valutazione OOF e Liguria'}
FIXED_H = {'verify_r1': .05, 'build': .75, 'preflight': .05, 'profile': .05, 'select': .05, 'finalize_refit': .05, 'evaluate': .15}


def read_json(path):
    try:
        with open(path, 'rb') as stream:
            return json.loads(stream.read().decode('utf-8'))
    except (OSError, ValueError):
        return None


def read_epochs(path):
    try:
        with open(path, 'r', encoding='utf-8', newline='') as stream:
            return list(csv.DictReader(stream))
    except OSError:
        return []


def hms(hours):
    if hours is None:
        return 'in stima'
    seconds = max(0, int(hours * 3600))
    return f'{seconds // 3600}h{seconds // 60 % 60:02d}m'


def bar(fraction, width=28):
    fraction = min(max(fraction or 0, 0), 1)
    full = int(round(fraction * width))
    return '[' + '#' * full + '-' * (width - full) + f'] {fraction * 100:5.1f}%'


def job_folder(run, name, fold):
    return run / name / ('refit' if fold in ('refit', 'development') else f'fold_{fold}')


def training_estimate(run, jobs, patience, epochs_max, epochs_guess, epoch_hint_s):
    # Residuo per job: epoche fino a best_epoch + patience (early stopping), al massimo epochs_max.
    lines, typical, maximum = [], [], []
    for name, fold in jobs:
        folder = job_folder(run, name, fold)
        done = (folder / 'complete.json').exists()
        rows = read_epochs(folder / 'epochs.csv')
        if done:
            lines.append(f'    {name:<12} fold {fold}: completato')
            typical.append(0.); maximum.append(0.)
            continue
        if rows:
            seconds = [float(r['epoch_seconds']) for r in rows]
            mean = sum(seconds[-5:]) / len(seconds[-5:])
            epoch = int(rows[-1]['epoch']); best = int(rows[-1]['best_epoch'])
            limit = epochs_max if fold == 'refit' else min(epochs_max, best + patience)
            left_typ = max(limit - epoch, 0); left_max = max(epochs_max - epoch, 0)
            val = rows[-1].get('validation_macro_aoi_rmse') or '-'
            lines.append(f'    {name:<12} fold {fold}: epoca {epoch}/{epochs_max} | {mean:5.0f} s/epoca | migliore {best} | '
                         f'val macro-RMSE {val[:7]} | residuo ~{hms(left_typ * mean / 3600)} (max {hms(left_max * mean / 3600)})')
            typical.append(left_typ * mean / 3600); maximum.append(left_max * mean / 3600)
        else:
            guess = (epoch_hint_s or 0) * epochs_guess / 3600 if epoch_hint_s else None
            lines.append(f'    {name:<12} fold {fold}: in attesa / avvio')
            typical.append(guess); maximum.append((epoch_hint_s or 0) * epochs_max / 3600 if epoch_hint_s else None)
    known = [t for t in typical if t is not None]
    return lines, (max(known) if known else None), (max([m for m in maximum if m is not None], default=None))


def snapshot(args):
    work, v2 = Path(args.work_root), Path(args.v2_root)
    pipe = read_json(work / 'pipeline.json') or {}
    stages = pipe.get('stages', {})
    run = work / 'runs' / 'full' / (pipe.get('run_name') or 'full_seed42')
    profile = read_json(work / 'profile_training.json') or {}
    epoch_hint = profile.get('estimated_train_epoch_seconds')
    cfg = (read_json(run / 'run.json') or {}).get('config', {}).get('training', {})
    patience, epochs_max = cfg.get('patience', 15), cfg.get('epochs', 150)
    out = [f"=== BIOMAP V2 spatial CNN — {datetime.now().strftime('%d/%m %H:%M:%S')} ===",
           f"Pipeline: {pipe.get('status', 'non avviata')} | fase: {LABEL.get(pipe.get('stage'), pipe.get('stage') or '-')}"]
    if pipe.get('error'):
        out.append('  ERRORE: ' + pipe['error'])
    remaining = 0.
    unknown = False
    for stage in ORDER:
        info = stages.get(stage, {})
        status = info.get('status', 'pending')
        mark = {'complete': 'OK', 'running': '>>', 'failed': '!!'}.get(status, '  ')
        left = 0. if status == 'complete' else None
        text = f'{mark} {LABEL[stage]}'
        if stage == 'extract':
            prog = read_json(v2 / 'stream_progress.json') or {}
            if status != 'complete':
                if prog:
                    frac = prog.get('pixels_done', 0) / max(prog.get('pixels_total', 1), 1)
                    left = prog.get('eta_hours_linear')
                    text += (f"\n     {bar(frac)} tile {prog.get('tiles_done')}/{prog.get('tiles_total')} | "
                             f"{prog.get('pixels_done', 0) / 1e6:.1f}/{prog.get('pixels_total', 0) / 1e6:.1f} M pixel | "
                             f"{prog.get('pixels_per_second_this_session', 0):.0f} px/s | residuo {hms(left)}"
                             f"\n     tile corrente {prog.get('current')} | memoria {info.get('private_gb', '-')} GB "
                             f"(picco {info.get('peak_private_gb', '-')}) | rilanci {info.get('restarts', 0)}")
                    age = None
                    if prog.get('updated_utc'):
                        age = (datetime.now(timezone.utc) - datetime.fromisoformat(prog['updated_utc'])).total_seconds()
                    if age and age > 900 and status == 'running':
                        text += f'\n     nota: avanzamento fermo da {int(age // 60)} min (acquisizione di una tile grande o rilancio)'
                else:
                    left = 16.
                    text += '\n     in avvio (stima iniziale ~16 h)'
        elif stage == 'build':
            st = read_json(work / 'status_build.json') or {}
            if status == 'running' and st.get('total'):
                left = (st.get('eta_seconds') or 0) / 3600 if st.get('eta_seconds') else FIXED_H['build']
                text += f"\n     {bar(st.get('done', 0) / st['total'])} {st.get('done')}/{st['total']} {st.get('unit', '')} | residuo {hms(left)}"
        elif stage in ('cv', 'refit') and status != 'complete':
            jobs = ([('spatial_cnn', f) for f in '012'] + [('mean_mlp', f) for f in '012']) if stage == 'cv' else \
                   [('spatial_cnn', 'refit'), ('mean_mlp', 'refit')]
            hint = epoch_hint * (2 / 3 if stage == 'cv' else 1) if epoch_hint else None
            lines, typ, mx = training_estimate(run, jobs, patience, epochs_max, args.epochs_guess, hint)
            if status == 'running' or stage == 'cv':
                left = typ
            if stage == 'refit' and status != 'running':
                sel = read_json(run / 'selection.json') or {}
                epochs = (sel.get('refit_epochs') or {}).get('spatial_cnn', args.epochs_guess)
                left = epoch_hint * epochs / 3600 if epoch_hint else None
            if status == 'running':
                text += '\n' + '\n'.join(lines) + (f'\n     residuo fase ~{hms(typ)} (massimo {hms(mx)})' if typ is not None else '')
        if left is None and status != 'complete':
            left = FIXED_H.get(stage)
        if status == 'failed':
            text += '\n     ' + str(info.get('error', ''))
        out.append(text)
        if status != 'complete':
            if left is None:
                unknown = True
            else:
                remaining += left
    if pipe.get('status') == 'complete':
        out.append('RUN COMPLETATO. Risultati in ' + str(run))
    else:
        end = datetime.now() + timedelta(hours=remaining)
        out.append(f"TEMPO RESIDUO STIMATO: {hms(remaining)}{' + fasi non ancora stimabili' if unknown else ''} "
                   f"-> fine prevista ~{end.strftime('%a %d/%m %H:%M')}")
        out.append('  (stime: estrazione dalla velocità misurata; training con early stopping, massimo tra parentesi)')
    return '\n'.join(out), pipe.get('status')


def main():
    ap = argparse.ArgumentParser(description='Monitor live V2 spatial (sola lettura)')
    ap.add_argument('--work-root', default=r'C:\BioMAP\v2_spatial_msa')
    ap.add_argument('--v2-root', default=r'C:\BioMAP\v2_spatial_work\dense_r2')
    ap.add_argument('--follow', action='store_true')
    ap.add_argument('--interval', type=float, default=15)
    ap.add_argument('--epochs-guess', type=int, default=40)
    args = ap.parse_args()
    args.interval = max(args.interval, 5)
    try:
        while True:
            text, status = snapshot(args)
            if args.follow and os.name == 'nt':
                os.system('cls')
            print(text, flush=True)
            if not args.follow or status in ('complete', 'failed'):
                break
            time.sleep(args.interval)
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
