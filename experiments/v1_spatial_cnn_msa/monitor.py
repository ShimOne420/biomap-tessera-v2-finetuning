# Scopo: leggere in un secondo terminale l'avanzamento V1 senza influenzare il run.
# Fasi: status atomico -> fase/celle/epoche -> velocità/ETA -> aggiornamento ogni cinque secondi.
# Input: work-root privata; output: solo terminale, nessuna scrittura o uso GPU.
# Parametri: --data-root ../data, --follow, --interval 5; Ctrl+C ferma soltanto il monitor.
# Risorse: un piccolo JSON; ETA è stima osservata, early stopping e fasi diverse cambiano il tempo.
# Ripresa: il monitor può essere riaperto; segnala heartbeat vecchio invece di presumere successo.
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import time


def duration(seconds):
    if seconds is None:return 'in stima'
    seconds=max(0,int(seconds));return f'{seconds//3600:02d}:{seconds//60%60:02d}:{seconds%60:02d}'


def main():
    p=argparse.ArgumentParser(description='Monitor live read-only V1 spatial')
    p.add_argument('--data-root',default='../data');p.add_argument('--work-root');p.add_argument('--follow',action='store_true');p.add_argument('--interval',type=float,default=5)
    args=p.parse_args();path=Path(args.work_root or Path(args.data_root)/'v1_spatial_msa')/'status.json'
    if args.interval<1:p.error('interval minimo 1 secondo')
    try:
        while True:
            try:
                s=json.loads(path.read_text(encoding='utf-8'));age=(datetime.now(timezone.utc)-datetime.fromisoformat(s['updated_utc'])).total_seconds()
                pipeline_path=path.with_name('pipeline.json');pipeline=None
                if pipeline_path.exists():
                    candidate=json.loads(pipeline_path.read_text(encoding='utf-8'))
                    # Ignore a previous pipeline when a newer standalone Python phase is running.
                    end=candidate.get('finished_utc')
                    if not end or datetime.fromisoformat(s['started_utc'])<=datetime.fromisoformat(end):pipeline=candidate
                print(f"[{s['updated_utc']}] {s['status']} {s['phase']} ({s['preset']}) | {s.get('done',0)}/{s.get('total',0)} {s.get('unit','items')} | trascorso {duration(s.get('elapsed_seconds'))} | ETA {duration(s.get('eta_seconds'))}",flush=True)
                if s.get('rate_per_second') is not None:print(f"  velocità {s['rate_per_second']:.3f} {s.get('unit','items')}/s",flush=True)
                if 'epoch' in s:print(f"  {s.get('current_model')} fold={s.get('fold')} epoca={s['epoch']}/{s.get('epochs_max')} batch={s.get('batch')}/{s.get('batches')} | budget residuo fase {duration(s.get('phase_eta_seconds'))} | {s.get('eta_note','')}",flush=True)
                if 'tile' in s:print('  tile:',s['tile'],'file:',s.get('current_file',''),flush=True)
                if s.get('bytes_total'):print(f"  file {s.get('bytes_done',0)/1024**2:.1f}/{s['bytes_total']/1024**2:.1f} MiB",flush=True)
                if s.get('gpu_peak_allocated_gib') is not None:print(f"  VRAM peak {s['gpu_peak_allocated_gib']:.2f} GiB",flush=True)
                if age>30 and s['status']=='running':print(f'  Heartbeat vecchio di {int(age)} s: non prova che il run sia ancora attivo.',flush=True)
                if s.get('error'):print('  ERRORE:',s['error'],flush=True)
                if pipeline:print('  sequenza:',pipeline['status'],pipeline.get('stage',''),flush=True)
                if not args.follow or (pipeline and pipeline['status']!='running') or (not pipeline and s['status']!='running'):break
            except (FileNotFoundError,json.JSONDecodeError):
                print('In attesa di status:',path,flush=True)
                if not args.follow:break
            time.sleep(args.interval)
    except KeyboardInterrupt:pass


if __name__=='__main__':main()
