# SCOPO
#   Coordina prepare con un processo separato per ogni area, per usare insieme più connessioni e core.
#   Le aree sono indipendenti: celle, punti e osservazioni non cambiano, cambia solo quante ne girano insieme.
#
# COME VIENE RICHIAMATO
#   cli.py chiama run_parallel(cfg, args) quando prepare riceve --parallel-aois:
#     python scripts/probe_msa.py prepare --config configs/msa_v1_liguria.yaml --data-root ..\data --parallel-aois 15
#   Prova su alcune aree per un tempo limitato, senza validazione finale:
#     python scripts/probe_msa.py prepare --config ... --data-root ... --parallel-aois 4 --aois a,b,c,d --stop-after-minutes 20
#
# LOGICA, IN ORDINE
#   1. Richiede engine=optimized e block_cache_gib=0: una cache condivisa non è sicura tra processi.
#   2. Acquisisce il target una sola volta, così i processi lo leggono senza scritture concorrenti.
#   3. pending_cells conta le celle mancanti per area e le aree complete vengono saltate.
#   4. Divide in parti uguali lo spazio disponibile: ogni processo riceve la sua quota con --max-new-gib.
#   5. worker_command avvia "prepare --aoi AREA" per ciascuna area, fino a N insieme.
#   6. Ogni 30 secondi summary legge i progress.json e stampa celle salvate, stima, velocità e ETA.
#   7. Un processo terminato con errore riparte dopo RESTART_DELAY_SECONDS, al massimo MAX_RESTARTS volte;
#      gli altri continuano. Riparte dai propri checkpoint.
#   8. Finite tutte le aree senza --aois, esegue un prepare completo, che valida tutte le celle e scrive
#      cells.csv. Con --aois o dopo --stop-after-minutes questo passo non viene eseguito.
#
# INPUT E OUTPUT
#   Input: stessa configurazione di prepare. Output: serie delle celle, cartella diagnostics/<timestamp>_parallel
#   con un log per area e parallel_progress.json.
#
# RISORSE
#   N processi, ciascuno con i propri day_workers e io_workers; connessioni massime N x io_workers.
#   Rete e limiti del server sono il vincolo atteso.
#
# RIPRESA
#   Rilanciare lo stesso comando: celle complete e checkpoint vengono riusati. Con Ctrl+C o con lo stop a
#   tempo i processi vengono terminati; le scritture atomiche proteggono i file già completati.
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
from .common import cells, root, save_json

POLL_SECONDS=30
RESTART_DELAY_SECONDS=60
MAX_RESTARTS=5


def pending_cells(cfg,frame):
    from .data import cell_directory
    counts={}
    for cell in frame.itertuples():
        directory=cell_directory(cfg,cell.cell_id)
        if not (directory/'series.npz').exists() and not (directory/'excluded.json').exists():
            counts[cell.aoi_id]=counts.get(cell.aoi_id,0)+1
    return counts


def used_bytes(base):
    total=0
    for path in Path(base).rglob('*'):
        try:
            if path.is_file():total+=path.stat().st_size
        except OSError:continue
    return total


def worker_command(cfg,args,area,share):
    command=[sys.executable,str(Path(sys.argv[0]).resolve()),'prepare','--data-root',cfg['data_root'],
        '--aoi',area,'--max-new-gib',f'{share:.3f}']
    if args.config:command+=['--config',str(Path(args.config).resolve())]
    if args.target_raster:command+=['--target-raster',args.target_raster]
    for expression in args.set or []:command+=['--set',expression]
    return command


def latest_progress(diagnostics,area):
    reports=sorted(diagnostics.glob(f'*_{area}_prepare'),key=lambda path:path.name)
    if not reports:return None
    try:return json.loads((reports[-1]/'progress.json').read_text(encoding='utf-8'))
    except (OSError,ValueError):return None


def summary(diagnostics,areas,pending,started,running,restarts):
    # Finished days give a live estimate; cells are counted only once saved to disk.
    saved=0;estimated=0;reads=0;failures=0;details={}
    for area in areas:
        state=latest_progress(diagnostics,area)
        if state is None:continue
        new=state.get('new_cells',0);saved+=new;counters=state.get('counters',{})
        reads+=counters.get('remote_blocks_read',0);failures+=counters.get('remote_read_failures',0)
        total=state.get('batch_days_total',0);done=state.get('batch_days_done',0);cells_in_batch=state.get('batch_cells',0)
        estimated+=max(new,cells_in_batch*done/total if total else 0)
        details[area]={'new_cells':new,'days_done':done,'days_total':total,'phase':state.get('phase'),'status':state.get('status')}
    elapsed=time.monotonic()-started;total=sum(pending.values())
    rate=estimated/elapsed*3600 if elapsed>0 else 0
    return {'elapsed_seconds':elapsed,'running':sorted(running),'pending_at_start':total,'saved_cells':saved,
        'estimated_cells':estimated,'estimated_cells_per_hour':rate,'eta_hours':(total-estimated)/rate if rate else None,
        'remote_blocks_read':reads,'blocks_per_second':reads/elapsed if elapsed>0 else 0,'remote_read_failures':failures,
        'restarts':restarts,'areas':details}


def run_parallel(cfg,args):
    from .data import prepare
    from .target_data import ensure_target
    if cfg['acquisition']['engine']!='optimized' or cfg['acquisition']['block_cache_gib']!=0:
        raise ValueError('--parallel-aois richiede acquisition.engine=optimized e block_cache_gib=0')
    frame=cells(cfg);known=set(frame.aoi_id)
    selected=[name.strip() for name in args.aois.split(',')] if args.aois else sorted(known)
    unknown=set(selected)-known
    if unknown:raise ValueError(f'Aree sconosciute: {sorted(unknown)}')
    ensure_target(cfg,frame,args.target_raster)
    counts=pending_cells(cfg,frame)
    queue=[area for area in sorted(selected,key=lambda name:(-counts.get(name,0),name)) if counts.get(area,0)]
    pending={area:counts[area] for area in queue}
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    diagnostics=root(cfg)/'diagnostics';logs=diagnostics/f'{stamp}_parallel';logs.mkdir(parents=True,exist_ok=True)
    limit=min(args.parallel_aois,len(queue)) if queue else 0
    cap=cfg['storage']['max_gib'];free=shutil.disk_usage(cfg['data_root']).free/1024**3-cfg['storage']['reserve_gib']
    available=min(cap-used_bytes(cfg['data_root'])/1024**3,free)
    share=max(available,0)/limit if limit else 0
    print(f'parallel: {sum(pending.values())} celle in {len(queue)} aree, {limit} processi, {share:.2f} GiB per processo; log {logs}',flush=True)
    running={};restarts={area:0 for area in queue};retry_at={};failed=[];started=time.monotonic();last_print=0
    stop=time.monotonic()+args.stop_after_minutes*60 if args.stop_after_minutes else None
    stopped=False

    def launch(area):
        output=(logs/f'{area}.log').open('a',encoding='utf-8')
        process=subprocess.Popen(worker_command(cfg,args,area,share),stdout=output,stderr=subprocess.STDOUT)
        running[area]=(process,output)

    def halt():
        for process,output in running.values():process.terminate()
        for process,output in running.values():
            process.wait();output.close()
        running.clear()

    try:
        while queue or running or retry_at:
            now=time.monotonic()
            if stop is not None and now>=stop:
                halt();stopped=True;break
            for area,(process,output) in list(running.items()):
                code=process.poll()
                if code is None:continue
                output.close();del running[area]
                if code==0:continue
                restarts[area]+=1
                if restarts[area]>MAX_RESTARTS:
                    failed.append(area);print(f'parallel: {area} fallita dopo {MAX_RESTARTS} riavvii; vedere il log',flush=True)
                else:
                    retry_at[area]=now+RESTART_DELAY_SECONDS
                    print(f'parallel: {area} terminata con codice {code}; riavvio {restarts[area]}/{MAX_RESTARTS}',flush=True)
            for area,when in list(retry_at.items()):
                if now>=when and len(running)<limit:
                    del retry_at[area];launch(area)
            while queue and len(running)<limit:
                launch(queue.pop(0));time.sleep(1)
            if now-last_print>=POLL_SECONDS:
                last_print=now;state=summary(diagnostics,list(pending),pending,started,running,restarts)
                save_json(logs/'parallel_progress.json',state)
                eta=f'{state["eta_hours"]:.1f}h' if state['eta_hours'] is not None else 'n/d'
                print(f'parallel: attive={len(running)} salvate={state["saved_cells"]} stimate={state["estimated_cells"]:.0f}/{state["pending_at_start"]} | {state["estimated_cells_per_hour"]:.0f}/h ETA={eta} | blocchi/s={state["blocks_per_second"]:.1f} errori={state["remote_read_failures"]}',flush=True)
            time.sleep(1)
    except KeyboardInterrupt:
        halt();raise
    state=summary(diagnostics,list(pending),pending,started,running,restarts)
    state['status']='stopped' if stopped else ('failed' if failed else 'complete');state['failed']=failed
    save_json(logs/'parallel_progress.json',state)
    print(f'parallel: {state["status"]}; salvate={state["saved_cells"]} stimate={state["estimated_cells"]:.0f}; report {logs/"parallel_progress.json"}',flush=True)
    if failed:raise RuntimeError(f'Aree non completate: {failed}. Rilanciare lo stesso comando per riprendere')
    if stopped or args.aois:return None
    args.aoi=None
    print(prepare(cfg,args),flush=True)
    return None
