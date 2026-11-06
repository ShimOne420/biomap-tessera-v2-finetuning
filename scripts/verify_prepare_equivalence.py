# SCOPO
#   Dimostra su dati reali che il motore corrente riproduce esattamente le celle già complete: le
#   riscarica in una cartella separata e le confronta con quelle di riferimento, senza tolleranze.
#
# COME VIENE RICHIAMATO
#   A mano, dalla cartella del progetto, prima di usare una modifica al motore su nuove celle:
#     python scripts/verify_prepare_equivalence.py --config configs/msa_v1_liguria.yaml --data-root ..\data
#       --workdir D:\verify --aois abruzzo_gran_sasso,belluno_dolomiti --cells 300
#
# LOGICA, IN ORDINE
#   1. main rifiuta una cartella di lavoro interna alla data-root di riferimento.
#   2. Avvia un processo per area; verify_area riporta l'errore di un'area senza perdere le altre.
#   3. compare_area sceglie N celle complete vicine al centro dell'area, come in un lotto reale.
#   4. Copia i cataloghi congelati, così le scene sono le stesse del riferimento.
#   5. Riacquisisce le celle con preparation.acquire_optimized e la configurazione indicata.
#   6. compare_cell confronta chiavi, dtype, forme e valori degli array con uguaglianza esatta, poi i campi
#      principali del manifest; registra anche se l'hash del file coincide.
#   7. main scrive equivalence_report.json e stampa l'esito di ogni area.
#
# INPUT E OUTPUT
#   Input: data-root di riferimento, solo in lettura. Output: serie ricalcolate nella cartella di lavoro ed
#   equivalence_report.json. Codice di uscita 1 con una differenza, un errore o nessuna cella confrontata.
#
# PARAMETRI
#   --config, --data-root, --workdir, --aois a,b, --cells N, --start-offset K (sposta il gruppo di celle),
#   --set sezione.chiave=valore.
#
# RISORSE
#   Internet, un processo per area. Su Windows usare una --workdir breve: percorsi oltre 260 caratteri
#   falliscono all'avvio dell'area.
#
# RIPRESA
#   Rieseguibile sulla stessa --workdir: le celle già scaricate non si riscaricano e il confronto si ripete.
import argparse
import json
import multiprocessing
from pathlib import Path
import shutil
import sys
import time
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from biomap_v2.common import cells,config,root
from biomap_v2 import data
from biomap_v2.preparation import acquire_optimized
from biomap_v2.preparation_profile import PreparationProfile

MANIFEST_FIELDS=['protocol','year','pixels_per_cell','sampling','bounds','valid_fraction','band_order','doy_policy','sar_units','scene_manifest']


def compare_cell(reference,candidate):
    differences=[]
    for name in ['series.npz','manifest.json']:
        if not (candidate/name).exists():differences.append(f'{name} assente')
    if differences:return differences,False
    with np.load(reference/'series.npz',allow_pickle=False) as a,np.load(candidate/'series.npz',allow_pickle=False) as b:
        if set(a.files)!=set(b.files):differences.append(f'chiavi {sorted(a.files)} != {sorted(b.files)}')
        for key in sorted(set(a.files)&set(b.files)):
            left=a[key];right=b[key]
            if left.dtype!=right.dtype:differences.append(f'{key}: dtype {left.dtype} != {right.dtype}')
            if left.shape!=right.shape:differences.append(f'{key}: forma {left.shape} != {right.shape}')
            elif not np.array_equal(left,right):differences.append(f'{key}: {int((left!=right).sum())} valori diversi')
    old=json.loads((reference/'manifest.json').read_text(encoding='utf-8'))
    new=json.loads((candidate/'manifest.json').read_text(encoding='utf-8'))
    for field in MANIFEST_FIELDS:
        if old.get(field)!=new.get(field):differences.append(f'manifest.{field} diverso')
    return differences,old.get('sha256')==new.get('sha256')


def verify_area(options):
    # One failing area must not discard the comparisons of the others.
    try:return compare_area(options)
    except Exception as error:
        return {'area':options[0],'status':f'ERRORE {type(error).__name__}: {error}','cells':0,'different':{}}


def compare_area(options):
    area,args_dict=options
    args=argparse.Namespace(**args_dict)
    reference_cfg=config('msa',argparse.Namespace(config=args.config,data_root=args.data_root,set=args.set))
    work_cfg=config('msa',argparse.Namespace(config=args.config,data_root=str(Path(args.workdir)/area),set=args.set))
    frame=cells(reference_cfg);frame=frame[frame.aoi_id==area].reset_index(drop=True)
    complete=[(data.cell_directory(reference_cfg,cell.cell_id)/'series.npz').exists() for cell in frame.itertuples()]
    frame=frame[complete].reset_index(drop=True)
    if not len(frame):return {'area':area,'status':'nessuna cella completa','cells':0,'different':[]}
    # A compact patch mirrors how one batch shares blocks; the offset moves it across the area.
    frame=frame.assign(_dist=(frame.lon-frame.lon.median()+args.start_offset*.01)**2+(frame.lat-frame.lat.median())**2)
    frame=frame.sort_values(['_dist','cell_id']).head(args.cells).drop(columns='_dist').reset_index(drop=True)
    source=root(reference_cfg)/'catalogs'/area;target=root(work_cfg)/'catalogs'/area
    if not target.exists():shutil.copytree(source,target)
    profile=PreparationProfile(work_cfg,len(frame),area);started=time.perf_counter()
    try:acquire_optimized(work_cfg,frame,argparse.Namespace(import_cache=None),profile)
    except BaseException:
        profile.finish('failed');raise
    profile.finish('complete');elapsed=time.perf_counter()-started
    different={};identical_files=0;excluded=0
    for cell in frame.itertuples():
        reference=data.cell_directory(reference_cfg,cell.cell_id);candidate=data.cell_directory(work_cfg,cell.cell_id)
        problems,same_sha=compare_cell(reference,candidate);identical_files+=int(same_sha)
        if problems:different[cell.cell_id]=problems
    state=profile.snapshot()
    return {'area':area,'status':'ok' if not different else 'DIFFERENZE','cells':len(frame),'identical_files':identical_files,
        'different':different,'seconds':elapsed,'remote_blocks_read':state['counters'].get('remote_blocks_read',0),
        'remote_read_failures':state['counters'].get('remote_read_failures',0)}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True);parser.add_argument('--data-root',required=True)
    parser.add_argument('--workdir',required=True);parser.add_argument('--aois',required=True)
    parser.add_argument('--cells',type=int,default=300);parser.add_argument('--start-offset',type=float,default=0)
    parser.add_argument('--set',action='append',default=[])
    args=parser.parse_args()
    workdir=Path(args.workdir).resolve();reference=Path(args.data_root).resolve()
    if workdir==reference or reference in workdir.parents:raise SystemExit('La cartella di lavoro deve stare fuori dalla data-root di riferimento')
    workdir.mkdir(parents=True,exist_ok=True)
    values=vars(args).copy();values['data_root']=str(reference);values['workdir']=str(workdir);values['config']=str(Path(args.config).resolve())
    areas=[area.strip() for area in args.aois.split(',')]
    with multiprocessing.get_context('spawn').Pool(len(areas)) as pool:
        results=pool.map(verify_area,[(area,values) for area in areas],chunksize=1)
    total=sum(result['cells'] for result in results);different=sum(len(result['different']) for result in results)
    report={'cells':total,'cells_with_differences':different,'files_byte_identical':sum(result.get('identical_files',0) for result in results),'areas':results}
    (workdir/'equivalence_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({key:report[key] for key in ['cells','cells_with_differences','files_byte_identical']}))
    for result in results:
        print(result['area'],result['status'],result['cells'],'celle',f'{result.get("seconds",0):.0f}s','blocchi',result.get('remote_blocks_read',0),'errori',result.get('remote_read_failures',0))
        for cell,problems in list(result['different'].items())[:5]:print(' ',cell,problems)
    if different or not total or any(result['status']!='ok' for result in results):raise SystemExit(1)


if __name__=='__main__':
    main()
