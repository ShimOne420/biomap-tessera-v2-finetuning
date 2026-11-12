# Scopo: facciata multi-AOI; audit e preflight precedono acquisizione ed estrazione.
# Percorso PC-only (dati pubblici): build-reference -> verify-legacy-sample -> stream -> probe v1/v2 -> evaluate-liguria
#   -> compare -> export. Vedere public_reference.py, streaming.py, legacy_check.py e docs/ml/DENSE_127909_PC_RUNBOOK.md.
# Tutti i lock, cataloghi e output sono nella work-root; sorgente e reference sono read-only.
import argparse
import json
import shutil
import sys
from pathlib import Path
import numpy as np
from filelock import FileLock
from biomap_v2.common import ROOT, digest, save_json, WEIGHTS_SHA256
from .state import writable, atomic_npz, cached, membership
from .reference import load, pack
from .inputs import identity, acquire, legacy_ready
from .extraction import extract
from .probing import probe, compare, liguria
from .reporting import export

# Esplicite le fasi che fanno rete: prepare e fetch-reference, mai all.
def parser():
    p=argparse.ArgumentParser(description='Campagna dense127909: 17 AOI frozen + Liguria separata')
    p.add_argument('command',choices=['pack-reference','build-reference','audit','validate','prepare','profile','extract','stream','verify-legacy-sample','probe','compare','evaluate-liguria','export','all'])
    p.add_argument('--reference',required=True);p.add_argument('--work-root',required=True)
    p.add_argument('--source-root',default='missing_source');p.add_argument('--project')
    p.add_argument('--v1-root');p.add_argument('--checkpoint');p.add_argument('--device',default='cpu')
    p.add_argument('--chunk-size',type=int,default=4096);p.add_argument('--pixel-batch',type=int,default=32)
    p.add_argument('--profile-chunks',type=int,default=3);p.add_argument('--amp',action='store_true')
    p.add_argument('--aoi',nargs='+');p.add_argument('--seed',type=int,default=20260925)
    p.add_argument('--encoder',choices=['v1','v2'],default='v2')
    p.add_argument('--reserve-gib',type=float,default=20);p.add_argument('--resume',action='store_true',help='Cache verificate riusate anche di default')
    # Percorso PC-only: reference da dati pubblici (build-reference) e V2 in streaming (stream).
    p.add_argument('--msa-area',help='GLOBIO msa_area_10sec.tif (build-reference)')
    p.add_argument('--cellarea',help='GLOBIO cellarea_km2_10sec.tif (build-reference)')
    p.add_argument('--liguria-bundle',help='Pacchetto denso ligure storico per le verifiche V1/V2')
    p.add_argument('--io-threads',type=int,default=8,help='Giorni Sentinel letti in parallelo (stream)')
    p.add_argument('--max-tiles',type=int,help='Ferma dopo N tile nuove: solo prove, non campagna completa')
    p.add_argument('--keep-pixel-embeddings',action='store_true',help='Conserva gli embedding V2 per pixel (~50 GB)')
    return p

# Dimensione e stato, non semplicemente presenza di un .lock o del memmap.
def audit(args,spec,frame):
    rows=[];reference=Path(args.reference)
    for area in spec['areas']:
        if args.aoi and area['aoi_id'] not in args.aoi:continue
        for name in area['tiles']:
            tile=reference/area['aoi_id']/name;src=Path(args.source_root)/area['aoi_id']/name
            geometry=json.loads((tile/'geometry.json').read_text());progress=src/'encoder_progress.json'
            old=json.loads(progress.read_text()) if progress.exists() else {}
            available=legacy_ready(src)
            rows.append({'aoi_id':area['aoi_id'],'tile':name,'pixels':geometry['pixels'],
                'legacy_arrays_present':available,'legacy_pixels_complete':old.get('pixels_complete',0)})
    result={'rows':rows,'n_development':127909,'n_liguria':4991,'v1_contributions':int(frame[frame.partition=='train'].n_valid.sum()),
        'checkpoint_valid':bool(args.checkpoint and Path(args.checkpoint).exists() and digest(args.checkpoint)==WEIGHTS_SHA256),
        'free_gib':shutil.disk_usage(args.work_root).free/1024**3,'source_read_only':True}
    save_json(Path(args.work_root)/'audit.json',result);return result

# Non confonde il pacchetto delle medie V1 con input temporali densi già acquisiti.
def validate(args,spec,frame):
    result=audit(args,spec,frame);missing=[]
    for row in result['rows']:
        tile=Path(args.reference)/row['aoi_id']/row['tile'];n=row['pixels']
        if row['legacy_arrays_present']:continue
        for low in range(0,n,args.chunk_size):
            high=min(n,low+args.chunk_size)
            p=Path(args.work_root)/'inputs'/row['aoi_id']/row['tile']/f'{low}-{high}.npz'
            if not cached(p,identity(tile,low,high)):missing.append(str(p))
    result['missing_input_chunks']=missing;result['preflight_passed']=not missing and result['checkpoint_valid']
    save_json(Path(args.work_root)/'preflight.json',result)
    return result

# Download esplicito: una cache per chunk, checkpoint giornalieri e spazio riservato.
def prepare(args,spec):
    work=Path(args.work_root);ref=Path(args.reference)
    for area in spec['areas']:
        aoi=area['aoi_id']
        if args.aoi and aoi not in args.aoi:continue
        for name in area['tiles']:
            tile=ref/aoi/name;n=json.loads((tile/'geometry.json').read_text())['pixels']
            src=Path(args.source_root)/aoi/name
            if legacy_ready(src):continue
            for low in range(0,n,args.chunk_size):
                high=min(n,low+args.chunk_size);tag=identity(tile,low,high)
                path=work/'inputs'/aoi/name/f'{low}-{high}.npz'
                if cached(path,tag):continue
                # Conservative headroom for annual raw observations plus day checkpoints.
                if shutil.disk_usage(work).free < args.reserve_gib*1024**3+4*1024**3:raise OSError('Spazio insufficiente: cache preservate')
                raw=acquire(tile,low,high,work);atomic_npz(path,tag,**raw)
                print(f'Input completati {aoi}/{name}: {high}/{n}',flush=True)

# Ogni invocazione registra revisione e argomenti effettivi; il lock protegge la work-root.
def main(argv=None):
    args=parser().parse_args(argv)
    if min(args.chunk_size,args.pixel_batch,args.profile_chunks,args.io_threads)<1:raise ValueError('Batch, chunk e thread devono essere positivi')
    if args.command=='build-reference':
        if not (args.v1_root and args.msa_area and args.cellarea):raise ValueError('--v1-root, --msa-area e --cellarea richiesti')
        from .public_reference import build
        return print(build(args))
    if args.command=='pack-reference':
        if not args.project or not args.v1_root:raise ValueError('--project e --v1-root richiesti')
        return print(pack(args.project,args.source_root,args.reference,args.v1_root))
    work=writable(args.work_root,[args.reference,args.source_root])
    if args.amp and not args.device.startswith('cuda'):raise ValueError('AMP supportata solo sul percorso CUDA; CPU FP32')
    if args.command=='audit' and not (Path(args.reference)/'reference.json').exists():
        f=membership(); result={'reference_ready':False,'n_development':127909,'n_liguria':4991,
            'missing':'Creare prima il pacchetto con pack-reference; vedere runbook',
            'source_geometry_tiles':len(list(Path(args.source_root).glob('*/grid_*/geometry.npz')))}
        save_json(work/'audit.json',result);print(json.dumps(result,indent=2));return
    spec,frame=load(args.reference)
    if args.aoi and not set(args.aoi)<=set(frame.aoi_id):raise ValueError('AOI sconosciuta')
    import subprocess
    with FileLock(str(work/'campaign.lock'),timeout=0):
        save_json(work/'last_invocation.json',{'args':vars(args),'git_sha':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()})
        if args.command=='audit':print(json.dumps(audit(args,spec,frame),indent=2));return
        if args.command=='validate':
            status=validate(args,spec,frame);print(json.dumps(status,indent=2))
            if not status['preflight_passed']:raise ValueError('Preflight incompleto: leggere preflight.json; eseguire prepare esplicitamente')
        elif args.command=='prepare':prepare(args,spec)
        elif args.command=='profile':extract(args,spec,frame,profile=True)
        elif args.command=='extract':extract(args,spec,frame)
        elif args.command=='stream':
            from .streaming import stream
            stream(args,spec,frame)
        elif args.command=='verify-legacy-sample':
            from .legacy_check import verify_legacy_sample
            print(json.dumps(verify_legacy_sample(args,spec,frame),indent=2))
        elif args.command=='probe':
            if args.aoi:raise ValueError('Probing completo: vietato filtro AOI')
            probe(args,frame)
        elif args.command=='evaluate-liguria':liguria(args,frame)
        elif args.command=='compare':compare(work)
        elif args.command=='export':export(work)
        elif args.command=='all':
            if args.aoi:raise ValueError('all richiede la campagna completa')
            if not validate(args,spec,frame)['preflight_passed']:raise ValueError('Input mancanti: eseguire prepare; all non scarica')
            if not (work/'profile.json').exists():raise ValueError('Eseguire profile prima di all')
            extract(args,spec,frame);probe(args,frame);liguria(args,frame);export(work)
            if (work/'probing/v1/protocol.json').exists():compare(work)
    return 0
