# Scopo: distinguere BLOCKED, READY_FOR_EXTRACTION e READY_FOR_TRAINING prima di ogni run V1.
# Fasi: membership/target -> inventario spaziale -> schema/hash cache -> device/disco/permessi.
# Input: percorsi CLI e config; output: preflight JSON privato e cause precise, nessun download.
# Parametri: --deep ricalcola hash sorgenti; --require-training impone cache pronta, --preset.
# Risorse: CPU/metadati; CUDA solo interrogata; nessun encoder o modifica di processi V2.
# Ripresa: rieseguibile dopo acquisizione/build; una cache assente non viene dichiarata pronta al training.
import json
from pathlib import Path
import os
import sqlite3
import shutil
import tempfile
from runtime import common_parser,configuration,membership,cache_path,save_json,Budget
from sources import inventory,target_values


def inspect(cfg,deep=False):
    report={'status':'BLOCKED','protocol':cfg['protocol'],'preset':cfg['preset'],'checks':{},'blockers':[],
        'encoder':{'mode':'precomputed V1.0 frozen','checkpoint_required':False,'checkpoint_locally_verified':False,'raw_sentinel_required':False}}
    try:
        frame=membership(cfg);report['checks']['membership']={'cells':len(frame),'development':int((frame.partition=='train').sum()),'liguria':int((frame.partition=='test').sum()),'aois':frame.aoi_id.nunique()}
    except Exception as error:report['blockers'].append('Membership: '+str(error));return report
    try:
        work=Path(cfg['work_root']);work.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryFile(dir=work) as f:f.write(b'preflight')
        report['checks']['storage']=Budget(cfg).check(32*1024**2)
    except Exception as error:report['blockers'].append('Storage: '+str(error))
    try:values,proof=target_values(cfg,frame);report['checks']['targets']={'valid':len(values),'provenance':proof}
    except Exception as error:values=None;proof=None;report['blockers'].append('Target: '+str(error))
    try:
        tiles,missing,bad=inventory(cfg,frame)
        report['checks']['spatial_sources']={'present_tiles':len(tiles),'missing_tiles':missing,'invalid_tiles':bad,'channels':128,
            'shapes':sorted({tuple([t['height'],t['width']]) for t in tiles}),'resolutions_m':sorted({tuple(t['resolution_m']) for t in tiles})}
        if bad:report['blockers'].append('Sorgenti spaziali incompatibili')
        if missing:report['checks']['acquisition_required']=True
    except Exception as error:tiles=[];missing=[];report['blockers'].append('Sorgenti: '+str(error))
    import torch
    requested=cfg['training']['device'];cuda=bool(torch.cuda.is_available())
    gpu={'requested':requested,'cuda_available':cuda,'torch':torch.__version__}
    if cuda:
        device=torch.device(requested if requested.startswith('cuda') else 'cuda:0')
        gpu.update(name=torch.cuda.get_device_name(device),vram_bytes=torch.cuda.get_device_properties(device).total_memory)
    report['checks']['device']=gpu
    if os.name=='nt':
        try:
            import ctypes
            class Memory(ctypes.Structure):
                _fields_=[('length',ctypes.c_ulong),('load',ctypes.c_ulong),('physical',ctypes.c_ulonglong),('available',ctypes.c_ulonglong),('page',ctypes.c_ulonglong),('page_available',ctypes.c_ulonglong),('virtual',ctypes.c_ulonglong),('virtual_available',ctypes.c_ulonglong),('extended',ctypes.c_ulonglong)]
            m=Memory();m.length=ctypes.sizeof(Memory)
            if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m)):raise OSError('GlobalMemoryStatusEx')
            report['checks']['ram']={'physical_bytes':m.physical,'available_bytes':m.available}
        except Exception:report['checks']['ram']={'physical_bytes':None,'available_bytes':None}
    else:report['checks']['ram']={'physical_bytes':os.sysconf('SC_PHYS_PAGES')*os.sysconf('SC_PAGE_SIZE'),'available_bytes':None}
    ready_cache=False;path=cache_path(cfg)
    if path.exists():
        try:
            from dataset import metadata,cell_frame,verify_sources
            from patch_builder import build_identity
            meta=metadata(path)
            if not meta.get('finalized'):raise ValueError('Transazioni complete ma indice non finalizzato')
            if values is None or missing:raise ValueError('Target o sorgenti mancanti')
            if meta['identity']!=build_identity(cfg,values,proof,tiles):raise ValueError('Config/geometrie/target/sorgenti differenti dalla cache')
            count=verify_sources(path,deep);cache=cell_frame(path)
            if cache.cell_id.duplicated().any() or not set(cache.cell_id)<=set(frame.cell_id):raise ValueError('ID cache non validi')
            if len(cache)!=len(frame) and not cfg['data']['allow_partial_cells']:raise ValueError('Sottoinsieme non autorizzato')
            dev=cache[cache.partition=='train']
            if set(dev.cv_fold)!={0,1,2} or dev.aoi_id.nunique()!=17:raise ValueError('Cache senza 17 AOI/tre fold di sviluppo')
            if set(cache[cache.partition=='test'].aoi_id)!={'liguria_levante'}:raise ValueError('Cache priva di Liguria')
            report['checks']['cache']={'eligible':len(cache),'excluded':len(frame)-len(cache),'verified_tiles':count,'deep':deep}
            ready_cache=True
        except Exception as error:report['blockers'].append('Cache: '+str(error))
    else:report['checks']['cache']={'available':False}
    if requested.startswith('cuda') and not cuda:report['checks']['training_device_blocker']='CUDA richiesta ma non disponibile'
    if not report['blockers']:
        report['status']='READY_FOR_TRAINING' if ready_cache and (cuda or not requested.startswith('cuda')) else 'READY_FOR_EXTRACTION'
    report['next_action']='train/profile' if report['status']=='READY_FOR_TRAINING' else ('fetch if needed, then build' if report['status']=='READY_FOR_EXTRACTION' else 'resolve blockers; targets/fetch are explicit commands')
    return report


def main():
    p=common_parser('Preflight V1 spatial, senza scaricare o addestrare')
    p.add_argument('--deep',action='store_true');p.add_argument('--require-training',action='store_true')
    args=p.parse_args();cfg=configuration(args);report=inspect(cfg,args.deep)
    save_json(Path(cfg['work_root'])/'preflight'/f"{cfg['preset']}.json",report)
    print(json.dumps(report,indent=2,ensure_ascii=False))
    if report['status']=='BLOCKED' or (args.require_training and report['status']!='READY_FOR_TRAINING'):raise SystemExit(2)


if __name__=='__main__':main()
