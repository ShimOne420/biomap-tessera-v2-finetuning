# Scopo: acquisizione annuale storica in nuove root e dataset di celle, senza leggere target per il sampling.
# Fasi: reference verificato -> piano pixel deterministico -> STAC copiato/nuovo -> giorni atomici -> NPY.
# Input: reference originale, scelta sampling esplicita; output: blocchi raw e manifest, nessun embedding inventato.
# Parametri: full_support/uniform_pixels/center_windows, K/lato, block_pixels e thread I/O; uso tramite acquire.
# Risorse: CPU/rete/disco; nessuna riduzione temporale. Ripresa per giorno/blocco, checksum prima del riuso.
# Dataset: memmap locali aperti e chiusi in ogni accesso; worker spawn picklable, default zero worker.
# Bridge sampling opt-in: ricostruisce richieste dal manifest frozen senza acquisire, ricampionare o approvare dati.
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import shutil
import time
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from biomap_v2.campaign import streaming as historical
from biomap_v2.campaign.reference import load as load_reference
from biomap_spatial.support import windows
from .config import require_ready
from .io import checked_roots,claim_root,signature,digest,save_json,commit_arrays,load_arrays,disk_budget,Progress

RAW_KEYS=('s2','mask','s2_doys','asc','asc_doys','desc','desc_doys')


def spatial_sampling_requests(manifest_cell,mode):
    """Opt-in contract bridge only: preserve tile, geometry positions, masks and contribution weights.

    No acquisition, encoder call, default change or implicit approval. Temporal Sentinel masks remain
    a future real-data validation requirement; benchmark embeddings cannot reconstruct those masks.
    """
    from biomap_sampling.data import future_requests
    return future_requests(manifest_cell,mode)


def spatial_sampling_requests_from_plan(plan_root,cell_id,mode,reference_root):
    """Verified exact bridge for future authorized raw acquisition; never invokes an encoder/network."""
    from biomap_sampling.data import Plan,future_requests,verify_requests
    plan=plan_root if isinstance(plan_root,Plan) else Plan(plan_root)
    result=future_requests(plan.cell(cell_id),mode)
    verify_requests(result,reference_root);result['plan_identity']=plan.manifest['identity']
    return result


def reference(cfg):
    root=Path(cfg['paths']['reference'])
    spec,frame=load_reference(root)
    from biomap_spatial.sources import validate_frame
    validate_frame(frame,'benchmark')
    frame=frame.sort_values('reference_index').reset_index(drop=True)
    if frame.reference_index.duplicated().any():raise ValueError('reference_index duplicato')
    if cfg['experiment']['profile']=='smoke':
        frame=frame[(frame.partition=='train') & (frame.aoi_id==cfg['sampling']['smoke_aoi'])].head(cfg['sampling']['smoke_cells']).copy()
        if len(frame)!=cfg['sampling']['smoke_cells']:raise ValueError('AOI/celle smoke non disponibili')
    if frame.empty:raise ValueError('Reference vuoto')
    return spec,frame


def rank_positions(seed,aoi,tile,positions):
    # Stable across Python/NumPy versions; independent of targets and iteration ordering.
    return np.array([int.from_bytes(hashlib.blake2b(f'{seed}/{aoi}/{tile}/{p}'.encode(),digest_size=8).digest(),'little') for p in positions],dtype=np.uint64)


def select_aoi(entries,sampling,seed):
    """entries: tile -> (reference IDs, flat indices, width). K is GLOBAL per cell across tiles."""
    result={k:[] for k in entries};cells={}
    for tile,(ids,flat,width) in entries.items():
        ordered=np.argsort(ids,kind='stable');ordered=ordered[ids[ordered]>=0]
        groups=np.split(ordered,np.flatnonzero(np.diff(ids[ordered]))+1) if len(ordered) else []
        for pos in groups:cells.setdefault(int(ids[pos[0]]),[]).append((tile,pos))
    for cell,parts in sorted(cells.items()):
        if sampling['method']=='full_support':
            for tile,pos in parts:result[tile].extend(pos.tolist())
        elif sampling['method']=='uniform_pixels':
            all_parts=[(tile,int(p)) for tile,pos in parts for p in pos]
            ranked=sorted(all_parts,key=lambda x:(rank_positions(seed,str(cell),x[0],[x[1]])[0],x))
            for tile,p in ranked[:sampling['pixels_per_cell']]:result[tile].append(p)
        elif sampling['method']=='center_windows':
            side=sampling['window_side']
            for tile,pos in parts:
                _,flat,width=entries[tile];rr=flat[pos]//width;cc=flat[pos]%width
                middle=np.array([(rr.min()+rr.max())/2,(cc.min()+cc.max())/2])
                anchor=np.argmin((rr-middle[0])**2+(cc-middle[1])**2)
                r0=int(rr[anchor])-side//2;c0=int(cc[anchor])-side//2
                chosen=pos[(rr>=r0)&(rr<r0+side)&(cc>=c0)&(cc<c0+side)]
                result[tile].extend(chosen.tolist())
        else:raise ValueError('Sampling undecided: piano non eseguibile')
    return {k:np.array(sorted(v),dtype=np.int64) for k,v in result.items()}


def plan(cfg):
    checked_roots(cfg);spec,frame=reference(cfg);root=Path(cfg['paths']['reference']);eligible=set(frame.reference_index)
    selection=[]
    for area in spec['areas']:
        aoi=area['aoi_id']
        if not (frame.aoi_id==aoi).any():continue
        lookup=np.load(root/aoi/'lookup.npy');entries={}
        for tile in area['tiles']:
            folder=root/aoi/tile;geo=json.loads((folder/'geometry.json').read_text(encoding='utf-8'))
            with np.load(folder/'geometry.npz',allow_pickle=False) as z:
                ids=lookup[z['cell_ids']];ids=np.where(np.isin(ids,list(eligible)),ids,-1)
                entries[tile]=(ids,z['flat_indices'].copy(),geo['width'])
        chosen=select_aoi(entries,cfg['sampling'],cfg['experiment']['seed'])
        for tile,positions in chosen.items():
            if not len(positions):continue
            ids,flat,width=entries[tile];folder=root/aoi/tile
            selection.append({'aoi':aoi,'tile':tile,'positions':positions,'ids':ids[positions],
                'flat':flat[positions],'width':width,'geometry':digest(folder/'geometry.npz')})
    counts={int(x):0 for x in frame.reference_index}
    for s in selection:
        for cell,n in zip(*np.unique(s['ids'],return_counts=True)):counts[int(cell)]+=int(n)
    if any(n<1 for n in counts.values()):raise ValueError('Cella senza supporto: non viene eliminata')
    if cfg['sampling']['method']=='full_support' and any(counts[int(r.reference_index)]!=r.n_valid for r in frame.itertuples()):
        raise ValueError('Supporto pieno diverso dai conteggi storici')
    identity=signature({'reference':digest(root/'reference.json'),'sampling':cfg['sampling'],'profile':cfg['experiment']['profile'],
        'seed':cfg['experiment']['seed'],'preprocessing':historical.code_identity(),'adapter_source_sha256':digest(Path(__file__)),
        'acquisition':cfg['acquisition'],
        'tiles':[{'aoi':s['aoi'],'tile':s['tile'],'geometry':s['geometry'],'positions_sha':hashlib.sha256(s['positions'].tobytes()).hexdigest()} for s in selection]})
    return frame,selection,counts,identity


def estimates(cfg):
    if cfg['sampling']['method']=='undecided':
        return {'status':'DECISIONS_REQUIRED','reason':'Sampling undecided: nessun piano o download automatico'}
    frame,tiles,counts,identity=plan(cfg);n=sum(counts.values())
    # Explicit illustrative scene counts; actual catalog is needed for a storage admission decision.
    return {'status':'ESTIMATED_NOT_ADMITTED','identity':identity,'cells':len(frame),'aois':int(frame.aoi_id.nunique()),
        'tiles':len(tiles),'pixels':n,'illustrative_s2_days':100,'illustrative_s1_days':160,
        'raw_bytes_illustrative':n*(21*100+4*160),'raw_peak_bytes_illustrative':2*n*(21*100+4*160),
        'prefix_bytes_fp32_104_160':n*440*(104+160)*4,
        'prefix_bytes_fp32_minimum_9_tokens':n*440*9*4,
        'prefix_bytes_fp32_maximum_512_tokens':n*440*512*4,
        'checkpoint_archive_bytes_upper_estimate':None if cfg['training']['epochs'] is None else 4*cfg['training']['epochs']*(21031506*4+{1:6494594,2:11208818,3:15923042}[cfg['finetuning']['last_n_blocks']]*8+1024**2),
        'network_100mbps_lower_bound_seconds_illustrative':n*(21*100+4*160)/12500000,
        'wall_time_seconds':None,'time_note':'Lower bound payload only; COG blocks, all annual scenes, retries and geography dominate; calibrate real pilot',
        'gpu_memory_estimate':'Not measured; microbatch and optimizer pilot mandatory, no automatic OOM method change',
        'cache_decision':'ONLINE_RECOMMENDED; cached corpus disabled, FP32 prototype only'}


def catalogs(cfg,entry,raw_root):
    aoi,tile=entry['aoi'],entry['tile'];folder=raw_root/'catalogs'/aoi/tile;folder.mkdir(parents=True,exist_ok=True)
    old=cfg['paths'].get('catalog_root')
    for name in ['sentinel-2-l2a_edge_bbox.json','sentinel-1-rtc_edge_bbox.json']:
        dest=folder/name
        if not dest.exists() and old and (Path(old)/aoi/tile/name).is_file():shutil.copyfile(Path(old)/aoi/tile/name,dest)
    # Full tile bbox, even for a sampled subset: exactly the historical selection rule.
    tile_path=Path(cfg['paths']['reference'])/aoi/tile
    import requests
    for attempt in range(cfg['acquisition']['catalog_attempts']):
        try:
            streams,box=historical.catalog(tile_path,historical.Pixels(tile_path,entry['positions'][:1]),folder);break
        except (requests.RequestException,OSError):
            if attempt+1==cfg['acquisition']['catalog_attempts']:raise
            time.sleep(min(2**attempt,8))
    hashes={p.name:digest(p) for p in folder.glob('*_edge_bbox.json')}
    return streams,box,hashes


def acquire(cfg,resume=False):
    require_ready(cfg,'acquire');checked_roots(cfg)
    frame,entries,counts,identity=plan(cfg);root=Path(cfg['paths']['raw_root'])
    if root.exists() and (root/'ownership.json').exists() and not resume:raise ValueError('Root esistente: usare --resume')
    claim_root(root,identity,'raw');save_json(root/'effective_config.json',cfg)
    progress=Progress(root/'progress.json');started=time.perf_counter();done=0;blocks=[];timings=historical.Timings();new_done=0;new_seconds=0.
    source=historical.reader(timings,root/'reader')
    try:
        catalog_records={};remaining_bytes=0
        for entry in entries:
            tick=time.perf_counter();streams,box,cat_hash=catalogs(cfg,entry,root)
            bpp=21*len(streams['s2'])+4*(len(streams['asc'])+len(streams['desc']))
            catalog_records[(entry['aoi'],entry['tile'])]={'bbox':box,'hashes':cat_hash,'bytes_per_pixel':bpp,'seconds':time.perf_counter()-tick}
            for low in range(0,len(entry['positions']),cfg['acquisition']['block_pixels']):
                high=min(low+cfg['acquisition']['block_pixels'],len(entry['positions']))
                key=f"{entry['aoi']}/{entry['tile']}/{low:08d}-{high:08d}"
                if not (root/'blocks'/key/'complete.json').exists():remaining_bytes+=(high-low)*(2*bpp+64)+4096*(sum(len(x) for x in streams.values())+20)
            del streams
        save_json(root/'storage_admission.json',{'additional_bytes_conservative':remaining_bytes,'reserve_gib':cfg['storage']['reserve_gib'],
            'basis':'all actual annual STAC day counts; raw+immutable day copies+NPY/filesystem allowance, not a payload download forecast'})
        disk_budget(root,cfg['storage']['reserve_gib'],remaining_bytes)
        for entry in entries:
            geometry_dest=root/'geometries'/entry['aoi']/(entry['tile']+'.json')
            geometry_dest.parent.mkdir(parents=True,exist_ok=True)
            geometry_src=Path(cfg['paths']['reference'])/entry['aoi']/entry['tile']/'geometry.json'
            if not geometry_dest.exists():shutil.copyfile(geometry_src,geometry_dest)
            tick=time.perf_counter();streams,box,cat_hash=catalogs(cfg,entry,root)
            query_seconds=time.perf_counter()-tick
            # Plan with ACTUAL catalog lengths, peak includes all immutable day-cache + final raw copies.
            bytes_per_pixel=21*len(streams['s2'])+4*(len(streams['asc'])+len(streams['desc']))
            disk_budget(root,cfg['storage']['reserve_gib'])
            for low in range(0,len(entry['positions']),cfg['acquisition']['block_pixels']):
                high=min(low+cfg['acquisition']['block_pixels'],len(entry['positions']))
                key=f"{entry['aoi']}/{entry['tile']}/{low:08d}-{high:08d}"
                tag=signature({'dataset':identity,'block':key,'catalogs':cat_hash})
                final=root/'blocks'/key
                meta={'aoi':entry['aoi'],'tile':entry['tile'],'low':low,'high':high,'bbox':box,'catalogs':cat_hash}
                block_started=time.perf_counter();reused=final.exists()
                if reused:arrays,_=load_arrays(final,tag,mmap=True)
                else:
                    pixels=historical.Pixels(Path(cfg['paths']['reference'])/entry['aoi']/entry['tile'],entry['positions'][low:high]);raw={}
                    for name in ['s2','asc','desc']:
                        days=sorted(streams[name].items())
                        def one(day_items,name=name):
                            day,items=day_items;path=root/'days'/key/(name+'_'+day)
                            day_tag=signature({'block':tag,'stream':name,'day':day,'scenes':sorted(x['id'] for x in items)})
                            if path.exists():values,_=load_arrays(path,day_tag)
                            else:
                                bands,valid=historical.compose_day(name,items,pixels,source)
                                values={'bands':bands,'valid':valid};commit_arrays(path,values,day_tag)
                            return values['bands'],values['valid']
                        with ThreadPoolExecutor(max_workers=cfg['acquisition']['io_workers']) as pool:parts=list(pool.map(one,days))
                        width,dtype=(10,np.uint16) if name=='s2' else (2,np.int16)
                        raw[name]=np.stack([x[0] for x in parts],1) if parts else np.zeros((high-low,0,width),dtype)
                        raw[name+'_doys']=np.array([historical.doy(day) for day,_ in days],np.int16)
                        if name=='s2':raw['mask']=np.stack([x[1] for x in parts],1) if parts else np.zeros((high-low,0),bool)
                    if historical.coverage(raw,len(pixels))['pixels_without_any_input']:raise ValueError('Pixel senza S1/S2: nessuna imputazione o esclusione')
                    arrays={**raw,'positions':entry['positions'][low:high],'flat':entry['flat'][low:high],'cell_ref':entry['ids'][low:high]}
                    commit_arrays(final,arrays,tag,meta)
                done+=high-low
                seconds=time.perf_counter()-block_started
                if not reused:new_done+=high-low;new_seconds+=seconds
                historical.append_jsonl(root/'acquisition_log.jsonl',{'block':key,'reused':reused,'pixels':high-low,'seconds':seconds,'reader':timings.reset()})
                blocks.append({'key':key,'identity':tag,'marker_sha256':digest(final/'complete.json'),**meta})
                elapsed=time.perf_counter()-started;rate=new_done/max(new_seconds,1e-9) if new_done else None
                progress.update(force=True,phase='acquire',done=done,total=sum(counts.values()),unit='pixels',rate_per_second=rate,
                    eta_seconds=(sum(counts.values())-done)/rate if rate else None,current=key,
                    newly_acquired_pixels=new_done,reused_pixels=done-new_done,rate_basis='new raw acquisition only; remaining cached blocks make ETA conservative')
                print(f'Acquisiti/verificati {done}/{sum(counts.values())} pixel: {key}',flush=True)
            save_json(root/'last_tile_timing.json',{'tile':entry['tile'],'catalog_seconds':query_seconds,'reader':timings.reset()})
    except BaseException as error:
        progress.update(force=True,phase='interrupted',error=repr(error),eta_seconds=None);raise
    finally:source.close()
    frame=frame.copy();frame['selected_pixel_contributions']=[counts[int(i)] for i in frame.reference_index]
    from biomap_v2.dense_fast.io_safe import unique_tmp,replace_with_retry
    cell_path=root/'cells.parquet'
    if cell_path.exists():
        if not pd.read_parquet(cell_path).equals(frame):raise ValueError('Celle esistenti incompatibili: nessun overwrite')
    else:
        tmp=unique_tmp(cell_path);frame.to_parquet(tmp,index=False)
        if not pd.read_parquet(tmp).equals(frame):raise ValueError('Rilettura celle discordante')
        replace_with_retry(tmp,cell_path)
    dataset={'identity':identity,'blocks':blocks,'counts':{str(k):v for k,v in counts.items()},
        'cells_sha256':digest(root/'cells.parquet'),'cells':len(frame),'profile':cfg['experiment']['profile'],
        'geometries':{p.relative_to(root).as_posix():digest(p) for p in (root/'geometries').rglob('*.json')},
        'support':'full_historical' if cfg['sampling']['method']=='full_support' else 'sampled_new_support_requires_matched_control'}
    save_json(root/'dataset.json',dataset);progress.update(force=True,phase='complete',eta_seconds=0)
    return dataset


class CellDataset(Dataset):
    def __init__(self,root,cell_ids=None,verify=True):
        self.root=Path(root);self.spec=json.loads((self.root/'dataset.json').read_text(encoding='utf-8'))
        self.geometries={}
        for name,sha in self.spec['geometries'].items():
            if digest(self.root/name)!=sha:raise ValueError('Geometria raw modificata')
            self.geometries[name]=json.loads((self.root/name).read_text(encoding='utf-8'))
        if digest(self.root/'cells.parquet')!=self.spec['cells_sha256']:raise ValueError('Celle/target modificati')
        self.frame=pd.read_parquet(self.root/'cells.parquet')
        if cell_ids is not None:self.frame=self.frame[self.frame.cell_id.isin(cell_ids)].reset_index(drop=True)
        self.parts={int(r):[] for r in self.frame.reference_index}
        for block in self.spec['blocks']:
            folder=self.root/'blocks'/block['key']
            if digest(folder/'complete.json')!=block['marker_sha256']:raise ValueError('Manifest blocco modificato')
            arrays,_=load_arrays(folder,block['identity'],verify=verify,mmap=True)
            for r in np.unique(arrays['cell_ref']):
                if int(r) in self.parts:self.parts[int(r)].append(block)
            del arrays
        for r in self.frame.itertuples():
            if not self.parts[int(r.reference_index)]:raise ValueError('Cella mancante')

    def __len__(self):return len(self.frame)

    def __getitem__(self,index):
        row=self.frame.iloc[index];grouped={}
        for block in self.parts[int(row.reference_index)]:grouped.setdefault((block['aoi'],block['tile']),[]).append(block)
        groups=[];patches=[]
        for (aoi,tile),blocks in sorted(grouped.items()):
            parts=[];flats=[]
            for block in blocks:
                arrays,_=load_arrays(self.root/'blocks'/block['key'],block['identity'],verify=False,mmap=True)
                take=np.flatnonzero(arrays['cell_ref']==row.reference_index)
                parts.append({k:np.array(arrays[k][take] if k in {'s2','mask','asc','desc'} else arrays[k]) for k in RAW_KEYS})
                flats.append(np.array(arrays['flat'][take]));del arrays
            for p in parts[1:]:
                for k in ['s2_doys','asc_doys','desc_doys']:
                    if not np.array_equal(p[k],parts[0][k]):raise ValueError('Date diverse tra blocchi della stessa tile')
            raw={k:np.concatenate([p[k] for p in parts]) if k in {'s2','mask','asc','desc'} else parts[0][k] for k in RAW_KEYS}
            flat=np.concatenate(flats);geometry=self.geometries[f'geometries/{aoi}/{tile}.json']
            width=geometry['width'];rr=flat//width;cc=flat%width;r0=int(rr.min());c0=int(cc.min())
            mask=np.zeros((int(rr.max())-r0+1,int(cc.max())-c0+1),bool);mask[rr-r0,cc-c0]=True
            group=len(groups);groups.append({'raw':raw,'owner':0,'aoi':aoi,'fold':int(row.cv_fold),'tile':tile,
                'positions':flat,'grid':geometry,'support_policy':'preserve_legacy_contributions'})
            for r,c,m in windows(mask,64):
                use=np.flatnonzero((rr-r0>=r)&(rr-r0<r+m.shape[0])&(cc-c0>=c)&(cc-c0<c+m.shape[1]))
                patches.append({'group':group,'owner':0,'take':use,'flat_local':(rr[use]-r0-r)*m.shape[1]+cc[use]-c0-c,
                    'row':r0+r,'col':c0+c,'height':m.shape[0],'width':m.shape[1]})
        count=sum(len(g['raw']['s2']) for g in groups)
        if count!=self.spec['counts'][str(int(row.reference_index))]:raise ValueError('Supporto perso nel loader')
        return {'groups':groups,'patches':patches,'cell_ids':[row.cell_id],'target':[float(row.target)],
            'aoi_ids':[row.aoi_id],'folds':[int(row.cv_fold)]}


def collate_cells(cells):
    result={'groups':[],'patches':[],'cell_ids':[],'target':[],'aoi_ids':[],'folds':[]}
    for owner,cell in enumerate(cells):
        offset=len(result['groups'])
        result['groups'].extend([{**g,'owner':owner} for g in cell['groups']])
        result['patches'].extend([{**p,'owner':owner,'group':p['group']+offset} for p in cell['patches']])
        result['cell_ids'].extend(cell['cell_ids']);result['target'].extend(cell['target'])
        result['aoi_ids'].extend(cell['aoi_ids']);result['folds'].extend(cell['folds'])
    return result
