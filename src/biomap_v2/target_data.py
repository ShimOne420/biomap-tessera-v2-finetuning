# Scopo: diagnosticare e acquisire target senza perdere cache o cambiare celle/split.
# Fasi: controlla copertura e griglie -> esporta cause per cella -> valida MSA-area/cellarea.
# Input: configurazione, geometrie, raster ufficiali/custom; output: acquisizioni versionate e CSV/JSON diagnostici.
# Parametri: probe_msa.py diagnose-targets oppure download-targets, --config e --data-root.
# Esempio: python scripts/probe_msa.py diagnose-targets --data-root ../data.
# Risorse: CPU/rasterio; Internet solo per download-targets/prepare se copertura insufficiente.
# Ripresa: conserva raster precedenti, verifica entrambe le griglie e non imputa/esclude target.
from pathlib import Path
from datetime import datetime, timezone
import json
import shutil
import time
import uuid
import numpy as np
import pandas as pd
import rasterio
from .common import budget, digest, root, save_json


def stamp():
    return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')


def locate_target(cfg, supplied=None):
    if supplied or cfg['target']['mode']=='custom':
        path=Path(supplied or cfg['target']['raster']).expanduser().resolve()
        if not path.is_file():raise FileNotFoundError(path)
        return path
    base=root(cfg)/'targets';pointer=base/'source.json'
    if pointer.is_file():
        info=json.loads(pointer.read_text(encoding='utf-8'))
        if info.get('raster') and Path(info['raster']).is_file():return Path(info['raster'])
    name='msa_area_10sec.tif' if cfg['indicator']=='msa' else 'bii-2020_v2-1-1.tif'
    path=base/name
    if not path.is_file():raise FileNotFoundError('Target assente: eseguire download-targets prima di diagnose-targets')
    return path


def is_ratio(cfg,path):
    return cfg['indicator']=='msa' and cfg['target']['mode']=='official' and Path(path).name=='msa_area_10sec.tif'


def raster_info(path):
    with rasterio.open(path) as ds:
        nodata=ds.nodata
        return {'name':Path(path).name,'crs':str(ds.crs),'bounds':list(ds.bounds),
                'resolution':list(ds.res),'transform':list(ds.transform)[:6],'width':ds.width,'height':ds.height,
                'nodata':float(nodata) if nodata is not None and np.isfinite(nodata) else str(nodata)}


def check_grid(cfg,path,other=None):
    with rasterio.open(path) as ds:
        resolution=1/360 if cfg['indicator']=='msa' else 1/12
        if ds.crs!=rasterio.crs.CRS.from_epsg(4326) or not np.allclose(ds.res,[resolution,resolution],rtol=0,atol=1e-8):
            raise ValueError('Target: CRS/risoluzione non coerenti con il manifest')
        if ds.transform.b!=0 or ds.transform.d!=0:raise ValueError('Raster ruotato non supportato')
        if other:
            with rasterio.open(other) as second:
                if ds.crs!=second.crs or ds.shape!=second.shape or not np.allclose(tuple(ds.transform),tuple(second.transform),rtol=0,atol=1e-10):
                    raise ValueError('Numeratore e denominatore non sono co-registrati: target non calcolato')


def sample_raster(path,frame):
    with rasterio.open(path) as ds:
        r,c=rasterio.transform.rowcol(ds.transform,frame.lon.to_numpy(),frame.lat.to_numpy())
        inside=(r>=0)&(r<ds.height)&(c>=0)&(c<ds.width)
        x,y=rasterio.transform.xy(ds.transform,r,c)
        aligned=np.isclose(x,frame.lon.to_numpy(),atol=1e-8,rtol=0)&np.isclose(y,frame.lat.to_numpy(),atol=1e-8,rtol=0)
        result=np.full(len(frame),np.nan,dtype=np.float64);masked=np.ones(len(frame),dtype=bool)
        indices=np.flatnonzero(inside)
        if len(indices):
            samples=np.ma.stack(list(ds.sample(zip(frame.lon.to_numpy()[indices],frame.lat.to_numpy()[indices]),masked=True)))[:,0]
            masked[indices]=np.ma.getmaskarray(samples)
            result[indices]=np.ma.filled(samples.astype(np.float64),np.nan)
    return result,inside,masked,aligned


def covers(path,frame):
    with rasterio.open(path) as ds:
        if ds.crs!=rasterio.crs.CRS.from_epsg(4326):return False
        r,c=rasterio.transform.rowcol(ds.transform,frame.lon.to_numpy(),frame.lat.to_numpy())
        return bool(((r>=0)&(r<ds.height)&(c>=0)&(c<ds.width)).all())


def pointer(cfg,info):
    base=root(cfg)/'targets';base.mkdir(parents=True,exist_ok=True)
    existing=base/'source.json'
    if existing.exists():
        old=json.loads(existing.read_text(encoding='utf-8'))
        if old!=info:
            backup=base/'source_history'/f'{stamp()}.json';backup.parent.mkdir(exist_ok=True)
            shutil.copy2(existing,backup)
    save_json(existing,info)


def ensure_target(cfg,frame,supplied=None):
    from .data import download, unzip_named, session
    base=root(cfg)/'targets';base.mkdir(parents=True,exist_ok=True)
    try:path=locate_target(cfg,supplied)
    except FileNotFoundError:
        if supplied or cfg['target']['mode']=='custom':raise
        path=None
    if supplied or cfg['target']['mode']=='custom':
        check_grid(cfg,path)
        pointer(cfg,{'mode':'imported','raster':str(path),'sha256':digest(path),'year':cfg['year'],
                     'units':cfg['target']['units'],'provenance':cfg['target']['provenance']})
        return path
    reusable=path is not None and covers(path,frame)
    denominator=path.with_name('cellarea_km2_10sec.tif') if reusable and cfg['indicator']=='msa' else None
    if denominator is not None:
        reusable=denominator.is_file() and covers(denominator,frame)
    if reusable:
        check_grid(cfg,path,denominator)
    elif cfg['indicator']=='bii':
        destination=base/'acquisitions'/stamp();destination.mkdir(parents=True)
        archive=destination/'nhm.zip';download(cfg['target']['url'],archive,cfg)
        unzip_named(archive,['bii-2020_v2-1-1.tif','LICENSE.txt','README.txt'],destination,cfg)
        path=destination/'bii-2020_v2-1-1.tif';check_grid(cfg,path)
    else:
        destination=base/'acquisitions'/stamp();destination.mkdir(parents=True)
        api='https://globiowebapi.globio.info/api/serv.py';ident='biomap_v2_'+uuid.uuid4().hex[:16]
        bbox=[float(frame.west.min()),float(frame.south.min()),float(frame.east.max()),float(frame.north.max())]
        s=session()
        def call(action,**kw):
            r=s.get(api,params={'action':action,**kw},timeout=90);r.raise_for_status();return r.json()
        call('StartSummary',id=ident,year='2020',scenario_name='Recent',scenario_layer='msa_2020',
             region_name='BioMAP V2',region_layer='',region_id='-1',extent=','.join(map(str,bbox)))
        def wait(status):
            for _ in range(60):
                response=call('GetStatus',id=ident);current=response.get('info',{}).get('status')
                if current==status:return
                if current not in {'summary running','zip running'}:raise RuntimeError(f'GLOBIO stato inatteso: {current}')
                time.sleep(3)
            raise TimeoutError('GLOBIO non pronto: rieseguire download-targets')
        wait('summary ready');call('StartZip',id=ident,name='',surname='',email='',organization='');wait('zip ready')
        archive=destination/'globio.zip';download(api,archive,cfg,params={'action':'GetZipData','id':ident})
        unzip_named(archive,['msa_area_10sec.tif','cellarea_km2_10sec.tif'],destination,cfg)
        path=destination/'msa_area_10sec.tif';denominator=destination/'cellarea_km2_10sec.tif'
        check_grid(cfg,path,denominator)
        save_json(destination/'download.json',{'bbox':bbox,'year':2020,'archive_sha256':digest(archive)})
    if not covers(path,frame) or (denominator is not None and not covers(denominator,frame)):
        raise ValueError('Il nuovo export non copre tutte le celle: raster conservati, nessuna esclusione automatica')
    info={'source':cfg['target']['source'],'year':cfg['year'],'raster':str(path.resolve()),'sha256':digest(path),
          'units':cfg['target']['units'],'coverage_checked_cells':len(frame)}
    if cfg['indicator']=='msa':
        info.update(formula='MSA-area / cell-area',cellarea_sha256=digest(path.with_name('cellarea_km2_10sec.tif')))
    else:info['license']='CC-BY-NC-SA-4.0'
    pointer(cfg,info)
    return path


def diagnose(cfg,frame,path):
    frame=frame.reset_index(drop=True);path=Path(path)
    directory=root(cfg)/'diagnostics'/f'{stamp()}_targets';directory.mkdir(parents=True)
    pair=is_ratio(cfg,path);denominator=path.with_name('cellarea_km2_10sec.tif') if pair else None
    info={'numerator':raster_info(path),'denominator':raster_info(denominator) if denominator and denominator.exists() else None}
    if denominator and not denominator.exists():
        save_json(directory/'summary.json',{'status':'missing_denominator',**info})
        raise FileNotFoundError(f'Denominatore assente; diagnosi: {directory}')
    try:check_grid(cfg,path,denominator)
    except ValueError as error:
        save_json(directory/'summary.json',{'status':'invalid_grid','error':str(error),**info})
        raise ValueError(f'{error}; diagnosi: {directory}') from None
    num,inside,nodata,aligned=sample_raster(path,frame)
    out=frame.copy();out['numerator']=num;out['numerator_outside']=~inside;out['numerator_nodata']=nodata&inside
    out['numerator_nonfinite']=~np.isfinite(num)&inside&~nodata
    if pair:
        den,dinside,dnodata,daligned=sample_raster(denominator,frame)
        out['denominator']=den;out['denominator_outside']=~dinside;out['denominator_nodata']=dnodata&dinside
        out['denominator_zero']=np.isfinite(den)&(den==0);out['denominator_negative']=np.isfinite(den)&(den<0)
        out['denominator_nonfinite']=~np.isfinite(den)&dinside&~dnodata
        values=np.divide(num,den,out=np.full_like(num,np.nan),where=np.isfinite(den)&(den>0))
        aligned &= daligned
    else:values=num/100 if cfg['target']['units']=='percent' else num
    out['target']=values;out['misaligned']=~aligned
    out['target_out_of_range']=np.isfinite(values)&((values<0)|(values>1))
    flags=[c for c in out if c.endswith(('_outside','_nodata','_nonfinite','_zero','_negative')) or c in {'misaligned','target_out_of_range'}]
    out['valid_target']=np.isfinite(values)&(values>=0)&(values<=1)&aligned
    out['exclusion_reason']=[';'.join(name for name,yes in zip(flags,row) if yes) for row in out[flags].to_numpy()]
    out.to_csv(directory/'cells.csv',index=False)
    grouped=out.groupby(['partition','aoi_id'],sort=True).agg(original_cells=('cell_id','size'),valid_targets=('valid_target','sum')).reset_index()
    grouped['invalid_targets']=grouped.original_cells-grouped.valid_targets;grouped.to_csv(directory/'by_partition_aoi.csv',index=False)
    summary={'status':'ok' if bool(out.valid_target.all()) else 'invalid_targets','n_cells':len(out),
        'valid_targets':int(out.valid_target.sum()),'invalid_targets':int((~out.valid_target).sum()),
        'reasons':{c:int(out[c].sum()) for c in flags},'raster_info':info,'raster_sha256':digest(path),
        'denominator_sha256':digest(denominator) if denominator else None,
        'counts_by_partition':out.groupby('partition').valid_target.agg(['size','sum']).reset_index().to_dict('records'),
        'bias_assessment':'by_partition_aoi.csv identifies geographic coverage bias; missing targets have no observable target distribution; no exclusions applied'}
    save_json(directory/'summary.json',summary)
    print(f'Target: {summary["valid_targets"]}/{len(out)} validi; diagnosi {directory}',flush=True)
    return out,directory


def validated_targets(cfg,frame,path):
    diagnosis,directory=diagnose(cfg,frame,path)
    if not diagnosis.valid_target.all():
        raise ValueError(f'Target non validi: nessuna cella eliminata/imputata. Leggere {directory / "summary.json"}')
    out=frame.copy();out['target']=diagnosis.target.to_numpy()
    return out
