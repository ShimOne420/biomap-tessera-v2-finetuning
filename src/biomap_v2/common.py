# Scopo: configurazione, provenienza, controlli risorse e metriche comuni alle tre CLI.
# Fasi: carica YAML/override, valida split, crea run e salva risultati verificabili.
# Input: configurazione, CSV celle, predizioni; output: JSON/CSV/PNG e report Markdown.
# Parametri: --config, --data-root, --set sezione.chiave=valore; vedere README per esempi.
# Risorse: CPU; il doctor verifica separatamente CUDA con forward e backward reali.
# Ripresa: scritture JSON atomiche e checksum; non cancella dati originali o run esistenti.
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import os
import shutil
import subprocess
import time
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
TESSERA_COMMIT = 'e368539f7c4711e5f16fdd09801573830e34baf9'
WEIGHTS_REVISION = '8c80b926f67cb0b44c2cd41d0226e61741503b9a'
WEIGHTS_SHA256 = '3823be7db9d9cfc93f3c2a47c7699be82821ab4e1117d4d2befdb746941ee96e'


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False, default=str), encoding='utf-8')
    os.replace(tmp, path)


def config(indicator, args):
    path = Path(args.config) if args.config else ROOT / 'configs' / f'{indicator}.yaml'
    cfg = yaml.safe_load(path.read_text(encoding='utf-8'))
    if cfg['indicator'] != indicator:
        raise ValueError('Indicatore CLI e configurazione discordanti')
    for expression in args.set or []:
        key, sep, value = expression.partition('=')
        if not sep:
            raise ValueError('--set richiede sezione.chiave=valore')
        target = cfg
        parts = key.split('.')
        for part in parts[:-1]:
            if part not in target or not isinstance(target[part], dict):
                raise ValueError(f'Parametro sconosciuto: {key}')
            target = target[part]
        if parts[-1] not in target:
            raise ValueError(f'Parametro sconosciuto: {key}')
        target[parts[-1]] = yaml.safe_load(value)
    cfg['data_root'] = str(Path(args.data_root or cfg['data_root']).expanduser().resolve())
    cfg['cells'] = str((ROOT / cfg['cells']).resolve()) if not Path(cfg['cells']).is_absolute() else cfg['cells']
    if cfg['sampling']['pixels_per_cell'] < 1:
        raise ValueError('pixels_per_cell deve essere positivo')
    n = cfg['sampling']['pixels_per_cell']
    if int(np.sqrt(n)) ** 2 != n:
        raise ValueError('pixels_per_cell deve essere un quadrato, per esempio 25, 49 o 100')
    if cfg['training']['epochs'] < 1 or cfg['training']['patience'] < 1:
        raise ValueError('epochs e patience devono essere positivi')
    if cfg['training']['batch_cells'] < 1 or cfg['training']['effective_batch_cells'] < 1 or cfg['model']['pixel_batch'] < 1:
        raise ValueError('Batch di celle/pixel devono essere positivi')
    if not cfg['training']['seeds'] or len(set(cfg['training']['seeds'])) != len(cfg['training']['seeds']):
        raise ValueError('seeds deve essere una lista non vuota senza duplicati')
    if not cfg['probing']['models'] or set(cfg['probing']['models'])-{'Ridge','HistGB','XGBoost'}:
        raise ValueError('probing.models deve contenere regressori supportati')
    if cfg['year'] != 2020 and cfg['target']['mode'] == 'official':
        raise ValueError('I target ufficiali configurati sono del 2020; per altri anni usare un raster custom documentato')
    return cfg


def root(cfg):
    return Path(cfg['data_root']) / cfg['indicator']


def budget(cfg, additional=0):
    base = Path(cfg['data_root'])
    base.mkdir(parents=True, exist_ok=True)
    used = sum(p.stat().st_size for p in base.rglob('*') if p.is_file())
    reserve = cfg['storage']['reserve_gib'] * 1024 ** 3
    cap = cfg['storage']['max_gib'] * 1024 ** 3
    if used + additional > cap or shutil.disk_usage(base).free < reserve + additional:
        raise RuntimeError('Budget disco insufficiente: cambiare --data-root o storage.max_gib; nessun originale cancellato')


def cells(cfg):
    frame = pd.read_csv(cfg['cells'], dtype={'cell_id': str})
    required = {'cell_id', 'lon', 'lat', 'partition', 'aoi_id', 'west', 'south', 'east', 'north'}
    if not required.issubset(frame):
        raise ValueError(f'CSV celle: mancano {sorted(required-set(frame))}')
    if frame.cell_id.duplicated().any() or not set(frame.partition).issubset({'train', 'validation', 'test'}):
        raise ValueError('Celle duplicate o partizioni non valide')
    if not np.isfinite(frame[['lon','lat','west','south','east','north']]).all().all():
        raise ValueError('Coordinate non finite')
    if ((frame.west >= frame.east) | (frame.south >= frame.north)).any():
        raise ValueError('Geometrie non valide')
    # Protect the sealed Umbria box, including overlapping target cells.
    if ((frame.west < 12.7) & (frame.east > 12.4) & (frame.south < 43.2) & (frame.north > 43.0)).any():
        raise ValueError('Il manifest interseca Umbria nord sigillata')
    return frame


def make_run(cfg, kind):
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    out = root(cfg) / 'runs' / f'{stamp}_{kind}'
    out.mkdir(parents=True)
    save_json(out / 'config.json', cfg)
    save_json(out / 'provenance.json', {'tessera_commit': TESSERA_COMMIT,
        'weights_revision': WEIGHTS_REVISION, 'cells_sha256': digest(cfg['cells']),
        'created_utc': stamp, 'role': 'exploratory_existing_spatial_holdouts'})
    return out


def metrics(y, pred):
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    y, pred = np.asarray(y), np.asarray(pred)
    if len(y) == 0 or not np.isfinite(y).all() or not np.isfinite(pred).all():
        raise ValueError('Metriche: campione vuoto o valori non finiti')
    nz = y != 0
    return {'n': len(y), 'mae': float(mean_absolute_error(y, pred)),
        'rmse': float(np.sqrt(mean_squared_error(y, pred))),
        'mape_pct': float(np.mean(np.abs((pred[nz]-y[nz])/y[nz]))*100) if nz.any() else None,
        'mape_n': int(nz.sum()), 'mape_excluded_zero': int((~nz).sum()),
        'bias': float(np.mean(pred-y)), 'r2': float(r2_score(y,pred)) if len(y)>1 and np.std(y)>0 else None,
        'pearson': float(np.corrcoef(y,pred)[0,1]) if len(y)>1 and np.std(y)>0 and np.std(pred)>0 else None}


def geographic_rmse(frame, pred, block_m=40000, grouping="spatial_block"):
    if grouping == "aoi":
        errors = (np.asarray(pred)-frame.target.to_numpy())**2
        return float(np.sqrt(pd.Series(errors).groupby(frame.aoi_id.reset_index(drop=True)).mean()).mean())
    from pyproj import Transformer
    x,y = Transformer.from_crs(4326,3035,always_xy=True).transform(frame.lon.to_numpy(),frame.lat.to_numpy())
    groups = pd.Series([f'{int(a//block_m)}:{int(b//block_m)}' for a,b in zip(x,y)])
    errors = (np.asarray(pred)-frame.target.to_numpy())**2
    return float(np.sqrt(pd.Series(errors).groupby(groups).mean()).mean())


def report(out, predictions):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows = []
    for (model, part), f in predictions.groupby(['model','partition']):
        rows.append({'model': model, 'partition': part, **metrics(f.target,f.prediction)})
        fig,ax = plt.subplots(figsize=(5,5))
        ax.scatter(f.target,f.prediction,s=8,alpha=.5); ax.plot([0,1],[0,1],'k--')
        ax.set(xlabel='Target modellato',ylabel='Predizione',title=f'{model} / {part}, n={len(f)}')
        fig.tight_layout();fig.savefig(out/f'{model}_{part}.png',dpi=140);plt.close(fig)
    frame = pd.DataFrame(rows)
    dummy = frame[frame.model=='DummyMean'].set_index('partition').rmse.to_dict()
    frame['rmse_gain_vs_dummy_pct'] = [100*(1-r.rmse/dummy[r.partition]) if r.partition in dummy and dummy[r.partition]>0 else np.nan for r in frame.itertuples()]
    frame.to_csv(out/'metrics.csv',index=False)
    predictions.to_csv(out/'predictions.csv',index=False)
    lines=['# Risultati '+out.name,'','Target modellati: GLOBIO MSA oppure NHM BII. Nessuna validazione di biodiversità in campo.',
        '','Test storici già osservati: benchmark esplorativo. Selezione del modello solo sulla validation geografica.',
        '','MAPE esclude target zero; denominatori in metrics.csv. Supporto: posizioni stratificate nella cella nativa.',
        '','| Modello | Split | n | MAE | RMSE | MAPE % |','|---|---|---:|---:|---:|---:|']
    for r in frame.itertuples():
        lines.append(f'| {r.model} | {r.partition} | {r.n} | {r.mae:.5f} | {r.rmse:.5f} | {r.mape_pct if pd.notna(r.mape_pct) else "n.a."} |')
    cfg=json.loads((out/'config.json').read_text()) if (out/'config.json').exists() else {}
    reference=ROOT/'configs/historical_metrics.csv'
    if reference.exists() and cfg.get('indicator') in {'msa','bii'}:
        historical=pd.read_csv(reference)
        historical=historical[historical.indicator==cfg['indicator']]
        historical.to_csv(out/'historical_reference.csv',index=False)
        lines += ['', '## Riferimento V1 storico', '',
            'Metriche precedenti in historical_reference.csv. Supporto/celle eleggibili diversi: confronto descrittivo, non differenza appaiata e non prova causale V1/V2.']
    (out/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return frame


def doctor(cfg, require_cuda=True):
    import platform
    import torch
    from importlib.metadata import version
    info={'platform':platform.platform(),'python':platform.python_version(),'processor':platform.processor(),
          'torch':torch.__version__,'cuda_runtime':torch.version.cuda,'cuda_available':torch.cuda.is_available(),
          'free_gib':shutil.disk_usage(Path(cfg['data_root']).anchor).free/1024**3,
          'packages':{k:version(k) for k in ['numpy','pandas','rasterio','scikit-learn']}}
    if require_cuda and not torch.cuda.is_available():
        raise RuntimeError('CUDA non disponibile. Installare driver NVIDIA e wheel PyTorch cu128; --device cpu solo per test')
    device = torch.device(cfg['model']['device'])
    if device.type=='cuda':
        i=device.index or 0
        info.update(gpu=torch.cuda.get_device_name(i),vram_gib=torch.cuda.get_device_properties(i).total_memory/1024**3,
                    capability=torch.cuda.get_device_capability(i),architectures=torch.cuda.get_arch_list())
    x=torch.randn(32,32,device=device,requires_grad=True)
    (x@x.T).square().mean().backward()
    if not torch.isfinite(x.grad).all():raise RuntimeError('Backward GPU non finito')
    try:
        info['nvidia_smi']=subprocess.run(['nvidia-smi','--query-gpu=name,driver_version,memory.total','--format=csv'],capture_output=True,text=True,timeout=10).stdout.strip()
    except (OSError, subprocess.TimeoutExpired):info['nvidia_smi']='non disponibile'
    if platform.system()=='Windows':
        query='Get-CimInstance Win32_ComputerSystem | Select-Object TotalPhysicalMemory | ConvertTo-Json'
        result=subprocess.run(['powershell','-NoProfile','-Command',query],capture_output=True,text=True,timeout=15)
        if result.returncode==0:info['ram_bytes']=json.loads(result.stdout)['TotalPhysicalMemory']
    save_json(root(cfg)/'doctor.json',info)
    print(json.dumps(info,indent=2))
    budget(cfg)
    return info
