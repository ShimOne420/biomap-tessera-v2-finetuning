# Scopo: configurazione esplicita del fine-tuning; undecided/null sono validi solo per preparazione.
# Fasi: YAML/extends -> override CLI -> validazione -> gate acquisizione/training -> manifest risolto.
# Input: YAML e percorsi passati dall'utente; output: dict senza default personali o download impliciti.
# Parametri: aggregazione, sampling approvato, LR/batch/epoche e ultimi 1-3 blocchi. Esempio: load(path).
# Risorse: sola CPU/metadati; ripresa: identita immutabile di protocollo, supporto e configurazione.
from copy import deepcopy
from pathlib import Path
import yaml


def merge(base, override):
    result=deepcopy(base)
    for key,value in override.items():
        result[key]=merge(result[key],value) if isinstance(value,dict) and isinstance(result.get(key),dict) else deepcopy(value)
    return result


def load(path, overrides=()):
    path=Path(path)
    cfg=yaml.safe_load(path.read_text(encoding='utf-8'))
    parent=cfg.pop('extends',None)
    if parent:cfg=merge(load(path.parent/parent),cfg)
    for expression in overrides:
        key,sep,value=expression.partition('=')
        if not sep:raise ValueError('Override richiede sezione.chiave=valore')
        target=cfg;parts=key.split('.')
        for part in parts[:-1]:
            if part not in target or not isinstance(target[part],dict):raise ValueError('Sezione sconosciuta: '+key)
            target=target[part]
        if parts[-1] not in target:raise ValueError('Parametro sconosciuto: '+key)
        target[parts[-1]]=yaml.safe_load(value)
    validate(cfg)
    return cfg


def validate(cfg):
    if cfg['experiment']['target']!='globio4_overall_msa_2020' or cfg['training']['loss']!='mse' or cfg['training']['optimizer']!='adamw':raise ValueError('Target/loss/optimizer congelati dal nuovo protocollo')
    if cfg['validation']['protocol']!='original_spatial_3fold' or cfg['validation']['selection_metric']!='macro_aoi_rmse':raise ValueError('Validazione geografica obbligatoria')
    if cfg['validation']['rtol']!=2e-5 or cfg['validation']['atol']!=2e-5:raise ValueError('Non allargare tolleranze storiche per aggirare una discrepanza')
    if cfg['experiment']['profile'] not in {'benchmark','smoke','synthetic'}:raise ValueError('Profilo non valido')
    if cfg['encoder']['name']!='geotessera_v2_medium' or cfg['finetuning']['method']!='partial':raise ValueError('Supportato soltanto V2 Medium partial')
    if cfg['finetuning']['last_n_blocks'] not in {1,2,3}:raise ValueError('last_n_blocks deve essere 1,2,3')
    for name in ['train_output_projection','train_normalization']:
        if cfg['finetuning'][name] not in {'auto',True,False}:raise ValueError('Policy auto/bool richiesta')
    if cfg['aggregation']['method'] not in {'undecided','dense_mean','spatial_cnn'}:raise ValueError('Aggregazione non supportata')
    if cfg['sampling']['method'] not in {'undecided','full_support','uniform_pixels','center_windows'}:raise ValueError('Sampling non supportato')
    if cfg['cache']['mode'] not in {'auto','online','cached'}:raise ValueError('Cache mode non valida')
    if cfg['training']['encoder_mode'] not in {'partial','frozen'}:raise ValueError('encoder_mode non valido')
    if cfg['training']['precision'] not in {'fp32','float16','bfloat16'}:raise ValueError('Precisione non valida')
    if cfg['training']['normalization'] not in {'train_initial_pixel_standard','none'}:raise ValueError('Normalizzazione non valida')
    for name in ['learning_rate_encoder','learning_rate_head','batch_size','epochs']:
        value=cfg['training'][name]
        if value is not None and value<=0:raise ValueError('Parametro training positivo: '+name)
    if cfg['training']['num_workers'] not in range(5):raise ValueError('num_workers 0..4')
    if cfg['training']['pixel_batch']<1 or cfg['acquisition']['block_pixels']<1 or cfg['storage']['reserve_gib']<0:
        raise ValueError('Risorse non valide')
    if cfg['acquisition']['io_workers'] not in range(1,5):raise ValueError('Thread I/O espliciti 1..4')
    if cfg['acquisition']['catalog_attempts'] not in range(1,7):raise ValueError('Tentativi catalogo espliciti 1..6')
    for name in ['batch_size','epochs']:
        if cfg['training'][name] is not None and (not isinstance(cfg['training'][name],int) or isinstance(cfg['training'][name],bool)):raise ValueError('Valore intero richiesto: '+name)
    if cfg['sampling']['method']=='uniform_pixels' and (cfg['sampling']['pixels_per_cell'] is None or cfg['sampling']['pixels_per_cell']<1):
        raise ValueError('uniform_pixels richiede pixels_per_cell esplicito')
    if cfg['sampling']['method']=='center_windows' and (cfg['sampling']['window_side'] is None or cfg['sampling']['window_side']<1):
        raise ValueError('center_windows richiede window_side esplicito')


def blockers(cfg, stage='train'):
    result=[]
    for name in ['reference','raw_root','output_root']:
        if not cfg['paths'].get(name):result.append('Percorso mancante: '+name)
    if cfg['sampling']['method']=='undecided':result.append('Sampling ancora undecided')
    if not cfg['sampling']['approved']:result.append('Sampling non approvato esplicitamente')
    if stage=='train':
        if cfg['aggregation']['method']=='undecided':result.append('aggregation=undecided: training bloccato')
        for name in ['learning_rate_encoder','learning_rate_head','batch_size','epochs']:
            if cfg['training'][name] is None:result.append('Parametro training ancora null: '+name)
        if not cfg['encoder']['checkpoint']:result.append('Checkpoint locale mancante')
        if cfg['cache']['mode']=='cached':result.append('Cache prefisso: prototipo verificato, corpus/trainer cached non abilitato; usare online')
    return result


def require_ready(cfg,stage='train'):
    errors=blockers(cfg,stage)
    if errors:raise ValueError('; '.join(errors))
    if cfg['training']['precision']!='fp32' and not cfg['execution']['device'].startswith('cuda'):
        raise ValueError('Precisione ridotta richiede CUDA esplicita')
    return {'cache_mode':'online','cache_resolution':'auto risolta esplicitamente online; nessuna generazione massiva',
            'aggregation':cfg['aggregation']['method']}
