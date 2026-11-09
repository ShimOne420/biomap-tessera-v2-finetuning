# SCOPO: orchestrare E2 sui tre fold geografici e il refit completo, senza usare Liguria.
# FASI: warmup indipendente per fold -> CNN frozen -> OOF -> epoche mediane -> refit.
# INPUT: indice/cache verificati; OUTPUT: run separata, checkpoint, OOF e selection.json.
# La selezione usa RMSE macro-AOI; il refit non usa validation o target liguri.
# Uso: python -m biomap_ml.train ... --run-id e2_msa_seed42 --phase all
import json
from pathlib import Path
import shutil
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import torch
from biomap_v2.common import digest, geographic_rmse, save_json, TESSERA_COMMIT, WEIGHTS_REVISION
from .cache import PointDataset
from .engine import fit, predict
from .settings import identity, save_csv, run_directory
from .source import indexed
from .spatial import SpatialCNN


# L'identità esclude solo i percorsi locali; iperparametri, dati e device restano vincolati.
def run_identity(cfg, audit):
    return identity({'dataset_id': audit['fingerprint']['dataset_id'], 'training': cfg['training'],
                     'runtime': cfg['runtime'], 'protocol': cfg['protocol']})


# Predizioni appaiate alla riga originale, con Dummy stimato esclusivamente sul training pertinente.
def prediction_table(frame, values, dummy, fold, role):
    out = frame.copy()
    out['prediction'] = values
    out['dummy_prediction'] = float(dummy)
    out['evaluation_fold'] = fold
    out['evaluation_role'] = role
    out['model'] = 'E2_FrozenCNN'
    out['error'] = out.prediction-out.target
    out['absolute_error'] = out.error.abs()
    out['absolute_percentage_error_pct'] = np.where(out.target != 0, out.absolute_error/out.target.abs()*100, np.nan)
    out['dummy_error'] = out.dummy_prediction-out.target
    return out


# Carica esclusivamente la testa del checkpoint e verifica il riferimento all'encoder/dataset.
def predict_checkpoint(cfg, frame, audit, checkpoint):
    payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if payload['dataset_id'] != audit['fingerprint']['dataset_id'] or payload['encoder_sha256'] != audit['fingerprint']['checkpoint_sha256']:
        raise ValueError('Checkpoint non compatibile con encoder o dataset')
    head = SpatialCNN().to(cfg['runtime']['device'])
    head.load_state_dict(payload['head'])
    dataset = PointDataset(cfg, frame, audit)
    try:
        return predict(head, dataset, cfg)
    finally:
        dataset.close()


# La cartella deterministica permette resume, ma non sovrascrive una campagna differente.
def initialize_run(cfg, audit, run_id, resume):
    out = run_directory(cfg, run_id)
    marker = out/'run.json'
    signature = run_identity(cfg, audit)
    if marker.exists():
        saved = json.loads(marker.read_text(encoding='utf-8'))
        if saved['identity'] != signature or not resume:
            raise ValueError('Run esistente: usare --resume con stessa configurazione e stessi dati')
    else:
        out.mkdir(parents=True, exist_ok=True)
        save_json(out/'config.json', cfg)
        save_json(out/'audit.json', audit)
        save_json(marker, {'run_id': run_id, 'identity': signature, 'experiment': 'E2',
                  'created_utc': datetime.now(timezone.utc).isoformat(), 'tessera_commit': TESSERA_COMMIT,
                  'weights_revision': WEIGHTS_REVISION, 'encoder_updated': False, 'status': 'initialized'})
    return out


# I dati di test sono filtrati prima di costruire qualsiasi dataset supervisionato.
def train_campaign(cfg, run_id, phase='all', resume=False):
    frame, audit = indexed(cfg)
    out = initialize_run(cfg, audit, run_id, resume)
    development = frame[frame.partition == 'train'].reset_index(drop=True)
    empty = development.iloc[:0]
    if phase in {'cv', 'all'}:
        oof, best_epochs = [], []
        for fold in range(3):
            training = development[development.cv_fold != fold].reset_index(drop=True)
            validation = development[development.cv_fold == fold].reset_index(drop=True)
            if set(training.aoi_id) & set(validation.aoi_id):
                raise ValueError('Leakage: AOI comuni a training e validation')
            directory = out/f'fold_{fold}'
            warmup = fit(cfg, training, validation, audit, directory/'warmup', 'warmup',
                         cfg['training']['warmup_epochs'], resume=resume)
            best = fit(cfg, training, validation, audit, directory/'E2', 'E2',
                       cfg['training']['epochs'], initial=warmup, resume=resume)
            payload = torch.load(best, map_location='cpu', weights_only=False)
            best_epochs.append(payload['best_epoch'])
            values = predict_checkpoint(cfg, validation, audit, best)
            table = prediction_table(validation, values, training.target.mean(), fold, 'out_of_fold_development')
            save_csv(directory/'validation_predictions.csv', table)
            oof.append(table)
        oof = pd.concat(oof, ignore_index=True)
        if oof.cell_id.duplicated().any() or set(oof.cell_id) != set(development.cell_id):
            raise ValueError('OOF incompleto o celle valutate più volte')
        save_csv(out/'oof_predictions.csv', oof)
        save_json(out/'selection.json', {'criterion': 'out_of_fold_macro_aoi_rmse',
                  'macro_aoi_rmse': geographic_rmse(oof, oof.prediction, grouping='aoi'),
                  'dummy_macro_aoi_rmse': geographic_rmse(oof, oof.dummy_prediction, grouping='aoi'),
                  'best_post_warmup_epochs': best_epochs, 'refit_epochs': int(np.median(best_epochs)),
                  'n_development': len(development), 'liguria_used_for_selection': False})
    if phase in {'refit', 'all'}:
        selection = json.loads((out/'selection.json').read_text(encoding='utf-8'))
        warmup = fit(cfg, development, empty, audit, out/'refit_warmup', 'warmup',
                     cfg['training']['warmup_epochs'], resume=resume)
        final = fit(cfg, development, empty, audit, out/'refit_E2', 'refit',
                    selection['refit_epochs'], initial=warmup, resume=resume)
        final_info = out/'final.json'
        if final_info.exists():
            previous = json.loads(final_info.read_text(encoding='utf-8'))
            if digest(final) != previous['checkpoint_sha256'] or digest(out/'final.pt') != previous['checkpoint_sha256']:
                raise ValueError('Refit diverso da un finale già registrato: usare una nuova run')
        else:
            shutil.copy2(final, out/'final.pt')
            save_json(final_info, {'checkpoint_sha256': digest(out/'final.pt'),
                  'dummy_mean': float(development.target.mean()), 'refit_epochs': selection['refit_epochs'],
                  'n_training': len(development), 'encoder_updated': False,
                  'test_evaluated': False, 'selection_identity': identity(selection)})
    marker = json.loads((out/'run.json').read_text(encoding='utf-8'))
    marker['status'] = 'refit_complete' if (out/'final.pt').is_file() else 'cv_complete'
    save_json(out/'run.json', marker)
    return out


# Il benchmark ligure è un comando separato e riutilizza le predizioni esistenti al rilancio.
def evaluate_campaign(cfg, run_id):
    frame, audit = indexed(cfg)
    out = run_directory(cfg, run_id)
    marker = json.loads((out/'run.json').read_text(encoding='utf-8'))
    if marker['identity'] != run_identity(cfg, audit):
        raise ValueError('Configurazione/dati diversi dalla campagna')
    final = json.loads((out/'final.json').read_text(encoding='utf-8'))
    if digest(out/'final.pt') != final['checkpoint_sha256']:
        raise ValueError('Checkpoint finale modificato')
    test = frame[frame.partition == 'test'].reset_index(drop=True)
    destination = out/'test_predictions.csv'
    if destination.exists():
        if not final['test_evaluated'] or digest(destination) != final['test_predictions_sha256']:
            raise ValueError('Predizioni test esistenti non verificate: nessun nuovo test automatico')
    else:
        values = predict_checkpoint(cfg, test, audit, out/'final.pt')
        table = prediction_table(test, values, final['dummy_mean'], -1, 'liguria_previously_observed_benchmark')
        save_csv(destination, table)
        final.update(test_evaluated=True, test_predictions_sha256=digest(destination),
                     evaluated_utc=datetime.now(timezone.utc).isoformat())
        save_json(out/'final.json', final)
    from .products import export_results
    export_results(cfg, out)
    return out
