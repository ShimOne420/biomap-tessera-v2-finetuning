# SCOPO
#   Oracolo per i test: esegue il codice ORIGINALE dell'Esperimento B (experiments/v1_spatial_cnn_msa) in un
#   processo separato, così i moduli omonimi (runtime, dataset, train...) non si mescolano con quelli V2.
# COME VIENE RICHIAMATO
#   python b_oracle.py <cfg.json> <frame.csv> <out_dir>   (da test_v2_spatial.py, mai a mano)
# LOGICA IN ORDINE
#   build dell'indice B -> train(phase=all) -> predizioni Liguria -> esporta celle, patch e predizioni in CSV.
# INPUT E OUTPUT
#   Input: configurazione B già risolta e membership sintetica. Output: CSV nella cartella indicata.
# PARAMETRI
#   Solo i tre percorsi posizionali.
# RISORSE
#   CPU, un thread Torch.
# RIPRESA
#   Nessuna: ogni chiamata usa una work-root nuova.
import json
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

B = Path(__file__).resolve().parents[2] / 'v1_spatial_cnn_msa'
sys.path.insert(0, str(B))
import torch  # noqa: E402

torch.set_num_threads(1)
from runtime import Progress  # noqa: E402
from patch_builder import build  # noqa: E402
from train import train, predict_external, load_run  # noqa: E402


def main():
    cfg = json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
    frame = pd.read_csv(sys.argv[2], dtype={'cell_id': str})
    out = Path(sys.argv[3])
    out.mkdir(parents=True, exist_ok=True)
    p = Progress(cfg, 'oracle')
    try:
        path = build(cfg, frame, p)
        run = Path(cfg['work_root']) / 'runs' / 'oracle'
        train(cfg, run, 'all', False, p)
        predict_external(cfg, run, load_run(cfg, run))
    finally:
        p.close()
    with sqlite3.connect(path) as db:
        cells = pd.read_sql('SELECT cell_id,n_total,n_valid,mean,variance,eligible FROM cells ORDER BY cell_id', db)
        patches = pd.read_sql('SELECT cell_id,tile_id,row0,col0,height,width,mask,n_valid FROM patches ORDER BY patch_id', db)
    cells['mean'] = [np.frombuffer(x, dtype=np.float32).tobytes().hex() for x in cells['mean']]
    cells['variance'] = [np.frombuffer(x, dtype=np.float32).tobytes().hex() for x in cells['variance']]
    patches['mask'] = [bytes(x).hex() for x in patches['mask']]
    cells.to_csv(out / 'cells.csv', index=False)
    patches.to_csv(out / 'patches.csv', index=False)
    pd.read_csv(run / 'oof_predictions.csv', dtype={'cell_id': str}).to_csv(out / 'oof.csv', index=False)
    pd.read_csv(run / 'liguria_predictions.csv', dtype={'cell_id': str}).to_csv(out / 'liguria.csv', index=False)


if __name__ == '__main__':
    main()
