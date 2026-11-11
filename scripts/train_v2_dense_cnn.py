# Scopo: avvio esplicito del confronto di teste su embedding V2 densi completati.
# Esempio: python scripts/train_v2_dense_cnn.py run --source-root F:\dense --out F:\cnn --device cuda:0
# Nessuna acquisizione/encoding/fine-tuning; output separato. monitor legge solo progress.json.
import os
import sys
from pathlib import Path
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ.setdefault(key,'1')
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from biomap_dense_heads.cli import main
if __name__=='__main__':sys.exit(main())
