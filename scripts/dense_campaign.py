# SCOPO: campagna frozen densa su 127.909 celle di sviluppo + 4.991 Liguria.
# INPUT: reference portabile, sorgenti dense read-only, checkpoint Medium verificato.
# FASI: audit -> prepare esplicito -> validate -> profile -> extract -> probe -> export.
# OUTPUT: cache riprendibili, feature 128D, modelli/OOF/metriche/Excel nella work-root.
# ESEMPIO: python scripts/dense_campaign.py audit --reference F:\reference --work-root F:\dense_run
# Le regressioni sono CPU; extract/profile possono usare CUDA. Nessun fine-tuning.
import os
import sys
from pathlib import Path
for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ.setdefault(name,'1')
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from biomap_v2.campaign.cli import main
if __name__=='__main__':
    sys.exit(main())
