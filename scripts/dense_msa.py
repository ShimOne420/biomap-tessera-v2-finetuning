# Scopo: entrypoint del benchmark denso V1/V2 per GLOBIO MSA.
# Fasi: check-v1, check-v2, check-encoder, run, all (vedere --help).
# Input: percorsi via flag o variabili BIOMAP_*.
# Output: outputs/dense_benchmark e reports/REPORT_V1_V2_DENSE_MSA.md.
# Esempio: python scripts/dense_msa.py all --v1-tiles ... --v2-tiles ... --target ...
# Risorse: CPU; GPU/MPS solo se mancano embedding V2 da completare.
# Ripresa: embedding V2 con memmap e progresso atomico per tile.
import os
import sys
from pathlib import Path

# Evita thread BLAS/OpenMP che possono bloccare HistGB in ambienti sandbox.
for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_name, '1')

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from biomap_v2.dense.cli import main

if __name__ == '__main__':
    sys.exit(main())
