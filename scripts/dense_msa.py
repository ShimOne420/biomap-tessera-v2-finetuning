# Scopo: entrypoint del benchmark denso V1/V2 per GLOBIO MSA.
# Fasi: pack -> fetch-bundle -> verify-bundle -> all; comandi raw ancora disponibili.
# Input: percorsi via flag o variabili BIOMAP_*.
# Output: nuova cartella in outputs/dense_benchmark, report del run, CSV e Excel.
# Esempio: python scripts/dense_msa.py all --bundle F:\dense_liguria --checkpoint ... --out ...
# Risorse: regressori CPU; GPU facoltativa per la sola verifica del campione encoder.
# Ripresa: dati/feature dense riusati, export rigenera Excel senza ripetere training.
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
