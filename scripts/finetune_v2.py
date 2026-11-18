# Scopo: entry point nuovo per fine-tuning parziale GeoTESSERA V2 Medium e controllo frozen appaiato.
# Fasi: config/decisioni -> doctor -> acquire -> verifica input -> train/CV/refit -> evaluate -> report.
# Input: YAML e percorsi espliciti; output: nuove raw/output root, nessuna scrittura nelle campagne r1/r2.
# Parametri: vedere --help e help per ciascun sottocomando; esempio: python scripts/finetune_v2.py doctor.
# Risorse: Python 3.12/Torch; CPU per test, CUDA solo se scelta; doctor non scarica dati/pesi.
# Ripresa: --resume e checkpoint per epoca/blocco; monitor --progress legge JSON e chiude subito il file.
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from biomap_finetune.cli import main

if __name__=='__main__':main()
