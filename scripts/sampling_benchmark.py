# Scopo: entry point unico del benchmark di campionamento su embedding frozen V2 r2.
# Fasi: doctor/dry-run -> prepare/cache -> smoke/screen -> nested approvato -> report/monitor.
# Input: YAML e percorsi CLI, nessun default personale; output: sole root nuove, mai sorgenti/indice.
# Parametri/esempio: python scripts/sampling_benchmark.py doctor --help; usare --resume per ripresa.
# Risorse: CPU o GPU della sola CNN; nessun Sentinel, checkpoint encoder o fine-tuning viene richiesto.
# Ripresa: shard/fit/epoca atomici con SHA e RNG; budget non completato non e un test passato.
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from biomap_sampling.cli import main

if __name__=='__main__':main()
