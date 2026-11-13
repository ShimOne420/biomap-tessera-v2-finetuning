# Scopo: ingresso unico al nuovo esperimento Spatial CNN V2, senza modificare entry point legacy.
# Fasi: delega doctor/prepare/preflight/profile/train/evaluate/compare/monitor al package biomap_spatial.
# Input: configs/spatial_v2_msa.yaml e override CLI; output: soltanto la output-root privata.
# Esempio: python scripts/spatial_cnn.py doctor --source-root C:/BIOMAP/work --reference C:/BIOMAP/reference.
# Risorse: CPU per audit/indice, GPU soltanto per la testa dopo preflight; nessun download implicito.
# Ripresa: train --resume; prepare idempotente per tile; monitor --watch in terminale separato.
from biomap_spatial.cli import main

if __name__=='__main__':
    raise SystemExit(main())
