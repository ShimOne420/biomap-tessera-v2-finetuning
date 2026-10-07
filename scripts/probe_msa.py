# Scopo: preparazione, probing e valutazione MSA con TESSERA V2 frozen.
# Fasi: doctor/download-weights -> prepare -> run -> evaluate/export-embeddings.
# Input: configs/msa.yaml, CSV celle, target e serie Sentinel.
# Output: cache verificate, configurazioni effettive, checkpoint, predizioni e report nella data-root.
# Parametri: --data-root, --config, --set sezione.chiave=valore, --checkpoint, --smoke; vedere --help.
# Esempio: python scripts/probe_msa.py run --data-root E:\BioMAP.
# Risorse: Python 3.12, Internet per acquisizione e NVIDIA CUDA per campagna; CPU solo per test.
# Ripresa: prepare riusa cache con checksum; feature frozen riusate soltanto se dati e checkpoint coincidono.
import sys
from pathlib import Path

# Resolve package from the checkout even when editable .pth files are not processed.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from biomap_v2.cli import main

if __name__ == '__main__':
    main(indicator='msa', finetune=False)
