# Scopo: preparazione, probing e valutazione BII con TESSERA V2 frozen.
# Fasi: doctor/download-weights -> prepare -> run -> evaluate/export-embeddings.
# Input: configs/bii.yaml, CSV celle, target e serie Sentinel.
# Output: cache verificate, configurazioni effettive, checkpoint, predizioni e report nella data-root.
# Parametri: --data-root, --config, --set sezione.chiave=valore, --checkpoint, --smoke; vedere --help.
# Esempio: python scripts/probe_bii.py run --data-root E:\BioMAP.
# Risorse: Python 3.12, Internet per acquisizione e NVIDIA CUDA per campagna; CPU solo per test.
# Ripresa: prepare riusa cache con checksum; feature frozen riusate soltanto se dati e checkpoint coincidono.
from biomap_v2.cli import main

if __name__ == '__main__':
    main(indicator='bii', finetune=False)
