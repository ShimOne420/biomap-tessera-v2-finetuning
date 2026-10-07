# Scopo: fine-tuning progressivo e valutazione MSA/BII aggiornando i pesi TESSERA V2.
# Fasi: doctor/download-weights -> prepare -> run -> evaluate/export-embeddings.
# Input: configs/msa oppure bii.yaml, CSV celle, target e serie Sentinel.
# Output: cache verificate, configurazioni effettive, checkpoint, predizioni e report nella data-root.
# Parametri: --data-root, --config, --set sezione.chiave=valore, --checkpoint, --smoke; vedere --help.
# Esempio: python scripts/finetune.py run --indicator bii --data-root E:\BioMAP.
# Risorse: Python 3.12, Internet per acquisizione e NVIDIA CUDA per campagna; CPU solo per test.
# Ripresa: prepare riusa cache con checksum; --resume RUN/last.pt riprende una fase.
from biomap_v2.cli import main

if __name__ == '__main__':
    main(indicator=None, finetune=True)
