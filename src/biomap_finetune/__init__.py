# Scopo: namespace isolato del partial fine-tuning V2, senza import GPU, acquisizioni o side effect.
# Input/output: configurazioni e batch gestiti dai sottomoduli; uso: scripts/finetune_v2.py --help.
# Risorse: nessuna all'import; ripresa atomica gestita dal nuovo engine, nessuna cache legacy modificata.
__version__ = '0.1.0'
