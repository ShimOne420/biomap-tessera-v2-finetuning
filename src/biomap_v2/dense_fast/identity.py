# SCOPO
#   Identità del protocollo r2-fast: nome del protocollo e impronte SHA256 del codice che determina gli
#   input e gli embedding. Ogni cache (giorni, blocchi codificati, tile.npz) registra questa identità.
#   È VOLUTAMENTE diversa dalla r1 di campaign/streaming.py: le cache r1 e r2 non sono mai scambiabili,
#   anche se il risultato numerico atteso è identico. L'equivalenza si dimostra con i test, non fingendo
#   la stessa identità.
# COME VIENE RICHIAMATO
#   from biomap_v2.dense_fast.identity import PROTOCOL, code_identity, PRODUCTION_WORK_ROOTS
# LOGICA IN ORDINE
#   1. PROTOCOL: nome nuovo e stabile.
#   2. CODE_FILES: file legacy riusati (stessa semantica) + tutti i moduli dense_fast.
#   3. code_identity(): dizionario percorso -> SHA256, calcolato all'avvio del run.
#   4. refuse_production(): impedisce di scrivere nella work-root della campagna r1 in corso.
# INPUT E OUTPUT
#   Input: i file sorgente elencati. Output: dizionari in memoria; nessuna scrittura.
# PARAMETRI
#   PRODUCTION_WORK_ROOTS: cartelle in cui questo codice non deve mai scrivere.
# RISORSE
#   Lettura e hash di una decina di file piccoli.
# RIPRESA
#   Nessuno stato.
from pathlib import Path

from biomap_v2.common import ROOT, digest

PROTOCOL = 'dense127909-stream-legacy-exact-r2-fast'

CODE_FILES = [
    'src/biomap_v2/campaign/streaming.py',
    'src/biomap_v2/campaign/extraction.py',
    'src/biomap_v2/raster_cache.py',
    'src/biomap_v2/data.py',
    'src/biomap_v2/reference.py',
    'src/biomap_v2/encoder.py',
    'src/biomap_v2/dense_fast/coords.py',
    'src/biomap_v2/dense_fast/handles.py',
    'src/biomap_v2/dense_fast/reader.py',
    'src/biomap_v2/dense_fast/acquisition.py',
    'src/biomap_v2/dense_fast/pipeline.py',
]

# La campagna r1 in produzione: mai scriverci, nemmeno per prova.
PRODUCTION_WORK_ROOTS = [Path(r'F:\biomap_dense_public\work')]


def code_identity():
    return {name: digest(ROOT / name) for name in CODE_FILES}


def refuse_production(work_root):
    work = Path(work_root).resolve()
    for production in PRODUCTION_WORK_ROOTS:
        production = production.resolve()
        if work == production or production in work.parents:
            raise ValueError(f'Work-root {work} coincide con la campagna r1 in produzione: rifiutato')
    marker = work / 'last_invocation.json'
    if marker.exists() and 'dense_fast' not in marker.read_text(encoding='utf-8', errors='replace'):
        raise ValueError(f'{work} contiene un run legacy (last_invocation.json): usare una work-root nuova')
    return work
