# SCOPO
#   Verificare che il nuovo percorso PC (reference pubblico + acquisizione streaming) produca per la
#   Liguria gli stessi input Sentinel e gli stessi embedding V2 del benchmark denso storico del Mac.
#   Usa encoder_sample.npz del pacchetto ligure: 32 pixel della tile grid_9.35_44.25 con le serie
#   S1/S2 grezze storiche, i giorni e gli embedding FP32 salvati allora.
#
# COME SI RICHIAMA
#   python scripts\dense_campaign.py verify-legacy-sample --reference <ref> --work-root <work>
#     --checkpoint <student_medium.pt> --liguria-bundle <pacchetto> --device cuda:0
#   (anche come funzione con un reference che contiene solo la Liguria, per le prove preliminari).
#
# CONTROLLI, IN ORDINE
#   1. La tile della nuova geometria ha lo stesso numero di pixel della geometria storica.
#   2. I 32 pixel, presi alle stesse posizioni nell'ordine riga-maggiore, vengono riacquisiti con
#      streaming.acquire_points (stesso codice del run completo) in work/tmp/legacy_check.
#   3. Confronto esatto di giorni (DOY), bande S2, maschera S2, S1 asc/desc con i valori storici.
#   4. Codifica con il Medium locale sia degli input storici sia di quelli nuovi; scarto massimo
#      dagli embedding storici. Tolleranza 1e-4 come check-encoder.
#   I risultati vanno in work/legacy_check.json; la cartella temporanea viene cancellata.
#   Un esito "different" NON viene nascosto: il report riporta dove gli input divergono.
import json
from pathlib import Path

import numpy as np

from biomap_v2.common import save_json


def _compare(name, old, new):
    entry = {'shape_historical': list(old.shape), 'shape_new': list(new.shape)}
    if old.shape != new.shape:
        entry['equal'] = False
        return entry
    entry['equal'] = bool(np.array_equal(old, new))
    if not entry['equal'] and old.size:
        diff = np.abs(old.astype(np.float64) - new.astype(np.float64))
        entry['n_different'] = int((diff > 0).sum())
        entry['max_abs'] = float(diff.max())
    return entry


def verify_legacy_sample(args, spec=None, frame=None, tolerance=1e-4):
    from .streaming import Timings, reader, Pixels, catalog, acquire_points, load_model, remove_temporary
    from .extraction import infer
    bundle = Path(args.liguria_bundle)
    manifest = json.loads((bundle / 'bundle.json').read_text(encoding='utf-8'))
    support = json.loads((bundle / 'support_verification.json').read_text(encoding='utf-8'))
    tile_name = manifest['encoder_sample']['tile']
    tile = Path(args.reference) / 'liguria_levante' / tile_name
    geometry = json.loads((tile / 'geometry.json').read_text(encoding='utf-8'))
    historical_pixels = next(t['pixels'] for t in support['tiles'] if t['tile'] == tile_name)
    with np.load(bundle / 'encoder_sample.npz', allow_pickle=False) as z:
        sample = {k: z[k] for k in z.files}
    report = {'tile': tile_name, 'pixels_new_geometry': geometry['pixels'], 'pixels_historical_geometry': historical_pixels}
    if geometry['pixels'] != historical_pixels:
        report['status'] = 'geometry_differs'
        save_json(Path(args.work_root) / 'legacy_check.json', report)
        return report
    positions = sample['pixel_indices'].astype(np.int64)
    pixels = Pixels(tile, positions)
    work = Path(args.work_root)
    timings = Timings()
    source = reader(timings, work)
    temporary = work / 'tmp' / 'legacy_check'
    try:
        # Catalogo della geometria intera, come nel run: i 32 pixel ricevono gli stessi giorni.
        streams, report['catalog_bbox'] = catalog(tile, pixels, work / 'api' / 'liguria_levante' / tile_name)
        raw = acquire_points(pixels, streams, temporary / 'days', {'protocol': 'legacy-check', 'tile': tile_name,
                             'positions': positions.tolist()}, source, getattr(args, 'io_threads', 1))
    finally:
        source.close()
    report['inputs'] = {key: _compare(key, sample[key], raw[key])
                        for key in ['s2_doys', 'asc_doys', 'desc_doys', 's2', 'mask', 'asc', 'desc']}
    report['inputs_identical'] = all(v['equal'] for v in report['inputs'].values())
    model = load_model(args)
    historical_inputs = {k: sample[k] for k in ['s2', 'mask', 'asc', 'desc', 's2_doys', 'asc_doys', 'desc_doys']}
    on_old = infer(model, historical_inputs, args.device, args.pixel_batch, False)
    on_new = infer(model, raw, args.device, args.pixel_batch, False)
    report['embedding_max_abs_historical_inputs'] = float(np.abs(on_old - sample['expected']).max())
    report['embedding_max_abs_new_inputs'] = float(np.abs(on_new - sample['expected']).max())
    report['tolerance'] = tolerance
    report['device'] = str(args.device)
    report['reader_timings'] = timings.reset()
    report['status'] = ('identical' if report['inputs_identical'] and report['embedding_max_abs_new_inputs'] <= tolerance
                        else 'different')
    save_json(work / 'legacy_check.json', report)
    if temporary.exists():
        remove_temporary(temporary, work)
    return report
