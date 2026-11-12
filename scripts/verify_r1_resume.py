# SCOPO
#   Verifica in SOLA LETTURA che la campagna r1 (dense127909-stream-legacy-exact-r1) possa essere
#   ripresa senza perdere né invalidare le tile completate. Si usa prima di fermare il run, dopo averlo
#   fermato e dopo il riavvio. Ripete esattamente il controllo che la r1 fa alla ripresa
#   (campaign/state.cached + identità di streaming.process_tile), senza scrivere nulla.
# COME VIENE RICHIAMATO
#   python scripts\verify_r1_resume.py --reference F:\biomap_dense_public\reference
#       --work-root F:\biomap_dense_public\work
#       --production-repo C:\Users\brome\Desktop\BIOMAP_TESSERA_V2\biomap-tessera-v2-finetuning
#       [--checkpoint C:\Users\brome\Desktop\BIOMAP_TESSERA_V2\data\weights\student_medium.pt] [--json out.json]
#   Codice di uscita 0 = ripresa sicura; 1 = almeno un problema (elencato).
# LOGICA IN ORDINE
#   1. Identità del codice ATTUALE della worktree di produzione: SHA256 dei 5 CODE_FILES r1.
#   2. (opzionale) SHA256 del checkpoint == WEIGHTS_SHA256 del Medium.
#   3. Per ogni tile del piano (ordine reference.json) con chunks/<aoi>/<tile>/tile.json:
#      identità del marker == {protocol r1, SHA256 di geometry.npz, 2020, checkpoint, fp32, codice attuale};
#      SHA256 di tile.npz == marker; array cell/sums/counts leggibili, coerenti, finiti, conteggi > 0.
#      Qualsiasi differenza = la r1 solleverebbe "Cache incompatibile/corrotta" alla ripresa.
#   4. Tile in corso: cartelle tmp/<aoi>/<tile> con giorni e blocchi codificati già salvati (riusati).
#   5. Argomenti dell'ultimo avvio (last_invocation.json): da riusare identici al riavvio.
# INPUT E OUTPUT
#   Input (sola lettura, condivisione completa): reference.json, geometry.npz, chunks/, tmp/, codice,
#   checkpoint, last_invocation.json. Output: testo e, se richiesto, JSON fuori dalla work-root.
# PARAMETRI
#   --checkpoint per verificare anche i pesi (legge ~84 MB).
# RISORSE
#   Legge i tile.npz completati e i geometry.npz corrispondenti (qualche centinaio di MB in tutto).
# RIPRESA
#   Nessuno stato; rieseguibile in qualsiasi momento, anche con la campagna attiva.
import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'src'))

import numpy as np  # noqa: E402

from biomap_v2.dense_fast.shared_read import read_text, sha256_file, load_npz  # noqa: E402

PROTOCOL_R1 = 'dense127909-stream-legacy-exact-r1'
CODE_FILES_R1 = ['src/biomap_v2/campaign/streaming.py', 'src/biomap_v2/raster_cache.py', 'src/biomap_v2/data.py',
                 'src/biomap_v2/reference.py', 'src/biomap_v2/encoder.py']
WEIGHTS_SHA256 = '3823be7db9d9cfc93f3c2a47c7699be82821ab4e1117d4d2befdb746941ee96e'


def verify(reference, work, repo, checkpoint=None):
    reference, work, repo = Path(reference), Path(work), Path(repo)
    problems, warnings = [], []
    code = {name: sha256_file(repo / name) for name in CODE_FILES_R1}
    if checkpoint:
        if sha256_file(checkpoint) != WEIGHTS_SHA256:
            problems.append(f'Checkpoint {checkpoint}: SHA256 diverso dal Medium verificato')
    spec = json.loads(read_text(reference / 'reference.json'))
    plan = [(a['aoi_id'], t) for a in spec['areas'] for t in a['tiles']]
    complete, pixels_complete, pixels_total = [], 0, 0
    for aoi, tile in plan:
        geometry = json.loads(read_text(reference / aoi / tile / 'geometry.json'))
        pixels_total += geometry['pixels']
        marker = work / 'chunks' / aoi / tile / 'tile.json'
        if not marker.exists():
            continue
        info = json.loads(read_text(marker))
        expected = {'protocol': PROTOCOL_R1, 'geometry': sha256_file(reference / aoi / tile / 'geometry.npz'),
                    'year': 2020, 'checkpoint': WEIGHTS_SHA256, 'precision': 'fp32', 'code': code}
        name = f'{aoi}/{tile}'
        if info['identity'] != expected:
            diff = sorted(k for k in set(expected) | set(info['identity']) if info['identity'].get(k) != expected.get(k))
            problems.append(f'{name}: identità diversa nei campi {diff} -> la r1 si fermerebbe alla ripresa')
            continue
        payload = work / 'chunks' / aoi / tile / 'tile.npz'
        if not payload.exists() or sha256_file(payload) != info['sha256']:
            problems.append(f'{name}: tile.npz assente o SHA256 diverso dal marker')
            continue
        z = load_npz(payload)
        if not (len(z['cell']) == len(z['sums']) == len(z['counts']) and z['sums'].shape[1:] == (128,)
                and np.isfinite(z['sums']).all() and (z['counts'] > 0).all()):
            problems.append(f'{name}: array incoerenti')
            continue
        if int(z['counts'].sum()) != geometry['pixels']:
            problems.append(f'{name}: somma dei conteggi {int(z["counts"].sum())} != pixel della geometria {geometry["pixels"]}')
            continue
        complete.append(name)
        pixels_complete += geometry['pixels']
    in_progress = []
    tmp = work / 'tmp'
    if tmp.exists():
        for folder in sorted(p for p in tmp.glob('*/*') if p.is_dir()):
            days = list((folder / 'days').glob('*.json')) if (folder / 'days').exists() else []
            encoded = list((folder / 'encoded').glob('*.json')) if (folder / 'encoded').exists() else []
            in_progress.append({'tile': f'{folder.parent.name}/{folder.name}', 'days_saved': len(days), 'encoded_blocks_saved': len(encoded)})
    invocation = json.loads(read_text(work / 'last_invocation.json')) if (work / 'last_invocation.json').exists() else {}
    if len(in_progress) > 1:
        warnings.append('Più di una tile con temporanei: normale solo se una tile è stata interrotta in passato')
    return {'ok': not problems, 'tiles_total': len(plan), 'tiles_complete_verified': len(complete),
            'pixels_complete': pixels_complete, 'pixels_total': pixels_total,
            'code_identity_current': code, 'in_progress': in_progress, 'last_invocation_args': invocation.get('args'),
            'last_invocation_git_sha': invocation.get('git_sha'), 'problems': problems, 'warnings': warnings,
            'last_complete': complete[-1] if complete else None}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--reference', required=True)
    p.add_argument('--work-root', required=True)
    p.add_argument('--production-repo', required=True)
    p.add_argument('--checkpoint')
    p.add_argument('--json')
    args = p.parse_args()
    result = verify(args.reference, args.work_root, args.production_repo, args.checkpoint)
    print(json.dumps({k: v for k, v in result.items() if k != 'code_identity_current'}, indent=2))
    print('RIPRESA SICURA' if result['ok'] else 'ATTENZIONE: NON RIAVVIARE PRIMA DI AVER CAPITO I PROBLEMI SOPRA')
    if args.json:
        out = Path(args.json).resolve()
        if Path(args.work_root).resolve() in out.parents:
            sys.exit('Rifiuto di scrivere nella work-root di produzione')
        out.write_text(json.dumps(result, indent=2), encoding='utf-8')
    sys.exit(0 if result['ok'] else 1)


if __name__ == '__main__':
    main()
