# SCOPO
#   Scaricare sul PC le tile GeoTESSERA **V1 storiche** (dataset 1.0, variante "vultr", anno 2020)
#   delle 18 AOI del manifest configs/msa_v1_liguria_cells.csv. Servono SOLO a due cose:
#     1. ricostruire le feature V1 dense (128 medie int8*scale) come nel benchmark storico;
#     2. definire il supporto pixel (centri pixel V1 a scala finita) su cui l'encoder V2 Medium
#        viene poi eseguito localmente. Gli embedding V2 NON vengono scaricati da qui.
#
# PERCHÉ LA VERSIONE È FISSATA
#   geotessera 0.11 usa per default dataset_version="v1.1" (variante NPY "cambridge"), un run
#   diverso dal V1 storico. Il registro della libreria dichiara 1.0/"vultr" = "Legacy model",
#   pubblicato in npy/v1 su https://data.source.coop/tessera/tessera. Qui versione e variante
#   sono passate esplicitamente e verificate dopo l'apertura del registro; nessun default.
#
# COME SI RICHIAMA (solo con l'ambiente isolato, non con .venv del benchmark)
#   .venv-geotessera\Scripts\python.exe scripts\download_v1_tiles.py --out F:\biomap_dense_public\v1_tiles
#   Opzioni: --aoi <id> [<id> ...] per un sottoinsieme; --dry-run per elencare le tile senza scaricare.
#
# LOGICA
#   1. Estensione di ogni AOI = min/max dei bordi delle celle del manifest (stesso box del target
#      storico ritagliato: per la Liguria 9.35-9.65 x 44.15-44.35, verificato sul pacchetto).
#   2. Tile candidate = tile 0.1° (centri a x.x5) il cui quadrato CHIUSO interseca il box AOI.
#      Riproduce l'inventario storico: Liguria 10 tile, Abruzzo 16 tile (bordi su multipli di 0.1).
#   3. Si scaricano solo le tile presenti nel registro 2020 (le tile interamente in mare non esistono).
#   4. Layout locale: <out>/<aoi>/tessera/2020/global_0.1_degree_representation/2020/grid_*/...
#      e global_0.1_degree_tiff_all/grid_*.tiff, esattamente quello letto da dense/v1_support.py.
#   5. La libreria scrive in modo atomico e verifica l'MD5 dell'ETag; i file già presenti sono riusati.
#      Alla fine si scrive <out>/<aoi>/v1_download_manifest.json con SHA256 di ogni file.
#
# RIPRESA
#   Errori HTTP transitori: 6 tentativi per tile con attesa crescente (30 s ... 16 min).
#   Rilanciare lo stesso comando: i file completi vengono saltati, quelli interrotti non esistono
#   (scrittura su file temporaneo + rename). Nessun file viene cancellato.
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATASET_VERSION = '1.0'
DATASET_VARIANT = 'vultr'
YEAR = 2020
EPS = 1e-9


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def aoi_boxes(manifest):
    frame = pd.read_csv(manifest)
    return {aoi: (float(g.west.min()), float(g.south.min()), float(g.east.max()), float(g.north.max()))
            for aoi, g in frame.groupby('aoi_id', sort=True)}


# Centri a x.x5: il quadrato [c-0.05, c+0.05] deve toccare il box chiuso dell'AOI.
def candidate_tiles(box):
    import math
    west, south, east, north = box

    # Tile k copre [k/10, k/10+0.1]; tocca [lo, hi] se k/10 <= hi e k/10+0.1 >= lo.
    def centres(lo, hi):
        first, last = math.ceil(round((lo - 0.1) * 10, 6)), math.floor(round(hi * 10, 6))
        return [round(k / 10 + 0.05, 2) for k in range(first, last + 1)]
    return [(lon, lat) for lon in centres(west, east) for lat in centres(south, north)]


def main(argv=None):
    parser = argparse.ArgumentParser(description='Download tile GeoTESSERA V1 storiche (1.0/vultr, 2020)')
    parser.add_argument('--out', required=True)
    parser.add_argument('--aoi', nargs='+')
    parser.add_argument('--manifest', default=str(ROOT / 'configs/msa_v1_liguria_cells.csv'))
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)
    import geotessera
    from geotessera import GeoTessera
    from geotessera.registry import dataset_path, TESSERA_NPY_MIRROR_URL
    boxes = aoi_boxes(args.manifest)
    if args.aoi and not set(args.aoi) <= set(boxes):
        raise SystemExit(f'AOI sconosciute: {sorted(set(args.aoi) - set(boxes))}')
    summary = {}
    for aoi, box in boxes.items():
        if args.aoi and aoi not in args.aoi:
            continue
        directory = Path(args.out) / aoi / 'tessera' / str(YEAR)
        directory.mkdir(parents=True, exist_ok=True)
        gt = GeoTessera(dataset_version=DATASET_VERSION, dataset_variant=DATASET_VARIANT,
                        embeddings_dir=directory, cache_dir=Path(args.out) / '_registry_cache',
                        bbox=box, bbox_margin_deg=0.15)
        # Verifica esplicita: il registro aperto deve essere proprio 1.0/vultr (npy/v1).
        if gt.dataset_variant != DATASET_VARIANT or gt.registry._dataset_path != dataset_path(DATASET_VERSION, DATASET_VARIANT) \
                or gt.registry._dataset_path != 'v1':
            raise SystemExit(f'Registro inatteso: {gt.dataset_variant} {gt.registry._dataset_path}')
        available = {(round(lon, 2), round(lat, 2)) for _, lon, lat in
                     gt.registry.load_blocks_for_region((box[0] - 0.1, box[1] - 0.1, box[2] + 0.1, box[3] + 0.1), YEAR)}
        wanted = [t for t in candidate_tiles(box) if t in available]
        print(f'{aoi}: box {box}, {len(candidate_tiles(box))} candidate, {len(wanted)} nel registro {YEAR}', flush=True)
        files = {}
        for index, (lon, lat) in enumerate(wanted, 1):
            name = f'grid_{lon:.2f}_{lat:.2f}'
            if args.dry_run:
                print(f'  {name}', flush=True)
                continue
            started = time.perf_counter()
            # Errori HTTP transitori (es. 520 del gateway): fino a 6 tentativi con attesa crescente.
            for attempt in range(6):
                if gt.download_tile(lon, lat, YEAR):
                    break
                if attempt == 5:
                    raise SystemExit(f'Download fallito: {aoi}/{name}; rilanciare il comando (i file completi restano)')
                print(f'  {name}: tentativo {attempt + 1} fallito, nuovo tentativo tra {30 * 2 ** attempt}s', flush=True)
                time.sleep(30 * 2 ** attempt)
            paths = [directory / 'global_0.1_degree_representation' / str(YEAR) / name / f'{name}.npy',
                     directory / 'global_0.1_degree_representation' / str(YEAR) / name / f'{name}_scales.npy',
                     directory / 'global_0.1_degree_tiff_all' / f'{name}.tiff']
            for path in paths:
                if not path.is_file():
                    raise SystemExit(f'File atteso assente dopo il download: {path}')
                files[str(path.relative_to(directory)).replace('\\', '/')] = {'sha256': sha256(path), 'bytes': path.stat().st_size}
            print(f'  {index}/{len(wanted)} {name} ({time.perf_counter() - started:.1f}s)', flush=True)
        if args.dry_run:
            continue
        record = {'aoi_id': aoi, 'box': box, 'year': YEAR, 'geotessera_version': geotessera.__version__,
                  'dataset_version': DATASET_VERSION, 'dataset_variant': DATASET_VARIANT,
                  'dataset_path': gt.registry._dataset_path, 'source': f'{TESSERA_NPY_MIRROR_URL}/{gt.registry._dataset_path}',
                  'tile_rule': 'closed 0.1 deg tile square intersects AOI box; present in 2020 registry',
                  'tiles': [f'grid_{lon:.2f}_{lat:.2f}' for lon, lat in wanted], 'files': files}
        (Path(args.out) / aoi / 'v1_download_manifest.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
        summary[aoi] = len(wanted)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == '__main__':
    sys.exit(main())
