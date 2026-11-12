# SCOPO
#   Copiare, in sola lettura dalla campagna r1, i cataloghi STAC già salvati di alcune tile dentro una
#   work-root di prova SEPARATA. Così il test reale della pipeline r2 usa esattamente le stesse scene della
#   produzione: un'eventuale differenza nei risultati non può dipendere da un catalogo STAC cambiato nel
#   frattempo su Planetary Computer.
# COME VIENE RICHIAMATO
#   python scripts\seed_catalogs.py --from F:\biomap_dense_public\work --to F:\dense_fast_check_20261112
#       --tiles belluno_dolomiti/grid_11.95_45.95 belluno_dolomiti/grid_12.05_45.95 belluno_dolomiti/grid_11.95_46.25
# LOGICA IN ORDINE
#   1. Rifiuta una destinazione uguale o interna alla work-root di produzione.
#   2. Per ogni tile legge con condivisione completa api/<aoi>/<tile>/{sentinel-2-l2a,sentinel-1-rtc}_edge_bbox.json.
#   3. Li scrive nella stessa posizione relativa della destinazione e verifica lo SHA256 delle copie.
# INPUT E OUTPUT
#   Input: cataloghi della produzione (sola lettura). Output: copie nella work-root di prova + riepilogo.
# PARAMETRI
#   --from, --to, --tiles.
# RISORSE
#   Pochi MB.
# RIPRESA
#   Rieseguibile: sovrascrive solo le copie nella destinazione.
import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'src'))

from biomap_v2.dense_fast.identity import refuse_production  # noqa: E402
from biomap_v2.dense_fast.shared_read import read_bytes  # noqa: E402

FILES = ['sentinel-2-l2a_edge_bbox.json', 'sentinel-1-rtc_edge_bbox.json']


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--from', dest='source', required=True)
    p.add_argument('--to', dest='target', required=True)
    p.add_argument('--tiles', nargs='+', required=True)
    args = p.parse_args()
    source, target = Path(args.source).resolve(), Path(args.target).resolve()
    if target == source or source in target.parents:
        sys.exit('La destinazione non può essere dentro la work-root sorgente')
    refuse_production(target)
    report = []
    for tile in args.tiles:
        aoi, name = tile.split('/', 1)
        for f in FILES:
            data = read_bytes(source / 'api' / aoi / name / f)
            out = target / 'api' / aoi / name / f
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(data)
            same = hashlib.sha256(out.read_bytes()).hexdigest() == hashlib.sha256(data).hexdigest()
            report.append({'tile': tile, 'file': f, 'items': len(json.loads(data)), 'bytes': len(data), 'copy_verified': same})
    print(json.dumps(report, indent=2))
    if not all(r['copy_verified'] for r in report):
        sys.exit(1)


if __name__ == '__main__':
    main()
