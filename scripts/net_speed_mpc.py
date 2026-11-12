# SCOPO
#   Test di velocità verso Microsoft Planetary Computer (la sorgente reale della campagna): scarica in
#   parallelo un numero limitato di byte da alcuni COG Sentinel-2 elencati nei cataloghi della campagna e
#   misura i Mbit/s. Da usare SOLO con la campagna ferma (prima e dopo il cambio di cavo), per non rubarle banda.
# COME VIENE RICHIAMATO
#   python scripts\net_speed_mpc.py --catalog F:\biomap_dense_public\work\api\belluno_dolomiti\grid_11.95_45.95\sentinel-2-l2a_edge_bbox.json
#       [--files 8] [--mb-per-file 48] [--parallel 8] [--approved]
# LOGICA IN ORDINE
#   1. Legge il catalogo (lettura condivisa) e prende i primi --files asset di banda B04/B08/B11/B12.
#   2. Ottiene il token SAS anonimo di Planetary Computer per l'account/container dell'asset.
#   3. Scarica in parallelo i primi --mb-per-file MB di ciascun file (richiesta Range), scartando i dati.
#   4. Stampa byte totali, tempo reale e Mbit/s complessivi. Richiede --approved (usa rete).
# INPUT E OUTPUT
#   Input: catalogo STAC già salvato dalla campagna. Output: solo testo; nessun file scritto.
# PARAMETRI
#   --files (8), --mb-per-file (48), --parallel (8): con i valori predefiniti ~384 MB.
# RISORSE
#   Rete: circa 30 s a 100 Mbit/s, 3-6 s a 1 Gbit/s.
# RIPRESA
#   Nessuno stato.
import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlparse

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'src'))

import requests  # noqa: E402

from biomap_v2.dense_fast.shared_read import read_text  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--catalog', required=True)
    p.add_argument('--files', type=int, default=8)
    p.add_argument('--mb-per-file', type=int, default=48)
    p.add_argument('--parallel', type=int, default=8)
    p.add_argument('--approved', action='store_true')
    args = p.parse_args()
    if not args.approved:
        sys.exit('Il test scarica dati da Planetary Computer: rilanciare con --approved, a campagna ferma')
    items = json.loads(read_text(args.catalog))
    hrefs = [it['assets'][b]['href'] for it in items for b in ('B04', 'B08', 'B11', 'B12') if b in it['assets']][:args.files]
    tokens = {}

    def signed(href):
        u = urlparse(href)
        key = (u.netloc.split('.')[0], u.path.split('/')[1])
        if key not in tokens:
            r = requests.get(f'https://planetarycomputer.microsoft.com/api/sas/v1/token/{key[0]}/{key[1]}', timeout=30)
            r.raise_for_status()
            tokens[key] = r.json()['token']
        return href.split('?')[0] + '?' + tokens[key]

    urls = [signed(h) for h in hrefs]
    limit = args.mb_per_file * 1024 * 1024

    def fetch(url):
        got = 0
        with requests.get(url, headers={'Range': f'bytes=0-{limit - 1}'}, stream=True, timeout=60) as r:
            r.raise_for_status()
            for chunk in r.iter_content(1024 * 1024):
                got += len(chunk)
        return got

    started = time.perf_counter()
    with ThreadPoolExecutor(args.parallel) as pool:
        total = sum(pool.map(fetch, urls))
    wall = time.perf_counter() - started
    print(json.dumps({'files': len(urls), 'bytes': total, 'seconds': round(wall, 2),
                      'mbit_per_s': round(total * 8 / wall / 1e6, 1)}, indent=2))


if __name__ == '__main__':
    main()
