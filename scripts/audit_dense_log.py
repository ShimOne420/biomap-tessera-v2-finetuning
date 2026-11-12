# SCOPO
#   Analisi in SOLA LETTURA dei tempi della campagna V2 densa (r1 o r2) da stream_log.jsonl: misure per
#   tile, per fascia di dimensione, costo fisso stimato, frazione di GPU ferma, latenze di lettura remota.
# COME VIENE RICHIAMATO
#   python scripts\audit_dense_log.py --log F:\biomap_dense_public\work\stream_log.jsonl [--json out.json]
#   (--json scrive SOLO nel percorso indicato, che non deve stare nella work-root di produzione)
# LOGICA IN ORDINE
#   1. Apre il log con condivisione Read|Write|Delete (Windows) e lo chiude subito: la campagna può
#      continuare ad aggiungere righe e a sostituire i propri file mentre leggiamo.
#   2. Per ogni tile: pixel, acquisizione, codifica, scene, contatori del lettore (solo SCL/S1: le bande
#      S2 riproiettate non passano dal lettore e NON sono incluse in remote_read).
#   3. Tile riusate dalla cache (acquisizione < 30 s) escluse dalle stime di costo.
#   4. Fasce: <10k, 10k-100k, 100k-600k, >=600k pixel; regressione tempo = a*tile + b*pixel.
#   5. GPU ferma stimata = acquisizione / (acquisizione + codifica): nella r1 le due fasi sono in serie.
# INPUT E OUTPUT
#   Input: stream_log.jsonl. Output: testo a schermo e, se richiesto, un JSON fuori dalla produzione.
# PARAMETRI
#   --log, --json.
# RISORSE
#   Trascurabili.
# RIPRESA
#   Nessuno stato.
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np


def read_shared(path):
    if os.name == 'nt':
        import ctypes
        import msvcrt
        from ctypes import wintypes
        kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel32.CreateFileW.restype = wintypes.HANDLE
        handle = kernel32.CreateFileW(str(path), 0x80000000, 0x1 | 0x2 | 0x4, None, 3, 0x80, None)
        if handle == wintypes.HANDLE(-1).value:
            raise OSError(ctypes.get_last_error(), f'Impossibile aprire {path}')
        fd = msvcrt.open_osfhandle(handle, os.O_RDONLY)
        with open(fd, 'rb') as stream:
            return stream.read().decode('utf-8', errors='replace')
    return Path(path).read_text(encoding='utf-8', errors='replace')


def analyse(text):
    rows = []
    for line in text.splitlines():
        if not line.strip():
            continue
        e = json.loads(line)
        reader = e.get('reader') or {}
        sec, cnt = reader.get('seconds', {}), reader.get('counters', {})
        rows.append({'tile': f"{e['aoi_id']}/{e['tile']}", 'pixels': e['coverage']['pixels'],
                     'acquire': e['acquire_seconds'], 'encode': e['encode_seconds'],
                     's2': e['scenes']['s2'], 'asc': e['scenes']['asc'], 'desc': e['scenes']['desc'],
                     'remote_read': sec.get('remote_read', 0.), 'remote_open': sec.get('remote_open', 0.),
                     'coordinate_transform': sec.get('coordinate_transform', 0.),
                     'blocks': cnt.get('remote_blocks_read', 0), 'opens': cnt.get('remote_asset_opens', 0),
                     'requests': cnt.get('asset_requests', 0), 'failures': cnt.get('remote_read_failures', 0),
                     'retries': cnt.get('remote_retries', 0), 'decoded': cnt.get('decoded_raster_bytes', 0)})
    fresh = [r for r in rows if r['acquire'] >= 30]
    bins = [('<10k', 0, 10_000), ('10k-100k', 10_000, 100_000), ('100k-600k', 100_000, 600_000), ('>=600k', 600_000, 10**9)]
    by_bin = {}
    for name, low, high in bins:
        sel = [r for r in fresh if low <= r['pixels'] < high]
        if not sel:
            continue
        px = sum(r['pixels'] for r in sel)
        wall = sum(r['acquire'] + r['encode'] for r in sel)
        by_bin[name] = {'tiles': len(sel), 'pixels_mean': px / len(sel),
                        'acquire_mean_s': float(np.mean([r['acquire'] for r in sel])),
                        'encode_mean_s': float(np.mean([r['encode'] for r in sel])),
                        'pixels_per_wall_second': px / wall if wall else None,
                        'gpu_idle_fraction': sum(r['acquire'] for r in sel) / wall if wall else None}
    x = np.array([[1., r['pixels']] for r in fresh])
    y = np.array([r['acquire'] + r['encode'] for r in fresh])
    coef = np.linalg.lstsq(x, y, rcond=None)[0] if len(fresh) >= 5 else [np.nan, np.nan]
    enc = np.array([[r['pixels']] for r in fresh if r['pixels'] > 100_000])
    enc_y = np.array([r['encode'] for r in fresh if r['pixels'] > 100_000])
    total_acq = sum(r['acquire'] for r in fresh)
    total_enc = sum(r['encode'] for r in fresh)
    blocks = sum(r['blocks'] for r in fresh)
    return {
        'tiles_logged': len(rows), 'tiles_fresh': len(fresh), 'tiles_reused_from_cache': len(rows) - len(fresh),
        'wall_hours_fresh': (total_acq + total_enc) / 3600,
        'acquire_share_of_wall': total_acq / (total_acq + total_enc) if fresh else None,
        'encode_ms_per_pixel_large_tiles': float(1000 * enc_y.sum() / enc.sum()) if len(enc) else None,
        'fixed_seconds_per_tile_fit': float(coef[0]), 'ms_per_pixel_fit': float(1000 * coef[1]),
        'reader_remote_read_seconds_per_block_thread_summed': (sum(r['remote_read'] for r in fresh) / blocks) if blocks else None,
        'reader_remote_open_seconds_per_open_thread_summed': (sum(r['remote_open'] for r in fresh) / max(1, sum(r['opens'] for r in fresh))),
        'reader_requests_per_tile_mean': float(np.mean([r['requests'] for r in fresh])) if fresh else None,
        'reader_failures_total': sum(r['failures'] for r in rows), 'reader_retries_total': sum(r['retries'] for r in rows),
        'reader_decoded_gb_total': sum(r['decoded'] for r in rows) / 1e9,
        'coordinate_transform_thread_summed_s_large_mean': float(np.mean([r['coordinate_transform'] for r in fresh if r['pixels'] > 600_000] or [0])),
        'bins': by_bin}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--log', required=True)
    p.add_argument('--json')
    args = p.parse_args()
    result = analyse(read_shared(args.log))
    print(json.dumps(result, indent=2))
    if args.json:
        out = Path(args.json).resolve()
        if Path(args.log).resolve().parent in out.parents or out.parent == Path(args.log).resolve().parent:
            sys.exit('Rifiuto di scrivere nella cartella del log di produzione')
        out.write_text(json.dumps(result, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
