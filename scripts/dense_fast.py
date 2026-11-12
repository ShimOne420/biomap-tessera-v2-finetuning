# SCOPO
#   Interfaccia a riga di comando dell'estrazione V2 densa ottimizzata (protocollo r2-fast) e dei suoi
#   strumenti di verifica e misura. Non tocca mai la campagna r1 in produzione.
# COME VIENE RICHIAMATO (dalla radice della worktree, con l'interprete .venv del progetto)
#   python scripts\dense_fast.py stream --reference F:\biomap_dense_public\reference --work-root F:\dense_fast_run
#       --checkpoint ..\data\weights\student_medium.pt --device cuda:0 [--io-workers 16] [--batch-pixels 1500000]
#       [--prefetch 1] [--aoi A B] [--tiles aoi/tile ...] [--max-tiles N] [--keep-pixel-embeddings]
#   python scripts\dense_fast.py compare-tiles --left <work_root_r1> --right <work_root_r2> [--tiles aoi/tile ...]
#   python scripts\dense_fast.py bench --reference ... --bench-root F:\dense_bench --checkpoint ... --device cuda:0
#       --tiles aoi/tile ... --approved-network-benchmark
#   python scripts\dense_fast.py verify-synthetic
# LOGICA IN ORDINE
#   stream:          carica reference (campaign.reference.load), lock della work-root, last_invocation.json,
#                    pipeline.stream_fast.
#   compare-tiles:   per ogni tile presente in entrambe le work-root confronta cell, sums e counts dei tile.npz
#                    (bit per bit; se diversi riporta la differenza massima). Solo lettura su entrambe.
#   bench:           per le tile indicate esegue prima la r1 (campaign.streaming.process_tile, tile per tile in
#                    ordine di piano) in <bench-root>\legacy, poi la r2 in <bench-root>\fast, misurando il tempo
#                    REALE di ciascuna; poi compare-tiles tra le due. Richiede --approved-network-benchmark perché
#                    usa rete e GPU in concorrenza con la campagna in corso.
#   verify-synthetic: esegue i test offline di equivalenza (tests/test_dense_fast.py) con pytest.
# INPUT E OUTPUT
#   Vedere dense_fast/pipeline.py. bench scrive <bench-root>\bench_result.json.
# PARAMETRI
#   --io-workers (16): richieste remote in parallelo (pool globale condiviso da S2/S1 e dalle tile del lotto).
#   --batch-pixels (1.000.000): pixel massimi per lotto di tile vicine. RAM ~3,5 kB per pixel per lotto in memoria.
#                   Compromesso misurato nel benchmark simulato: lotti grandi condividono più byte ma la GPU parte
#                   solo a lotto finito; con ~1M le tile piccole viaggiano insieme e quelle grandi (~930k) da sole.
#   --prefetch (1): lotti acquisiti in attesa della GPU.
#   --memory-cache-mib (4096): cache RAM dei blocchi nativi SCL/S1.
#   --gdal-cache-mib (2048), --vsi-cache-mib (512): cache GDAL dei blocchi e dei byte HTTP.
#   --chunk-size (4096) e --pixel-batch (256): come la campagna r1; cambiarli cambia l'equivalenza bit per bit.
# RISORSE
#   Rete, GPU, RAM come sopra. Nessuna scrittura fuori dalla work-root o dalla bench-root.
# RIPRESA
#   stream e bench si riprendono rilanciando lo stesso comando (cache con identità r2).
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

for name in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ.setdefault(name, '1')
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'src'))
import numpy as np  # noqa: E402


def parser():
    p = argparse.ArgumentParser(description='Estrazione V2 densa r2-fast (worktree di ottimizzazione)')
    p.add_argument('command', choices=['stream', 'compare-tiles', 'bench', 'bench-legacy', 'bench-fast', 'verify-synthetic'])
    p.add_argument('--reference')
    p.add_argument('--work-root')
    p.add_argument('--bench-root')
    p.add_argument('--left')
    p.add_argument('--right')
    p.add_argument('--checkpoint')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--chunk-size', type=int, default=4096)
    p.add_argument('--pixel-batch', type=int, default=256)
    p.add_argument('--io-workers', type=int, default=16)
    p.add_argument('--batch-pixels', type=int, default=1_000_000)
    p.add_argument('--prefetch', type=int, default=1)
    p.add_argument('--memory-cache-mib', type=int, default=4096)
    p.add_argument('--gdal-cache-mib', type=int, default=2048)
    p.add_argument('--vsi-cache-mib', type=int, default=512)
    p.add_argument('--aoi', nargs='+')
    p.add_argument('--tiles', nargs='+', help='aoi/tile espliciti (prove e benchmark)')
    p.add_argument('--max-tiles', type=int)
    p.add_argument('--keep-pixel-embeddings', action='store_true')
    p.add_argument('--legacy-io-threads', type=int, default=16, help='Thread della r1 nel benchmark (come produzione)')
    p.add_argument('--approved-network-benchmark', action='store_true')
    return p


def compare_tiles(left, right, tiles=None):
    left, right = Path(left), Path(right)
    names = sorted(p.parent.relative_to(left / 'chunks').as_posix() for p in (left / 'chunks').glob('*/*/tile.npz'))
    if tiles:
        names = [t for t in names if t in set(tiles)]
    report = []
    for name in names:
        other = right / 'chunks' / name / 'tile.npz'
        if not other.exists():
            continue
        with np.load(left / 'chunks' / name / 'tile.npz') as a, np.load(other) as b:
            same_cells = np.array_equal(a['cell'], b['cell'])
            row = {'tile': name, 'cells': int(len(a['cell'])), 'cell_equal': same_cells,
                   'counts_equal': same_cells and np.array_equal(a['counts'], b['counts']),
                   'sums_bitwise_equal': same_cells and np.array_equal(a['sums'], b['sums'])}
            if same_cells and not row['sums_bitwise_equal']:
                row['sums_max_abs_diff'] = float(np.abs(a['sums'] - b['sums']).max())
                row['mean_max_abs_diff'] = float(np.abs(a['sums'] / a['counts'][:, None]
                                                        - b['sums'] / b['counts'][:, None]).max())
            report.append(row)
    return {'left': str(left), 'right': str(right), 'compared': len(report),
            'all_bitwise_equal': bool(report) and all(r['sums_bitwise_equal'] and r['counts_equal'] for r in report),
            'tiles': report}


def legacy_bench(args, spec, frame, root, plan):
    from types import SimpleNamespace
    from biomap_v2.campaign import streaming as legacy
    work = Path(root)
    work.mkdir(parents=True, exist_ok=True)
    largs = SimpleNamespace(work_root=str(work), reference=args.reference, checkpoint=args.checkpoint, device=args.device,
                            chunk_size=args.chunk_size, pixel_batch=args.pixel_batch, io_threads=args.legacy_io_threads,
                            keep_pixel_embeddings=False, amp=False, aoi=None, max_tiles=None)
    timings = legacy.Timings()
    state = {'frame': frame, 'model': None, 'code': legacy.code_identity(), 'started': time.perf_counter(),
             'tiles_done': 0, 'tiles_total': len(plan), 'pixels_total': 0, 'pixels_done': 0, 'pixels_at_start': 0}
    source = legacy.reader(timings, work)
    ref = Path(args.reference)
    per_tile = []
    started = time.perf_counter()
    try:
        for aoi, tile_name in plan:
            lookup = np.load(ref / aoi / 'lookup.npy')
            with np.load(ref / aoi / tile_name / 'geometry.npz') as z:
                ids = lookup[z['cell_ids']]
            t0 = time.perf_counter()
            legacy.process_tile(largs, aoi, tile_name, ids, state, source, timings)
            per_tile.append({'tile': f'{aoi}/{tile_name}', 'wall_seconds': time.perf_counter() - t0})
    finally:
        source.close()
    return {'wall_seconds': time.perf_counter() - started, 'tiles': per_tile}


def peak_vram():
    try:
        import torch
        return int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else None
    except Exception:
        return None


def main(argv=None):
    args = parser().parse_args(argv)
    if args.command == 'verify-synthetic':
        return subprocess.call([sys.executable, '-m', 'pytest', '-q', str(REPO / 'tests' / 'test_dense_fast.py')], cwd=REPO)
    if args.command == 'compare-tiles':
        print(json.dumps(compare_tiles(args.left, args.right, args.tiles), indent=2))
        return 0
    from filelock import FileLock
    from biomap_v2.campaign.reference import load
    from biomap_v2.dense_fast.identity import refuse_production, PROTOCOL
    from biomap_v2.dense_fast.io_safe import save_json
    from biomap_v2.dense_fast.pipeline import stream_fast
    spec, frame = load(args.reference)
    if args.command == 'stream':
        work = refuse_production(args.work_root)
        work.mkdir(parents=True, exist_ok=True)
        with FileLock(str(work / 'campaign.lock'), timeout=0):
            save_json(work / 'last_invocation.json', {'tool': 'dense_fast', 'protocol': PROTOCOL, 'args': vars(args),
                      'git_sha': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip()})
            stream_fast(args, spec, frame)
        return 0
    if not args.approved_network_benchmark:
        raise SystemExit('bench usa rete e GPU in concorrenza con la campagna: serve --approved-network-benchmark')
    if not args.tiles or not args.bench_root:
        raise SystemExit('bench richiede --bench-root e --tiles aoi/tile ...')
    bench = refuse_production(args.bench_root)
    bench.mkdir(parents=True, exist_ok=True)
    order = [(a['aoi_id'], t) for a in spec['areas'] for t in a['tiles']]
    wanted = {tuple(x.split('/', 1)) for x in args.tiles}
    plan = [key for key in order if key in wanted]
    if len(plan) != len(wanted):
        raise SystemExit('Tile richieste non presenti nel reference')
    if args.command == 'bench-legacy':
        result = legacy_bench(args, spec, frame, bench / 'legacy', plan)
        result.update(memory=_rss_bytes(), peak_vram_bytes=peak_vram())
        save_json(bench / 'legacy_result.json', result)
        return 0
    if args.command == 'bench-fast':
        fast = bench / 'fast'
        fast.mkdir(parents=True, exist_ok=True)
        args.work_root, args.aoi = str(fast), None
        started = time.perf_counter()
        status = stream_fast(args, spec, frame)
        save_json(bench / 'fast_result.json', {'wall_seconds': time.perf_counter() - started, 'status': status,
                                                'memory': _rss_bytes(), 'peak_vram_bytes': peak_vram()})
        return 0
    # bench: due processi separati (nessuna cache GDAL condivisa), prima r1 poi r2, poi confronto.
    common = ['--reference', args.reference, '--bench-root', str(bench), '--checkpoint', args.checkpoint,
              '--device', args.device, '--chunk-size', str(args.chunk_size), '--pixel-batch', str(args.pixel_batch),
              '--approved-network-benchmark', '--tiles', *args.tiles]
    measures = {}
    for name, extra in [('legacy', ['--legacy-io-threads', str(args.legacy_io_threads)]),
                        ('fast', ['--io-workers', str(args.io_workers), '--batch-pixels', str(args.batch_pixels),
                                  '--prefetch', str(args.prefetch), '--memory-cache-mib', str(args.memory_cache_mib),
                                  '--gdal-cache-mib', str(args.gdal_cache_mib), '--vsi-cache-mib', str(args.vsi_cache_mib)])]:
        net_before, started = _nic_bytes(), time.perf_counter()
        code = subprocess.call([sys.executable, str(Path(__file__).resolve()), f'bench-{name}', *common, *extra], cwd=REPO)
        measures[name] = {'exit_code': code, 'wall_seconds_outer': time.perf_counter() - started,
                          'nic_received_bytes_machine_wide': _delta(_nic_bytes(), net_before)}
        if code:
            raise SystemExit(f'bench-{name} fallito con codice {code}')
    legacy = json.loads((bench / 'legacy_result.json').read_text())
    fast = json.loads((bench / 'fast_result.json').read_text())
    result = {'tiles': args.tiles, 'legacy': {**legacy, **measures['legacy']}, 'fast': {**fast, **measures['fast']},
              'speedup_wall': legacy['wall_seconds'] / fast['wall_seconds'],
              'equivalence': compare_tiles(bench / 'legacy', bench / 'fast'),
              'note': 'Byte di rete misurati sulla scheda: includono il traffico di qualunque altro processo.'}
    save_json(bench / 'bench_result.json', result)
    print(json.dumps({k: v for k, v in result.items() if k != 'equivalence'} | {'all_bitwise_equal': result['equivalence']['all_bitwise_equal']}, indent=2, default=str))
    return 0


def _nic_bytes():
    try:
        out = subprocess.check_output(['powershell', '-NoProfile', '-Command',
                                       '(Get-NetAdapterStatistics | Measure-Object -Property ReceivedBytes -Sum).Sum'], text=True)
        return int(out.strip())
    except Exception:
        return None


def _delta(after, before):
    return after - before if after is not None and before is not None else None


# psutil non è installato nel .venv di produzione: memoria del processo via API Windows.
def _rss_bytes():
    try:
        import ctypes
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [('cb', wintypes.DWORD), ('PageFaultCount', wintypes.DWORD),
                        ('PeakWorkingSetSize', ctypes.c_size_t), ('WorkingSetSize', ctypes.c_size_t),
                        ('QuotaPeakPagedPoolUsage', ctypes.c_size_t), ('QuotaPagedPoolUsage', ctypes.c_size_t),
                        ('QuotaPeakNonPagedPoolUsage', ctypes.c_size_t), ('QuotaNonPagedPoolUsage', ctypes.c_size_t),
                        ('PagefileUsage', ctypes.c_size_t), ('PeakPagefileUsage', ctypes.c_size_t)]
        counters = Counters()
        counters.cb = ctypes.sizeof(Counters)
        ctypes.windll.psapi.GetProcessMemoryInfo(ctypes.windll.kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb)
        return {'working_set': int(counters.WorkingSetSize), 'peak_working_set': int(counters.PeakWorkingSetSize)}
    except Exception:
        return None


if __name__ == '__main__':
    sys.exit(main())
