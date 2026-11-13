# SCOPO
#   Orchestratore unico e riprendibile del run V2 spatial completo (17 AOI di sviluppo + Liguria, 132.900 celle):
#   estrazione degli embedding V2 per pixel con r2-fast, verifica contro r1, indice spaziale, training in
#   parallelo, selezione, refit e valutazione. Ogni fase già completa viene saltata al rilancio.
# COME VIENE RICHIAMATO (dalla radice del clone, con l'interprete del run)
#   C:\BioMAP\venv-v2spatial\Scripts\python.exe experiments\v2_spatial_cnn_msa\run_pipeline.py
#       --work-root C:\BioMAP\v2_spatial_msa --v2-root C:\BioMAP\v2_spatial_work\dense_r2
#       --v1-root F:\biomap_dense_public\v1_tiles --reference F:\biomap_dense_public\reference
#       --r1-work F:\biomap_dense_public\work --checkpoint <...>\data\weights\student_medium.pt
#       [--run-name full_seed42] [--set training.mixed_precision=true] [--until build]
#   Monitor in un altro terminale: python experiments\v2_spatial_cnn_msa\monitor_v2.py --work-root ... --v2-root ... --follow
# LOGICA IN ORDINE
#   1. extract: scripts/dense_fast.py stream --keep-pixel-embeddings nella v2-root (mai la work-root r1);
#      rilancio automatico dopo errori (ripresa per tile r2) e riciclo se la memoria privata supera il limite;
#      ci si ferma dopo troppi rilanci o tre rilanci consecutivi senza tile nuove.
#   2. verify_r1: medie per cella di r2 (v2.parquet) contro quelle della campagna r1, cella per cella:
#      uguali bit per bit -> OK; differenza massima <= 1e-5 -> OK con avviso registrato; oltre -> STOP.
#   3. build: patch_builder.py build (patch/maschere dell'Esp. B, valori V2, SHA256 di ogni file V2).
#   4. preflight --require-training; 5. profile (tempi/VRAM per batch, nessun cambio automatico di batch).
#   6. cv: 3 processi CNN (un fold ciascuno) + 1 processo per MLP/HistGB, in parallelo sulla stessa GPU;
#      job falliti rilanciati con --resume (max 3). 7. select: selection.json in un solo processo.
#   8. refit: un processo per modello in parallelo; 9. finalize_refit; 10. evaluate: OOF e Liguria.
# INPUT E OUTPUT
#   Input: percorsi sopra (V1, reference e r1 in sola lettura). Output: v2-root (embedding) e work-root
#   (indice, run, log in logs\, pipeline.json con fase, tempi e dettagli per il monitor).
# PARAMETRI
#   --memory-limit-gb (24): memoria privata oltre cui l'estrazione viene riciclata (RAM fisica 31 GB).
#   --max-restarts (30), --io-workers (16), --cnn-jobs-parallel (3), --until <fase>, --set (al training).
# RISORSE
#   Estrazione: rete + GPU + ~10 GB RAM; training: GPU (4 processi) + CPU (worker di lettura) + NVMe.
# RIPRESA
#   Rilanciare lo stesso comando: fasi complete saltate, r2 riprende per tile, training da last.pt.
import argparse
import ctypes
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
STAGES = ['extract', 'verify_r1', 'build', 'preflight', 'profile', 'cv', 'select', 'refit', 'finalize_refit', 'evaluate']
R2_PROTOCOL = 'dense127909-stream-legacy-exact-r2-fast'
CNN_FOLDS = ['spatial_cnn:0', 'spatial_cnn:1', 'spatial_cnn:2']
MEAN_CV = 'mean_mlp:0,mean_mlp:1,mean_mlp:2,mean_histgb:0,mean_histgb:1,mean_histgb:2'
REFITS = ['spatial_cnn:refit', 'mean_mlp:refit', 'mean_histgb:refit']


def now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str), encoding='utf-8')
    for _ in range(40):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            time.sleep(.05)


class Pipeline:
    def __init__(self, args):
        self.args = args
        self.work = Path(args.work_root)
        self.path = self.work / 'pipeline.json'
        old = json.loads(self.path.read_text(encoding='utf-8')) if self.path.exists() else {}
        stages = old.get('stages', {})
        self.state = {'status': 'running', 'stage': None, 'pid': os.getpid(), 'started_utc': now(),
                      'run_name': args.run_name, 'v2_root': args.v2_root, 'work_root': args.work_root,
                      'stages': {s: stages.get(s, {'status': 'pending'}) for s in STAGES}, 'history': old.get('history', [])}
        self.state['history'].append({'started_utc': self.state['started_utc'], 'pid': os.getpid(), 'argv': sys.argv[1:]})
        self.save()

    def save(self):
        self.state['updated_utc'] = now()
        write_json(self.path, self.state)

    def begin(self, stage):
        info = self.state['stages'][stage]
        info.update(status='running', started_utc=info.get('started_utc') or now())
        self.state['stage'] = stage
        self.save()

    def detail(self, stage, **values):
        self.state['stages'][stage].update(values)
        self.save()

    def done(self, stage, **values):
        self.state['stages'][stage].update(status='complete', finished_utc=now(), **values)
        self.save()

    def fail(self, stage, error):
        self.state['stages'][stage].update(status='failed', error=error, failed_utc=now())
        self.state.update(status='failed', error=f'{stage}: {error}', finished_utc=now())
        self.save()


class MEMORY(ctypes.Structure):
    _fields_ = [('cb', ctypes.c_ulong), ('PageFaultCount', ctypes.c_ulong), ('PeakWorkingSetSize', ctypes.c_size_t),
                ('WorkingSetSize', ctypes.c_size_t), ('QuotaPeakPagedPoolUsage', ctypes.c_size_t),
                ('QuotaPagedPoolUsage', ctypes.c_size_t), ('QuotaPeakNonPagedPoolUsage', ctypes.c_size_t),
                ('QuotaNonPagedPoolUsage', ctypes.c_size_t), ('PagefileUsage', ctypes.c_size_t),
                ('PeakPagefileUsage', ctypes.c_size_t), ('PrivateUsage', ctypes.c_size_t)]


def private_gb(process):
    # Memoria privata (commit) del processo figlio, tramite l'handle già aperto da subprocess.
    if os.name != 'nt':
        return None
    m = MEMORY()
    m.cb = ctypes.sizeof(MEMORY)
    ok = ctypes.windll.psapi.GetProcessMemoryInfo(ctypes.c_void_p(int(process._handle)), ctypes.byref(m), m.cb)
    return m.PrivateUsage / 1024 ** 3 if ok else None


def kill_tree(process):
    if os.name == 'nt':
        subprocess.run(['taskkill', '/PID', str(process.pid), '/T', '/F'], capture_output=True)
    else:
        process.kill()
    process.wait()


def child_env(gpu=True):
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1', HF_HUB_OFFLINE='1')
    if gpu:
        env.pop('CUDA_VISIBLE_DEVICES', None)
    return env


def launch(cmd, log, gpu=True):
    log = Path(log)
    log.parent.mkdir(parents=True, exist_ok=True)
    stream = log.open('a', encoding='utf-8')
    stream.write(f'\n===== {now()} {" ".join(map(str, cmd))}\n')
    stream.flush()
    flags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == 'nt' else 0
    process = subprocess.Popen([str(c) for c in cmd], stdout=stream, stderr=subprocess.STDOUT, cwd=str(REPO),
                               env=child_env(gpu), creationflags=flags)
    process.log_stream = stream
    return process


def extraction_complete(root):
    marker = Path(root) / 'extraction_complete.json'
    if not marker.exists():
        return False
    value = json.loads(marker.read_text(encoding='utf-8'))
    return value.get('method') == R2_PROTOCOL and value.get('complete_campaign') is True


def count_tiles(root):
    return sum(1 for _ in (Path(root) / 'pixel_embeddings').glob('*/*.npy'))


def stage_extract(p, a):
    if extraction_complete(a.v2_root):
        return {'note': 'già completa'}
    cmd = [sys.executable, REPO / 'scripts' / 'dense_fast.py', 'stream', '--reference', a.reference,
           '--work-root', a.v2_root, '--checkpoint', a.checkpoint, '--device', 'cuda:0',
           '--keep-pixel-embeddings', '--io-workers', a.io_workers]
    restarts = stall = 0
    while True:
        before = count_tiles(a.v2_root)
        process = launch(cmd, Path(a.work_root) / 'logs' / 'extract.log')
        reason = None
        peak = 0.
        while process.poll() is None:
            time.sleep(15)
            mem = private_gb(process)
            if mem is not None:
                peak = max(peak, mem)
                p.detail('extract', child_pid=process.pid, private_gb=round(mem, 2), peak_private_gb=round(peak, 2),
                         restarts=restarts, tiles_with_pixels=count_tiles(a.v2_root))
                if mem > a.memory_limit_gb:
                    reason = f'memoria privata {mem:.1f} GB > {a.memory_limit_gb} GB: riciclo'
                    kill_tree(process)
        process.log_stream.close()
        if extraction_complete(a.v2_root):
            return {'restarts': restarts, 'tiles_with_pixels': count_tiles(a.v2_root)}
        after = count_tiles(a.v2_root)
        stall = stall + 1 if after == before else 0
        restarts += 1
        reason = reason or f'uscita {process.returncode}'
        p.detail('extract', restarts=restarts, last_restart_reason=reason, last_restart_utc=now())
        if restarts > a.max_restarts:
            raise RuntimeError(f'rilanci esauriti ({restarts}); ultimo motivo: {reason}; vedere logs/extract.log')
        if stall >= 3:
            raise RuntimeError(f'tre rilanci consecutivi senza tile nuove; ultimo motivo: {reason}')
        time.sleep(60)


def stage_verify(p, a):
    import numpy as np
    import pandas as pd
    r1_marker = Path(a.r1_work) / 'extraction_complete.json'
    if not r1_marker.exists():
        raise RuntimeError('Campagna r1 non completa: confronto impossibile')
    left = pd.read_parquet(Path(a.r1_work) / 'v2.parquet')
    right = pd.read_parquet(Path(a.v2_root) / 'v2.parquet')
    if 'cell_id' not in left or 'cell_id' not in right:
        raise RuntimeError('v2.parquet senza cell_id')
    left = left.set_index(left.cell_id.astype(str)).sort_index()
    right = right.set_index(right.cell_id.astype(str)).sort_index()
    if not left.index.equals(right.index):
        raise RuntimeError(f'Celle diverse tra r1 ({len(left)}) e r2 ({len(right)})')
    columns = [c for c in left.columns if c in right.columns and c != 'cell_id' and np.issubdtype(left[c].dtype, np.number)]
    a1 = left[columns].to_numpy(dtype=np.float64)
    a2 = right[columns].to_numpy(dtype=np.float64)
    diff = np.abs(a1 - a2)
    exact = bool(np.array_equal(a1, a2, equal_nan=True))
    report = {'cells': len(left), 'columns': len(columns), 'bitwise_equal': exact,
              'max_abs_diff': float(np.nanmax(diff)) if diff.size else 0., 'cells_different': int((diff > 0).any(axis=1).sum()),
              'pixel_tiles': count_tiles(a.v2_root)}
    write_json(Path(a.work_root) / 'verify_r1.json', report)
    if not exact and report['max_abs_diff'] > 1e-5:
        raise RuntimeError(f'r2 diversa da r1 oltre la soglia: {report}')
    if not exact:
        report['warning'] = 'differenza numerica minima rispetto a r1, registrata'
    return report


def experiment(script, a, *extra, status='status.json', lock='experiment.lock'):
    base = [sys.executable, HERE / script]
    common = ['--work-root', a.work_root, '--v1-root', a.v1_root, '--v2-root', a.v2_root, '--reference', a.reference,
              '--preset', a.preset, '--status-name', status, '--lock-name', lock]
    for item in a.set:
        common += ['--set', item]
    return base + list(extra) + common


def run_one(p, stage, cmd, log, attempts=1):
    for attempt in range(1, attempts + 1):
        process = launch(cmd, Path(p.args.work_root) / 'logs' / log)
        p.detail(stage, child_pid=process.pid, attempt=attempt)
        code = process.wait()
        process.log_stream.close()
        if code == 0:
            return
        if attempt < attempts:
            time.sleep(30)
    raise RuntimeError(f'{log}: uscita {code}; vedere logs/{log}')


def run_parallel(p, stage, jobs, phase, attempts=3):
    # jobs: lista di (nome, stringa --jobs). Il primo parte da solo finché run.json non esiste (evita scritture
    # concorrenti dello stesso file), poi partono gli altri. Ogni job fallito viene rilanciato con --resume.
    a = p.args
    run_json = Path(a.work_root) / 'runs' / a.preset / a.run_name / 'run.json'

    def start(name, spec):
        cmd = experiment('train.py', a, '--run-name', a.run_name, '--phase', phase, '--jobs', spec, '--resume',
                         status=f'status_{name}.json', lock=f'job_{name}.lock')
        return launch(cmd, Path(a.work_root) / 'logs' / f'{name}.log')

    running, tries = {}, {name: 1 for name, _ in jobs}
    first, rest = jobs[0], jobs[1:]
    running[first[0]] = (start(*first), first[1])
    while not run_json.exists() and running[first[0]][0].poll() is None:
        time.sleep(5)
    for name, spec in rest:
        running[name] = (start(name, spec), spec)
        time.sleep(3)
    failures = {}
    while running:
        time.sleep(10)
        p.detail(stage, jobs={n: {'pid': pr.pid, 'try': tries[n]} for n, (pr, _) in running.items()}, failures=failures)
        for name, (process, spec) in list(running.items()):
            code = process.poll()
            if code is None:
                continue
            process.log_stream.close()
            del running[name]
            if code != 0:
                failures[name] = failures.get(name, 0) + 1
                if tries[name] >= attempts:
                    for other, _ in running.values():
                        kill_tree(other)
                    raise RuntimeError(f'job {name} fallito {tries[name]} volte; vedere logs/{name}.log')
                tries[name] += 1
                time.sleep(30)
                running[name] = (start(name, spec), spec)
    return {'jobs': [n for n, _ in jobs], 'failures': failures}


def main():
    ap = argparse.ArgumentParser(description='Pipeline completa V2 spatial (estrazione -> training -> valutazione)')
    ap.add_argument('--work-root', required=True)
    ap.add_argument('--v2-root', required=True)
    ap.add_argument('--v1-root', required=True)
    ap.add_argument('--reference', required=True)
    ap.add_argument('--r1-work', required=True)
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--run-name', default='full_seed42')
    ap.add_argument('--preset', default='full', choices=['pilot', 'intermediate', 'full'])
    ap.add_argument('--io-workers', type=int, default=16)
    ap.add_argument('--memory-limit-gb', type=float, default=24)
    ap.add_argument('--max-restarts', type=int, default=30)
    ap.add_argument('--set', action='append', default=[])
    ap.add_argument('--until', choices=STAGES, help='fermarsi dopo questa fase')
    a = ap.parse_args()
    if Path(a.work_root).name != 'v2_spatial_msa':
        ap.error('--work-root deve terminare con v2_spatial_msa')
    if Path(a.v2_root).resolve() == Path(a.r1_work).resolve():
        ap.error('--v2-root non può essere la work-root della campagna r1')
    p = Pipeline(a)
    plan = {
        'extract': lambda: stage_extract(p, a),
        'verify_r1': lambda: stage_verify(p, a),
        'build': lambda: run_one(p, 'build', experiment('patch_builder.py', a, 'build', status='status_build.json'), 'build.log'),
        'preflight': lambda: run_one(p, 'preflight', experiment('preflight.py', a, '--require-training', status='status_preflight.json'), 'preflight.log'),
        'profile': lambda: run_one(p, 'profile', experiment('train.py', a, '--profile', status='status_profile.json'), 'profile.log'),
        'cv': lambda: run_parallel(p, 'cv', [(f'cnn{i}', spec) for i, spec in enumerate(CNN_FOLDS)] + [('mean_cv', MEAN_CV)], 'cv'),
        'select': lambda: run_one(p, 'select', experiment('train.py', a, '--run-name', a.run_name, '--phase', 'cv', status='status_select.json'), 'select.log'),
        'refit': lambda: run_parallel(p, 'refit', [('refit_cnn', REFITS[0]), ('refit_mlp', REFITS[1]), ('refit_histgb', REFITS[2])], 'refit'),
        'finalize_refit': lambda: run_one(p, 'finalize_refit', experiment('train.py', a, '--run-name', a.run_name, '--phase', 'refit', status='status_finalize.json'), 'finalize_refit.log'),
        'evaluate': lambda: (run_one(p, 'evaluate', experiment('evaluate.py', a, 'oof', '--run-name', a.run_name, status='status_eval_oof.json'), 'evaluate.log'),
                             run_one(p, 'evaluate', experiment('evaluate.py', a, 'liguria', '--run-name', a.run_name, status='status_eval_liguria.json'), 'evaluate.log')),
    }
    for stage in STAGES:
        if p.state['stages'][stage].get('status') == 'complete':
            continue
        p.begin(stage)
        try:
            result = plan[stage]()
            p.done(stage, result=result if isinstance(result, dict) else None)
        except BaseException as error:
            p.fail(stage, f'{type(error).__name__}: {error}')
            raise
        if a.until == stage:
            p.state.update(status='paused', paused_after=stage)
            p.save()
            return
    p.state.update(status='complete', finished_utc=now(), stage=None)
    p.save()
    print('Pipeline completa:', Path(a.work_root) / 'runs' / a.preset / a.run_name)


if __name__ == '__main__':
    main()
