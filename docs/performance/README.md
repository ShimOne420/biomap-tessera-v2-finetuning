# Ottimizzazione dell'estrazione V2 densa: indice

Branch `optimization/v2-dense-performance`, worktree `..\biomap-v2-performance-dev`. Lavoro del 12/11/2026, svolto con la campagna r1 in corso.

| Documento | Contenuto |
|---|---|
| [AUDIT_V2_DENSE.md](AUDIT_V2_DENSE.md) | Colli di bottiglia misurati sulla campagna r1, in ordine di peso |
| [ACCELERAZIONE_CAMPAGNA_ATTUALE.md](ACCELERAZIONE_CAMPAGNA_ATTUALE.md) | Cosa si può fare senza interrompere la r1: rete a 100 Mbit/s, secondo worker, impostazioni |
| [EQUIVALENZA.md](EQUIVALENZA.md) | Verifiche bit per bit r1/r2 fatte e da fare |
| [BENCHMARK.md](BENCHMARK.md) | Benchmark simulato locale: da 1,3 a 2,6 volte, byte da 1,3 a 3,6 volte in meno |
| [ARCHITETTURA_V1_V2.md](ARCHITETTURA_V1_V2.md) | Stadi separati; mappe per pixel per la CNN |
| [DEPLOY.md](DEPLOY.md) | Comandi esatti: test, benchmark, estrazione, ripresa |
| [RISCHI.md](RISCHI.md) | Rischi e mitigazioni |
| `data/` | JSON dell'analisi e dei benchmark |

## File creati o modificati in questa worktree
- **`f5bcfd9`, snapshot legacy:** copia byte per byte del codice di produzione non versionato (`campaign/streaming.py`, `legacy_check.py`, `public_reference.py`, `uncertainty.py`, `cli.py`, `probing.py`, script e runbook). Gli SHA256 dei `CODE_FILES` coincidono con l'identità delle tile r1.
- **Nuovo pacchetto `src/biomap_v2/dense_fast/`:** `identity.py`, `io_safe.py`, `coords.py`, `handles.py`, `reader.py`, `acquisition.py`, `pipeline.py`, `spatial.py`.
- **Script:** `scripts/dense_fast.py` (`stream`, `compare-tiles`, `bench`, `verify-synthetic`), `scripts/bench_synthetic.py`, `scripts/audit_dense_log.py`.
- **Test:** `tests/test_dense_fast.py` (7 test offline).
- **Altro:** `.gitattributes` (LF per `*.py` e `*.ps1`) e questi documenti.
- **Nessun file legacy modificato.** Il codice r1 nella worktree è identico a quello di produzione.

## Conferma sulla produzione
Verificata alla fine del lavoro contro una fotografia presa all'inizio:
- `git status` e HEAD della worktree `biomap-tessera-v2-finetuning`: **identici**.
- SHA256 dei 41 file di codice (`src/biomap_v2/*.py`, `campaign/*.py`, `scripts/*.py`, `*.ps1`): **identici**.
- SHA256 dei 62 `tile.json` completati all'inizio: **identici**. Nel frattempo la campagna è arrivata a 71 tile.
- Il processo della campagna (PID 4356, avviato alle 10:33) è lo stesso e non è stato toccato.
- **Letture fatte in produzione:** codice sorgente, `stream_log.jsonl`, `stream_console.log` e `legacy_check.json` (aperti con condivisione completa), `tile.json` e `geometry.json`, contatori di rete, scheda di rete, `nvidia-smi` (3 letture), elenco dei processi.
- **Mai letto in ciclo:** `stream_progress.json`.
- **Scritture in `F:\biomap_dense_public`:** nessuna.
- **Unica operazione sul repository condiviso:** `git worktree add` (metadati in `.git\worktrees`) e i commit su questo branch.
