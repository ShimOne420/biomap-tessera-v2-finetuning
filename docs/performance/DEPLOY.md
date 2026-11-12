# Comandi (nessun deploy eseguito)

Tutti i comandi partono dalla worktree di ottimizzazione:

```powershell
cd C:\Users\brome\Desktop\BIOMAP_TESSERA_V2\biomap-v2-performance-dev
$py = "..\biomap-tessera-v2-finetuning\.venv\Scripts\python.exe"   # stesso interprete della produzione, usato in sola lettura
```

## 1. Test isolati (offline: nessuna rete, nessuna GPU)
```powershell
& $py scripts\dense_fast.py verify-synthetic
```
Atteso: sette punti (`.......`) e codice di uscita 0 (`$LASTEXITCODE`). Con la configurazione pytest del progetto la riga "7 passed" può non comparire.

## 2. Benchmark simulato locale (nessuna rete esterna, nessuna GPU)
```powershell
& $py -W ignore scripts\bench_synthetic.py all --root $env:TEMP\dense_bench_syn_$(Get-Date -Format yyyyMMddHHmm) --tile-px 200 --tiles-x 2 --tiles-y 2 --batch-pixels 30000
```

## 3. Piccolo benchmark reale (DA APPROVARE: rete e GPU in concorrenza con la campagna)
```powershell
& $py scripts\dense_fast.py bench --reference F:\biomap_dense_public\reference --bench-root F:\dense_bench_20261112 --checkpoint ..\data\weights\student_medium.pt --device cuda:0 --tiles belluno_dolomiti/grid_11.95_45.95 appennino_tosco_emiliano/grid_10.65_44.25 belluno_dolomiti/grid_11.95_46.25 --approved-network-benchmark
```
Confronto con la produzione, solo lettura:
```powershell
& $py scripts\dense_fast.py compare-tiles --left F:\biomap_dense_public\work --right F:\dense_bench_20261112\fast
```

## 4. Estrazione ottimizzata (DA APPROVARE; mai nella work-root r1)
```powershell
& $py -u scripts\dense_fast.py stream --reference F:\biomap_dense_public\reference --work-root F:\dense_fast_run --checkpoint ..\data\weights\student_medium.pt --device cuda:0 --io-workers 16 --batch-pixels 1000000 --prefetch 1
```
- **RAM:** circa (prefetch + 1) × batch-pixels × 3,5 kB, più le cache. Con i valori sopra sono circa 7 GB più 7,5 GB di cache.
- **Spazio:** tenere almeno 20 GB liberi su F:.
- **Protezione:** se `--work-root` è `F:\biomap_dense_public\work` (o una sua sottocartella), il programma si rifiuta di partire.

## 5. Ripresa dopo un'interruzione
Si rilancia lo **stesso comando** del punto 4. Vengono riusati:
- le tile complete (`chunks\…\tile.npz` con identità r2 e SHA256 verificato);
- i blocchi codificati (`tmp\…\encoded`);
- i giorni Sentinel già salvati (`tmp\…\days`).

Un'identità diversa (codice cambiato) produce un errore esplicito `Cache incompatibile/corrotta`, mai una sovrascrittura.

## 6. Monitoraggio (solo lettura)
```powershell
powershell -ExecutionPolicy Bypass -File scripts\watch_v2_dense_eta.ps1 -WorkRoot F:\dense_fast_run
& $py scripts\audit_dense_log.py --log F:\dense_fast_run\stream_log.jsonl
```
Nella r2 il log ha una riga per tile e `stream_batches.jsonl` una riga per lotto, con il tempo reale di acquisizione e le statistiche degli handle.
