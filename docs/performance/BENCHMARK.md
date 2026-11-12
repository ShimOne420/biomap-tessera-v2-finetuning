# Benchmark r1 contro r2-fast

## Cosa è stato misurato e cosa no
- **Misurato:** un benchmark **simulato e completamente locale** (`scripts/bench_synthetic.py`).
  - Un server HTTP sul PC serve GeoTIFF sintetici, con latenza fissa per richiesta (40 ms) e banda globale limitata (100 Mbit/s come il collegamento reale, oppure 1 Gbit/s).
  - L'encoder è finto e deterministico, con un'attesa di 0,327 ms per pixel, pari alla GPU reale misurata.
  - r1 e r2 girano in **due processi separati**, così non condividono le cache GDAL.
  - Tempi reali con `perf_counter`, mai sommati sui thread. Richieste e byte contati dal server.
- **Non misurato:** il comportamento su Microsoft Planetary Computer (latenza vera, throttling, dimensioni reali dei COG) e la GPU vera. Serve `scripts\dense_fast.py bench ... --approved-network-benchmark`, che **non è stato eseguito**: usa rete e GPU in concorrenza con la campagna in corso.

## Risultati (dati in `data/synthetic_*.json`)
| Scenario | Rete | Lotto r2 (pixel) | r1 | r2 | Velocità | Byte r1/r2 | Richieste r1/r2 | Bit per bit |
|---|---|---|---|---|---|---|---|---|
| 6 tile piccole (2.520 pixel), 22 scene S2 + 20 S1 | 100 Mbit/s | 1.500.000 (un lotto) | 40,8 s | 15,5 s | **2,62×** | 3,59 | 2,39 | 6/6, parquet sì |
| 4 tile medie (28.000 pixel) | 100 Mbit/s | 1.500.000 (un lotto) | 103,1 s | 76,0 s | 1,36× | 1,62 | 1,55 | 4/4, parquet sì |
| 4 tile medie | 100 Mbit/s | 60.000 (2 tile) | 102,5 s | 65,1 s | 1,57× | 1,47 | — | 4/4, parquet sì |
| 4 tile medie | 100 Mbit/s | 30.000 (1 tile) | 102,5 s | 58,9 s | **1,74×** | 1,30 | — | 4/4, parquet sì |
| 4 tile medie | 1 Gbit/s | 1.500.000 | 55,5 s | 46,8 s | 1,19× | 1,62 | 1,55 | 4/4, parquet sì |
| 4 tile medie | 1 Gbit/s | 30.000 (1 tile) | 55,0 s | 41,4 s | 1,33× | 1,31 | — | 4/4, parquet sì |

## Lettura
- **Tile piccole:** il guadagno viene soprattutto dai byte non riscaricati e dalle richieste non ripetute. È il caso delle 125 tile sotto i 100.000 pixel della campagna reale.
- **Tile grandi:** il guadagno viene dalla sovrapposizione con la GPU. Se il lotto è troppo grande la GPU parte solo a lotto finito e il guadagno scende (1,36 contro 1,74 volte). Per questo il valore predefinito è `--batch-pixels 1.000.000`: le tile piccole viaggiano insieme, quelle grandi (circa 930.000 pixel) da sole.
- **A 1 Gbit/s il limite diventa la GPU:** 36,6 s di codifica simulata su 41,4 totali. Oltre questo serve accelerare l'encoder, ma con il vincolo FP32 e bit per bit c'è poco margine (vedi `RISCHI.md`).
- **Il collegamento di rete pesa quanto il codice:** la sola r1 passa da 103 a 55 s con 1 Gbit/s (1,86 volte).

## Stima per la campagna reale (da verificare)
Se i rapporti simulati valessero anche su Planetary Computer, un ricalcolo completo andrebbe da 1,7 a 2,6 volte più veloce a 100 Mbit/s. Combinando r2 e 1 Gbit/s, nello scenario medio la stima è circa 2,5 volte (103 s → 41 s). **Non è una promessa:** la latenza reale (1,26 s per blocco sommato sui thread, M) e i COG reali sono diversi da quelli simulati.

## Per misurare sul serio (dopo approvazione)
```powershell
cd C:\Users\brome\Desktop\BIOMAP_TESSERA_V2\biomap-v2-performance-dev
..\biomap-tessera-v2-finetuning\.venv\Scripts\python.exe scripts\dense_fast.py bench --reference F:\biomap_dense_public\reference --bench-root F:\dense_bench_20261112 --checkpoint ..\data\weights\student_medium.pt --device cuda:0 --tiles belluno_dolomiti/grid_11.95_45.95 appennino_tosco_emiliano/grid_10.65_44.25 belluno_dolomiti/grid_11.95_46.25 --approved-network-benchmark
```
Il comando produce `bench_result.json` con:
- tempo reale r1 e r2 (processi separati);
- velocità;
- memoria di picco e VRAM;
- byte della scheda di rete, che includono anche il traffico della campagna se è ancora attiva;
- confronto bit per bit delle tile.

Le cache sono fredde a ogni processo. Per la cache calda, rilanciare `bench-fast` sulla stessa `--bench-root` dopo aver cancellato solo `fast\chunks`.
