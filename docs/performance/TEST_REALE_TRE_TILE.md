# Test reale della pipeline r2-fast su tre tile già completate

**Preparato, non eseguito.** Richiede la tua approvazione: usa rete verso Planetary Computer e, per pochi secondi, la GPU.

## Tile scelte
Tre tile vicine di `belluno_dolomiti`, già complete in produzione e con il catalogo STAC salvato:

| Tile | Pixel | Tempo r1 in produzione (acquisizione) |
|---|---|---|
| `grid_11.95_45.95` | 8 | 71 s |
| `grid_11.95_46.25` | 1.266 | 82 s |
| `grid_12.05_45.95` | 11.168 | 191 s |

Insieme fanno 12.442 pixel. Stanno in un solo lotto r2, quindi provano anche la condivisione dei byte tra tile vicine. La r1 le ha acquisite in 344 s complessivi, una dopo l'altra.

## Separazione dalla produzione
- **Cartella di lavoro:** `F:\dense_fast_check_20261112`, fuori da `F:\biomap_dense_public`. La pipeline rifiuta la work-root di produzione.
- **Cataloghi:** copiati dalla produzione con lettura condivisa (`seed_catalogs.py`), così le scene sono **le stesse** usate dalla r1 e il confronto bit per bit è significativo.
- **Confronto:** legge `tile.npz` e `tile.json` delle tre tile, scritti una volta sola e mai più sostituiti. Non apre `stream_progress.json`.
- **Scritture:** nessuna in produzione.

## Quando eseguirlo
- **Ideale:** a campagna ferma, tra il passo 2 e il passo 5 di `PROCEDURA_CAVO_E_RIPRESA.md`. Non ruba banda alla r1 e il tempo misurato è pulito. Durata stimata: pochi minuti.
- **In alternativa, a campagna attiva:** si dividono i 100 Mbit/s. Il risultato di equivalenza resta valido, i tempi diventano rumorosi.

## Comandi
```powershell
$prod = 'C:\Users\brome\Desktop\BIOMAP_TESSERA_V2\biomap-tessera-v2-finetuning'
$dev  = 'C:\Users\brome\Desktop\BIOMAP_TESSERA_V2\biomap-v2-performance-dev'
$py   = "$prod\.venv\Scripts\python.exe"
$test = 'F:\dense_fast_check_20261112'
$tiles = 'belluno_dolomiti/grid_11.95_45.95','belluno_dolomiti/grid_11.95_46.25','belluno_dolomiti/grid_12.05_45.95'
cd $dev

# 1. Cataloghi identici a quelli della produzione (lettura condivisa, scrittura solo in $test)
& $py scripts\seed_catalogs.py --from F:\biomap_dense_public\work --to $test --tiles $tiles

# 2. Estrazione r2 delle tre tile (stessi chunk 4096 e batch 256 della produzione, FP32)
& $py -u scripts\dense_fast.py stream --reference F:\biomap_dense_public\reference --work-root $test --checkpoint C:\Users\brome\Desktop\BIOMAP_TESSERA_V2\data\weights\student_medium.pt --device cuda:0 --tiles $tiles

# 3. Confronto bit per bit con la produzione (solo lettura su entrambe) e tempi
& $py scripts\dense_fast.py compare-tiles --left F:\biomap_dense_public\work --right $test --tiles $tiles | Tee-Object "$test\compare_with_r1.json"
```

## Criteri di esito
- **Equivalenza:** `"all_bitwise_equal": true`, cioè `cell`, `counts` e `sums` identici per tutte e 3 le tile. Sono attese anche le somme identiche: stessa GPU, stesso batch, stessi chunk, stesse scene.
- **Tempo:**
  - `right_sessions[0].wall_seconds` (r2, le tre tile insieme) contro la somma di `left_timings` (r1, circa 344 s di acquisizione più la codifica);
  - in `F:\dense_fast_check_20261112\stream_batches.jsonl` ci sono i riusi degli handle e il tempo reale di acquisizione del lotto.
- **Se l'equivalenza non è bit per bit:** riportare `sums_max_abs_diff` e `mean_max_abs_diff` e **non** usare la r2 finché la causa non è chiarita. Le cause possibili sono asset MPC rigenerati dopo l'11/11 oppure una differenza reale da indagare. La tolleranza non si allarga.

## Pulizia
La cartella `F:\dense_fast_check_20261112` contiene solo i risultati del test e i cataloghi copiati. Si può cancellare a mano quando non serve più.
