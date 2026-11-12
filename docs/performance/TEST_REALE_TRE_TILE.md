# Test reale della pipeline r2-fast su tre tile già completate

**Eseguito il 12/11/2026 alle 15:09–15:16 (ora locale), a campagna r1 attiva, su autorizzazione dell'utente.** Esito: **identico bit per bit**. I tempi sono falsati dalla banda condivisa con la r1 (vedi "Risultati").

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

## Risultati (12/11/2026)
Il report completo è in `F:\dense_fast_check_20261112\compare_with_r1.json`.

**Equivalenza:** `all_bitwise_equal: true`. Per tutte e 3 le tile sono identici bit per bit:
- `cell`;
- `counts`;
- `sums`, cioè la somma float64 degli embedding a 128 dimensioni per cella; le celle sono 1, 1 e 36.

Gli embedding dei singoli pixel non sono confrontabili, perché la r1 li cancella a fine tile.

**Tempi:**

| | r1 (produzione, 11/11, da sola) | r2-fast (oggi, con la r1 attiva) |
|---|---|---|
| Acquisizione | 71 + 82 + 191 = **344 s**, in sequenza | **434 s**, lotto unico |
| Codifica | 0,5 + 1,0 + 4,1 = **5,6 s** | 0,8 + 0,8 + 3,8 = **5,3 s** |
| Totale | **circa 350 s** | **440 s** di sessione (446 s compreso l'avvio) |

**Lettura dei tempi:**
- Il test **non** dimostra un'accelerazione: a parità di rete condivisa la r2 è stata 1,26 volte più lenta.
- La r2 ha diviso i 100 Mbit/s con la campagna. Il tempo medio per blocco remoto (sommato sui thread) è stato 2,31 s contro 1,26 s misurati sulla r1 da sola, cioè circa il doppio.
- Correggendo per la banda dimezzata si otterrebbero circa 220 s, cioè circa 1,6 volte più veloce. È una **stima, non una misura**.
- Riuso degli handle: 1.110 aperture e 710 riusi.
- **Effetto sulla r1:** la tile `chianti_siena/grid_11.15_43.55` (18.465 pixel), finita durante il test, ha richiesto 281 s di acquisizione contro i 150 s della tile precedente da 24.648 pixel. Il test è costato alla campagna circa 2–4 minuti.

## Benchmark su una tile grande (preparato, NON eseguito: serve autorizzazione)
- **Tile:** `basilicata_interna/grid_16.15_40.55`, 965.181 pixel.
- **Riferimento r1:** 1.022 s di acquisizione e 355 s di codifica, in una sola sessione senza riprese.
- **Quando:** **solo a campagna ferma**, tra il passo 2 e il passo 5 di `PROCEDURA_CAVO_E_RIPRESA.md`. A campagna attiva il tempo non è interpretabile, come dimostra il test sopra. Durata a 100 Mbit/s: circa 20–25 minuti.
- **Come leggerlo:** misurato prima del cambio del cavo è il confronto pulito con la r1 a 100 Mbit/s; misurato dopo indica la velocità della r2 a 1 Gbit/s.

```powershell
$big = 'F:\dense_fast_check_big_20261112'
$bt  = 'basilicata_interna/grid_16.15_40.55'
cd $dev
& $py scripts\seed_catalogs.py --from F:\biomap_dense_public\work --to $big --tiles $bt
& $py -u scripts\dense_fast.py stream --reference F:\biomap_dense_public\reference --work-root $big --checkpoint C:\Users\brome\Desktop\BIOMAP_TESSERA_V2\data\weights\student_medium.pt --device cuda:0 --tiles $bt
& $py scripts\dense_fast.py compare-tiles --left F:\biomap_dense_public\work --right $big --tiles $bt | Tee-Object "$big\compare_with_r1.json"
```

**Criteri:**
- `all_bitwise_equal: true`;
- velocità = (1.022 + 355) / `right_sessions[0].wall_seconds`;
- in `$big\stream_log.jsonl` c'è anche `gpu_wait_before_batch_seconds`, che dice quanto la codifica si sovrappone all'acquisizione.

## Pulizia
La cartella `F:\dense_fast_check_20261112` contiene solo i risultati del test e i cataloghi copiati. Si può cancellare a mano quando non serve più.
