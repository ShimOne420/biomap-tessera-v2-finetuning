# Benchmark del campionamento spaziale (frozen) — dati reali del PC e compito per l'agente

Data: 18/11/2026. Autore: agente di esecuzione sul PC universitario.
Base: `feature/finetuning-v2` @7fee23d. Si aggiunge al piano "Spatial sampling benchmark & partial fine-tuning"
che l'utente ti ha consegnato e al brief `docs/development/FINETUNING_HANDOFF_PC.md` (che resta valido).
Tutti i numeri qui sotto sono **misurati sul PC** il 18/11/2026, salvo dove è scritto "stima".

## 1. Divisione del lavoro

- **Tu (Mac):** scrivi il codice, i test CPU su dati sintetici e il runbook PowerShell. Fai push su un branch nuovo
  (es. `feature/sampling-benchmark`) e apri una PR verso `feature/finetuning-v2`. Non serve accesso al PC.
- **Io (PC):** clono il tuo branch in una cartella nuova, eseguo test, smoke e benchmark sugli embedding reali
  e riporto i risultati. Né io né te avviamo acquisizioni Sentinel o fine-tuning dell'encoder in questa fase.

Il benchmark addestra **solo la Spatial CNN frozen** sugli embedding V2 già calcolati per ogni pixel.
Serve a scegliere il campionamento che poi useranno *sia* Frozen *sia* Partial.

## 2. Dati disponibili sul PC (passarli SEMPRE come argomenti CLI, nessun percorso cablato)

### 2.1 Embedding per pixel r2 (`--pixel-root`, sul PC `C:\BioMAP\v2_spatial_pixels_r2`)

- Protocollo `dense127909-stream-legacy-exact-r2-fast`, checkpoint SHA `3823be7d…`, fp32. Bit-identico a r1.
- `pixel_embeddings/<aoi>/<tile>.npy`: array `(N_pixel_validi_tile, 128)` float32, ordine righe =
  `geometry.npz flat_indices` del riferimento. Accanto c'è `<tile>.json` con `grid` (`width`, `height`, `transform`
  a 10 m, `crs` UTM) e SHA. 18 cartelle AOI, 273 tile, 48 GB in totale.
- `chunks/<aoi>/<tile>/tile.npz`: somme e conteggi per cella (dense mean). `v2.parquet`: medie per cella.
- **Lettura:** riusa `biomap_spatial.sources.V2Source(spec, output).patches(cell_id)` (stesso codice del run storico).
  Non reimplementare il lettore e non modificare `sources.py` / `support.py` (sono nell'identità dell'indice).

### 2.2 Indice spaziale (`--index-root`, sul PC `C:\BioMAP\spatial_runs\index`)

- `spatial_index.sqlite3`. Tabelle:
  - `meta(identity)`;
  - `tiles(tile)` con 273 righe;
  - `moments(cell_id, n, sums, squares)` con 132.900 righe;
  - `patches(id, cell_id, tile, row0, col0, height, width, mask BLOB bit-packed)` con 147.302 righe e indice su `cell_id`.
- `cells.parquet` ha queste colonne:
  - identità e posizione: `cell_id`, `lon`, `lat`, `west`, `south`, `east`, `north`;
  - split: `partition` (`train` / `test`), `aoi_id`, `cv_fold` (0/1/2; −1 = Liguria);
  - riferimento e target: `reference_index`, `target`;
  - supporto: `n_valid`, `n_total`, `valid_fraction`, `patches`, `duplicate_contributions`;
  - statistiche: `mean`, `variance`, `heterogeneity`.
- Identità indice `6d8d1d9f…`. Va aperto in **sola lettura**; non va ricostruito.

### 2.3 Geometria reale delle celle (decide quali configurazioni sono possibili)

| Grandezza | Valore |
|---|---|
| Contributi validi totali | 99.527.159 (sviluppo 95.981.208, Liguria 3.545.951) |
| `n_valid` per cella | mediana 713, p5 660, p95 981, p99 1.695, max 5.188 |
| Patch per cella | 1: 118.827 · 2: 13.796 · 3: 225 · 4: 52 |
| Bbox patch (righe × colonne) | mediana 31 × 23 px; min 1 × 1; max 63 × 50 |

Conseguenze da gestire nel campionatore:

1. **La cella tipica è circa 31 × 23 pixel.** 12×12, 2×8×8 e 16×16 ci stanno quasi sempre. La cella completa ha circa il 72% di posizioni
   con campo ricettivo 5×5 interno, una patch 8×8 il 25% e una 12×12 il 44%.
2. **Celle con 2 o più patch.** Sono quasi sempre celle coperte da **due tile sovrapposte**, oppure a cavallo di due zone UTM (le 107 di
   `belluno_dolomiti`). In molti casi due patch descrivono **lo stesso terreno**.
   - Per `multi_2x8x8` e `multi_4x4x4` le patch vanno distribuite **nello spazio geografico** della cella, non "una per patch
     dell'indice". Altrimenti due finestre possono vedere lo stesso terreno.
   - Il `center_windows` esistente sceglie una finestra **per ogni tile**: su queste celle raddoppia il supporto. Non va usato come
     modello per le nuove modalità.
3. **Patch piccole o maschere irregolari.** Ai bordi delle AOI e sull'acqua può non esistere una finestra interamente valida.
   - Regola: la finestra resta sulla griglia nativa ed è ammessa con maschera parziale; si registra il numero di pixel validi.
   - Si definisce un minimo (parametro) sotto il quale la cella è marcata `insufficient`, contata e riportata. **Mai eliminata in
     silenzio.**
   - La valutazione usa comunque il supporto completo per tutte le celle.

### 2.4 Fold

| `cv_fold` | AOI | Celle |
|---|---|---|
| 0 | appennino_tosco_emiliano, basilicata_interna, sardegna_interna, sicilia_interna, trentino_alpi | 38.808 |
| 1 | abruzzo_gran_sasso, belluno_dolomiti, campania_pilot, puglia_tavoliere, sardegna_costiera, sicilia_etna | 44.895 |
| 2 | calabria_sila, chianti_siena, emilia_romagna_bologna, friuli_prealpi, roma_periurbano, vercelli_novara | 44.206 |
| −1 | liguria_levante (**esclusa dal benchmark**) | 4.991 |

## 3. Riferimento storico (run `v2_spatial3_seed42`, supporto completo)

Configurazione del run, da riusare identica:
- modello `spatial_cnn`: input 128, `conv_channels` [64, 32], kernel 3, hidden 64, dropout 0.1, output lineare,
  94.561 parametri;
- normalizzazione `pixel_weighted_standard`;
- training: `max_patch_side` 64, 50 epoche massime, patience 8, `batch_cells` 4, `accumulate_cells` 16, lr 1e-3,
  weight decay 1e-4, clip 1.0, seed 42.

| Fold | RMSE macro AOI | MAE macro AOI | Epoca migliore |
|---|---|---|---|
| 0 | 0.17541 | 0.13286 | 2 |
| 1 | 0.18883 | 0.14414 | 8 |
| 2 | 0.16371 | 0.11719 | 2 |
| Tutti (17 AOI) | 0.17602 | 0.13131 | refit 2 |

**Il numero da confrontare in uno screening a un fold è quello del fold, non 0.17602.**
Le metriche per AOI sono in `oof_by_aoi.csv` del run; le predizioni in `predictions.parquet` (colonne `cell_id`, `aoi_id`,
`cv_fold`, `target`, `prediction`, `model` ∈ {`predictor`, `dummy_mean`}, `fold`, …).

Due fatti che il benchmark deve tenere in conto:

1. **Il risultato storico è ottimista.** Il run ha scelto l'epoca con `selection: oof_macro_aoi_rmse`, cioè sul fold di validazione stesso.
2. **La validazione è molto rumorosa tra un'epoca e l'altra.** Fold 2: 0.1836 all'epoca 9 e 0.1699 all'epoca 10, cioè Δ≈0.014.
   Differenze tra configurazioni sotto ~0.01 su un solo fold e un solo seed **non sono interpretabili** senza una misura del rumore.

**Prestazioni del run storico:**
- ogni epoca dura 520–590 s per fold, compresa la validazione densa;
- picco GPU circa 200 MB.

Il collo di bottiglia è quindi la lettura dei 48 GB di r2, non la GPU.

## 4. Cosa costruire

### 4.1 Modalità di campionamento (estendere `biomap_finetune/data.py`, senza cambiare le modalità esistenti)

`single_8x8`, `multi_2x8x8`, `single_12x12`, `multi_4x4x4`, `full_cell_subset` (tutti i pixel di un sottoinsieme
di circa 11.000 celle di sviluppo, stratificato per AOI), più `full` = riferimento.

Requisiti comuni a tutte le modalità:
- **Determinismo:** seed esplicito (proposta: 20260925, come il resto del framework) e hash stabile delle posizioni
  (`rank_positions` esistente). Indipendenti dal target.
- **Geometria:** griglia nativa, nessun pixel fuori dalla cella, nessun mosaico tra tile o CRS.
- **Tracciabilità:** coordinate registrate e un manifest con SHA del piano.

Per `full_cell_subset` la popolazione di training cambia: va riportata come esperimento distinto.

### 4.2 Cache degli embedding campionati (indispensabile per i tempi)

- Per ogni configurazione e per le celle di training del fold: estrai **una volta** dagli `.npy` r2 i pixel scelti, con maschere e
  posizioni, in un file compatto. La scrittura deve essere riprendibile e atomica.
- **Stima dimensione:** 512 byte/pixel × pixel.
  - 64 px/cella su 128 mila celle ≈ 4,2 GB;
  - 144 px/cella ≈ 9,4 GB;
  - `full_cell_subset` ≈ 4–5 GB.
- Il training legge solo la cache: niente rilettura dei 48 GB a ogni epoca.
- La **valutazione** usa invece il supporto completo delle celle del fold di validazione, tramite `V2Source` e lo stesso `collate_cells`.
  Una sola passata densa per checkpoint scelto, non a ogni epoca.

### 4.3 Training e selezione

- **Modello:** CNN e aggregazione identiche a quelle storiche (`biomap_spatial.model.SpatialCNN`, pooling pesato per pixel validi,
  una predizione e una loss MSE per cella).
- **Normalizzazione:** stimata sul supporto di training campionato di quella configurazione, e registrata.
- **Early stopping su validazione geografica INTERNA:**
  - tenere fuori 1 o 2 AOI di training del fold, con regola dichiarata;
  - la metrica di validazione è calcolata sul supporto completo di quelle AOI;
  - il fold esterno non sceglie nulla.
- **Controlli obbligatori, sullo stesso fold e con lo stesso codice:**
  1. `full` riaddestrato con la selezione interna, con 2 seed. Verifica che il codice nuovo riproduca il risultato storico e misura il
     rumore tra seed.
  2. Ogni configurazione candidata, con almeno lo stesso numero di seed se il budget lo consente.
- **Regola di decisione:** va scritta nella config **prima** del run e la fissa l'utente. Proposta: la configurazione più economica in
  disco la cui differenza di RMSE macro rispetto a `full` resta entro il rumore tra seed misurato al punto 1.

### 4.4 Fold e budget dello screening

- **Fold proposto:** fold 0 (5 AOI, 38.808 celle di validazione, 89.101 di training). È un parametro CLI.
- **Budget obiettivo:** l'intero screening (circa 7 addestramenti, più cache e valutazioni dense) entro circa 12 ore sul PC. Il runbook
  deve indicare una stima per passo e un dry-run che stampi questa stima prima di partire.

### 4.5 Report

Per ogni configurazione:
- **Metriche:** RMSE e MAE macro AOI; RMSE pooled, R², bias, Pearson; quantili p50/p90/p95/p99 dell'errore assoluto; % celle con
  errore > 0.3 e > 0.4. Separate per AOI e per fascia di MSA: [0, .1), [.1, .2), [.2, .4), [.4, .6), [.6, .8), [.8, 1].
- **Copertura del campionamento:** pixel selezionati e validi, % di posizioni con RF 5×5 completo, distanza tra le patch, celle
  `insufficient`.
- **Costi:** tempo per epoca, VRAM, dimensione della cache.
- **Costo raw stimato** per il fine-tuning sulle celle di sviluppo, con 2.956 byte/pixel come stima da sostituire con i cataloghi reali
  (vedi `estimates()`):

| Configurazione | Pixel sviluppo (max) | Raw stimati |
|---|---|---|
| single_8x8 / multi_4x4x4 | 8,19 M | ~24 GB |
| multi_2x8x8 | 16,37 M | ~48 GB |
| single_12x12 | 18,42 M | ~54 GB |
| full | 95,98 M | ~284 GB |

- **Confronto con `full`:** bootstrap appaiato a cluster di AOI. Riusa `bootstrap_delta` di `biomap_v2.dense.evaluation` (2000 ricampionamenti,
  seed 20260925) per restare coerente con i confronti già fatti.

## 5. Vincoli PC (misurati il 18/11/2026)

- **Spazio libero:** C: 57 GB, F: 52 GB. Sono dischi separati: **non sommarli**. Il run storico usava `reserve_gib: 20`.
- Con questi spazi `multi_2x8x8` e `single_12x12` **non entrano** su un singolo disco per i raw del fine-tuning senza liberare spazio.
  Il report deve dirlo in chiaro. La cache degli embedding del benchmark invece ci sta.
- La cache dell'uscita dei blocchi congelati è già stata valutata come non praticabile (`CACHE_FEASIBILITY.md`, `estimates()`:
  centinaia di KB/pixel). Non riproporla.
- **Ambiente:** Windows 11, Python 3.12.10, torch 2.10.0+cu128, RTX 5080 16 GB, 31 GB RAM.
  - DataLoader con `spawn`.
  - `num_workers` configurabile (storico 4).
  - Niente `fork`.
  - Niente percorsi POSIX.
- **Contratto di esecuzione** (come da handoff):
  - una sola CLI con `--dry-run`;
  - output in una cartella nuova che non sovrascrive mai;
  - ripresa atomica;
  - `progress.json` aggiornato durante l'epoca, con `cells_done`, `cells_total`, `cells_per_second`, `eta_current_epoch_seconds`,
    `phase`, `fold`, `epoch`;
  - smoke reale sotto 15 minuti, su un'AOI e poche celle;
  - intestazione dettagliata in italiano in ogni script nuovo o modificato.

## 6. Cosa non fare

- Non usare la Liguria (`cv_fold == -1`) in alcuna fase del benchmark.
- Non modificare:
  - `biomap_spatial` (modello, `sources.py`, `support.py`);
  - i 5 CODE_FILES r1;
  - i run e gli output storici.
- Non ricostruire l'indice spaziale.
- Non mettere in git dati, embedding, checkpoint, IP o dettagli di rete: il repository è pubblico.
- Non decidere il numero di blocchi allenabili (1 o 2): è una decisione separata dell'utente, successiva al benchmark.
- Non dichiarare test superati che non hai eseguito. Le parti che richiedono il PC vanno marcate "da verificare sul PC".

## 7. Cosa mi serve nella PR

1. Elenco esatto dei comandi del runbook, in ordine:
   - test CPU;
   - dry-run;
   - smoke;
   - cache;
   - controllo `full` × 2 seed;
   - candidati;
   - report.
2. Per ogni comando: output atteso, durata stimata e come riprendere dopo un'interruzione.
3. Numero di test eseguiti, passati e saltati sul Mac, con il commit.
4. Elenco dei file nuovi e modificati, con la conferma che i file protetti non sono cambiati (fixture SHA esistente).
