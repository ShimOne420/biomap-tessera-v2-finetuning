# Equivalenza numerica r2-fast contro r1

## Principio
La r2 non cambia **nessuna regola** del protocollo storico:
- catalogo STAC per tile sul bbox dei bordi dei pixel;
- ordine delle scene per (copertura nuvolosa, id);
- SCL al pixel con classi non valide {0,1,2,3,8,9};
- bande S2 bilineari sul reticolo V1, senza nodata, troncate a uint16;
- S1 in 20·log10, +50, ·200, int16, con riempimento separato di VV e VH;
- DOY reale;
- encoder Medium con SHA256 verificato, in FP32 senza AMP, a blocchi di 4.096 pixel e batch 256;
- somme per cella in float64 blocco per blocco;
- aggregazione finale nell'ordine del piano.

Cambia solo l'organizzazione del lavoro: lotti di tile, handle riusati, cache, thread e sovrapposizione con la GPU. Per questo l'obiettivo è l'uguaglianza **bit per bit**, non una tolleranza.

Identità: la r2 ha protocollo `dense127909-stream-legacy-exact-r2-fast` e propri `CODE_FILES`. Le cache r1 e r2 non sono mai intercambiabili. L'equivalenza si dimostra con i confronti qui sotto, non riusando l'identità r1.

## Verifiche eseguite (offline, nessuna rete, nessuna GPU)
`tests/test_dense_fast.py`, 7 test, tutti superati:

| Test | Cosa confronta | Esito |
|---|---|---|
| `test_rowcol_exact_matches_rasterio_chunks` | righe e colonne su 200.006 punti, anche fuori raster | identiche, stesso dtype int32 |
| `test_fast_reader_matches_legacy` | `values()` SCL/VV/VH, r1 contro r2 con cache RAM, due passate | identici (NaN compresi) |
| `test_acquire_batch_matches_legacy` | 2 tile vicine con cataloghi diversi: S2, mask, S1 asc/desc, DOY | identici; anche alla seconda chiamata dalla cache dei giorni |
| `test_end_to_end_legacy_vs_fast` | `campaign.streaming.stream` (r1) contro `stream_fast` (r2) con encoder finto deterministico per pixel, chunk 97 e batch 16 | `tile.npz` (cell, sums, counts) e `v2.parquet` identici; ripresa idempotente |
| `test_replace_retries_while_reader_holds_file` | `os.replace` bloccato da un lettore senza condivisione Delete | ritentato, salvataggio riuscito, nessun temporaneo |
| `test_failed_replace_keeps_previous_file` | salvataggio fallito | file precedente intatto, nessun temporaneo |
| `test_make_batches_groups_neighbours_within_aoi` | formazione dei lotti | come atteso |

**Test di mutazione:** invertendo di proposito l'ordine delle scene nella r2 (plugin pytest temporaneo, codice invariato), `test_acquire_batch_matches_legacy` e `test_end_to_end_legacy_vs_fast` **falliscono**. I test colgono quindi differenze reali.

**Benchmark simulato** (`scripts/bench_synthetic.py`, server HTTP locale con latenza e banda limitata):
- 6 configurazioni, dalle 4 alle 6 tile ciascuna;
- in tutte, `tile.npz` e `v2.parquet` r1 e r2 identici bit per bit.

## Verifiche ancora da fare su dati reali (richiedono approvazione)
1. **Tile piccole già completate in produzione**, ricalcolate con la r2 in una work-root di prova:
   - `belluno_dolomiti/grid_11.95_45.95` (8 pixel), `appennino_tosco_emiliano/grid_10.65_44.25` (659), `belluno_dolomiti/grid_11.95_46.25` (1.266);
   - poi `scripts\dense_fast.py compare-tiles --left F:\biomap_dense_public\work --right <prova>`, che legge soltanto entrambe le cartelle.
   - **Criterio:** `all_bitwise_equal: true`. Sono attese anche le stesse somme, perché GPU, batch e chunk sono gli stessi della produzione.
2. **Encoder sul campione storico ligure** (32 pixel, `F:\dense_liguria_bundle_20261110\encoder_sample.npz`):
   - stesso criterio del `legacy_check.json` esistente: input identici, differenza massima sugli embedding non oltre 1e-4.
   - La tolleranza vale **solo** contro lo storico del Mac (GPU diversa) e non viene mai allargata.

Se una verifica reale non fosse bit per bit, si riportano differenza massima e causa. La r2 non va usata finché la causa non è chiarita.
