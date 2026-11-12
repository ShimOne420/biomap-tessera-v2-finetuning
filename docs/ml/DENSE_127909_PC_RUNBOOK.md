# Campagna densa 127.909: esecuzione completa dal PC universitario (Windows)

Data: 10 novembre 2026. Branch: `dense-public-pc`. Questo percorso non richiede artefatti del Mac.
Il confronto è GeoTESSERA V1 contro TESSERA V2 Medium. Le feature sono la media densa delle 128
componenti sugli stessi pixel V1, con target GLOBIO4 MSA 2020. I due encoder sono congelati e non
c'è fine-tuning.

## 1. Configurazione

| Elemento | Valore |
|---|---|
| V1 | GeoTESSERA dataset **1.0, variante `vultr`** ("Legacy model"), anno 2020, da `https://data.source.coop/tessera/tessera/npy/v1`, con `geotessera 0.11.0` in un venv isolato |
| V2 | TESSERA V2 **Student Medium** eseguito in locale, checkpoint SHA256 `3823be7d…`, FP32, sugli input Sentinel‑1/2 2020 di Microsoft Planetary Computer |
| Aggregazione | `dense/v1_support.aggregate_v1`, invariata: centri pixel UTM → EPSG:4326 → cella GLOBIO nativa 10″, bordo `trunc` storico, scale non finite escluse, somme float64 / n_valid, copertura ≥ 0,8. V2 usa gli stessi pixel e la stessa media |
| Target | GLOBIO MSA 2020: `float32(MSA-area / cell-area)`, nodata −999, ritagliato al box di ogni AOI |
| Celle | 132.900: 127.909 di sviluppo (17 AOI, 3 fold storici) + 4.991 Liguria |
| Regressori | DummyMean, Ridge, HistGB, XGBoost, MLP, con parametri congelati (`probing.params`) |

Il default della libreria è `v1.1`. Per la Liguria non contiene alcuna tile: la versione va quindi
sempre fissata. V2 **non** si scarica da GeoTESSERA: la libreria pubblica solo un V2 `2B-L` in beta,
che è un modello diverso dal Medium.

## 2. Verifiche eseguite su dati reali

| Verifica | Esito |
|---|---|
| Scale e landmask V1 1.0/vultr (10 tile liguri) contro gli SHA256 storici | 10/10 identiche |
| V1 ricalcolato contro le medie storiche Liguria (4.991 celle) | differenza max **0,0**; n_valid, n_total e target identici |
| Target GLOBIO ricalcolato contro il target storico | differenza **0** |
| Geometria V2 contro la storica (pixel per tile) | 10/10 identiche, 3.545.951 pixel |
| Input Sentinel dei 32 pixel storici: giorni, S2, maschera, S1 asc/desc | **identici al bit** |
| Embedding V2 dei 32 pixel (CUDA) contro lo storico | max \|Δ\| 3,3·10⁻⁶ (tolleranza 10⁻⁴) |
| Tile reale `grid_9.35_44.25`: medie V2 di 52 celle contro lo storico | max \|Δ\| 5,1·10⁻⁷ |

## 3. Differenze trovate rispetto al codice di campagna precedente

L'acquisizione di `campaign/inputs.acquire` e `data.power_to_storage` **non** riproduce gli input V2
storici. Il nuovo `campaign/streaming.py` applica le regole storiche, verificate al bit:

| Elemento | Storico, adottato | Codice precedente |
|---|---|---|
| Bande S2 | riproiezione **bilineare** sul reticolo V1, troncate a uint16, senza nodata dichiarato | valore puntuale nearest |
| S1 | `(20·log10(x)+50)·200`, in float32 | `10·log10(x)` |
| Catalogo STAC | bbox dei bordi dei pixel della geometria | bbox della tile intera (S1 asc: 113 date invece di 58) |
| S1 con due scene nello stesso giorno | ogni polarizzazione si riempie da sola | entrambe dalla stessa scena |
| DOY | reale (31/12/2020 = 366) | limitato a 365 |

La conversione S1 storica è anche quella coerente con le statistiche di normalizzazione del Medium
(`S1A_BAND_MEAN` 5909/3405). Con `10·log10` gli input S1 medi di questa tile sono 8069/6983:
VH risulta spostato di oltre 2 deviazioni standard. Le pipeline storiche (25 punti) non sono state
modificate.

## 4. Comandi

Vedere la risposta di consegna o la sezione "Comandi" del README del branch. In sintesi:
`download_v1_tiles.py` → `build-reference` → `verify-legacy-sample` → `run_v2_dense_stream.ps1`
→ `probe`/`evaluate-liguria` per v2 e v1 → `compare` → `export`.

## 5. Output nella work-root

| File | Contenuto |
|---|---|
| `chunks/<aoi>/<tile>/tile.npz` + `.json` | somme e conteggi V2 per cella, checkpoint della tile |
| `stream_progress.json`, `stream_log.jsonl`, `stream_errors.jsonl`, `stream_status.json` | avanzamento, tempi e copertura per tile, errori |
| `v2.parquet` | 132.900 righe con 128 medie V2; si pubblica solo se tutti i conteggi = n_valid V1 |
| `probing/v1`, `probing/v2` | OOF, modelli per fold e finali, metriche per AOI, fold e bin, selezione |
| `comparison_v1_v2_dense_127909.csv` | delta puntuali V2−V1 per AOI, modello e metrica |
| `paired_uncertainty_v1_v2_dense_127909.csv` | IC bootstrap appaiati (AOI per lo sviluppo, blocchi 2 km per la Liguria) |
| `historical_v1_reproduction.json` | V1 ricostruito contro la macro RMSE storica 0,17309 e la Liguria HistGB |
| `pixel_embeddings/` (opzionale) | embedding V2 per pixel, per un futuro CNN denso |

## 6. Misure e stime (RTX 5080, 10/11/2026)

**Misurato:**
- Codifica V2: 2.500–3.550 pixel/s con batch 256 (3 GB di VRAM).
- Acquisizione con 16 thread: 136 s per la tile da 54.193 pixel. La tile da 839.491 pixel ha
  richiesto circa 20 minuti, ma con un download concorrente in corso.
- Reference: 273 tile e 99.527.159 pixel. Build in circa 25 minuti. Probing V1 su 5 regressori in
  4 minuti.

**Stimato (non garantito):**
- Durata: circa 1–2,5 giorni in totale, dominati dalla rete. Codifica circa 8 ore.
- Disco temporaneo: circa 2–4 GB per tile, uno alla volta.
- RAM: circa 6–8 GB per la tile più grande.

**V1 ricostruito contro lo storico (HistGB):**
- Macro RMSE CV: 0,17270 contro 0,17309.
- Liguria MAE: 0,12161 contro 0,12206.
- La Liguria ha feature identiche al bit. Le 17 AOI hanno lo 0,35% di pixel in più rispetto al
  totale storico (95.981.208 contro 95.641.746).

## 7. Limiti

- La Liguria era già nota dal benchmark storico: non è un test cieco.
- Il V1 storico delle 17 AOI non è disponibile sul PC; la sua ricostruzione si verifica con la
  macro RMSE CV HistGB, non cella per cella. Per la Liguria la verifica è invece cella per cella.
- CNN E2 non si applica ai vettori medi: richiede una griglia `[128,5,5]`.
- Non c'è correzione per confronti multipli tra i 5 regressori.
