# Architettura per gli esperimenti futuri V1/V2

## Stadi separati
| Stadio | V2 (encoder locale) | V1 (embedding storici) | Codice |
|---|---|---|---|
| 1. Geometria e supporto | tile 0,1°, `geometry.npz` (flat_indices, cell_ids), `lookup.npy` | identico | `campaign/public_reference.py`, reference su F: |
| 2. Acquisizione | Sentinel-1/2 2020 da Planetary Computer, protocollo storico | **nessuna**: tile GeoTESSERA 1.0 int8 scaricate (`scripts/download_v1_tiles.py`) | V2: `dense_fast/acquisition.py`, `reader.py`, `handles.py` |
| 3. Preprocessing | serie per pixel, bin per numero di osservazioni, standardizzazione (`reference.encode_pixels`) | dequantizzazione int8 · scala | V2: `reference.py`; V1: `dense/v1_support.py` |
| 4. Encoder | TESSERA V2 Medium FP32 (`encoder.py`) | già applicato a monte | — |
| 5. Archiviazione | `tile.npz` (somme e conteggi per cella); per pixel opzionale (`--keep-pixel-embeddings`) | tile npy native | `dense_fast/pipeline.py`, `spatial.py` |
| 6. Aggregazione | media per cella = somme / conteggi, nell'ordine del piano | stessa formula | `pipeline.finish` |
| 7. Regressione | Dummy, Ridge, HistGB, XGBoost, MLP | stessa | `campaign/probing.py` |

V1 e V2 **non** condividono l'acquisizione: il V1 non parte da Sentinel ma da embedding già calcolati. Condividono geometria, mappatura cella-pixel, aggregazione, formato delle feature (`mean_000…mean_127`) e regressori. Gli input dei due modelli sono diversi e non vanno uniformati.

## Esperimento A — V2 Dense Mean + regressori
- Si fa direttamente con `scripts\dense_fast.py stream` su una work-root nuova.
- Produce `v2.parquet` con le stesse colonne della r1, da usare con `campaign probe/compare/evaluate-liguria`.

## Esperimento B — V1 feature map spaziali + CNN 2D
- La CNN ha bisogno di **mappe**, non di vettori medi. Ricostruirle da `mean_*` non ha senso, e questo codice non lo fa.
- **V1:** le mappe esistono già nelle tile int8 scaricate. Vanno solo ritagliate sulla geometria di ogni cella, nello stesso reticolo (lavoro di `experiments/v1_spatial_cnn_msa/`, di un'altra sessione: non toccato).
- **V2:** con `--keep-pixel-embeddings` la r2 salva per ogni tile:
  - `pixel_embeddings/<aoi>/<tile>.npy`, in float32, righe nell'ordine di `flat_indices`;
  - un JSON con SHA256, identità, SHA256 di `geometry.npz` e reticolo V1 (width, height, transform, crs).
- `dense_fast.spatial.pixel_map()` ricostruisce la finestra `(H, W, 128)`, con NaN fuori geometria, maschera di validità e affine della finestra. È allineata pixel per pixel al reticolo V1: una CNN V1 e una V2 vedono le stesse posizioni.
- **Disco:** 512 byte per pixel, circa 51 GB per 99,5 milioni di pixel.

## Integrazione futura (non fatta, da approvare)
- Fondere `dense_fast` nel branch principale solo dopo le verifiche reali di `EQUIVALENZA.md`.
- Un eventuale lettore V1 "spaziale" comune dovrebbe stare in un modulo nuovo e usare `geometry.npz` come unica fonte della mappatura cella-pixel.
