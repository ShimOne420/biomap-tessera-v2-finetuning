# Benchmark denso V1 / V2 TESSERA — GLOBIO4 MSA 2020

## Sintesi

- 4991 celle appaiate; stesso supporto denso (media di 128 embedding), stessi fold spaziali e iperparametri per V1 e V2.
- HistGB: MAE V1 0.09727 → V2 0.09430 (Δ -0.00296, IC 95% blocchi [-0.00464, -0.00116]); RMSE V1 0.14394 → V2 0.14179 (Δ -0.00215, IC 95% blocchi [-0.00415, -0.00013]).
- Ridge: MAE V1 0.10329 → V2 0.09777 (Δ -0.00552, IC 95% blocchi [-0.00898, -0.00254]); RMSE V1 0.14728 → V2 0.14142 (Δ -0.00586, IC 95% blocchi [-0.00977, -0.00264]).
- Δ negativo = V2 con errore minore. Gli intervalli bootstrap a blocchi spaziali da 2 km escludono lo zero per Ridge e per HistGB, ma il margine di HistGB è piccolo.
- Il ricalcolo V1 con troncamento riproduce esattamente il parquet storico; il ricalcolo V2 riproduce le medie dense storiche e l’encoder attivo riproduce gli embedding V2 entro 2,4e-6.
- Resta un confronto interno alla Liguria, non il trasferimento sui 17 AOI di sviluppo: le feature V2 dense di quei AOI non esistono ancora.

## 1. Domanda e disegno

Questo esperimento isola la differenza fra **encoder V1 (int8 dequantizzato)** ed **encoder V2 Medium (fp32)** usando **lo stesso supporto di pixel e la stessa aggregazione densa**. Non usa i 25 punti stratificati del pipeline V2 operativo: ogni cella GLOBIO riceve la media dei 128 embedding su tutti i pixel V1 con scala finita e copertura ≥ 0,8. Le celle, il target e i fold sono identici per i due encoder.

La valutazione eseguita è una **cross-validazione spaziale dentro la Liguria Levante** (5 bande contigue in EPSG:3035, stesse bande per V1 e V2, OOF su tutte le celle). È un confronto appaiato e controllato; **non** sostituisce il benchmark storico di trasferimento sviluppo→Liguria, che richiederebbe le feature V2 dense dei 17 AOI di sviluppo.

## 2. Equivalenza degli artefatti

- V1 ricalcolato da tile grezze (`edge_op=trunc`): 4991 celle; max |Δ mean| = 0.00000000; n_valid diversi = 0; n_total diversi = 0; celle mancanti = 0.
- V1 ricalcolato da tile grezze (`edge_op=floor`): 4991 celle; max |Δ mean| = 3.34180279; n_valid diversi = 124; n_total diversi = 124; celle mancanti = 0.
- V2 riaggregato da `embeddings_fp32.npy` + `geometry.npz`: 4991 celle; max |Δ mean| = 0.00000000; n_valid diversi = 0.
- Encoder V2 attivo vs embedding V2 storici su 256 pixel: max |Δ| = 2.38e-06 (tolleranza 1e-04).

L’aggregazione V1 è stata ricalcolata con **troncamento** degli indici, come nel run storico; il confronto con `floor` documenta la correzione del bordo. Se il ricalcolo V1 non coincide con il parquet storico, il confronto non è valido e il comando termina con errore.

## 3. Metriche OOF dentro la Liguria

Stesse 4.991 celle, stessi fold spaziali, iperparametri V1 congelati (HistGB 90 iterazioni, 15 foglie, learning rate 0,05, L2 1,0, early stopping interno, seed 20260925; Ridge alpha 100).

| Encoder | Modello | n | MAE | RMSE | MAPE % | R² | Pearson | Bias | RMSE macro-fold |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| V1 | DummyMean | 4991 | 0.12507 | 0.16342 | 32.794 | -0.0082 | -0.1208 | 0.00000 | 0.16245 |
| V2 | DummyMean | 4991 | 0.12507 | 0.16342 | 32.794 | -0.0082 | -0.1208 | 0.00000 | 0.16245 |
| V1 | Ridge | 4991 | 0.10329 | 0.14728 | 26.199 | 0.1810 | 0.4613 | 0.00506 | 0.14639 |
| V2 | Ridge | 4991 | 0.09777 | 0.14142 | 25.384 | 0.2449 | 0.5045 | 0.00426 | 0.14088 |
| V1 | HistGB | 4991 | 0.09727 | 0.14394 | 25.449 | 0.2178 | 0.4761 | 0.00145 | 0.14335 |
| V2 | HistGB | 4991 | 0.09430 | 0.14179 | 24.921 | 0.2410 | 0.4990 | 0.00156 | 0.14117 |

MAPE esclude i target nulli; qui il target MSA non è mai nullo. `RMSE macro-fold` è la media delle RMSE dei cinque fold spaziali, non una media di errori quadratici.

## 4. Confronto appaiato V2 − V1

Δ negativo = V2 con errore minore. Gli intervalli usano 2.000 bootstrap: uno su celle e uno su blocchi spaziali da 2 km, per tenere conto dell’autocorrelazione. `P(V2 meglio)` è la frazione di repliche con Δ < 0.

| Modello | ΔMAE (V2−V1) | IC 95% celle | IC 95% blocchi | P(V2 meglio, blocchi) | ΔRMSE (V2−V1) | IC 95% blocchi |
|---|---:|---|---|---:|---:|---|
| DummyMean | 0.00000 | [0.00000, 0.00000] | [0.00000, 0.00000] | 0.000 | 0.00000 | [0.00000, 0.00000] |
| Ridge | -0.00552 | [-0.00655, -0.00453] | [-0.00898, -0.00254] | 1.000 | -0.00586 | [-0.00977, -0.00264] |
| HistGB | -0.00296 | [-0.00417, -0.00174] | [-0.00464, -0.00116] | 1.000 | -0.00215 | [-0.00415, -0.00013] |

## 5. Riferimento storico V1 (trasferimento sviluppo→Liguria)

Riproduzione esatta del modello V1 congelato sulle 4.991 celle liguri, a conferma del codice di metrica. Questo numero **non** è il risultato della CV dentro la Liguria e non è confrontabile con la tabella precedente: il modello ha visto i 17 AOI di sviluppo, non la Liguria.

- MAE 0.12206, RMSE 0.15808, MAPE 26.451%, R² 0.0566, bias -0.04652.
- Scostamento massimo dalle metriche storiche: 0.000000000000.

## 6. Interpretazione e limiti

- Il disegno isola l’encoder: stesse celle, stesso target, stesso supporto pixel, stessa formula di media, stessi fold e iperparametri. Le uniche differenze sono i pesi/rappresentazione V1 vs V2 e il formato numerico (int8 dequantizzato vs fp32).
- La CV è **dentro** un’area già osservata e solo ligure; non è un test cieco e non stima il trasferimento geografico su AOI nuove.
- Il confronto V1/V2 completo sui 17 AOI di sviluppo resta **non eseguito**: le feature V2 dense esistono solo per la Liguria; Abruzzo è quasi completo, le altre AOI sono solo inventariate. Il codice di preparazione V2 è riprendibile e documentato, ma servono dati e tempo su una macchina con GPU.
- I target GLOBIO4 MSA sono un indicatore modellato, non una validazione di biodiversità in campo. I risultati non autorizzano affermazioni causali oltre il confronto interno.
- HistGB con early stopping usa una validation interna casuale sul solo training di ciascun fold; è lo stesso comportamento del riferimento V1.

## 7. Riproduzione

```bash
python scripts/dense_msa.py check-v1 --v1-tiles <tessera/2020> --target <msa_2020_unitless.tif> \
  --v1-parquet <tessera_globio_msa.parquet> --out outputs/dense_benchmark
python scripts/dense_msa.py check-v2 --v2-tiles <Archivio2/AOI> --v1-parquet <tessera_globio_msa.parquet> \
  --v2-parquet <tessera_v2_medium_globio_msa.parquet> --out outputs/dense_benchmark
python scripts/dense_msa.py run --v1-parquet <tessera_globio_msa.parquet> \
  --v2-tiles <Archivio2/AOI> --out outputs/dense_benchmark --folds 5 --bootstrap 2000
```

Percorsi usati in questa run:

- `v2_tiles`: `/Volumes/Archivio2/BioMAP_TESSERA_V2_Medium_2020/liguria_levante`
- `v1_parquet`: `~/Desktop/Università/Tesi_BioMap/TCBiomap/biomap-tessera/data/areas/liguria_levante/processed/tessera_globio_msa.parquet`
- `v1_model`: `~/Desktop/Università/Tesi_BioMap/TCBiomap/biomap-tessera/outputs/runs/20260928T111520077028Z/selected_frozen_model.joblib`
- `reference`: `~/Desktop/Università/Tesi_BioMap/biomap-finetuning-dev/configs/msa_v1_liguria_reference.json`
- `out`: `<tmp>/opencode/dense_benchmark_liguria`

Artefatti della run (non versionati): `predictions.csv`, `metrics.csv`, `metrics_by_fold.csv`, `paired_delta.csv`, `deciles_*.csv`, `scatter_*.png`, `protocol.json`.
