# Report primo run di probing MSA (TESSERA V2 frozen)

Data del report: 9 novembre 2026. Fonti: file dei run in `data\msa_v1_liguria\runs` e `data\msa\runs`, riferimento V1 in `configs\msa_v1_liguria_reference.json`.

## In breve

- **Il run completo non è ancora stato fatto.** In `data` ci sono solo due *smoke test*: prove dell'intera catena su 12 celle, pensate per verificare che il codice funzioni, non per misurare le prestazioni.
- Lo smoke del 4 novembre ha funzionato dall'inizio alla fine: serie Sentinel, embedding TESSERA, GroupKFold per AOI, test sulla Liguria, confronto con il V1. Il codice stesso marca il confronto come `smoke_not_comparable`.
- Con 6 celle di addestramento i numeri non dicono nulla sul modello. Per esempio, HistGB predice lo stesso valore per tutte le celle e sul test va peggio della media (RMSE 0,267 contro 0,213).
- Il confronto vero con il V1 (HistGB, RMSE 0,158 e MAPE 26,45% sulle 4.991 celle liguri) arriverà solo dal run completo, quello che stiamo preparando ora.

## 1. Quali run esistono

| Run | Cartella | Data | Celle | Ruolo |
| --- | --- | --- | --- | --- |
| Smoke V1-Liguria | `data\msa_v1_liguria\runs\20261104T090455304923Z_msa_v1_liguria_smoke` | 4/11/2026 | 6 train + 6 test | Prova del protocollo che useremo per il run completo |
| Smoke vecchio profilo | `data\msa\runs\20261029T185912348296Z_probe_smoke` | 29/10/2026 | 4 train + 4 validation + 4 test | Prova del profilo precedente (`msa_cells.csv`, AOI "bii_1000"), ora superato |

Lo smoke del 4 novembre è quello rilevante: stesso YAML (`configs\msa_v1_liguria.yaml`), stesso elenco di celle, stessi fold e stesso confronto con il V1 del run completo. Quello del 29 ottobre usa un altro insieme di celle e un altro riferimento storico (`msa_v1_native1000`), quindi resta solo come storia del progetto.

### File prodotti da un run (smoke del 4 novembre)

| File | Contenuto |
| --- | --- |
| `protocol.json` | Conteggi effettivi (6 train, 6 test) e attesi (127.909, 4.991), flag `smoke: true` |
| `folds.json` | Per ogni fold: celle di addestramento e AOI di validazione |
| `cv_metrics_by_aoi.csv`, `cv_macro_metrics.csv` | Metriche di validazione incrociata per AOI e in media |
| `selection.json` | Modello scelto sulla validazione (HistGB) e hash del checkpoint TESSERA |
| `metrics.csv`, `predictions.csv` | Metriche e predizioni del modello finale su train e test |
| `predictions_cv.csv` | Predizioni fuori fold usate per la validazione |
| `historical_reference.csv` | Metriche V1 sulle 4.991 celle liguri |
| `comparison_v1_v2.csv` | V1 e V2 affiancati; colonna `delta` vuota perché smoke |
| `report.md`, `*.png`, `*.joblib` | Report automatico, grafici osservato/predetto, modelli salvati |

## 2. Risultati dello smoke del 4 novembre

### Celle usate

| Partizione | AOI | Celle | MSA osservato |
| --- | --- | --- | --- |
| Train | appennino_tosco_emiliano | 2 | 0,738 |
| Train | abruzzo_gran_sasso | 2 | 0,767 |
| Train | calabria_sila | 2 | 0,288 |
| Test | liguria_levante | 6 contigue | da 0,224 a 0,557 (media 0,421) |

Le 2 celle di ogni AOI sono adiacenti e hanno praticamente lo stesso target. In pratica il modello vede 3 valori distinti.

### Validazione incrociata (GroupKFold per AOI, 3 fold)

| Modello | RMSE macro | AOI peggiore |
| --- | --- | --- |
| DummyMean | 0,310 | calabria_sila, 0,465 |
| HistGB | 0,257 | calabria_sila, 0,460 |

In ogni fold il modello si addestra su 4 celle di 2 AOI e predice la terza. Calabria Sila (MSA 0,29) non si può predire da due AOI con MSA intorno a 0,75: qualunque modello sbaglia di circa 0,46.

### Test sulla Liguria (6 celle)

| Modello | MAE | RMSE | Bias | R² |
| --- | --- | --- | --- | --- |
| DummyMean | 0,176 | 0,213 | +0,176 | −2,13 |
| HistGB | 0,238 | 0,267 | +0,238 | −3,90 |

HistGB predice 0,660 per tutte le 12 celle, sia train sia test: con 6 esempi non riesce a fare alcuna suddivisione e si riduce a una costante. Poiché quella costante è più alta della media di train, sul test va peggio della media. È il comportamento atteso per uno smoke, non un difetto del modello o degli embedding.

### Cosa dimostra davvero lo smoke

- Download e preparazione delle serie funzionano: 12 celle su 12 valide, `excluded` vuoto.
- L'encoder TESSERA V2 Medium gira su GPU e produce embedding. Il checkpoint registrato in `selection.json` ha hash `3823be7d…`.
- Fold, AOI di validazione e test ligure seguono il riferimento V1 (`cells_sha256` coerente).
- Il confronto con il V1 viene prodotto ed è protetto: in smoke il codice rifiuta di calcolare il delta.

## 3. Confronto con il V1

### Il riferimento V1

Run storici `20260928T111520077028Z` (sviluppo) e `20260928T133950415054Z` (test), HistGB su embedding V1 frozen.

| Misura | V1 HistGB | V1 DummyMean |
| --- | --- | --- |
| Celle di test (Liguria) | 4.991 | 4.991 |
| MAE | 0,122 | 0,284 |
| RMSE | 0,158 | 0,302 |
| MAPE | 26,45% | 44,83% |
| Bias | −0,047 | −0,254 |
| R² | 0,057 | −2,43 |
| Pearson r | 0,445 | n.d. |
| RMSE CV macro sulle 17 AOI | 0,173 | n.d. |

Un dettaglio da tenere presente per il confronto: la Liguria ha MSA medio 0,646, molto sopra la media delle celle di sviluppo (0,392). La media di train, quindi, sottostima di 0,25. Il V1 recupera quasi tutto questo scarto (bias −0,047) ma spiega poco della variazione tra celle liguri (R² 0,06, r 0,44).

### Smoke V2 contro V1

| Misura | V1 HistGB (4.991 celle) | V2 smoke HistGB (6 celle) |
| --- | --- | --- |
| RMSE | 0,158 | 0,267 |
| MAE | 0,122 | 0,238 |
| MAPE | 26,45% | 73,33% |

**Questi numeri non vanno confrontati.** Il V2 è stato addestrato su 6 celle invece di 127.909 e testato su 6 celle contigue invece di 4.991. Lo dice anche il file stesso (`comparison: smoke_not_comparable`). La differenza misura solo la dimensione del campione, non la qualità di TESSERA V2.

### Le uniche indicazioni quantitative sul V2, per ora

Nella prova in sola lettura fatta per la strategia di fine-tuning (6.000 celle di 4 AOI, embedding frozen, una AOI esclusa a turno), HistGB ha ottenuto RMSE 0,223 sulle AOI escluse contro 0,258 della media. Il V1 ha RMSE CV 0,173, ma con 17 AOI e circa 85.000 celle di addestramento per fold, contro 4.500 celle di 3 AOI. Anche questo confronto non è appaiato: indica solo che gli embedding V2 contengono segnale sull'MSA, non se battono il V1.

## 4. Cosa cambia con il run che stiamo facendo ora

| Aspetto | Smoke del 4 novembre | Run completo in corso |
| --- | --- | --- |
| Celle di sviluppo | 6 (3 AOI) | 127.909 (17 AOI), comprese le 4 celle di Puglia accettate con copertura ridotta |
| Celle di test Liguria | 6 contigue | 4.991, le stesse del V1 |
| Validazione | 3 fold da 2 celle | 3 fold GroupKFold, da 38.808 a 44.895 celle di validazione per fold |
| Confronto V1 | `smoke_not_comparable`, delta vuoto | Delta V2 − V1 calcolato in `comparison_v1_v2.csv` |
| Tempo | pochi secondi | `prepare` circa 1–2 h, poi `run` per gli embedding di 132.900 celle (ore) |
| Utilità | verifica tecnica | primo risultato scientifico V2 |

Il codice di probing non cambia: modelli, parametri HistGB, fold e metriche sono gli stessi dello smoke. Dal 4 novembre sono cambiati solo la velocità di `prepare` (motore *optimized*, che produce le stesse serie) e l'eccezione per le 4 celle di Puglia (`acquisition.low_coverage_cells`).

## 5. Come leggere il run completo

1. **Controllare il protocollo.** `protocol.json` deve riportare train 127.909, test 4.991, `smoke: false`, 17 AOI di sviluppo.
2. **Guardare la validazione prima del test.** `cv_macro_metrics.csv` per HistGB va confrontato con l'RMSE CV del V1 (0,173). È il confronto più solido, perché su 17 AOI.
3. **Poi il test ligure.** In `comparison_v1_v2.csv`, RMSE sotto 0,158 e MAPE sotto 26,45% significano che il V2 frozen fa meglio del V1. Guardare anche bias e R²: il V1 ha R² 0,06, quindi c'è molto margine.
4. **Leggere il confronto con cautela.** Il V1 usava la media densa dei pixel, il V2 la media di 25 punti stratificati. Un delta misura l'insieme encoder più campionamento, non il solo encoder. Inoltre il test ligure è già stato osservato in passato, quindi è un benchmark esplorativo.
5. **Guardare le AOI.** `cv_metrics_by_aoi.csv` mostra dove il modello sbaglia. Nella prova preliminare, l'errore dipendeva soprattutto dal livello medio dell'AOI.

## 6. Limiti di questo report

- Non contiene risultati del run completo, perché il run non è ancora stato eseguito.
- I numeri V2 citati vengono da 12 celle (smoke) o da 6.000 celle di 4 AOI (prova preliminare): nessuno dei due è confrontabile con il V1.
- I valori V1 sono letti da `configs\msa_v1_liguria_reference.json` e `historical_reference.csv`; il run V1 non è stato ricalcolato.
