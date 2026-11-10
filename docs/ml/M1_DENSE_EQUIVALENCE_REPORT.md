# M1 — Equivalenza densa reale della Liguria

Verifica del 10 novembre 2026, senza addestrare nuovi regressori.

| Controllo | Risultato |
|---|---:|
| Celle V1/V2 | 4.991 |
| Pixel densi comuni | 3.545.951 |
| Tile liguri complete | 10/10 |
| Celle mancanti o aggiunte nella replica V1 | 0 |
| Conteggi pixel validi/totali discordanti | 0 |
| Max differenza coordinate e target V1 | 0 |
| Max differenza medie e std V1 | 0 |
| Max differenza medie V2 | 6,938893903907228×10⁻¹⁸ |
| Pixel di controllo encoder CPU | 32 |
| Max differenza embedding nel controllo CPU | 2,1457672119140625×10⁻⁶ |
| Tolleranza encoder fissata | 10⁻⁴ |

Il controllo fisico ha confrontato coordinate, CRS, reticolo, indici univoci, hash delle
scale V1 e delle landmask, hash della geometria, checkpoint Medium e input codificati.
Ogni pixel V2 deve corrispondere a un pixel V1 a scala finita e alla stessa cella GLOBIO.
L'uguaglianza dei soli conteggi non è stata considerata sufficiente.

Il parquet storico usa l'assegnazione `trunc`: questo protocollo è conservato per entrambi
gli encoder. Il codice V1 corrente contiene la correzione `floor` ai bordi; una campagna
che adotti la correzione deve applicarla a entrambi e versionare i nuovi artefatti.

Il pacchetto contiene i cinque fold già congelati, per evitare differenze di ordinamento
o backend numerico tra Mac e Windows. Il checkpoint non è incluso. Il campione di 32
pixel permette una nuova verifica encoder sul PC ricevente; non equivale a ricodificare
tutti i 3,5 milioni di pixel.

## Identità degli artefatti

- Parquet V1 sorgente SHA256: `7b04b40f83fed0cf8275abf4bcf9ff588614a995c63ffa8898a622b3a9536627`.
- Parquet appaiato: `859cda01da9edefba14ee0703319ff29402414c60e9336b877b53cae76d18299`.
- Target normalizzato: `cde571d5e2b02885939f673846348dcf5db9e3d3d0ae4696f04a7e3017a7d0a2`.
- ZIP: `e437e2122080618988cb38eb2a24cac12b021bed073c2579b287bae08adbe87e`.
- Manifest: `75d134e40d748a44e537bb5f8a9bd42dc98f9e5720373bb74f925fe8930e16ea`.

Il dettaglio di ogni tile è in `support_verification.json` nel pacchetto. I risultati di
questa tabella sono controlli di input/implementazione, non MAE/RMSE/MAPE di un nuovo modello.

La campagna sulle 17 AOI non è completata. La Liguria sarà valutata con CV interna e non
presentata come trasferimento da 127.909 celle di training. Il confronto storico V1 con
MAPE 26,45% appartiene a quel diverso protocollo di trasferimento.
