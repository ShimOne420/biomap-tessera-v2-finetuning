# Modifiche tecniche — probing V2 denso come V1

Data: 10 novembre 2026. Questo documento spiega le modifiche; i comandi sono nel README
principale e la strategia è in PIANO_OPERATIVO.md.

## Il problema risolto

La campagna V2 a 25 punti conserva 25 posizioni per cella, non tutti i pixel. Cambiare la
funzione `mean` o impostare un flag non ricrea i punti mancanti. Il problema del PC
universitario non era un regressore mancante: i dati densi risiedevano sul Mac/SSD.

Il confronto denso locale già presente nel repository dipendeva da percorsi Mac, tile
V1, intera cartella Sentinel V2 e modelli storici. Inoltre il codice pubblicato aveva
tre soli regressori, sovrascriveva il report versionato e conteneva conclusioni numeriche
fisse. Le modifiche universitarie descritte nell'allegato non erano ancora committate:
sono state riprodotte nel repository condiviso, senza dichiarare di aver recuperato il loro diff.

Ora è disponibile un pacchetto di **11.904.407 byte** (circa 11,35 MiB), con feature dense
appaiate già verificate e un campione per controllare l'encoder. È un asset della Release
GitHub, non un dataset pesante dentro Git. Dopo il pull, `fetch-bundle` scarica e verifica
il pacchetto: non occorre copiare le intere serie Sentinel.

## La metodologia V1 recuperata

Origine locale: `TCBiomap/biomap-tessera/src/biomap_tessera/msa/aggregation.py`; artefatto
Liguria del 28 settembre: `data/areas/liguria_levante/processed/tessera_globio_msa.parquet`.

Per ogni tile V1:

1. Leggere il suo CRS UTM e la trasformazione, senza imporre lo stesso UTM a tutte le tile.
2. Ricostruire i centri dei pixel ed esprimerli in EPSG:4326.
3. Assegnare ogni centro alla cella del target MSA nativo di 10 arcsec.
4. Dequantizzare V1 come `int8 * scale`; scale non finite identificano pixel non validi.
5. Accumulare le 128 componenti e dividere per il numero di pixel validi della cella.
6. Conservare celle con copertura almeno 0,8 e feature finite.

Non viene calcolata una media del target a una griglia più grossolana. La regressione
principale usa le **128 medie**; le deviazioni standard storiche non sono aggiunte di nascosto.

### Precisazione importante sui bordi

Il codice V1 attuale usa `floor`, correggendo un precedente troncamento verso zero. Il
parquet storico ligure fu generato con `trunc`. Per confrontare gli artefatti storici
usiamo **trunc in entrambi gli encoder**, e lo dichiariamo nel manifest. Non correggiamo
un solo encoder: cambierebbe il supporto vicino al bordo nord/ovest. Un futuro benchmark
con `floor` richiederà ricostruire entrambi gli encoder sugli stessi nuovi pixel.

## Applicazione a V2

V2 usa il Medium pre-addestrato, congelato, con checkpoint SHA256
`3823be7db9d9cfc93f3c2a47c7699be82821ab4e1117d4d2befdb746941ee96e`.

Gli embedding FP32 V2 già presenti sono associati agli stessi centri pixel V1. Il nuovo
controllo fisico verifica geometrie, hash delle scale/landmask, CRS, trasformazioni,
indici pixel univoci e appartenenza alle celle. Verificare solo `n_valid` non basta:
due insiemi di punti diversi potrebbero avere lo stesso numero di elementi.

`v2_support.aggregate_v2` applica la stessa media aritmetica sui punti validi, in chunk
di 8.192 pixel, usando accumulatori float64. Non carica l'intera tile in float64 e non
rigenera gli embedding già completati. La precisione V1 int8 dequantizzata e quella V2
FP32 rimangono diverse: il confronto controlla il pooling ma include questa differenza numerica.

## File e collegamenti

L'unico script da avviare è **`scripts/dense_msa.py`**.

| Modulo in `src/biomap_v2/dense/` | Operazione |
|---|---|
| `cli.py`, `cli_args.py` | Comandi, controlli preliminari, orchestrazione e run separati |
| `grid.py`, `v1_support.py` | Griglia target e replica dell'aggregazione storica V1 |
| `support.py` | Identità fisica dei pixel V1/V2, non soltanto numerosità |
| `v2_support.py` | Media densa V2, senza campionamento o nuova inferenza |
| `sample.py` | Campione Sentinel portabile e verifica del Medium sul PC ricevente |
| `bundle.py`, `download.py` | Creazione, hash, ZIP e download del pacchetto |
| `paired.py` | Stesse celle e target; rifiuto di coperture differenti |
| `evaluation.py` | Training esclusivamente dei regressori e predizioni OOF |
| `report.py`, `exports.py` | Report dinamico, grafici, CSV e Excel |

Ogni modulo nuovo/modificato ha meno di 300 righe. `scripts/probe_msa.py` e il dataset a
25 punti restano disponibili come percorso storico; non sono stati trasformati in un
falso percorso denso. Non è stato modificato il codice del run universitario in corso.

## Regressori e protocollo

| Regressore | Configurazione iniziale |
|---|---|
| DummyMean | Media del solo training del fold |
| Ridge | StandardScaler del training + alpha 100 |
| HistGB | 90 iterazioni, 15 foglie, learning rate 0,05, L2 1; early stopping interno al training |
| XGBoost | 400 alberi, profondità 2, learning rate 0,05, obiettivo squared error, CPU |
| MLP | Scaler del training, layer 256–128, ReLU/Adam, LR 0,001, max 500, early stopping sul training |

L'MLP è una configurazione esplorativa dichiarata prima del nuovo run, non una replica
certificata di tutti gli MLP storici V1. Il recupero di questi ultimi rimane un'attività M0.
CNN E2 non è eseguita sui vettori medi: richiede il reticolo spaziale 5×5 e costituisce
un esperimento separato. Non si esegue fine-tuning di TESSERA.

Il pacchetto include cinque fold geometrici congelati prima del training. Tutti i modelli
usano le stesse celle e gli stessi fold per entrambi gli encoder. Il benchmark è una
CV interna alla Liguria: **non** è il confronto sviluppo→Liguria con 127.909 celle.

## Sicurezza dei risultati e riproducibilità

- Ogni output di training ha una cartella nuova; un run esistente viene rifiutato.
- Il report è nella run. Il report storico in `reports/` non viene riscritto.
- Le conclusioni derivano da metriche, bootstrap e controlli effettivamente eseguiti.
- CSV/JSON conservano fonte, hash, versione del codice/software, fold e parametri.
- Sono salvati i modelli di ogni fold, le predizioni per cella, gli errori e il confronto Dummy.
- Excel del run e registro denso comune vengono derivati dai CSV. `export` li rigenera senza rifare il training.
- Il campione encoder legge gli NPY/GZIP senza decomprimere nuovi file nelle sorgenti.
- I pesi Medium non sono inclusi nel pacchetto; sul PC universitario sono già disponibili.

## Limiti e verifiche reali

La verifica reale sul Mac ha confermato 4.991 celle e **3.545.951 pixel** nelle 10 tile liguri.
V1 riaggregato: differenza massima dalle medie storiche **0**. V2 riaggregato: **6,94×10⁻¹⁸**.
Conteggi discordanti: zero. Il Medium CPU su 32 pixel del campione differisce dagli
embedding salvati di al massimo **2,15×10⁻⁶**, sotto la tolleranza fissata di 10⁻⁴.

Queste sono verifiche tecniche, non nuove metriche di regressione. Il nuovo run a cinque
regressori sarà eseguito dall'utente sul PC universitario. Su Archivio2 risultano 10/10
tile complete in Liguria, 14/16 in Abruzzo, zero complete nelle altre 16 AOI. Non è stato
avviato un nuovo download/encoding dell'intera campagna.
