# Evidenze di verifica E2

Verifica locale del 9 novembre 2026 nella copia isolata `biomap-finetuning-dev`.

## Ambiente e comandi eseguiti

- macOS Apple Silicon, Python 3.12.14, PyTorch 2.10.0, device CPU.
- NumPy 2.2.6, pandas 2.2.3, rasterio 1.4.3, pyproj 3.7.1, scikit-learn 1.6.1.
- Nessun download del checkpoint Medium o di Sentinel; encoder ridotto nelle fixture.
- `python -m pip check`: nessuna dipendenza incompatibile rilevata.
- `python -m compileall -q src/biomap_ml` e `git diff --check`: esito positivo.

La suite completa (`python -m pytest -q -ra`) ha completato **69 test con exit code 0**.
Dopo l'aggiunta del registro delle invocazioni della cache, sono stati rieseguiti i 14
test E2: **14 passed, 18 warnings, 27,92 secondi**. I 55 test legacy non sono stati
modificati. La CI Windows/Linux usa la suite completa e il lockfile del repository.

I warning locali riguardano deprecazioni Affine/pyparsing e il rilevamento dei core
fisici joblib. Non rappresentano errori dei target o misure di accuratezza scientifica.

## Cosa verificano i 14 test E2

| Area | Evidenza |
|---|---|
| Geometria | Nord in alto; punti mancanti; duplicati/fuori cella/non allineati rifiutati |
| Supporto | Pesi geodetici positivi e normalizzati; maschera non confusa con target zero |
| CNN | Forward in [0,1], pooling corretto, gradienti finiti e parametri aggiornabili |
| Cache | SHA record, corruzione rilevata, ripresa senza ricaricare/ricodificare l'encoder |
| Isolamento | Guardia che rifiuta scritture sorgente e confronto hash prima/dopo |
| Split | OOF copre una volta lo sviluppo della fixture; test non presente nel training |
| Dummy | Media del solo training pertinente, confronto sulle stesse celle |
| Resume | Interruzione durante epoca 2: stessi pesi del run ininterrotto |
| OOM | Ripristino epoca dopo OOM simulato, microbatch ridotto, batch effettivo invariato |
| Configurazione | Resume con LR diverso rifiutato |
| Test | Secondo evaluate non richiama l'inferenza; mantiene il checksum delle predizioni |
| Excel | Conteggi/righe coerenti coi CSV, rebuild idempotente, file aperto segnalato |
| Raster | Valori TIFF coerenti con CSV entro 1e-7, EPSG:4326, NoData -9999 |
| CLI | Sei comandi `python -m ... --help` funzionano anche da directory diversa |

La CNN minima ha **13.505 parametri**. Il controllo frozen della cache confronta lo stato
encoder prima/dopo; il training E2 non contiene un encoder nei gruppi optimizer.

## Correzioni emerse durante la verifica

La riconciliazione Excel ha individuato una riga vuota prima dell'header in `Risultati`,
generata dal controllo della prima cella. È stato introdotto un flag esplicito di header
scritto; il test verifica il numero esatto di righe dopo due ricostruzioni.

L'uguaglianza numerica del Dummy dopo il round-trip CSV usa tolleranza assoluta 1e-15,
per distinguere una differenza di rappresentazione floating point da una diversa media.

I DOY al cap 365 e le modalità Sentinel opzionalmente vuote sono gestiti secondo il
formato della preparazione esistente; non vengono cancellate osservazioni per forzare
una falsa unicità dei DOY.

## Verifiche non eseguite e condizioni di utilizzo

Non è stato eseguito un audit delle 132.900 serie reali sul PC universitario. Non è stato
usato un modello sintetico per aggirare i controlli nel codice operativo: le sostituzioni
di SHA/manifest sono confinate alle fixture pytest.

Non sono misurati VRAM, throughput CUDA, costo della cache reale o metriche MSA E2.
Il probing attivo non è stato interrotto. E4/E5 non sono implementati in questa fase.
La CI CPU Windows/Linux non certifica CUDA, prestazioni scientifiche o licenze dei dati.

Per l'esito della CI del branch consultare la draft PR GitHub e il suo SHA; gli esiti
remoti sono distinti dalle prove locali descritte qui.
