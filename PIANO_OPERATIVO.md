# Aggiornamento: campagna densa 127.909

La nuova CLI è `scripts/dense_campaign.py`, disponibile su `main` (PR #2 integrata). [Runbook universitario](docs/ml/DENSE_127909_UNIVERSITY_RUNBOOK.md) e [audit con blocchi reali](docs/ml/DENSE_127909_AUDIT.md). Non esegue training o acquisizioni automaticamente. Il reference multi-AOI è ancora bloccato da target storici iCloud non leggibili; non confondere codice implementato con dati pronti. Nessuna nuova metrica scientifica prodotta.

---

# Piano operativo — priorità al probing V2 denso

Data: 10 novembre 2026. La roadmap allegata dall'utente sostituisce il precedente piano
di avvio E4/E5: **in questa fase gli encoder restano congelati**.

## Obiettivo immediato

Rendere eseguibile sul PC universitario un confronto V1/V2 con **lo stesso supporto di
pixel e la stessa media densa**, eliminando la differenza V1 denso contro V2 a 25 punti.

La prima consegna utilizzabile è il benchmark denso **Liguria Levante**, dove i dati sono
già completi. Questo non completa la campagna sulle 17 AOI: i dati V2 densi di sviluppo
sono ancora incompleti. Non sostituire silenziosamente quella campagna con la Liguria.

## M0 — Audit e conservazione

**Completato per il percorso attivo:** recuperati codice V1, parquet storico, geometrie
V2, conteggi, target e architettura E2. E2 usa davvero un reticolo spaziale 5×5, non una
falsa immagine ottenuta dalle 128 medie. I vecchi esperimenti rimangono disponibili.

**Da completare:** ricostruzione completa di architetture e risultati degli MLP V1 storici.
L'MLP del nuovo benchmark è una baseline esplorativa dichiarata, non una replica di un
run storico non verificato. Le metriche V2/25 punti presenti nell'analisi allegata sono
informazioni riportate dall'utente, non risultati ricontrollati in questa attività.

**Criterio:** ogni risultato usato deve avere run, popolazione, feature, fold e metriche
identificabili. Nessun risultato di fixture è un risultato scientifico MSA.

## M1 — Aggregazione densa V2 identica al V1

### M1a: Liguria, completata e trasferibile

1. Verifica del supporto fisico delle 10 tile: CRS, centri, maschere, indici, hash e celle.
2. Ricalcolo V1 con il criterio storico `trunc`: coincide con il parquet storico.
3. Riaggregazione di 3.545.951 pixel V2 FP32 sulle stesse 4.991 celle, senza sottocampionamento.
4. Conservazione delle 128 medie, copertura, conteggi e target nel pacchetto appaiato.
5. Congelamento dei cinque fold geometrici prima di addestrare i regressori.
6. Campione Sentinel di 32 pixel per verificare il Medium sul PC ricevente.
7. Pubblicazione del piccolo ZIP come asset Release e hash nello script di download.

**Criteri di equivalenza:** identici identificativi e conteggi; tolleranza medie V1 10⁻⁶,
V2 10⁻⁵; nessuna cella esclusa silenziosamente. Tolleranza spot-check encoder 10⁻⁴.

**Evidenza reale:** medie V1 max differenza 0; medie V2 6,94×10⁻¹⁸; conteggi discordanti
zero; spot-check CPU 32 pixel max differenza 2,15×10⁻⁶. Sono controlli tecnici, non metriche
di regressione nuove.

Il `trunc` storico include una gestione dei bordi oggi corretta in `floor` nel vecchio
progetto. Per una futura campagna corretta con `floor`, cambiare entrambi gli encoder e
versionare il nuovo protocollo: non mescolare feature di bordo prodotte con regole diverse.

### M1b: 17 AOI di sviluppo, non completata

Inventario attuale: Abruzzo 14/16 tile complete; altre 16 AOI di sviluppo senza tile
codificate complete; Liguria 10/10. L'inventario non implica che tutte le serie Sentinel
delle tile mancanti siano già scaricate. Il codice del run attuale non avvia una nuova
campagna di acquisizione: prima occorre congelarne campione, budget e input disponibili.

Per completare M1b:

1. Inventariare per AOI tutti i pixel V1, le serie dense disponibili e le parti mancanti.
2. Fissare griglia/bordi/supporto e mantenerli identici fra encoder.
3. Misurare su una tile tempi, RAM, VRAM, disco e throughput dell'inferenza V2.
4. Preparare la campagna densa con ripresa per tile/chunk: riusare i completati senza ricodificarli.
5. Completare prima le due tile Abruzzo mancanti, se il pilot e i dati lo consentono.
6. Proseguire sulle restanti aree soltanto con un budget misurato sostenibile.
7. Validare tutti i conteggi e rigenerare le feature per le 127.909 celle originali.

Una media delle serie a 25 punti non permette di svolgere questi passaggi: servono
gli input/embedding dei pixel densi aggiuntivi.

## M2 — Benchmark V1/V2 appaiato

### Run immediato sul PC universitario

1. Attendere la fine dell'eventuale processo che usa il checkout, o usare un clone separato.
2. Salvare le modifiche universitarie locali e preservare gli stash esistenti.
3. Pull di **main**, aggiornamento dipendenze e test CPU del percorso denso.
4. `fetch-bundle`: download piccolo, controllo SHA256 e verifica del pacchetto.
5. `all --bundle`: controllo del Medium locale sul campione, poi training dei cinque regressori.
6. Verificare `run_state.json`, metriche, predizioni, modelli, figure e Excel.

I comandi completi copiabili sono nel [README](README.md). La GPU è usata solo per il
piccolo controllo encoder se si sceglie CUDA; le regressioni tabellari sono CPU.

### Protocollo del benchmark ligure

- Cinque bande spaziali contigue in EPSG:3035, congelate nel pacchetto.
- 4.991 celle OOF: training circa 3.992/3.993, valutazione 998/999 per fold.
- Stessi target, celle, supporti, fold e iperparametri per ogni coppia encoder/regressore.
- Dummy stimato nel solo training; scaler Ridge/MLP stimato solo nel training.
- Seed 20260925 e configurazioni dichiarate prima del run; nessuna ricerca di iperparametri.
- Confronto di DummyMean, Ridge, HistGB, XGBoost e MLP, anche se un modello perde.
- Nessun refit o scelta guidata da un test geografico esterno in questa fase.

**Interpretazione:** è una CV entro Liguria già osservata, non trasferimento geografico
né nuovo test cieco. Il confronto vero 17 AOI→Liguria attende M1b e riuserà i tre fold
AOI storici: 89.101/38.808, 83.014/44.895, 83.703/44.206, poi refit e test ligure separato.
Umbria nord rimane esclusa.

## M3 — Regression head e M4 — diagnostica

Dopo il run leggere MAE, RMSE, MAPE con denominatori, bias, R² e Pearson; confrontare
Dummy e delta V2−V1 con intervalli bootstrap per blocchi spaziali. Non classificare
come successo un intervallo che include zero o una riduzione ottenuta su popolazioni diverse.

Analizzare residui, distribuzioni, estremi del target, variabilità dei fold e shift delle
feature. Qualsiasi ricerca successiva di iperparametri richiede validation interna e
budget uguale fra encoder. Non scegliere l'architettura guardando ripetutamente Liguria.

Ablation `mean` contro `mean+std` o covariate GLOBIO/pressioni sono esperimenti distinti.
Il benchmark corrente non altera queste scelte retroattivamente.

## M5 — CNN spaziale, rinviata

Prima consolidare il probing denso. Successivamente preparare un pilot multi-AOI da
5.000–10.000 celle con patch native, maschere, padding e pooling; confrontare CNN e
baseline dense sulle stesse celle. E2 a 25 punti non è questa CNN densa futura.

Gli encoder resteranno congelati anche nel pilot. Non avviare partial/LoRA in base al
vecchio piano archiviato: la nuova roadmap richiede prima i confronti frozen.

## M6–M7 — Generalizzazione e consolidamento

Una volta completati i dati multi-AOI, eseguire valutazione geografica con selezione
interna, confronto dei regressori e bootstrap spaziale. Conservare Liguria come benchmark
storico ed eventuale territorio nuovo sigillato per il test finale.

Ogni run deve conservare revisione codice, versioni software, hash, configurazione,
membership, modello per fold, predizioni, metriche e tempi. CSV/JSON sono la fonte di
verità; il registro Excel è ricostruibile e non sovrascrive i risultati precedenti.

## Obiettivi di accettazione della consegna corrente

- Codice e documentazione recuperabili da main; dati piccoli recuperabili da Release.
- Run Windows eseguibile senza percorsi Mac o copie integrali delle serie Sentinel.
- Test CPU, CLI e controlli di integrità superati; CI Windows/Linux verificata.
- Report dinamico del singolo run, nessuna conclusione del Mac copiata come nuovo risultato.
- Feature dense liguri verificate; campagna densa 17 AOI esplicitamente ancora aperta.
- Nessun training scientifico universitario avviato automaticamente.
