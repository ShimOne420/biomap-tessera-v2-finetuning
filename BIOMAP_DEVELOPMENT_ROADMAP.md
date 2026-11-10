# Roadmap condivisa fornita dall’utente — 10 novembre 2026

Questo è il documento strategico di riferimento. Lo stato effettivo è in BIOMAP_DEVELOPMENT_STATUS.md; le metriche citate nel testo sono riferimenti riportati dall’analista e vanno ricondotte ai rispettivi artefatti prima di usarle come evidenza. La consegna corrente copre M1a/M2 Liguria e non la campagna completa sulle 17 AOI.

Ti propongo un Piano di Sviluppo Tecnico e Sperimentale per BIOMAP-FM, strutturato come documento operativo da condividere con Codex, Claude e il senior developer.

Il piano distingue gli esperimenti necessari per confrontare correttamente GeoTESSERA V1 e V2, quelli dedicati all’ottimizzazione delle regression head e, infine, lo sviluppo di una CNN spaziale capace di utilizzare gli embedding senza aggregazione preventiva.

Ho organizzato le attività in milestone progressive, con obiettivi, implementazioni, deliverable e criteri di completamento.

BIOMAP-FM

Piano di sviluppo tecnico e sperimentale

GeoTESSERA V1/V2 — MSA GLOBIO4 — Regression Heads — Spatial CNN

Versione: 1.0
Data: 10 novembre 2026
Destinatari: Codex, Claude, Senior Developer
Stato: Roadmap operativa per le prossime attività di ricerca e sviluppo

⸻

1. Obiettivo generale del progetto

L’obiettivo della prossima fase di sviluppo di BIOMAP-FM è migliorare la capacità predittiva degli embedding geospaziali prodotti da GeoTESSERA nella stima del Mean Species Abundance (MSA), utilizzando il dataset GLOBIO4 come riferimento.

Il progetto adotta un approccio di frozen foundation model probing.

Gli encoder GeoTESSERA V1 e V2 rimangono congelati: non è previsto, in questa fase, il fine-tuning dei foundation model.

L’attività di ricerca riguarda principalmente:

1. La costruzione di rappresentazioni geospaziali coerenti e confrontabili.
2. Il confronto tra GeoTESSERA V1 e GeoTESSERA V2.
3. L’identificazione delle regression head più efficaci.
4. Il miglioramento della generalizzazione geografica.
5. Lo sviluppo di una regression head convoluzionale capace di sfruttare direttamente la struttura spaziale degli embedding.
6. La riduzione dei costi computazionali attraverso ottimizzazioni che non compromettano la correttezza scientifica.

L’obiettivo non è semplicemente ottenere il RMSE più basso possibile, ma individuare una metodologia robusta, riproducibile e scientificamente difendibile.

⸻

2. Stato attuale della ricerca

2.1 Dataset e rappresentazioni

Il target principale è GLOBIO4 MSA 2020, con valori normalizzati nell’intervallo [0,1].

La risoluzione nativa del target è pari a 10 secondi d’arco.

Gli embedding GeoTESSERA sono disponibili su una griglia spaziale a risoluzione più elevata, indicativamente 10 metri.

Le campagne precedenti hanno utilizzato 17 aree geografiche italiane, per un totale di 127.909 celle di sviluppo.

È inoltre disponibile un benchmark geografico su Liguria Levante, composto da 4.991 celle.

2.2 Esperimenti storici

Sono state utilizzate due differenti strategie di rappresentazione.

GeoTESSERA V1

Aggregazione densa degli embedding validi all’interno delle celle GLOBIO.

La rappresentazione principale è costituita da 128 feature medie per cella.

Sono state sperimentate anche rappresentazioni con media e deviazione standard.

I regressori storicamente valutati comprendono:

* Ridge;
* Lasso;
* OLS;
* MLP;
* XGBoost;
* HistGradientBoosting.

GeoTESSERA V2

Negli esperimenti più recenti è stata utilizzata una rappresentazione ottenuta mediante campionamento di 25 posizioni per cella GLOBIO.

Sono stati confrontati, tra gli altri, HistGradientBoosting e CNN E2.

2.3 Risultati di riferimento

Esperimento storico	Regressore	RMSE Liguria	MAPE Liguria
V1 — Dense	HistGB	0,15808	26,45%
V2 — 25 punti	HistGB	0,18080	27,85%
V2 — 25 punti	CNN E2	0,18009	28,67%

Questi risultati provengono da campagne differenti.

Non devono essere interpretati come un confronto causale tra encoder o metodi di aggregazione.

Il nuovo piano di sviluppo nasce proprio dalla necessità di eliminare tali differenze sperimentali.

⸻

3. Principi metodologici obbligatori

Tutte le attività successive devono rispettare i seguenti principi.

P1 — Comparabilità

Quando si confrontano due encoder, deve cambiare soltanto la versione dell’encoder.

Quando si confrontano due regressori, devono rimanere invariati encoder, rappresentazione, target, celle e protocollo di valutazione.

P2 — Separazione degli esperimenti

Gli esperimenti sull’aggregazione, sulle regression head e sulle rappresentazioni spaziali devono essere identificati separatamente.

Non attribuire miglioramenti a una singola componente quando più elementi della pipeline sono cambiati contemporaneamente.

P3 — Validazione geografica

La valutazione deve considerare la dipendenza spaziale tra osservazioni.

Utilizzare split geografici e procedure di cross-validation appropriate.

Evitare leakage tra training, validation e test.

P4 — Riproducibilità

Ogni run deve conservare:

* configurazione;
* versione del codice;
* versione degli embedding;
* dataset e manifest;
* split;
* seed;
* iperparametri;
* metriche;
* predizioni;
* tempi computazionali.

P5 — Efficienza

Le ottimizzazioni devono preservare il significato scientifico dell’esperimento.

Non modificare silenziosamente il metodo di aggregazione, la risoluzione o la selezione dei pixel per accelerare i run.

⸻

4. Roadmap generale

Milestone	Attività	Priorità	Dipendenze
M0	Audit del repository e consolidamento dei benchmark	Critica	Nessuna
M1	Aggregazione densa V2 identica a V1	Critica	M0
M2	Dataset appaiato V1–V2 e benchmark encoder	Critica	M1
M3	Benchmark completo delle regression head	Alta	M2
M4	Analisi degli errori e ottimizzazione dei regressori	Alta	M3
M5	Prototipo CNN spaziale su embedding non aggregati	Ricerca avanzata	M1, M3
M6	Validazione multi-AOI e confronto finale	Alta	M4, M5
M7	Consolidamento della pipeline e report scientifico	Alta	M6

Le milestone rappresentano una sequenza logica, non una stima temporale vincolante.

⸻

MILESTONE 0 — Audit e consolidamento

Obiettivo

Ricostruire lo stato effettivo del repository prima di introdurre ulteriori modifiche.

Attività

M0.1 — Audit delle pipeline

Identificare:

* implementazione dell’aggregazione densa V1;
* implementazione dell’aggregazione a 25 punti V2;
* formato e dimensionalità degli embedding;
* gestione delle tile;
* gestione dei CRS;
* trattamento dei NoData;
* procedure di costruzione del target;
* codice di training;
* codice di valutazione.

M0.2 — Audit degli esperimenti neurali

Recuperare i run MLP eseguiti su V1.

Per ogni run individuare:

* architettura;
* input;
* numero di parametri;
* preprocessing;
* iperparametri;
* metriche;
* split geografici.

Analizzare inoltre l’architettura effettiva della CNN E2.

È necessario stabilire se le convoluzioni operino su una struttura spaziale o su feature già aggregate.

M0.3 — Consolidamento degli artefatti

Identificare le cache dense V1 riutilizzabili.

Verificare l’integrità dei manifest e dei dataset.

Non sovrascrivere gli esperimenti storici.

Deliverable

M0_REPOSITORY_AUDIT.md

M0_HISTORICAL_EXPERIMENTS.md

Criterio di completamento

Le pipeline storiche devono essere ricostruibili e le differenze tra V1 e V2 documentate.

⸻

MILESTONE 1 — Aggregazione densa GeoTESSERA V2

PRIORITÀ ASSOLUTA

Questa è la prima milestone sperimentale da completare.

Obiettivo

Applicare a GeoTESSERA V2 lo stesso identico algoritmo di aggregazione densa utilizzato storicamente per GeoTESSERA V1.

Il risultato deve consentire un confronto metodologicamente controllato tra i due encoder.

M1.1 — Identificazione dell’algoritmo V1

Recuperare l’implementazione originale.

Verificare:

* selezione dei pixel;
* regole di appartenenza geografica;
* trattamento dei bordi;
* gestione dei NoData;
* ricostruzione degli embedding;
* operazioni di media;
* precisione numerica.

Non assumere che una nuova implementazione teoricamente equivalente produca necessariamente gli stessi risultati.

M1.2 — Adattamento a V2

Riutilizzare preferibilmente la stessa funzione di aggregazione.

La pipeline dovrà essere compatibile con gli embedding V2 senza modificarne il significato.

Per ogni cella GLOBIO, calcolare la media di tutti gli embedding validi selezionati secondo il protocollo V1.

Non utilizzare campionamento a 25 punti.

Non introdurre sottocampionamenti intermedi.

M1.3 — Ottimizzazione computazionale

Ottimizzare:

* lettura delle tile;
* elaborazione per gruppi geografici;
* cache persistenti;
* trasformazioni CRS;
* operazioni vettorizzate;
* parallelizzazione;
* scrittura incrementale.

Le ottimizzazioni devono essere precedute da profilazione.

M1.4 — Verifica di equivalenza

Se il codice V1 viene rifattorizzato, confrontare le nuove feature con quelle storiche.

Verificare:

* stessi identificativi;
* stessi pixel;
* stessi conteggi validi;
* stessi valori aggregati entro una tolleranza numerica documentata.

M1.5 — Esecuzione completa

Eseguire l’aggregazione densa V2 sulle celle eleggibili.

Utilizzare cache riprendibili.

Registrare:

* tempo totale;
* celle elaborate;
* copertura;
* errori;
* memoria massima;
* throughput.

Deliverable

M1_V2_DENSE_FEATURES

M1_DENSE_EQUIVALENCE_REPORT.md

M1_COMPUTATIONAL_PROFILE.md

Criterio di completamento

GeoTESSERA V2 deve disporre di feature dense costruite mediante il medesimo algoritmo utilizzato per V1.

Le differenze di copertura e disponibilità dei dati devono essere documentate.

⸻

MILESTONE 2 — Benchmark controllato GeoTESSERA V1 vs V2

Obiettivo

Stabilire quale encoder fornisca rappresentazioni più efficaci per prevedere MSA.

M2.1 — Dataset appaiato

Costruire l’intersezione delle celle valide V1–V2.

Ogni osservazione deve contenere:

* cell_id;
* AOI;
* target MSA;
* embedding medio V1;
* embedding medio V2;
* numero di pixel validi;
* split geografico.

Le due versioni devono utilizzare gli stessi target e le stesse partizioni.

M2.2 — Benchmark dei regressori

Confrontare:

1. DummyRegressor;
2. Ridge;
3. HistGradientBoosting;
4. XGBoost;
5. MLP;
6. CNN E2, previa verifica della compatibilità architetturale.

Per ogni regressore, utilizzare lo stesso protocollo di training e valutazione per V1 e V2.

Gli iperparametri possono essere selezionati separatamente per ciascun encoder, purché il budget di ricerca e le regole di selezione siano equivalenti.

M2.3 — Metriche

Calcolare:

* MAE;
* RMSE;
* MAPE;
* R²;
* Pearson;
* bias;
* macro-AOI RMSE;
* metriche per AOI.

M2.4 — Analisi delle differenze

Confrontare:

* prestazioni globali;
* prestazioni per territorio;
* stabilità tra fold;
* comportamento sui valori estremi;
* generalizzazione geografica.

Deliverable

M2_V1_V2_PAIRED_BENCHMARK.md

M2_MODEL_COMPARISON.csv

Criterio di completamento

Deve essere possibile stabilire se V2 migliori rispetto a V1, con quali regressori e in quali territori.

Le conclusioni devono distinguere risultati osservati e robustezza statistica.

⸻

MILESTONE 3 — Benchmark delle regression head

Obiettivo

Individuare la regression head più efficace utilizzando una rappresentazione fissa degli embedding.

Il benchmark principale utilizzerà GeoTESSERA V2 con aggregazione densa.

M3.1 — Baseline lineare

Addestrare Ridge.

Verificare la quantità di informazione MSA accessibile attraverso una relazione lineare.

M3.2 — Gradient boosting

Confrontare:

* HistGradientBoosting;
* XGBoost.

Utilizzare una ricerca degli iperparametri controllata.

M3.3 — MLP

Addestrare una rete fully connected.

Partire da architetture relativamente semplici.

Valutare:

* numero di layer;
* dimensione dei layer;
* attivazioni;
* learning rate;
* weight decay;
* dropout;
* early stopping.

Recuperare e confrontare i risultati MLP storici V1.

M3.4 — CNN E2

Verificare prima l’architettura esistente.

Se compatibile con feature dense, eseguire il probing sullo stesso dataset.

Se l’architettura richiede input spaziali, non modificarla silenziosamente per adattarla a un vettore.

In tal caso, documentare la differenza e sviluppare una variante distinta, identificata con un nuovo nome sperimentale.

M3.5 — Confronto equo

Tutti i regressori devono utilizzare:

* stesse celle;
* stesso target;
* stessi split;
* stessa rappresentazione informativa;
* stessa procedura di valutazione.

Deliverable

M3_REGRESSION_HEAD_BENCHMARK.md

Criterio di completamento

Identificare il miglior regressore secondo il protocollo di validazione prestabilito.

Non selezionare il modello sulla base del risultato ottenuto sul benchmark esterno.

⸻

MILESTONE 4 — Diagnostica e miglioramento delle prestazioni

Obiettivo

Comprendere perché i regressori commettano errori e quali interventi possano migliorare la generalizzazione.

M4.1 — Analisi degli errori

Studiare:

* distribuzione dei residui;
* bias;
* errori per decile MSA;
* errori per AOI;
* sovrastima dei valori bassi;
* sottostima dei valori alti.

M4.2 — Analisi del domain shift

Confrontare le distribuzioni delle feature tra AOI.

Valutare se i territori con errori elevati presentino distribuzioni differenti rispetto al training.

M4.3 — Ottimizzazione MLP/CNN

Analizzare:

* learning curves;
* overfitting;
* underfitting;
* stabilità tra seed;
* regolarizzazione;
* capacità del modello.

M4.4 — Ablation study sulle feature

Confrontare, in esperimenti separati:

* mean;
* mean + standard deviation;
* eventuali ulteriori statistiche descrittive.

Non modificare il benchmark principale retroattivamente.

M4.5 — Valutazione delle pressioni ambientali

Esaminare se alcune pressioni utilizzate da GLOBIO siano scarsamente rappresentate negli embedding satellitari.

L’eventuale introduzione di covariate aggiuntive deve costituire un esperimento separato.

Deliverable

M4_ERROR_ANALYSIS.md

M4_ABLATION_STUDY.md

Criterio di completamento

Individuare le principali cause plausibili degli errori e verificare sperimentalmente gli interventi correttivi.

⸻

MILESTONE 5 — Spatial CNN sugli embedding GeoTESSERA

Obiettivo

Sviluppare una CNN che utilizzi direttamente la disposizione spaziale degli embedding GeoTESSERA, evitando la loro aggregazione preventiva in un singolo vettore medio.

Questa milestone è distinta dal benchmark con aggregazione densa.

M5.1 — Concetto architetturale

La pipeline tradizionale è:

GeoTESSERA → aggregazione densa → 128 feature → regressore → MSA.

La nuova pipeline sarà:

GeoTESSERA → patch spaziale di embedding → CNN 2D → pooling → regression head → MSA.

GeoTESSERA rimane congelato.

Viene addestrata soltanto la CNN e la relativa regression head.

M5.2 — Costruzione delle patch

Per ogni cella GLOBIO:

1. Recuperare la geometria della cella.
2. Identificare le tile GeoTESSERA corrispondenti.
3. Recuperare gli embedding nativi.
4. Ricostruire la loro disposizione spaziale.
5. Applicare una maschera geografica.
6. Produrre un tensore compatibile con la CNN.

La patch dovrà conservare, per quanto possibile, la risoluzione nativa degli embedding.

Non utilizzare una media preventiva.

M5.3 — Gestione delle differenti risoluzioni

GLOBIO e GeoTESSERA hanno risoluzioni differenti.

Una cella GLOBIO può contenere centinaia di embedding.

Il modello deve quindi apprendere una funzione che associ una rappresentazione spaziale ad alta risoluzione a un singolo valore MSA.

Valutare:

* patch di dimensioni uniformi;
* padding;
* maschere;
* masked pooling;
* gestione dei CRS;
* gestione delle celle ai bordi delle tile.

La strategia scelta deve preservare la corretta corrispondenza geografica.

M5.4 — Architettura iniziale

Sviluppare una CNN 2D leggera.

Architettura indicativa:

Input spaziale → Conv2D → Conv2D → Masked Global Pooling → MLP → MSA.

La configurazione definitiva dovrà essere scelta dopo aver verificato la forma effettiva degli embedding e i vincoli computazionali.

M5.5 — Pilot sperimentale

Non avviare immediatamente il training sull’intero dataset.

Costruire inizialmente un pilot con 5.000–10.000 celle distribuite su più AOI.

Utilizzare split geografici.

Confrontare la CNN spaziale con le baseline dense sulle stesse celle.

M5.6 — Analisi computazionale

Stimare:

* memoria delle patch;
* spazio disco;
* throughput di lettura;
* memoria GPU;
* tempo per epoca;
* tempo di costruzione delle patch.

Utilizzare:

* lazy loading;
* cache per tile;
* mini-batch;
* checkpoint;
* mixed precision, se compatibile;
* GPU.

M5.7 — Ablation study spaziale

Confrontare:

A. Embedding medi + MLP.

B. Embedding medi + HistGB.

C. Patch spaziali + CNN 2D.

D. Patch spaziali appiattite + MLP, se computazionalmente sostenibile.

L’obiettivo è distinguere il vantaggio dell’informazione spaziale da quello dell’architettura.

Deliverable

M5_SPATIAL_CNN_PROTOTYPE

M5_PATCH_DATASET

M5_SPATIAL_CNN_REPORT.md

Criterio di completamento

La CNN spaziale deve essere addestrabile e valutabile su un dataset geograficamente coerente.

Il pilot deve stabilire se la rappresentazione spaziale offre un vantaggio rispetto alle feature aggregate.

Non è necessario ottenere un miglioramento per considerare valido l’esperimento scientifico.

⸻

MILESTONE 6 — Validazione geografica avanzata

Obiettivo

Valutare la robustezza dei modelli su territori differenti da quelli utilizzati per l’addestramento.

M6.1 — Leave-One-AOI-Out

Valutare una procedura nella quale una AOI viene esclusa dal training e utilizzata per la valutazione.

Ripetere per più AOI.

La selezione degli iperparametri deve avvenire mediante una procedura interna che non utilizzi la AOI di test.

M6.2 — Confronto delle architetture

Confrontare:

* miglior modello lineare;
* miglior boosting;
* miglior MLP;
* CNN E2;
* CNN spaziale.

Utilizzare lo stesso protocollo geografico.

M6.3 — Robustezza statistica

Utilizzare bootstrap spaziale o procedure appropriate per dati geograficamente dipendenti.

Riportare intervalli di confidenza delle differenze di prestazione.

M6.4 — Benchmark Liguria

Mantenere Liguria Levante come benchmark storico.

Non utilizzarla ripetutamente per selezionare architetture o iperparametri.

Per una valutazione finale realmente indipendente, predisporre un territorio non ancora utilizzato durante lo sviluppo.

Deliverable

M6_SPATIAL_GENERALIZATION_REPORT.md

Criterio di completamento

Identificare quali modelli generalizzano meglio e con quale livello di robustezza.

⸻

MILESTONE 7 — Consolidamento e documentazione scientifica

Obiettivo

Trasformare gli esperimenti in una pipeline stabile, riproducibile e utilizzabile per il progetto BIOMAP-FM.

M7.1 — Consolidamento del codice

Organizzare moduli riutilizzabili per:

* lettura GeoTESSERA;
* costruzione delle feature dense;
* costruzione delle patch;
* training;
* validazione;
* inferenza;
* esportazione dei risultati.

M7.2 — Registry degli esperimenti

Registrare tutti i run in un sistema unico.

Ogni esperimento deve essere identificabile e riproducibile.

M7.3 — Report finale

Produrre un documento contenente:

1. Metodologia.
2. Dataset.
3. Esperimenti.
4. Risultati.
5. Analisi statistica.
6. Analisi computazionale.
7. Limiti.
8. Conclusioni.
9. Sviluppi futuri.

M7.4 — Selezione del modello operativo

Selezionare la configurazione più appropriata considerando:

* accuratezza;
* generalizzazione;
* stabilità;
* costo computazionale;
* complessità;
* riproducibilità.

Il modello più complesso non deve essere preferito automaticamente.

Deliverable

BIOMAP_FINAL_TECHNICAL_REPORT.md

BIOMAP_REPRODUCIBLE_PIPELINE

⸻

5. Strategia di ottimizzazione computazionale

Le risorse computazionali devono essere gestite in modo efficiente.

La priorità è evitare ricalcoli inutili.

Aggregazione densa

Riutilizzare le feature V1 esistenti quando verificabili.

Ottimizzare l’estrazione V2 mediante elaborazione per tile e cache persistenti.

Regression head tabellari

Una volta disponibili le feature dense, Ridge, HistGB, XGBoost e MLP devono condividere lo stesso dataset.

Non ripetere l’estrazione degli embedding per ogni regressore.

CNN spaziale

Le patch devono essere costruite mediante una pipeline separata, riutilizzando il codice di lettura geospaziale.

Non caricare l’intero dataset in memoria.

Utilizzare mini-batch e caricamento progressivo.

Benchmark computazionali

Prima di ogni esecuzione estesa:

1. Eseguire uno smoke test.
2. Misurare il throughput.
3. Stimare tempo e memoria.
4. Verificare la disponibilità delle risorse.
5. Avviare il run completo soltanto dopo la validazione tecnica.

⸻

6. Regole operative per Codex e Claude

Questo documento costituisce la roadmap di riferimento del progetto.

Gli agenti devono utilizzarlo per mantenere continuità tra le sessioni di sviluppo.

Prima di ogni attività

1. Consultare la roadmap.
2. Identificare la milestone corrente.
3. Verificare lo stato delle milestone precedenti.
4. Analizzare il codice esistente.
5. Evitare implementazioni duplicate.

Durante lo sviluppo

* Non modificare esperimenti storici.
* Non sovrascrivere artefatti.
* Non introdurre cambiamenti metodologici non documentati.
* Non confondere risultati di validation e test.
* Non effettuare fine-tuning di GeoTESSERA.
* Non utilizzare il benchmark esterno per ottimizzare direttamente gli iperparametri.
* Documentare le modifiche al protocollo.

Dopo ogni attività

Aggiornare un file di avanzamento:

BIOMAP_DEVELOPMENT_STATUS.md

Il file deve contenere:

* milestone corrente;
* attività completate;
* attività in corso;
* problemi aperti;
* decisioni tecniche;
* risultati;
* prossimi step.

Ogni run deve essere collegato ai relativi artefatti.

⸻

7. Priorità immediate

Le prossime attività devono essere eseguite nel seguente ordine.

PRIORITÀ 1

Completare l’aggregazione densa GeoTESSERA V2 utilizzando l’algoritmo V1.

PRIORITÀ 2

Costruire il dataset appaiato V1–V2 e confrontare i regressori.

PRIORITÀ 3

Analizzare i risultati e identificare la migliore configurazione V2.

PRIORITÀ 4

Recuperare i risultati MLP storici e analizzare la CNN E2.

PRIORITÀ 5

Ottimizzare le regression head mantenendo invariata la rappresentazione densa.

PRIORITÀ 6

Sviluppare un prototipo di CNN spaziale sugli embedding nativi.

PRIORITÀ 7

Effettuare validazione geografica avanzata e consolidare i risultati.

⸻

8. Obiettivo scientifico finale

Il progetto deve rispondere a quattro domande fondamentali.

RQ1 — Encoder

GeoTESSERA V2 produce embedding più informativi di GeoTESSERA V1 per la previsione dell’MSA, quando entrambi utilizzano lo stesso metodo di aggregazione?

RQ2 — Regression head

Quale regressore riesce a sfruttare meglio le informazioni contenute negli embedding GeoTESSERA?

RQ3 — Rappresentazione spaziale

La disposizione geografica degli embedding contiene informazioni predittive che vengono perse attraverso l’aggregazione mediante media?

RQ4 — Generalizzazione

Quale combinazione di encoder, rappresentazione e regressore offre le prestazioni più robuste su territori geograficamente differenti?

⸻

9. Istruzione finale agli agenti

La roadmap deve essere trattata come riferimento operativo persistente.

Non avviare contemporaneamente tutte le milestone.

Procedere in modo incrementale, verificando i risultati di ogni fase prima di passare alla successiva.

La prima attività obbligatoria è il completamento del benchmark GeoTESSERA V1–V2 con aggregazione densa identica.

Successivamente, utilizzare le feature dense per confrontare Ridge, HistGradientBoosting, XGBoost, MLP e CNN, verificando preliminarmente la compatibilità dell’architettura CNN con tale rappresentazione.

Soltanto dopo il consolidamento di questi risultati procedere allo sviluppo della CNN spaziale sugli embedding non aggregati.

Ogni decisione deve essere motivata da evidenze sperimentali.

Obiettivo conclusivo: costruire una pipeline di probing geospaziale accurata, efficiente, riproducibile e capace di generalizzare su territori differenti, identificando il contributo effettivo dell’encoder, della rappresentazione e della regression head alla previsione dell’MSA GLOBIO4.

Come utilizzarlo operativamente

Suggerisco di inserire questo documento nel repository come BIOMAP_DEVELOPMENT_ROADMAP.md e di farlo leggere a entrambi gli agenti prima di iniziare una nuova sessione.

Il senior developer potrà utilizzarlo per verificare la correttezza delle scelte architetturali, mentre Codex e Claude potranno lavorare sulle singole milestone mantenendo una visione comune del progetto.

La priorità rimane M1: completare l’aggregazione densa V2. Non anticiperei lo sviluppo della CNN spaziale finché non avremo stabilito quanto migliora V2 utilizzando la stessa rappresentazione di V1.

Un’ultima precisazione: per la futura CNN spaziale, il vantaggio scientifico non sarà semplicemente utilizzare più embedding, ma verificare se conservarne la disposizione geografica permetta di prevedere meglio l’MSA rispetto alla loro media. Questa distinzione dovrà guidare la progettazione degli esperimenti successivi.