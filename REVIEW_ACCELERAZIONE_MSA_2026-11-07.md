# Revisione indipendente dell'accelerazione della preparazione MSA

**Data:** 7 novembre 2026. **Repository:** `ShimOne420/biomap-tessera-v2-finetuning`.

**Versione esaminata:** [`2e24f979ec78a9509ee1797ef130914b829ef4f8`](https://github.com/ShimOne420/biomap-tessera-v2-finetuning/commit/2e24f979ec78a9509ee1797ef130914b829ef4f8), confrontata con `dc8604a`, ultima versione della precedente analisi. Il [diff completo](https://github.com/ShimOne420/biomap-tessera-v2-finetuning/compare/dc8604abedd68733aa5a5a06e80b27acf302b0e9...2e24f979ec78a9509ee1797ef130914b829ef4f8) contiene due commit del 6 novembre: `a4d8adf` e `2e24f97`, con modifiche a 13 file.

## 1. Valutazione complessiva

La strategia dell'altro agente ha una motivazione tecnica solida: affronta il costo dei moltissimi piccoli file temporanei e organizza l'acquisizione in modo da riusare spazialmente una lettura su migliaia di celle. La parallelizzazione per giorni e aree è coerente con l'indipendenza di queste unità; il mosaicking resta separato per cella.

Ho confermato che il nuovo codice supera i **55 test ordinari** e che la CI del commit esaminato è riuscita su Windows e Linux. Non risultano modificati manifest delle celle, target, split, encoder o protocollo di probing. Non emerge dal diff una scorciatoia scientifica basata sulla riduzione dei campioni o sull'imputazione di osservazioni.

La revisione individua però **cinque criticità riprodotte localmente**, relative a validazione dei checkpoint, contabilità disco, stato finale, avanzamento dopo un riavvio e consolidamento dei file compressi. Non dimostrano che i dati reali già acquisiti siano errati; impediscono di considerare automaticamente conclusa e certificata la campagna sulla base del solo stato o dell'ETA.

**Il risultato delle cinque ore non è verificato.** Il report dell'altro agente dichiara una stima di 10–14 ore per le celle mancanti e non il completamento della campagna in cinque ore. Non ho accesso al processo attivo o ai dati del PC universitario.

Questa consegna aggiunge documentazione, evidenze e casi diagnostici separati: non modifica i moduli operativi usati dal run.

## 2. Fonti e metodo

Sono stati utilizzati:

- Il codice GitHub fissato al commit indicato, senza confonderlo con il vecchio checkout locale.
- Il report dell'agente, [`REPORT_ACCELERAZIONE_PREPARE_MSA.md`](https://github.com/ShimOne420/biomap-tessera-v2-finetuning/blob/2e24f979ec78a9509ee1797ef130914b829ef4f8/REPORT_ACCELERAZIONE_PREPARE_MSA.md).
- Il confronto dei file e della configurazione rispetto a `dc8604a`.
- La suite ordinaria rieseguita sul Mac: **55 passed, 113 warnings, 38,05 s**, CPU/Python 3.12.
- La [CI del commit esaminato, run `42252057039`](https://github.com/ShimOne420/biomap-tessera-v2-finetuning/actions/runs/42252057039), riuscita per entrambi i sistemi operativi.
- Cinque prove diagnostiche locali su fixture sintetiche e directory temporanee, senza rete e senza dati universitari.

Le misure Windows citate nel report precedente — compreso il confronto su 1.200 celle — sono **dichiarazioni documentate dall'altro agente**, non misure ripetute in questa revisione. Nel repository non sono presenti i corrispondenti report JSON/log grezzi necessari a verificarle indipendentemente. L'assenza di questi artefatti non dimostra che le misure siano sbagliate, ma limita il livello di conferma.

Il report del 6 novembre contiene inoltre la frase «modifiche ... non ancora ... committate»: è un'indicazione redazionale ormai superata, perché le modifiche sono nei due commit esaminati.

## 3. Che cosa è cambiato

### 3.1 Dal piccolo lotto al lotto di area

Il parametro `batch_cells` passa da **16 a 8.192**. Ho contato le celle nel manifest: le 18 aree hanno fra 4.991 e 7.776 celle. Con questo specifico dataset, quindi, 8.192 è sufficiente a contenere tutte le celle mancanti di una singola area in un lotto.

Prima, un medesimo blocco raster poteva essere richiesto da numerosi piccoli lotti. Adesso tutti i punti dell'area possono essere raggruppati durante la lettura dello stesso asset. Il guadagno deriva dalla maggiore condivisione della lettura, non dalla fusione di celle, dalla media dei target o dalla riduzione dei 25 punti.

Dire che ogni blocco viene letto una sola volta va inteso **per asset e chiamata di acquisizione del lotto**, con il recupero delle letture riuscite durante un retry. Non significa una sola lettura globale fra aree diverse, run diversi o tutti gli asset.

### 3.2 Dai checkpoint per cella ai checkpoint per lotto

La precedente preparazione scriveva intermedi giornalieri separati per ogni cella e poi li cancellava dopo il consolidamento. La nuova versione salva un archivio per coppia sorgente/giorno e per lotto sotto `batch_checkpoints/<area>/<identità_lotto>`.

Se un lotto contiene N celle e D giorni/sorgenti, il numero di archivi giornalieri scende, in ordine di grandezza, da N×D a D. Le coppie finali `series.npz`/`manifest.json` restano individuali: il costo di quei file non scompare.

L'area viene poi consolidata in gruppi di **512 celle**. In questo modo non occorre tenere simultaneamente in RAM tutte le serie annuali dell'area. È una scelta sensata, ma l'attuale modalità di rilettura NPZ comporta lavoro ridondante, descritto nella criticità R5.

Gli intermedi nel vecchio formato vengono ancora riconosciuti e riusati; la migrazione non impone di cancellare i dati già scaricati.

### 3.3 Cache di blocchi disattivata

Nel profilo Liguria `block_cache_gib` passa da 8 a **0** e `memory_cache_mib` da 256 a **0**. Il codice distingue ora esplicitamente una cache disattivata: non apre, svuota o cancella l'indice SQLite già presente.

Questo è coerente con il lotto di area: se una lettura serve contemporaneamente tutti i punti interessati, scrivere e rileggere una cache di blocchi può diventare un costo evitabile. Il report precedente attribuisce proprio a scritture, cache e churn LRU una parte rilevante del lavoro.

Il compromesso è un minore riuso persistente a livello di singolo blocco: dopo un arresto forzato si conservano i giorni completati, ma si può dover ripetere la parte di un giorno non ancora pubblicata. Non si deve generalizzare la scelta «cache zero» a ogni futuro dataset, soprattutto con aree sovrapposte o lotti piccoli.

### 3.4 Concorrenza su due livelli e processi per area

`io_workers` passa da 4 a **8**; viene aggiunto `day_workers=4`. Un pool gestisce le giornate indipendenti, l'altro le letture delle bande. Tenerli separati evita che tutti i worker siano occupati da task che attendono altri task nello stesso pool.

Il nuovo `parallel_prepare.py` avvia processi separati per area con `--parallel-aois`. Con quattro processi e otto worker di lettura per processo, il limite dei task di banda attivi è circa **32**, non 128: i quattro giorni condividono il pool delle otto letture.

Le aree acquisiscono lock distinti. Un prepare completo prende anche i lock delle aree, evitando di scrivere sulle stesse celle insieme ai worker. Il coordinatore prepara i target una volta; i worker per area li leggono senza aggiornare il puntatore ai raster.

I worker non scrivono il manifest globale `cells.csv`. Al termine della modalità parallela completa viene chiamato il prepare globale, che deve verificare e pubblicare il dataset. Le modalità con `--aois` o arresto a tempo non eseguono quel passaggio.

### 3.5 Coordinate, tentativi ed esclusioni

Il reader memorizza la trasformazione PROJ per punti/CRS e raggruppa gli indici dei blocchi con operazioni vettoriali. La trasformazione resta fatta per cella e il campionamento resta sulla griglia nativa. I limiti delle cache di coordinate sono espliciti.

I tentativi raster passano da 3 a **8**. Il retry intercetta anche errori `requests`, aggiunge attesa crescente con jitter e conserva le finestre già assegnate durante la chiamata: un errore successivo non forza a rileggere i blocchi appena ottenuti. Timeout e disattivazione del retry GDAL annidato rimangono.

Le esclusioni vengono rese persistenti con identità di protocollo, anno, campionamento e geometria. Evitare di riscaricare una cella con insufficiente copertura è ragionevole quando cataloghi e regole sono congelati. Un'esclusione non viene però trasformata in una cella valida: il controllo dei conteggi completi resta attivo.

### 3.6 Coordinamento, quote e avanzamento

Il coordinatore aggiunge selezione delle aree, arresto a tempo, log separati e fino a cinque riavvii automatici per area. Assegna una quota iniziale di spazio aggiuntivo ai worker.

Per mostrare avanzamento prima del salvataggio delle celle, stima il lavoro attraverso i giorni conclusi. Il secondo commit esclude i report precedenti all'invocazione e introduce cinque minuti di baseline prima di calcolare velocità/ETA. Sono miglioramenti utili, ma rimangono i difetti di quota e conteggio descritti sotto.

## 4. Mappa delle modifiche nel codice

Tutti i riferimenti seguenti puntano alla versione esaminata, non a un `main` che potrebbe cambiare.

| File | Responsabilità delle modifiche | Giudizio |
|---|---|---|
| [`preparation.py`](https://github.com/ShimOne420/biomap-tessera-v2-finetuning/blob/2e24f979ec78a9509ee1797ef130914b829ef4f8/src/biomap_v2/preparation.py) | Checkpoint di lotto, pool giornaliero, consolidamento, esclusioni, ripresa e lock | Architettura motivata; rafforzare validazione dei checkpoint e lettura compressa |
| [`raster_cache.py`](https://github.com/ShimOne420/biomap-tessera-v2-finetuning/blob/2e24f979ec78a9509ee1797ef130914b829ef4f8/src/biomap_v2/raster_cache.py) | Disattivazione non distruttiva della cache, riuso PROJ, raggruppamento vettoriale, retry | Coerente con grandi lotti; da misurare su nuove condizioni di rete |
| [`parallel_prepare.py`](https://github.com/ShimOne420/biomap-tessera-v2-finetuning/blob/2e24f979ec78a9509ee1797ef130914b829ef4f8/src/biomap_v2/parallel_prepare.py) | Processi, quote, riavvii, avanzamento e lancio della validazione finale | Utile, ma contiene problemi riprodotti di stato, quote e conteggi |
| [`preparation_profile.py`](https://github.com/ShimOne420/biomap-tessera-v2-finetuning/blob/2e24f979ec78a9509ee1797ef130914b829ef4f8/src/biomap_v2/preparation_profile.py) | Etichetta area, giorni, quota locale | Misure più informative; quota locale non equivale a garanzia globale |
| [`data.py`](https://github.com/ShimOne420/biomap-tessera-v2-finetuning/blob/2e24f979ec78a9509ee1797ef130914b829ef4f8/src/biomap_v2/data.py) | Modalità `--aoi`, lock multipli, target in sola lettura, manifest globale separato | Separazione corretta dei ruoli |
| [`cli.py`](https://github.com/ShimOne420/biomap-tessera-v2-finetuning/blob/2e24f979ec78a9509ee1797ef130914b829ef4f8/src/biomap_v2/cli.py), [`common.py`](https://github.com/ShimOne420/biomap-tessera-v2-finetuning/blob/2e24f979ec78a9509ee1797ef130914b829ef4f8/src/biomap_v2/common.py) | Opzioni parallele e nuovi limiti di configurazione | Necessarie; ulteriori controlli degli argomenti sono opportuni |
| [`msa_v1_liguria.yaml`](https://github.com/ShimOne420/biomap-tessera-v2-finetuning/blob/2e24f979ec78a9509ee1797ef130914b829ef4f8/configs/msa_v1_liguria.yaml) | Parametri I/O più aggressivi | Nessuna modifica ai parametri scientifici del profilo |
| [`verify_prepare_equivalence.py`](https://github.com/ShimOne420/biomap-tessera-v2-finetuning/blob/2e24f979ec78a9509ee1797ef130914b829ef4f8/scripts/verify_prepare_equivalence.py) | Riacquisizione separata e confronto esatto con celle esistenti | Strumento utile; richiede condizioni di esecuzione più rigorose |
| Test e tre documenti aggiornati/aggiunti | Equivalenza, ripresa, quota locale, uso e misure dichiarate | Copertura migliorata, ma non completa del coordinatore |

## 5. Criticità riprodotte

Le cinque prove sono disponibili in [`tests/review_acceleration_cases.py`](tests/review_acceleration_cases.py); i risultati sono registrati in [`REVIEW_MSA_2026-11-07_EVIDENZE.json`](REVIEW_MSA_2026-11-07_EVIDENZE.json). I casi dimostrano comportamenti problematici della versione esaminata. Il loro esito positivo significa «caso riprodotto», non «difetto corretto».

### R1 — Validazione insufficiente dei nuovi checkpoint giornalieri

**Priorità alta, integrità degli input.** [`day_task`, righe 252–257](https://github.com/ShimOne420/biomap-tessera-v2-finetuning/blob/2e24f979ec78a9509ee1797ef130914b829ef4f8/src/biomap_v2/preparation.py#L252) riusa un checkpoint se il file esiste. [`consolidate`, righe 300–305](https://github.com/ShimOne420/biomap-tessera-v2-finetuning/blob/2e24f979ec78a9509ee1797ef130914b829ef4f8/src/biomap_v2/preparation.py#L300) controlla identificativi delle scene e forme degli array, ma non il dtype atteso e non un checksum esterno registrato per quel checkpoint.

**Prova:** ho interrotto una fixture dopo il primo giorno e sostituito il checkpoint con un NPZ leggibile di uguale forma/scena, convertendo `bands` da `uint16` a `float32` e modificandone un valore. La ripresa è stata accettata e ha prodotto una serie S2 finale `float32`, senza segnalare la non conformità.

Questo non simula necessariamente un normale bit flip: ZIP dispone di controlli CRC e molti danneggiamenti fisici verrebbero rilevati. Dimostra però che un checkpoint formalmente leggibile ma semanticamente non conforme può essere accettato e poi ricevere un nuovo checksum finale valido. Quel checksum certifica i byte appena scritti, non la correttezza dell'input intermedio.

L'identità del lotto include protocollo, anno, sampling e ID ordinati, ma non geometrie o hash dei cataloghi. I cataloghi congelati e le geometrie immutate riducono il rischio nella campagna attuale; non sostituiscono un legame esplicito nella provenienza.

**Intervento proposto:** validare chiavi, dtype, forme e maschere, registrare checksum/identità dei checkpoint e collegarli a geometrie e cataloghi. Verificare anche le serie reali già prodotte per schema e provenienza, senza presumere che siano errate e senza cancellarle.

### R2 — La quota locale non garantisce il limite globale

**Priorità alta, affidabilità operativa.** La quota è calcolata una volta in [`run_parallel`, righe 125–127](https://github.com/ShimOne420/biomap-tessera-v2-finetuning/blob/2e24f979ec78a9509ee1797ef130914b829ef4f8/src/biomap_v2/parallel_prepare.py#L125). Ogni `SpaceGuard` mantiene poi il proprio inventario iniziale più le proprie scritture.

Quando un'area termina e il suo posto viene riutilizzato, il nuovo processo riceve ancora la quota originaria. I processi già attivi non vedono le nuove scritture degli altri nella loro contabilità. Non esiste una prenotazione globale atomica condivisa.

**Prova:** con un limite di 200 MiB e due posti da 100 MiB, il worker A scrive 90 MiB e termina; C ne prende il posto e scrive 75 MiB, mentre B arriva a 100 MiB. Tutti i controlli locali passano, ma i file totalizzano **265 MiB**. La prova usa file sparsi temporanei e la stessa misura logica delle dimensioni usata dall'inventario.

Il controllo dello spazio fisicamente libero rimane, ma non rende vera l'affermazione che la somma delle quote rispetti sempre il limite globale. Non è una prova che il disco universitario abbia già superato 50 GiB.

**Intervento proposto:** contabilità e prenotazioni centralizzate, aggiornate anche attraverso sostituzioni e riavvii dei worker. Evitare di risolvere tornando a una scansione completa prima di ogni file, che reintrodurrebbe il vecchio collo di bottiglia.

### R3 — `complete` viene scritto prima della validazione finale

**Priorità alta per interpretare la conclusione.** [`run_parallel`, righe 175–182](https://github.com/ShimOne420/biomap-tessera-v2-finetuning/blob/2e24f979ec78a9509ee1797ef130914b829ef4f8/src/biomap_v2/parallel_prepare.py#L175) salva `status=complete` nel report e solo dopo invoca il prepare globale.

**Prova:** ho simulato un fallimento della validazione finale. Il comando solleva correttamente un errore, ma `parallel_progress.json` rimane con stato **`complete`**.

Il controllo finale non viene aggirato e il processo non ritorna successo in questo caso. Il problema è il report, che può far credere conclusa una validazione ancora in corso o fallita. Inoltre una modalità limitata a `--aois` può arrivare a `complete` senza validazione dell'intero benchmark, per scelta esplicita del codice.

**Intervento proposto:** separare `acquisition_complete`, `validating`, `validated`, `failed` e `stopped`; aggiornare il report anche quando il controllo globale fallisce. Per dichiarare conclusa la campagna servono esito positivo del comando completo e manifest globale verificato, non quel solo campo.

### R4 — Conteggi non cumulativi dopo un riavvio

**Priorità media, misure e pianificazione.** [`latest_progress` e `summary`, righe 80–106](https://github.com/ShimOne420/biomap-tessera-v2-finetuning/blob/2e24f979ec78a9509ee1797ef130914b829ef4f8/src/biomap_v2/parallel_prepare.py#L80) considerano soltanto l'ultimo report per area. Al riavvio il nuovo processo conta le celle precedenti come cached, ma il coordinatore somma soltanto le nuove del processo corrente.

**Prova:** un primo tentativo salva 200 celle; il secondo riconosce quelle 200 e ne salva 800. Le celle effettivamente concluse sono 1.000, ma il riepilogo riporta **800**. I blocchi letti scendono analogamente da un totale di 3.000 a **2.000** nel report, perché il primo tentativo non viene sommato. Può rimanere un ETA positivo nonostante l'area sia completata.

La selezione dei report della stessa invocazione e la baseline di cinque minuti sono utili, ma non risolvono questa perdita di storia. Nei profili con più lotti per area la stima basata sul lotto corrente richiede ulteriori cautele; nel manifest attuale tutte le aree entrano nel lotto da 8.192.

**Intervento proposto:** aggregare tentativi e celle concluse rispetto alla membership mancante all'avvio, con stato persistente del coordinatore e senza doppi conteggi. La contabilità dovrebbe restare monotona e distinguere celle, giorni e traffico.

### R5 — Decompressione ripetuta nel consolidamento

**Priorità media, prossima ottimizzazione di prestazioni.** In [`consolidate`, righe 295–305](https://github.com/ShimOne420/biomap-tessera-v2-finetuning/blob/2e24f979ec78a9509ee1797ef130914b829ef4f8/src/biomap_v2/preparation.py#L295) ogni gruppo di 512 celle riapre tutti gli NPZ giornalieri. Ogni membro `bands`/`valid` viene inoltre letto una volta per verificarne la forma e un'altra per selezionare le posizioni. Un accesso a un membro NPZ compresso materializza l'intero array, non solo quelle righe.

**Prova:** con 513 celle, cioè due gruppi, il membro `bands` viene materializzato quattro volte; lo stesso avviene per `valid`. Per un'area di 7.776 celle ci sono 16 gruppi: il codice può quindi materializzare ogni membro **32 volte per giorno**.

Questo non cambia i valori, ma può diventare un collo di bottiglia CPU/disco, soprattutto dopo un aumento della banda. La quantità di lavoro ridondante è verificata; il suo peso temporale sulla campagna reale non è ancora misurato.

**Intervento proposto:** eliminare intanto le doppie letture dentro ogni accesso e valutare checkpoint con accesso effettivo per blocchi di celle, per esempio `.npy` con memmap o un formato chunked. Qualunque cambio di formato deve conservare identità, validazione e ripresa.

## 6. Copertura delle verifiche ed equivalenza scientifica

Il diff non modifica `msa_v1_liguria_cells.csv`, il riferimento storico, `target_data.py`, `encoder.py`, `student.py`, `benchmark.py` o `experiments.py`. Restano 127.909 celle di sviluppo, 4.991 Liguria, tre fold geografici e 25 punti campionati per cella; il filtro dei punti validi rimane quello originale. Non sono cambiati SCL, conversione S1, priorità delle scene o gestione del giorno dell'anno.

I 55 test ordinari coprono array, dtype, scene, ripresa, compatibilità con vecchi intermedi, lock, cache disattivata e policy di rete. Il parametro `batch_cells=8192` compare nei test, ma la fixture ha comunque otto celle: prova l'accettazione della configurazione, **non** il carico completo di 8.192 celle. La prova diagnostica aggiuntiva da 513 celle esercita il confine dei gruppi di consolidamento, non un'acquisizione reale di area intera.

Il report dell'agente dichiara 1.200 celle reali senza differenze: 300 per Abruzzo, Appennino, Basilicata e Belluno. È una verifica significativa se la riacquisizione è stata realmente fresca, ma copre quattro aree su 18, circa lo 0,9% del benchmark e non il test Liguria. Non è una prova esaustiva su tutto il dataset.

Lo script di equivalenza è utile, con queste limitazioni:

- Riusa una workdir già popolata: una seconda esecuzione potrebbe confrontare candidati prodotti da una versione precedente invece di esercitare il nuovo motore. Per ogni revisione serve una workdir nuova e un commit registrato.
- Il contatore dei file «identici byte per byte» confronta i checksum dichiarati nei manifest; `compare_cell` non ricalcola indipendentemente entrambi gli hash dei file.
- Il percorso di lavoro non deve sovrapporsi alla sorgente neppure attraverso la sottocartella aggiunta per area. Il controllo esistente rifiuta una workdir interna alla sorgente, ma non descrive tutti i casi di alias/sovrapposizione.
- Il coordinatore completo non ha un test automatico end-to-end di processi, riavvii, quote e fallimento della validazione finale. I nuovi casi diagnostici evidenziano proprio alcune lacune di quel livello.

Per estendere la verifica proporrei campioni preregistrati e distribuiti fra aree, bordi raster, CRS, copertura e vecchi intermedi. L'inclusione di Liguria deve servire alla verifica degli input, senza usare le metriche del test per scegliere iperparametri del modello. Umbria nord resta esclusa.

## 7. Come leggere i numeri di velocità

Il report precedente dichiara:

| Misura | Risultato dichiarato | Livello di conferma in questa revisione |
|---|---|---|
| Vecchio run lungo | 22.240 nuove celle in 41,8 ore, circa 532 celle/h | Documento consultato; log originali non disponibili qui |
| Due processi | 23,8 blocchi/s | Come sopra |
| Quattro processi | 23,5 blocchi/s | Come sopra |
| Otto processi | 13,4 blocchi/s e cinque errori recuperati | Come sopra |
| Confronto reale | 1.200 celle senza differenze | Script ispezionato; confronto non ripetuto su quei dati |
| Campagna residua | Stima 10–14 ore per circa 105.000 celle | Estrapolazione, non tempo finale misurato |

Il plateau fra due e quattro processi e il peggioramento con otto sono compatibili con saturazione e contesa. Una connessione negoziata a 100 Mbps avrebbe un limite teorico di circa 12,5 MB/s, ma il suo stato e il traffico effettivo del PC non sono stati letti direttamente in questa revisione. Le dimensioni dei vecchi blocchi NPZ non equivalgono ai byte HTTP dei COG: la stima di banda è indicativa.

Quattro processi possono aumentare le aree avanzate contemporaneamente, ma i dati dichiarati non dimostrano che completino prima di due. A parità di throughput, due possono ridurre pressione su memoria, disco e server. La scelta va fatta sul tempo completo e sull'affidabilità, non sul numero di processi.

Anche l'affermazione che cinque ore siano impossibili a 100 Mbps è troppo categorica finché non si conosce il numero di blocchi risparmiati dal lotto di area. **Condizionatamente** a 23,5 blocchi/s costanti, cinque ore consentirebbero 423.000 letture. Su 105.000 celle residue significherebbe circa quattro blocchi/cella, prima di considerare il consolidamento. Il modello precedente usa 10,66 blocchi/cella, ma il nuovo riuso mira proprio a diminuire quel rapporto. Non è dimostrato né che scenda abbastanza né che non possa farlo.

I tempi dei worker possono sovrapporsi: le percentuali ricavate dividendo somme di tempi per il tempo di parete non costituiscono automaticamente una partizione esclusiva del runtime. Scaricamento, lettura checkpoint, decompressione, assemblaggio e pubblicazione finale devono essere misurati separatamente.

## 8. Strategia raccomandata

### Fase A — Accertare lo stato reale della campagna

Non aggiornare i moduli Python mentre il processo li sta usando e non cancellare cache o checkpoint. Leggere report e log del run già attivo; un ETA o il solo `status=complete` non sono certificazioni del benchmark.

Da PowerShell, nella cartella del repository:

```powershell
$DataRoot = (Resolve-Path '..\data').Path
$Diag = Join-Path $DataRoot 'msa_v1_liguria\diagnostics'
$Parallel = Get-ChildItem $Diag -Directory -Filter '*_parallel' |
    Sort-Object Name -Descending | Select-Object -First 1
Get-Content (Join-Path $Parallel.FullName 'parallel_progress.json') -Raw
Get-ChildItem $Parallel.FullName -Filter '*.log'
```

Questi comandi leggono lo stato e non lanciano una nuova acquisizione. I log dei worker non sostituiscono l'esito del prepare globale, che è eseguito nel processo coordinatore: conservare anche l'output della finestra principale.

Prima del probing completo verificare il manifest globale, i conteggi 127.909/4.991, le identità e gli hash, la copertura dei target, lo schema delle serie e l'esito positivo dei controlli finali. Se il manifest manca, la preparazione non è pronta per il benchmark. Una futura azione dedicata di sola validazione eviterebbe di dover usare `prepare`, che potrebbe anche acquisire celle mancanti.

### Fase B — Correggere le garanzie operative

Affrontare prima R1–R3: validazione/provenienza dei checkpoint, quota disco realmente globale e stato finale scritto dopo la validazione. Aggiungere prove end-to-end del coordinatore per riavvii, arresti, errori di avvio e fallimenti finali. Rendere cumulativi i contatori di R4.

Fra i controlli ulteriori: rifiutare un numero di processi non positivo (`--parallel-aois -1` oggi può lasciare la coda in attesa senza avviare worker), proteggere il coordinatore da avvii duplicati e assicurare la chiusura dei figli anche per eccezioni diverse da Ctrl+C. Queste sono osservazioni da ispezione, non ulteriori casi sperimentali dichiarati riusciti.

### Fase C — Ottimizzare il collo di bottiglia misurato

Misurare RSS/RAM, CPU, spazio, byte di rete e tempi di consolidamento su almeno un'area grande. Affrontare la decompressione ripetuta di R5 prima di aumentare indiscriminatamente il numero di processi.

Per controllare la rete sul PC, senza modificarne la configurazione:

```powershell
Get-NetAdapter -Physical | Select-Object Name, Status, LinkSpeed, InterfaceDescription
Get-NetAdapterStatistics | Select-Object Name, ReceivedBytes, SentBytes
```

Identificare l'interfaccia realmente usata dal run. Se risulta a 100 Mbps, verificare con il supporto universitario cavo, presa e porta di rete. Non basta la velocità nominale della scheda e non serve cambiare configurazioni amministrative alla cieca.

Dopo un eventuale miglioramento della rete, confrontare due e quattro processi su lavoro residuo comparabile, con cache/checkpoint iniziali documentati. Provare otto solo se il throughput totale cresce senza peggiorare errori, memoria e consolidamento. Le prove su aree diverse non sono confronti perfettamente controllati: costo S2/S1, giorni e copertura possono variare.

### Fase D — Consolidare la riproducibilità e passare a TESSERA

Archiviare commit, configurazione effettiva, hash di membership/cataloghi, output della validazione finale, statistiche per area e artefatti del confronto reale. Non pubblicare URL firmati, chiavi o dati con licenze incompatibili.

Solo dopo l'accettazione dei dati procedere con embedding frozen V2 e probing geografico invariato. L'eventuale fine-tuning viene dopo il confronto frozen. Una preparazione più rapida non dimostra un miglioramento dell'accuratezza del modello e non elimina la differenza di supporto spaziale già nota fra V1 denso e V2 a punti.

## 9. Riprodurre questa revisione

La suite ordinaria si esegue con il comando usuale. I casi diagnostici sono separati dalla raccolta automatica perché documentano difetti attualmente presenti; dopo la loro correzione devono essere trasformati in regressioni che richiedono il comportamento corretto.

```powershell
& .\.venv\Scripts\python.exe -m pytest
& .\.venv\Scripts\python.exe -m pytest -q -s tests\review_acceleration_cases.py
```

Il secondo comando usa solo fixture temporanee e non legge la data-root universitaria. Nei risultati registrati produce cinque righe `AUDIT_EVIDENCE`: checkpoint non conforme accettato, 265/200 MiB, conteggio 800/1.000 dopo riavvio, quattro materializzazioni per due gruppi, stato complete nonostante validazione fallita.

**Giudizio finale:** mantenere la direzione dell'ottimizzazione, perché risolve costi reali della precedente architettura. Prima di dichiarare concluso il benchmark, verificare il run reale e rafforzare le garanzie indicate. I dati disponibili sostengono un miglioramento plausibile e alcuni vantaggi verificati, non un tempo finale di cinque ore già dimostrato né una certificazione integrale delle 132.900 celle.
