# Report tecnico — Correzioni della preparazione dati MSA

**Progetto:** BioMap / TESSERA V2, confronto con il riferimento TESSERA V1 Liguria.  
**Data:** 4 novembre 2026.  
**Versione analizzata prima dell'intervento:** `b0783a6`.  
**Commit delle correzioni:** [`46692f4`](https://github.com/ShimOne420/biomap-tessera-v2-finetuning/commit/46692f40953cfd78303072e26c10a77b675fb0a2) e [`6f5d28b`](https://github.com/ShimOne420/biomap-tessera-v2-finetuning/commit/6f5d28bbfce5c721c87ac97824f3f4b977590d31).

## 1. Esito dell'intervento

La preparazione MSA è stata modificata per ridurre lavoro ripetuto su rete e disco, elaborando insieme celle vicine e condividendo i blocchi raster Sentinel già letti. Sono stati aggiunti profilazione, avanzamento periodico, limiti alle risorse, gestione controllata degli errori e recupero delle scritture interrotte.

Il profilo `configs/msa_v1_liguria.yaml` attiva automaticamente il nuovo motore. Il comando di preparazione, gli ingressi principali del repository e le dipendenze installate rimangono compatibili.

**Risultato verificato:** 41 test superati sul Mac e nella CI CPU Windows/Linux; nella fixture di otto celle le aperture raster passano da 362 a 70, conservando gli stessi array e gli stessi input del modello.

**Risultato ancora da misurare:** celle/ora e durata dell'intera campagna sulla rete del PC universitario. La riduzione delle aperture non viene presentata come un'accelerazione temporale già dimostrata.

## 2. Problema iniziale e diagnosi

Il comando interessato era:

```powershell
python scripts/probe_msa.py prepare --config configs/msa_v1_liguria.yaml --data-root ../data
```

La velocità segnalata era di circa quattro celle ogni trenta minuti, cioè otto celle/ora. Applicando questa velocità costante a 132.900 celle si ottengono 16.612,5 ore, circa 692 giorni continuativi. Questa è una semplice estrapolazione, non una misura della campagna completa.

L'inattività della RTX 5080 è coerente con il codice: `prepare` acquisisce target e osservazioni, legge raster, applica maschere e scrive serie su disco. Il calcolo degli embedding con TESSERA avviene successivamente. L'intervento si concentra quindi su I/O e organizzazione del preprocessing.

La diagnosi deriva dall'ispezione del codice precedente. Non erano disponibili misure dettagliate dei singoli stadi sul PC universitario prima della correzione; non è quindi possibile assegnare retroattivamente una percentuale esatta del rallentamento a ciascuna causa.

| Collo di bottiglia precedente | Comportamento osservabile nel codice | Conseguenza |
|---|---|---|
| Aperture raster ripetute | `Reader.values` riapriva ogni asset per ogni cella e scena | Ripetizione di aperture e accesso ai metadati degli stessi COG |
| Campionamento puntuale | `rasterio.sample` richiedeva finestre 1×1 per i punti | Lavoro ripetuto su blocchi che possono contenere molti punti e celle |
| Cache confinata alla cella | Giorni/serie erano riusabili nella stessa cella | Mancava il riuso persistente delle finestre fra celle vicine |
| Cataloghi decodificati ripetutamente | Il YAML già condivideva i JSON per AOI, ma ogni cella li ricaricava | Ripetizione di parsing e costruzione delle serie temporali |
| Controllo disco ripetitivo | `common.budget` scandiva ricorsivamente la data-root prima di molte scritture | Costo crescente con il numero di file accumulati |
| Esecuzione seriale | Una cella alla volta e letture delle bande in sequenza | Latenza di rete poco sovrapposta |
| Retry e attese | Retry esterni più retry GDAL; timeout raster non espliciti | Attese difficili da interpretare e potenzialmente moltiplicate |
| Visibilità insufficiente | Avanzamento soprattutto alla conclusione di una cella | Difficile distinguere lavoro in corso, rete lenta e blocco |

La condivisione dei cataloghi STAC **era già presente** nel profilo Liguria: non è corretto attribuire al vecchio codice una nuova ricerca remota completa per ogni cella. La nuova correzione evita soprattutto la ripetizione del caricamento/decodifica e dell'organizzazione dei cataloghi all'interno dello stesso processo.

## 3. Perimetro dei file modificati

| File | Tipo di intervento |
|---|---|
| [`src/biomap_v2/data.py`](src/biomap_v2/data.py) | Integrazione dei motori, reader condiviso, HTTP/SAS, retry, profilazione e gestione del ciclo di preparazione |
| [`src/biomap_v2/preparation.py`](src/biomap_v2/preparation.py) | Nuovo orchestratore a lotti, concorrenza, mosaicking per cella, scritture recuperabili e lock OS |
| [`src/biomap_v2/raster_cache.py`](src/biomap_v2/raster_cache.py) | Nuovo lettore di finestre native e cache condivisa RAM/disco |
| [`src/biomap_v2/preparation_profile.py`](src/biomap_v2/preparation_profile.py) | Nuovi tempi, contatori, heartbeat, ETA e contabilità incrementale del disco |
| [`src/biomap_v2/common.py`](src/biomap_v2/common.py) | Valori predefiniti e validazione dei parametri I/O |
| [`configs/msa_v1_liguria.yaml`](configs/msa_v1_liguria.yaml) | Attivazione del motore ottimizzato e parametri commentati |
| [`tests/test_preparation_performance.py`](tests/test_preparation_performance.py) | Venti casi di test aggiuntivi, comprese le parametrizzazioni |
| [`README.md`](README.md), [`README_RUN_MSA_V1_V2.md`](README_RUN_MSA_V1_V2.md) | Collegamenti, istruzioni di aggiornamento/ripresa e descrizione della nuova preparazione |
| [`PREPARAZIONE_MSA_PRESTAZIONI.md`](PREPARAZIONE_MSA_PRESTAZIONI.md) | Guida operativa con diagnosi, rischi, comandi e interpretazione dei report |
| [`VALIDATION.md`](VALIDATION.md) | Evidenze dei test e verifiche ancora necessarie |

I nuovi moduli e i test iniziano con i commenti descrittivi richiesti: scopo, fasi, input/output, parametri, risorse e ripresa. Non sono stati aggiunti nuovi script principali. SQLite, threading e gestione dei processi usano la libreria standard; non occorre installare nuove dipendenze.

## 4. Correzione dell'orchestrazione: lotti di celle vicine

**Riferimenti:** `data.prepare`, `data._prepare_run`, `preparation.acquire_optimized`, `BatchAcquirer`.

Il nuovo flusso verifica dapprima le serie complete già disponibili. Le celle ancora da acquisire vengono ordinate per AOI e prossimità geografica, quindi suddivise in lotti: il valore iniziale è 16 celle. L'ordinamento riguarda soltanto l'I/O; il manifest finale viene ricostruito nell'ordine delle celle originario.

Per ciascun lotto si riuniscono le coordinate dei punti e si richiedono le bande della stessa scena insieme. Il reader può così condividere metadati e finestre raster fra più celle. I cataloghi vengono organizzati una volta per AOI nei tre flussi originali: Sentinel-2, Sentinel-1 ascendente e Sentinel-1 discendente.

Il mosaicking rimane **indipendente per cella**. Le scene vengono considerate nello stesso ordine per nuvolosità e identificativo; una cella che ha già tutti i punti validi smette di ricevere contributi dalle scene successive, anche se altre celle del lotto devono continuare. Questo evita che il raggruppamento introduca osservazioni differenti.

## 5. Correzione delle letture raster e cache condivisa

**Riferimenti:** `raster_cache.NativeReader`, `BlockStore`, `key_for`, `metadata_from`.

### 5.1 Finestre sulla griglia originale

Il lettore calcola gli stessi indici riga/colonna dei punti e li raggruppa in finestre native, con lato massimo di 512 pixel. Legge dati e maschere, poi estrae gli stessi pixel per i punti richiesti. Non interpola e non ricampiona i raster.

Per preservare i risultati ai bordi dei pixel mantiene la trasformazione delle coordinate per singola cella e la suddivisione in gruppi di 256 punti usata dal campionamento rasterio. Gestisce esplicitamente NoData, maschere e coordinate fuori dal raster. I piani di coordinate sono riusati quando griglia e punti coincidono.

### 5.2 Cache a due livelli

I blocchi vengono conservati in RAM e, se previsto dal budget, su disco in `msa_v1_liguria/observation_cache/native_v1`. L'indice SQLite registra metadati, dimensioni, checksum e informazioni per la sostituzione LRU dei blocchi meno recentemente utilizzati.

La chiave identifica scena, collezione, banda, data, metadati/versione disponibili e URL dell'asset privo della query SAS. La posizione della finestra e i metadati raster entrano nell'identità del blocco. Il rinnovo di una firma temporanea non provoca quindi un nuovo download degli stessi pixel.

Quando tutti i blocchi richiesti sono disponibili, una lettura successiva può evitare completamente firma SAS e apertura remota. Un test verifica questo comportamento anche dopo la chiusura e ricreazione del reader.

Le soglie predefinite sono 256 MiB per i blocchi in RAM e 8 GiB per i blocchi su disco. Sono limiti della cache dei blocchi, **non** della memoria totale del processo o di tutti i metadati SQLite. La LRU rimuove soltanto blocchi di propria competenza; non elimina le serie complete delle celle.

## 6. Parallelizzazione controllata e risorse GDAL

**Riferimenti:** `BatchAcquirer.read_bands`, `BatchAcquirer.close`, `NativeReader._remote`.

È stato introdotto un `ThreadPoolExecutor` con quattro worker iniziali, configurabile da uno a otto. Le letture delle bande si sovrappongono, mentre i risultati vengono ricomposti nell'ordine originale delle bande. Mosaicking e assemblaggio delle serie restano deterministici.

Ogni dataset GDAL appartiene al worker che lo apre. Lock dedicati proteggono token, indice/cache e accessi allo stesso asset. In caso di errore i task non iniziati vengono annullati; quelli già attivi vengono attesi prima della chiusura delle risorse.

`GDAL_CACHEMAX` viene impostato a 64 MiB una sola volta nel thread principale e ripristinato alla chiusura. Questo parametro è globale al processo: modificarlo dentro contesti di worker sovrapposti avrebbe potuto causare ripristini incoerenti.

Più worker non implicano necessariamente più velocità: rete, server, CPU e disco possono diventare il limite. Per questo la concorrenza è moderata e misurabile.

## 7. Correzione dei controlli sul disco

**Riferimento:** `preparation_profile.SpaceGuard`.

Il percorso ottimizzato sostituisce le scansioni ricorsive ripetute con un inventario iniziale della data-root e un conteggio aggiornato dopo scritture, sostituzioni e rimozioni dei propri file. A ogni scrittura viene comunque controllato lo spazio libero effettivo. L'inventario viene ricostruito a ogni invocazione, senza riusare un totale persistente potenzialmente obsoleto.

Restano il limite globale di 50 GiB e la riserva minima di 20 GiB. Gli 8 GiB della cache nativa rientrano nel limite globale. Una maggiorazione prudenziale di 32 MiB considera i piccoli file di report e metadati, mentre l'occupazione SQLite viene aggiornata separatamente.

Il commit `6f5d28b` corregge anche una possibile condizione di concorrenza: durante un inventario lungo, l'heartbeat può sostituire atomicamente un file temporaneo fra il controllo di esistenza e la lettura della sua dimensione. Il codice tollera questa specifica scomparsa (`FileNotFoundError`) invece di interrompere la preparazione. Un test riproduce il caso.

Il controllo non prenota spazio contro altri programmi esterni: la data-root va dedicata al run durante l'acquisizione. Non è ancora dimostrato che tutte le serie della campagna entrino nei 50 GiB.

## 8. Timeout, retry e gestione delle sessioni

**Riferimenti:** `data.session`, `data.query`, `Reader.signed`, `Reader.clear_tokens`, `Reader.close`, `NativeReader._remote`.

| Aspetto | Correzione |
|---|---|
| HTTP STAC | Timeout distinti connessione/lettura; sessione chiusa anche in caso di errore |
| Retry POST STAC | Le POST di ricerca vengono incluse esplicitamente fra i metodi ritentabili |
| Errori HTTP temporanei | Due retry aggiuntivi predefiniti per 429/500/502/503/504, rispettando `Retry-After` |
| Letture raster | Tre tentativi esterni predefiniti, backoff e invalidazione dei token dopo errori raster |
| Retry annidati | Retry interno GDAL disattivato nel motore ottimizzato |
| Timeout raster | Connessione 10 s e operazione HTTP 30 s, configurabili |
| Token SAS | Cache condivisa sincronizzata; sessioni HTTP per thread quando necessarie |
| Diagnosi degli errori | Contatori per fallimenti/retry e messaggi con scena e banda |
| Informazioni riservate | Filtri sui logger rasterio usati per rimuovere query degli URL; verbose HTTP/GDAL disattivato |

I trenta secondi non sono una scadenza per l'intera banda: una banda può comportare più operazioni HTTP. Anche attese richieste dal server possono estendere la durata complessiva. Un errore di rete non viene trasformato in un'osservazione valida o in un valore imputato.

## 9. Profilazione, avanzamento ed ETA

**Riferimento:** `preparation_profile.PreparationProfile`.

Ogni invocazione crea una cartella `diagnostics/TIMESTAMP_prepare`. La configurazione effettiva viene salvata in `config.json`; `profile.json` e `progress.json` riportano stato, fase, tempi, contatori e avanzamento. `assets.csv`, scritto alla conclusione o all'interruzione gestita, aggrega informazioni per scena/banda senza URL firmati. Gli hash dei cataloghi usati sono registrati separatamente per AOI.

I principali tempi distinguono acquisizione/validazione dei target, query STAC, caricamento dei cataloghi, richiesta SAS, apertura raster, lettura remota, trasformazione delle coordinate, preprocessing, lettura/scrittura cache, checksum e controlli disco.

L'heartbeat stampa ogni dieci secondi celle elaborate, celle riusate, celle nuove, velocità, ETA e fase corrente. La velocità usa le celle nuove elaborate: il riconoscimento rapido di migliaia di celle già in cache non viene contato come nuovo throughput di acquisizione. Il contatore delle celle nuove include anche le celle elaborate ma risultate non eleggibili, riportate separatamente fra le escluse.

Prima del primo lotto concluso l'ETA può essere sconosciuto. Successivamente è una stima media del lavoro residuo: cambia con territorio, scene disponibili, rete e riuso della cache.

Per interpretare i report occorre distinguere:

- **Tempo di parete** e somma dei tempi dei worker: questi ultimi possono sovrapporsi e superare la durata del processo.
- **Aperture raster** e richieste HTTP Range: una singola apertura/lettura può generare più richieste interne a GDAL.
- **Byte decodificati** e traffico HTTP: i dati/mask decodificati non misurano i byte compressi trasferiti.
- **Rete** e cache: il caricamento di un catalogo locale non è una nuova interrogazione remota.

## 10. Interruzione, ripresa e conservazione della cache

**Riferimenti:** `preparation_lock`, `atomic_pair`, `recover_pair`, `completed_cell`, `BatchAcquirer.finish_cell`.

Un blocco del sistema operativo impedisce due `prepare` concorrenti nello stesso namespace: usa `msvcrt` su Windows e `fcntl` sui sistemi POSIX. Il blocco viene rilasciato alla chiusura del processo; il solo file `prepare.lock` residuo non significa che un processo sia ancora attivo.

Prima di riusare una serie completa vengono controllati protocollo, anno, campionamento, geometria e checksum. Le vecchie serie compatibili vengono conservate senza riscriverle. I giorni parziali già salvati vengono verificati anche per scene, forma, tipo e maschera.

Le nuove scritture usano file temporanei e una procedura recuperabile:

1. Scrittura dell'NPZ di staging.
2. Calcolo del checksum e scrittura del manifest pending.
3. Pubblicazione atomica dell'NPZ tramite sostituzione del nome.
4. Pubblicazione atomica del manifest.
5. Al riavvio, eventuale completamento della coppia pendente dopo verifica del checksum.

I due file non vengono rinominati con un'unica operazione indivisibile: il manifest pending consente di recuperare l'intervallo fra le due pubblicazioni. Non vengono accettati silenziosamente file corrotti o privi di manifest.

Dopo il consolidamento della serie completa gli array vengono riletti e confrontati con quelli in memoria; soltanto dopo questa verifica vengono rimossi gli intermedi giornalieri ridondanti di quella cella. Una cella esclusa mantiene gli intermedi utili alla diagnosi/ripresa.

Ctrl+C viene registrato come `interrupted`; gli altri errori come `failed`. La ripresa avviene con lo stesso comando. Le verifiche coprono interruzioni di processo e transazioni incomplete simulate; non costituiscono una prova di recupero da guasti fisici del disco o perdita di alimentazione.

## 11. Parametri aggiunti e compatibilità

`common.config` inserisce i valori predefiniti prima degli override CLI e controlla intervalli e compatibilità. I vecchi YAML restano sul motore `legacy`; il profilo Liguria sceglie esplicitamente `optimized`, che richiede i cataloghi condivisi.

| Parametro `acquisition.*` | Valore nel profilo Liguria | Funzione |
|---|---:|---|
| `engine` | `optimized` | Selezione del percorso; `legacy` resta il riferimento |
| `shared_catalogs` | `true` | Cataloghi annuali condivisi per AOI |
| `batch_cells` | 16 | Celle raggruppate per l'I/O |
| `io_workers` | 4 | Letture di bande concorrenti, ammesse da 1 a 8 |
| `block_cache_gib` | 8 | Budget dei blocchi persistenti |
| `memory_cache_mib` | 256 | Budget dei blocchi in RAM |
| `progress_seconds` | 10 | Intervallo dell'heartbeat |
| `read_attempts` | 3 | Tentativi esterni raster |
| `connect_timeout_seconds` | 10 | Timeout di connessione |
| `read_timeout_seconds` | 30 | Timeout di lettura/operazione HTTP |
| `http_retries` | 2 | Retry aggiuntivi per HTTP STAC/token |

Cambiare questi parametri non cambia i punti o i target. Può però modificare l'ordine I/O, il consumo delle risorse e la probabilità di incontrare timeout. Il ritorno a `legacy` è utile per verifiche controllate, non per accelerare la campagna.

## 12. Vincoli scientifici mantenuti

La correzione riguarda l'acquisizione e la sua affidabilità, non l'esperimento statistico. Restano:

- 132.900 celle: 127.909 sviluppo in 17 AOI e 4.991 test Liguria.
- Tre fold geografici originali; nessuna nuova validation casuale e nessun uso del test per scegliere parametri.
- Target GLOBIO4 overall MSA 2020, senza imputazioni o sostituzioni.
- Gli stessi 25 punti inizialmente campionati per cella; il filtro originale può conservarne solo quelli validi.
- Le stesse scene/date, priorità di mosaicking, orbite e soglie di copertura.
- Le bande S2 nello stesso ordine e le stesse esclusioni SCL: 0, 1, 2, 3, 8, 9.
- Controllo di finitezza delle bande S2 e clipping originale in `uint16`.
- Per S1, intensità positiva e finita, stessa conversione `10*log10`, offset/scalatura/clipping e storage `int16` tramite `power_to_storage`.
- Giorno dell'anno limitato a 365, inclusa la stessa gestione del 31 dicembre 2020.
- Stessa normalizzazione, binning, pooling e ingresso dell'encoder nei componenti successivi.

Le membership CSV, il riferimento storico, i target e il codice di selezione/valutazione non sono stati modificati dai due commit. Se la copertura Sentinel riduce le celle eleggibili, il controllo del benchmark completo continua a rifiutare il risultato come replica equivalente. Umbria nord resta fuori dallo sviluppo.

Rimane il limite scientifico già dichiarato: il riferimento V1 usava medie dense, mentre questo esperimento V2 usa punti stratificati. Accelerare l'I/O non elimina quella differenza di supporto spaziale e non dimostra una superiorità dell'encoder V2.

## 13. Test ed evidenze quantitative

Sono stati aggiunti 20 casi di test, portando la suite da 21 a 41. La fixture principale contiene otto celle vicine, due date S2 con priorità di scene diverse, entrambi i flussi S1, nuvole, NoData, valori estremi e una cella deliberatamente non eleggibile in entrambi i percorsi. STAC e firma degli URL sono sostituiti da fixture controllate; le letture rasterio dei GeoTIFF locali sono reali.

| Verifica | Evidenza |
|---|---|
| Array delle serie | Uguaglianza esatta di bande, maschere, punti e date fra riferimento e ottimizzato |
| Celle/target | Stesse celle eleggibili e stesso CSV salvo il campo checksum dell'archivio |
| Input TESSERA | Uguaglianza esatta dopo normalizzazione/binning e degli embedding di uno student ridotto |
| Griglie e NoData | Due CRS; bordi; NoData assente, numerico e NaN |
| Concorrenza/lotti | Risultati invarianti cambiando confini dei lotti e numero di worker; limite di concorrenza rispettato |
| Cache esistente | Hash e timestamp delle vecchie serie complete immutati |
| Cache persistente | Nuovo reader riusa i blocchi senza firma o accesso remoto |
| Integrità | Corruzione rilevata; LRU rimuove soltanto i propri blocchi |
| Ripresa | Giorni già validi riusati dopo interruzione; coppie pending recuperate |
| Rete | Rinnovo dell'indirizzo firmato dopo errore; timeout/paginazione STAC e policy retry POST controllati |
| Disco | Un solo inventario per più scritture; tollerata la sostituzione di un temporaneo durante la scansione |
| Informazioni riservate | Firma rimossa dai messaggi rasterio di test |

Nel confronto principale sono stati osservati:

| Misura sulla fixture | Riferimento | Ottimizzato |
|---|---:|---:|
| Aperture raster | 362 | 70 |
| Invocazioni del componente catalogo sostituito dalla fixture | 16 | 2 |
| Tempo totale di un'esecuzione locale registrata | 0,497 s | 0,828 s |

Le aperture diminuiscono di circa l'80,7%, ma sui piccoli file locali il nuovo percorso è più lento a causa dell'overhead di cache, SQLite e thread. **Non si tratta quindi di una prova di speedup temporale**, né di una misura delle richieste HTTP reali. Il vantaggio atteso riguarda la riduzione del lavoro remoto ripetuto; va quantificato con i report del PC universitario.

La CI del codice finale è riuscita su entrambi i sistemi: [run Windows/Linux `41815671119`](https://github.com/ShimOne420/biomap-tessera-v2-finetuning/actions/runs/41815671119). Su Windows include il setup PowerShell documentato. La CI CPU non certifica throughput Planetary Computer, CUDA o occupazione finale della campagna.

## 14. Uso operativo e verifiche ancora aperte

Dopo aver fermato il vecchio processo e ricevuto il codice con `git pull --ff-only origin main`, si riprende dalla stessa cartella dati:

```powershell
& .\.venv\Scripts\python.exe scripts\probe_msa.py prepare --config configs\msa_v1_liguria.yaml --data-root ..\data
```

Non occorre cancellare la cache valida, reinstallare dipendenze o scaricare nuovamente i pesi. Per i passaggi completi e la lettura dei report vedere [la guida prestazioni](PREPARAZIONE_MSA_PRESTAZIONI.md).

Sul PC universitario devono essere misurati il throughput dopo i primi lotti, il tasso di hit della cache, gli errori/retry, l'occupazione finale e la stabilità su più AOI. I confronti di velocità devono distinguere cache fredda/calda e celle nuove/riusate; le condizioni possono variare lungo la campagna.

La cache assume che un asset identificato dal catalogo congelato rimanga stabile. Un contenuto sostituito dal fornitore dietro lo stesso URL e senza aggiornamenti dei metadati non è rilevabile su un hit che evita la rete. Quando un asset viene riaperto, i metadati raster vengono confrontati con quelli conservati.

L'intervento consegna un'acquisizione con riuso fra celle, meno operazioni ripetute e misure leggibili per valutarne il comportamento reale. Non è stata eseguita da questa macchina l'intera campagna annuale di 132.900 celle e non viene dichiarata una sua durata finale già verificata.
