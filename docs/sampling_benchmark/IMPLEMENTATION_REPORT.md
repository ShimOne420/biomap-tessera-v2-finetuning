# BIOMAP — report di implementazione del benchmark di campionamento

Data: 18 novembre 2026. Base verificata: **07dca6b100797e1d77af4cd5264200ab3d457032**, documento dei fatti PC. Branch di consegna: **feature/sampling-benchmark**, destinazione PR: **feature/finetuning-v2**. Stato: infrastruttura implementata e verificata su fixture CPU; **validazione su dati reali e RTX 5080 da eseguire sul PC**, dopo autorizzazione. Non è un report di risultati MSA.

## 1. Isolamento e provenienza

È stato controllato il checkout locale, effettuato fetch e verificato il commit richiesto anche tramite GitHub. Lo sviluppo avviene in un worktree nuovo, senza aggiornare il checkout precedente o accedere al PC. La base contiene già le correzioni Spatial CNN per schema parquet e celle multi-UTM. I fatti del PC sono letti dal brief versionato, non dedotti da directory presenti sul Mac.

Nessun dato Sentinel/V1/V2 o checkpoint esterno è stato scaricato. I test costruiscono minuscoli artefatti r2 sintetici e usano il reader originale. La campagna attiva, i checkpoint storici e l'indice universitario non sono accessibili da questo ambiente e non sono stati modificati.

## 2. Mappa delle modifiche

L'unica nuova entry point è `scripts/sampling_benchmark.py`. `configs/spatial_sampling_benchmark.yaml` documenta percorsi, sampling, cache, training, budget e decisione. I percorsi PC stanno nel runbook e vengono passati come argomenti: nessun percorso personale è cablato nel runtime.

Il package `src/biomap_sampling` separa responsabilità mantenendo una sola CLI: `geometry.py` seleziona finestre native e calcola diagnostiche; `data.py` prepara manifest, cache e ponte spaziale; `training.py` addestra la testa originale; `study.py` governa nested/seed/decisioni; `metrics.py` distingue macro e pooled; `reporting.py` calcola budget e report; `state.py` centralizza configurazione e controlli. Ogni modulo, script e test nuovo ha un commento iniziale su scopo, input/output, parametri, risorse e ripresa.

`src/biomap_finetune/data.py` riceve soltanto due funzioni opt-in di ricostruzione dal manifest; le modalità di acquisizione e il percorso attivo restano invariati. Il workflow CPU esistente viene abilitato anche alle PR verso `feature/finetuning-v2`. README e AGENTS indirizzano al nuovo percorso. Nessun file di `src/biomap_spatial`, CODE_FILE r1, lock o regola EOL viene modificato.

## 3. Campionamento e integrità geometrica

S64 usa una finestra 8×8, M128 due 8×8, S144 una 12×12, M64 quattro 4×4. Il seed 20260925 entra nello spareggio hash; i target non vengono passati al campionatore. Si esplorano le finestre candidate sull'intera cella, privilegiando copertura valida e poi dispersione geografica. La prima finestra favorisce la vicinanza al centroide; le successive massimizzano la distanza minima dalle precedenti. Questo criterio è dichiarato nel protocollo, non adattato ai risultati.

La disponibilità viene aggiornata su tutte le tile dopo ciascuna patch. In overlap non si campiona ripetutamente lo stesso terreno. In multi-UTM vengono trasformati i poligoni dei footprint soltanto per verificare intersezioni: valori e griglie restano nativi. Si documenta la tolleranza di area 1e-6 m² per contatti numerici. Non vengono creati mosaici o contesti esterni.

Le celle piccole mantengono maschere parziali e, se necessario, meno patch. `insufficient` e `partial_mask` sono diagnostiche esplicite, non motivi di esclusione. Supporto vuoto causa errore. FULL e FULL-SUBSET conservano il contributo storico completo, inclusi overlap originali. FULL-SUBSET è AOI-stratificato, target-independent e diagnostico; non concorre alla raccomandazione primaria.

Il campo ricettivo completo viene calcolato sulla maschera reale con erosione 5×5: 16 posizioni in 8×8 piena, 64 in 12×12, zero in 4×4. Le frazioni sono rapportate ai contributi e non dichiarate percentuali di area geografica unica.

## 4. Manifest, cache e contratto futuro

Il piano registra identità sorgente/codice/checkpoint, cella/AOI/fold, seed, tile, patch sorgente, CRS, trasformazione, finestra, maschera, coordinate WGS84, righe/colonne, flat index, posizione originale degli embedding e peso. Gli shard hanno checksum e marker atomici; il piano complessivo collega queste identità. La funzione di ricostruzione legge il piano già scelto: non ripete il sampling.

Il ponte verifica SHA della geometria originale, griglia e corrispondenza esatta fra posizioni, maschere e pesi. I test confrontano i float32 originali con quelli ricostruiti dalla cache e le predizioni della stessa CNN. Questo soddisfa il contratto spaziale su fixture; **non dimostra parità temporale Sentinel reale**. Gli embedding non contengono le maschere temporali originarie: la futura acquisizione richiederà una verifica distinta e autorizzata. Il ponte non è un nuovo downloader e non abilita l'encoder.

La cache conserva vettori float32 non normalizzati, condividendo una sola copia di ogni coppia tile/posizione riutilizzata fra configurazioni. Provenienze diverse con numeri identici non vengono confuse. Un nuovo SQLite compatto serve ai worker, senza riscrivere il reader o l'indice universitario. I momenti per cella vengono precomputati, mentre media/scala del normalizzatore derivano soltanto dal training del fit corrente.

La geometria immutabile viene verificata con SHA una volta per tile e riutilizzata durante la preparazione: calcolarne l'hash per ciascuna patch/cella avrebbe introdotto letture ripetute inutili su tutta la popolazione. Un test conta queste verifiche e impedisce la regressione. Il manifest distingue l'ID originale della patch SQLite dall'ID della nuova patch campionata. Questo riuso non altera né coordinate né valori e non evita la verifica iniziale del reader.

La baseline FULL legge gli originali, evitando una seconda copia dell'intera campagna. La validation densa interna usa uno scratch di un solo gruppo, massimo20GiB predefiniti, nel nuovo output. Il passaggio ad altro gruppo chiude memmap/SQLite e rimuove solo lo scratch con marcatore di proprietà del benchmark. Non viene cancellata alcuna cache preesistente o checkpoint. L'early stopping richiede validation completa a ogni epoca; la valutazione outer avviene dopo la decisione interna.

## 5. Nested, refit e separazione scientifica

Per ogni outer0/1/2 gli altri due gruppi si alternano come training/validation interna. ID e AOI sono controllati prima del training. Una vista di sviluppo impedisce che FULL incorpori le righe Liguria esposte dal reader globale. Normalizzazione e early stopping macro-AOI RMSE sono interni. Le stesse condizioni valgono per FULL e candidati.

Le due migliori epoche interne sono numerate da uno. Il refit usa `(e1+e2+1)//2`: 2 e3 danno3; 2 e8 danno5. Ricrea la CNN con il seed previsto e un normalizzatore del training esterno. Non parte dai pesi interni. Checkpoint di inizializzazione e hash dei training ID rendono ispezionabile questo contratto; la ripresa ripristina solo lo stesso fit/refit.

FULL42/43 calibrano il margine assoluto, sigillato prima dei candidati del relativo outer. Si provano i candidati42, poi43/44 e FULL44 per confermare quelli ammessi, in ordine di costo. A pari costo si usa stabilità geografica e poi RMSE. La soglia non aumenta dopo un risultato sfavorevole. La raccomandazione globale usa soltanto predizioni interne; ogni cella vi appare in due contesti, non come due osservazioni indipendenti. Le decisioni per outer rimangono separate e sigillate prima del refit e della valutazione esterna.

Il margine da due inizializzazioni è esplorativo. Delta, variabilità fra seed, bootstrap e costo sono riportati separatamente; **non è una prova di non inferiorità**. R², errori estremi, AOI e fattibilità disco richiedono approvazione umana. Un risultato senza candidato ammissibile è valido. L'OOF dei vincitori valuta la procedura di selezione, non una configurazione globale scelta retroattivamente.

## 6. Metriche e provenienza storica

Sono esportati RMSE/MAE macro-AOI, metriche pooled, R², bias, Pearson, quantili p50/p90/p95/p99 dell'errore assoluto, frequenze strettamente oltre0,3/0,4, tabelle per AOI e fasce MSA. R²/Pearson non definiti vengono marcati null. I confronti verificano stessi ID, target, AOI e fold, senza restringere silenziosamente la popolazione all'intersezione.

Il bootstrap pooled esistente resta separato dal nuovo bootstrap macro sulle differenze per cluster AOI,2000 repliche di default. Le predizioni interne concatenate per la raccomandazione mantengono l'ID originale e un ID del contesto, evitando di dichiarare raddoppiato il numero di celle indipendenti.

Doctor verifica la provenienza del riferimento0,17602 soltanto se trova predizioni canoniche, selection e marker CV coerenti con SHA. Ricomputa macro e fold, riportando anche se il valore arrotondato coincide. Sul Mac questi artefatti reali non esistono: la provenienza empirica rimane **da verificare sul PC**, pur essendo il numero documentato dal brief. Non è un target da riprodurre esattamente: la selezione storica usava il fold valutato, mentre il nuovo FULL è nested. Un singolo fold non è confrontato con la media di tutti.

## 7. Budget e operatività

Screening: sette fit su training interno gruppo1/validation gruppo2 dell'outer0, mai usando quest'ultimo per scartare candidati. La deadline12h interrompe in modo controllato e conserva le epoche complete. Non riduce epoche, pazienza o candidati. Nested:42 fit interni iniziali; almeno48 se nessun candidato passa e si completa FULL44,60 con una conferma completa; massimo96 interni e18 refit, a cui aggiungere cache/audit/valutazioni. Il preventivo è un intervallo operativo, non una durata promessa.

I costi temporali iniziali usano il riferimento PC520–590s/epoca, non una misura della nuova cache. Dopo il pilot il software usa le history disponibili, dichiarandone la provenienza. Cache/hash/valutazione finale non misurati rimangono null. Il report salva risorse per fit: picco CUDA e picco RAM del processo, con nota che la RAM dei worker e il picco incrementale del singolo fit non sono misurati separatamente.

Per i raw futuri si usa la stima dichiarata2956byte/pixel. Lo spazio corrente viene letto per ogni destinazione, con riserva20GiB; C e F non si sommano. I raw M128/S144 non sono ammessi sui dischi descritti dal brief senza liberare spazio o dimostrare altra soluzione. Compressione e cache di attivazioni non vengono presentate come risparmi provati. Un blocco finale encoder è soltanto una proposta per un futuro pilot, senza alterare il percorso fine-tuning corrente.

## 8. Ripresa, Windows e monitor

La CLI espone doctor/prepare/cache/smoke/screen/run/report/monitor, `--dry-run`, `--resume`, override YAML e gate `--approved-plan`. Le root devono essere nuove, possedute e disgiunte dai percorsi protetti. Lock di workspace impediscono scritture concorrenti. Il monitor è read-only e apre/chiude subito progress.json: fase, fold, epoca, celle, velocità ed ETA della fase corrente. Non confondere ETA di epoca e durata dell'intero nested.

Scritture atomiche con rilettura/checksum, optimizer e RNG preservati, marker per epoca; l'epoca incompleta viene ricalcolata. I file orfani non vengono trattati come checkpoint validi o sovrascritti. Su Windows si usa spawn esplicito, timeout dei worker e un ambiente separato. Lo smoke usa zero worker DataLoader ed è supervisionato da un processo con limite900s: il timeout è un mancato completamento, non un successo.

## 9. Validazione osservata

La suite specifica del benchmark verifica determinismo, overlap, multi-UTM, maschere, RF, roundtrip sorgente/cache/manifest, forward identico della CNN, worker spawn, split, epoche, inizializzazione, ripresa, corruzione, metriche e percorso nested sintetico completo. Verifica anche che cambiare i target outer non cambi la decisione interna e che decisioni manomesse o approvazioni mancanti blocchino l'esecuzione.

Le verifiche iniziali hanno individuato e corretto: confronto parquet sensibile ai soli indici pandas; filtro FULL refit che, senza guardia, avrebbe incluso righe esterne; un nome locale che oscurava l'helper delle distanze. La guardia degli split ha bloccato il problema FULL prima del training. Nessuna correzione ha coinvolto la pipeline storica. Un tentativo iniziale con worker spawn e thread nativi predefiniti non è terminato: è stato interrotto; la ripetizione con thread limitati e esecuzione nativa è passata. Cambiando entrambe le condizioni non si può attribuire con certezza il blocco a una sola causa. Non sono stati aggiunti skip o tolleranze per nasconderlo.

Gli esiti numerici finali di suite completa, clone pulito e CI sono registrati nella sezione di consegna sotto. I test sintetici non misurano throughput reale, consumo disco dei manifest reali, CUDA5080 o accuratezza MSA.

## 10. Consegna e verifiche ancora necessarie

La sequenza PC è nel [runbook](WINDOWS_RUNBOOK.md): clone/venv/test → doctor/dry-run → smoke autorizzato → piano/cache → screening autorizzato → budget/approvazione separata → nested/refit/outer → report. Il [protocollo](PROTOCOL.md) è la fonte delle regole scientifiche.

**Da eseguire sul PC:** compatibilità dell'indice originale con le EOL del clone, audit canonico storico, roundtrip su celle reali incluse multi-UTM, tempi/cache/RAM/VRAM, smoke entro15min, costi per volume, screening e solo dopo approvazione nested. Non certificare questi aspetti dai test CPU. La futura acquisizione richiede inoltre identità di tile, maschere temporali, normalizzazione Sentinel e prova di parità con il percorso storico.

### Registro finale di validazione

Commit codice verificato: **5f57ea66b8c29141286a155fae2870beb8d5dc8d**. macOS26.6.2 arm64, Python3.12.14, Torch2.10.0, CPU, OMP/OPENBLAS/MKL a un thread. È stato riutilizzato un ambiente di sviluppo già installato, puntando gli import al nuovo checkout: non è una prova di reinstallazione completa sul Mac.

| Verifica osservata | Esito |
|---|---|
| Suite specifica sampling | **29 passati**, zero falliti/saltati,18,74s |
| Suite CPU completa del repository | **272 passati,1 saltato**, zero falliti,80,11s |
| Check ambiente, sintassi e import | **Passato**,158 file Python analizzati, backward CPU passato |
| Diff contro07dca6b: biomap_spatial, biomap_v2, lock, EOL | **Nessuna modifica** |
| Contratto r1 con fixture versionata SHA/EOL | Incluso nella suite completa e passato |

L'unico skip della suite Mac è `tests/test_dense_fast.py:277`, blocco di condivisione specifico Windows. La suite emette1447 warning, prevalentemente nelle fixture geospaziali preesistenti: non sono stati disabilitati o trasformati in skip. Il check osserva anche il caso S1 di frontiera già documentato; suite verde non è garanzia di aritmetica bit-identica fra piattaforme. MPS non richiesto e non usato.

Clone pulito e CI tre sistemi sono riportati qui al termine delle rispettive esecuzioni. Nessun risultato remoto è presunto.
