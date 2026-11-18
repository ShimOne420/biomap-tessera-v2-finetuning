# Benchmark del campionamento — protocollo preregistrato

Base: `feature/finetuning-v2` @ `07dca6b`. Namespace nuovo: **sampling-v2-nested-r1**. I fatti hardware e le metriche storiche provengono dal brief PC del 18/11/2026; non sono misure effettuate da questo Mac. Il software prepara un esperimento frozen: non carica GeoTESSERA, non contatta Planetary Computer e non acquisisce Sentinel.

## Popolazioni e modello

Sviluppo: 127.909 celle, 17 AOI, fold originali 0/1/2. Liguria esclusa da training, selezione, cache e metriche. Il reader V2Source originale verifica l'integrità della campagna globale secondo il proprio contratto, ma il benchmark invoca `patches` solo per celle di sviluppo. Indice esistente in sola lettura; nessuna ricostruzione.

La CNN originale resta immutata: 128 input, due PartialConv 3×3, 64/32 canali, PixelNorm, GELU, pooling pesato per contributi validi, testa hidden64/dropout0,1/output lineare. Campo ricettivo nominale 5×5, una predizione e MSE per cella. Batch4, accumulo16, AdamW lr0,001/weight_decay0,0001, clipping1, massimo50 epoche, patience8, FP32. I worker sono configurabili, esplicitamente spawn; default2, zero nello smoke.

## Geometrie

S64 = 1×8×8, M128 = 2×8×8, S144 = 1×12×12, M64 = 4×4×4. Seed geografico20260925, indipendente dai seed della CNN. FULL-SUBSET è un sottoinsieme globale di circa11.000 celle, con quote AOI proporzionali arrotondate mediante maggiori resti e ranking hash stabile, mai usando i target. Ogni fit ne usa l'intersezione con il proprio training; la validation resta completa. FULL-SUBSET non partecipa alla scelta primaria perché cambia la popolazione.

Il campionatore riceve soltanto ID/centro/geometria. Finestre candidate centrate su pixel eleggibili; massima copertura, poi vicinanza al centroide per la prima finestra e massima distanza minima dalle precedenti per le successive. Spareggio BLAKE2b con seed, ID, tile e posizione. Gli hash vengono calcolati solo sui pari merito della distanza: ottimizzazione equivalente all'ordinamento completo, senza cambiare il criterio.

La scelta avviene sull'intera cella, non una volta per ogni tile. Dopo una scelta vengono esclusi i footprint già utilizzati, anche sulle tile sovrapposte. Per griglie nello stesso CRS si confrontano gli intervalli nativi; fra CRS diversi si trasformano soltanto i poligoni di footprint per il controllo di intersezione. Non si trasformano o interpolano gli embedding. Intersezioni di area≤1e-6m² sono trattate come contatto numerico di bordo, non sovrapposizione di terreno.

Finestre con maschere parziali sono ammesse. Possono esserci meno patch/pixel del massimo, senza eliminare celle: `insufficient` e `partial_mask` lo dichiarano. Il minimo è un parametro diagnostico, default1. Una cella senza alcun supporto determina errore, non imputazione. FULL mantiene i contributi originali, inclusi overlap: non si riscrive la baseline per deduplicarla.

Il RF completo si calcola con erosione5×5 della **maschera effettiva**, considerando esterno e buchi invalidi: 8×8 piena→16, 12×12→64, 4×4→0. Copertura = contributi selezionati/contributi storici; non viene spacciata per percentuale di area unica quando il denominatore storico contiene overlap. Si riportano distanze fra patch/dal centroide e casi multi-tile/multi-UTM.

## Manifest e cache

Ogni piano ha codice/sorgente/checkpoint/seed e checksum. Ogni patch mantiene tile, CRS/transform, finestra, maschera bit-packed, posizione nella geometria originale, flat index, righe/colonne, coordinate WGS84 e peso esatto. Le finestre non vengono unite artificialmente durante il pooling.

Il contratto `spatial_sampling_requests_from_plan` ricostruisce gruppi e patch per l'acquisizione futura senza campionare nuovamente; verifica hash della geometria, griglia, posizioni, maschere e pesi. L'output rimane `approved_for_acquisition=false`. Le maschere temporali Sentinel non sono disponibili negli embedding e non vengono dichiarate ricostruite o verificate: servirà una successiva acquisizione autorizzata con il protocollo denso storico e la prova di parità degli input.

La cache contiene float32 non normalizzati. I contributi con identica coppia tile/posizione vengono memorizzati una volta e condivisi fra configurazioni/fold/seed. Non si deduplicano valori uguali con provenienze diverse. Shard atomici, fsync, rilettura esatta, SHA256 e marker; SQLite proprio del benchmark in sola lettura ai worker. I momenti per configurazione sono calcolati una volta, ma il normalizzatore viene stimato solo sulle righe del training del fit. Nessuna statistica globale viene usata come normalizzatore.

FULL legge gli embedding originali senza duplicare circa49GB. La validation interna può usare uno scratch completo di **un solo gruppo**, default20GiB. Si sostituisce soltanto questo scratch derivato, creato dal nuovo workspace e verificato come proprio; le sorgenti, gli indici, le cache e i checkpoint preesistenti non vengono eliminati. I memmap/SQLite vengono chiusi prima della sostituzione. Il checkpoint esterno è valutato una volta, dopo la decisione; l'early stopping interno richiede invece validation a ogni epoca.

## Nested, epoche e inizializzazione

Per ogni outer k, gli altri due gruppi si alternano come inner training/validation. Nessuna AOI attraversa i ruoli. Validation interna densa e macro-AOI RMSE scelgono l'epoca; normalizzazione sul training campionato. Ogni ruolo ha controlli espliciti di ID/AOI/partition.

Per ogni configurazione e seed: `refit_epochs=(best_epoch_1+best_epoch_2+1)//2`, epoche numerate da uno e arrotondamento dei mezzi verso l'alto. Il refit ricrea la CNN con il seed previsto e ricalcola il normalizzatore su tutto il training esterno pertinente. Non parte da un checkpoint interno. La ripresa dello **stesso refit** ripristina soltanto i suoi checkpoint, optimizer e RNG.

Prima della valutazione esterna si salva la decisione interna con sigillo SHA256 separato. La valutazione verifica il sigillo e lega i checkpoint di refit alla decisione. Modifiche successive determinano errore. I checkpoint per epoca sono immutabili; file privi di marker non contano, una nuova pubblicazione non li sovrascrive. Una deadline può interrompere un minibatch: la ripresa ricalcola l'epoca incompleta dallo stato precedente.

## Selezione esplorativa e seed

Controlli FULL42/FULL43 completati prima dei candidati42; margine = differenza assoluta dei loro macro-AOI RMSE interni appaiati. Il margine è sigillato e non aumenta dopo aver visto i candidati. Non si usa l'oscillazione fra epoche di circa0,014 del run storico come stima del rumore fra seed.

Fra i candidati entro il margine, ordinamento: costo raw previsto crescente; a pari costo, p90 degli eccessi di RMSE per AOI crescente; poi RMSE medio e ID. Conferma43/44 e FULL44: la differenza media sui tre seed deve restare entro il margine già fissato. In caso negativo si prova il successivo candidato inizialmente ammissibile, senza allargare la soglia. È possibile che non ne rimanga nessuno.

La raccomandazione globale usa solo risultati interni, con ogni cella predetta in due contesti di training; questi non sono due nuove celle indipendenti. Le decisioni per outer sono separate e non vengono cambiate dalla raccomandazione globale. Le metriche OOF esterne dei vincitori stimano la **procedura di selezione**, non certificano retroattivamente un unico campionamento globale scelto a posteriori.

Delta rispetto a FULL, variabilità fra seed, intervalli bootstrap e costo sono separati. Il bootstrap legacy è pooled; il nuovo macro-AOI ricampiona le differenze per AOI,2000 repliche e seed20260925. Intervalli condizionati ai modelli stimati, non prova formale di non inferiorità. R² ed errori estremi richiedono revisione umana: nessuna approvazione automatica del fine-tuning.

## Budget e autorizzazioni

Lo screening esegue sette fit su una sola suddivisione interna del training dell'outer0. Non usa i target outer0 per eliminare configurazioni e non elimina globalmente candidati per gli altri outer.12ore è una deadline operativa con resume, non una garanzia di completamento.

Prima del nested, dry-run con budget_id, numero di fit, costi e dati mancanti. Il comando `run` richiede quel budget_id con `--approved-plan` dopo approvazione separata. Nessuna campagna reale è stata autorizzata da questa consegna. Lo smoke reale ha limite900s e zero worker figli del processo supervisionato; al timeout non è dichiarato passato.

I raw futuri vengono stimati sui pixel effettivi di sviluppo con2956byte/pixel dichiarati dal PC. C:57GB e F:52GB sono rilevazioni del brief, non capacità da sommare; il software legge lo spazio corrente per ciascuna destinazione. Considerare manifest, staging, scratch, checkpoint e20GiB di riserva per volume. M128/S144 possono essere eseguibili nel benchmark ma non ancora acquisibili. Compressione lossless/compattazione devono essere misurate su dati autorizzati prima di farne un criterio di ammissione; nessun risparmio inventato.

Un blocco finale per modalità è solo una raccomandazione per il futuro pilot: non viene caricato l'encoder né alterato il default del percorso fine-tuning esistente. Restano necessari approvazione del campionamento, verifiche reali di input/VRAM/disco e autorizzazione all'acquisizione.
