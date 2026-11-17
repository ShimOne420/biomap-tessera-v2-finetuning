# Implementazione partial fine-tuning GeoTESSERA V2 Medium

Data: 17 novembre 2026. Destinatario: agente del PC universitario e revisore della PR. **Infrastruttura implementata e verificata su CPU; esecuzione scientifica su input reali/GPU non ancora verificata.** Fattibilità: **SUPPORTED_WITH_CHANGES**, attraverso un package isolato, non modifiche all'estrazione legacy.

## Base Git e isolamento

Sono stati controllati repository, origin, branch, HEAD, worktree, modifiche locali e informazioni remote. Il clone Mac era pulito; è stato eseguito fetch e creato `feature/finetuning-v2` da `1737b1fdd30d5de9a50967363ca39042b26a055b`, con `d1c0d93` verificato come antenato. La base contiene le correzioni schema parquet reale, chiusura memmap Windows e celle multi-UTM. Main verificato era `b7da382d4424c1ddd35a285830f95bb2d42c6436`; non è stato modificato né si è presunto il merge della Spatial CNN.

L'audit architetturale è stato scritto e committato **prima dell'implementazione**, commit `576b8ec`. La PR del fine-tuning dipende dal branch fix della Spatial CNN e deve essere revisionata lì, senza merge automatico. L'elenco aggiornato dei commit della consegna è nella PR; questo report non contiene uno SHA autoreferenziale del proprio futuro commit.

Non è stato contattato il PC universitario. Nessun Sentinel o nuovo checkpoint è stato scaricato, nessuna campagna scientifica o cache massiva è stata avviata. I soli processi Python locali osservati inizialmente erano strumenti dell'editor. Durante la verifica finale il branch fix è avanzato a `36e4bb846f4eb99c4099244a113f269670b33190` con soli aggiornamenti del handoff. Sono stati incorporati senza merge/force push i due commit documentali, tramite cherry-pick (`79d6c7f`, `fe01c50`). La base codice resta invariata a1737b1f.

Il run Spatial universitario e i suoi output non sono stati ispezionati né modificati.

## Audit e prova sul modello reale

È stata trovata una copia locale già materializzata di `student_medium.pt`. Verificati dimensione 84.163.403 byte e SHA256 `3823be7db9d9cfc93f3c2a47c7699be82821ab4e1117d4d2befdb746941ee96e`. Il loader originale usa strict=True; nessuna chiave mancante/inattesa è stata tollerata. Il checkpoint ha quattro blocchi temporali per modalità, latent_dim 110, dimensione interna **440**, feed-forward 1792, quattro attention head e 21.031.506 parametri. L'output è 128D. Non si sono usati i default del costruttore come prova della sua architettura.

La piccola prova CPU autorizzata ha confrontato il forward originale e quello separato su due pixel sintetici: errore massimo osservato **0** per N=1,2,3, entro la tolleranza 2e-5. I conteggi trainabili misurati sono 6.494.594, 11.208.818 e 15.923.042, testa esclusa. Con N=2, backward e AdamW hanno aggiornato **58 tensori di parametri encoder**, nessuno congelato. Un secondo probe con loss per cella e ciascuna aggregazione ha aggiornato 64 tensori complessivi per Dense Mean e 68 per Spatial CNN, includendo testa e suffisso, senza modificare i pesi congelati. La forma delle predizioni era `[1]`.

I numeri di tensori aggiornati non sono conteggi di elementi scalari; questi ultimi sono separati nei manifest. Il file checkpoint di partenza è rimasto invariato sul disco. Gli input di queste prove sono artificiali: le loss non sono risultati MSA e i tempi CPU non stimano la GPU. Evidenze tecniche sanificate sono in `validation/MAC_CHECKPOINT_PROBE.json` e `validation/MAC_CHECKPOINT_ARCHITECTURE.json`.

## File nuovi e motivazioni

| File/gruppo | Funzione e motivo |
|---|---|
| `scripts/finetune_v2.py` | Un'unica CLI nuova; non estende il vecchio percorso S1 10·log10 |
| `src/biomap_finetune/config.py` | YAML/extends/override, decisioni null/undecided, gate e protocollo esplicito |
| `encoder.py` | Token storici, confine effettivo, suffix trainabile, due aggregazioni differenziabili e prototipo prefisso FP32 |
| `data.py` | Piano di tutte le celle, sampling non basato sui target, cataloghi read-only copiati, acquisizione annuale, blocchi atomici e loader per cella |
| `io.py` | Root separate/proprietà, SHA/riletture, checkpoint immutabili, progress e retry Windows |
| `validation.py` | Probe checkpoint reale, aggiornamenti effettivi, confronto campionario con mappe frozen r2 e gate training |
| `training.py` | Normalizzazione iniziale training-only, AdamW, MSE cella, CV AOI, refit, Liguria, resume RNG/optimizer/scaler |
| `dense_evaluation.py` | Streaming adattato sul supporto completo, checkpoint OOF/refit, staging limitato e ripresa per tile/AOI |
| `comparison.py` | Nuovo frozen/partial appaiato, popolazioni identiche, delta macro-AOI e bootstrap AOI |
| `cli.py`, `__init__.py` | Orchestrazione, dry-run, smoke supervisionato, monitor senza handle persistenti |
| `configs/v2_partial_finetuning.yaml` | Parametri scientifici non decisi lasciati null; aggregazione undecided |
| `configs/v2_partial_smoke.yaml` | Profilo tecnico piccolo esplicito; non approva il metodo del benchmark |
| `tests/test_partial_finetuning.py` | Test CPU/offline di gradienti, protocollo, dataset, aggregazioni, acquisizione/resume e CV |
| `tests/fixtures/finetuning_protected_1737b1f.json` | Riferimenti derivati dai blob pubblicati e dalle EOL r1, utilizzabili anche da checkout CI shallow |
| `docs/finetuning/*` e runbook PC | Audit, progetto, budget, decisioni e istruzioni eseguibili per l'altro agente |

README riceve soltanto un collegamento al nuovo esperimento, mantenendo l'attuale percorso frozen. Non sono modificati moduli condivisi, dipendenze o workflow CI. Il workflow esistente su Ubuntu/macOS/Windows Python 3.12 raccoglie automaticamente i nuovi test ed è avviabile manualmente sul branch.

## Correzioni e scelte tecniche durante l'implementazione

Il nuovo forward evita NumPy dopo l'encoder e mantiene autograd fino ai blocchi finali. Il prefisso è no_grad ordinario, non inference_mode; il pooling e le mappe restano Torch. I gruppi optimizer contengono solo parametri autorizzati; il primo step registra i cambiamenti reali. L'automatismo per norm/fusione si risolve sulla struttura verificata e viene salvato nei manifest/checkpoint, non su nomi inventati.

Le validità S1 vengono calcolate prima dello z-score asc/desc separato; viene preservato DOY 366. I test confrontano il nuovo preprocessing direttamente con `reference.encode_pixels`, incluse modalità mancanti e bin misti. Non sono state cambiate date o regole SCL, interpolazione S2 o conversione S1. L'acquisizione importa l'implementazione r1 congelata.

Il sampling uniforme è globale per cella fra tile, non K per ogni tile. Il piano usa raggruppamento ordinato per evitare scansioni complete della geometria per ogni singola cella. Le finestre mantengono griglie native e contributi di overlap; non vengono assemblati mosaici tra UTM. Il loader produce più patch ma un solo target per cella, e rifiuta supporti persi. Nuove geometrie copiate sono verificate tramite SHA.

Per evitare costose acquisizioni annuali indipendenti per cella, il lavoro è raggruppato per tile e blocchi configurabili di pixel. I giorni validi sono persistenti e riusabili; i cataloghi r1 esistenti vengono copiati, mai aggiornati alla fonte. Il default operativo è 8192 pixel/blocco e due thread I/O: non è una misura di throughput e può richiedere calibrazione. Le serie hanno tutte le date annuali selezionate dal catalogo storico. Il preflight di spazio considera tutti i cataloghi reali prima dei raster; il piano offline dichiara invece scenari ipotetici.

Le transazioni usano file/directory temporanei univoci, fsync, rilettura e checksum prima del marker; il replace ha retry per Windows. I memmap non restano aperti mentre un file viene spostato. I checkpoint per epoca non vengono sovrascritti né cancellati; la ripresa ricostruisce l'ultimo marker verificato anche se i pointer sono mancanti. Un file incompleto resta diagnostico e non viene trattato come epoca completa. Nel test, perdita dei pointer e presenza di un checkpoint orfano non cambiano il risultato finale.

Il monitor chiude il JSON prima di attendere, evitando di mantenere handle Windows sul progress. Lo smoke usa un processo spawn supervisionato: deadline massima 900 secondi per l'intera pipeline. Se non termina, conserva i validi e fallisce; non dichiara “meno di 15 minuti” senza misura. La CLI rifiuta il profilo sintetico come scorciatoia per un run scientifico.

Il controllo frozen usa gli stessi input/head/seed del partial, ed è il confronto principale. Ogni confronto verifica popolazioni intere e target/fold, senza scegliere l'intersezione più conveniente. I confronti con i vecchi dense e con la Spatial CNN corrente restano da verificare quando cambia supporto, seed o protocollo; il report li segnala, senza attribuire una differenza interamente ai pesi.

## Validazione eseguita e limiti

Baseline sulla base obbligatoria, prima del nuovo codice: **213 passed, 1 skipped, 0 failed**, CPU macOS, Python 3.12.14 e Torch 2.10.0. Lo skip è quello preesistente Windows-only. La suite nuova copre N=1/2/3, gradienti e optimizer, due head, una previsione per cella, pesi maschere/supporto, modalità mancanti, cache prefisso online/eval/train, checksum, root/spazio, checkpoint e ripresa esatta, vero worker spawn, acquisizione interrotta simulata e raw uguali al percorso storico, CV/refit frozen e esclusione Liguria dalla selezione. Nessun download nei test.

La versione finale del codice `ef1f3d7` ha completato la suite: **243 passed, 1 skipped, 0 failed**, in 64,19s. Da clone Git pulito separato: **243 passed, 1 skipped**, in 73,64s, riusando soltanto la venv di sviluppo per le dipendenze. Il check ambiente/import/sintassi/backward è passato. La prima CI su de943eb è passata sui tre OS; la CI finale del codice esteso è registrata nella sezione conclusiva. I test sintetici non validano le librerie native Windows sui dati reali. La differenza S1 float32 di frontiera documentata dal progetto resta possibile tra CPU: non è stata corretta alterando CODE_FILES o allargando tolleranze. La prova campionaria input r2 sul PC deve bloccare discrepanze non spiegate.

Il test iniziale di protezione confrontava erroneamente il blob Git LF con il checkout r1 CRLF: è stato corretto derivando le EOL dal contratto versionato, non riscrivendo file legacy o inventando nuovi valori attesi. Il nuovo fixture mantiene byte raw per r1 e confronto contenuto LF per gli altri file non coperti da pin EOL. Il test r1 esistente continua a verificare la propria fixture storica indipendente.

## Funzioni non implementate e lavoro residuo

Non è implementato un trainer di corpus da cache intermedia: il prototipo FP32 verifica il confine, ma una cache completa è proibitiva. Non sono implementati LoRA, nuovi encoder, riduzione delle date, imputazioni, ricerca automatica degli iperparametri o conversione di mappe finali in attivazioni del prefisso. Il report legacy automatico e la validazione scientifica dei vecchi CSV non sono certificati. Non è misurato il batch sostenibile della RTX 5080, né la durata completa dell'acquisizione; doctor non dichiara READY.

Sul PC restano da eseguire: verifica degli output Spatial attuali, scelta dell'aggregazione, smoke reale entro il budget o diagnosi del timeout, confronto input/frozen con copertura dichiarata, misura risorse su supporto rappresentativo, decisione sampling/LR/batch/epoche e destinazione disco, acquisizione approvata e solo poi controllo frozen + partial su CV, refit e valutazione Liguria. Il training scientifico e l'acquisizione massiva non sono stati avviati da questa sessione.

La preferenza resta N=2, ma non è certificata sostenibile per tutte le celle su 16 GB. N=1 riduce parametri/stati ed è una candidata esplicita se il pilot lo richiede; N=3 aumenta molto il costo. La raccomandazione cache è online. Dense Mean e Spatial CNN sono entrambe predisposte; la scelta finale resta all'utente.

## Aggiornamento essenziale del handoff durante la consegna

I nuovi documenti del PC precisano che le serie legacy a 25 punti esistono: non sono utilizzabili per questo esperimento a causa di S1 10log10, S2 nearest e DOY limitato. Sono state aggiunte alle root protette configurabili e al doctor come artefatto incompatibile escluso. Nessuna conversione è implementata.

La scelta ora fissata dal handoff è acquisire un campione dei pixel originali V1 per ogni cella, usando le regole dense storiche. Il gate benchmark richiede uniform_pixels; full_support è escluso dal training e center_windows rimane tecnico. La proposta motivata è 32, seed20260925, fino a4.252.800 contributi. La config dedicata v2_partial_sample32.yaml lascia approved=false: il download è indipendente da testa/LR ma richiede approvazione esplicita del supporto. L'aggregazione resta undecided.

È stato aggiunto dense-eval per entrambe le head. Questo comando riusa lo streaming originale, applica i veri checkpoint adattati e non spaccia i suoi output per una cache r1/r2. Sviluppo OOF usa il modello del fold che esclude ogni AOI; Liguria usa refit. Supporto e conteggi sono quelli storici completi, selezione invariata sul campione. Produce metriche/predizioni dense separate e un protocollo nuovo; gestisce celle multi-tile/multi-UTM senza ricampionamento. Il raw transitorio nuovo è eliminato solo dopo commit della tile; le mappe nuove solo dopo commit dell'AOI. Tutte le sorgenti/cache/checkpoint preesistenti restano intatti. È l'unica pulizia automatica, necessaria per lo streaming e confinata a staging di proprietà del nuovo run. Test sintetici verificano entrambe le aggregazioni e resume senza ulteriori acquisizioni delle AOI concluse.

Il runbook è aggiornato con acquisizione32 indipendente e dense-eval, stime dichiarate del PC e controllo spazio. Gli esiti finali qui sotto sostituiscono quelli intermedi.

## Verifica finale del codice e pubblicazione

Il codice finale è il commit `ef1f3d7` su **feature/finetuning-v2**. PR aperta: [#7](https://github.com/ShimOne420/biomap-tessera-v2-finetuning/pull/7), base fix/spatial-v2-real-parquet-schema. Nessun merge o push su main. Working tree verificato pulito dopo i commit del codice; nessuna differenza nei cinque CODE_FILES, sources.py/support.py, .gitattributes, lock o entry point legacy rispetto alla base 1737b1f.

I 30 nuovi casi CPU sono tutti passati, compreso dense-eval su entrambe le head e confronto numerico con le teste del forward training, supporto completo sintetico, resume e sorgenti immutate. L'intera suite locale e da clone pulito ha243 passed/1 skipped; lo skip Windows-only resta motivato e preesistente. I warning sono1446, prevalentemente deprecazioni geospaziali e diagnostiche preesistenti; non sono stati silenziati per rendere verde la suite. Il clone pulito usa un checkout indipendente e dipendenze riusate, non una nuova installazione macOS. La CI esegue invece il setup su runner puliti.

Per lo spazio, C: e F: non sono un unico pool: pianificare raw e checkpoint per ciascuna destinazione.32 pixel sono solo una proposta. Con copie giornaliere e archivio immutabile di entrambi i run, le epoche massime possono richiedere più spazio del disponibile; spostare la nuova output-root su un'altra unità o scegliere un budget approvato prima del training. Non liberare automaticamente spazio cancellando dati/cache/checkpoint esistenti.

## Esiti CI osservati sul commit del codice ef1f3d7

Workflow finale [44123439025](https://github.com/ShimOne420/biomap-tessera-v2-finetuning/actions/runs/44123439025): **success**, tre job completati.

| Ambiente | Passati | Falliti | Saltati | Durata pytest |
|---|---:|---:|---:|---:|
| Mac locale |243|0|1|64,19s|
| Clone pulito Mac |243|0|1|73,64s|
| GitHub Ubuntu |243|0|1|112,33s|
| GitHub macOS |243|0|1|167,17s|
| GitHub Windows |244|0|0|322,60s|

Su Windows viene eseguito anche il controllo di condivisione file che sugli altri OS è saltato. I warning CI sono rispettivamente 1447/1445/1442; nessun errore è stato nascosto con skip o tolleranze aumentate. Il commit successivo aggiorna soltanto questa documentazione e le evidenze JSON: i test sopra si riferiscono esplicitamente a ef1f3d7, il codice rimane identico. Nessuna nuova esecuzione scientifica viene implicata dalla CI.

**Consegna tecnica completata. Fattibilità partial: supportata con modifiche isolate. Stato operativo: NOT_VERIFIED_FOR_REAL_TRAINING** finché il PC non completa smoke reale, parità input e misura risorse e l'utente non approva supporto, aggregazione e iperparametri. Report e runbook sono pubblicati nello stesso branch della PR; la PR resta aperta e non viene unita automaticamente.
