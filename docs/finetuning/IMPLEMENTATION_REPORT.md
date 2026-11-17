# Implementazione partial fine-tuning GeoTESSERA V2 Medium

Data: 17 novembre 2026. Destinatario: agente del PC universitario e revisore della PR. **Infrastruttura implementata e verificata su CPU; esecuzione scientifica su input reali/GPU non ancora verificata.** Fattibilità: **SUPPORTED_WITH_CHANGES**, attraverso un package isolato, non modifiche all'estrazione legacy.

## Base Git e isolamento

Sono stati controllati repository, origin, branch, HEAD, worktree, modifiche locali e informazioni remote. Il clone Mac era pulito; è stato eseguito fetch e creato `feature/finetuning-v2` da `1737b1fdd30d5de9a50967363ca39042b26a055b`, con `d1c0d93` verificato come antenato. La base contiene le correzioni schema parquet reale, chiusura memmap Windows e celle multi-UTM. Main verificato era `b7da382d4424c1ddd35a285830f95bb2d42c6436`; non è stato modificato né si è presunto il merge della Spatial CNN.

L'audit architetturale è stato scritto e committato **prima dell'implementazione**, commit `576b8ec`. La PR del fine-tuning dipende dal branch fix della Spatial CNN e deve essere revisionata lì, senza merge automatico. L'elenco aggiornato dei commit della consegna è nella PR; questo report non contiene uno SHA autoreferenziale del proprio futuro commit.

Non è stato contattato il PC universitario. Nessun Sentinel o nuovo checkpoint è stato scaricato, nessuna campagna scientifica o cache massiva è stata avviata. I soli processi Python locali osservati inizialmente erano strumenti dell'editor. Il run Spatial universitario e i suoi output non sono stati ispezionati né modificati.

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

**Gli esiti finali della suite completa e della CI saranno registrati qui dopo l'esecuzione**, prima della consegna della PR. I test sintetici non validano le librerie native Windows sui dati reali. La differenza S1 float32 di frontiera documentata dal progetto resta possibile tra CPU: non è stata corretta alterando CODE_FILES o allargando tolleranze. La prova campionaria input r2 sul PC deve bloccare discrepanze non spiegate.

Il test iniziale di protezione confrontava erroneamente il blob Git LF con il checkout r1 CRLF: è stato corretto derivando le EOL dal contratto versionato, non riscrivendo file legacy o inventando nuovi valori attesi. Il nuovo fixture mantiene byte raw per r1 e confronto contenuto LF per gli altri file non coperti da pin EOL. Il test r1 esistente continua a verificare la propria fixture storica indipendente.

## Funzioni non implementate e lavoro residuo

Non è implementato un trainer di corpus da cache intermedia: il prototipo FP32 verifica il confine, ma una cache completa è proibitiva. Non sono implementati LoRA, nuovi encoder, riduzione delle date, imputazioni, ricerca automatica degli iperparametri o conversione di mappe finali in attivazioni del prefisso. Il report legacy automatico e la validazione scientifica dei vecchi CSV non sono certificati. Non è misurato il batch sostenibile della RTX 5080, né la durata completa dell'acquisizione; doctor non dichiara READY.

Sul PC restano da eseguire: verifica degli output Spatial attuali, scelta dell'aggregazione, smoke reale entro il budget o diagnosi del timeout, confronto input/frozen con copertura dichiarata, misura risorse su supporto rappresentativo, decisione sampling/LR/batch/epoche e destinazione disco, acquisizione approvata e solo poi controllo frozen + partial su CV, refit e valutazione Liguria. Il training scientifico e l'acquisizione massiva non sono stati avviati da questa sessione.

La preferenza resta N=2, ma non è certificata sostenibile per tutte le celle su 16 GB. N=1 riduce parametri/stati ed è una candidata esplicita se il pilot lo richiede; N=3 aumenta molto il costo. La raccomandazione cache è online. Dense Mean e Spatial CNN sono entrambe predisposte; la scelta finale resta all'utente.
