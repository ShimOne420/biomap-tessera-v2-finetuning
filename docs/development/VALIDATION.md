# Validazione dell'infrastruttura di sviluppo

Verifica locale: **13 novembre 2026**, macOS 26.6.2 arm64. Branch `feature/mac-dev-infrastructure`, base integrata `cf9f06c334ba036506ceca267f6cddc1dabd4c9a`. Questa consegna non avvia alcun esperimento scientifico.

## Ambiente realmente provato

È stata creata una **venv nuova `.venv-dev`** eseguendo la procedura documentata con Python 3.12.14. Setup completato, progetto installato editable e `pip check` senza dipendenze incoerenti. Torch 2.10.0; dipendenze provenienti dal lock scientifico mantenuto invariato.

`check_dev_environment.py` ha verificato sintassi di 119 file, import delle librerie/package installati, import isolati di B e un forward/backward CPU. Nella prima sessione sandbox `mps_built=true` ma `mps_available=false`; nel clone persistente, con accesso completo al Mac, MPS è disponibile e la piccola prova esplicita `--probe-mps` di forward/backward è passata. Non è un test dell'encoder o dell'intera pipeline MPS. CUDA non è disponibile. Il primo caricamento delle wheel nuove ha avuto un'attesa iniziale prima dell'output, poi è terminato correttamente senza modificare pipeline o librerie.

## Suite standard unificata

```bash
.venv-dev/bin/python -m pytest -ra --junitxml=test-results/cpu.xml
```

Nel clone persistente fuori da iCloud: **166 passati, 0 falliti, 1 saltato**, 1.426 warning, **93,92 secondi**. Il file JUnit riporta 167 casi raccolti, dei quali **21 dell'Esperimento B**, ora inclusi nella selezione standard. I nuovi controlli infrastrutturali/contratto r1 aggiungono 21 casi alla suite preesistente. Anche la prima prova Desktop era passata (166/0/1, 1.427 warning, 119,20 s); è stata ripetuta nel clone persistente dopo aver rilevato il problema dei `.pth` nascosti.

Lo skip è soltanto `tests/test_dense_fast.py:277`, già presente: test di condivisione file specifico Windows. Non sono stati aggiunti skip per nascondere errori o assenza dei dati universitari. Il job Windows è mantenuto proprio per eseguire quel ramo. I warning, inclusi quelli di deprecazione Affine, restano visibili nel log: non sono stati filtrati.

I test sono stati eseguiti con thread numerici limitati e Hugging Face offline. La fixture comune blocca gli accessi esterni Python/Rasterio più comuni; le prove rimangono locali e sintetiche, senza dataset, pesi o credenziali.

## Diagnostica storica eseguita separatamente

```bash
.venv-dev/bin/python -m pytest -ra tests/review_acceleration_cases.py
```

Risultato: **5 passati, 0 falliti, 0 saltati**, 4 warning, 0,76 secondi. Quel file è dichiaratamente diagnostico fin dalla versione precedente: le asserzioni documentano comportamenti difettosi della revisione storica, **non sono criteri positivi di accettazione**. La raccolta standard lo esclude già per nome `review_*`.

I cinque casi riguardano accettazione del dtype errato in un checkpoint, budget locale insufficiente come quota globale, contatori persi alla ripresa, decodifiche ripetute degli NPZ e stato complete prima della validazione finale. Il loro passaggio significa che le dimostrazioni dei difetti restano riproducibili. Non sono stati rinominati, eliminati, trasformati in skip o utilizzati per affermare che quei problemi siano risolti. Una futura correzione richiede convertirli in regressioni con comportamento desiderato, in una PR distinta.

## Integrità r1

I cinque `CODE_FILES` r1 e `requirements-lock.txt` sono identici alla base `cf9f06c`; nessun file degli esperimenti è stato modificato. Il nuovo contratto è stato derivato dal commit pubblico `1cdc76274738659cdfe1727c48ad9784b94cbd7f`, applicando le regole EOL di quel commit e verificando i byte contro il checkout integrato. Le firme attese sono versionate nella fixture, non ricalcolate dai file correnti come valori attesi.

I test controllano hash raw/EOL, protocollo, default, dispatch e compatibilità/rifiuto dei marker su una cache sintetica. Non certificano gli hash attuali del PC universitario, che non è accessibile.

## Limiti e verifiche remote

La CI predisposta ha tre job CPU standard Ubuntu/macOS/Windows e usa lo stesso comando unificato. Lo stato del run della PR è consultabile su GitHub: non va confuso con la prova Mac locale riportata qui. La disponibilità di MPS, CUDA, i processi/marker universitari e il completamento della campagna non sono verificati da questi test.

Nel [primo run CI](https://github.com/ShimOne420/biomap-tessera-v2-finetuning/actions/runs/43419082195), il job macOS è fallito al controllo import per assenza di `libomp.dylib`; la suite non è stata avviata su quel job dopo il fallimento. La dipendenza era già disponibile sul Mac locale. È stato aggiunto un passo Homebrew specifico del runner macOS e documentato il prerequisito: nessuna versione del lock o sorgente r1 è stata modificata per aggirare l'errore.

Il [secondo run CI, commit 405d0d9](https://github.com/ShimOne420/biomap-tessera-v2-finetuning/actions/runs/43419391006), ha completato tutti e tre i job:

| Runner | Passati | Falliti | Saltati | Tempo suite |
|---|---:|---:|---:|---:|
| macOS | 166 | 0 | 1 Windows-only | 194,59 s |
| Windows | 167 | 0 | 0 | 235,70 s |
| Ubuntu | 165 | **1** | 1 Windows-only | 61,43 s |

L'errore Ubuntu è `tests/test_dense_public.py::test_s1_storage_historical_convention`: output `5097` invece del riferimento `5098`, nel calcolo float32 prima della quantizzazione int16. Il codice e i pacchetti sono quelli pinned; la differenza di runtime/implementazione matematica richiede analisi. Non è stata dimostrata qui una causa ISA specifica, né un impatto sul numero completo di pixel. Non sono stati aggiunti skip, cambiati valori attesi o modificati CODE_FILES/dispatch NumPy. La CI segnala la divergenza senza aggirarla: questa blocca la piena riproducibilità numerica Ubuntu e il merge della PR.

Un successivo controllo import nel clone Desktop ha rilevato `ModuleNotFoundError`: il `.pth` editable risultava `UF_HIDDEN` e veniva ignorato da Python 3.12. È stato quindi preparato un clone Git indipendente e una nuova venv sotto `~/Developer`, fuori da iCloud. Il `.pth` e la configurazione Git del clone persistente non risultano hidden/dataless; gli import editable e la suite sono passati senza `PYTHONPATH`. I checkout storici sono stati preservati; il setup dipende solo da pacchetti ufficiali e libomp, senza dati della produzione. Gli esiti del clone persistente sono registrati nel riepilogo JSON e nella PR.

Nessuna modifica al codice scientifico è stata necessaria per far passare la suite. La documentazione costi/limiti cita le fonti ufficiali verificate alla data dell'intervento. Un riepilogo leggibile automaticamente è in [validation_mac.json](validation_mac.json).
