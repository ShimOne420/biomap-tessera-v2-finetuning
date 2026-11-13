# BIOMAP: istruzioni per sviluppatori e agenti

## Contesto e ambito

Questo repository contiene codice di ricerca. Un clone di sviluppo è indipendente dal PC Windows universitario: non presumere di poter accedere a processi, cache, checkpoint, configurazioni o risultati di quel PC. La presenza di documentazione di un run non ne dimostra lo stato live.

Il lavoro infrastrutturale e integrato. Per la Spatial CNN corrente usare `feature/spatial-cnn-v2`, partito dal main verificato `b7da382`. Il percorso prioritario e `scripts/spatial_cnn.py`; non cancellare A/B o modificare le loro entry point. Non fare push su `main` e non effettuare merge automatici. Controllare sempre `git status`, branch, HEAD, origin e worktree prima di modificare file. Non sovrascrivere lavoro non committato, non fare reset/clean/force push. Usare branch e output separati per esperimenti paralleli. Non fare checkout o pull sotto un processo scientifico attivo.

## Ambiente e verifica

- Python richiesto: **3.12**. Il Python predefinito del sistema può avere una versione diversa.
- Setup dev: `python3.12 scripts/setup_dev.py`; su Windows `py -3.12 scripts/setup_dev.py`.
- La venv privata è `.venv-dev`; `.venv` e gli ambienti degli esperimenti non vengono riutilizzati.
- macOS/Linux: `.venv-dev/bin/python scripts/check_dev_environment.py` e `.venv-dev/bin/python -m pytest -ra`.
- Windows PowerShell: `& .\.venv-dev\Scripts\python.exe scripts\check_dev_environment.py` e `& .\.venv-dev\Scripts\python.exe -m pytest -ra`.
- Il comando pytest standard include `tests` e `experiments/v1_spatial_cnn_msa/tests`.
- Il lock scientifico resta `requirements-lock.txt`; `requirements-dev.txt` lo include e blocca Torch. Non rigenerare indiscriminatamente il lock con freeze di un computer diverso.
- I test devono funzionare su CPU, con fixture locali e senza token/dataset/checkpoint. La fixture radice blocca connessioni esterne comuni; non è un sandbox per codice nativo arbitrario. Non disattivarla per far passare un download.
- Non eliminare test, aumentare tolleranze scientifiche o aggiungere skip per aggirare un errore. Uno skip deve descrivere una reale dipendenza di piattaforma.
- La divergenza S1 nativa di frontiera (`5097` contro `5098`) resta registrata da `check_dev_environment.py`. I test separano kernel controllato e kernel reale lontano dalle soglie, con assert esatte: leggere `docs/development/S1_NUMERICAL_PORTABILITY.md`. Non cambiare CODE_FILES, dispatch NumPy, fixture r1 o tolleranze per rendere verde la CI. Suite verde non significa identita numerica fra piattaforme.

## Contratto r1 da preservare

Il protocollo r1 è `dense127909-stream-legacy-exact-r1`. Le cache dipendono dai byte raw dei cinque `CODE_FILES`; **non modificarli né formattarli** per attività infrastrutturali:

| File sotto `src/biomap_v2/` | EOL |
|---|---|
| `campaign/streaming.py` | LF |
| `raster_cache.py` | CRLF |
| `data.py` | CRLF |
| `reference.py` | CRLF |
| `encoder.py` | CRLF |

Non introdurre regole EOL globali o impostazioni editor che riscrivano questi file. `.gitattributes` è l'autorità; il contratto verificato è in `tests/fixtures/r1_contract_1cdc762.json`. I riferimenti derivano dal commit pubblico indicato nella fixture, non da hash inventati o da output attesi della macchina corrente. Se falliscono, investigare: non aggiornare il JSON solo per ottenere test verdi. Un cambiamento scientifico richiede un protocollo/namespace nuovo e una decisione esplicita; non invalida le cache esistenti silenziosamente.

## Componenti distinti

- r1: `scripts/dense_campaign.py stream` → `campaign/streaming.py`, congelato.
- r2-fast: `scripts/dense_fast.py`, work-root e identità separate; cache r1/r2 non intercambiabili.
- Esperimento A: `biomap_dense_heads`, medie V2 128D; Conv1D sui **canali**, MLP e baseline. Il loader scientifico attuale accetta solo marker r1.
- Esperimento B: `experiments/v1_spatial_cnn_msa`, mappe V1 native e CNN 2D mascherata, una loss per cella. Non ricostruire immagini da vettori medi.
- `biomap_ml` e la preparazione/probing generale sono ulteriori percorsi implementati: non sostituirli automaticamente al protocollo di produzione.

## Vincoli scientifici e collaborazione

Conservare ID, target GLOBIO, fold geografici, conteggi e provenienza. Separare sviluppo/CV da Liguria, già osservata; niente selezione sul test. Normalizzazione training-only; confronti appaiati sulle medesime celle/supporto. Test sintetici non sono risultati MSA né certificazione CUDA. Documentare piattaforma, test superati/falliti/saltati e limiti.

Ogni nuovo script/modulo di supporto/test deve iniziare con un commento dettagliato su scopo, fasi, input/output, parametri/esempio, risorse e ripresa. Tenere dati, credenziali, venv, cache e checkpoint fuori da Git. Fare commit soltanto del proprio lavoro, aprire PR revisionabili e indicare dipendenze da altri branch. Non modificare esperimenti altrui oltre al minimo necessario per la CI.

Guida operativa: [docs/development/MACOS_SETUP.md](docs/development/MACOS_SETUP.md). Produzione, sviluppo e test devono avere directory e processi separati.
