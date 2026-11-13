# Sviluppo BIOMAP su macOS, Linux e Windows

Questa procedura prepara **soltanto sviluppo e test CPU**: non avvia acquisizioni, non scarica pesi/dataset e non accede al PC universitario. Richiede Git e Python **3.12**; Python 3.14 eventualmente predefinito sul Mac non soddisfa il vincolo del progetto. Gli esperimenti scientifici restano attività separate.

## 1. Checkout e branch

Usare un clone di codice indipendente, preferibilmente fuori da Desktop/Documenti sincronizzati iCloud: gli oggetti `.git` dataless possono bloccare fetch e lettura della storia anche con working tree pulito. Non spostare o sovrascrivere un checkout con modifiche locali. Prima di ogni attività:

```bash
git remote -v
git status --short --untracked-files=all
git branch --show-current
git rev-parse HEAD
git worktree list
```

Per una directory nuova, questi comandi creano il clone persistente fuori da iCloud. Se la destinazione esiste già, verificarla senza sovrascriverla:

```bash
mkdir -p "$HOME/Developer"
git clone --single-branch --branch feature/mac-dev-infrastructure https://github.com/ShimOne420/biomap-tessera-v2-finetuning.git "$HOME/Developer/biomap-tessera-v2-finetuning"
cd "$HOME/Developer/biomap-tessera-v2-finetuning"
```

Durante la validazione è emerso anche un secondo effetto di iCloud: il file `.pth` dell'installazione editable sul Desktop ha acquisito `UF_HIDDEN`. Il `site.py` di Python 3.12 ignora i `.pth` nascosti, causando `ModuleNotFoundError` pur con pacchetto installato e `pip check` corretto. Non lo aggiriamo aggiungendo `PYTHONPATH`: il clone e la venv persistenti sotto `~/Developer` evitano quel percorso sincronizzato. Una `.venv-dev` su Desktop non è l'ambiente definitivo raccomandato.

Il lavoro infrastrutturale corrente è su `feature/mac-dev-infrastructure`. È stato avviato dalla versione integrata `cf9f06c`, perché durante la preparazione `main` era ancora `e720ad0c`. La PR espone la dipendenza dall'integrazione; non effettua un merge. Non assumere che un semplice pull di un main precedente includa gli esperimenti.

## 2. Python e venv privata su macOS

Verificare il Python installato. Con Homebrew già disponibile, Python 3.12 può trovarsi in `/opt/homebrew/bin/python3.12` (Apple Silicon) o `/usr/local/bin/python3.12` (Intel). Non serve cambiare il Python globale:

```bash
python3.12 --version
python3.12 scripts/setup_dev.py --dry-run
python3.12 scripts/setup_dev.py
```

Se Python 3.12 manca, installarlo personalmente da [python.org](https://www.python.org/downloads/macos/) oppure con il proprio gestore di pacchetti prima del setup. Lo script non installa Python, driver, Homebrew o toolchain di sistema.

Su macOS XGBoost richiede anche il runtime OpenMP **libomp**, esterno alla venv. Con Homebrew già disponibile verificarlo e, se manca, installarlo esplicitamente prima del controllo import:

```bash
brew list --versions libomp
# Solo se il controllo precedente segnala che libomp non è installato:
brew install libomp
```

Il setup Python non modifica automaticamente i pacchetti di sistema. La CI installa esplicitamente questo prerequisito sul runner macOS. L'errore `Library not loaded: @rpath/libomp.dylib` non si risolve eliminando XGBoost dai test o cambiando il lock. [Indicazioni XGBoost per macOS/OpenMP](https://xgboost.readthedocs.io/en/release_3.1.0/build.html).

Il setup crea `.venv-dev` con `venv`, installa pip 25.1.1 e Torch 2.10.0, poi `requirements-dev.txt`, che include integralmente il lock scientifico esistente. Installa il progetto editable con `--no-deps --no-build-isolation` e verifica `pip check`. Su macOS usa le wheel Torch di PyPI con CPU e, quando disponibile, MPS. Su Linux/Windows seleziona prima l'indice CPU ufficiale per evitare dipendenze CUDA.

Le dipendenze richiedono rete e spazio su disco; i dati di ricerca no. L'ambiente principale `.venv`, quello della CNN V1 e quelli di altri agenti non sono riutilizzati. Il parametro `--venv /percorso/.venv-dev` consente un'altra root privata; il basename deve rimanere `.venv-dev`. Nessuna attivazione della shell è necessaria.

Equivalente manuale su macOS, utile per ispezionare ogni fase:

```bash
python3.12 -m venv .venv-dev
.venv-dev/bin/python -m pip install pip==25.1.1
.venv-dev/bin/python -m pip install torch==2.10.0
.venv-dev/bin/python -m pip install -r requirements-dev.txt
.venv-dev/bin/python -m pip install -e . --no-deps --no-build-isolation
.venv-dev/bin/python -m pip check
```

Il lock radice non è stato rigenerato. Non copiare il `pip freeze` del Mac nell'ambiente universitario. Se il proprio Mac non dispone di una wheel compatibile con i pin, fermarsi con l'errore concreto e discuterne un profilo separato; non cambiare le versioni scientifiche automaticamente.

## 3. Import e comando unico di test

```bash
.venv-dev/bin/python scripts/check_dev_environment.py --json .venv-dev/environment.json
.venv-dev/bin/python -m pytest -ra
```

Il primo comando controlla la sintassi, gli import del progetto installato e delle librerie, importa B in un processo isolato e verifica un piccolo forward/backward CPU. Rifiuta import provenienti da un altro clone. Nessun encoder o peso viene caricato.

Il secondo esegue **tutti i test CPU standard**, inclusi `tests/`, contratto r1 e `experiments/v1_spatial_cnn_msa/tests/`. Non occorre ricordare un secondo percorso pytest. Il riepilogo `-ra` espone skip ed errori; non nasconde i warning.

Il runner Ubuntu ha evidenziato una differenza numerica S1 di frontiera: `5097` contro `5098`. I test verificano ora il contratto float32/troncamento con kernel controllati e il kernel nativo su valori lontani dalle soglie; non usano tolleranze allargate o skip. `check_dev_environment.py` continua a registrare output nativo, bit float32 del logaritmo e dispatch, senza modificarli. La suite verde non implica identità numerica del preprocessing fra piattaforme: leggere la [decisione S1](S1_NUMERICAL_PORTABILITY.md) e gli esiti in [VALIDATION.md](VALIDATION.md).

Il file storico `tests/review_acceleration_cases.py` resta diagnostico ed esplicitamente separato: afferma la presenza di difetti di una vecchia revisione e non è un criterio di accettazione. Il motivo e il suo esito separato sono documentati in [VALIDATION.md](VALIDATION.md).

Per salvare anche un resoconto JUnit locale:

```bash
.venv-dev/bin/python -m pytest -ra --junitxml=test-results/cpu.xml
```

La fixture radice impedisce richieste HTTP Python, socket e letture Rasterio esterne comuni; file locali e server loopback sintetici sono ammessi. Le prove r1/r2 usano un encoder finto e raster generati nei temporanei, non Sentinel. Questi controlli non sono un sandbox per codice nativo/subprocessi arbitrari.

## 4. MPS facoltativo

`mps_built` significa che il pacchetto supporta il backend; `mps_available` indica se è utilizzabile nella sessione effettiva. Il report non confonde le due condizioni. Per una prova esplicita e minuscola:

```bash
.venv-dev/bin/python scripts/check_dev_environment.py --probe-mps
```

Se MPS non è disponibile, il comando richiesto fallisce in modo esplicito; la suite CPU continua a essere il riferimento. Non abilitiamo fallback silenziosi per far apparire superato un test MPS. [Documentazione PyTorch MPS](https://docs.pytorch.org/docs/2.10/notes/mps.html).

MPS non sostituisce automaticamente CUDA FP32: disponibilità, operazioni deterministiche e differenze numeriche richiedono verifiche dedicate prima di un esperimento reale. Il setup non cambia il device o il protocollo della r1.

## 5. Linux e Windows

Su Linux si usano gli stessi comandi `python3.12 scripts/setup_dev.py`, `.venv-dev/bin/python ...`. Lo script sceglie l'indice Torch CPU.

Su Windows, da PowerShell nella radice di un clone di sviluppo:

```powershell
py -3.12 scripts\setup_dev.py --dry-run
py -3.12 scripts\setup_dev.py
& .\.venv-dev\Scripts\python.exe scripts\check_dev_environment.py
& .\.venv-dev\Scripts\python.exe -m pytest -ra
```

Il percorso di produzione con RTX 5080, CUDA 12.8 e `.venv` rimane distinto. Non eseguire il setup dentro un checkout usato da un run in corso. Il nuovo setup CPU non chiama il vecchio `doctor`, non crea cache di ricerca e non usa percorsi `C:\Users\...` o `F:`.

I wrapper scientifici PowerShell, le metriche WinAPI e il test di condivisione file Windows non diventano portabili per effetto del setup. Su macOS/Linux il test Windows già esistente è saltato con motivo esplicito; Windows lo esegue. CUDA e campagna completa non sono verificate dai test CPU.

## 6. Collaborare senza interferire con la produzione

Lavorare su branch dedicati, con commit del solo lavoro pertinente e PR verso main. Non fare merge automatici o push diretto su main. Prima di aggiornare un checkout verificare processi che lo usano e modifiche non committate; non usare reset/clean/force push per risolvere un conflitto.

Non è richiesto possedere i dataset universitari per modificare infrastruttura o metriche su fixture. Per run scientifici servono acquisizione/provenienza/preflight separati. Non inventare marker di completamento o risultati per soddisfare un test.

Le [regole per gli agenti](../../AGENTS.md) elencano i file r1 con EOL misti da lasciare intatti. Il [contratto e la CI](CI_AND_R1.md) descrivono protezioni, provenienza dei riferimenti, costi e limiti. Gli esiti effettivi di questa consegna sono in [VALIDATION.md](VALIDATION.md).
