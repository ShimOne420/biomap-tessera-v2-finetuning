# CI CPU, contratto r1 e limiti operativi

## Workflow

`.github/workflows/tests.yml` si attiva sulle **Pull Request verso main**, sui push a main e manualmente. I push al branch con PR non generano un secondo run identico: la validazione è quella della PR. Si usano soltanto runner standard `ubuntu-latest`, `macos-latest`, `windows-latest`, Python 3.12 e `.venv-dev`.

Ogni job installa i pin di sviluppo, verifica sintassi/import/CPU backward e lancia `python -m pytest -ra`, includendo l'Esperimento B. Il job Windows mantiene la copertura del locking specifico della piattaforma. Non servono GPU, token personali, dataset o file del PC universitario. Il normale token interno al checkout ha solo `contents: read` e non viene persistito; nessun secret viene iniettato nei test.

Il runner macOS installa esplicitamente `libomp` tramite Homebrew, con auto-update disabilitato. Il primo run ha individuato l'assenza di questo runtime richiesto dalla wheel XGBoost: è un prerequisito del sistema, non una dipendenza Python rimossa dal lock. Gli import restano obbligatori.

Sono impostati timeout di **20 minuti per job**, cancellazione dei run superati della stessa PR, thread numerici limitati, backend grafico non interattivo e modalità Hugging Face offline. Non si eseguono benchmark di rete/GPU né training scientifici. La suite usa piccoli fit sintetici CPU per verificare il codice.

La cache pip serve solo le dipendenze, con chiave legata ai file di lock/setup; non contiene cache r1, pesi o dataset. Non vengono caricati artefatti sperimentali. Se la cache viene eliminata, l'installazione deve comunque funzionare dai pin. I runner `*-latest` possono evolvere: il report distingue versioni dei pacchetti e piattaforma osservata, senza promettere identità numerica tra hardware diversi.

## Costi e limiti verificati il 13 novembre 2026

Il repository GitHub è pubblico. L'esecuzione su **runner GitHub-hosted standard** di repository pubblici è gratuita secondo la documentazione corrente; i **larger runners sono a pagamento anche per repository pubblici**. Questo workflow non li seleziona e non usa runner universitari. [GitHub Actions billing](https://docs.github.com/en/billing/concepts/product-billing/github-actions).

Gratuito non significa senza limiti: la piattaforma impone concorrenza, code e durata massima; un job hosted ha limite di sei ore. Il workflow si ferma già a venti minuti per job e usa tre job, senza scheduling periodico. Le quote di concorrenza effettive dipendono dal piano dell'account e possono accodare le esecuzioni. Non è stato interrogato il piano di fatturazione privato dell'utente. [Actions limits](https://docs.github.com/en/actions/reference/limits).

Cache e artifact storage hanno limiti separati: la documentazione indica un'allowance cache di 10 GB per repository; policy di eviction e quote vanno monitorate. Non attiviamo storage aggiuntivo a pagamento e non promettiamo artefatti illimitati. Sono memorizzate solo wheel pip, con possibili chiavi distinte per sistema operativo. [Storage billing](https://docs.github.com/en/billing/concepts/product-billing/github-actions), [Dependency caching](https://docs.github.com/en/actions/reference/workflows-and-actions/dependency-caching).

Le condizioni GitHub possono cambiare: ricontrollare queste fonti prima di passare a repository privato, runner larger, artifact voluminosi o strategie di cache diverse.

## Provenienza del riferimento r1

La fixture `tests/fixtures/r1_contract_1cdc762.json` deriva dal **commit pubblico `1cdc76274738659cdfe1727c48ad9784b94cbd7f`**, che introduce le regole EOL della r1. Per ciascun `CODE_FILE` contiene Git blob ID, SHA256 della blob e SHA256 atteso dei byte di checkout.

Derivazione verificata: lettura `git show <commit>:<path>`, normalizzazione di storage Git e applicazione dell'EOL specifico dichiarato da `.gitattributes` nello stesso commit. I byte risultanti sono stati confrontati uno a uno con il checkout dell'integrazione `cf9f06c`. Non è stato usato un hash calcolato e atteso dallo stesso file corrente durante il test: il valore atteso è versionato e indipendente.

Questa verifica prova la corrispondenza allo snapshot Git pubblicato. Il PC universitario è inaccessibile: non viene certificata l'identità dei suoi file attuali o dei suoi marker live. Non occorre leggere la produzione per eseguire la CI.

## Cosa viene protetto

`tests/test_r1_contract.py` controlla:

- protocollo r1, lista ordinata `CODE_FILES`, SHA256 raw e fine riga dei cinque file;
- checksum Medium, maschera SCL e numero di retry fissati nello snapshot;
- regole `.gitattributes` miste, senza wildcard che le sovrascriva;
- byte normalizzati del validatore delle cache `campaign/state.py`;
- default della CLI `stream` e contenuto del launcher Windows di produzione;
- dispatch di `stream` alla r1, senza attivare l'altro percorso `extract`;
- riuso di una tile sintetica con identità conforme senza rete/encoder;
- rifiuto di ciascun campo identità mutato e di un payload con checksum errato, senza ricomputazione silenziosa.

Restano attivi i test storici di preprocessing e quelli di equivalenza r1/r2 su raster sintetici. Nessun `CODE_FILE` r1 viene modificato dal lavoro infrastrutturale.

Se un controllo fallisce, **non rigenerare la fixture dal checkout modificato**. Verificare prima EOL, editor, Git attributes e diff rispetto al commit fonte. Modifiche scientifiche deliberate richiedono un nuovo protocollo e un processo di revisione; non devono forzare la compatibilità di cache preesistenti. Queste protezioni rilevano incidenti e regressioni: non impediscono a un autore di una PR di modificare insieme test e riferimenti, che rimangono oggetto di revisione.

## Componenti sperimentali e confini

| Componente | Input e comportamento |
|---|---|
| r1 | `dense_campaign.py stream`, serie Sentinel annuali e encoder V2 congelato; byte/EOL protetti |
| r2-fast | `dense_fast.py`, organizzazione I/O sperimentale; protocollo e work-root distinti |
| A | 128 medie V2 per cella, Dummy/Ridge/HistGB, MLP e Conv1D sui canali; loader scientifico r1-only |
| B | Mappe V1 native `[H,W,128]`, patch/maschere, CNN 2D e baseline sul medesimo supporto B |

La CI non modifica gli esperimenti né certifica completamento della campagna, equivalenza su tutta la produzione, prestazioni GPU o miglioramento MSA. L'ambiente dev è per implementare e verificare codice; il run di ricerca richiede dati, provenienza, device e preflight propri.
