# S1 float32: decisione sui test e limiti di riproducibilità

## Scopo e decisione

Questa revisione della PR infrastrutturale #4 corregge l'assunzione che un singolo risultato `log10` float32 al confine di un intero sia identico su tutti gli host. Non corregge o modifica il preprocessing r1. Il contratto della funzione viene verificato con assert esatte su kernel controllati; il kernel reale viene verificato su casi lontani dalle soglie e osservato separatamente sul caso di frontiera.

La decisione segue la richiesta di consolidare l'integrazione e sistemare il test Ubuntu senza cambiare i cinque CODE_FILES. Non consiste nell'accettare indistintamente `5097` oppure `5098`, nell'aumentare una tolleranza, applicare skip/xfail, forzare un dispatch NumPy o rilanciare il job finché diventa verde.

## Diagnosi verificata

La funzione congelata `campaign/streaming.py::s1_storage` converte prima l'input in float32. Sui valori positivi finiti calcola `((20 * log10(value)) + 50) * 200`, con costanti float32; applica clipping e conversione int16 per troncamento. Gli altri valori restano zero.

Il test precedente usava `0.05949767306447029` e imponeva sempre `5098`. Sul Mac con NumPy 2.2.6 è stato misurato:

| Log float32 | Bit IEEE754 | Prima di int16 | Risultato |
|---|---|---|---|
| `-1.2254999876022339` | `0xbf9cdd2f` | `5098.0` | `5098` |
| `-1.2255001068115234` | `0xbf9cdd30` | `5097.99951171875` | `5097` |

La prima riga deriva dal kernel reale misurato sul Mac. La seconda deriva dal successivo valore float32 rappresentabile verso meno infinito, costruito e verificato localmente: riproduce esattamente il valore pre-cast già registrato su Ubuntu. Questo dimostra la sensibilità del troncamento a un ULP del logaritmo. Non dimostra da solo quale CPU, ISA o libreria abbia causato la differenza del runner; il processore assegnato da GitHub rimane un'ipotesi da verificare.

Anche la [CI di main dopo il merge dell'integrazione](https://github.com/ShimOne420/biomap-tessera-v2-finetuning/actions/runs/43460386810) ha riprodotto il fallimento Ubuntu sullo stesso test: 123 passati, 1 fallito, 1 skip nella selezione precedente all'unificazione. Il precedente verde dell'integrazione non dimostrava stabilità fra runner. Non è stato effettuato un re-run per nascondere questo esito.

## Modifiche ai test

`tests/test_dense_public.py::test_s1_storage_historical_convention` conserva il nome, ma ora ha due casi parametrizzati. Il solo `log10` viene sostituito temporaneamente da pytest con uno dei due risultati float32 esatti della tabella. La funzione r1 sotto test rimane quella reale e immutata. Il kernel controllato verifica che riceva soltanto i due valori positivi finiti, già convertiti in float32.

Per il primo caso il risultato deve essere esattamente `5098`; per il secondo esattamente `5097`. Non sono due attesi alternativi per lo stesso kernel. Le asserzioni proteggono conversione prima del logaritmo, filtro degli invalidi, scala/offset, dtype int16, shape e troncamento. Per esempio, un cambiamento a float64 viene rifiutato prima del logaritmo; un arrotondamento al posto del troncamento fallisce sul secondo caso.

Il nuovo `test_s1_storage_native_kernel_away_from_integer_boundaries` usa il kernel NumPy reale, senza monkeypatch. Verifica una matrice con valori `.02`, `.2`, `.5`, `1`, `2`, zero, negativo, NaN, infinito, valore molto piccolo e valore da saturare. Le uscite intere esatte sono `[3204, 7204, 8795, 10000, 11204, 0, 0, 0, 0, 0, 32767]`. Per i valori trasformati non interi la distanza dalla soglia è superiore a 0.1: non servono tolleranze numeriche. Questo percorso verifica realmente logaritmo, scale/offset, clipping, dtype e conservazione della shape.

La sostituzione del kernel è limitata al test ed è ripristinata da pytest. Non sono installati wrapper globali, cambiate variabili ISA o modificati i kernel usati dalla preparazione dei dati.

## Diagnostica che rimane visibile

`scripts/check_dev_environment.py` continua a eseguire il caso di frontiera con il kernel reale. Registra versione NumPy, dispatch, risultato prima/dopo int16 e ora anche i bit float32 del logaritmo. Il JSON indica esplicitamente `cross_platform_bitwise_guarantee: false` e se il risultato coincide con il valore Mac registrato. I log CI mantengono questa informazione anche quando la suite dei contratti passa.

Questa diagnostica non è un certificato di equivalenza dei dati né un gate che rifiuta il solo valore `5097`. La separazione è intenzionale: portabilità del contratto di sviluppo e identità numerica del preprocessing fra ambienti sono verifiche diverse.

## File e identità protetti

Nessuna modifica a `streaming.py`, `raster_cache.py`, `data.py`, `reference.py`, `encoder.py`, `.gitattributes`, `requirements-lock.txt`, alla fixture SHA/EOL o a r2-fast. Restano attivi i test della cache r1, della ripresa e dell'equivalenza r1/r2 sul medesimo host. Non sono modificati target, geometrie, split o campionamento.

Il riferimento SHA/EOL rimane `tests/fixtures/r1_contract_1cdc762.json`, derivato dal commit pubblico `1cdc76274738659cdfe1727c48ad9784b94cbd7f`, non dai file modificati della macchina corrente. Il checksum dei file protegge il codice; da solo non prova equivalenza fra implementazioni matematiche di runtime differenti.

## Implicazioni scientifiche

Una CI verde dopo questa revisione significa che i contratti CPU e i test numerici dichiarati passano. **Non significa che la r1 sia bit-identica fra macOS, Linux e Windows.** Non è stato quantificato l'effetto sui circa 99,5 milioni di pixel, sugli embedding o sulle metriche MSA. Il PC universitario non è stato consultato né modificato: gli SHA e gli artefatti live non sono verificati da questo intervento.

La campagna Windows continua con il suo ambiente. Non spostare una ripresa su un altro host presumendo che il solo hash del codice certifichi l'equivalenza; conservare l'ambiente e la provenienza della campagna. Un futuro preprocessing canonico fra piattaforme, o una modifica dell'identità che includa il runtime, richiede protocollo/namespace separato e confronto dedicato. Non viene retrofittato sulle cache r1.

## Consolidamento Git

La [PR #5](https://github.com/ShimOne420/biomap-tessera-v2-finetuning/pull/5) integra `cf9f06c` con merge commit `8822f60e90dee58b8a8b13172062c75b69e8c499`, preservando la storia e senza force push. L'albero candidato era uguale all'albero dell'integrazione verificata. La PR #3 è risultata automaticamente merged. Il diff della PR #4 ora riguarda l'infrastruttura successiva all'integrazione, più la presente revisione test/diagnostica/documentazione.

Il merge infrastrutturale richiede la nuova CI completa Ubuntu/macOS/Windows, non un vecchio verde. Nessun branch del framework viene iniziato prima della verifica della base consolidata. I risultati finali sono registrati in `VALIDATION.md` e nel riepilogo JSON dedicato della revisione.
