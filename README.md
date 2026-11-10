# Aggiornamento: campagna densa 127.909

La nuova CLI è `scripts/dense_campaign.py`, branch `dense-127909`. [Runbook universitario](docs/ml/DENSE_127909_UNIVERSITY_RUNBOOK.md) e [audit con blocchi reali](docs/ml/DENSE_127909_AUDIT.md). Non esegue training o acquisizioni automaticamente. Il reference multi-AOI è ancora bloccato da target storici iCloud non leggibili; non confondere codice implementato con dati pronti. Nessuna nuova metrica scientifica prodotta.

---

# BIOMAP-FM — probing TESSERA V2 denso come V1

**Workflow attivo:** encoder V1/V2 congelati → stessa media densa di 128 componenti →
Dummy, Ridge, HistGB, XGBoost e MLP. Nessun fine-tuning in questa fase.

Il primo pacchetto eseguibile copre **Liguria Levante: 4.991 celle, 3.545.951 pixel densi,
cinque fold spaziali interni**. Non è la campagna sviluppo→Liguria sulle 127.909 celle:
gli embedding V2 densi delle 17 AOI di sviluppo non sono ancora completi.

## Quale documento leggere

- Questo README: apertura del progetto, download dati e comandi Windows.
- [README_MODIFICHE_DENSE.md](README_MODIFICHE_DENSE.md): metodo recuperato e modifiche tecniche.
- [PIANO_OPERATIVO.md](PIANO_OPERATIVO.md): milestone, controlli e decisioni successive.
- [BIOMAP_DEVELOPMENT_STATUS.md](BIOMAP_DEVELOPMENT_STATUS.md): stato effettivo e problemi aperti.
- [Roadmap condivisa](BIOMAP_DEVELOPMENT_ROADMAP.md): riferimento strategico fornito dall'utente.

I vecchi README a 25 punti e il piano E2/E4/E5 sono in `docs/archive/` come documentazione
storica. **E4/E5 sono sospesi dalla nuova roadmap frozen.** Il codice legacy non viene cancellato.

## 1. Aggiornare il PC universitario senza perdere modifiche

Eseguire il pull soltanto quando nessun run sta usando il checkout da aggiornare. Se il
run è ancora attivo, attendere oppure clonare in una cartella separata: non cambiare i
moduli Python importati dal processo. Non aggiornare il suo ambiente durante il run.

Aprire PowerShell nel terminale VS Code:

```powershell
Set-Location 'C:\Users\brome\Desktop\BIOMAP_TESSERA_V2\biomap-tessera-v2-finetuning'
git status --short
git stash list
```

L'allegato segnala modifiche universitarie **non committate** e un vecchio README nello
stash. Non perderle: se ci sono modifiche tracked, prima del pull salvarle localmente:

```powershell
# Solo se git status mostra modifiche locali; questo salva i file tracked.
git stash push -m 'backup-university-before-dense-portable'
```

Se ci sono file di codice nuovi non tracciati, salvarli/committarli su un branch di backup
prima del pull. Non cancellarli per forzare l'aggiornamento. Non applicare automaticamente
lo stash sul codice nuovo: XGBoost/MLP/report dinamici sono già integrati. Il vecchio stash
resta recuperabile tramite `git stash list` e `git stash show -p <nome>`.

```powershell
git switch main
git pull --ff-only origin main
if ($LASTEXITCODE -ne 0) { throw 'Pull non completato: non forzare o cancellare modifiche' }
code .
$TaskPython = (Resolve-Path '.\.venv\Scripts\python.exe').Path
& $TaskPython -m pip install -r requirements-lock.txt
if ($LASTEXITCODE -ne 0) { throw 'Dipendenze non installate' }
& $TaskPython -m pip check
& $TaskPython -m pytest tests\test_dense_benchmark.py tests\test_dense_portable.py -q --disable-warnings
if ($LASTEXITCODE -ne 0) { throw 'Test falliti: non avviare il benchmark' }
```

Per un clone nuovo senza `.venv`, usare prima `scripts/setup_windows.ps1` in quella copia
con `-DataRoot '.\local_data'`; non esegue download delle serie. Per le regressioni dense
basta CPU. Il controllo del piccolo campione encoder può usare il checkpoint CUDA già disponibile.

## 2. Scaricare i dati del run: non serve l'SSD del Mac

`git pull` porta codice e configurazioni. Il pacchetto dati è un asset della
[Release dense-liguria-20261110](https://github.com/ShimOne420/biomap-tessera-v2-finetuning/releases/tag/dense-liguria-20261110).
Il comando seguente lo scarica, verifica dimensione/SHA256 e lo estrae senza sovrascrivere dati:

```powershell
$TaskBundle = 'F:\dense_liguria_bundle_20261110'
& $TaskPython scripts\dense_msa.py fetch-bundle --out $TaskBundle
if ($LASTEXITCODE -ne 0) { throw 'Download o verifica pacchetto falliti' }
& $TaskPython scripts\dense_msa.py verify-bundle --bundle $TaskBundle
if ($LASTEXITCODE -ne 0) { throw 'Dati del pacchetto non validi' }
```

Se F: non è disponibile, scegliere un percorso nuovo su C:. Il download è **11.904.407 byte**,
non decine di GB. Contiene medie dense V1/V2, target, fold, prove di equivalenza e un campione
Sentinel di 32 pixel. **Non** contiene tutte le immagini originali né il checkpoint Medium.
Gli embedding non vengono ricalcolati per ogni regressore.

Se il pacchetto è già presente e verificato, `fetch-bundle` lo riusa. Con un errore di rete
si ferma: nessuna sequenza infinita di tentativi o fallback al campione a 25 punti.

## 3. Eseguire il probing V1/V2 denso

Il checkpoint universitario è già presente, secondo l'allegato. Adattare solo il suo percorso:

```powershell
$TaskCkpt = (Resolve-Path '..\data\weights\student_medium.pt').Path
$TaskOut = Join-Path $PWD ('outputs\dense_benchmark\' + (Get-Date -Format 'yyyyMMddTHHmmss') + '_dense_5reg')
& $TaskPython scripts\dense_msa.py all `
  --bundle $TaskBundle --checkpoint $TaskCkpt --device cuda:0 `
  --folds 5 --bootstrap 2000 --out $TaskOut
if ($LASTEXITCODE -ne 0) { throw 'Run incompleto: leggere errore e run_state.json' }
```

**Questo è il comando del run.** Prima dei fit verifica tutti gli hash e ricodifica solo
il campione di 32 pixel con il Medium locale, confrontandolo con gli embedding originali.
Poi esegue i regressori su CPU, usando gli stessi fold congelati per V1 e V2.

Se vuoi evitare del tutto la GPU, usa `--device cpu`: cambia il device della verifica del
campione, non le feature dense. Per il solo training tabellare, dopo aver verificato il
pacchetto, è disponibile `run --bundle ... --out ...`; il report dichiarerà che in quella
invocazione non è stato ripetuto il controllo encoder. Il comando consigliato resta `all`.

Non rilanciare `probe_msa.py prepare` per questo benchmark: quello è il percorso a 25 punti.
Non usare `finetune.py` e non passare i vettori medi alla CNN E2 come se fossero patch.

## 4. Dove trovare e leggere i risultati

```powershell
Get-Content "$TaskOut\REPORT_V1_V2_DENSE_MSA.md"
Import-Csv "$TaskOut\metrics.csv" |
  Select-Object encoder,model,n,mae,rmse,mape_pct,bias,r2,pearson_r,rmse_gain_vs_dummy_pct |
  Format-Table
Get-Content "$TaskOut\run_state.json"
```

| File/cartella | Contenuto |
|---|---|
| `protocol.json` | Versioni codice/software, hash, regressori, iperparametri e ruolo della valutazione |
| `folds.csv` | Appartenenza geografica per cella |
| `predictions.csv` | Cella, coordinate, target, encoder, regressore, predizione, errore e copertura |
| `metrics.csv`, `metrics_by_fold.csv` | MAE, RMSE, MAPE, bias, R², Pearson, guadagni rispetto al Dummy |
| `paired_delta.csv` | Differenze V2−V1 e intervalli bootstrap |
| `models/` | Modello addestrato per ogni encoder/regressore/fold |
| `scatter_*.png` | Osservato contro predetto per entrambi gli encoder |
| `risultati.xlsx` | Summary, risultati, fold e valori cella per cella di questa run |

Il registro comune è `outputs\dense_benchmark\DENSE_MSA_experiment_registry.xlsx`, con una
riga per encoder/modello/run. Il registro CSV è la fonte ricostruibile. I workbook storici
dei run a 25 punti non vengono sovrascritti.

Se Excel blocca l'aggiornamento, chiuderlo e rigenerare gli Excel **senza ripetere i fit**:

```powershell
& $TaskPython scripts\dense_msa.py export --out $TaskOut
```

Un output già usato per il training viene rifiutato. Per un nuovo esperimento scegliere
un nuovo `$TaskOut`; le vecchie predizioni restano intatte.

## 5. Come interpretare il risultato

Per ogni regressore confrontare V1/V2 sulle stesse celle e con lo stesso Dummy. Una
riduzione RMSE positiva indica vantaggio sul Dummy; delta V2−V1 negativo indica minore
errore V2. Guardare anche bias, estremi del target e intervalli bootstrap spaziali.

MAPE è relativo al target di ogni cella. MAE 0,1 significa 10 punti della scala MSA,
non automaticamente un errore relativo del 10%. Il target è modellato da GLOBIO.

La CV ligure usa circa 3.993/3.992 celle di training e 998/999 di valutazione per fold.
Ogni cella ha una sola predizione OOF. Non c'è un nuovo holdout finale separato e non si
possono attribuire questi numeri al training storico sulle 17 AOI.

## 6. Preparazione del pacchetto da raw: solo per manutenzione

Questo è il comando già eseguito sul Mac per produrre il pacchetto. Non serve sul PC
universitario se si usa `fetch-bundle`:

```text
python scripts/dense_msa.py pack --v1-tiles <tessera/2020> --v2-tiles <AOI_V2>
  --target <msa_2020_unitless.tif> --v1-parquet <V1.parquet> --v2-parquet <V2.parquet>
  --out <nuova_cartella_bundle> --folds 5 --pixels 32
```

Replica il V1 storico, verifica gli stessi pixel e riaggrega V2 senza nuovi download o
encoding. I comandi raw `check-v1`, `check-v2`, `check-encoder`, `all` restano disponibili.
