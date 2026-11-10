> Documento storico archiviato il 10 novembre 2026. Per il workflow attivo leggere [README](../../README.md) e [PIANO_OPERATIVO](../../PIANO_OPERATIVO.md). I comandi seguenti descrivono il vecchio stato del repository.

# Piano operativo BIOMAP-FM — E2, E4, E5

## Stato e obiettivo della consegna del 9 novembre 2026

Questa consegna implementa **E2: TESSERA v2 Medium congelato + CNN spaziale**. Il probing
universitario può continuare sul proprio checkout. Nessun training scientifico E2 è stato
avviato durante lo sviluppo. Le verifiche CPU con dati sintetici non sono risultati MSA.

E4 (partial fine-tuning) ed E5 (LoRA) rimangono le due fasi successive: **questa versione
non ne implementa o avvia il training**. I checkpoint di warmup e la CNN di E2 costituiscono
il riferimento da mantenere identico nelle due implementazioni future.

| Esperimento | Encoder | Testa | Obiettivo |
|---|---|---|---|
| E2, implementato qui | Medium congelato | CNN 5×5 | Misurare il riferimento senza aggiornare TESSERA |
| E4, successivo | Ultimi 2 blocchi, pooling, fusione | Stessa CNN | Misurare l'effetto del partial fine-tuning |
| E5, successivo | Pesi base congelati, adapter LoRA | Stessa CNN | Misurare un adattamento con meno parametri |

Il probing con embedding medi e regressori resta un riferimento aggiuntivo. Una sua
differenza rispetto a E2 include il cambiamento della testa e dell'aggregazione: non misura
da sola l'effetto dell'aggiornamento dell'encoder.

## 1. Regole durante il probing attivo

- Non aggiornare il checkout `biomap-tessera-v2-finetuning` che sta eseguendo il probing.
- Non modificare `src/biomap_v2`, YAML, `cells.csv`, serie, lock o run di quel checkout.
- Usare la copia indipendente `biomap-finetuning-dev`, branch `finetuning-dev`.
- La sorgente `data\msa_v1_liguria` è letta soltanto. Ogni nuovo file è in `data_dev\E2`.
- Aspettare la fine del probing prima di generare la cache puntuale su GPU, profilare CUDA
  o addestrare E2. I comandi non monitorano né interrompono il probing automaticamente.
- Non rilanciare download o preparazione Sentinel. Se manca un input, il comando si ferma
  con il percorso e la causa; non cambia automaticamente campione o protocollo.

La separazione è effettiva anche nei percorsi risolti: source/work uguali, annidati o
collegati tramite symlink vengono rifiutati. Il lock E2 è creato solo nella work-root.

## 2. Dati e protocollo scientifico

**Target:** GLOBIO overall MSA 2020 sulla griglia originale EPSG:4326 di 10 secondi d'arco
(`1/360` di grado). Non sono quadrati esatti di 300 m. Il rapporto `MSA-area / cell-area`
si applica soltanto al raster numeratore; un target già unitario non viene diviso di nuovo.

**Campione:** 127.909 celle di sviluppo, 17 AOI; 4.991 celle di Liguria Levante come
benchmark già osservato. Non si costruisce un nuovo campione casuale. Umbria nord resta
esclusa; questa implementazione non la prepara o valuta.

| Fold (`cv_fold`) | Training | Validation |
|---|---:|---:|
| 0 | 89.101 | 38.808 |
| 1 | 83.014 | 44.895 |
| 2 | 83.703 | 44.206 |

Ogni AOI di sviluppo appartiene a un solo fold. Ogni cella riceve una sola predizione OOF
(out-of-fold), da un modello che non ha appreso dai target della sua AOI.

Le quattro eccezioni Puglia restano esplicite, con gli stessi soli punti validi:

| cell_id | Punti validi attesi | Copertura |
|---|---:|---:|
| `17443:70453` | 18/25 | 72% |
| `17443:70454` | 19/25 | 76% |
| `17445:70460` | 19/25 | 76% |
| `17490:70434` | 15/25 | 60% |

L'audit verifica i conteggi effettivi e il flag `low_coverage_accepted`; non abbassa la
soglia generale 0,8. Una cella senza input informativo non viene accettata.

## 3. Architettura E2 e preparazione della cache

Le serie Sentinel dei singoli punti vengono normalizzate con le costanti e l'ordine bande
ufficiali. Sentinel-1 ascendente e discendente sono normalizzati separatamente e poi uniti
nel tempo. Il formato radar salvato è intensità MPC convertita in dB e poi in int16
`(dB + 50) * 200`. Non riapplicare conversioni manuali alle serie preparate.

Per ogni punto, Medium produce 128 componenti. Le coordinate dei punti ricostruiscono
il reticolo 5×5 con nord in alto: **non** si fa reshape di un embedding medio per cella.
I punti mancanti hanno feature placeholder nulle e maschera zero, non target zero.

```text
serie S1/S2 per punto → Medium frozen → [128,5,5] + maschera
    → Conv 1×1 (129→32) → Conv 3×3 (32→32) → GroupNorm → ReLU
    → Conv 1×1 (32→1) → sigmoid → pooling con maschera e aree degli strati
    → una previsione MSA nella cella GLOBIO
```

Le aree degli strati sono calcolate sull'ellissoide WGS84. Il pooling rinormalizza sui
soli strati validi. Questa è una stima da campionamento spaziale, non una media densa di
tutti i pixel a 10 m. Le predizioni interne della CNN non sono MSA validato a 10 m.

La cache SQLite conserva griglia, maschera e pesi in FP32, con checksum per record.
Dipende da checkpoint, dataset, preprocessing e protocollo. Un rilancio riutilizza le
celle già registrate e non carica neppure l'encoder se la cache è completa. Un arresto
può richiedere di ricalcolare solo l'ultima transazione non ancora committata (max 63 celle).
Sono circa 1,61 GiB di payload per 132.900 celle, oltre agli indici e al journal SQLite.
Spazio e tempo delle serie originali restano separati e non vengono stimati come zero.

## 4. Setup nel PC universitario: una sola volta

Aprire un nuovo terminale PowerShell. **Non eseguire questi comandi nel checkout del
probing.** Il clone separato non cambia il codice che il processo attivo sta usando.

```powershell
Set-Location 'C:\Users\brome\Desktop\BIOMAP_TESSERA_V2'
git clone --branch finetuning-dev https://github.com/ShimOne420/biomap-tessera-v2-finetuning.git biomap-finetuning-dev
Set-Location '.\biomap-finetuning-dev'
git branch --show-current
git log -1 --oneline
code .
```

Se la copia separata esiste già, verificare il branch e fare `git pull --ff-only origin
finetuning-dev` **in quella copia**, mentre nessun suo comando E2 è in corso. Un pull di
`main` non recupera questo branch finché la PR non viene integrata.

Installare l'ambiente dedicato. Il doctor del setup scrive soltanto nella cartella nuova:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\setup_windows.ps1 -DataRoot '.\data_dev\setup'
$taskPython = (Resolve-Path '.\.venv\Scripts\python.exe').Path
& $taskPython -m pip check
& $taskPython -m pytest
if ($LASTEXITCODE -ne 0) { throw 'Test falliti: non avviare E2' }
```

Il setup GPU esegue un piccolo controllo CUDA: eseguirlo dopo il probing. Per sole prove
di sviluppo CPU usare `-CpuOnly` in un ambiente separato, senza sostituire l'ambiente
CUDA del run attivo.

## 5. Percorsi e controlli preliminari

Impostare le variabili nel terminale di VS Code della **copia separata**:

```powershell
$taskPython = (Resolve-Path '.\.venv\Scripts\python.exe').Path
$sourceData = 'C:\Users\brome\Desktop\BIOMAP_TESSERA_V2\biomap-tessera-v2-finetuning\data\msa_v1_liguria'
$workRoot = Join-Path $PWD 'data_dev\E2'
$e2Args = @('--config', 'configs\e2_msa.yaml', '--source-data-root', $sourceData, '--work-root', $workRoot)
Test-Path "$sourceData\cells.csv"
Test-Path "$sourceData\series"
Test-Path "$(Split-Path $sourceData -Parent)\weights\student_medium.pt"
& $taskPython -m biomap_ml.tools.check_environment @e2Args --device cpu
```

Tutti i `Test-Path` devono dare `True`. Se i pesi si trovano altrove, aggiungere a `$e2Args`
`--checkpoint` e il percorso del **Medium originale**, non un checkpoint adattato:

```powershell
# Solo se serve: indicare il percorso reale già presente sul PC.
# $e2Args += @('--checkpoint', 'D:\BioMAP\weights\student_medium.pt')
```

`--source-data-root` indica la cartella con `cells.csv`, non il suo genitore `data`.
Se il target non si trova più nel percorso registrato da `targets/source.json`, fermarsi
e diagnosticare il trasferimento: non correggere il puntatore del run attivo durante il probing.

Audit completo, senza nuovo training:

```powershell
& $taskPython -m biomap_ml.prepare @e2Args --stage index --device cpu
if ($LASTEXITCODE -ne 0) { throw 'Audit fallito: leggere la causa prima di procedere' }
Get-Content "$workRoot\dataset\audit.json"
```

Controllare 127.909 sviluppo, 4.991 test, 17 AOI e i tre fold riportati sopra. L'audit legge
e verifica tutte le serie: può richiedere I/O consistente e rallentare il probing, quindi
è preferibile eseguirlo dopo la sua conclusione.

## 6. Dopo la fine del probing: cache, profilo e training E2

Generare gli embedding puntuali una volta sola. Il checkpoint rimane invariato e l'output
viene salvato in `data_dev\E2\cache`, senza modificare gli embedding medi del probing.

```powershell
& $taskPython -m biomap_ml.prepare @e2Args --stage cache
if ($LASTEXITCODE -ne 0) { throw 'Cache non completa: non avviare training' }
& $taskPython -m biomap_ml.tools.profile_training @e2Args --steps 20
if ($LASTEXITCODE -ne 0) { throw 'Profiling fallito' }
Get-Content "$workRoot\profile_E2.json"
```

Il profilo misura la CNN con cache, non tutta la futura strategia E4/E5. Distingue il
budget di training dalle componenti escluse (encoder/cache, validation, checkpoint).

**Comando del run E2 completo: tre fold e refit. Non esegue ancora il test Liguria.**

```powershell
& $taskPython -m biomap_ml.train @e2Args --run-id e2_msa_seed42 --phase all
if ($LASTEXITCODE -ne 0) { throw 'Training interrotto: usare la procedura resume' }
```

Per eseguire soltanto la CV usare `--phase cv`; per fare il refit dopo una CV conclusa,
usare lo stesso run-id con `--phase refit --resume`.

## 7. Regole di training, selezione e resume

Per ogni fold il warmup dura 5 epoche, indipendentemente dal punteggio. E2 riparte dalla
sua testa con un optimizer nuovo, poi addestra al massimo altre 50 epoche con patience 8.
Il migliore checkpoint del fold minimizza RMSE medio per AOI nella sua validation.

La metrica complessiva di selezione è RMSE medio delle 17 AOI sulle predizioni OOF. Il
budget del refit è la mediana delle migliori epoche **dopo** warmup dei tre fold. Il
refit ripete warmup e quel numero fissato di epoche sulle sole 127.909 celle: non vede
Liguria per early stopping e non riusa un warmup supervisionato appartenente a un altro fold.

Parametri iniziali: seed 42, microbatch 2, batch effettivo 16, AdamW LR `1e-3`, weight decay
0,01, MSE e clipping 1. AMP usa BF16 se supportato, altrimenti FP16/GradScaler. Sono
configurati in `configs/e2_msa.yaml`; cambiarli dopo il test richiede una nuova run.

In caso di interruzione:

```powershell
& $taskPython -m biomap_ml.train @e2Args --run-id e2_msa_seed42 --phase all --resume
```

Si riparte dall'ultima epoca completata. Optimizer, scaler, RNG, global step e configurazione
sono salvati. Un OOM ripristina l'inizio epoca da disco e dimezza il microbatch, mantenendo
batch effettivo e celle. Con microbatch 1 il comando si ferma; non riduce punti o sequenze.

## 8. Benchmark Liguria, Excel e mappe

Congelato il refit, eseguire il test e le esportazioni:

```powershell
& $taskPython -m biomap_ml.evaluate @e2Args --run-id e2_msa_seed42
if ($LASTEXITCODE -ne 0) { throw 'Valutazione/esportazione incompleta: leggere la causa' }
& $taskPython -m biomap_ml.predict @e2Args --run-id e2_msa_seed42
```

Risultati principali:

```text
data_dev/E2/
  dataset/index.csv, audit.json
  cache/points_<dataset_id>.sqlite3 + JSON
  profile_E2.json
  E2_experiment_registry.xlsx
  runs/e2_msa_seed42/
    config.json, audit.json, run.json
    fold_0..2/warmup/last.pt, E2/best.pt, E2/last.pt, history.csv
    oof_predictions.csv, selection.json
    refit_warmup/, refit_E2/, final.pt, final.json
    test_predictions.csv, metrics.csv, report.md
    validation_quintiles.csv, liguria_test_quintiles.csv, test_bootstrap.json
    plots/validation.png, liguria_test.png
    geotiff/*.tif, metadata.json
```

L'Excel di E2 è **separato** da quello del probing attivo. Contiene `Summary`, `Risultati`
e i fogli per cella OOF/test di ogni run. È ricostruito dai CSV delle run senza duplicare
righe. Se è aperto, chiuderlo e rilanciare `evaluate`: le predizioni esistenti vengono
riutilizzate, non si ripete il test o il training. Nessuna copia `_latest.xlsx` sostituisce
silenziosamente il canonico.

Le mappe sono COG EPSG:4326 sulla griglia nativa, con predetto, osservato, Dummy, errori e
copertura; mancanze a NoData -9999. Non è esportata un'incertezza calibrata inventata.

## 9. Milestone e decisioni successive

| Milestone | Consegna/controllo | Stato in questa fase |
|---|---|---|
| M0 | Base universitaria pubblicata; checkout isolato | Base `741bca9`, branch separato |
| M1 | Audit dati, target, griglia e quattro celle | Codice e fixture; audit reale da eseguire sul PC |
| M2 | Griglia 5×5 e CNN comune | Implementate e testate su CPU |
| M3 | Cache puntuale e training E2 | Implementati; campagna reale non avviata |
| M4 | E4 partial: ultimi 2 blocchi + pooling/fusione | Prossima implementazione |
| M5 | E5 LoRA: rank 8, alpha 16, dropout 0,05 su FFN/fusione | Prossima implementazione |
| M6 | Test, resume, OOM, metriche, Excel, mappe | Evidenze nel documento di handover/CI |
| M7 | Branch, draft PR, controlli Windows/Linux | Consultare PR e ultimo SHA pubblicato |
| M8 | Preflight GPU e run scientifici universitari | Successivi al probing, avviati dall'utente |

Prima di E4/E5 conservare cache, manifest, checkpoint warmup dei tre fold e refit, parametri
della CNN e predizioni E2. E4/E5 devono attraversare l'encoder durante il training: la
cache frozen non permette di aggiornare il backbone. Il loro confronto con E2 userà le
stesse celle, target, maschere, pesi, testa e criterio di selezione.

Finestre preliminari del piano: E2 30 min–2 h con cache pronta; cache 1–6 h una tantum;
E4 12–48 h ed E5 12–60 h con circa 20 epoche effettive/fit. **Non sono misure di questa
implementazione né garanzie.** Il profilo universitario deve aggiornarle; 50 epoche possono
aumentare la parte training di circa 2,5×. LoRA non è automaticamente più veloce.

Concluderemo sul miglioramento nel riprodurre GLOBIO, non sulla biodiversità misurata in
campo. La Liguria resta un benchmark noto. Nessun vincitore E2/E4/E5 è dichiarato ora.

Dettaglio del codice e dei controlli: [handover](docs/ml/TESSERA_IMPLEMENTATION_HANDOVER.md).
