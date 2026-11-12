# V2 Dense frozen — Conv1D sui canali e confronto controllato

**Nessun fine-tuning. Nessun nuovo download o encoding.** L'esperimento usa esclusivamente la matrice densa completata. La Conv1D non è una CNN spaziale: lavora sull'ordine delle 128 componenti apprese. Il [report dettagliato](REPORT_IMPLEMENTAZIONE.md) spiega audit, architettura, protocollo, file, test e limiti.

## 1. Copia e ambiente separati sul PC universitario

Non fare pull, checkout o installazioni nel progetto usato dal run attivo o dall'altro agente. Creare un clone distinto. Questi comandi non avviano training; l'installazione usa rete/disco, quindi programmarla senza rallentare l'acquisizione in corso.

```powershell
Set-Location 'C:\Users\brome\Desktop\BIOMAP_TESSERA_V2'
git clone --single-branch --branch v2-dense-cnn https://github.com/ShimOne420/biomap-tessera-v2-finetuning.git biomap-v2-dense-cnn
Set-Location '.\biomap-v2-dense-cnn'
powershell -ExecutionPolicy Bypass -File scripts\setup_dense_heads.ps1
$Py = (Resolve-Path '.\.venv\Scripts\python.exe').Path
& $Py -m pytest tests\test_dense_cnn_heads.py -q
```

Setup usa Python3.12 e Torch2.10 CUDA12.8 come il repository condiviso, in una .venv nuova. Non installa dati o pesi. Per un ambiente di sole prove CPU, passare `-CpuOnly` al setup; prima del run GPU servirà il setup CUDA.

Smoke isolato facoltativo: esegue 26 fit minuscoli, tutti sintetici, su CPU e in una cartella nuova. Non chiamarne le metriche risultati MSA:

```powershell
$Smoke = "F:\biomap_heads_smoke_$(Get-Date -Format yyyyMMddTHHmmss)"
& $Py scripts\train_v2_dense_cnn.py smoke --out $Smoke --device cpu
```

## 2. Attendere il completamento dell'estrazione e indicare l'output reale

```powershell
# Percorso confermato dall'operatore del run attuale.
$Dense = 'F:\biomap_dense_public\work'
$Run = 'F:\biomap_v2_dense_cnn_run01'
```

$Run deve essere una cartella nuova, separata da $Dense (neppure una sottocartella). Attesi in $Dense:

- `v2.parquet`: cell_id, target MSA unitario e mean_000..mean_127. Metadati aoi_id/partition/cv_fold/lon/lat sono controllati o uniti dal manifest congelato.
- `extraction_complete.json`: SHA256 del parquet, n_cells=132900, complete_campaign=true e method=dense127909-stream-legacy-exact-r1. Un marker senza questo metodo è rifiutato.
- 127.909 celle sviluppo/17 AOI più 4.991 Liguria; fold storici invariati. I numeri approssimati nel prompt non sostituiscono il manifest.

Se il produttore salva il parquet/marker con nomi diversi, specificare `--features 'PERCORSO_REALE.parquet' --completion 'PERCORSO_REALE.json'`. È supportata anche la sequenza f000..f127. Se il target è in un file separato, usare `--targets 'FILE_REALE.csv'` (cell_id,target); il codice non ricalcola o sostituisce un target già presente. Il method è verificato anche per nomi di file personalizzati. Se manca un marker di completamento con checksum, il run deve fermarsi: non inventarlo mentre l'estrazione scrive.

## 3. Preflight e profiling, solo quando i dati sono completi

```powershell
& $Py scripts\preflight_v2_dense_cnn.py --source-root $Dense --out $Run --device cuda:0
if ($LASTEXITCODE -ne 0) { throw 'Dati/ambiente non pronti: non avviare il training' }

$Profile = 'F:\biomap_v2_dense_cnn_profile01'
& $Py scripts\train_v2_dense_cnn.py profile --source-root $Dense --out $Profile --device cuda:0 --profile-cells 4096 --profile-epochs 3
if ($LASTEXITCODE -ne 0) { throw 'Profiling non completato' }
Get-Content "$Profile\profile.json"
```

Preflight legge due volte gli hash per intercettare scritture concorrenti, verifica valori finiti, target [0,1], celle/fold/AOI, assenza di sovrapposizione e device con una piccola operazione CUDA. Scrive solo nell'output nuovo. Non aspetta in background né interviene sull'estrazione.

Il profilo misura tre epoche di una sola Conv1D su un sottoinsieme di sviluppo, mai su Liguria. Restituisce tempo/epoca, picco CUDA, dimensione matrice e una stima lineare del budget neurale. Non include le baseline CPU e l'export; non è una promessa di durata né una valutazione scientifica.

## 4. UN comando per il run completo, dopo la fine del run attuale

```powershell
& $Py scripts\train_v2_dense_cnn.py run --source-root $Dense --out $Run --device cuda:0
```

Esegue preflight, tre fold per Dummy/Ridge/HistGB e le quattro configurazioni neurali (Conv1D/MLP × identity/sigmoid), scelta su macro-RMSE AOI, refit e test ligure separato. Totale 26 fit: 12 classici, 12 neurali di CV e 2 refit neurali. Parametri predefiniti nel config.yaml della cartella esperimento. Per una configurazione propria copiare il YAML e passare `--config 'MIO_FILE.yaml'` prima di iniziare.

Per riprendere dall'ultima epoca completata, senza ricodificare dati:

```powershell
& $Py scripts\train_v2_dense_cnn.py run --source-root $Dense --out $Run --device cuda:0 --resume
```

Non aggiornare codice/configurazione/dati mentre il training gira. Resume richiede identità di dati, codice, configurazione, hardware/device e split. Un fit completo viene recuperato; predizioni e report sono ricostruiti senza append duplicati. L'epoca parziale interrotta viene ripetuta.

## 5. SECONDO terminale — monitor live

Aprire un nuovo terminale PowerShell, mantenendo il primo aperto:

```powershell
Set-Location 'C:\Users\brome\Desktop\BIOMAP_TESSERA_V2\biomap-v2-dense-cnn'
$Py = (Resolve-Path '.\.venv\Scripts\python.exe').Path
$Run = 'F:\biomap_v2_dense_cnn_run01'
& $Py scripts\train_v2_dense_cnn.py monitor --out $Run --interval 5
```

Mostra stato/fase/modello, fold, fit completati/26, epoca, celle elaborate, ETA epoca/fit e campagna. Le ETA sul budget massimo sono estrapolazioni dalle epoche misurate, non limiti garantiti: architetture, baseline CPU, export ed early stopping cambiano la durata. Prima delle prime misure compare “non disponibile”. Durante fit sklearn senza callback un aggiornamento vecchio non dimostra blocco. Nella fase export non resta training, ma Excel/grafici possono richiedere tempo.

Su Windows il lettore richiede FILE_SHARE_DELETE e chiude subito il file dopo una copia in memoria, prima di decodificare JSON; il writer ha retry limitati: leggere il progresso non deve impedire l’atomic replace o terminare il training. Il monitor legge solo il progress.json del NUOVO output, mai stream_progress.json/stream_status.json/stream_log.jsonl/stream_errors.jsonl della campagna attiva, non l'estrazione e non la GPU. Ctrl+C qui ferma solo il monitor. Termina automaticamente su complete/failed/interrupted. `--once` produce una sola fotografia dello stato.

## 6. Dove leggere risultati e recuperare un blocco Excel

Tutto in $Run: protocol.json, preflight.json, selection.json, models/*/best.pt/last.pt/history.csv (neurali), joblib (baseline), predictions.csv, metrics.csv, report.md, results.xlsx e PNG. Il report di risultato viene generato dai file reali, non da numeri attesi.

```powershell
Get-Content "$Run\selection.json"
Import-Csv "$Run\metrics.csv" | Where-Object { $_.aggregation -eq 'pooled' } | Format-Table model,partition,mae,rmse,mape_pct,bias,r2,pearson_r
# Se Excel ha bloccato l'export: chiuderlo, poi questo comando NON rifà il training.
& $Py scripts\train_v2_dense_cnn.py export --out $Run
```

CSV/JSON sono la fonte di verità. Excel è derivato, con metriche e dettagli per modello/partizione. Il confronto storico scalare è separato in historical_reference_indicative.json; per un confronto verificato fornire predizioni vecchie con celle/target/modello/partizione:

```powershell
& $Py scripts\train_v2_dense_cnn.py compare --out $Run --historical 'PERCORSO_REALE_PREVISIONI_STORICHE.csv'
```

Sono richieste colonne cell_id,model_name,partition,target,prediction. I modelli e le partizioni dichiarati nel file storico devono avere esattamente tutte le medesime celle del nuovo run. Un confronto non appaiabile viene rifiutato; un valore storico aggregato da solo non viene presentato come risultato nuovo.
