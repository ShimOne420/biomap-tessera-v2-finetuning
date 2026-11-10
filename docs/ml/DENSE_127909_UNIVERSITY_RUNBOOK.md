# Runbook universitario — campagna densa 127.909

## Stato e condizioni

Codice integrato su `main` tramite PR #2, su richiesta esplicita dell’utente. I run legacy non sono stati modificati. **Nessuna campagna completa eseguita durante lo sviluppo.** Le prove CPU non certificano CUDA. Il riferimento completo non è ancora pubblicato: il target storico Abruzzo sul Mac è dataless iCloud e la lettura raster è fallita. Finché `pack-reference` non completa tutti i controlli, l'estrazione completa non è avviabile. Non usare il bundle ligure come reference multi-AOI.

## 1. Recuperare il codice senza rompere il run attivo

Attendere la fine del run che usa il checkout, oppure creare un secondo clone. Non aggiornare in-place src/config mentre processi Windows li rileggono. I dati del probing restano read-only. Salvare modifiche e file non tracciati prima del checkout; non applicare alla cieca stash precedenti.

```powershell
Set-Location 'C:\Users\brome\Desktop\BIOMAP_TESSERA_V2\biomap-tessera-v2-finetuning'
git status --short
# Solo se ci sono modifiche da conservare:
# git stash push -u -m 'backup prima dense127909'
git fetch origin
git switch main
git pull --ff-only origin main
$Py = (Resolve-Path '.\.venv\Scripts\python.exe').Path
& $Py -m pip install -r requirements-lock.txt
& $Py -m pip install -e . --no-deps --no-build-isolation
& $Py -m pytest -q
& $Py scripts\dense_campaign.py --help
```

Usare il setup Windows già presente per creare .venv se manca; preservare l'installazione CUDA funzionante. Non sostituire Torch CUDA con CPU. S1 MPC può richiedere `PC_SDK_SUBSCRIPTION_KEY`; impostarla nell'ambiente, non nei file Git. Errori 401/403 vanno risolti prima del download, senza cambiare sorgente o modalità.

## 2. Preparare il riferimento (una volta, sulla macchina con gli artefatti V1)

Ripristinare tramite Finder “Scarica ora” tutti i TIFF target e gli artefatti iCloud necessari. La cartella storica deve contenere il run `20260928T111520077028Z`, i target indicati dal suo `training_dataset_report.json`, e il parquet Liguria. La sorgente legacy deve contenere geometry.json/npz per tutte le 18 AOI. I percorsi seguenti sono **segnaposto da sostituire**:

```powershell
& $Py scripts\dense_campaign.py pack-reference `
 --project 'D:\PERCORSO_PROGETTO_V1_COMPLETO' `
 --v1-root 'F:\PERCORSO_TILE_V1_PER_AOI' `
 --source-root 'F:\PERCORSO_CAMPAGNA_V2_LEGACY' `
 --reference 'F:\dense127909_reference' --work-root 'F:\dense127909_work'
```

La v1-root deve contenere <aoi>/tessera/2020 con tile int8, scale e landmask. Il pack ricalcola le feature V1, determina trunc/floor tramite confronto numerico, verifica i centri fisici, conta duplicazioni cross-tile senza eliminarle, e controlla target, join, conteggi, checksum e fold; interrompe se non coincidono. Copiare l'intera reference completata sul PC universitario. Pubblicazione Release e checksum del pacchetto saranno possibili solo dopo la verifica reale: **non esiste ancora un URL multi-AOI da scaricare**. Le serie 25 punti non sono sufficienti. Non sono inseriti checkpoint o dati satellitari nei commit.

## 3. Percorsi universitari e audit

```powershell
$Reference = 'F:\dense127909_reference'
$Source = 'F:\PERCORSO_CAMPAGNA_V2_LEGACY' # sorgente opzionale: se assente prepare usa API
$Work = 'F:\dense127909_work'             # cache e risultati nuovi; esterna a Source/Reference
$Checkpoint = (Resolve-Path '..\data\weights\student_medium.pt').Path
$Common = @('--reference',$Reference,'--source-root',$Source,'--work-root',$Work,'--checkpoint',$Checkpoint)
& $Py scripts\dense_campaign.py audit @Common
```

Audit funziona anche senza reference completa e ne segnala l'assenza. Audit non scarica né addestra. Leggere `audit.json`: presenza array non equivale a verifica dei giorni/checksum. Le verifiche approfondite avvengono alla lettura; non si dichiara completamento in base a un lock.

## 4. Acquisizione esplicita degli input mancanti

```powershell
& $Py scripts\dense_campaign.py prepare @Common --reserve-gib 20
# Ripresa: stesso comando, stessi parametri e stessa work-root.
& $Py scripts\dense_campaign.py validate @Common
```

Prepare esegue rete e scrive solo nella work-root. Congela cataloghi STAC e salva giorni/chunk con marker e hash. Mantiene un margine di 20 GiB più 4 GiB prima di ogni chunk, ma il totale della campagna dipende dalle scene: non partire con soli pochi GB. Archivi legacy presenti sono letti senza modifiche. Download con dataset incompleto o problemi di accesso non va confuso con completamento.

## 5. Profiling reale e estrazione

```powershell
# Pilot iniziale limitato: scegliere AOI con input già disponibili.
& $Py scripts\dense_campaign.py profile @Common --device cuda:0 --aoi liguria_levante --profile-chunks 3
# Leggere profile.json; il pilot ligure non rappresenta automaticamente tutte le AOI.
& $Py scripts\dense_campaign.py extract @Common --device cuda:0
# Ripresa dopo stop: comando identico; --resume è esplicito ma il riuso è già predefinito.
& $Py scripts\dense_campaign.py extract @Common --device cuda:0 --resume
```

Default FP32, 4.096 pixel/chunk, 32 pixel/batch encoder. OOM dimezza solo il batch fino a 1. Non riduce punti o sequenze. AMP è opzionale CUDA e richiede work-root separata; non mescolare precisioni. `profile.json` misura tempo, pixel/s, celle complete, memoria GPU e RAM dove supportata; RAM Windows usa GetProcessMemoryInfo; può risultare null se la chiamata non è disponibile. Stime lineari limitate al campione, senza tempi garantiti. I GZIP sono decompressi nella staging scrivibile, una volta, non nella sorgente.

`v2.parquet` viene pubblicato solo se tutti i conteggi richiesti coincidono. Le estrazioni filtrate scrivono `v2_subset.parquet` e non sono una campagna completa. Le cache di chunk evitano doppie somme dopo interruzioni.

## 6. Probing e confronto

```powershell
& $Py scripts\dense_campaign.py probe @Common --encoder v2
& $Py scripts\dense_campaign.py evaluate-liguria @Common --encoder v2
# V1: avvio esplicito, non automatico.
& $Py scripts\dense_campaign.py probe @Common --encoder v1
& $Py scripts\dense_campaign.py evaluate-liguria @Common --encoder v1
& $Py scripts\dense_campaign.py compare @Common
& $Py scripts\dense_campaign.py export @Common
```

Cinque regressori, tre fold geografici originali, refit sviluppo. Selezione RMSE macro su 17 AOI OOF, mai Liguria. Le regressioni sono CPU. Dummy e scaler usano solo training. `all` riunisce estrazione/probing V2/Liguria/export dopo preflight e profile; non scarica e non lancia V1 automaticamente:

```powershell
& $Py scripts\dense_campaign.py all @Common --device cuda:0
```

## 7. Risultati e verifica finale

Nella work-root: `v2.parquet`, `chunks`, `inputs`, `probing/v1`, `probing/v2`, `metrics.csv`, `cell_predictions.csv`, `results.xlsx`, `plots`, `report.md`, `comparison_v1_v2_dense_127909.csv`. Modelli fold/finali e hash separati. CSV/JSON sono la fonte primaria. Chiudere results.xlsx prima di export; un errore Excel si recupera con export senza rifit. Il comando è idempotente per la medesima campagna; usare una work-root nuova per nuovi esperimenti.

Confermare 127.909 sviluppo, 4.991 Liguria, OOF una volta per cella/modello, cinque regressori e supporti identici V1/V2. Umbria nord esclusa. MAPE non è MAE×100. Un buon accordo con GLOBIO riproduce un target modellato, non prova accuratezza della biodiversità sul campo.
