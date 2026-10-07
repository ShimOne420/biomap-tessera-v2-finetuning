# BioMAP — TESSERA V2: probing e fine-tuning MSA/BII

Repository operativo per **Windows nativo, NVIDIA RTX 5080 16 GB, 32 GB RAM**. L'obiettivo è confrontare gli embedding V2 frozen con quelli ottenuti aggiornando i pesi dello student **Medium** (~21 milioni di parametri), per riprodurre **GLOBIO overall MSA 2020** e **NHM BII v2.1.1 2020**.

**Stato:** codice e protocollo implementati; test CPU, parità con l'inferenza ufficiale e controlli delle dipendenze sono descritti in `VALIDATION.md`. La prova CUDA/VRAM effettiva richiede il computer universitario. Nessun risultato di fine-tuning scientifico è già garantito. I test storici sono già osservati e hanno ruolo esplorativo. Umbria nord resta sigillata.

## Come è organizzato

Solo tre ingressi Python e un setup PowerShell. Ogni script/modulo/test inizia con un commento che spiega cosa fa, input, output, parametri, risorse e ripresa.

| Ingresso | Cosa fa |
|---|---|
| `scripts/probe_msa.py` | Prepara MSA, calcola embedding frozen o adattati, addestra regressori e valuta |
| `scripts/probe_bii.py` | Stesso percorso per BII, con griglia/target specifici |
| `scripts/finetune.py` | Aggiorna encoder e testa; scegli MSA/BII con `--indicator` |
| `scripts/setup_windows.ps1` | Crea ambiente Python e verifica CUDA |

`configs/` contiene YAML commentati, geometrie/split e provenienza. `src/biomap_v2/` contiene funzioni condivise: acquisizione, forward differenziabile, esperimenti e CLI. `tests/` contiene fixture sintetiche esplicitamente separate dai benchmark. La CI è in `.github/workflows/`.

## 1. Preparare il computer universitario

Aprire **PowerShell**. Installare Git e Python 3.12 a 64 bit, se assenti; con `winget`:

```powershell
winget install --id Git.Git -e
winget install --id Python.Python.3.12 -e
```

Chiudere e riaprire PowerShell dopo l'installazione. Verificare:

```powershell
git --version
py -3.12 --version
nvidia-smi
```

`nvidia-smi` deve elencare la RTX 5080. Se manca o il driver è troppo vecchio per CUDA 12.8, aggiornare il driver NVIDIA con il supporto dell'università. Non serve installare separatamente CUDA Toolkit per usare le wheel PyTorch.

Se non hai permessi per installare Git/Python o aggiornare il driver, chiedere all'amministratore queste tre componenti; il setup non aggira le restrizioni del PC.

## 2. Clonare e scegliere disco interno o SSD

```powershell
Set-Location ([Environment]::GetFolderPath('Desktop'))
git clone https://github.com/ShimOne420/biomap-tessera-v2-finetuning.git
Set-Location biomap-tessera-v2-finetuning
$DataRoot = Join-Path $env:USERPROFILE 'BioMAP_Data'
```

Per usare l'SSD, sostituire **soltanto** la variabile con il percorso effettivo del disco:

```powershell
$DataRoot = 'E:\BioMAP_Data'
```

La lettera `E:` è un esempio: controllarla in Esplora file. Repository e ambiente possono restare sul desktop; serie, pesi e run vanno nella data-root. La cache interna è limitata a 50 GiB con almeno 20 GiB liberi. Il limite considera **tutta** la data-root, anche entrambi gli indicatori e i run. Per aumentarlo su SSD usare `--set storage.max_gib=250` in tutti i comandi interessati, dopo aver controllato lo spazio.

## 3. Installare e controllare l'ambiente

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\setup_windows.ps1 -DataRoot $DataRoot
$Python = Join-Path $PWD '.venv\Scripts\python.exe'
& $Python -m pip check
& $Python scripts\probe_msa.py doctor --data-root $DataRoot
& $Python scripts\probe_bii.py doctor --data-root $DataRoot
```

Il setup installa Python package bloccati e **PyTorch 2.10.0 cu128** da indice ufficiale. `doctor` registra piattaforma, driver, GPU, VRAM, RAM su Windows, spazio libero e prova matematica forward/backward in `msa/doctor.json` o `bii/doctor.json`. Verificare che `cuda_available` sia `true` e che la GPU sia quella desiderata.

La sola disponibilità CUDA non prova che l'intero training entri nella VRAM: la prova tecnica dei punti 6–7 serve a misurarlo. Se ci sono più GPU, usare `--device cuda:1` ecc. Non usare memoria GPU condivisa come se fosse VRAM dedicata.

Su Windows, errori XGBoost riguardanti `vcomp140.dll` possono richiedere Microsoft Visual C++ Redistributable x64. Non installare wheel non ufficiali per risolvere incompatibilità della RTX 5080.

## 4. Scaricare i pesi Medium

```powershell
& $Python scripts\probe_msa.py download-weights --data-root $DataRoot
```

MSA e BII condividono `$DataRoot\weights\student_medium.pt` (~84 MB). Revisione Hugging Face e SHA256 sono fissati; un hash diverso interrompe il comando. Non si scarica il teacher 2B. Le funzioni di inferenza V1 non sono compatibili con questi pesi.

## 5. Accesso ai dati Sentinel e target

Le geometrie e le partizioni storiche sono già nel repository: **1.000 training, 122 validation e 245 test per ciascun indicatore**. Gli identificativi sono diversi per MSA e BII: stessa numerosità non significa stessi target/supporti.

`prepare` scarica separatamente NHM BII (5 arcmin, percentuale) e GLOBIO MSA (10 arcsec, MSA-area/cell-area), poi serie S2 L2A e S1 RTC del 2020. Il download Sentinel usa punti stratificati e blocchi COG; non scarica un'intera scena o embedding globali. È comunque la fase più lenta: non è un download di soli pesi.

Planetary Computer può richiedere un account/chiave per Sentinel-1 RTC. Consultare la [documentazione di accesso](https://planetarycomputer.microsoft.com/docs/concepts/sas/). Se la lettura è negata con 401/403, impostare la chiave **solo nella sessione PowerShell**, senza inserirla nei YAML o in Git:

```powershell
$env:PC_SDK_SUBSCRIPTION_KEY = [System.Net.NetworkCredential]::new('', (Read-Host 'Chiave Planetary Computer' -AsSecureString)).Password
```

In alternativa importare cache già generate con questo progetto. Una vecchia cache Mac non è automaticamente compatibile: unità radar, geometria, anno e manifest devono coincidere. Il preprocessore precedente applicava una conversione da ampiezza; qui l'intensità MPC usa `10*log10(power)`, equivalente alla conversione ufficiale di `sqrt(power)`.

Le impostazioni di default accettano solo il 2020. Per un altro anno servono target corrispondenti: configurare `target.mode=custom`, raster/provenienza/unità, geometrie coerenti e una nuova data-root.

## 6. Prima prova piccola su dati reali

Preparare 12 celle per indicatore, distribuite fra train/validation/test:

```powershell
& $Python scripts\probe_msa.py prepare --data-root $DataRoot --limit 12
& $Python scripts\probe_bii.py prepare --data-root $DataRoot --limit 12
& $Python scripts\probe_msa.py run --data-root $DataRoot --smoke
& $Python scripts\probe_bii.py run --data-root $DataRoot --smoke
```

`--limit` vale per `prepare`; salva `cells_smoke.csv`. `--smoke` seleziona quel file nei comandi successivi. Le serie sono annuali anche nello smoke: pochi giorni non rappresenterebbero il protocollo reale. Il limite è 12 celle totali, non 12 per split.

Queste metriche misurano il funzionamento del percorso; **non sono risultati scientifici**. Controllare copertura, assenza di valori non finiti, tempi e memoria. Se un errore rete interrompe `prepare`, rieseguire lo stesso comando: cataloghi, giorni e celle completati vengono verificati e riusati. Dopo il consolidamento verificato di una cella, solo i suoi intermedi per giorno vengono rimossi; i dati originali esterni restano intatti.

## 7. Prova tecnica del fine-tuning MSA e BII

```powershell
& $Python scripts\finetune.py run --indicator msa --data-root $DataRoot --smoke --stage all --set training.epochs=2 --set 'training.seeds=[42]'
& $Python scripts\finetune.py run --indicator bii --data-root $DataRoot --smoke --stage all --set training.epochs=2 --set 'training.seeds=[42]'
```

`all` esegue warmup head-only, controllo frozen aggiuntivo e fine-tuning parziale. `phase.json` deve indicare `encoder_changed: false` per il controllo e `true` per il ramo partial. Sono verifiche automatiche, non deduzioni dalla sola loss.

La GPU usa inizialmente quattro celle per microbatch, 16 effettive tramite accumulo e 32 pixel per forward. In caso di CUDA OOM viene ripetuta l'epoca con meno celle, ripristinando pesi, optimizer e RNG. Se non entra una cella, ridurre `model.pixel_batch`, per esempio:

```powershell
& $Python scripts\finetune.py run --indicator msa --data-root $DataRoot --smoke --stage all --set model.pixel_batch=8 --set training.epochs=2 --set 'training.seeds=[42]'
```

Modificare `model.pixel_batch` non cambia il campione. Ridurre i pixel campionati cambia invece il protocollo e richiede una nuova preparazione/data-root.

## 8. Preparare il campione completo e fare probing V2

```powershell
& $Python scripts\probe_msa.py prepare --data-root $DataRoot
& $Python scripts\probe_bii.py prepare --data-root $DataRoot
& $Python scripts\probe_msa.py run --data-root $DataRoot
& $Python scripts\probe_bii.py run --data-root $DataRoot
```

Senza `--smoke` si usa `cells.csv`. Il numero finale può essere minore di 1.367 se alcune celle non raggiungono la copertura Sentinel richiesta: le esclusioni sono nel manifest e tutte le varianti usano lo stesso insieme eleggibile. Errori rete non vengono convertiti silenziosamente in esclusioni.

Regressori: Dummy Mean sempre presente, Ridge con scaler train-only, HistGradientBoosting e XGBoost. Hyperparametri nel YAML, selezione solo tramite media RMSE dei blocchi geografici di validation. Il test non sceglie il regressore.

## 9. Fine-tuning progressivo completo della campagna

```powershell
& $Python scripts\finetune.py run --indicator msa --data-root $DataRoot --stage all
& $Python scripts\finetune.py run --indicator bii --data-root $DataRoot --stage all
```

Default: seed 42/43/44, massimo 50 epoche **per fase**, early stopping dopo otto epoche senza miglioramento, testa 128→64→1 (ReLU, sigmoid), MSE e AdamW. Learning rate testa 1e-3, encoder 1e-5.

Il warmup produce una testa iniziale. Da **quello stesso checkpoint**, il controllo head-only continua l'addestramento e il ramo partial sblocca gli ultimi blocchi S1/S2, attention pooling e fusione. I due rami hanno uguale budget massimo di selezione; l'early stopping può interromperli in epoche diverse. Encoder dropout disattivato in entrambi, mantenendo i gradienti nel ramo fine-tuned. Il seed viene fissato per ogni fase.

Per aggiornare **tutti** i pesi, eseguire una campagna esplicita, con warmup e controllo abbinato:

```powershell
& $Python scripts\finetune.py run --indicator msa --data-root $DataRoot --stage full
& $Python scripts\finetune.py run --indicator bii --data-root $DataRoot --stage full
```

Full parte dallo stesso tipo di warmup head-only, non automaticamente dal migliore partial; questa scelta permette il confronto con il controllo a parità di inizializzazione/budget. Valutare full dopo aver verificato partial sulla validation. Non scegliere la strategia dai risultati di test.

## 10. Personalizzare senza modificare gli script

Modificare `configs/msa.yaml` / `configs/bii.yaml`, oppure usare override ripetibili:

```powershell
& $Python scripts\finetune.py run --indicator bii --data-root $DataRoot --stage partial --set training.encoder_lr=0.000005 --set training.head_lr=0.0005 --set training.unfreeze_blocks=2 --set training.epochs=30
& $Python scripts\probe_msa.py run --data-root $DataRoot --set probing.ridge_alpha=10 --set 'probing.models=[Ridge,HistGB]'
```

Le chiavi sconosciute sono rifiutate per evitare errori di battitura. Ogni run salva gli override effettivi in `config.json`. Per cambiare indicatore basta `--indicator msa/bii`; non occorre riscrivere training o loss.

Per cambiare campionamento, usare una radice distinta e lo stesso override sia in `prepare` sia nei run:

```powershell
$DataRoot49 = 'E:\BioMAP_49pixels'
& $Python scripts\probe_msa.py prepare --data-root $DataRoot49 --set sampling.pixels_per_cell=49
& $Python scripts\probe_msa.py run --data-root $DataRoot49 --set sampling.pixels_per_cell=49
& $Python scripts\finetune.py run --indicator msa --data-root $DataRoot49 --set sampling.pixels_per_cell=49
```

`pixels_per_cell` deve essere un quadrato. Le soglie di qualità, anno e sampling restano congelati durante un confronto. Le letture COG usano il pixel nativo più vicino per banda: è un protocollo puntuale esplicito, non una parità dimostrata con tutto il mosaicking/resampling spaziale del preprocessore ufficiale. La parità verificata riguarda normalizzazione, binning e forward a parità di array. Non cambiare target/split durante un run.

## 11. Ripresa e importazione da SSD

`prepare` è riprendibile con lo stesso comando. Gli NPZ per giorno e cella hanno checksum; una cache corrotta blocca il comando, non viene accettata come valida.

Per cache prodotte da **questo repository**, indicando la loro radice originaria:

```powershell
& $Python scripts\probe_msa.py prepare --data-root $DataRoot --import-cache 'F:\BioMAP_Data'
& $Python scripts\probe_bii.py prepare --data-root $DataRoot --import-cache 'F:\BioMAP_Data'
```

Non estrarre direttamente le vecchie serie Mac nella nuova cache: la vecchia trasformazione S1 e la geometria devono essere prima verificate/adattate. L'import conserva i file originali e verifica protocollo/hash.

Per target già scaricati (non serve riscaricarli):

```powershell
& $Python scripts\probe_msa.py prepare --data-root $DataRoot --target-raster 'E:\Targets\msa_2020_unitless.tif'
& $Python scripts\probe_bii.py prepare --data-root $DataRoot --target-raster 'E:\Targets\bii-2020_v2-1-1.tif'
```

MSA importato deve essere il ratio unitless, BII la percentuale NHM; CRS, risoluzione e centri della griglia vengono verificati. Per dati alternativi impostare anche provenienza/unità nel YAML.

Ogni fase di training salva `last.pt` e `best.pt` insieme all'optimizer e RNG. Per riprendere **la singola fase interrotta**, sostituire il percorso con quello stampato/salvato nel run:

```powershell
$Interrupted = 'E:\BioMAP_Data\msa\runs\RUN_ID\partial_seed42\last.pt'
& $Python scripts\finetune.py run --indicator msa --data-root $DataRoot --resume $Interrupted
```

La ripresa richiede gli stessi override e lo stesso manifest. Un'epoca interrotta viene ripetuta dall'ultimo checkpoint completo. Per uno smoke ripetere anche `--smoke --set training.epochs=2 --set 'training.seeds=[42]'`. Conservare anche `best.pt`; non spostare `last.pt` da solo. La ripresa di una fase non avvia automaticamente gli altri seed o una nuova campagna.

## 12. Valutare, esportare embedding adattati e ripetere il probing

Ogni comando `run` stampa il percorso del run. Sostituire `RUN_ID` con il nome effettivo:

```powershell
$ProbeRun = 'E:\BioMAP_Data\msa\runs\RUN_ID_PROBE'
& $Python scripts\probe_msa.py evaluate --data-root $DataRoot --run-dir $ProbeRun
$FineRun = 'E:\BioMAP_Data\msa\runs\RUN_ID_FINETUNE'
& $Python scripts\finetune.py evaluate --indicator msa --data-root $DataRoot --run-dir $FineRun
```

`selection.json` sceglie lo stage dalla RMSE validation media fra seed, poi un checkpoint rappresentativo vicino alla mediana validation dello stage. Non sceglie automaticamente il seed più fortunato. Contiene il checkpoint scelto **sulla validation**. Non scegliere quello con il test migliore. Leggere il percorso automaticamente:

```powershell
$Selection = Get-Content (Join-Path $FineRun 'selection.json') -Raw | ConvertFrom-Json
$Best = Join-Path $FineRun $Selection.selected_checkpoint
& $Python scripts\finetune.py export-embeddings --indicator msa --data-root $DataRoot --checkpoint $Best
& $Python scripts\probe_msa.py run --data-root $DataRoot --checkpoint $Best
```

Per BII cambiare `msa` in `bii` e usare il relativo run:

```powershell
$FineRunBii = 'E:\BioMAP_Data\bii\runs\RUN_ID_FINETUNE'
$SelectionBii = Get-Content (Join-Path $FineRunBii 'selection.json') -Raw | ConvertFrom-Json
$BestBii = Join-Path $FineRunBii $SelectionBii.selected_checkpoint
& $Python scripts\finetune.py export-embeddings --indicator bii --data-root $DataRoot --checkpoint $BestBii
& $Python scripts\probe_bii.py run --data-root $DataRoot --checkpoint $BestBii
```

I checkpoint `.pt` e i modelli `.joblib` sono file eseguibili durante il caricamento: usare esclusivamente checkpoint propri o ufficiali verificati. Il progetto non carica file ricevuti da fonti sconosciute.

## 13. Leggere e confrontare i risultati

Sotto `$DataRoot\msa` o `bii`:

- `targets/`: raster e provenienza; `series/`: cataloghi, giorni e serie per cella.
- `cells.csv` e manifest: target, split, checksum, copertura ed esclusioni congelate.
- `embeddings_HASH.csv`: 128 medie per cella, con hash del checkpoint e del supporto.
- `runs/RUN_ID/`: configurazione, provenienza, modelli, `selection.json`, predizioni e report.
- Fasi fine-tuning: `last.pt`, `best.pt`, `history.csv`, `phase.json` con memoria/tempi e pesi modificati.

`historical_reference.csv` conserva le metriche dei run V1 precedenti con ruolo descrittivo e supporto diverso, senza calcolare delta spacciati per appaiati. `metrics.csv` contiene MAE, RMSE, MAPE, R², Pearson, bias, n ed esclusioni MAPE; nel run completo anche guadagno rispetto al Dummy. `seed_summary.csv` riporta media/deviazione per stage e split. I grafici mostrano target/predizione. Non si garantisce un miglioramento: può emergere parità o peggioramento.

Il primo confronto appaiato è **V2 frozen + stessa testa vs V2 adattato + stessa testa**; il confronto con Ridge/HistGB/XGBoost misura utilità pratica. Gli indicatori sono prodotti modellati, non misure indipendenti di biodiversità in campo.

Per MSA storico V1, il pooling era la media dei pixel validi nella cella nativa; qui è un campionamento stratificato 25 punti. Riportare questa differenza: non attribuire automaticamente ogni scarto a V1/V2. BII storico usava già 5×5, ma copertura e preprocessing possono cambiare. Un confronto V1 rigorosamente appaiato richiede ricalcolare V1 sulle stesse posizioni eleggibili: questo repository fornisce il protocollo e gli split, **non include un nuovo encoder V1**. I vecchi risultati non vengono presentati come equivalenti al nuovo supporto.

## 14. Estendere MSA alle 17 AOI di sviluppo

`configs/msa_17areas.yaml` include tutte le celle geometriche delle 17 aree storiche più Liguria. È una campagna grande, da avviare dopo il pilot e su SSD. Usa un nuovo split fisso: 13 AOI training, quattro validation (Vercelli, Chianti, Sila, Sardegna interna), Liguria test già osservato. Selezione macro-AOI, non replica della precedente CV a cinque fold. Campania ed Emilia conservano soltanto le celle storiche training; i loro vecchi test e buffer sono esclusi dall’addestramento.

```powershell
$LargeRoot = 'E:\BioMAP_17areas'
& $Python scripts\probe_msa.py prepare --config configs\msa_17areas.yaml --data-root $LargeRoot --limit 12
& $Python scripts\probe_msa.py run --config configs\msa_17areas.yaml --data-root $LargeRoot --smoke
& $Python scripts\probe_msa.py prepare --config configs\msa_17areas.yaml --data-root $LargeRoot
& $Python scripts\probe_msa.py run --config configs\msa_17areas.yaml --data-root $LargeRoot
& $Python scripts\finetune.py run --indicator msa --config configs\msa_17areas.yaml --data-root $LargeRoot
```

Il profilo aumenta il limite a 250 GiB ma **non promette** che tutta la campagna entri: spazio e costo vanno misurati sul pilot. Se manca spazio il comando si ferma conservando dati riutilizzabili. Umbria nord non viene campionata.

## 15. Collaboratori e manutenzione

Sul repository GitHub: **Settings → Collaborators → Add people**, poi inserire gli username dei due colleghi e inviare gli inviti. Pubblico significa leggibile da tutti; la scrittura richiede comunque invito accettato. Non usare GitHub per conservare chiavi o dati soggetti a condizioni separate.

```powershell
git switch -c experiment/bii-learning-rate
# Modificare configurazione o codice, poi verificare:
& $Python -m pytest
git add configs src scripts tests README.md
git commit -m "Configure BII fine-tuning experiment"
git push -u origin experiment/bii-learning-rate
```

Aprire una pull request. La CI verifica Windows/Linux CPU; non esegue download della campagna né sostituisce il test CUDA. Salvare i run localmente/SSD e condividere metriche e manifest con licenze adeguate.

Riferimenti e licenze: [ATTRIBUTIONS.md](ATTRIBUTIONS.md). Stato dei controlli: [VALIDATION.md](VALIDATION.md).
