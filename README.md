# BIOMAP — Spatial CNN su GeoTESSERA V2

Il percorso prioritario è ora **embedding autentici per pixel → Spatial CNN 2D mascherata → una previsione GLOBIO4 MSA per cella**. Il benchmark usa **127.909 celle di sviluppo, 17 AOI, tre fold geografici originali e 4.991 celle Liguria esterne**: 132.900 celle totali. Liguria è già osservata e rimane esclusa dalla selezione del modello.

Il vecchio esperimento A su medie Dense/Conv1D non è più un passo da eseguire nella strategia principale. Il suo codice e i test restano disponibili per riproducibilità, insieme a B V1, r1 e r2-fast. Il controllo Mean MLP del nuovo esperimento è opzionale e usa le medie degli **stessi pixel**, non una mappa inventata dai vettori del vecchio parquet.

**Condizione indispensabile:** la work-root V2 deve contenere `pixel_embeddings/<aoi>/<tile>.npy` e sidecar validi. Il solo `v2.parquet` non basta. La r1 conserva questi file soltanto con `--keep-pixel-embeddings`, disabilitato di default. Non sappiamo se il run universitario attuale li abbia conservati: il primo comando dati lo controlla. Se mancano, [seguire il percorso dedicato](docs/framework/windows_runbook.md#se-mancano-gli-embedding-per-pixel); non è possibile promettere il training immediato al termine dell'estrazione delle sole medie.

## 1. Codice separato e ambiente Windows

I comandi seguenti si eseguono personalmente sul PC, dopo la conclusione del run attuale per le attività GPU. Nessun processo universitario viene controllato da questo repository. Usare un clone e una venv nuovi preserva checkout e ambiente della campagna.

Se Git/Python mancano, installarli tramite i canali consentiti dall'università; per un PC su cui winget è autorizzato:

```powershell
winget install --id Git.Git -e
winget install --id Python.Python.3.12 -e
```

Aprire un nuovo PowerShell dopo l'installazione. Finché la PR non è stata integrata, il codice è sul branch pubblicato:

```powershell
New-Item -ItemType Directory -Force 'C:\BIOMAP' | Out-Null
git clone --branch feature/spatial-cnn-v2 https://github.com/ShimOne420/biomap-tessera-v2-finetuning.git 'C:\BIOMAP\spatial-code'
Set-Location 'C:\BIOMAP\spatial-code'
py -3.12 -m venv .venv-spatial
$SpatialPython = (Resolve-Path '.\.venv-spatial\Scripts\python.exe').Path
& $SpatialPython -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128
if ($LASTEXITCODE -ne 0) { throw 'Installazione Torch fallita' }
& $SpatialPython -m pip install -r requirements-lock.txt
if ($LASTEXITCODE -ne 0) { throw 'Installazione dipendenze fallita' }
& $SpatialPython -m pip install -e . --no-deps --no-build-isolation
if ($LASTEXITCODE -ne 0) { throw 'Installazione progetto fallita' }
& $SpatialPython -m pip check
if ($LASTEXITCODE -ne 0) { throw 'Dipendenze incoerenti' }
```

Non aggiornare la `.venv` della campagna. Non occorre attivare la nuova venv. Il lock scientifico è invariato. Il canale cu128 per Torch 2.10.0 è documentato nelle [istruzioni ufficiali PyTorch](https://pytorch.org/get-started/previous-versions/#v2100); il test sul driver/GPU effettivi rimane il passo profile. Per aggiornare successivamente **questo clone pulito**, non quello sotto un run:

```powershell
Set-Location 'C:\BIOMAP\spatial-code'
if (git status --porcelain) { throw 'Salvare prima le modifiche locali; nessun reset o clean' }
git pull --ff-only origin feature/spatial-cnn-v2
if ($LASTEXITCODE -ne 0) { throw 'Pull non riuscito: non forzare' }
```

## 2. Test CPU e percorsi reali

```powershell
& $SpatialPython scripts\check_dev_environment.py
if ($LASTEXITCODE -ne 0) { throw 'Import/ambiente non validi' }
& $SpatialPython -m pytest -ra
if ($LASTEXITCODE -ne 0) { throw 'Test falliti: non avviare il run' }

# Cambiare queste due stringhe se i dati completati sono in un altro percorso o SSD.
$V2Work = 'C:\BIOMAP\v2_dense_work'
$Reference = 'C:\BIOMAP\reference'
$SpatialOut = 'C:\BIOMAP\spatial_runs'
$SpatialArgs = @('--config','configs\spatial_v2_msa.yaml','--source-root',$V2Work,'--reference',$Reference,'--output-root',$SpatialOut,'--device','cuda:0')
```

`$V2Work` è la work-root esistente con `extraction_complete.json`; `$Reference` contiene `reference.json`. `$SpatialOut` deve essere una cartella nuova e separata, anche dall'albero delle sorgenti. La radice può essere su C: o SSD; non ci sono percorsi Mac obbligatori. L'indice occupa centinaia di MB, non duplica i circa **47,5 GiB** degli embedding per 99,5 milioni di pixel. Riserva minima predefinita: 20 GiB. Il conteggio reale dei pixel deriva dagli artefatti, non da questa stima.

## 3. Controllare la presenza delle mappe

```powershell
& $SpatialPython scripts\spatial_cnn.py doctor @SpatialArgs
if ($LASTEXITCODE -ne 0) { throw 'Pixel/reference mancanti: leggere docs/framework/windows_runbook.md prima di proseguire' }
```

Doctor è un inventario rapido: `PRESENT_NOT_VERIFIED` significa solo file presenti. Non carica encoder, pesi o GPU. Se mancano i pixel, fermarsi qui e seguire il [runbook](docs/framework/windows_runbook.md): aggiungere il flag sulla vecchia cache già completata non li ricostruisce.

## 4. Preparare l'indice riprendibile

```powershell
& $SpatialPython scripts\spatial_cnn.py prepare @SpatialArgs
if ($LASTEXITCODE -ne 0) { throw 'Preparazione incompleta: leggere errore; gli originali restano intatti' }
```

Verifica tutti gli SHA/marker, ordine, griglia, geometrie/lookup e somme/conteggi. Costruisce `$SpatialOut\index\spatial_index.sqlite3` e `cells.parquet` con finestre/mask e momenti per cella. La prima verifica legge sequenzialmente tutti i file pixel: non è istantanea e non usa la GPU. Ogni tile è una transazione; per riprendere si rilancia lo **stesso comando**, senza cancellare l'indice o cambiare config.

## 5. Preflight e prova tecnica GPU

```powershell
& $SpatialPython scripts\spatial_cnn.py preflight @SpatialArgs
if ($LASTEXITCODE -ne 0) { throw 'Preflight non pronto' }
& $SpatialPython scripts\spatial_cnn.py profile @SpatialArgs
if ($LASTEXITCODE -ne 0) { throw 'Batch/GPU non validi: correggere la config prima del training' }
Get-Content "$SpatialOut\profile.json"
```

La GPU viene interrogata solo dopo i controlli dati. `profile` esegue un forward/backward della **sola CNN**, sul batch configurato di quattro celle; salva tempo, parametri e memoria CUDA. Non certifica il batch massimo o la durata completa. Per provare un batch diverso usare, per esempio, `--batch-cells 8 --set training.accumulate_cells=16` in tutte le fasi del relativo run. La verifica reale sulla RTX 5080 resta da eseguire sul PC.

## 6. Spatial CNN V2: CV sulle 17 AOI

```powershell
$RunId = 'v2_spatial3_seed42'
& $SpatialPython scripts\spatial_cnn.py train @SpatialArgs --run-id $RunId --phase cv
if ($LASTEXITCODE -ne 0) { throw 'CV interrotta/fallita: leggere manifest e checkpoint' }
Get-Content "$SpatialOut\$RunId\selection.json"
```

Tre fold originali, scaler stimato solo sul training di ciascun fold, early stopping sulla RMSE macro-AOI della validation. MSE una volta per cella, AdamW, batch 4 con accumulo verso 16, massimo 50 epoche e pazienza 8. Il numero di patch non cambia il peso del target. L'encoder V2 resta congelato e non viene caricato.

## 7. Monitor live in un secondo terminale

Preparazione:

```powershell
Set-Location 'C:\BIOMAP\spatial-code'
& .\.venv-spatial\Scripts\python.exe scripts\spatial_cnn.py monitor --config configs\spatial_v2_msa.yaml --output-root 'C:\BIOMAP\spatial_runs' --watch --interval 5
```

Training (il run-id deve corrispondere):

```powershell
Set-Location 'C:\BIOMAP\spatial-code'
& .\.venv-spatial\Scripts\python.exe scripts\spatial_cnn.py monitor --config configs\spatial_v2_msa.yaml --output-root 'C:\BIOMAP\spatial_runs' --run-id v2_spatial3_seed42 --watch --interval 5
```

Mostra fold, epoca, celle elaborate, velocità, ETA dell'epoca e, dopo la prima epoca, ETA approssimativa del budget della fase. L'early stopping può anticipare il termine; CV e refit sono fasi distinte. Il monitor non apre dati sorgente né mantiene handle sui JSON. Ctrl+C chiude soltanto il monitor.

## 8. Ripresa e refit, poi Liguria separata

Per riprendere una CV interrotta, con gli stessi parametri:

```powershell
& $SpatialPython scripts\spatial_cnn.py train @SpatialArgs --run-id $RunId --phase cv --resume
if ($LASTEXITCODE -ne 0) { throw 'Ripresa CV non completata' }
```

La selezione è già nella CV; il refit usa tutte le 127.909 celle di sviluppo per la mediana delle epoche migliori dei tre fold. La Liguria non determina le epoche.

```powershell
& $SpatialPython scripts\spatial_cnn.py train @SpatialArgs --run-id $RunId --phase refit
if ($LASTEXITCODE -ne 0) { throw 'Refit non completato; usare --resume per riprenderlo' }
& $SpatialPython scripts\spatial_cnn.py evaluate @SpatialArgs --run-id $RunId
if ($LASTEXITCODE -ne 0) { throw 'Valutazione esterna incompleta' }
Get-Content "$SpatialOut\$RunId\report.md"
```

## 9. Ablation opzionale: media, 1×1 e 3×3

Prima di refit/Liguria, si possono eseguire altri due run di CV sugli stessi pixel, senza lanciare il vecchio esperimento Dense/Conv1D:

```powershell
& $SpatialPython scripts\spatial_cnn.py train @SpatialArgs --architecture mean_mlp --run-id v2_mean_seed42 --phase cv
if ($LASTEXITCODE -ne 0) { throw 'CV Mean MLP incompleta' }
& $SpatialPython scripts\spatial_cnn.py train @SpatialArgs --kernel 1 --run-id v2_pixel1_seed42 --phase cv
if ($LASTEXITCODE -ne 0) { throw 'CV pixelwise incompleta' }
& $SpatialPython scripts\spatial_cnn.py compare @SpatialArgs --compare-runs v2_mean_seed42 v2_pixel1_seed42 v2_spatial3_seed42 --comparison-name ablation_seed42
Get-Content "$SpatialOut\comparisons\ablation_seed42\selection.json"
```

Per rifittare il vincitore, mantenere il suo stesso override: `--architecture mean_mlp`, `--kernel 1`, oppure nessun override per 3×3. Il confronto registra capacità e campo recettivo; non attribuisce automaticamente il guadagno alla sola spazialità. Repliche con `--seed 43` / `44` richiedono run-id distinti e confronti separati per seed. La proiezione opzionale è `--projection-channels 64`; è una nuova variante, non un riallineamento semantico degli encoder.

## 10. Output, documentazione e percorsi legacy

Ogni run contiene configurazione effettiva, manifest, checkpoint per fold, `checkpoint.pt` finale, predizioni OOF/esterne separate, metriche pooled/per fold/per AOI/macro-AOI, Dummy, MAPE con denominatori, diagnosi per MSA/patch/copertura/eterogeneità, tempi e memoria CUDA. I file pesanti restano fuori da Git. MAE 0,1 indica 0,1 della scala MSA; il target è modellato, non biodiversità misurata sul campo.

- [Architettura e scelte scientifiche](docs/framework/architecture.md).
- [Report dettagliato di implementazione e test](docs/framework/implementation.md).
- [Runbook Windows: dati mancanti, pesi e acquisizione esplicita](docs/framework/windows_runbook.md).
- [Audit di fattibilità del fine-tuning, non implementato in questa attività](docs/framework/finetuning_feasibility.md).
- [Sviluppo Mac Python 3.12 e CI CPU](docs/development/MACOS_SETUP.md), [regole agenti](AGENTS.md).
- Legacy preservati: [campagna r1](docs/ml/DENSE_127909_PC_RUNBOOK.md), [r2-fast](docs/performance/README.md), [A Dense heads](experiments/v2_dense_cnn_msa/README.md), [B V1 Spatial CNN](experiments/v1_spatial_cnn_msa/README.md).
