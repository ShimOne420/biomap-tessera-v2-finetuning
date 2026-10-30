# Primo run MSA: V2 frozen contro il riferimento V1 Liguria

Questa guida usa il PC universitario Windows e i dati nella cartella sorella `..\data`. Le modifiche si ricevono con `git pull`; i run si eseguono sul PC con RTX 5080. Non servono file sull'SSD del Mac.

## 1. Quale esperimento stiamo riproducendo

Sono stati verificati report, metriche, membership e codice del progetto V1:

- Training/selezione: `20260928T111520077028Z`.
- Test geografico Liguria: `20260928T133950415054Z`.
- Encoder V1 frozen, regressore **HistGradientBoosting**, 128 medie, campionamento naturale.
- **127.909 celle di sviluppo in 17 AOI**, poi refit su tutte queste celle.
- **4.991 celle di test Liguria Levante**, escluse da fit e selezione.
- **Nessuna validation fissa**: tre fold geografici GroupKFold, con ogni AOI interamente in un fold.

| Fold storico | Training nel fold | Validation nel fold |
|---|---:|---:|
| 1 | 89.101 | 38.808 |
| 2 | 83.014 | 44.895 |
| 3 | 83.703 | 44.206 |

Le validation OOF sommano a 127.909 celle, viste una sola volta fuori dal proprio fit. Non sono celle aggiuntive al training finale. Le vecchie celle test/buffer Campania–Emilia restano fuori dallo sviluppo.

Il profilo nuovo è **`configs/msa_v1_liguria.yaml`**, non `msa.yaml` (che rimane il pilot 1.000/122/245), né il precedente profilo 13/4 AOI. Il manifest nuovo contiene gli stessi identificativi del training V1 e i fold letti direttamente dagli artefatti OOF storici, non rigenerati casualmente.

Riferimento V1 su tutte le 4.991 celle liguri: **MAE 0,12206058; RMSE 0,15807774; MAPE 26,451133%; R² 0,05660319**. È il benchmark geografico discusso nei report, non una dichiarazione del minimo errore fra test su territori diversi.

## 2. Cosa resta diverso e come leggere il risultato

Il nuovo run mantiene celle, anno 2020, target `MSA-area / cell-area`, fold, regressore e iperparametri HistGB del candidato V1 selezionato: 90 iterazioni massime, 15 foglie, learning rate 0,05, L2=1, early stopping interno al solo training, seed 20260925. Non ripete la vecchia ricerca di 19 candidati: valuta il candidato scelto V1 e il Dummy, con selezione solo sullo sviluppo geografico. Fit e predict usano float32 come nel codice V1.

**V1 usava medie dense di pixel; V2 usa 25 punti stratificati per cella.** Il confronto numerico è utile sullo stesso split, ma non dimostra da solo che il solo encoder V2 sia superiore. Anche il preprocessing spaziale non è una replica integrale del vecchio mosaicking. Per attribuire un vantaggio esclusivamente all'encoder occorre un successivo ricalcolo V1/V2 con identico supporto di pixel. Liguria è già stata osservata: non è un nuovo test cieco. Umbria nord resta esclusa.

Questa è una campagna **di probing**, senza aggiornare i pesi TESSERA. Non usare questo YAML con `finetune.py`.

## 3. Ricevere le modifiche sul PC universitario

Aprire PowerShell:

```powershell
Set-Location 'C:\Users\brome\Desktop\BIOMAP_TESSERA_V2\biomap-tessera-v2-finetuning'
git status --short
git pull --ff-only origin main
$Python = Join-Path $PWD '.venv\Scripts\python.exe'
$DataRoot = (Resolve-Path '..\data').Path
$Cfg = 'configs\msa_v1_liguria.yaml'
```

Se `git status` segnala modifiche locali, non cancellarle per forzare il pull. Se il percorso esterno non esiste ancora, crearlo con `New-Item -ItemType Directory -Force '..\data'`, poi ridefinire `$DataRoot`.

Le variabili vanno ridefinite se si apre una nuova finestra PowerShell. Eseguire un comando per volta e fermarsi se termina con errore.

```powershell
& $Python -m pip check
& $Python -m py_compile src\biomap_v2\data.py src\biomap_v2\target_data.py src\biomap_v2\benchmark.py
& $Python scripts\probe_msa.py doctor --config $Cfg --data-root $DataRoot
```

Il doctor deve vedere CUDA e la RTX 5080. Se `.venv` manca, usare il setup già documentato:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\setup_windows.ps1 -DataRoot $DataRoot
```

Non reinstallare l'ambiente se quello già usato per lo smoke funziona. Nessuna nuova dipendenza è necessaria per questa modifica.

## 4. Cartelle: smoke precedente e nuovo benchmark separati

```text
BIOMAP_TESSERA_V2/
  biomap-tessera-v2-finetuning/   codice GitHub
  data/
    weights/student_medium.pt   pesi condivisi, verificati e riusati
    msa/                       smoke precedente, lasciato intatto
    msa_v1_liguria/             NUOVO profilo e run
      targets/
      catalogs/
      series/
      diagnostics/
      runs/
```

Il namespace è configurabile e non è hardcoded nel codice. Tutti i comandi della campagna devono usare **lo stesso `$Cfg` e `$DataRoot`**. Non spostare la vecchia cache `msa` nella nuova cartella: il protocollo/membership devono essere verificati.

## 5. Scaricare personalmente pesi e target

```powershell
& $Python scripts\probe_msa.py download-weights --config $Cfg --data-root $DataRoot
& $Python scripts\probe_msa.py download-targets --config $Cfg --data-root $DataRoot
& $Python scripts\probe_msa.py diagnose-targets --config $Cfg --data-root $DataRoot
```

Il primo comando riusa `data\weights\student_medium.pt` se presente e corretto. Il secondo acquisisce i raster GLOBIO per **tutte** le celle del profilo, prima del download Sentinel. Il terzo produce una diagnosi distinta per numeratore/denominatore: copertura, CRS/griglia, NoData, valori non finiti, area zero/negativa e ratio fuori range. Deve risultare **132.900/132.900 target validi**.

I report sono in `data\msa_v1_liguria\diagnostics\TIMESTAMP_targets\summary.json`, `cells.csv` e `by_partition_aoi.csv`. Se ci sono target invalidi, il comando fallisce senza imputare zeri, fare clipping o modificare gli split. Leggere quei file prima di acquisire le serie annuali.

## 6. Problema precedente dei 444 target mancanti

Il vecchio `prepare --limit 12` scaricava un ritaglio GLOBIO limitato e `prepare` completo lo riusava solo perché esisteva. Il rettangolo ricostruito dalle prime 12 celle lascia fuori **441 delle 1.367 celle**. Questo spiega gran parte delle 444 NaN segnalate; le altre tre richiedono i raster effettivi Windows, che non sono accessibili dal Mac. Non sono state arbitrariamente classificate come valide o da escludere.

La correzione verifica la copertura di **entrambi** i raster e acquisisce una nuova coppia se quella presente è troppo piccola. Le acquisizioni precedenti restano al loro posto; i nuovi export sono sotto `targets\acquisitions\TIMESTAMP`. La vecchia provenienza è salvata in `source_history` quando il puntatore viene aggiornato.

Per diagnosticare anche il vecchio pilot, separatamente dal nuovo run:

```powershell
& $Python scripts\probe_msa.py diagnose-targets --config configs\msa.yaml --data-root $DataRoot
```

È previsto che questo comando segnali il vecchio problema. Per recuperare quel pilot senza cancellare i suoi raster:

```powershell
& $Python scripts\probe_msa.py download-targets --config configs\msa.yaml --data-root $DataRoot
```

Non serve completare il pilot 1.367 celle per iniziare il profilo Liguria, che usa una cartella separata.

## 7. Preparare e provare 12 celle prima della campagna grande

```powershell
& $Python scripts\probe_msa.py prepare --config $Cfg --data-root $DataRoot --limit 12
& $Python scripts\probe_msa.py run --config $Cfg --data-root $DataRoot --smoke
```

Si scaricano serie Sentinel-1/2 annuali su 12 celle: sei di sviluppo, distribuite sui tre fold, e sei liguri. I target sono già completi dal passo 5. La metrica dello smoke **non si confronta** con il 26,45% sulle 4.991 celle V1.

Per i cataloghi Sentinel vengono condivisi gli item per AOI; le osservazioni mancanti restano distinte da quelle valide. Le letture COG supportano `.tif` e `.tiff`, tre tentativi controllati e non espongono SAS URL negli errori.

Se Sentinel-1 restituisce 401/403 può essere necessaria la chiave Planetary Computer già descritta nel README principale. Impostarla solo nella sessione, mai nei file Git:

```powershell
$env:PC_SDK_SUBSCRIPTION_KEY = [System.Net.NetworkCredential]::new('', (Read-Host 'Chiave Planetary Computer' -AsSecureString)).Password
```

Poi rieseguire lo stesso `prepare`.

## 8. Preparare tutti i dati e lanciare il probing completo

```powershell
& $Python scripts\probe_msa.py prepare --config $Cfg --data-root $DataRoot
& $Python scripts\probe_msa.py run --config $Cfg --data-root $DataRoot
```

La preparazione completa deve produrre **127.909 train e 4.991 test**. Se la copertura Sentinel non basta, vengono salvati conteggi e cause in `diagnostics`, ma il dataset ridotto **non viene accettato come replica equivalente**. I dati validi già acquisiti restano disponibili per la ripresa. Nessuna sostituzione casuale di celle.

Questa è una campagna molto più grande del pilot 1.000 celle. Il download può essere lungo; non è stato misurato sul PC universitario. Il limite resta 50 GiB sull'intera data-root, con almeno 20 GiB di riserva. Non è garantito che tutta la campagna entri prima di misurarne il costo. Se manca spazio, il comando si ferma senza cancellare dati. Aumentare il limite solo dopo aver controllato lo spazio reale, tramite `--set storage.max_gib=...`.

Per riprendere dopo interruzione, rieseguire lo **stesso comando**. Le serie riusano i giorni/celle verificati; l'estrazione degli embedding riparte dalle celle salvate nel database SQLite del checkpoint, senza ricalcolare tutte quelle precedenti. I run finali hanno sempre una nuova cartella e non sovrascrivono lo smoke.

## 9. Leggere e verificare il confronto

Il comando stampa la cartella del run sotto `data\msa_v1_liguria\runs`. Per ricontrollare i risultati:

```powershell
$Run = 'INCOLLARE_QUI_IL_PERCORSO_RUN_STAMPATO'
& $Python scripts\probe_msa.py evaluate --config $Cfg --data-root $DataRoot --run-dir $Run
```

Artefatti principali:

- `protocol.json`: conteggi effettivi, hash del manifest e ruolo della validation.
- `folds.json`: n training/validation e AOI per ciascun fold.
- `predictions_cv.csv`: predizioni OOF sulle 127.909 celle di sviluppo.
- `cv_metrics_by_aoi.csv` e `cv_macro_metrics.csv`: metriche e criterio di selezione, senza test.
- `selection.json`: candidato scelto sullo sviluppo e checkpoint frozen utilizzato.
- `predictions.csv` e `metrics.csv`: predizioni/metriche del refit, distinguendo train e test.
- `historical_reference.csv`: riferimento **Liguria V1 corretto**, non BII o MSA1000.
- `comparison_v1_v2.csv`: valori V1/V2 e delta sulle stesse numerosità; nello smoke i delta sono vuoti.
- `report.md`: lettura del protocollo e limiti del confronto.

Delta negativo in MAE/RMSE/MAPE significa errore numericamente più basso nel nuovo run. Non equivale da solo a una superiorità causale del solo encoder, dati il diverso pooling e il test già osservato. Non usare Liguria per cambiare iperparametri dopo averne visto il risultato.
