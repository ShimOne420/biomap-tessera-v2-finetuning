# Runbook Windows: dati, pesi e avvio Spatial V2

Il [README principale](../../README.md) contiene l'ordine operativo completo. Questo documento risolve il caso in cui il run Dense attuale abbia salvato soltanto le medie. Non accedere alla GPU della campagna o aggiornare il suo checkout durante il run; i comandi sono destinati all'esecuzione personale successiva. Il branch pubblicato è `feature/spatial-cnn-v2`, senza merge automatico su main.

## Verificare le due root

La reference contiene `reference.json`, `v1.parquet`, e per ogni AOI `lookup.npy` e le cartelle `grid_*` con `geometry.npz/json`. Contiene anche i target GLOBIO4 congelati; non occorre riscaricarli o rigenerare i fold per il nuovo trainer.

La work-root V2 completata contiene `extraction_complete.json`, `v2.parquet`, `chunks/<aoi>/<tile>/tile.npz/json` e, per la CNN, `pixel_embeddings/<aoi>/<tile>.npy/json`. Non cercare questi dati nel repository Git: sono artefatti esterni.

```powershell
Test-Path "$Reference\reference.json"
Test-Path "$V2Work\extraction_complete.json"
Get-Content "$V2Work\extraction_complete.json"
Test-Path "$V2Work\pixel_embeddings"
& $SpatialPython scripts\spatial_cnn.py doctor @SpatialArgs
```

Doctor elenca quante tile per pixel mancano e fino a 20 esempi. Anche un file presente può essere parziale o corrotto: soltanto prepare/preflight verificano gli SHA, i marker e l'ordine. Non usare `Get-ChildItem -Recurse` sui temporanei della campagna; il nuovo monitor legge soltanto i suoi JSON di avanzamento.

## Se mancano gli embedding per pixel

Non è un errore correggibile con reshape o broadcasting. La media 128D è una trasformazione che ha eliminato la disposizione spaziale. La r1 non mantiene necessariamente i raw Sentinel dopo la tile. Non modificare né cancellare la cache completata.

Inoltre il ramo r1 riusa subito `tile.npz` completato: rilanciare la vecchia root con il nuovo flag non riapre quelle tile. Occorre una nuova estrazione esplicita **con una work-root nuova**, sulla reference già congelata. Questa attività può richiedere di riacquisire le serie annuali ed è distinta dal training della CNN. Il codice preparato qui non la avvia automaticamente.

### Checkpoint, solo se serve una nuova estrazione

Se si possiedono già tutti i pixel, la CNN non ha bisogno di scaricare o caricare pesi GeoTESSERA. Se serve una nuova estrazione, usare il checkpoint Medium locale già verificato, cambiando questa stringa al percorso effettivo:

```powershell
$Checkpoint = 'C:\BIOMAP\weights\ckpt\student_medium.pt'
& $SpatialPython -c "import sys; from biomap_v2.common import digest,WEIGHTS_SHA256; assert digest(sys.argv[1]) == WEIGHTS_SHA256, 'Checkpoint SHA256 discordante'; print('Checkpoint verificato')" $Checkpoint
if ($LASTEXITCODE -ne 0) { throw 'Pesi mancanti/non validi; non avviare estrazione' }
```

Se il file non è disponibile, questo è un **download esplicito personale**, con revisione fissata dal repository:

```powershell
& $SpatialPython -c "from huggingface_hub import hf_hub_download; from biomap_v2.common import WEIGHTS_REVISION,digest,WEIGHTS_SHA256; p=hf_hub_download('geotessera/TESSERA-V-2.0-2B-M','ckpt/student_medium.pt',revision=WEIGHTS_REVISION,local_dir=r'C:\BIOMAP\weights'); assert digest(p)==WEIGHTS_SHA256; print(p)"
if ($LASTEXITCODE -ne 0) { throw 'Download/verifica pesi falliti' }
```

Il download Hugging Face usa la propria cache riprendibile; rilanciare lo stesso comando. Non esiste un download pubblico preconfezionato delle mappe Spatial V2 delle 17 AOI verificato da questo intervento. Il piccolo bundle storico della release contiene medie, non sostituisce i circa 99,5 milioni di embedding per pixel.

### Nuova estrazione, preservando r1

Terminata la campagna attuale, verificare spazio e scegliere una root che non sia quella di produzione. Gli embedding da soli richiedono circa 50,9 GB decimali / 47,5 GiB a 99,5 milioni × 128 × 4 byte, oltre a temporanei, indice e riserva. Non è stata misurata qui la dimensione di picco dei temporanei reali.

```powershell
$PixelWork = 'C:\BIOMAP\v2_spatial_pixels_new'
if (Test-Path $PixelWork) { throw 'Per il primo avvio scegliere una cartella nuova; non usare la cache della campagna precedente' }
Get-PSDrive -PSProvider FileSystem
& $SpatialPython scripts\dense_campaign.py stream --reference $Reference --work-root $PixelWork --checkpoint $Checkpoint --device cuda:0 --chunk-size 4096 --pixel-batch 32 --io-threads 8 --keep-pixel-embeddings
if ($LASTEXITCODE -ne 0) { throw 'Estrazione incompleta; sorgente/vecchia campagna non alterate' }
```

Questo comando chiama l'entry point r1 esistente, senza modificarlo, e conserva le mappe autentiche. Per riprendere si rilancia soltanto il comando `dense_campaign.py stream` con gli stessi argomenti e la stessa nuova root, senza la guardia del primo avvio. La r1 controlla i propri marker. Se un'interruzione avviene tra la pubblicazione della tile aggregata e quella dei pixel, il nuovo doctor/preflight può ancora rilevare pixel mancanti: non eliminare cache per forzare il passaggio, serve una diagnosi separata.

Quando la nuova estrazione è completa:

```powershell
$V2Work = $PixelWork
$SpatialArgs = @('--config','configs\spatial_v2_msa.yaml','--source-root',$V2Work,'--reference',$Reference,'--output-root',$SpatialOut,'--device','cuda:0')
& $SpatialPython scripts\spatial_cnn.py doctor @SpatialArgs
```

Proseguire dai passi prepare/preflight/profile del README. Non servono ulteriori acquisizioni nel trainer. r2-fast rimane sperimentale e separata: il nuovo reader ne verifica lo schema, ma il percorso consigliato sopra conserva il protocollo r1 della campagna corrente.

## Ripresa del training e modifiche di configurazione

Ripresa dall'ultima epoca completata con `train --phase cv --resume` o `train --phase refit --resume`, sempre stesso run-id e stessi override. Un'interruzione a metà epoca ripete quell'epoca; non dichiara conservati minibatch non checkpointati.

Cambiare feature-space, protocollo, seed, batch, precisione, modello o codice richiede una nuova identità/run; il checkpoint non viene riadattato silenziosamente. Per una prova di ripresa è disponibile `--max-epochs-this-invocation 1`: limita il lavoro della singola invocazione senza cambiare il budget congelato, registra `interrupted` e consente poi `--resume`. L'exit code può essere zero perché si tratta di una pausa tecnica richiesta: leggere `manifest.json`, non assumere CV completa. Refit legge `selection.json`, che esiste soltanto a CV conclusa.

Le impostazioni si modificano da YAML oppure, per esempio, `--set training.learning_rate=0.0005`, `--set model.projection_channels=null`, `--normalization none`. Gli argomenti CLI prevalgono. `cell_standard` è definita sulle medie: con CNN viene rifiutata. Per l'ablation usare la stessa strategia di normalizzazione in tutti i run.

## macOS e stato di prontezza

Sul Mac usare `.venv-dev` e [setup CPU](../development/MACOS_SETUP.md). Tutte le nuove prove usano fixture minuscole e non accedono al PC. Non occorre installare CUDA sul Mac. MPS, se richiesto esplicitamente con `--device mps`, resta una verifica tecnica separata sul predittore; la suite CI usa CPU. Il comportamento reale su RTX 5080 e il batch sostenibile restano verifiche dell'utente tramite profile, non risultati dichiarati da questo documento.
