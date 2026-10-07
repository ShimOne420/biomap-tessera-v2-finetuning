# Verifiche e limiti — 7 ottobre 2026

## Verificato localmente

- Ambiente nuovo, separato dal progetto storico: Python 3.12.14, PyTorch 2.10.0 e dipendenze di `requirements-lock.txt`; `pip check` senza conflitti.
- Test CPU su fixture sintetiche, senza confonderli con dati scientifici: parità normalizzazione/binning/pooling rispetto al codice ufficiale; conversione corretta intensità SAR; gradienti e aggiornamento pesi; controllo frozen immutato; target/griglia e checksum; prepare riprendibile; probing; training progressivo; metriche con target zero; resume identico al run ininterrotto; recupero OOM simulato mantenendo batch effettivo.
- Checkpoint **Medium ufficiale reale**, 21.031.506 parametri, SHA256 verificato contro Hugging Face. Confronto fra forward differenziabile e inferenza ufficiale sugli stessi array sintetici: massimo errore assoluto `3.337860107421875e-06`, output tutti finiti, tolleranza `rtol=2e-5, atol=2e-5`.
- Risoluzione delle wheel Windows x64/Python 3.12 per le dipendenze bloccate, controllata tramite pip. PyTorch cu128 installato sul PC universitario dal setup ufficiale.
- Query live Planetary Computer su S1 RTC e S2 L2A: disponibili gli asset attesi e la paginazione STAC. La descrizione ufficiale S1 RTC conferma intensità gamma, non ampiezza.
- Esportazione dei due split storici senza target/feature: 1.000/122/245 per indicatore. Provenienza e hash in `configs/provenance.json`.

Il Mac richiede OpenMP per XGBoost; nelle verifiche locali è stata usata la libreria già inclusa con PyTorch tramite `DYLD_LIBRARY_PATH`. Non è una configurazione necessaria per Windows. Avvisi di deprecazione di matplotlib/pyparsing/rasterio non sono fallimenti dei test.

## Verifica automatica Windows/Linux

La CI esegue la suite CPU su entrambi i sistemi. Su Windows esegue anche lo **stesso setup PowerShell documentato**, con wheel CPU. Controllare il risultato in GitHub Actions; CPU CI non certifica CUDA o memoria RTX 5080.

## Da verificare in università

1. Driver NVIDIA, esecuzione CUDA sulla RTX 5080 e microbatch sostenibile con i pesi Medium reali.
2. Credenziali/accessibilità degli asset S1 RTC e disponibilità dei target al momento del download.
3. Prova `prepare --limit 12`, probing `--smoke` e fine-tuning `--smoke` su serie annuali reali.
4. Copertura delle celle e costo effettivo di preparazione/training: nessun tempo o spazio totale garantito prima della misura.

Non è stata eseguita una campagna scientifica di fine-tuning né generata una nuova baseline V2 completa. Il codice salva risultati reali solo dopo l'esecuzione richiesta.

## Limiti del confronto

I holdout storici sono già osservati. La campagna sulle 17 aree usa un nuovo split fisso 13 training/4 validation, non la vecchia CV. Umbria nord non è inclusa.

Le letture spaziali puntuali COG usano il pixel nativo più vicino per banda; la parità con l'intera catena ufficiale di mosaicking e resampling **non è dimostrata**. È verificata invece la parità di normalizzazione, binning e forward a parità di array. Il 31 dicembre 2020 è mappato al DOY 365 per rispettare il contratto pubblico V2; politica salvata nel manifest.

Il nuovo MSA usa 25 posizioni per cella, mentre il riferimento V1 storico usava medie dense. Le metriche V1 esportate sono descrittive e non vengono trasformate in confronti causali appaiati. Ricalcolare V1 sulle stesse posizioni richiede un percorso encoder V1 separato.

Il fine-tuning può ridurre oppure aumentare l'errore. Le predizioni emulano prodotti GLOBIO/NHM e non validano biodiversità osservata sul campo.
