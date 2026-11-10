# Avanzamento BIOMAP-FM

Aggiornato al 10 novembre 2026. Milestone corrente: **M1a–M2, probing denso ligure**.

## Completato

- Audit V1: media densa di 128 componenti, conteggi, scale finite e griglia GLOBIO nativa.
- Identificata la differenza fra `trunc` storico e correzione `floor` del codice attuale.
- Verificato il supporto fisico di 3.545.951 pixel V1/V2, non solo i conteggi per cella.
- Ricalcolate 4.991 righe V1: valori, identificativi e conteggi coincidono col parquet storico.
- Riaggregate 4.991 righe V2: differenza massima medie 6,94×10⁻¹⁸, conteggi concordi.
- Creato pacchetto portabile di 11.904.407 byte, con fold congelati e campione encoder.
- Verificato il Medium CPU su 32 pixel: max |Δ| 2,15×10⁻⁶, tolleranza 10⁻⁴.
- Integrati Dummy/Ridge/HistGB/XGBoost/MLP, report dinamico e protezione dei risultati.
- Codice per download del pacchetto, verifica hash e run senza percorsi Mac.
- Release pubblicata e download effettivo con `fetch-bundle` verificato, inclusi estrazione e hash.
- Vecchi README e piano E2/E4/E5 spostati in archivio; nuova roadmap frozen attiva.

## Esecuzione successiva dell'utente

1. Salvare modifiche universitarie locali e aggiornare main fuori da un run attivo.
2. Eseguire test, `fetch-bundle`, `verify-bundle` e `all --bundle` del README.
3. Leggere metriche, delta, fold, grafici ed Excel del nuovo run.

Il nuovo benchmark a cinque regressori **non è stato avviato sul PC universitario** da
questa sessione. Le fixture e le verifiche di equivalenza non sono risultati scientifici
del nuovo probing. Il report storico resta in `reports/REPORT_V1_V2_DENSE_MSA.md`.

## Problemi aperti

- Campagna densa 17 AOI ancora incompleta: Liguria 10/10 tile; Abruzzo 14/16; altre 16 AOI
  senza tile codificate complete. Non confondere inventario delle geometrie con serie acquisite.
- Recupero sistematico degli MLP storici V1 ancora da completare.
- Nessuna misurazione di tempo/VRAM del nuovo probing sul PC universitario.
- Il benchmark locale non misura trasferimento geografico su una nuova AOI.

## Decisioni tecniche

- Nessun fine-tuning; E4/LoRA sospesi dalla roadmap del 10 novembre.
- Medie dense reali, senza interpolare i 25 punti per simulare dati mancanti.
- Pacchetto piccolo in GitHub Release, codice/configurazioni in Git; pesi separati.
- Stessa rappresentazione, stesse celle, fold e iperparametri fra V1 e V2.
- E2 spaziale non viene adattata silenziosamente ai vettori medi.
- CSV/JSON fonte di verità, Excel ricostruibile, run storici preservati.
- Il recupero di un errore finale Excel/report ricostruisce gli output dai CSV e aggiorna
  lo stato, senza ripetere i regressori.

## Evidenze e prossime milestone

[Audit tecnico e modifiche](README_MODIFICHE_DENSE.md),
[piano operativo](PIANO_OPERATIVO.md),
[equivalenza reale](docs/ml/M1_DENSE_EQUIVALENCE_REPORT.md).

M1b e benchmark multi-AOI richiedono prima input densi completi e un pilot computazionale
misurato. M3/M4 seguono i risultati del probing; M5 CNN densa e M6 generalizzazione sono
successivi. Non avviare contemporaneamente tutte le milestone.
