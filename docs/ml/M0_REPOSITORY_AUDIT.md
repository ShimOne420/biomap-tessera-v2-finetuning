# M0 — Audit del percorso denso

Base condivisa verificata: `2ae5ea9f49b0047e0c16f2b9caa3cfbb448e68af` (main).
Sono state lette le note universitarie del 10 novembre: i cinque regressori e i report
dinamici descritti lì non erano pubblicati. Questa consegna li integra nella base remota,
senza fingere che i file non committati del PC siano stati recuperati via Git.

## Pipeline distinte

| Pipeline | Supporto | Stato |
|---|---|---|
| V1 storico | Tutti i pixel validi, medie 128D | Parquet e codice recuperati e verificati |
| V2 operativo precedente | 25 punti stratificati per cella | Conservato come storico, non usato per il nuovo denso |
| E2 CNN | Reticolo reale 5×5 dei punti, feature+maschera | Non applicabile direttamente al vettore medio |
| V2 denso corrente | Stessi pixel densi V1, medie 128D | Liguria completa e verificata |

## Input e controlli

Target MSA GLOBIO overall 2020, EPSG:4326, 10 arcsec, valori unitari già ottenuti come
MSA-area/cell-area. Il pacchetto conserva il target normalizzato: non si divide di nuovo.
V1 è int8 dequantizzato con scale per pixel; V2 è FP32 Medium. CRS e trasformazioni sono
per tile. Scale non finite vengono escluse, NoData non viene trasformato in MSA zero.

La replica dell'artefatto storico usa `trunc`. La correzione `floor` presente nel codice
V1 attuale non viene applicata a un solo encoder. `support.py` controlla gli stessi punti
e celle, non soltanto i conteggi; `paired.py` rifiuta duplicati, target e coperture diversi.

## Training e valutazione

`scripts/dense_msa.py` è l'entrypoint unico. La nuova valutazione usa medie dense, cinque
fold spaziali liguri congelati e regressori identici fra encoder. Scaler e Dummy sono
stimati solo nel training pertinente. MLP e HistGB, se configurati, usano early stopping
interno al training del fold. Non si ottimizzano iperparametri sui risultati OOF.

Il nuovo run è interno alla Liguria. Il trasferimento 17 AOI→Liguria e i relativi tre
fold non vengono simulati con questo campione. La campagna densa di sviluppo è ancora aperta.

## Conservazione e documentazione

Nessuna riscrittura di dati, modelli o report storici. Report nuovi nella run; output
esistente rifiutato; CSV/JSON fonte di verità; Excel ricostruibile. I vecchi README sono
archiviati, non confusi con il workflow attivo. Il nuovo piano mantiene gli encoder frozen.
