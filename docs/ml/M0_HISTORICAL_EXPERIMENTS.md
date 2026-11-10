# M0 — Riferimenti storici e limiti di recupero

## Riferimento V1 verificabile nel repository

`configs/msa_v1_liguria_reference.json` conserva l'esperimento con 127.909 celle di
sviluppo, 17 AOI, tre fold geografici e 4.991 celle nel benchmark Liguria.
HistGB ha MAE 0,12206, RMSE 0,15808 e MAPE 26,4511%. Quel numero appartiene al
trasferimento sviluppo→Liguria, non alla nuova CV interna nella Liguria.

Il report `reports/REPORT_V1_V2_DENSE_MSA.md` conserva il benchmark denso locale precedente
con Dummy/Ridge/HistGB. È un artefatto storico separato: i nuovi run non lo sovrascrivono.

## MLP V1 ed E2

La roadmap dell'utente segnala esperimenti MLP V1 precedenti. In questa fase non è stata
ricostruita per ciascuno la combinazione di architettura, parametri, split e risultato:
questa parte di M0 resta aperta. Non è corretto dire che la nuova MLP 256–128 replichi
un run storico non verificato.

L'MLP attuale è una baseline esplorativa parametrizzata, dichiarata prima del run e
identica per entrambi gli encoder. E2 usa input `[128,5,5]` più maschera, non una media
128D rimodellata. Non viene inclusa nel benchmark tabellare e non è la futura CNN densa.

I risultati V2/25 punti citati nel piano allegato restano attribuiti all'analisi fornita
dall'utente: non sono stati ricalcolati o presentati come nuove misure di questa consegna.
