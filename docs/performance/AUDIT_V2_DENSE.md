# Analisi delle prestazioni della campagna V2 densa (r1)

Data: 12 novembre 2026. La campagna `dense127909-stream-legacy-exact-r1` è in corso in `F:\biomap_dense_public\work`. Tutte le misure qui sotto sono state prese **in sola lettura**: i file sono stati aperti in condivisione e il processo non è stato toccato.

Legenda della colonna "Tipo":
- **M** = misurato;
- **S** = misurato su un benchmark simulato locale (vedi `BENCHMARK.md`);
- **E** = stimato.

## Numeri chiave
Fonte: `stream_log.jsonl`, 61 tile acquisite da zero più 2 riprese dalla cache. Script: `scripts/audit_dense_log.py`. Output: `data/audit_r1_20261112.json`.

| Grandezza | Valore | Tipo |
|---|---|---|
| Tempo totale delle 61 tile | 9,05 h | M |
| Quota di tempo in acquisizione, con GPU ferma | **79,5%** | M |
| Codifica GPU (FP32, batch 256, tile > 100.000 pixel) | 0,327 ms/pixel | M |
| Costo fisso per tile (regressione tempo = a + b·pixel) | **232 s** | M (stima di regressione) |
| Costo variabile | 0,90 ms/pixel | M (stima di regressione) |
| Lettura remota SCL/S1 per blocco (tempo sommato sui thread) | 1,26 s | M |
| Asset richiesti per tile (solo SCL/S1) | 651 in media | M |
| Errori di lettura remota in tutta la campagna | 1 (1 ritentativo) | M |
| Collegamento Ethernet negoziato | **100 Mbit/s** (scheda Realtek da 2,5 GbE, negoziazione automatica) | M |
| Traffico in ingresso durante l'acquisizione (campione di 15 s) | 56 Mbit/s | M |
| RAM del processo di produzione | 13,4 GB su 31 GB | M |
| Tempo dichiarato `coordinate_transform` (tile grandi, sommato sui thread) | 868 s | M |
| Costo reale di `rowcol` su 2 milioni di punti | 0,5 s (r1) contro 0,12 s (r2) | M |

### Prestazioni per fascia di dimensione della tile
| Fascia | Tile | Pixel medi | Acquisizione media | Codifica media | Pixel per secondo | GPU ferma |
|---|---|---|---|---|---|---|
| meno di 10.000 pixel | 14 | 3.991 | 212 s | 1,9 s | **19** | 99% |
| 10.000–100.000 | 21 | 26.132 | 247 s | 8,1 s | 102 | 97% |
| 100.000–600.000 | 10 | 490.063 | 548 s | 198 s | 657 | 73% |
| 600.000 e oltre | 16 | 929.467 | 767 s | 281 s | 887 | 73% |

Su tutta la campagna, 125 tile su 273 hanno meno di 100.000 pixel. Insieme contengono 2,2 milioni di pixel, cioè il 2,3% del totale (M).

## Colli di bottiglia in ordine di peso
1. **Acquisizione e codifica in serie, una tile alla volta** (`streaming.py`, `stream` e `process_tile`). La GPU resta ferma il 79,5% del tempo (M). Sovrapporre le due fasi recupera al massimo il tempo di codifica, circa il 20% del totale. Nel benchmark simulato, con la sola sovrapposizione: da 1,3 a 1,7 volte (S).
2. **Costo fisso per tile di circa 232 s**, indipendente dal numero di pixel (M). Ogni tile rilegge tutte le scene dell'anno: circa 140 giorni S2 e circa 190 giorni S1, cioè 650 asset SCL/S1 più le bande S2. Le tile sotto i 10.000 pixel vanno a 19 pixel al secondo. Si stimano 7–8 ore della campagna per il 2,3% dei pixel (E).
3. **Gli stessi byte vengono riscaricati da tile vicine** (cache RAM dei blocchi spenta, `memory_cache_mib: 0`).
   - Ogni banda S2 di ogni scena viene riaperta da zero per ogni tile (`reproject_band`), e il suo traffico non compare nei contatori del lettore.
   - Nel benchmark simulato la r1 scarica **3,6 volte** i byte necessari con tile piccole e 1,6 volte con tile grandi (S).
   - Con un collegamento da 100 Mbit/s, ogni byte risparmiato è tempo risparmiato.
4. **Banda di rete limitata a 100 Mbit/s dal collegamento fisico** (M). La scheda supporta 2,5 Gbit/s. Il limite viene dal cavo, dalla presa o dallo switch, non dal software. Nel benchmark simulato la r1 passa da 103 s a 55 s solo passando da 100 Mbit/s a 1 Gbit/s, cioè **1,86 volte** (S).
5. **Parallelismo a fasi.** S2, S1 ascendente e S1 discendente vengono elaborati uno dopo l'altro; alla fine di ogni fase il pool aspetta il giorno più lento. Dentro un giorno scene e bande vanno in sequenza.
6. **Contesa sul GIL di Python, non calcolo vero.** I 600–870 s di `coordinate_transform` sono tempo trascorso dai thread ad aspettarsi: `rowcol` costa realmente 0,5 s per 2 milioni di punti (M). La versione r2 lo porta a 0,12 s.
7. **Codifica FP32 a 0,327 ms/pixel**, circa 7,3 h per la parte rimanente (E). Il protocollo vieta l'AMP. Batch più grandi cambierebbero l'equivalenza bit per bit: non toccati.
8. **Fragilità operativa.**
   - `common.save_json` non ritenta `os.replace`. Qualsiasi lettura concorrente di `stream_progress.json` fa uscire il run, anche con condivisione Delete (verificato il 12/11). Arresti avvenuti l'11/11 alle 14:06 e alle 20:09 e il 12/11 alle 12:35 (M).
   - **Il fermo di circa 14,5 h dell'11/11 è dovuto a Windows Update** (M, registro eventi): riavvio forzato alle 20:30 per KB5132061, tre riavvii tra 20:30 e 20:33, subito dopo la fine delle ore attive (8–20). Lo script di ripresa non sopravvive a un riavvio.
   - Lo script di ripresa `run_v2_dense_stream.ps1` (processo 16748) è attivo e rilancia il run dopo un'uscita. Il 12/11 alle 12:37 l'ha fatto in 60 s.

## Cosa non è stato misurato
- **Tempo e byte delle sole bande S2 nella campagna reale:** non passano dal profilo del lettore. Si possono ricavare solo dal benchmark simulato o da una prova reale da approvare.
- **Throttling di Planetary Computer con più richieste in parallelo:** non verificabile senza traffico reale aggiuntivo.
