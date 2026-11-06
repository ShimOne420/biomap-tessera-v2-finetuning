# Report tecnico — Accelerazione di `prepare` MSA

**Progetto:** BioMap / TESSERA V2, confronto con il riferimento TESSERA V1 Liguria.
**Data:** 6 novembre 2026.
**Versione di partenza:** commit `dc8604a`. Le modifiche descritte qui non sono ancora state committate.
**Profilo interessato:** `configs/msa_v1_liguria.yaml`, 132.900 celle in 18 aree.

## 1. Esito

| Domanda | Risposta | Evidenza |
|---|---|---|
| Il risultato cambia? | No, sul campione verificato | 1.200 celle reali su 4 aree: 0 differenze, file identici byte per byte |
| I dati già scaricati restano validi? | Sì | Formato e identità delle celle invariati; nessuna cella riscaricata |
| Il codice regge i casi limite? | Sì, sui casi coperti | 55 test superati sulla suite completa |
| Quanto è più veloce? | La campagna è in corso e il dato finale manca | Banda locale saturata con 2 processi; vedere sezione 5 |
| L'obiettivo di 5 ore è raggiunto? | **No, non con il collegamento attuale** | La scheda di rete è collegata a 100 Mbps |

**Limite attuale:** la velocità dipende dalla banda. La scheda Realtek 2,5 GbE risulta negoziata a **100 Mbps**: aggiungere processi oltre questo limite non accelera e con 8 processi peggiora.

## 2. Diagnosi di partenza

Misure del run dal 4 al 6 novembre (41,8 ore, 22.240 celle nuove, 532 celle/ora, circa 6,8 s per cella):

| Voce | Tempo misurato | Effetto |
|---|---|---|
| Scrittura e checksum dei file temporanei per giorno e cella | 59.746 s, circa il 40% del tempo totale | Circa 350 coppie di file per cella, scritte una alla volta e poi cancellate |
| Lettura della cache blocchi su disco | 18.504 s, circa il 12% | Eseguita sotto un lock globale, con commit SQLite a ogni lettura |
| Letture di rete | 99.497 s sommati sui worker | Parallelismo solo tra bande della stessa scena |
| Cache in RAM | 0 hit | Da 256 MiB, troppo piccola per il giro di blocchi di un lotto |
| Cache su disco | 237.047 scritture, 237.446 espulsioni | Piena al limite di 8 GiB |

La rete era stabile: 12 errori in 41,8 ore. Lo spazio disco non era un limite.

## 3. Modifiche al comportamento

| # | Modifica | Prima | Ora | Perché |
|---|---|---|---|---|
| 1 | Checkpoint | Una coppia di file per giorno e per cella | Un file per giorno e per lotto in `batch_checkpoints` | Elimina milioni di scritture e conserva la ripresa |
| 2 | Lotti | 16 celle | 8.192 celle, cioè l'area intera | Ogni blocco serve una volta sola e ogni banda si apre una volta per lotto |
| 3 | Cache blocchi | 8 GiB su disco, 256 MiB in RAM | Disattivata | Con un lotto per area i blocchi non si rileggono |
| 4 | Parallelismo | Un giorno alla volta, 4 worker | 4 giorni insieme, 8 worker | I giorni sono indipendenti tra loro |
| 5 | Processi | Uno | Uno per area con `--parallel-aois` | Le aree sono indipendenti tra loro |
| 6 | Tentativi di lettura | 3, attesa fino a 4 s | 8, attesa casuale fino a 60 s; i blocchi già letti non si rileggono | Con più carico un singolo errore non deve fermare un'area |
| 7 | Coordinate | Conversione PROJ ripetuta | Riuso per cella e CRS, raggruppamento vettoriale | Stessa funzione con stessi argomenti, senza ricalcolo |
| 8 | Spazio disco | Conteggio per un solo processo | Quota fissa per processo | I processi non si vedono: la somma delle quote resta nel limite |
| 9 | Celle escluse | Riesaminate a ogni ripresa | `excluded.json` con identità è definitivo | Senza intermedi per cella andrebbero riscaricate |

## 4. Modifiche file per file

| File | Tipo | Contenuto |
|---|---|---|
| [src/biomap_v2/preparation.py](src/biomap_v2/preparation.py) | Modificato | Checkpoint giornalieri, pool dei giorni, consolidamento a gruppi, ripresa, blocchi per area, esclusioni definitive |
| [src/biomap_v2/raster_cache.py](src/biomap_v2/raster_cache.py) | Modificato | Cache disattivabile senza toccare quella esistente, riuso PROJ, tentativi robusti |
| [src/biomap_v2/parallel_prepare.py](src/biomap_v2/parallel_prepare.py) | **Nuovo** | Coordinatore dei processi: quote, riavvii, avanzamento, validazione finale |
| [src/biomap_v2/preparation_profile.py](src/biomap_v2/preparation_profile.py) | Modificato | Avanzamento in giorni, etichetta area nei report, quota disco |
| [src/biomap_v2/data.py](src/biomap_v2/data.py) | Modificato | Opzione `--aoi`, blocchi per area, target in sola lettura nei processi |
| [src/biomap_v2/cli.py](src/biomap_v2/cli.py) | Modificato | Opzioni `--aoi`, `--parallel-aois`, `--aois`, `--stop-after-minutes`, `--max-new-gib` |
| [src/biomap_v2/common.py](src/biomap_v2/common.py) | Modificato | Parametro `day_workers` e nuovi limiti di validazione |
| [configs/msa_v1_liguria.yaml](configs/msa_v1_liguria.yaml) | Modificato | `batch_cells` 8192, `io_workers` 8, `day_workers` 4, `block_cache_gib` 0, `memory_cache_mib` 0, `read_attempts` 8 |
| [scripts/verify_prepare_equivalence.py](scripts/verify_prepare_equivalence.py) | **Nuovo** | Confronto esatto con celle reali già complete |
| [tests/test_preparation_performance.py](tests/test_preparation_performance.py) | Modificato | Vedere sezione 6.1 |
| [README.md](README.md), [PREPARAZIONE_MSA_PRESTAZIONI.md](PREPARAZIONE_MSA_PRESTAZIONI.md) | Modificati | Comandi attuali, avvertenza sui numeri storici |

Ogni script aggiunto o modificato ha un'intestazione che spiega cosa fa, la logica passo per passo e come si richiama.

## 5. Velocità misurata

Prova sulle celle mancanti reali, 12 minuti per livello, con 8 worker e 4 giorni per processo. I blocchi al secondo sono la media dell'intera prova, avvio incluso.

| Processi | Thread di rete | Blocchi/s totali | Tempo medio per blocco | Errori di lettura |
|---|---|---|---|---|
| 2 | 16 | **23,8** | 0,50-0,85 s | 0 |
| 4 | 32 | **23,5** | 0,87-1,70 s | 0 |
| 8 | 64 | **13,4** | 3,2-5,8 s | 5, tutti recuperati |

**Interpretazione:** il totale non cresce con i processi, mentre il tempo per blocco cresce con il numero di thread. Il lavoro locale è trascurabile: al massimo 32 s di coordinate in 12 minuti per processo. A 100 Mbps il collegamento consente al massimo circa 12,5 MB/s; con una dimensione media dei blocchi compressi di circa 575 KB corrisponde a circa 22 blocchi/s, coerente con la misura. I 575 KB sono stati misurati sulla cache su disco, quindi questa stima non è una misura diretta del traffico. Il calo a 8 processi non è stato analizzato a fondo: l'ipotesi è la contesa sul collegamento saturo.

**Scelta adottata:** 4 processi. Il totale è uguale a 2 processi, senza errori, e più aree avanzano insieme.

**Durata stimata con 100 Mbps: circa 10-14 ore** per le celle mancanti. Il riferimento è il vecchio motore, con 10,66 blocchi per cella su circa 105.000 celle: circa 1,1 milioni di blocchi, cioè circa 13 ore a 23,5 blocchi/s. Il limite basso assume meno letture grazie al lotto per area. Il numero di blocchi per area intera non è ancora misurato.

**Con un collegamento da 1 Gbps** il limite di banda salirebbe di 10 volte. L'obiettivo di 5 ore diventerebbe plausibile, ma **non è provato**: i limiti di Planetary Computer con più carico non sono stati misurati.

**Riferimento:** nella prova di equivalenza su 300 celle per area, con 3-4 processi che dividevano la stessa banda, ogni area ha raggiunto circa 1.470-1.540 celle/ora, contro le 532-632 del vecchio motore. È un riferimento parziale, perché un gruppo di 300 celle non equivale a un'area intera.

## 6. Verifiche

### 6.1 Test automatici

Suite completa: **55 test superati**, senza rete. Nel file di test sulle prestazioni:

| Area | Cosa viene provato |
|---|---|
| Equivalenza | Lotti da 1 a 8.192, fino a 16 worker per bande e per giorni, con e senza cache: array, dtype, manifest delle scene e frazioni identici al percorso originale |
| Ripresa | Interruzione durante i giorni e durante il salvataggio delle celle, con risultato identico |
| Vecchio formato | Gli intermedi per cella vengono riusati e poi rimossi |
| Esclusioni | Una cella esclusa non si riscarica |
| Processi | `--aoi` non scrive `cells.csv`; blocchi reciproci; quota disco; cache esistente intatta |
| Rete | Un nuovo tentativo non rilegge i blocchi già ottenuti |
| Avanzamento | La velocità ignora i giorni recuperati dai checkpoint |

Un test è stato riscritto: `test_interrupted_batch_resumes_verified_days` richiedeva i file giornalieri per cella, che non vengono più scritti. Ora verifica il checkpoint unico e confronta il risultato con un riferimento senza interruzioni.

Il coordinatore `run_parallel` non ha un test automatico completo: è stato provato sul campo nelle tre prove della sezione 5, compreso l'arresto a tempo. La validazione finale automatica non è ancora stata eseguita sulla campagna reale.

### 6.2 Confronto con il processo lungo

Comando: `scripts/verify_prepare_equivalence.py`. Le celle sono state riscaricate da zero in una cartella separata, con la configurazione nuova e i cataloghi congelati.

| Area | Celle | Differenze | File identici byte per byte | Blocchi letti | Errori di lettura | Tempo |
|---|---|---|---|---|---|---|
| Abruzzo | 300 | 0 | 300 | 3.230 | 0 | 11,7 min |
| Appennino | 300 | 0 | 300 | 5.150 | 0 | 12,2 min |
| Basilicata | 300 | 0 | 300 | 8.001 | 21, tutti recuperati | 20,7 min |
| Belluno | 300 | 0 | 300 | 3.326 | 0 | 12,1 min |
| **Totale** | **1.200** | **0** | **1.200** | | | |

Il confronto usa uguaglianza esatta su chiavi, dtype, forme e valori degli array, oltre ai campi del manifest: protocollo, anno, punti, limiti, frazione valida, bande, politica dei giorni, unità SAR e scene.

**Cosa non copre:** le aree mai lavorate dal vecchio processo non hanno un riferimento, quindi lì l'equivalenza è dedotta dal codice e dai test. Il confronto copre 1.200 delle circa 27.700 celle già complete, in gruppi compatti da 300 celle e non su aree intere.

Durante la prova Appennino è fallita al primo tentativo per un percorso oltre il limite di 260 caratteri di Windows, causato dalla cartella temporanea di prova, ed è stata rilanciata da una cartella più corta. Con la cartella `data` reale i percorsi sono più corti. La cartella di prova è stata rimossa.

## 7. Cosa è rimasto invariato

- **Scienza:** 132.900 celle, target, 25 punti, anno 2020, bande, maschera SCL, NoData, conversione Sentinel-1, ordine delle scene e arresto del mosaicking per cella.
- **Dati esistenti:** `series.npz` e `manifest.json` completi, `PROTOCOL`, cataloghi congelati e formato dei file. Il campo `io_engine` resta `batched_native_v1`.
- **Controlli:** validazione delle celle complete, checksum e rilettura di verifica di ogni serie, controllo finale dei conteggi prima di scrivere `cells.csv`.
- **Percorso `legacy`:** invariato, resta il riferimento dei test.

## 8. Come si usa

Avvio e ripresa, con lo stesso comando:

```powershell
Set-Location 'C:\Users\brome\Desktop\BIOMAP_TESSERA_V2\biomap-tessera-v2-finetuning'
& .venv\Scripts\python.exe scripts\probe_msa.py prepare --config configs\msa_v1_liguria.yaml --data-root ..\data --parallel-aois 4
```

Riga di avanzamento: il valore affidabile è `blocchi/s`. `salvate` cresce solo a fine area; `stimate` ed `ETA` sono indicativi, anche perché i giorni Sentinel-2 vengono per primi e costano più di quelli Sentinel-1. Log e report sono in `..\data\msa_v1_liguria\diagnostics\<timestamp>_parallel`.

## 9. Rischi e punti aperti

| Punto | Stato | Cosa fare |
|---|---|---|
| Collegamento a 100 Mbps | Aperto, **è il limite principale** | Controllare cavo, presa e porta dello switch; la negoziazione può essere limitata anche dalla rete dell'università |
| Limiti di Planetary Computer | Non misurati oltre 100 Mbps | Dopo un eventuale aumento di banda, ripetere la prova a 4, 8 e 15 processi |
| Interruzione di un'area | Perde solo i giorni in corso | Rilanciare lo stesso comando |
| Area fallita più di 5 volte | Le altre continuano, la validazione finale non parte | Leggere il log dell'area e rilanciare |
| Cache `observation_cache\native_v1` | Inutilizzata, circa 8 GiB | Non cancellata; può essere rimossa solo dopo la fine della campagna, con una decisione esplicita |
| Numero di blocchi per area intera | Non misurato | Sarà noto al completamento della prima area |
| Fasi successive | Non toccate | Embedding su GPU e probing vengono dopo `prepare` e non sono stati misurati qui |
