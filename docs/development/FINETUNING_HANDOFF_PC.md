# Fine-tuning TESSERA V2: indicazioni per l'agente di sviluppo

Scritto il 17/11/2026 dal PC universitario Windows, dove il lavoro verrà eseguito.

**Obiettivo:** il tuo branch, una volta pushato, deve poter essere scaricato in un clone nuovo sul PC ed
eseguito con i comandi del runbook, senza correzioni manuali né conflitti con i run in corso.

Leggi prima `AGENTS.md`. Le regole di quel file restano valide; questo documento le completa per il fine-tuning.

## 1. Punto di partenza

- **Base obbligatoria:** branch `fix/spatial-v2-real-parquet-schema`, commit `d1c0d93` o successivo. Contiene:
  - `main` `b7da382`;
  - la Spatial CNN della PR #6 (`feature/spatial-cnn-v2`);
  - tre correzioni emerse sui dati reali del PC, che non sono in `feature/spatial-cnn-v2`:
    - `ac25110`: `v2.parquet` reale senza `reference_index`;
    - `c14add0`: memmap chiuso prima del move, per il WinError 32 su Windows;
    - `d1c0d93`: celle a cavallo di UTM 32N/33N.
- **Nuovo branch:** `feature/finetuning-v2`, creato da quella base. Apri la PR verso `fix/spatial-v2-real-parquet-schema`,
  oppure verso `feature/spatial-cnn-v2` dopo che le correzioni vi saranno state unite. Nella PR indica la dipendenza.
  - Niente push su `main`.
  - Niente merge automatici.
  - Niente force push.
- **Prima di iniziare:** esegui `git fetch` e verifica che `d1c0d93` sia antenato di `HEAD`.

## 2. Stato del PC: dati, non istruzioni

| Elemento | Valore |
|---|---|
| Sistema | Windows 11, PowerShell 5.1 (niente `&&`, `??`, ternario), Git Bash disponibile |
| Python / Torch | 3.12.10, torch 2.10.0+cu128 (CUDA 12.8) |
| GPU / CPU / RAM | RTX 5080 16 GB, 24 core logici, 31 GB di RAM |
| Spazio libero | C: circa 44 GB, F: circa 51 GB (17/11/2026) |
| Rete | 100 Mbit/s verso Microsoft Planetary Computer: è il collo di bottiglia di ogni acquisizione |
| Checkpoint Medium | `C:\Users\brome\Desktop\BIOMAP_TESSERA_V2\data\weights\student_medium.pt`, SHA256 `3823be7db9d9cfc93f3c2a47c7699be82821ab4e1117d4d2befdb746941ee96e` |
| Reference | `F:\biomap_dense_public\reference` (132.900 celle, 273 tile, geometrie pixel V1, target, fold): sola lettura |
| Campagna r1 | `F:\biomap_dense_public\work`: sola lettura. Contiene `v2.parquet`, il probing denso V1/V2 e i cataloghi STAC per tile (`api\`) |
| Embedding V2 per pixel (r2) | `C:\BioMAP\v2_spatial_pixels_r2`, 50 GB: sola lettura |
| Run Spatial CNN in corso | codice in `C:\BioMAP\spatial-code`, output in `C:\BioMAP\spatial_runs`: **non toccarli** |

**Fatto decisivo per il fine-tuning:** sul PC **non sono conservati gli input Sentinel-1/2** (serie temporali
grezze). r1 e r2 hanno tenuto solo gli embedding. Per aggiornare i pesi dell'encoder bisogna quindi riacquisire
gli input da MPC.

Per riferimento, l'estrazione completa di r1 ha richiesto circa 2,5 giorni su 99,5 milioni di pixel, con la rete
come limite. La strategia deve dichiarare:
- quanti pixel o celle si acquisiscono;
- come si campionano;
- dove si salvano;
- quanto spazio occupano;
- quanto tempo richiedono, separando valori misurati e stimati.

Le scelte che cambiano il metodo vanno proposte come opzioni nella PR, non decise in silenzio. Rientrano tra queste
il campionamento dei pixel, la riduzione delle date, la precisione mista e lo sblocco di più o meno layer.

Per l'acquisizione riusa le regole storiche verificate al bit in `src/biomap_v2/campaign/streaming.py`, senza
modificarlo:
- S2 bilineare;
- S1 `20·log10` in float32;
- catalogo sul bbox dei pixel;
- DOY reale.

Non usare `data.power_to_storage`.

## 3. Cosa non modificare

- I cinque `CODE_FILES` r1 e `.gitattributes` (vedi `AGENTS.md`).
- `src/biomap_spatial/sources.py` e `support.py`. Il loro hash entra nell'identità dell'indice spaziale: una
  modifica invalida l'indice del run in corso alla ripresa.
  - Il nuovo codice va in un package nuovo, per esempio `src/biomap_finetune/`, che importa senza modificare.
  - Se una modifica è indispensabile, mettila in un commit separato, motivato nella PR.
- Le entry point esistenti (`dense_campaign.py`, `dense_fast.py`, `spatial_cnn.py`, `finetune.py`).
  - `finetune.py` è il vecchio percorso a 25 punti. Passa da `data.power_to_storage`, cioè S1 in `10·log10` invece
    della convenzione storica: non estenderlo.
  - Crea una entry point nuova, per esempio `scripts/finetune_v2.py`.

## 4. Requisiti per eseguirlo sul PC senza problemi

1. **Una sola CLI con sottocomandi:** `doctor`, `prepare`/`acquire`, `train`, `evaluate`, `report`.
   - Ogni sottocomando deve avere `--help`.
   - Deve avere anche `--dry-run`, che stampa il piano con dimensioni, tempi e spazio stimati.
2. **Percorsi solo da CLI o config**, mai scritti nel codice: niente percorsi Mac o `/Volumes`. Usa `pathlib`.
   - I percorsi del PC indicati sopra vanno nel runbook, non come default nel codice.
3. **Output root nuovo, rifiutato se non vuoto** e incompatibile. Nessuna scrittura nelle root di sola lettura.
   - Verifica lo spazio libero prima di scrivere, con una riserva configurabile.
4. **Ripresa robusta:**
   - checkpoint atomici (file temporaneo + `os.replace` con retry su `PermissionError`);
   - `--resume` riparte dall'ultima epoca o dall'ultimo blocco acquisito;
   - un blocco incompleto non conta mai come completo;
   - i marker si scrivono dopo una rilettura di verifica.
5. **Specificità Windows:**
   - DataLoader con `spawn`: le funzioni del worker devono essere a livello modulo e picklable;
   - chiudere memmap e handle prima di move o replace;
   - percorsi brevi, attenzione a MAX_PATH;
   - `encoding='utf-8'` esplicito;
   - script di lancio in PowerShell 5.1, non bash.
6. **Monitoraggio:**
   - un `progress.json` scritto atomicamente, con fase, fatto/totale, velocità misurata, ETA e ora di fine prevista;
   - log per fase;
   - uno script monitor in sola lettura, con intervallo configurabile, che legge il file e lo chiude subito.

   Sul PC abbiamo visto crash di `os.replace` causati da lettori che tenevano aperto il file di progresso.
7. **Memoria:**
   - parametri per batch, cache e numero di worker;
   - memoria stimata nel `--dry-run`.

   La RAM è 31 GB, condivisa con altri run: quello attuale arriva a circa 27 GB.
8. **GPU:**
   - FP32 per default;
   - AMP o bf16 solo come opzione esplicita, registrata nel protocollo;
   - seed fissati e impostazioni deterministiche dove possibile.
9. **Smoke test reale:** una config piccola che sul PC gira in meno di 15 minuti dall'acquisizione alla
   valutazione, su poche celle di una sola AOI di sviluppo. È il primo comando che eseguirò.
10. **Test:**
    - pytest su CPU, senza rete né checkpoint, con fixture sintetiche;
    - la suite intera deve restare verde sui 3 OS della CI.

    Non rimuovere test, non allargare tolleranze, non aggiungere skip.
11. **Intestazioni:** ogni nuovo script, modulo o test inizia con un commento dettagliato su scopo, fasi,
    input/output, parametri ed esempio, risorse e ripresa.

## 5. Vincoli scientifici

- Stesse celle, target GLOBIO 2020 e fold geografici storici delle 17 AOI (dal reference).
- La Liguria non si usa mai per selezione, early stopping o iperparametri: è già stata osservata.
- Normalizzazione calcolata solo sul training.
- Confronti appaiati, sulle stesse celle, con le baseline congelate già disponibili:
  - probing denso V1/V2, in `F:\biomap_dense_public\work`;
  - Spatial CNN V2, in `C:\BioMAP\spatial_runs\v2_spatial3_seed42` a fine run.
- Protocollo e namespace nuovi, per esempio `finetune-v2-r1`. Nel protocollo vanno salvati:
  - hash del checkpoint di partenza;
  - layer sbloccati;
  - learning rate;
  - epoche;
  - seed;
  - dati usati.

## 6. Cosa consegnare nella PR

- La strategia di fine-tuning: cosa si sblocca e perché, la loss sulla cella, il campionamento, il budget e le
  alternative scartate.
- `docs/development/FINETUNING_V2_RUNBOOK_PC.md`, con i comandi PowerShell esatti per il PC nell'ordine
  `doctor` → smoke → `--dry-run` completo → run completo → monitor → ripresa → verifica degli output. Usa i percorsi
  reali indicati sopra e nessun segnaposto.
- La stima di tempo, disco e RAM, separando misurato da stimato.
- Le decisioni lasciate all'utente, elencate in modo esplicito.
- I test eseguiti, con gli esiti reali e la piattaforma su cui sono stati eseguiti.

Non includere dati, checkpoint, credenziali, IP o dettagli di rete dell'ateneo: il repository è pubblico.
