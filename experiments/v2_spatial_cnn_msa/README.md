# Esperimento V2 spatial CNN (MSA, 17 AOI + Liguria)

L'esperimento addestra la stessa CNN spaziale dell'Esperimento B ([v1_spatial_cnn_msa](../v1_spatial_cnn_msa)), sulle stesse celle, patch, maschere, fold e target. Cambiano solo i valori dei pixel: al posto degli embedding TESSERA V1 (int8 × scala) si usano gli **embedding TESSERA V2 per pixel**. Li estrae la pipeline r2-fast (`scripts/dense_fast.py stream --keep-pixel-embeddings`) sullo stesso reticolo dei pixel V1.

## Cosa resta identico a B

- `model.py` e `sources.py` sono **identici byte per byte** a B. Anche le funzioni di metrica sono identiche. I test lo verificano.
- Patch e maschere vengono dalla stessa funzione `fragments()`: centro del pixel nella cella e scala V1 finita.
- Il 100% dei pixel del supporto di B è codificato da r1/r2: 96,9 milioni di pixel su 281 tile, con 0 pixel mancanti (verificato il 13/11/2026).
- Normalizzazione stimata sul solo training, AdamW, clip, early stopping, selezione sulla macro-RMSE OOF, refit, Liguria osservata solo alla fine.
- **Prova di equivalenza.** `tests/test_v2_spatial.py` esegue il codice originale di B in un processo separato. Quando i valori V2 coincidono con i V1 dequantizzati, indice, medie e predizioni OOF e Liguria sono **identici bit per bit**.

## Cosa cambia: solo l'esecuzione

| Ottimizzazione | Effetto sul risultato |
|---|---|
| Worker del DataLoader persistenti (`num_workers`, `prefetch_factor`, `pin_memory`) | Nessuno: lo stesso consumo del generatore casuale di `num_workers=0` dà pesi identici (test) |
| I 3 fold della CNN e i baseline come processi paralleli sulla stessa GPU | Nessuno: ogni job ha gli stessi semi; risultati identici al run seriale (test) |
| Refit dei modelli in parallelo | Nessuno, per lo stesso motivo |
| Estrazione r2-fast invece di r1 | Medie per cella confrontate con la campagna r1 prima della build (gate automatico) |
| Mixed precision (`training.mixed_precision`) | **Disattivata** di default; si usa solo se verificata sui dati reali e dichiarata con `--set` |

## Avvio

Il run richiede una work-root nuova chiamata `v2_spatial_msa` e una work-root r2 separata da quella della campagna r1.

```powershell
$py = 'C:\BioMAP\venv-v2spatial\Scripts\python.exe'
& $py experiments\v2_spatial_cnn_msa\run_pipeline.py `
  --work-root C:\BioMAP\v2_spatial_msa --v2-root C:\BioMAP\v2_spatial_work\dense_r2 `
  --v1-root F:\biomap_dense_public\v1_tiles --reference F:\biomap_dense_public\reference `
  --r1-work F:\biomap_dense_public\work `
  --checkpoint C:\Users\brome\Desktop\BIOMAP_TESSERA_V2\data\weights\student_medium.pt --run-name full_seed42
```

Fasi, in ordine:
1. estrazione;
2. verifica r2 = r1;
3. build;
4. preflight;
5. profilo;
6. CV parallela;
7. selezione;
8. refit parallelo;
9. chiusura del refit;
10. valutazione OOF e Liguria.

Per riprendere basta rilanciare lo stesso comando: le fasi complete vengono saltate.

## Monitor live (sola lettura)

```powershell
& $py experiments\v2_spatial_cnn_msa\monitor_v2.py --work-root C:\BioMAP\v2_spatial_msa --v2-root C:\BioMAP\v2_spatial_work\dense_r2 --follow
```

Il monitor mostra, per ogni fase, avanzamento, velocità e tempo residuo, poi l'ora di fine stimata. Per il training la stima tiene conto dell'early stopping; tra parentesi c'è il massimo di 150 epoche.

## Risorse

- **Embedding V2:** circa 51 GB float32 su C:.
- **Indice SQLite:** circa 1 GB.
- **RAM:** circa 10 GB durante l'estrazione.
- **GPU:** l'estrazione usa la GPU; il training usa fino a 4 processi sulla stessa GPU.
- **Riciclo dell'estrazione:** se la memoria privata supera `--memory-limit-gb` (24), l'estrazione viene riavviata.

## Test

```powershell
& $py -m pytest experiments\v2_spatial_cnn_msa\tests
```

I test vanno eseguiti in un processo pytest separato da quelli di B, perché i moduli hanno gli stessi nomi.
