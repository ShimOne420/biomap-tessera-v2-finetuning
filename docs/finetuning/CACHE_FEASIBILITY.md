# Cache del prefisso: fattibilità e budget

**Raccomandazione: ONLINE_RECOMMENDED.** Una cache completa delle attivazioni è **CACHE_NOT_FEASIBLE sui dischi descritti nel handoff**. `cache.mode=auto` viene risolto e registrato come online; non genera cache. `cached` è rifiutato dal trainer di corpus. È disponibile soltanto un prototipo FP32 locale, verificato nei test, senza comando di generazione massiva. Questa è una limitazione intenzionale dichiarata, non una cache completa già pronta.

## Artefatti da non confondere

| Artefatto | Provenienza/formato e uso | Disponibilità e integrità |
|---|---|---|
| Input Sentinel annuali | S2 uint16 `[N,T2,10]`, mask bool `[N,T2]`; S1 asc/desc int16 `[N,Ta/Td,2]`; DOY int16 | Assenti sul PC secondo handoff. Da acquisire nella nuova raw-root, anno 2020, preprocessing storico |
| Medie dense V2 | `F:\biomap_dense_public\work\v2.parquet`, 128 float per cella e colonne target/geografia | Dichiarate dal PC; non aperte dal Mac. Non permettono backward encoder |
| Mappe frozen V2 | `C:\BioMAP\v2_spatial_pixels_r2\pixel_embeddings\<aoi>\<tile>.npy`, float32 `[N,128]`, sidecar SHA/geometry/order | Circa 50 GB dichiarati; da verificare sul PC. Ordine `geometry.npz`, CRS/transform nel reference. Non sono stati intermedi |
| Checkpoint Medium | `student_medium.pt`, 84.163.403 byte, SHA `3823be7d…941ee96e` | Copia locale già presente verificata sul Mac, caricamento strict e forward/backward riusciti; percorso PC da verificare con doctor |
| Prefisso proposto | Due array float32 `[N,Tbin_S2,440]` e `[N,Tbin_S1,440]` | Nessun corpus disponibile o creato. Solo piccoli prototipi sintetici temporanei |
| Reference | `F:\biomap_dense_public\reference`, parquet celle/target/fold, lookup NPY, geometrie NPZ/JSON | 132.900 celle/273 tile dichiarate; loader controlla manifest e membership quando eseguito sul PC |

Le dimensioni e i percorsi PC sono informazioni fornite dal handoff, non osservazioni live. I NPZ `chunks/tile.npz` legacy contengono somme/conteggi, non vanno interpretati come serie Sentinel. Il riferimento completo ai metadati di normalizzazione, dimensioni e pesi è nell'audit architetturale. L'anno/checkpoint delle mappe deve essere provato dai marker della campagna, non dedotto dal nome di cartella.

## Confine e tensori necessari

Con gli ultimi due blocchi sbloccati, si salvano le uscite dopo embedding+DOY+layer 0/1, separatamente per S2 e S1. La dimensione interna reale è **440**, non 128 o 256. `Tbin` varia per gruppo di pixel: multipli di otto fino a 256 con campionamento/padding storico; una modalità mancante ha un placeholder di lunghezza uno. La fusione delle modalità avviene solo dopo i due suffissi e attention pooling. Non esiste una singola immagine intermedia 128D equivalente.

Un corpus avrebbe bisogno anche di ordine e raggruppamento dei pixel, bin temporali, mapping patch/cella, CRS/transform, supporto, identità dei raw e del prefisso, configurazione N, dtype/versione e marker. DOY è già incorporato nelle attivazioni; i Transformer correnti non ricevono una nuova maschera temporale da costruire arbitrariamente. Non aggiungere una maschera differente dal percorso ufficiale. Il prototipo memorizza entrambi i tensori, hash di token/pesi congelati e confine, ma non implementa il catalogo geografico completo.

## Calcolo riproducibile dello spazio

Per ogni gruppo `g`, bytes = `N_g × (T2_g + T1_g) × 440 × bytes_per_scalar`. Sommare gruppi/tile, poi aggiungere metadata, staging e margine filesystem. Non si assume compressione gratuita; i float intermedi possono comprimersi poco. FP16/BF16 dimezzerebbero il payload, ma **non sono implementati per la cache** e richiederebbero una validazione distinta della precisione e del gradiente.

Su **99,5 milioni** di pixel, usati qui solo come scala del handoff:

| Scenario | Payload FP32 decimale | FP16/BF16 teorico |
|---|---:|---:|
| Limite inferiore 8 token in una modalità + placeholder 1 | 1,58 TB | 0,79 TB |
| Esempio 104 token S2 + 160 S1 | 46,23 TB | 23,12 TB |
| Due modalità a 256 token | 89,66 TB | 44,83 TB |

Il minimo presuppone almeno una modalità valida per pixel. Le lunghezze effettive non sono state misurate sul PC. Anche il limite inferiore supera enormemente C:44 GB e F:51 GB liberi dichiarati, prima delle riserve. Le unità in tabella sono TB decimali; il controllo spazio usa GiB binari. Il piano restituisce byte per evitare ambiguità.

L'I/O per epoca della cache sarebbe almeno il suo payload letto una volta, moltiplicato per le epoche e per le ripetizioni CV/refit; include poi i costi di letture sparse e decompressione eventuale. A titolo illustrativo, 46,23 TB / 500 MB/s sono circa **25,7 ore per sola lettura sequenziale ideale di un'epoca**, senza forward/backward. 500 MB/s è un'ipotesi, non una misura dei dischi PC. Una cache molto piccola per un pilot può essere utile per studio, ma non giustifica l'estensione al corpus.

## Raw e checkpoint online

Esempio esplicitamente ipotetico: 100 giorni S2 e 160 giorni S1 combinati. I raw richiedono `N × (21×100 + 4×160) = N×2740` byte, più DOY/metadata/indici. A 99,5 milioni sono **272,63 GB**, con picco circa doppio se si conservano anche le copie giornaliere per ripresa. Il nuovo acquire calcola il budget usando i giorni **reali di tutti i cataloghi** prima delle letture raster; il dry-run senza cataloghi dichiara le ipotesi.

Con K=25 contributi su tutte le 132.900 celle il tetto è 3.322.500 pixel: circa 9,10 GB di raw nell'esempio, circa 18,21 GB con duplicazione giornaliera, prima degli overhead. K=25 è uno scenario di budget, **non un metodo selezionato**. Una finestra 2×2 per cella e tile non equivale necessariamente a quattro pixel per cella quando vi sono più tile.

Per N=2, i pesi encoder FP32 sono 84,13 MB e i due momenti AdamW dei parametri sbloccati circa 89,67 MB; aggiungere testa, stati, copie, gradienti e grafo. Un checkpoint può quindi essere dell'ordine di 175 MB, non solo il peso iniziale. Con 50 epoche e quattro fit (3 CV + refit), l'archivio immutabile può superare **34 GB**. Il piano riporta questo tetto se il numero di epoche è deciso. Non viene cancellato né sovrascritto un checkpoint per far posto: se manca spazio occorre un'altra destinazione o una decisione esplicita sul budget prima di partire.

La memoria per stati/pesi non misura la VRAM completa. Il grafo del suffisso, attenzione temporale, patch/head e batch celle possono dominarla. Il microbatch pixel limita un forward, ma concatenare output differenziabili conserva grafi per l'intera cella. Activation checkpointing riduce parte della memoria al costo di ricalcolo. Il batch sostenibile sulla RTX 5080 non è ancora misurato. Non eseguire questo pilot in parallelo al run che usa già circa 27 GB su 31 di RAM.

## Tempi e verifica numerica

100 Mbit/s corrispondono a 12,5 MB/s teorici di payload. Non sono una previsione di acquisizione: finestre COG, bande, scene e blocchi remoti possono essere letti più volte. Ridurre pixel mantenendo 17 AOI e tutte le date non riduce necessariamente il traffico nella stessa proporzione. Le 2,5 giornate della r1 sono un riferimento riportato dall'utente, non un benchmark di questo adapter. Nessuna promessa di completamento in cinque ore o meno di 15 minuti è derivata da una semplice proporzione.

Il prototipo confronta online e cached FP32 a parità di pesi in eval, e in train con seed identico per il Dropout del suffisso/fusione. Il prefisso resta sempre eval. Il caricamento rifiuta prefissi di pesi o confine diversi e verifica SHA. Test di backward verificano che la cache non annulli il gradiente del suffisso. Le tolleranze sono 2e-5; questi test sintetici non autorizzano FP16/BF16 né un corpus cached.

La decisione finale di budget richiede lo smoke reale, i contatori STAC/reader, spazio libero corrente e tempi osservati di acquisizione e di una cella training. Il risultato può legittimamente essere **INSUFFICIENT_INFORMATION** per durata e batch GPU anche se la raccomandazione online è già solida.
