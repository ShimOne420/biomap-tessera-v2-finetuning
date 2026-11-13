# Fattibilità del fine-tuning GeoTESSERA V2 per MSA

Audit statico del codice al 13 novembre 2026. Non sono stati scaricati pesi, eseguiti encoder preaddestrati o avviati esperimenti reali. I test del nuovo package eseguono soltanto il predittore CNN/MLP su fixture. La regressione legacy include anche PixelStudent ridotti, casuali e CPU, gia presenti nei test: non sono il Medium caricato da checkpoint. L'audit di fattibilita rimane statico; il fine-tuning resta una fase successiva.

## Fattibilità verificata dal codice

Il repository conserva `src/biomap_v2/student.py`, derivato dal codice ufficiale MIT e attribuito in `ATTRIBUTIONS.md`, commit TESSERA `e368539f7c4711e5f16fdd09801573830e34baf9`. `PixelStudent.forward(s2_x,s1_x)` chiama `encode` con normali operazioni Torch: due backbone temporali, attention pooling, concatenazione e `dim_reducer`, output 128D. Non c'è un detach nel forward del modello. La modalità eval non disabilita da sola i gradienti.

`encoder.load_encoder` costruisce il Medium dai parametri del checkpoint e carica lo state_dict in modalità strict. `common.py` blocca revisione dei pesi `8c80b926f67cb0b44c2cd41d0226e61741503b9a` e SHA256 `3823be7db9d9cfc93f3c2a47c7699be82821ab4e1117d4d2befdb746941ee96e`. Questa è una verifica del percorso di caricamento e dei riferimenti, non della presenza del file universitario.

Esiste già un percorso differenziabile `encoder.pixel_embeddings`: normalizza/binna le serie, converte in tensori, chiama il modello e può usare activation checkpointing. `encoder.trainable` permette testa soltanto, ultimi blocchi di entrambi i backbone con attention pooling/fusione, oppure encoder completo. Il percorso storico `Regression` fa però media→testa e non costituisce fine-tuning della nuova Spatial CNN.

Al contrario, `campaign/extraction.infer` usa `torch.inference_mode` e restituisce array NumPy. La r1 carica l'encoder con `requires_grad_(False)` e cancella i temporanei della tile dopo l'estrazione. Quel percorso è intenzionalmente frozen e **non va trasformato in trainer**, né modificato per questa attività.

La nuova CNN contiene operazioni Torch differenziabili rispetto ai tensori in ingresso e ai propri pesi; i test lo verificano con array sintetici. La proiezione iniziale appartiene al predittore, non all'encoder. La separazione model/protocol/source consente un futuro percorso live distinto, ma oggi il reader passa da NPY/NumPy e non collega il grafo all'encoder.

## Input e dipendenze mancanti per un esperimento reale

Servono le serie annuali Sentinel-2 e Sentinel-1 ascendenti/discendenti, bande, DOY, maschere, binning e normalizzazione identici al protocollo. Gli embedding precalcolati, anche per pixel, non consentono di ricostruire questi input o di calcolare gradienti rispetto all'encoder. La presenza dei raw completi sul PC non è verificata; la r1 elimina i temporanei acquisiti. Non è autorizzata una nuova acquisizione automatica in questo intervento.

Servono inoltre un adapter live che mantenga il grafo Torch, un piano di microbatch/ricomposizione spaziale e un trainer con gruppi di learning rate distinti. L'attuale engine addestra soltanto le teste e non espone opzioni che fingano di sbloccare il Foundation Model. Il lock attuale contiene Torch ma non PEFT. Nessuna nuova dipendenza pesante viene aggiunta per il solo audit.

## LoRA / PEFT: ipotesi, non compatibilità certificata

Con `enable_qk_norm=False`, il codice usa `nn.TransformerEncoderLayer`: attenzione con proiezione QKV combinata e ulteriori layer lineari. Con `enable_qk_norm=True`, `QKNormEncoderLayer` espone invece `q_proj`, `k_proj`, `v_proj`, `out_proj` separati. Il flag effettivo deve essere letto dal checkpoint reale: non si può applicare indistintamente una lista di nomi LoRA.

Una parametrizzazione a basso rango dei layer compatibili è plausibile, ma non verificata. Occorrono inventario dei moduli realmente caricati, iniezione controllata, test di equivalenza prima del training e gradienti solo sui parametri previsti. Non installiamo PEFT né dichiariamo che il modello sia già supportato da una sua factory.

## Risorse e rischi

La RTX 5080 dichiarata ha 16 GB: la disponibilità e il funzionamento CUDA universitari rimangono da misurare. Il profilo della sola CNN non stima il costo dell'encoder temporale. Un training end-to-end trattiene attivazioni per osservazioni e pixel, non soltanto la mappa 128D. Non è disponibile una misura affidabile del microbatch sostenibile o del tempo sui circa 99,5 milioni di contributi pixel.

I rischi principali sono VRAM, ricomputazione annuale, input raw non conservati, contesto fra patch, normalizzazione/DOY non equivalenti e modifica involontaria del supporto. La differenza S1 float32 tra host rimane documentata: un nuovo protocollo di fine-tuning deve dichiarare ambiente e provenienza, senza riscrivere le cache r1.

## Prossimo esperimento minimo, ancora da autorizzare/eseguire

Preparare una fixture annuale locale verificata su poche decine di pixel e una cella; caricare i pesi già disponibili sul PC, senza downloader implicito. Confrontare il forward frozen col percorso ufficiale, poi sbloccare soltanto la testa, quindi l'ultimo blocco/fusione in una namespace separata. Misurare gradienti, modifica effettiva dei soli pesi sbloccati, VRAM e ripresa; verificare che le patch producano una sola loss per cella. Solo dopo estendere la CV alle 17 AOI. Questo documento non esegue nessuno di questi passaggi.
