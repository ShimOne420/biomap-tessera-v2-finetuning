# GeoTESSERA V2 Medium: audit preliminare al partial fine-tuning

Data: 17 novembre 2026. Base verificata `1737b1fdd30d5de9a50967363ca39042b26a055b`, branch `feature/finetuning-v2`. L'audit precede l'implementazione. Main resta `b7da382`; la Spatial CNN e le tre correzioni PC sono nella base scelta, non si presume il merge della PR #6.

## Esito

**SUPPORTED_WITH_CHANGES.** Il modello ha un forward Torch differenziabile e blocchi selezionabili; l'API r1 di estrazione, invece, usa inference mode e NumPy. Occorre un adapter separato per il training: nessuna modifica ai file protetti o agli script del run in corso.

È stato trovato un checkpoint locale già materializzato. Prima di caricarlo è stato verificato SHA256 `3823be7db9d9cfc93f3c2a47c7699be82821ab4e1117d4d2befdb746941ee96e`, dimensione **84.163.403 byte**. Non è stato scaricato alcun peso. Il caricamento tramite `encoder.load_encoder`, con `load_state_dict(strict=True)`, è riuscito senza chiavi mancanti/inattese. Il percorso Mac non è pubblicato e non è un default del codice.

## Architettura effettiva del checkpoint

I valori sotto provengono da `payload['args']` e dallo state dict verificato, non dai default del costruttore:

| Campo | Valore reale |
|---|---:|
| repr_dim | 128 |
| latent_dim | 110 |
| dimensione temporale interna | 440 = 4 × 110 |
| num_layers per backbone | 4 |
| attention heads | 4 |
| feed-forward dimension | 1792 |
| max_seq_len / bin cap | 256 |
| enable_qk_norm | false |
| parametri | 21.031.506 |

`student.PixelStudent` contiene `s2_backbone` e `s1_backbone`. Ciascuno ha `embedding` (Linear/ReLU/Linear), `temporal_encoder` sinusoidale sul DOY reale, `transformer_encoder.layers.0` … `.3` e `attn_pool.query`. Sono Transformer temporali per pixel, non blocchi convoluzionali spaziali. Ogni encoder layer ha self-attention, feed-forward e LayerNorm, con connessioni residuali interne.

I due attention pooling restituiscono 440D ciascuno. La concatenazione 880D entra in `dim_reducer`: Linear 880→1760, LayerNorm affine, ReLU, Dropout 0,2, Linear 1760→128 e LayerNorm finale non affine. Il checkpoint ha 114 tensori nello state dict. Il nome del repository del teacher non indica che questo student abbia due miliardi di parametri.

## Confine prefisso/suffisso e policy proposta

Per `last_n_blocks = n`, selezionare gli ultimi n **di entrambi** i backbone, non gli ultimi moduli Python dell'intero modello. Con n=2: `.layers.2` e `.layers.3` di S1 e S2. Embedding iniziali e layer precedenti restano congelati, in eval. Il DOY encoder non ha parametri apprendibili.

`train_output_projection: auto` si risolve esplicitamente in fusione `dim_reducer` trainabile e attention pooling di entrambi i backbone trainabili. `train_normalization: auto` permette solo le affine norm dei layer trainabili e della fusione abilitata; le norm del prefisso restano congelate. La LayerNorm finale non affine non aggiunge parametri. Disabilitare la fusione non deve interrompere il gradiente verso i blocchi precedenti: si congelano i suoi parametri, senza avvolgerne il forward in no_grad.

Il prefisso deve rimanere in eval durante `model.train()`. Il suffisso e l'eventuale Dropout della fusione seguono il modo training; l'inferenza usa eval per tutto. La configurazione risolta e i nomi dei parametri saranno registrati, con conteggi misurati, non euristiche silenziose.

## Forward e preprocessing

`student.PixelStudent.forward/encode` conserva il grafo. `reference.encode_pixels` è decorata con no_grad e restituisce NumPy; `campaign.extraction.infer` aggiunge inference_mode; la r1 imposta anche requires_grad False. Nessuno di questi percorsi sarà modificato per renderlo trainabile.

Il nuovo adapter userà gli helper ufficiali di binning e le costanti S1 asc/desc separate, concatenazione S1 dopo normalizzazione, validità calcolata prima dello z-score e ordinamento originale dei pixel. È necessario accettare DOY 366 del 2020: il vecchio `encoder.groups` limita a 365 e non sarà esteso. L'acquisizione riuserà `streaming.compose_day`, con S2 bilineare, S1 20·log10 float32, stesso catalogo sul bbox completo dei pixel e nessuna riduzione arbitraria delle date. Non si userà `data.power_to_storage`.

## LoRA e limiti

Il checkpoint reale ha QK norm disabilitata: usa MultiheadAttention con QKV combinate (`self_attn.in_proj_weight`) e proiezione `out_proj`; non ha i `q_proj/k_proj/v_proj` separati della variante QK-norm presente nel sorgente. Linear del feed-forward e della fusione sono altri candidati potenziali. PEFT non è una dipendenza installata del progetto. Compatibilità di una particolare implementazione LoRA, risparmio VRAM e qualità non sono verificati; LoRA non viene implementata.

Il caricamento strict reale è verificato sul Mac. La piccola prova differenziabile, l'equivalenza del prefisso e i conteggi dopo freezing verranno registrati nel report finale. Nessuna serie Sentinel reale, metrica MSA o prova RTX 5080 è stata eseguita in questo audit.
