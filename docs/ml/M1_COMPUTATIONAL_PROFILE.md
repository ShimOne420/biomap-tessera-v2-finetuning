# M1 — Misure di preparazione del pacchetto

Misura locale del 10 novembre 2026, da `bundle.json`:

- Preparazione completa: **266,42 secondi**.
- Celle dense appaiate: **4.991**.
- Pixel densi corrispondenti: **3.545.951**.
- Workflow misurato: letture e checksum, controllo geometrico, replica V1, aggregazione
  V2 già codificato, costruzione tabella e campione.
- Payload ZIP finale con attribuzioni: **11.904.407 byte**.

Questo **non è il throughput dell'encoder**: gli embedding densi esistevano già. Non è
una previsione dei tempi della campagna 17 AOI o del PC universitario. Il costo di
creazione dello ZIP e delle attribuzioni successive non è una misura di inferenza.

L'aggregazione V2 usa chunk di 8.192 pixel con accumulatori float64 per cella. Questo
evita la conversione in RAM di un'intera tile in float64; non è stato misurato un picco
RAM/VRAM complessivo e non viene inventato. Le ottimizzazioni non cambiano media, pixel
selezionati, target o tolleranze.

Il nuovo run universitario a cinque regressori è ancora da eseguire. I tempi di ciascun
fit e il consumo reale sul PC dovranno guidare ulteriori ottimizzazioni, soprattutto
prima di acquisire o ricodificare tutte le altre AOI.
