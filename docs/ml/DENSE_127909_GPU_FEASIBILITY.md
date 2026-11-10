# Fattibilità GPU

RTX5080 16GB, Windows e 32GB RAM sono caratteristiche dichiarate, non hardware ispezionato da questa sessione. Nessun training scientifico o estrazione completa lanciati. Default FP32, batch32, chunk4096; inferenza frozen, no_grad ufficiale dentro inference_mode, output CPU e somme float64. OOM dimezza batch senza tagliare dati.

profile misura un numero limitato di chunk, forza il forward anche se esistono embedding legacy e salva tempi input/inferenza/scrittura, pixel/s, celle complete e VRAM. Il primo caricamento checkpoint è incluso nel tempo di inferenza. RAM via resource su Unix; su Windows via GetProcessMemoryInfo, non verificato sul PC reale. Le stime sono lineari, limitate al campione e non garantite. Profilare AOI e densità temporali diverse prima di budgettare l'intera campagna. Il caching compresso annuale rimane un costo I/O rilevante; decompressione solo nella work-root.

Nessuna promessa di ore o giorni è scientificamente supportata prima del pilot sul PC universitario. Test CPU e OOM simulato non certificano CUDA, VRAM o throughput reali.
