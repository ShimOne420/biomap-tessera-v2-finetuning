# Rischi

| Rischio | Dove | Probabilità | Mitigazione |
|---|---|---|---|
| **Cache incompatibili** tra r1 e r2 | identità diverse, voluto | certa | Le cache non si mescolano mai: work-root separate, `refuse_production`, `cached()` solleva errore su identità diversa. |
| **Identità instabile per i fine riga** | `core.autocrlf=true`: un checkout trasforma LF in CRLF e cambia gli SHA256 dei `CODE_FILES` | alta senza correzione | `.gitattributes` con `*.py text eol=lf` su questo branch. Il codice r1 di produzione **non è versionato**: va committato, con lo stesso `.gitattributes`, prima di qualsiasi riproduzione. |
| **Divergenza numerica su dati reali** | COG, GDAL o GPU reali diversi dai sintetici | bassa | Verifiche reali di `EQUIVALENZA.md` prima di usare la r2; criterio bit per bit. |
| **GDAL con handle riusati** | `HandlePool`: un handle per thread, scadenza 600 s, scartato a ogni errore | bassa | Un handle non viene mai condiviso tra thread. Test di equivalenza con riuso effettivo (`reuses > 0`). |
| **Throttling di Planetary Computer** | più richieste in parallelo e lotti | media | Pool globale `--io-workers`; letture con 6 tentativi e attesa crescente (r1 con jitter). Conviene partire da `--io-workers 16`, come la r1. |
| **Contesa sulla GPU** | r2 lanciata mentre la r1 gira | media | Non lanciare r2 né benchmark reali durante la campagna senza approvazione; un solo thread GPU per processo. |
| **Esaurimento della RAM** | lotti grandi, prefetch, cache; produzione già a 13,4 GB su 31 | media | `--batch-pixels 1.000.000`, `--prefetch 1`; cache configurabili. Niente secondo worker oggi. |
| **Esaurimento del disco** | tmp per tile e lotto (giorni ed encoded); pixel embeddings circa 51 GB | media | I temporanei si cancellano a ogni tile completata; `--keep-pixel-embeddings` solo se serve; tenere almeno 20 GB liberi su F: (oggi 54). |
| **Accesso concorrente ai file** | qualsiasi lettura di `stream_progress.json`; la condivisione Delete **non** basta (verificato) | alta con un monitor attivo: arresti r1 dell'11/11 e del 12/11 | r1: monitor `watch_v2_dense_safe.ps1`, che non apre quel file. r2: temporanei unici, `os.replace` con ritentativi, avanzamento mai fatale. |
| **Riavvii di Windows Update** | riavvio forzato fuori dalle ore attive (8–20) | avvenuto l'11/11 alle 20:30 (14,5 h perse) | Sospendere gli aggiornamenti per la durata della campagna (azione manuale dell'utente). Lo script di ripresa non sopravvive a un riavvio. |
| **Codice r1 non versionato** | worktree di produzione: `streaming.py` e altri non tracciati | alta (rischio di perdita o modifica accidentale) | Copia byte per byte nel commit `f5bcfd9` di questo branch. Commit nel branch di produzione consigliato dopo la fine della campagna. |
| **Collegamento di rete a 100 Mbit/s** | cavo, presa o switch | certa (misurata) | Vedi `ACCELERAZIONE_CAMPAGNA_ATTUALE.md` §1. |
| **Limite dell'encoder FP32** | a 1 Gbit/s il limite diventa la GPU | — | Margine possibile, da verificare: preparazione CPU dei batch sovrapposta alla GPU (stesso batch, risultato identico). AMP e batch diversi sono esclusi dal protocollo attuale. |
