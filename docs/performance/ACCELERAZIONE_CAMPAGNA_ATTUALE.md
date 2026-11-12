# Si può accelerare la campagna r1 in corso senza interromperla?

Il requisito è che il processo in produzione continui esattamente com'è. Ogni proposta è classificata così:
- **compatibile** con zero interruzioni;
- **richiede approvazione**, perché scrive in produzione o lancia nuovo carico;
- **incompatibile**, perché va fermato o riavviato il processo.

Nessuna di queste azioni è stata eseguita.

## 1. Collegamento di rete a 100 Mbit/s → 1 Gbit/s (consigliata, richiede approvazione)
- **Stato misurato:** la scheda Realtek 2.5GbE è in negoziazione automatica ma si è collegata a 100 Mbit/s. Il link è full duplex; durante l'acquisizione il traffico era di 56 Mbit/s.
- **Causa tipica:** cavo Cat5 o danneggiato (con 2 coppie su 4 il collegamento resta a 100 Mbit/s), presa a muro o porta dello switch a 100 Mbit/s.
- **Effetto atteso:** nel benchmark simulato la r1 è 1,86 volte più veloce a 1 Gbit/s (103 s contro 55 s). La rete di ateneo e Planetary Computer potrebbero dare meno: va verificato.
- **Rischio:** cambiare il cavo interrompe la rete per alcuni secondi.
  - Le letture r1 hanno 6 tentativi con attesa crescente, fino a 60 s, quindi un'interruzione di pochi secondi viene probabilmente assorbita.
  - Se il run uscisse, lo script di ripresa attivo (processo 16748) lo rilancia: si perdono al massimo i giorni non salvati della tile in corso.
- **Classificazione:** **richiede approvazione**, con breve rischio di interruzione. Il momento migliore è subito dopo una riga "Tile completa" in console.
- **Verifica dopo il cambio, solo lettura:** `Get-NetAdapter -Name Ethernet | Select LinkSpeed` deve indicare 1 Gbps.

## 2. Secondo worker su AOI disgiunte (oggi sconsigliato)
- **Fattibilità tecnica:** sì. Il lock `campaign.lock` vale per singola work-root. Le tile prodotte dallo stesso codice hanno identità identica, quindi copiate in `work\chunks` verrebbero riusate dal processo A.
- **Perché oggi no:**
  - **RAM:** A occupa 13,4 GB su 31; un secondo worker uguale porterebbe a circa 27 GB, con rischio di swap o di errore di memoria per **entrambi**.
  - **Banda:** A usa già circa 56 Mbit/s su 100. Due worker se la contenderebbero, con un guadagno stimato al massimo di 1,3–1,5 volte (E).
- **Con collegamento a 1 Gbit/s** diventa interessante, con due approvazioni:
  1. avvio di `dense_campaign.py stream --aoi <coda> --work-root F:\biomap_dense_public\work_b`, con `--io-threads 8` per limitare la RAM;
  2. copia delle tile finite (`tile.npz`, poi `tile.json`) in `work\chunks` **prima** che A raggiunga quelle AOI. La copia va fermata quando A è a meno di un'AOI di distanza.
- **Classificazione:** **richiede approvazione** (nuovo carico più scrittura in produzione). Oggi non consigliato per il limite di RAM.

## 3. Usare la pipeline r2-fast per il resto della campagna (incompatibile)
- **Perché è incompatibile:** la r2 ha un'identità diversa, voluta. Non riusa le tile r1 e richiederebbe di fermare la r1 oppure di ricalcolare da zero.
- **Ricalcolo da zero:** con le stime del benchmark (da 1,7 a 2,6 volte) e un collegamento a 100 Mbit/s non conviene rispetto alle circa 35 ore residue della r1.
- **Classificazione:** **incompatibile.** La r2 va usata per le prossime campagne (BII, mappe spaziali, eventuali ripetizioni).

## 4. Impostazioni di sistema (da applicare a mano, se approvate)
| Proposta | Effetto | Rischio |
|---|---|---|
| Piano energetico "Prestazioni elevate", sospensione e ibernazione disattivate | evita fermi notturni | nessuno |
| Esclusione di Defender per `F:\biomap_dense_public\work` | meno scansioni sui molti `.npz`/`.json` scritti | riduce la protezione su quella cartella |
| Disattivare "Energy Efficient Ethernet" (oggi attivo con massimo 1 Gbps) | a volte migliora la negoziazione | breve caduta del link al cambio |
| Gestione energetica GPU | nessun effetto atteso: la GPU è quasi sempre ferma in attesa della rete | — |

## 5. Monitoraggio sicuro (già disponibile)
- `scripts\watch_v2_dense_eta.ps1` apre `stream_progress.json` con `FileShare.ReadWrite|Delete` e lo chiude subito, ogni 60 s o più: non può impedire `os.replace`.
- **Da evitare:**
  - `Get-Content` in ciclo su quel file, che apre senza condivisione Delete ed è la causa probabile dei due arresti;
  - il file tenuto aperto in un editor.
- `scripts\audit_dense_log.py` legge `stream_log.jsonl` nello stesso modo.

## Raccomandazione
1. **Subito, senza rischi:** sospensione disattivata. Lo script di ripresa è già attivo. Nessun monitor aggressivo.
2. **Con approvazione:** sostituire il cavo o la porta per arrivare a 1 Gbit/s, subito dopo una tile completata. È la singola azione con il guadagno potenziale maggiore sulla campagna in corso.
3. **Solo dopo l'upgrade di rete, e solo se la RAM lo consente:** valutare il secondo worker.
