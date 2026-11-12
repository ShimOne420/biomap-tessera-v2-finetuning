# Procedura: arresto ordinato, cambio del cavo, ripresa della campagna r1

Preparata il 12/11/2026. **Nessun passo è stato eseguito.** Ogni comando PowerShell va lanciato in una finestra **diversa** da quella del run.

## Esito delle verifiche (sola lettura)

### Collegamento Ethernet
| Voce | Valore | Lettura |
|---|---|---|
| Scheda | scheda Ethernet 2,5 GbE | supporta 10/100/1000/2500 Mbit/s |
| Velocità e duplex | **Auto Negotiation** | nessuna velocità forzata via software |
| Velocità negoziata | **100 Mbps**, full duplex | il collegamento si è accordato su 100 |
| Errori e scarti | 0 errori, 0 scartati su 39,1 milioni di pacchetti ricevuti | nessun segno di cavo rumoroso |
| Risparmio energetico Realtek | Green Ethernet **attivo**, Power Saving Mode **attivo**, Gigabit Lite **attivo**, EEE e Advanced EEE disattivati | vedi sotto |
| Eventi di collegamento | il driver non ne registra: la storia della velocità non è ricostruibile | — |

**Conclusione.** Il limite **non dipende da un'impostazione software modificabile**. La velocità è in negoziazione automatica e il gigabit non è disattivato. In negoziazione automatica, un collegamento a 100 Mbit/s su una scheda gigabit ha di solito una di due cause:
1. **cavo o presa con solo 2 coppie funzionanti su 4.** Il 1000BASE-T usa tutte e 4 le coppie: con 2 coppie la negoziazione scende a 100. Succede con cavi Cat5 vecchi o danneggiati e con prese o patch panel cablati a 2 coppie;
2. **porta dello switch o della presa limitata a 100 Mbit/s**, per hardware o per configurazione dell'ateneo.

Green Ethernet e Power Saving Mode riducono i consumi sui cavi corti o in inattività, ma non portano normalmente a 100 un collegamento gigabit sano. Gigabit Lite è una modalità Realtek a 500 Mbit/s su 2 coppie, che funziona solo se anche l'altro lato è Realtek. Cambiare queste voci è un intervento software da provare **solo se** un cavo Cat6 sicuro su una porta gigabit sicura resta a 100. Cambiarle interrompe brevemente il collegamento e va approvato a parte.

**Intervento consigliato:** fisico.
- Usare un cavo Cat6 o Cat5e nuovo; la categoria è stampata sulla guaina.
- Provare, se esiste, un'altra presa o porta dello switch.
- Se il PC è collegato a una presa a muro e il problema resta, chiedere ai tecnici di rete dell'ateneo se la porta è gigabit.

### Riavvii e sospensione
- **Il fermo di 14,5 h dell'11/11 è stato causato da Windows Update.** Riavvio forzato alle 20:30 per KB5132061, tre riavvii fino alle 20:33, subito dopo la fine delle ore attive (8–20). Lo script di ripresa non sopravvive a un riavvio.
- **Oggi:** nessun riavvio richiesto da Windows Update. Sospensione e ibernazione su alimentazione da rete: **mai**.
- **Consiglio (azione manuale tua):** Impostazioni → Windows Update → **Sospendi aggiornamenti per 1 settimana**, fino alla fine della campagna.

### Monitor
Il monitor `watch_v2_dense_eta.ps1` (PID 25576) rilegge `stream_progress.json` e può far fermare la campagna. È la causa probabile dell'arresto del 12/11 alle 12:35. **Va chiuso** e sostituito da `watch_v2_dense_safe.ps1`, che non apre mai quel file.

### Ripresa
`scripts\verify_r1_resume.py`, eseguito il 12/11 alle 12:45 con la campagna attiva, ha dato **RIPRESA SICURA**:
- 75/273 tile complete e valide: identità uguale al codice attuale, SHA256 dei `tile.npz` uguale ai marker, conteggi uguali ai pixel della geometria;
- checkpoint Medium verificato;
- tile in corso `calabria_sila/grid_16.65_39.35`, con 164 giorni Sentinel già salvati.

**Perché la ripresa è sicura**, dal codice r1:
- ogni tile completa è scritta in modo atomico (`tile.npz` con fsync, poi il marker `tile.json`) e viene riletta prima di cancellare i temporanei;
- alla ripresa ogni tile con marker è verificata (identità e SHA256) e saltata;
- la tile interrotta riparte dai giorni Sentinel e dai blocchi codificati già salvati;
- un file senza marker non è mai considerato completo e viene rifatto.

L'unico modo di invalidare le cache è cambiare uno dei 5 `CODE_FILES` della worktree di produzione, il checkpoint o gli argomenti di identità. La procedura non tocca nessuno dei tre.

## Variabili usate nei comandi
```powershell
$prod = 'C:\Users\brome\Desktop\BIOMAP_TESSERA_V2\biomap-tessera-v2-finetuning'
$dev  = 'C:\Users\brome\Desktop\BIOMAP_TESSERA_V2\biomap-v2-performance-dev'
$py   = "$prod\.venv\Scripts\python.exe"
```

## 0. Subito, senza rischi per il run
1. Nella finestra del vecchio monitor (`watch_v2_dense_eta.ps1`) premere **Ctrl+C**. Si chiude solo il monitor.
2. Avviare il monitor sicuro:
   ```powershell
   cd $dev; powershell -ExecutionPolicy Bypass -File scripts\watch_v2_dense_safe.ps1
   ```
3. Sospendere Windows Update per 1 settimana, dalle Impostazioni.
4. (Facoltativo) Misurare il traffico attuale per il confronto, 10 minuti in sola lettura:
   ```powershell
   cd $dev; powershell -ExecutionPolicy Bypass -File scripts\measure_net.ps1 -Minutes 10
   ```

## 1. Fermare il run subito dopo una tile completata
```powershell
cd $dev
powershell -ExecutionPolicy Bypass -File scripts\stop_r1_after_tile.ps1            # prova: mostra PID e ultima tile
powershell -ExecutionPolicy Bypass -File scripts\stop_r1_after_tile.ps1 -Execute   # attende la tile e ferma
```
- Lo script aspetta che `stream_log.jsonl` cresca, il che accade solo quando una tile è stata resa persistente.
- Poi termina **prima** lo script di ripresa, così non rilancia, e subito dopo i due processi python.
- Si perdono al massimo pochi secondi della tile successiva.
- L'attesa può durare fino a circa 20 minuti se la tile in corso è grande. **Ctrl+C** durante l'attesa annulla senza fermare nulla.
- Esito atteso: `FERMATO alle …` e nessun processo rimasto.

## 2. Verificare che le tile completate siano persistenti
```powershell
cd $dev
& $py scripts\verify_r1_resume.py --reference F:\biomap_dense_public\reference --work-root F:\biomap_dense_public\work --production-repo $prod --checkpoint C:\Users\brome\Desktop\BIOMAP_TESSERA_V2\data\weights\student_medium.pt
Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -like '*dense_campaign.py*' -or $_.CommandLine -like '*run_v2_dense_stream*' } | Select-Object ProcessId, CommandLine
```
**Esito atteso:** `RIPRESA SICURA`, codice di uscita 0, `tiles_complete_verified` pari all'ultimo numero visto più 1, `problems: []`, nessun processo elencato.
**Se compare un problema: non riavviare.**

(Facoltativo, a campagna ferma) Velocità reale verso Planetary Computer **prima** del cambio, circa 384 MB:
```powershell
& $py scripts\net_speed_mpc.py --catalog F:\biomap_dense_public\work\api\belluno_dolomiti\grid_11.95_45.95\sentinel-2-l2a_edge_bbox.json --approved
```

## 3. Sostituire il cavo o cambiare porta
- Cavo Cat6 (o Cat5e) nuovo e integro, possibilmente corto, diretto dalla scheda alla presa o allo switch.
- Se possibile provare un'altra presa o porta. Sugli switch il colore del LED spesso indica la velocità: consultare il modello.
- Scollegare il vecchio cavo **dopo** il passo 2, collegare il nuovo e attendere 10–20 secondi per la negoziazione.

## 4. Verificare che il collegamento sia a 1 Gbit/s
```powershell
Get-NetAdapter -Name Ethernet | Select-Object Name, LinkSpeed, FullDuplex, Status
Test-NetConnection planetarycomputer.microsoft.com -Port 443 | Select-Object ComputerName, TcpTestSucceeded
& $py "$dev\scripts\net_speed_mpc.py" --catalog F:\biomap_dense_public\work\api\belluno_dolomiti\grid_11.95_45.95\sentinel-2-l2a_edge_bbox.json --approved
```
- **Atteso:** `LinkSpeed 1 Gbps` (o 2.5 Gbps), `TcpTestSucceeded True`, `mbit_per_s` sensibilmente sopra il valore del passo 2.
- **Se resta a 100 Mbps:** provare un'altra porta o un altro cavo. Altrimenti chiedere ai tecnici dell'ateneo.
- **In ogni caso il passo 5 si può fare:** la campagna riprende anche a 100 Mbit/s.

## 5. Riavviare la campagna originale senza invalidare le cache
Prima il controllo, poi **lo stesso comando dell'avvio del 12/11 alle 10:33**:
```powershell
cd $dev
& $py scripts\verify_r1_resume.py --reference F:\biomap_dense_public\reference --work-root F:\biomap_dense_public\work --production-repo $prod
cd $prod
powershell -ExecutionPolicy Bypass -File scripts\run_v2_dense_stream.ps1 -Reference F:\biomap_dense_public\reference -WorkRoot F:\biomap_dense_public\work -IoThreads 16
```
- **Parametri identici a `last_invocation.json`:**
  - checkpoint `..\data\weights\student_medium.pt` (valore predefinito dello script);
  - `cuda:0`;
  - `--pixel-batch 256`;
  - `--chunk-size 4096`;
  - `--io-threads 16`.
- Lanciare dalla cartella di produzione `$prod`, senza nessuna modifica al codice.
- Non passare `-PixelBatch` o `-Checkpoint` diversi.

## 6. Controllare la ripresa e misurare il nuovo throughput
1. **Nei primi 2–3 minuti, nella finestra del run:**
   - nessuna riga `Cache incompatibile/corrotta`;
   - nessun traceback;
   - le tile già fatte vengono saltate senza stampare nulla.
2. **Monitor sicuro (passo 0.2):** stesse tile complete di prima. "In corso" mostra la tile interrotta con i giorni già salvati.
3. **Errori:** nessuna riga nuova in `stream_errors.jsonl`. Lettura sicura, perché è un file a sola aggiunta:
   ```powershell
   & $py -c "from pathlib import Path; import sys; sys.path.insert(0, r'$dev\src'); from biomap_v2.dense_fast.shared_read import read_text; print(read_text(r'F:\biomap_dense_public\work\stream_errors.jsonl').count(chr(10)), 'errori totali (prima del riavvio: 3)')"
   ```
4. **Dopo la prima tile nuova completata:** rilanciare `verify_r1_resume.py`. Atteso: +1 tile, ancora `RIPRESA SICURA`.
5. **Traffico, 10 minuti:**
   ```powershell
   powershell -ExecutionPolicy Bypass -File "$dev\scripts\measure_net.ps1" -Minutes 10
   ```
   Confronto: circa 56 Mbit/s a 100 Mbps.
6. **Dopo 2–3 ore, confronto per fascia di tile prima e dopo.** Sostituire l'ora UTC del riavvio:
   ```powershell
   & $py "$dev\scripts\audit_dense_log.py" --log F:\biomap_dense_public\work\stream_log.jsonl --since 2026-11-12T15:00:00
   ```
   Riferimento prima del cambio (`data/audit_r1_20261112.json`):
   - 232 s fissi per tile e 0,90 ms per pixel;
   - tile sotto 10.000 pixel: 212 s di acquisizione;
   - tile da 600.000 pixel in su: 767 s di acquisizione e 887 pixel al secondo.
