# Audit denso 127.909 — 10 novembre 2026

Verificati main 9a2946b e manifest SHA LF b3649c5a225f75891b190378470d16d85ff43bdd66c3ae7e94363f4192568d6c. Il parquet storico multi_aoi_training_dataset.parquet ha 127.909 righe e 128 medie, conteggi pixel e target: 95.641.746 contributi salvati. Il codice V1 aggrega tile int8 dequantizzate con scale finite. Non è un campione 5×5. La pipeline V2 precedente msa_v1_liguria.yaml usa invece 25 punti. Il bundle ligure pubblicato contiene solo 4.991 celle e non sostituisce lo sviluppo.

17 AOI e fold sono in configs/msa_v1_liguria_reference.json. I conteggi validation sono 38.808/44.895/44.206; training 89.101/83.014/83.703. Test Liguria escluso dalla CV. Serie dense: Liguria 10/10 tile, Abruzzo 14 complete su 16 e una parziale; altre 16 AOI senza array legacy acquisiti nel percorso locale ispezionato.

BLOCCO REALE: pack-reference tentato, fallito leggendo il target Abruzzo GLOBIO conservato come dataless iCloud. Nessuna reference completa né nuova Release multi-AOI è stata prodotta. È necessario materializzare gli originali e completare il pack. Controlli fisici di maschere V1 e sovrapposizioni cross-tile su tutte le AOI non sono ancora dimostrati; il controllo conteggi/geometry non li sostituisce. La consegna software non è una dimostrazione di equivalenza reale completa.

Fatto: Liguria era stata verificata fisicamente nel percorso precedente. Ipotesi non certificata: geometrie legacy di tutte le AOI rappresentano esattamente i supporti storici. Non inventare nuovi risultati. Il pack conserva metadati e segnala conteggi discordanti; la validazione fisica integrale resta un gate scientifico aperto.
