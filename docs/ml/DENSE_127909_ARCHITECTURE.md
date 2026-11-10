# Architettura della campagna

scripts/dense_campaign.py -> campaign.cli -> reference/state -> inputs/extraction -> probing -> reporting.

reference congela V1/target/geometrie/lookup locale-globale. state valida il manifest e gestisce payload+marker atomici. inputs riutilizza array time-major NPY/GZIP oppure interroga STAC e legge COG tramite Reader; converte nel formato punto-major dell'inferenza ufficiale. extraction verifica Medium, congela i pesi e salva somme float64/conteggi per chunk. Il consolidamento rilegge chunk verificati, senza mutare un accumulatore persistente: una ripresa non raddoppia i conteggi. probing usa i tre fold originali, salva fit e fingerprint. reporting ricostruisce tabelle, grafici e workbook.

Work-root e reference/sorgenti sono separate; FileLock per l'intera invocazione. I vecchi comandi e i moduli E2 non sono modificati. Non viene fatto fine-tuning. File di implementazione sotto 300 righe, funzioni commentate. La nuova facciata è separata per non cambiare il significato di all del benchmark ligure.
