# Specifica dataset denso

132.900 righe totali: partition=train per 127.909 sviluppo, test per 4.991 Liguria. cell_id globale immutato, AOI, cv_fold 0/1/2 oppure -1, lon/lat e bbox dal manifest originale. Target unitario GLOBIO overall MSA 2020 su EPSG:4326 10 arcsec. Pack richiede raster unitario originale e verifica valori contro V1; non applica divisioni a target già normalizzati. Input MSA-area va normalizzato e documentato prima del pack.

Feature mean_000..127. n_valid/n_total/valid_fraction sono conservati da V1. lookup.npy mappa cell_id locale geometry al row index globale del reference. geometry flat_indices identifica centri sul reticolo UTM V1; nessuna falsa immagine da un vettore medio e nessuna interpolazione dei 25 punti. Serie s2 [N,T,10], asc/desc [N,T,2], mask [N,T]; i legacy sono time-major e vengono trasposti. DOY reale limitato a 365; S2 ordine ufficiale; S1 intensity -> 10log10 -> (+50)*200 int16.

I contributi di bordo trunc/floor e duplicazioni storiche devono essere investigati prima di una dichiarazione di equivalenza fisica. Nessuna correzione unilaterale. Pixel privi di tutte le osservazioni bloccano inferenza, senza eliminare celle. Eccezioni 25 punti non trasferite. Hash, parametrizzazione AMP e checkpoint distinguono cache incompatibili.
