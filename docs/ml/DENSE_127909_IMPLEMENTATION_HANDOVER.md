# Handover implementazione densa

Leggere prima UNIVERSITY_RUNBOOK, poi ARCHITECTURE e DATASET_SPEC. Aggiunta una CLI indipendente e sette moduli campaign. Nessuna modifica al codice legacy di probing/encoder/preparazione o ai dati originali. Branch dense-127909, nessun merge automatico.

Implementato: manifest/fold, reference pack con controlli di conteggi, acquisizione STAC esplicita e cache giornaliere, lettura legacy, inferenza ufficiale congelata a chunk, riuso progressi compatibili, somme/conteggi atomici, controllo OOM, probing cinque regressori, refit, Liguria separata, confronto, grafici/report/Excel, source-root read-only e lock.

Test locali: suite precedente 86 casi passata; nuovi test sintetici verificano manifest, transazioni/cache, sorgente protetta, allineamento, chunking, OOM, configurazioni, parser e integrazione probing/export con estimator sintetico. Nessun test sintetico costituisce risultato MSA. Esito finale e CI sono nella consegna GitHub.

LIMITI APERTI: reference completa bloccata da target iCloud non leggibile; non pubblicato asset multi-AOI. Validazione fisica cross-tile/maschere su tutte le AOI ancora da completare. RAM Windows implementata tramite API, non verificata sul PC reale. Nessuna validazione API/GPU end-to-end sulle 17 AOI. Bootstrap spaziale e analisi dettagliata di shift non inclusi in questa prima implementazione. Non chiamare questa consegna certificazione production-ready o campagna già eseguita.

Recupero: materializzare target originali, rieseguire pack, verificare equivalenza fisica, poi prepare/validate/profile. Caches incompatibili bloccano; non cancellarle per forzare un risultato. Cambi scientifici richiedono work-root nuova. Un crash tra payload e marker produce cache incompleta ricostruibile; dopo marker hash errato è errore. Export non riaddestra e blocchi Excel sono espliciti.
