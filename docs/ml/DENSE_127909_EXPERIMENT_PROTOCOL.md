# Protocollo frozen

Tre fold AOI originali; nessuna partizione casuale nuova. Cinque regressori: DummyMean, Ridge scaler+alpha100, HistGB90/15/.05/L2=1, XGBoost400/depth2/.05, MLP256-128/max_iter500/early_stopping interno training. Seed20260925; nessuna ricerca. Early stopping tabellare usa solo dati training, non la validation geografica esterna. Scaler esclusivamente training.

OOF sulle 127.909 celle; selezione mediante media RMSE sulle 17 AOI. Refit sviluppo e benchmark Liguria separato. V1 probing esplicito, con medesimi parametri. Confronto solo dati appaiati e reference/fold/parametri coincidenti; differenze V2−V1 non attribuite automaticamente ai soli pesi se precisione/preprocessing differiscono. Umbria nord esclusa.

MAE, RMSE, MAPE target nonzero, bias predetto−osservato, R² e Pearson; null per correlazioni costanti. Valori fuori [0,1] non clipped silenziosamente. MAPE esclude zero e ne registra conteggio. MAE×100 = punti percentuali scala. OOF micro e metriche per AOI; macro salvate separatamente. Liguria già osservata non è test cieco. Non dichiarare vincitore senza risultati del run.
