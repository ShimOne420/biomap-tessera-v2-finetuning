# SCOPO
#   Pacchetto dell'estrazione V2 densa ottimizzata (protocollo r2-fast). Stessa semantica numerica
#   della campagna r1 (campaign/streaming.py), con lettura remota, coordinate e pipeline più veloci.
# COME VIENE RICHIAMATO
#   Non si esegue da solo: lo usano scripts/dense_fast.py e tests/test_dense_fast.py.
# LOGICA IN ORDINE
#   identity (protocollo e codice) -> io_safe (scritture atomiche) -> coords (righe/colonne esatte)
#   -> handles (dataset aperti per thread) -> reader (lettore con cache) -> acquisition (giorni Sentinel)
#   -> pipeline (acquisizione e GPU sovrapposte) -> spatial (mappe per pixel opzionali).
# INPUT E OUTPUT
#   Nessuno a livello di pacchetto.
# PARAMETRI, RISORSE, RIPRESA
#   Vedere i singoli moduli.
