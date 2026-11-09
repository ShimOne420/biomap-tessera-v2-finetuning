# SCOPO: auditare e indicizzare le serie già scaricate, poi creare la cache puntuale E2.
# INPUT: --source-data-root .../msa_v1_liguria; OUTPUT esclusivamente in --work-root.
# COMANDI: --stage index (controlli), --stage cache (encoder frozen), --stage all (entrambi).
# RIPRESA: la cache riconosce le celle completate; nessuna ricodifica o acquisizione automatica.
from .settings import parser, command_context
from .source import build_index, indexed
from .cache import build_cache


# Questa CLI non richiama la preparazione legacy e non modifica il dataset di produzione.
def main():
    p = parser('E2: controllo in sola lettura e cache 5x5 da serie Sentinel esistenti')
    p.add_argument('--stage', choices=['index', 'cache', 'all'], default='index')
    args = p.parse_args()
    with command_context(args) as (cfg, legacy):
        frame, audit = build_index(cfg, legacy) if args.stage in {'index', 'all'} else indexed(cfg)
        if args.stage in {'cache', 'all'}:
            print(build_cache(cfg, legacy, frame, audit))


if __name__ == '__main__':
    main()
