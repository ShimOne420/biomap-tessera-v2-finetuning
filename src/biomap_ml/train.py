# SCOPO: avviare E2 Frozen CNN sui tre fold storici e fare il refit completo.
# INPUT: indice e cache già verificati; OUTPUT: work-root/runs/<run-id>, senza test implicito.
# ESEMPIO: python -m biomap_ml.train --source-data-root ... --work-root ... --run-id e2_msa_seed42
# RIPRESA: aggiungere --resume; riparte dall'ultima epoca completata senza cambiare split o seed.
from .settings import parser, command_context
from .campaign import train_campaign


# E4/E5 non vengono avviati o simulati da questo comando: la fase attuale consegna E2.
def main():
    p = parser('E2 Frozen CNN: tre fold geografici e refit, senza fine-tuning encoder')
    p.add_argument('--run-id', required=True)
    p.add_argument('--phase', choices=['cv', 'refit', 'all'], default='all')
    p.add_argument('--resume', action='store_true')
    args = p.parse_args()
    with command_context(args) as (cfg, _):
        print(train_campaign(cfg, args.run_id, args.phase, args.resume))


if __name__ == '__main__':
    main()
