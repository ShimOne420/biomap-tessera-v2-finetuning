# SCOPO: valutare una volta il checkpoint finale su Liguria e aggiornare CSV/grafici/Excel.
# INPUT: run-id con CV/refit conclusi; OUTPUT: test_predictions, metriche, report, Excel separato.
# Un secondo comando riutilizza le predizioni registrate, senza rifit o selezione sul test.
from .settings import parser, command_context
from .campaign import evaluate_campaign


# Gli errori Excel rimangono visibili: nessun file _latest sostituisce il workbook canonico.
def main():
    p = parser('E2: benchmark Liguria già osservato e registro risultati separato')
    p.add_argument('--run-id', required=True)
    args = p.parse_args()
    with command_context(args) as (cfg, _):
        print(evaluate_campaign(cfg, args.run_id))


if __name__ == '__main__':
    main()
