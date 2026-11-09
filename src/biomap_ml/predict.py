# SCOPO: esportare le predizioni liguri già verificate sulla griglia nativa GLOBIO.
# INPUT: run-id dopo evaluate; OUTPUT: geotiff/ con COG, errori, copertura e metadati.
# Non ricampiona a 10 metri e non stima celle mancanti; NoData rimane -9999.
import json
from pathlib import Path
import pandas as pd
from biomap_v2.common import digest
from .settings import parser, command_context, run_directory
from .products import export_rasters


# Il checksum impedisce di esportare un CSV modificato manualmente dopo la valutazione.
def main():
    p = parser('E2: GeoTIFF/COG delle predizioni Liguria già registrate')
    p.add_argument('--run-id', required=True)
    args = p.parse_args()
    with command_context(args) as (cfg, _):
        out = run_directory(cfg, args.run_id)
        final = json.loads((out/'final.json').read_text(encoding='utf-8'))
        if not final['test_evaluated'] or digest(out/'test_predictions.csv') != final['test_predictions_sha256']:
            raise ValueError('Eseguire prima evaluate: predizioni test non verificate')
        print(export_rasters(cfg, out, pd.read_csv(out/'test_predictions.csv')))


if __name__ == '__main__':
    main()
