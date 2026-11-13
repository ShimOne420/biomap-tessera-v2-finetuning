# Scopo: il v2.parquet reale scritto da r1 (campaign/streaming.py) e da r2-fast (dense_fast/pipeline.py) ha colonne
# esplicite senza reference_index: il reader deve accettarlo, senza richiedere colonne che solo la reference possiede.
import numpy as np
import pandas as pd
import pytest

from biomap_spatial.sources import V2Source, prepare_index, R1, R2
from biomap_spatial.utils import digest, save_json
from spatial_fixtures import make_artifacts

# Colonne effettivamente scritte da streaming.stream() e dense_fast.pipeline.finish() (stesso elenco in entrambi).
REAL_COLUMNS = ['cell_id', 'aoi_id', 'partition', 'cv_fold', 'lon', 'lat', 'target', 'n_valid', 'n_total', 'valid_fraction']


@pytest.mark.parametrize('protocol', [R1, R2])
def test_real_extraction_parquet_without_reference_index_is_accepted(tmp_path, protocol):
    spec, ref, work, out = make_artifacts(tmp_path, protocol)
    v2 = pd.read_parquet(work / 'v2.parquet')
    v2 = v2[REAL_COLUMNS + [c for c in v2 if c.startswith('mean_')]]
    v2.to_parquet(work / 'v2.parquet', index=False)
    marker = work / 'extraction_complete.json'
    import json
    value = json.loads(marker.read_text()); value['sha256'] = digest(work / 'v2.parquet'); save_json(marker, value)
    source = V2Source(spec, out, False)
    frame = prepare_index(source)
    assert len(frame) == 8 and frame.n_valid.eq(4).all()
    source.close()
