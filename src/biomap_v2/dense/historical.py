# SCOPO: conservare il controllo opzionale del modello V1 storico senza riaddestrarlo.
# È un riferimento di trasferimento sviluppo→Liguria, distinto dalla CV dentro Liguria.
# Caricare joblib soltanto da una sorgente fidata, come gli artefatti del proprio progetto.
import json
from pathlib import Path
import joblib
from biomap_v2.common import digest
from .evaluation import full_metrics
from .paired import features


# L'assenza del flag non richiede di copiare il vecchio modello; un flag errato dà errore.
def evaluate_reference(frame, model_path=None, reference_path=None):
    if not model_path:
        return None
    payload = joblib.load(model_path)
    model = payload['model'] if isinstance(payload, dict) else payload
    result = {'role': 'historical_development_to_liguria_not_within_AOI_CV',
              'model_sha256': digest(model_path), 'metrics': full_metrics(frame.target, model.predict(features(frame, 'v1')))}
    if reference_path:
        reference = json.loads(Path(reference_path).read_text(encoding='utf-8'))['metrics']['HistGB']
        deltas = [abs(result['metrics'][key]-reference[key]) for key in ['mae', 'rmse', 'mape_pct', 'bias', 'r2']
                  if reference.get(key) is not None and result['metrics'].get(key) is not None]
        result['max_abs_reference_delta'] = max(deltas) if deltas else None
    return result
