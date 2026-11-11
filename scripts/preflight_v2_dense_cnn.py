# Scopo: controllare sola lettura gli artifact completati, senza avviare training/estrazione.
# Input: source-root con v2.parquet e marker extraction_complete.json; output: preflight.json.
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from biomap_dense_heads.cli import main
if __name__=='__main__':sys.exit(main(['preflight',*sys.argv[1:]]))
