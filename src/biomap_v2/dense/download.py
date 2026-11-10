# SCOPO: scaricare soltanto il piccolo pacchetto portabile pubblicato nella Release GitHub.
# Il codice e i dati hanno canali distinti: git pull aggiorna gli script, fetch-bundle i dati.
# Controlla SHA256 prima di estrarre e non consente sovrascritture o percorsi ZIP esterni.
import json
from pathlib import Path
import shutil
import zipfile
import requests
from biomap_v2.common import digest
from .bundle import load_bundle


# Nessun loop di tentativi: un errore di rete/hash viene mostrato e il download può essere rilanciato.
def fetch_bundle(specification, destination):
    spec = json.loads(Path(specification).read_text(encoding='utf-8'))
    out = Path(destination).resolve()
    if out.exists():
        if (out/'bundle.json').is_file() and digest(out/'bundle.json') == spec['manifest_sha256']:
            load_bundle(out)
            print(f'Pacchetto già presente e verificato: {out}', flush=True)
            return out
        raise FileExistsError('Destinazione esistente diversa dal pacchetto pubblicato')
    out.parent.mkdir(parents=True, exist_ok=True)
    archive = out.with_suffix('.download.zip')
    if not archive.is_file() or digest(archive) != spec['zip_sha256']:
        temporary = archive.with_suffix('.partial')
        with requests.get(spec['url'], stream=True, timeout=(20, 180)) as response:
            response.raise_for_status()
            count = 0
            with temporary.open('wb') as stream:
                for chunk in response.iter_content(1024**2):
                    count += len(chunk)
                    if count > spec['zip_bytes']:
                        raise ValueError('Download più grande del pacchetto dichiarato')
                    stream.write(chunk)
        if count != spec['zip_bytes'] or digest(temporary) != spec['zip_sha256']:
            raise ValueError('Dimensione/SHA256 download discordanti: nessun file estratto')
        temporary.replace(archive)
    staging = out.with_name(out.name+'.extracting')
    if staging.exists():
        raise FileExistsError(f'Estrazione precedente presente: ispezionare {staging} o scegliere un nuovo --out')
    staging.mkdir()
    with zipfile.ZipFile(archive) as z:
        for name in z.namelist():
            if '/' in name or '\\' in name or ':' in name or name in {'', '.', '..'}:
                raise ValueError('Percorso ZIP non valido')
        z.extractall(staging)
    if digest(staging/'bundle.json') != spec['manifest_sha256']:
        raise ValueError('Manifest estratto diverso da quello pubblicato')
    frame, _ = load_bundle(staging)
    if len(frame) != spec['n_cells']:
        raise ValueError('Numero di celle diverso dal pacchetto pubblicato')
    staging.replace(out)
    print(f'Pacchetto pronto: {out}', flush=True)
    return out
