# Scopo: creare un ambiente di sviluppo CPU isolato, Python 3.12, su macOS/Windows/Linux.
# Fasi: controlli versione/percorso -> venv privata -> Torch ufficiale -> lock -> progetto -> pip check.
# Input: requirements-dev.txt e lock esistente; output: .venv-dev, senza dati o checkpoint.
# Parametri: --venv PERCORSO/.venv-dev; --dry-run mostra i comandi senza scrivere/installare.
# Esempio: python3.12 scripts/setup_dev.py; poi .venv-dev/bin/python -m pytest.
# Risorse: rete per wheel ufficiali, spazio per dipendenze; nessun uso GPU o ambiente di produzione.
# Ripresa: ripetibile nella sola venv privata; un errore interrompe subito la procedura.
import argparse
import os
from pathlib import Path
import subprocess
import sys
import venv

ROOT = Path(__file__).resolve().parents[1]
TORCH_VERSION = '2.10.0'
CPU_INDEX = 'https://download.pytorch.org/whl/cpu'


def environment_python(directory):
    return directory / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')


def install_commands(python, platform):
    prefix = [str(python), '-m', 'pip']
    torch = prefix + ['install', f'torch=={TORCH_VERSION}']
    # macOS wheels contain CPU/MPS support; the Linux/Windows index avoids CUDA dependencies.
    if platform != 'darwin':
        torch += ['--index-url', CPU_INDEX]
    return [
        prefix + ['install', 'pip==25.1.1'],
        torch,
        prefix + ['install', '-r', str(ROOT / 'requirements-dev.txt')],
        prefix + ['install', '-e', str(ROOT), '--no-deps', '--no-build-isolation'],
        prefix + ['check'],
    ]


def destination(value):
    path = Path(value).expanduser().absolute()
    if path.name != '.venv-dev' or path.is_symlink():
        raise ValueError('Usare una directory privata chiamata .venv-dev, senza symlink; .venv di produzione è esclusa')
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description='Setup privato BIOMAP CPU, nessun dato o peso')
    parser.add_argument('--venv', default=str(ROOT / '.venv-dev'))
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)
    if sys.version_info[:2] != (3, 12):
        parser.error('Richiesto Python 3.12: usare python3.12 (macOS/Linux) o py -3.12 (Windows)')
    directory = destination(args.venv)
    python = environment_python(directory)
    if directory.exists():
        if not (directory / 'pyvenv.cfg').is_file() or not python.is_file():
            raise ValueError('La directory esiste ma non è una venv completa: scegliere un altro percorso, senza cancellarla')
        subprocess.run([str(python), '-c', 'import sys; assert sys.version_info[:2] == (3,12)'], check=True)
    commands = install_commands(python, sys.platform)
    if args.dry_run:
        print('Venv prevista:', directory)
        for command in commands:
            print(subprocess.list2cmdline(command))
        return 0
    if not directory.exists():
        venv.EnvBuilder(with_pip=True).create(directory)
    for command in commands:
        print('+', subprocess.list2cmdline(command), flush=True)
        subprocess.run(command, cwd=ROOT, check=True)
    print('Pronto:', python)
    print('Import e hardware:', subprocess.list2cmdline([str(python), 'scripts/check_dev_environment.py']))
    print('Tutti i test CPU:', subprocess.list2cmdline([str(python), '-m', 'pytest']))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
