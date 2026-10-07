# Scopo: unica interfaccia condivisa dai tre script principali, senza codice duplicato.
# Fasi: parsing, YAML/override, controllo GPU, dispatch verso prepare/probe/train/evaluate.
# Input: argomenti CLI; output: dati e run nella data-root scelta dall'utente.
# Parametri: --help elenca azioni e opzioni; --set sezione.chiave=valore modifica YAML.
# Esempio: probe_bii.py run --data-root E:\BioMAP --set probing.ridge_alpha=10.
# Risorse: import pesanti soltanto dopo parsing; --help funziona senza GPU.
# Ripresa: --resume last.pt per training, prepare riprende cache; errori interrompono il comando.
import argparse
from .common import config, doctor


def main(indicator=None,finetune=False):
    parser=argparse.ArgumentParser(description='BioMAP TESSERA V2: target modellati, split geografici, Windows/CUDA')
    parser.add_argument('action',choices=['doctor','download-weights','prepare','run','evaluate','export-embeddings'])
    if finetune:parser.add_argument('--indicator',choices=['msa','bii'],required=True)
    parser.add_argument('--config',help='Configurazione YAML alternativa')
    parser.add_argument('--data-root',help='Cartella dati locale o SSD')
    parser.add_argument('--set',action='append',default=[],metavar='KEY=VALUE',help='Override YAML, ripetibile')
    parser.add_argument('--device',help='cuda, cuda:0 oppure cpu per test')
    parser.add_argument('--checkpoint',help='Checkpoint ufficiale o best.pt adattato; caricare solo file fidati')
    parser.add_argument('--limit',type=int,help='prepare: limite celle, distribuito fra tutti gli split')
    parser.add_argument('--smoke',action='store_true',help='run/evaluate: usa cells_smoke.csv; risultati tecnici, non benchmark')
    parser.add_argument('--target-raster',help='Raster locale; MSA unitless, BII nelle unità dichiarate nel YAML')
    parser.add_argument('--import-cache',help='Radice di una cache esportata da questo progetto, con manifest')
    parser.add_argument('--stage',choices=['head','partial','full','all'],default='all',help='all = warmup, controllo frozen, partial; full esplicito')
    parser.add_argument('--resume',help='Percorso last.pt: riprende la singola fase interrotta')
    parser.add_argument('--run-dir',help='evaluate: cartella run contenente predictions.csv')
    args=parser.parse_args()
    indicator=args.indicator if finetune else indicator
    if args.device:args.set.append('model.device='+args.device)
    cfg=config(indicator,args)
    if finetune and args.action=='run' and args.checkpoint:
        cfg['model']['checkpoint']=str(__import__('pathlib').Path(args.checkpoint).resolve())
    if args.action=='doctor':return doctor(cfg,require_cuda=cfg['model']['device'].startswith('cuda'))
    if args.action=='download-weights':
        from .encoder import weights
        print(weights(cfg));return
    if args.action=='prepare':
        from .data import prepare
        print(prepare(cfg,args));return
    if args.action=='run':
        if cfg['model']['device'].startswith('cuda'):doctor(cfg)
        from .experiments import probe,train
        return train(cfg,args) if finetune else probe(cfg,args)
    if args.action=='evaluate':
        if not args.run_dir and not args.checkpoint:parser.error('evaluate richiede --run-dir o --checkpoint')
        if args.checkpoint and not args.run_dir:args.run_dir='.'
        from .experiments import evaluate
        return evaluate(cfg,args)
    if args.action=='export-embeddings':
        from .data import prepared
        from .encoder import embeddings
        features,path=embeddings(cfg,prepared(cfg,args.smoke),args.checkpoint)
        print(f'{len(features)} embedding cella, pesi {path}');return features
