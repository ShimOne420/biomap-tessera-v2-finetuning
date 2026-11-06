# SCOPO
#   Unica interfaccia a riga di comando condivisa dai tre script principali, senza codice duplicato.
#
# COME VIENE RICHIAMATO
#   Non si esegue da solo: scripts/probe_msa.py, probe_bii.py e lo script di fine-tuning chiamano main().
#     python scripts/probe_msa.py --help
#     python scripts/probe_msa.py prepare --config configs/msa_v1_liguria.yaml --data-root ..\data
#     python scripts/probe_msa.py prepare --config ... --data-root ... --parallel-aois 15
#     python scripts/probe_bii.py run --data-root E:\BioMAP --set probing.ridge_alpha=10
#
# LOGICA, IN ORDINE
#   1. Legge gli argomenti, poi carica YAML e override con common.config().
#   2. Esegue il controllo GPU solo nelle azioni che la usano.
#   3. Passa all'azione richiesta: doctor, download-weights, download-targets, diagnose-targets, prepare,
#      run, evaluate oppure export-embeddings.
#   4. Per prepare: con --parallel-aois chiama parallel_prepare.run_parallel, altrimenti data.prepare.
#
# OPZIONI DI prepare
#   --limit N: prova su N celle totali.
#   --aoi AREA: una sola area, senza scrivere cells.csv.
#   --max-new-gib X: con --aoi, spazio massimo aggiuntivo del processo.
#   --parallel-aois N: un processo per area, al massimo N insieme, poi validazione finale.
#   --aois a,b: con --parallel-aois, solo queste aree, senza validazione finale.
#   --stop-after-minutes M: con --parallel-aois, si ferma dopo M minuti conservando i checkpoint.
#
# INPUT E OUTPUT
#   Input: argomenti CLI. Output: dati e run nella data-root scelta dall'utente.
#
# RISORSE
#   Import pesanti soltanto dopo il parsing; --help funziona senza GPU.
#
# RIPRESA
#   --resume last.pt per il training; prepare riprende da solo; un errore interrompe il comando.
import argparse
from .common import config, doctor


def main(indicator=None,finetune=False):
    parser=argparse.ArgumentParser(description='BioMAP TESSERA V2: target modellati, split geografici, Windows/CUDA')
    parser.add_argument('action',choices=['doctor','download-weights','download-targets','diagnose-targets','prepare','run','evaluate','export-embeddings'])
    if finetune:parser.add_argument('--indicator',choices=['msa','bii'],required=True)
    parser.add_argument('--config',help='Configurazione YAML alternativa')
    parser.add_argument('--data-root',help='Cartella dati locale o SSD')
    parser.add_argument('--set',action='append',default=[],metavar='KEY=VALUE',help='Override YAML, ripetibile')
    parser.add_argument('--device',help='cuda, cuda:0 oppure cpu per test')
    parser.add_argument('--checkpoint',help='Checkpoint ufficiale o best.pt adattato; caricare solo file fidati')
    parser.add_argument('--limit',type=int,help='prepare: limite celle, distribuito fra tutti gli split')
    parser.add_argument('--aoi',help='prepare: elabora una sola area, senza scrivere cells.csv')
    parser.add_argument('--parallel-aois',type=int,metavar='N',help='prepare: un processo per area, massimo N insieme; poi validazione finale')
    parser.add_argument('--aois',help='prepare --parallel-aois: elenco di aree separate da virgole, senza validazione finale')
    parser.add_argument('--stop-after-minutes',type=float,help='prepare --parallel-aois: ferma i processi dopo questo tempo, conservando i checkpoint')
    parser.add_argument('--max-new-gib',type=float,help='prepare --aoi: spazio massimo aggiuntivo per questo processo')
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
    if finetune and cfg['evaluation'].get('mode')=='historical_groupkfold':
        parser.error('Il profilo storico GroupKFold è riservato al probing; il fine-tuning richiede validation fissa')
    if args.action in {'download-targets','diagnose-targets'}:
        from .common import cells
        from .target_data import ensure_target,locate_target,diagnose
        frame=cells(cfg)
        path=ensure_target(cfg,frame,args.target_raster) if args.action=='download-targets' else locate_target(cfg,args.target_raster)
        diagnosis,directory=diagnose(cfg,frame,path)
        if not diagnosis.valid_target.all():
            raise ValueError(f'Target incompleti: nessuna esclusione applicata; leggere {directory}')
        return directory
    if args.action=='doctor':return doctor(cfg,require_cuda=cfg['model']['device'].startswith('cuda'))
    if args.action=='download-weights':
        from .encoder import weights
        print(weights(cfg));return
    if args.action=='prepare':
        if args.parallel_aois:
            if args.aoi or args.limit:parser.error('--parallel-aois non si combina con --aoi o --limit')
            from .parallel_prepare import run_parallel
            return run_parallel(cfg,args)
        if args.aois or args.stop_after_minutes:parser.error('--aois e --stop-after-minutes richiedono --parallel-aois')
        if args.max_new_gib is not None:
            if not args.aoi:parser.error('--max-new-gib richiede --aoi')
            cfg['storage']['max_new_gib']=args.max_new_gib
        from .data import prepare
        print(prepare(cfg,args));return
    if args.action=='run':
        if cfg['model']['device'].startswith('cuda'):doctor(cfg)
        from .experiments import probe,train
        return train(cfg,args) if finetune else probe(cfg,args)
    if args.action=='evaluate':
        if cfg['evaluation'].get('mode')=='historical_groupkfold':
            if not args.run_dir or args.checkpoint:parser.error('Valutazione benchmark: indicare --run-dir, senza --checkpoint')
            from .benchmark import evaluate_historical
            return evaluate_historical(cfg,args)
        if not args.run_dir and not args.checkpoint:parser.error('evaluate richiede --run-dir o --checkpoint')
        if args.checkpoint and not args.run_dir:args.run_dir='.'
        from .experiments import evaluate
        return evaluate(cfg,args)
    if args.action=='export-embeddings':
        from .data import prepared
        from .encoder import embeddings
        features,path=embeddings(cfg,prepared(cfg,args.smoke),args.checkpoint)
        print(f'{len(features)} embedding cella, pesi {path}');return features
