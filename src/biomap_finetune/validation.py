# Scopo: prove esplicite di architettura, parita input e gradienti; nessuna certificazione GPU implicita.
# Fasi: checkpoint SHA -> forward ufficiale/split -> backward/step -> confronto pixel r2 su raw acquisiti.
# Input: checkpoint locale e nuove root; output: audit JSON con copertura e limiti, mai scritture legacy.
# Parametri: tolleranze storiche 2e-5, last_n_blocks; uso doctor --verify-inputs dopo acquire.
# Risorse: pochi pixel CPU/CUDA; ripresa: prova legata al dataset e checkpoint, cache finale non e prefisso.
import json
from pathlib import Path
import time
import numpy as np
import torch
from .encoder import load_medium,PartialMedium
from .io import digest,save_json,signature,code_identity,checked_roots
from .data import load_arrays
from biomap_v2.common import WEIGHTS_SHA256


def changed_parameters(model,before):
    changed=[];illegal=[];gradients=[]
    for name,p in model.named_parameters():
        if p.grad is not None and bool((p.grad!=0).any()):gradients.append(name)
        if not torch.equal(p.detach().cpu(),before[name]):
            changed.append(name)
            if not p.requires_grad:illegal.append(name)
    if illegal:raise RuntimeError('Pesi congelati modificati: '+','.join(illegal))
    if not changed:raise RuntimeError('Nessun parametro aggiornato')
    return {'changed_names':changed,'nonzero_gradient_names':gradients,'frozen_changed':illegal}


def architecture_probe(cfg):
    model,args=load_medium(cfg['encoder']['checkpoint']);device=torch.device(cfg['execution']['device'])
    adapter=PartialMedium(model,cfg['finetuning']['last_n_blocks'],cfg['finetuning']['train_output_projection'],
        cfg['finetuning']['train_normalization'],activation_checkpointing=cfg['training']['activation_checkpointing']).to(device).eval()
    s2=torch.ones(2,8,11,device=device);s1=torch.ones(2,8,3,device=device)
    s2[:,:,-1]=torch.arange(1,9,device=device);s1[:,:,-1]=torch.arange(1,9,device=device)
    if device.type=='cuda':torch.cuda.reset_peak_memory_stats(device);torch.cuda.synchronize(device)
    started=time.perf_counter()
    with torch.no_grad():
        original=model(s2,s1);split=adapter(s2,s1)
        torch.testing.assert_close(split,original,rtol=2e-5,atol=2e-5)
    before={n:p.detach().cpu().clone() for n,p in adapter.named_parameters()}
    adapter.train();optimizer=torch.optim.AdamW([p for p in adapter.parameters() if p.requires_grad],lr=1e-5)
    # Non-constant objective (embedding mean/squared norm are almost constant after final LayerNorm).
    objective=(adapter(s2,s1)*torch.linspace(-1,1,128,device=device)).sum();objective.backward()
    optimizer.step();changes=changed_parameters(adapter,before)
    for modality in ['s2_backbone','s1_backbone']:
        if not any(modality+'.transformer_encoder' in n for n in changes['changed_names']):raise RuntimeError('Suffisso senza aggiornamenti: '+modality)
    if device.type=='cuda':torch.cuda.synchronize(device)
    return {'status':'TECHNICAL_PROBE_PASSED','checkpoint':WEIGHTS_SHA256,'device':str(device),'parameters':sum(p.numel() for p in model.parameters()),
        'policy':adapter.policy(),'forward_max_abs_error':float((split-original).abs().max()),'changes':changes,
        'seconds':time.perf_counter()-started,'cuda_peak_bytes':torch.cuda.max_memory_allocated(device) if device.type=='cuda' else None,
        'limitation':'2 pixels x 8 tokens, not a sustainable cell microbatch benchmark; synthetic inputs only'}


def verify_inputs(cfg):
    checked_roots(cfg);root=Path(cfg['paths']['raw_root']);frozen=cfg['paths'].get('frozen_pixels')
    if not frozen:raise ValueError('Impostare frozen_pixels: confronto con pixel r2 richiesto')
    frozen=Path(frozen);campaign=json.loads((frozen/'extraction_complete.json').read_text(encoding='utf-8'))
    if campaign.get('complete_campaign') is not True or campaign.get('amp') is not False or campaign.get('method') not in {
        'dense127909-stream-legacy-exact-r1','dense127909-stream-legacy-exact-r2-fast'}:raise ValueError('Campagna frozen completa FP32 richiesta')
    if digest(frozen/'v2.parquet')!=campaign['sha256']:raise ValueError('Manifest/parquet frozen incoerenti')
    manifest=json.loads((root/'dataset.json').read_text(encoding='utf-8'))
    model,_=load_medium(cfg['encoder']['checkpoint']);model=PartialMedium(model,2).to(cfg['execution']['device']).eval()
    records=[]
    # One selected block per tile; at most 8 pixels per tile. Explicit coverage, not exhaustive identity.
    seen=set()
    for block in manifest['blocks']:
        tile=(block['aoi'],block['tile'])
        if tile in seen:continue
        seen.add(tile);arrays,_=load_arrays(root/'blocks'/block['key'],block['identity'],mmap=True)
        take=np.arange(min(len(arrays['s2']),8));raw={k:np.array(arrays[k][take] if k in {'s2','mask','asc','desc'} else arrays[k]) for k in ['s2','mask','s2_doys','asc','asc_doys','desc','desc_doys']}
        path=frozen/'pixel_embeddings'/block['aoi']/(block['tile']+'.npy')
        marker=json.loads(path.with_suffix('.json').read_text(encoding='utf-8'))
        geometry=Path(cfg['paths']['reference'])/block['aoi']/block['tile']/'geometry.npz'
        if marker['geometry']!=digest(geometry) or marker['sha256']!=digest(path):raise ValueError('Pixel frozen/geometry corrotti')
        baseline=np.load(path,mmap_mode='r',allow_pickle=False)
        if baseline.dtype!=np.float32 or baseline.ndim!=2 or baseline.shape[1]!=128:raise ValueError('Formato pixel frozen non supportato')
        expected=np.array(baseline[arrays['positions'][take]]);del baseline
        with torch.no_grad():actual=model.encode_raw(raw,cfg['training']['pixel_batch']).cpu().numpy()
        np.testing.assert_allclose(actual,expected,rtol=cfg['validation']['rtol'],atol=cfg['validation']['atol'])
        records.append({'aoi':tile[0],'tile':tile[1],'pixels':len(take),'max_abs_error':float(np.max(np.abs(actual-expected)))})
        del arrays
    proof={'status':'SAMPLED_INPUT_PARITY_PASSED','dataset_identity':manifest['identity'],'checkpoint':WEIGHTS_SHA256,
        'frozen_campaign_marker_sha256':digest(frozen/'extraction_complete.json'),
        'code':code_identity(),'rtol':cfg['validation']['rtol'],'atol':cfg['validation']['atol'],'coverage':records,
        'limitation':'Up to 8 selected pixels/tile; exact raw historical operations reused, not an exhaustive bitwise proof on new platform'}
    save_json(root/'input_validation.json',proof);return proof


def require_input_proof(cfg,dataset):
    if cfg['experiment']['profile']=='synthetic':return
    path=Path(cfg['paths']['raw_root'])/'input_validation.json'
    if not path.is_file():raise ValueError('Eseguire doctor --verify-inputs prima del training')
    proof=json.loads(path.read_text(encoding='utf-8'))
    if (proof.get('status')!='SAMPLED_INPUT_PARITY_PASSED' or proof['dataset_identity']!=dataset['identity'] or
        proof['checkpoint']!=WEIGHTS_SHA256 or proof['code']!=code_identity() or
        proof['rtol']!=cfg['validation']['rtol'] or proof['atol']!=cfg['validation']['atol']):raise ValueError('Prova input assente/incompatibile con codice corrente')
