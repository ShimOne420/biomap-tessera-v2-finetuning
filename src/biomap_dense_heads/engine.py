# Scopo: addestrare solo le teste, con scaler training-only e checkpoint per epoca.
# Resume conserva modello/optimizer/RNG e non rilegge mai serie Sentinel o encoder.
import random
import time
from pathlib import Path
import joblib
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from biomap_v2.common import digest
from .models import make
from .progress import epoch_progress

# Seed e algoritmi deterministici; documentare le eventuali differenze fra hardware.
def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False

# Payload piccolo, atomico: contiene soltanto la testa, non TESSERA.
def save(path,payload):
    tmp=path.with_suffix('.partial');torch.save(payload,tmp);tmp.replace(path)

# Forward in mini-batch senza gradienti, sullo scaler già stimato dal training.
def predict(model,x,batch_size,device):
    model.eval();parts=[]
    with torch.inference_mode():
        for low in range(0,len(x),batch_size):parts.append(model(torch.as_tensor(x[low:low+batch_size],device=device)).cpu().numpy())
    return np.concatenate(parts)

# Criterio condiviso: media RMSE sulle AOI validation, non RMSE pooled.
def macro_rmse(y,p,aois):
    return float(np.mean([np.sqrt(np.mean((y[aois==a]-p[aois==a])**2)) for a in np.unique(aois)]))

# Refit fisso: validation=None, numero epoche deciso prima dal mediano dei fold.
def fit(x,y,aois,train,validation,name,output,cfg,out,device,progress,fingerprint,epochs=None,resume=False):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);seed_all(cfg['seed'])
    torch.set_num_threads(cfg['threads']);limit=epochs or cfg['epochs']
    model=make(name,output,cfg).to(device);optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
    # StandardScaler vede ESCLUSIVAMENTE i campioni del train corrente.
    scaler=StandardScaler().fit(x[train]);xt=scaler.transform(x[train]).astype(np.float32);yt=y[train].astype(np.float32)
    xv=scaler.transform(x[validation]).astype(np.float32) if validation is not None else None
    history=[];start=0;best=float('inf');best_epoch=0;stale=0
    last=out/'last.pt';identity={'fingerprint':fingerprint,'model':name,'output':output,'config':cfg,'limit':limit,
        'train_sha':__import__('hashlib').sha256(np.asarray(train,dtype='<i8').tobytes()).hexdigest(),
        'validation_sha':__import__('hashlib').sha256(np.asarray(validation if validation is not None else [],dtype='<i8').tobytes()).hexdigest()}
    if last.exists():
        if not resume:raise ValueError('Fit già esistente: usare --resume')
        old=torch.load(last,map_location='cpu',weights_only=False)
        if old['identity']!=identity:raise ValueError('Checkpoint incompatibile con dati/configurazione/split')
        model.load_state_dict(old['model']);optimizer.load_state_dict(old['optimizer']);scaler=old['scaler']
        if old['complete']:
            import pandas as pd
            pd.DataFrame(old['history']).to_csv(out/'history.csv',index=False)
            return restore(out/'best.pt',device),old['history'],old['best_epoch']
        history=old['history'];start=old['epoch']+1;best=old['best'];best_epoch=old['best_epoch'];stale=old['stale']
        torch.set_rng_state(old['torch_rng']);np.random.set_state(old['numpy_rng']);random.setstate(old['python_rng'])
        if device.startswith('cuda'):torch.cuda.set_rng_state_all(old['cuda_rng'])
    started=time.perf_counter()
    def payload(epoch,complete):
        return {'identity':identity,'model':{k:v.detach().cpu() for k,v in model.state_dict().items()},
            'optimizer':optimizer.state_dict(),'scaler':scaler,'epoch':epoch,'history':history,'best':best,
            'best_epoch':best_epoch,'stale':stale,'complete':complete,'name':name,'output':output,'config':cfg,
            'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all() if device.startswith('cuda') else [],
            'numpy_rng':np.random.get_state(),'python_rng':random.getstate()}
    for epoch in range(start,limit):
        t=time.perf_counter();model.train();order=np.random.default_rng(cfg['seed']+epoch).permutation(len(xt));loss_sum=0.
        for low in range(0,len(order),cfg['batch_size']):
            idx=order[low:low+cfg['batch_size']];xb=torch.as_tensor(xt[idx],device=device);yb=torch.as_tensor(yt[idx],device=device)
            optimizer.zero_grad(set_to_none=True);loss=(model(xb)-yb).square().mean()
            if not torch.isfinite(loss):raise ValueError('Loss non finita')
            loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            if not torch.isfinite(norm):raise ValueError('Gradienti non finiti')
            optimizer.step();loss_sum+=float(loss.detach())*len(idx)
            if low==0 or low//cfg['batch_size']%20==0:
                elapsed=time.perf_counter()-t;done=low+len(idx)
                progress.update(epoch=epoch+1,max_epochs=limit,batch_cells_done=done,batch_cells_total=len(xt),
                    eta_epoch_seconds=elapsed*(len(xt)-done)/done,cells_per_second=done/elapsed)
        pred=predict(model,xv,cfg['batch_size'],device) if xv is not None else None
        score=macro_rmse(y[validation],pred,aois[validation]) if validation is not None else loss_sum/len(xt)
        if not np.isfinite(score):raise ValueError('Score validation/loss non finito')
        mse=float(np.mean((y[validation]-pred)**2)) if pred is not None else None
        history.append({'epoch':epoch+1,'training_mse':loss_sum/len(xt),'validation_mse':mse,
            'validation_macro_rmse':score if validation is not None else None,'seconds':time.perf_counter()-t})
        if score<best-cfg['min_delta'] or validation is None:
            best=score;best_epoch=epoch+1;stale=0;save(out/'best.pt',payload(epoch,False))
        else:stale+=1
        complete=(epoch+1==limit) or (validation is not None and stale>=cfg['patience'])
        save(last,payload(epoch,complete))
        epoch_progress(progress,history,epoch+1,limit,fit_elapsed_seconds=time.perf_counter()-started,
            best_epoch=best_epoch,validation_macro_rmse=score if validation is not None else None)
        if complete:break
    # Curva CSV e checkpoint permettono di valutare overfit e durata effettiva.
    import pandas as pd
    pd.DataFrame(history).to_csv(out/'history.csv',index=False)
    return restore(out/'best.pt',device),history,best_epoch

# Restituisce testa e scaler training-only; non richiede pesi del foundation model.
def restore(path,device):
    p=torch.load(path,map_location='cpu',weights_only=False)
    model=make(p['name'],p['output'],p['config']).to(device);model.load_state_dict(p['model']);model.eval()
    return model,p['scaler']
