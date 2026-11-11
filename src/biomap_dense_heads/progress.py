# Scopo: progresso JSON atomico e monitor live standalone, senza accesso ai dati sorgente.
import json
import time
import os
import warnings
from contextlib import contextmanager
from pathlib import Path
from datetime import datetime, timezone

# Windows: il lettore consente anche la sostituzione del file da parte del writer.
@contextmanager
def shared_open(path):
    if os.name!='nt':
        with open(path,encoding='utf-8') as handle:yield handle
        return
    import ctypes,msvcrt
    from ctypes import wintypes
    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    kernel.CreateFileW.argtypes=[wintypes.LPCWSTR,wintypes.DWORD,wintypes.DWORD,ctypes.c_void_p,wintypes.DWORD,wintypes.DWORD,wintypes.HANDLE]
    kernel.CreateFileW.restype=wintypes.HANDLE
    raw=kernel.CreateFileW(str(path),0x80000000,7,None,3,0x80,None) # SHARE_READ | WRITE | DELETE
    if raw==ctypes.c_void_p(-1).value:raise ctypes.WinError(ctypes.get_last_error())
    try:fd=msvcrt.open_osfhandle(raw,os.O_RDONLY|os.O_BINARY)
    except BaseException:
        kernel.CloseHandle.argtypes=[wintypes.HANDLE];kernel.CloseHandle(raw);raise
    with os.fdopen(fd,'r',encoding='utf-8') as handle:yield handle


class Progress:
    # Budget massimo stimato, non promessa di durata: early stopping può accorciare i fit.
    def __init__(self,out):
        self.path=Path(out)/'progress.json';self.data={'state':'starting','started_utc':datetime.now(timezone.utc).isoformat()}
    def update(self,**fields):
        self.data.update(fields);self.data['updated_utc']=datetime.now(timezone.utc).isoformat()
        tmp=self.path.with_name(f'.progress.{os.getpid()}.tmp')
        for attempt in range(8):
            try:
                tmp.write_text(json.dumps(self.data,allow_nan=False,default=str),encoding='utf-8')
                os.replace(tmp,self.path);return
            except PermissionError:
                if attempt<7:time.sleep(.025*(attempt+1))
        # Un lettore/antivirus non deve terminare il training per un aggiornamento UI.
        warnings.warn('Progress non aggiornabile temporaneamente; training continua, nessun dato sorgente modificato')

# La stima viene aggiornata dopo almeno un'epoca misurata, mai prima.
def epoch_progress(progress,history,epoch,limit,**fields):
    recent=[r['seconds'] for r in history[-5:]]
    average=sum(recent)/len(recent) if recent else None
    remaining=max(0,progress.data.get('fits_total',1)-progress.data.get('fits_done',0)-1)
    progress.update(epoch=epoch,max_epochs=limit,eta_fit_cap_seconds=average*(limit-epoch) if average else None,
        eta_campaign_budget_seconds=average*(limit-epoch+limit*remaining) if average else None,
        eta_note='budget massimo stimato; early stopping e fit successivi possono differire',**fields)

# Solo lettura, nessun lock condiviso col processo training e nessuna dipendenza GPU.
def monitor(out,interval=5,once=False):
    if interval<1:raise ValueError('Intervallo minimo 1 secondo')
    path=Path(out)/'progress.json'
    while True:
        if path.exists():
            try:
                with shared_open(path) as handle:r=json.load(handle)
            except (json.JSONDecodeError,OSError):r={'state':'reading_retry'}
            age=None
            if r.get('updated_utc'):age=(datetime.now(timezone.utc)-datetime.fromisoformat(r['updated_utc'])).total_seconds()
            def duration(seconds):
                if seconds is None:return 'non disponibile'
                return f'{int(seconds)//3600:02d}:{int(seconds)%3600//60:02d}:{int(seconds)%60:02d}'
            print(f"[{r.get('updated_utc','')}] {r.get('state')} | {r.get('phase')} | {r.get('model','')} | fold {r.get('fold')} | fit {r.get('fits_done','?')}/{r.get('fits_total','?')}\n"
                f"Epoca {r.get('epoch','?')}/{r.get('max_epochs','?')} | ETA epoca {duration(r.get('eta_epoch_seconds'))} | ETA fit a budget massimo {duration(r.get('eta_fit_cap_seconds'))}\n"
                f"ETA campagna estrapolata a budget massimo {duration(r.get('eta_campaign_budget_seconds'))}; stima non garantita, early stopping può accorciare.\n"
                f"Ultimo aggiornamento {int(age) if age is not None else '?'} s fa. {r.get('error','')}",flush=True)
            if r.get('state') in ['complete','failed','interrupted']:return 0 if r['state']=='complete' else 1
        else:print(f'In attesa di {path}; nessun training avviato da questo monitor.',flush=True)
        if once:return 0
        time.sleep(interval)
