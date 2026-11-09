"""Exact, source-only reuse for the original deterministic labeled As view."""
from collections import Counter
from contextlib import contextmanager
from dataclasses import asdict
import hashlib
import os
from pathlib import Path
import shutil
import threading
import time

import numpy as np
import torch


class Practical:
    def __init__(self,c):
        from leo_practical.channel import Config
        self.c=c;self.coverage=Counter();self.sessions={};self._receivers={}
        self.cache=Path(c['fast_as_cache']);self.cache.mkdir(parents=True,exist_ok=True)
        self.cache_hits=self.cache_misses=self.cache_write_failures=0
        self.scenes=('practical_low_suburban','practical_mid_urban','practical_low_urban')
        self.configs={scene:Config(fs_hz=25e6,fc_hz=2462e6,scenario=scene,processing_route='residual',
            mode='post_sync',equalization_enabled=False,reference_snr_db=30.,boundary='reflect') for scene in self.scenes}

    def __call__(self,x,meta,epoch,step):
        from leo_practical.channel import stable_seed,receiver_for_session,normalize_input
        from leo_practical.residual import ResidualChannel
        keys=('eq_i','rx_i','day_i','sig_i','base_index')
        values={k:meta[k].detach().cpu().tolist() for k in keys}
        if set(values['rx_i'])-set(self.c['source_rxs']) or set(values['day_i'])-set(self.c['source_days']) or set(values['eq_i'])!={1}:
            raise ValueError('As source contract differs')
        if x.dtype!=torch.float32:raise ValueError('Source IQ dtype differs')
        raw=np.asarray(x.detach().cpu().tolist(),dtype=np.float32)
        if raw.shape!=(len(x),2,256):raise ValueError('Source IQ shape differs')
        # This is an input-addressed memoization key, not an additional data
        # validation or model-selection gate. Identical values, IDs and step
        # produce the exact same original simulator output.
        key=hashlib.blake2b(digest_size=20)
        key.update(str((int(epoch),int(step),raw.dtype.str,raw.shape)).encode())
        key.update(raw.tobytes(order='C'))
        for k in keys:key.update(np.asarray(values[k],dtype='<i8').tobytes())
        digest=key.hexdigest();folder=self.cache/digest[:2];path=folder/(digest+'.npy')
        namespace='FcrossRX/train/e%d/step%d/L_S'%(epoch,step)
        plan=[]
        for i in range(len(x)):
            pid=tuple(int(values[k][i]) for k in keys)
            session='rx%d:day%d:eq1'%(values['rx_i'][i],values['day_i'][i])
            rng=np.random.default_rng(stable_seed(20261016421,namespace,'scene',pid))
            scene=str(rng.choice(self.scenes));cfg=self.configs[scene];token=(scene,session)
            if token not in self._receivers:
                hw=receiver_for_session(cfg,20261016423,session);record=asdict(hw)
                if self.sessions.setdefault(session,record)!=record:raise ValueError('As physical session changed')
                self._receivers[token]=hw
            self.coverage[scene]+=1
            plan.append((pid,scene,cfg,self._receivers[token]))
        if path.is_file():
            result=np.load(path,allow_pickle=False)
            if result.dtype!=np.float32 or result.shape!=(len(x),2,256) or not np.isfinite(result).all():
                raise ValueError('Invalid shared source As cache')
            self.cache_hits+=1
        else:
            out=[];raw64=raw.astype(np.float64,copy=False)
            for i,(pid,scene,cfg,hw) in enumerate(plan):
                signal,_=normalize_input(raw64[i,0]+1j*raw64[i,1])
                signal,info=ResidualChannel(cfg,stable_seed(20261016421,namespace,scene,pid),hw).process(signal)
                if not np.isfinite(signal).all() or info['channel_equalization_applied']:raise ValueError('As finite noEQ contract differs')
                out.append(np.stack((signal.real,signal.imag)))
            result=np.ascontiguousarray(out,np.float32);self.cache_misses+=1
            tmp=None
            try:
                if shutil.disk_usage(self.cache).free < self.c.get('fast_as_min_free_bytes',200*1024**3):
                    raise OSError('Shared cache paused: disk free space below reserve')
                folder.mkdir(parents=True,exist_ok=True)
                tmp=folder/(digest+'.tmp-'+str(os.getpid())+'-'+str(threading.get_ident())+'-'+str(time.monotonic_ns()))
                with tmp.open('xb') as f:np.save(f,result,allow_pickle=False)
                try:os.link(tmp,path)
                except FileExistsError:pass
            except OSError as error:
                self.cache_write_failures+=1
                if self.cache_write_failures==1:print('AS_CACHE_WRITE_UNAVAILABLE',repr(error),flush=True)
            finally:
                if tmp is not None and tmp.exists():tmp.unlink()
        calls=self.cache_hits+self.cache_misses
        if calls%222==0:
            print('AS_CACHE_STATS',dict(hits=self.cache_hits,misses=self.cache_misses,write_failures=self.cache_write_failures),flush=True)
        return torch.frombuffer(bytearray(result.tobytes()),dtype=torch.float32).reshape(result.shape).to(x.device)


@contextmanager
def installed(c):
    if not c.get('fast_as_cache'):
        yield
        return
    from experiments.cric_dura_comparison import t1_practical
    original=t1_practical.Practical;t1_practical.Practical=Practical
    try:yield
    finally:t1_practical.Practical=original
