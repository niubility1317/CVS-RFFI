"""Original T1 residual As simulator generalized only to declared source RX/day."""
from collections import Counter
from dataclasses import asdict
import numpy as np
import torch


class Practical:
    def __init__(self,c):
        self.c=c;self.coverage=Counter();self.sessions={}
    def __call__(self,x,meta,epoch,step):
        from leo_practical.channel import Config,stable_seed,receiver_for_session,normalize_input
        from leo_practical.residual import ResidualChannel
        keys=('eq_i','rx_i','day_i','sig_i','base_index')
        values={k:meta[k].detach().cpu().tolist() for k in keys}
        if set(values['rx_i'])-set(self.c['source_rxs']) or set(values['day_i'])-set(self.c['source_days']) or set(values['eq_i'])!={1}:raise ValueError('As source contract differs')
        scenes=('practical_low_suburban','practical_mid_urban','practical_low_urban')
        namespace='FcrossRX/train/e%d/step%d/L_S'%(epoch,step);raw=np.asarray(x.detach().cpu().tolist(),np.float64);out=[]
        for i in range(len(x)):
            pid=tuple(int(values[k][i]) for k in keys);session='rx%d:day%d:eq1'%(values['rx_i'][i],values['day_i'][i])
            rng=np.random.default_rng(stable_seed(20261016421,namespace,'scene',pid));scene=str(rng.choice(scenes))
            cfg=Config(fs_hz=25e6,fc_hz=2462e6,scenario=scene,processing_route='residual',mode='post_sync',equalization_enabled=False,reference_snr_db=30.,boundary='reflect')
            hw=receiver_for_session(cfg,20261016423,session)
            if self.sessions.setdefault(session,asdict(hw))!=asdict(hw):raise ValueError('As physical session changed')
            signal,_=normalize_input(raw[i,0]+1j*raw[i,1]);signal,info=ResidualChannel(cfg,stable_seed(20261016421,namespace,scene,pid),hw).process(signal)
            if not np.isfinite(signal).all() or info['channel_equalization_applied']:raise ValueError('As finite noEQ contract differs')
            self.coverage[scene]+=1;out.append(np.stack((signal.real,signal.imag)))
        a=np.ascontiguousarray(out,np.float32)
        return torch.frombuffer(bytearray(a.tobytes()),dtype=torch.float32).reshape(a.shape).to(x.device)
