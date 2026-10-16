"""Unmodified practical residual, raw IQ only, named independent random streams."""
from dataclasses import asdict
import numpy as np
import torch
from . import design as d

def imports():
    from experiments.cvs_phase1_overlay.model import native_modules
    native_modules()
    import leo_practical.channel as ch
    import leo_practical.residual as residual
    return ch,residual

class PairedChannel:
    def __init__(self):
        self.ch,self.residual=imports()
        self.configs={s:self.ch.Config(fs_hz=25000000.,fc_hz=2462000000.,scenario=s,processing_route='residual',mode='post_sync',equalization_enabled=False) for s in d.SCENES}
    def __deepcopy__(self,memo):
        # Modules are import handles, not model state. Recreate this stateless
        # view generator for native ownEMA/reader deep copies.
        value=type(self)();memo[id(self)]=value;return value
    def audit(self):return dict(channel_file=self.ch.__file__,residual_file=self.residual.__file__,actual_configs={k:asdict(v) for k,v in self.configs.items()},EQ_off_G_one=True,formal_test_unchanged=True)
    def apply(self,x,meta,*,namespace,scene=None):
        if x.ndim!=3 or x.shape[1:]!=(2,256) or not bool(torch.isfinite(x).all()):raise ValueError('Finite raw IQ256 required')
        if scene is not None and (scene not in d.SCENES or not namespace.startswith('source_eval/')):raise ValueError('Scene override only for declared source evaluation')
        required=('rx_i','day_i','eq_i','sig_i','base_index')
        if not all(k in meta and len(meta[k])==len(x) for k in required):raise ValueError('Actual physical source metadata required')
        values={k:meta[k].detach().cpu().tolist() if hasattr(meta[k],'detach') else list(meta[k]) for k in required}
        if any(int(r) not in (1,3,4,6,8) for r in values['rx_i']) or any(int(v) not in (1,2,3) for v in values['day_i']):raise ValueError('Source physical roles only')
        raw=np.asarray(x.detach().cpu().tolist(),dtype=np.float64);outputs=[];details=[]
        seed0=d.EVAL_SEED if namespace.startswith('source_eval/') else d.CHANNEL_SEED
        for i in range(len(x)):
            pid=tuple(int(values[k][i]) for k in required);session='rx%d:day%d:eq%d'%(pid[0],pid[1],pid[2])
            # Session uses the physical RX/day/eq tuple, no TX or pseudo label.
            rng=np.random.default_rng(self.ch.stable_seed(seed0,namespace,'scene',pid))
            selected=scene or d.SCENES[int(rng.integers(len(d.SCENES)))];cfg=self.configs[selected]
            seed=self.ch.stable_seed(seed0,namespace,selected,pid);receiver=self.ch.receiver_for_session(cfg,d.RECEIVER_SEED,session)
            record,_=self.ch.normalize_input(raw[i,0]+1j*raw[i,1]);out,m=self.residual.ResidualChannel(cfg,seed,receiver).process(record)
            if not np.isfinite(out).all():raise FloatingPointError('Invalid physical output, never drop packet')
            outputs.append(np.stack([out.real,out.imag]));details.append(dict(scene=selected,seed=seed,session=session,RX=pid[0],day=pid[1],state=m['state_end'],lock=m['receiver_lock_status'],quality_snr_db=m['quality_snr_db'],signal_to_added_noise_db=m['signal_to_added_noise_db'],agc_limited=m['agc_limited']))
        array=np.ascontiguousarray(outputs,dtype=np.float32)
        result=torch.frombuffer(bytearray(array.tobytes()),dtype=torch.float32).reshape(array.shape).to(x.device)
        return result,details
