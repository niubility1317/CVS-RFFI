"""Use the imported practical engine unchanged for every training observation."""
import math
import numpy as np
from dataclasses import asdict
from . import design as d

def imports():
    # Resolve precisely the same module ordering as the native trainer first.
    from experiments.cvs_phase1_overlay.model import native_modules
    native_modules()
    import leo_practical.channel as ch
    import leo_practical.residual as residual
    return ch,residual

def configurations():
    ch,_=imports()
    return {scene:ch.Config(fs_hz=25_000_000.,scenario=scene,processing_route='residual',
        mode='post_sync',equalization_enabled=False) for scene in d.SCENES}

def audit():
    ch,r=imports(); cfg=configurations()
    return dict(channel_module_file=ch.__file__,residual_module_file=r.__file__,
        version=ch.VERSION,actual_configs={k:asdict(v) for k,v in cfg.items()},
        sampling_rate_hz=25_000_000,IQ_length=256,EQ_off_G_equals_one=True,
        truth_total_SNR_known=False,formal_test_changed=False)

def identities(meta):
    keys=('rx_i','day_i','eq_i','base_index')
    def values(v): return v.detach().cpu().tolist() if hasattr(v,'detach') else list(v)
    if not all(k in meta for k in keys): raise ValueError('Native physical metadata incomplete')
    a={k:values(meta[k]) for k in keys}; n=len(a['rx_i'])
    if any(len(v)!=n for v in a.values()): raise ValueError('Metadata batch mismatch')
    ids=list(meta['physical_id']) if 'physical_id' in meta else ['source:'+str(a['base_index'][i]) for i in range(n)]
    # Stable physical handles are never decoded to TX labels.
    sessions=[f"rx{a['rx_i'][i]}:day{a['day_i'][i]}:eq{a['eq_i'][i]}" for i in range(n)]
    return ids,sessions

def numpy_iq(x):
    return np.asarray(x.detach().cpu().tolist(),dtype=np.float64)

def tensor_iq(a,device):
    import torch
    a=np.ascontiguousarray(a,dtype=np.float32)
    return torch.frombuffer(bytearray(a.tobytes()),dtype=torch.float32).reshape(a.shape).to(device)

def usability(meta):
    db=meta.get('signal_to_added_noise_db')
    return 0. if db is None else 1/(1+math.exp(-max(-60.,min(60.,(db-5)/5))))

class PairedChannel:
    def __init__(self,c):
        d.validate(c);self.c=c;self.configs=configurations();self.ch,self.residual=imports()

    def apply(self,x,meta,epoch,step,*,view='L',eta=0.):
        if not 0<=eta<=self.c['curriculum_plan']['eta_max']: raise ValueError('Unbounded curriculum mixture')
        ids,sessions=identities(meta); raw=numpy_iq(x);result=[];details=[]
        namespace=(f'validation/fixed/{view}' if view.startswith('source_validation')
                   else f'train/epoch{epoch}/step{step}/{view}')
        for i,pid in enumerate(ids):
            selection=np.random.default_rng(self.ch.stable_seed(self.c['seeds']['augmentation'],namespace,'selection',pid))
            record,_=self.ch.normalize_input(raw[i,0]+1j*raw[i,1])
            hard=bool(selection.random()<eta); attempts=8 if hard else 1;chosen=None
            for attempt in range(attempts):
                choices=self.c['curriculum_plan']['hard_scenes'] if hard or view=='U' else d.TRAIN_SCENES
                scene=str(selection.choice(choices));cfg=self.configs[scene]
                seed=self.ch.stable_seed(self.c['seeds']['augmentation'],namespace,scene,pid,attempt)
                hardware=self.ch.receiver_for_session(cfg,self.c['seeds']['receiver'],sessions[i])
                output,m=self.residual.ResidualChannel(cfg,seed,hardware).process(record)
                chosen=output,m
                if not hard or (3<=m['quality_snr_db']<=18 and m['receiver_lock_status']!='locked'): break
            output,m=chosen
            # Finite proposal fallback retains an actual original-engine draw.
            # It is disclosed separately, never weakened or dropped.
            m.update(sample_id=pid,session_id=sessions[i],namespace=namespace,hard_requested=hard,
                hard_accepted=hard and 3<=m['quality_snr_db']<=18 and m['receiver_lock_status']!='locked',
                proposal_count=attempt+1,q=usability(m),eta=float(eta))
            result.append(np.stack([output.real,output.imag]));details.append(m)
        return tensor_iq(np.stack(result),x.device),details
