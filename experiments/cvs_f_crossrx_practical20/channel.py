from collections import Counter
from dataclasses import asdict
import math
import numpy as np
from . import design as d
from experiments.cvs_urban_sigtime.channel import imports,identities,numpy_iq,tensor_iq

def serial(v):
    if isinstance(v,np.generic):return serial(v.item())
    if isinstance(v,np.ndarray):return serial(v.tolist())
    if isinstance(v,dict):return {str(k):serial(x) for k,x in v.items()}
    if isinstance(v,(tuple,list)):return [serial(x) for x in v]
    if isinstance(v,complex):return [v.real,v.imag]
    if isinstance(v,float) and not math.isfinite(v):return None
    return v

class Practical:
    def __init__(self,role):
        if role not in ('train','validation'):raise ValueError('Separate named train/validation streams only')
        self.role=role;self.ch,self.residual=imports();self.calls=0;self.packets=0;self.coverage=Counter();self.sessions={}
        self.configs={s:self.ch.Config(fs_hz=25_000_000.,scenario=s,processing_route='residual',mode='post_sync',
            equalization_enabled=False,reference_snr_db=30.,boundary='reflect') for s in d.SCENES}
    def audit(self):
        return dict(role=self.role,channel_module_file=self.ch.__file__,residual_module_file=self.residual.__file__,version=self.ch.VERSION,
            configurations={s:asdict(c) for s,c in self.configs.items()},EQ_off_G_equals_one=True,
            normalization='native source RMS before views; original simulator normalize_input then full receiver AGC, no added output RMS',
            ordinary_before_satellite=False,unknown_TX_inverse=False,true_total_SNR_known=False,formal_test_changed=False)
    def apply(self,x,meta,epoch=0,step=0,scene=None):
        if (self.role=='train' and (scene is not None or epoch not in range(1,21) or step not in range(1,4441))) or (self.role=='validation' and scene not in d.SCENES):
            raise ValueError('Fixed source role/scene/budget required')
        ids,sessions=identities(meta);raw=numpy_iq(x)
        if raw.shape!=(len(ids),2,256) or not np.isfinite(raw).all():raise ValueError('Finite source IQ256 required')
        rx=meta['rx_i'].tolist();days=meta['day_i'].tolist();eq=meta['eq_i'].tolist()
        if set(rx)-{1,3,4,6,8} or set(days)-{1,2,3} or set(eq)!={1}:raise ValueError('Only legal source IQ metadata')
        if self.role=='train' and set(rx)-{3,4,6,8}:raise ValueError('Held RX1 cannot enter training')
        if self.role=='validation' and set(rx)!={1}:raise ValueError('Only fixed held source RX1 enters final source evaluation')
        namespace=f'FcrossRX/train/e{epoch}/step{step}/L_S' if self.role=='train' else 'FcrossRX/source_validation/fixed/v1'
        seedbase=d.CHANNEL_SEEDS[self.role];output=[];records=[]
        for i,pid in enumerate(ids):
            selected=scene
            if selected is None:
                rng=np.random.default_rng(self.ch.stable_seed(seedbase,namespace,'scene',pid));selected=str(rng.choice(d.TRAIN_SCENES))
            cfg=self.configs[selected];seed=self.ch.stable_seed(seedbase,namespace,selected,pid)
            hw=self.ch.receiver_for_session(cfg,d.CHANNEL_SEEDS['receiver'],sessions[i]);hardware=asdict(hw)
            previous=self.sessions.setdefault(sessions[i],hardware)
            if previous!=hardware:raise ValueError('Hardware changed within RX/day/eq session')
            normalized,_=self.ch.normalize_input(raw[i,0]+1j*raw[i,1])
            y,m=self.residual.ResidualChannel(cfg,seed,hw).process(normalized)
            if m['channel_equalization_applied']:raise ValueError('Declared EQ off must retain G=1')
            m.update(sample_id=pid,session_id=sessions[i],namespace=namespace,scene=selected,ordinary_before_satellite=False)
            self.coverage[(int(rx[i]),int(days[i]),selected,m['state_end'],m['receiver_lock_status'])]+=1
            output.append(np.stack((y.real,y.imag)));records.append(serial(m))
        self.calls+=1;self.packets+=len(ids)
        return tensor_iq(np.stack(output),x.device),records
    def summary(self):
        return dict(calls=self.calls,packets=self.packets,role=self.role,
            strata=[dict(RX=k[0],day=k[1],scene=k[2],state=k[3],lock=k[4],count=n) for k,n in sorted(self.coverage.items())],
            session_count=len(self.sessions),sessions=self.sessions,geometry_noise_receiver_dependencies='original full practical residual',
            proposal_rejection=False,dropped_samples=0,ordinary_before_satellite=False,formal_test_changed=False)
