from collections import Counter
from dataclasses import asdict
import json,math,time
import numpy as np
from . import design as d
from .policy import eligible
from experiments.cvs_f_crossrx_folds20.channel import Practical as Nominal,serial
from experiments.cvs_urban_sigtime.channel import identities,numpy_iq,tensor_iq
def minimal(scene,m,index,seed):return dict(index=index,scene=scene,seed=int(seed),state=m['state_end'],lock=m['receiver_lock_status'],added_noise_db=m.get('signal_to_added_noise_db'),quality_snr_db=m.get('quality_snr_db'),eligible=eligible(scene,m))
class Practical(Nominal):
    def __init__(self,held_RX,policy):
        if held_RX!=6:raise ValueError('Fixed sourceRX6 first falsification')
        super().__init__('train',held_RX);self.policy=policy;self.trace=None;self.proposal_packets=0;self.hard_requested=0;self.hard_accepted=0;self.hard_fallback=0;self.nominal_selected=0;self.generation_seconds=0.
    def audit(self):return dict(super().audit(),curriculum=d.config(d.rows()[0])['curriculum'],all_labeled_CE=True,nominal_support_floor=.5)
    def apply(self,x,meta,epoch=0,step=0,scene=None):
        if scene is not None or epoch!=self.policy.epoch or epoch not in range(1,21) or step not in range(1,4441) or not 0<=self.policy.eta<=.5:raise ValueError('One current trainingL curriculum draw only')
        start=time.perf_counter();ids,sessions=identities(meta);raw=numpy_iq(x);rx=meta['rx_i'].tolist();days=meta['day_i'].tolist();eq=meta['eq_i'].tolist()
        if raw.shape!=(len(ids),2,256) or not np.isfinite(raw).all() or set(rx)-set(self.fit) or set(days)-{1,2,3} or set(eq)!={1}:raise ValueError('Only legal fitL source IQ256')
        namespace=f'FcrossRX/train/e{epoch}/step{step}/L_S';eta=self.policy.eta;output=[];records=[];trace=[]
        for i,pid in enumerate(ids):
            normalized,_=self.ch.normalize_input(raw[i,0]+1j*raw[i,1])
            base_scene=str(np.random.default_rng(self.ch.stable_seed(d.CHANNEL_SEEDS['train'],namespace,'scene',pid)).choice(d.TRAIN_SCENES))
            def draw(selected,index):
                seed=self.ch.stable_seed(d.CHANNEL_SEEDS['train'],namespace,selected,pid) if index==0 else self.ch.stable_seed(d.COURSE_SEED,namespace,'proposal',index,selected,pid)
                cfg=self.configs[selected];hw=self.ch.receiver_for_session(cfg,d.CHANNEL_SEEDS['receiver'],sessions[i]);hardware=asdict(hw)
                if self.sessions.setdefault(sessions[i],hardware)!=hardware:raise ValueError('Hardware must remain stable in RX/day/eq session')
                y,m=self.residual.ResidualChannel(cfg,seed,hw).process(normalized.copy());self.proposal_packets+=1
                if m['channel_equalization_applied']:raise ValueError('Original EQoff G=1 required')
                return y,m,minimal(selected,m,index,seed)
            y0,m0,p0=draw(base_scene,0);trials=[p0];u=float(np.random.default_rng(self.ch.stable_seed(d.COURSE_SEED,namespace,'mixture',pid)).random());hard=u<eta
            selected=base_scene;y,m=y0,m0;chosen=0;accepted=hard and p0['eligible']
            if hard and not accepted:
                for index in range(1,5):
                    candidate=str(np.random.default_rng(self.ch.stable_seed(d.COURSE_SEED,namespace,'urban_scene',index,pid)).choice(('practical_mid_urban','practical_low_urban')))
                    yy,mm,pp=draw(candidate,index);trials.append(pp)
                    if pp['eligible']:y,m,selected,chosen,accepted=yy,mm,candidate,index,True;break
            self.hard_requested+=int(hard);self.hard_accepted+=int(accepted);self.hard_fallback+=int(hard and not accepted);self.nominal_selected+=int(chosen==0)
            info=dict(eta=eta,u=u,hard_requested=hard,hard_accepted=accepted,selected_index=chosen,nominal_scene=base_scene,proposals=trials)
            m.update(sample_id=pid,session_id=sessions[i],namespace=namespace,scene=selected,ordinary_before_satellite=False,curriculum=info)
            self.coverage[(int(rx[i]),int(days[i]),selected,m['state_end'],m['receiver_lock_status'])]+=1
            output.append(np.stack((y.real,y.imag)));records.append(serial(m));trace.append(dict(sample_id=pid,session_id=sessions[i],**info))
        self.calls+=1;self.packets+=len(ids)
        if self.trace:self.trace.write(json.dumps(serial(dict(step=step,epoch=epoch,eta=eta,records=trace)),allow_nan=False)+'\n')
        self.generation_seconds+=time.perf_counter()-start;return tensor_iq(np.stack(output),x.device),records
    def summary(self):
        return dict(super().summary(),proposal_rejection=True,proposal_packets=self.proposal_packets,rejected_channel_draws=self.proposal_packets-self.packets,
            hard_requested=self.hard_requested,hard_accepted=self.hard_accepted,hard_fallback=self.hard_fallback,nominal_selected=self.nominal_selected,
            nominal_support_floor=.5,extra_proposals_max=4,finite_fallback='Original nominal draw, never the final failed proposal',generation_seconds=self.generation_seconds,
            added_noise_proxy_not_total_SNR=True,TX_used_for_selection=False,student_confidence_gate=False,physical_samples_dropped=0)
