"""Nominal full practical draw plus same-operator independent-noise HQ view."""
import numpy as np
from . import design as d
from experiments.cvs_urban_sigtime50.channel import imports,configurations,identities,numpy_iq,tensor_iq,usability,audit
from experiments.cvs_urban_bridge.views import coupled


class PairedChannel:
    def __init__(self,c):
        d.validate(c);self.c=c;self.configs=configurations();self.ch,self.residual=imports()

    def apply(self,x,meta,epoch,step,*,view='L',scene=None):
        if view!='L' and not view.startswith('source_validation_bridge'):
            raise ValueError('Bridge generator accepts labeled source views only')
        if scene is not None and (view=='L' or scene not in d.TRAIN_SCENES):
            raise ValueError('Fixed scene is source calibration only')
        ids,sessions=identities(meta);raw=numpy_iq(x);full=[];hq=[];details=[]
        namespace=f'validation/fixed/{view}' if view.startswith('source_validation') else f'train/epoch{epoch}/step{step}/{view}'
        seed_base=self.c['bridge']['calibration_stream'] if view.startswith('source_validation') else self.c['seeds']['augmentation']
        for i,pid in enumerate(ids):
            selection=np.random.default_rng(self.ch.stable_seed(seed_base,namespace,'selection',pid))
            selection.random()  # Preserve original eta=0 nominal RNG consumption.
            chosen=scene or str(selection.choice(d.TRAIN_SCENES));cfg=self.configs[chosen]
            seed=self.ch.stable_seed(seed_base,namespace,chosen,pid,0)
            hardware=self.ch.receiver_for_session(cfg,self.c['seeds']['receiver'],sessions[i])
            record,_=self.ch.normalize_input(raw[i,0]+1j*raw[i,1])
            waves,m,info,_=coupled(record,cfg,seed,hardware,
                noise_improvement_db=self.c['bridge']['noise_improvement_db'])
            m.update(sample_id=pid,session_id=sessions[i],namespace=namespace,hard_requested=False,
                hard_accepted=False,proposal_count=1,q=usability(m),eta=0.,bridge=info)
            full.append(np.stack((waves['full'].real,waves['full'].imag)))
            hq.append(np.stack((waves['bridge_independent'].real,waves['bridge_independent'].imag)));details.append(m)
        return tensor_iq(np.stack(full),x.device),tensor_iq(np.stack(hq),x.device),details
