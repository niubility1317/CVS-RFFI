"""Smooth schedules, source-label sampler, channel and final inference averaging."""
import copy,math
import torch
from torch.utils.data import Sampler

def smooth_pl_scale(epoch, config):
    if getattr(config,"pl_schedule","cosine")=="step_full":
        return float(epoch>=config.pseudo_start_epoch)
    t=max(0.,min(1.,(epoch-config.pseudo_start_epoch+1)/max(1,config.smooth_pl_ramp_end-config.pseudo_start_epoch+1)))
    return .5-.5*math.cos(math.pi*t)

def smooth_lr_factor(epoch, config):
    if epoch <= config.smooth_lr_warmup:
        return config.smooth_lr_initial+(1-config.smooth_lr_initial)*max(0,epoch-1)/max(1,config.smooth_lr_warmup-1)
    t=max(0.,min(1.,(epoch-config.smooth_lr_start)/max(1,config.smooth_lr_end-config.smooth_lr_start)))
    return config.smooth_lr_floor+(1-config.smooth_lr_floor)*(.5+.5*math.cos(math.pi*t))

class StratifiedCycleSampler(Sampler):
    """Interleave RX×TX strata without replacement; every source label appears once."""
    def __init__(self, labels, seed):
        self.seed=seed;self.epoch=0;self.groups={};self.n=len(labels)
        for i,row in enumerate(labels.tolist()):
            self.groups.setdefault((int(row[2]),int(row[0])),[]).append(i)
    def set_epoch(self,epoch):self.epoch=epoch
    def __len__(self):return self.n
    def __iter__(self):
        generator=torch.Generator().manual_seed(self.seed+self.epoch)
        pools=[torch.tensor(v)[torch.randperm(len(v),generator=generator)].tolist() for k,v in sorted(self.groups.items())]
        result=[]
        for depth in range(max(map(len,pools))):
            for g in torch.randperm(len(pools),generator=generator).tolist():
                if depth<len(pools[g]):result.append(pools[g][depth])
        return iter(result)

@torch.no_grad()
def update_inference_ema(ema, student, decay):
    if ema is None:
        ema=copy.deepcopy(student).eval();ema.requires_grad_(False);return ema
    for key,target in ema.state_dict().items():
        source=student.state_dict()[key]
        if target.is_floating_point() or target.is_complex():target.lerp_(source,1-decay)
        else:target.copy_(source)
    return ema.eval()
