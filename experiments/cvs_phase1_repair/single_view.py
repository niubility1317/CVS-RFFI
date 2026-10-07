"""Opt-in training-only single-view channel sampling; no change to released runs.

Reuses the physical renderer and schedule, but samples one side of the clean/
satellite objective per item. Never use this to create Phase2 query views.
"""
import math
import torch
import torch.nn.functional as F
from experiments.cvs_phase1_overlay.augmentation import OriginalLEO


class SingleViewLEO:
    def __init__(self, output, *, seed, satellite_weight=.68, start_epoch=80, renderer=None):
        if not math.isfinite(satellite_weight) or not 0 <= satellite_weight or start_epoch < 1:
            raise ValueError('Invalid loss weight or start epoch')
        self.weight = float(satellite_weight)
        self.probability = self.weight / (1 + self.weight)
        self.start_epoch = int(start_epoch)
        self.generator = torch.Generator(device='cpu').manual_seed(int(seed))
        self.renderer = renderer if renderer is not None else OriginalLEO(output)

    def __call__(self, x, batch, epoch, step):
        if len(batch['meta']) != len(x): raise ValueError('Physical metadata misaligned')
        info = dict(input_samples=len(x), model_samples=len(x), selected_samples=0,
                    rendered_samples=0, channel_applied=False, view='clean',
                    branch_probability=0., view_probability=0., loss_scale=1.)
        if epoch < self.start_epoch or self.weight == 0:
            return x, info
        chosen = (torch.rand(len(x), generator=self.generator) < self.probability).nonzero().flatten()
        info.update(selected_samples=len(chosen),branch_probability=self.probability,loss_scale=1+self.weight)
        if not len(chosen): return x, info
        index = chosen.to(x.device)
        # Pass only physical identity/receiver/day metadata, never labels.
        meta = [{k:batch['meta'][i][k] for k in ('sample_id','rx_i','day_i')} for i in chosen.tolist()]
        view = self.renderer(x.index_select(0,index), dict(meta=meta), epoch, step)
        if view.x.shape != x.index_select(0,index).shape: raise ValueError('Renderer shape differs')
        value = x.clone()
        value.index_copy_(0,index,view.x)
        info.update(rendered_samples=len(chosen) if view.applied else 0,
                    channel_applied=bool(view.applied),view=view.scenario,view_probability=view.view_prob)
        return value, info


def single_view_loss(model, x, y, domain, batch, augment, epoch, step):
    value, info = augment(x,batch,epoch,step)
    logits = model(value,y=y,domain_labels=domain)
    ce = F.cross_entropy(logits,y)
    return info['loss_scale']*ce, dict(info, sampled_ce=ce)
