"""Focused numerical/state checks, without data or target access."""
import argparse
import copy
import json
from pathlib import Path
import torch
from .frequency_optim import FrequencyAdamW


class Toy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.id_backbone=torch.nn.Module()
        self.id_backbone.sinc=torch.nn.Module()
        s=self.id_backbone.sinc;s.sample_rate=25e6
        s.low_hz_=torch.nn.Parameter(torch.tensor([[30.],[8327119.5]]))
        s.band_hz_=torch.nn.Parameter(torch.tensor([[365.],[4172798.5]]))
        self.id_backbone.dense=torch.nn.Linear(2,2)
        self.head=torch.nn.Linear(2,2)


def groups(model):
    return [dict(name='encoder',params=list(model.id_backbone.parameters()),lr=1e-4),
            dict(name='fusion_head',params=list(model.head.parameters()),lr=3e-4)]


def check():
    torch.manual_seed(17);model=Toy();reference=copy.deepcopy(model)
    initial=copy.deepcopy(model.state_dict());opt=FrequencyAdamW(model,groups(model),3e-6)
    assert all(torch.equal(initial[n],p) for n,p in model.state_dict().items())
    dense=[dict(g,params=[p for p in g['params'] if p.numel()!=2]) for g in groups(reference)]
    # Explicitly use matching dense tensors: shape is not ownership in production.
    dense[0]['params']=list(reference.id_backbone.dense.parameters());dense[1]['params']=list(reference.head.parameters())
    ropt=torch.optim.AdamW(dense,weight_decay=1e-4)
    for step in range(4):
        opt.zero_grad();ropt.zero_grad()
        for g,rg in zip(opt.param_groups,ropt.param_groups):
            g['lr']=(1e-4 if g['name']=='encoder' else 3e-4)/(step+1)
            rg['lr']=g['lr']
        for name,p in model.named_parameters():
            p.grad=torch.full_like(p,-1e-11 if 'hz_' in name else .2)
            rp=dict(reference.named_parameters())[name];rp.grad=p.grad.clone()
        opt.set_frequency_lr(3e-6/(step+1));metrics=opt.step();ropt.step()
        assert all(torch.equal(p,dict(reference.named_parameters())[n]) for n,p in model.named_parameters() if 'hz_' not in n)
        assert all(r['active_gradient_zero_update_count']==0 for r in metrics['frequency'])
        assert all(r['max_abs_delta_Hz']<=r['step_cap_Hz']+.5 for r in metrics['frequency'])
    # Full exact trajectory restore: native Hz + float64 master/moments.
    resumed=Toy();resumed.load_state_dict(model.state_dict());restored=FrequencyAdamW(resumed,groups(resumed),3e-6)
    restored.load_state_dict(copy.deepcopy(opt.state_dict()))
    for a,b in zip(model.parameters(),resumed.parameters()):a.grad=torch.full_like(a,.1);b.grad=a.grad.clone()
    opt.step();restored.step()
    assert all(torch.equal(a,b) for a,b in zip(model.parameters(),resumed.parameters()))
    before=copy.deepcopy(model.state_dict());opt.zero_grad();opt.step()
    assert all(torch.equal(before[n],p) for n,p in model.state_dict().items())
    # Mutated/SAM-perturbed raw frequency must not silently overwrite master.
    p=model.id_backbone.sinc.low_hz_
    with torch.no_grad():p.add_(100.)
    try:opt.step()
    except ValueError:pass
    else:raise AssertionError('Unrestored native frequency accepted')
    with torch.no_grad():p.copy_(before['id_backbone.sinc.low_hz_'])
    bad=copy.deepcopy(opt.state_dict());bad['sample_rates'][0]*=2
    try:opt.load_state_dict(bad)
    except ValueError:pass
    else:raise AssertionError('Incompatible units accepted')
    return dict(status='PASS',initial_native_conversion_exact=True,dense_matches_AdamW=True,
                tiny_frequency_gradient_actual_update=True,physical_step_bound=True,
                exact_resume_trajectory=True,scheduled_dense_lr_resume_exact=True,no_gradient_no_update=True,
                unrestored_SAM_rejected=True,incompatible_coordinate_rejected=True,
                target_access=False,real_training_updates=0)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args()
    result=check();Path(a.output).write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps(result))
