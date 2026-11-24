"""Preserve the fitted PAMF projections while exposing branch-local resolution."""
import torch
from torch import nn


def retain_projection(old, bins=4):
    new = nn.Linear(old.in_features*bins,old.out_features,bias=old.bias is not None,
                    device=old.weight.device)
    with torch.no_grad():
        new.weight.copy_(old.weight.repeat_interleave(bins,dim=1)/bins)
        if old.bias is not None:
            new.bias.copy_(old.bias)
    return new


def set_pool(core, path):
    pool = nn.Sequential(nn.AdaptiveAvgPool1d(4), nn.Flatten(1))
    if path == 'time':
        core.t_pool, core.t_proj = pool, retain_projection(core.t_proj)
    elif path == 'frequency':
        core.f_pool, core.f_proj = pool, retain_projection(core.f_proj)
    else:
        core.pa_pool, core.pa_proj[0] = pool, retain_projection(core.pa_proj[0])


def widen_reference(raw, width):
    path = raw.response_projection
    first, last = path[0],path[2]
    n = first.out_features
    device = first.weight.device
    a = nn.Linear(first.in_features,width,device=device)
    b = nn.Linear(width,last.out_features,device=device)
    with torch.no_grad():
        a.weight.mul_(.1)
        a.bias.zero_()
        a.weight[:n].copy_(first.weight)
        a.bias[:n].copy_(first.bias)
        b.weight.zero_()
        b.weight[:,:n].copy_(last.weight)
        b.bias.copy_(last.bias)
    path[0], path[2] = a,b


def adapt(raw, row):
    core = raw.core.id_backbone
    if path := row.get('branch_pool'):
        set_pool(core,path)
    if scale := row.get('freq_scale'):
        core.freq_bands *= scale
    if width := row.get('reference_width'):
        widen_reference(raw,width)
