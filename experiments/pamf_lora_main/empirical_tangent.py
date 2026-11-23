"""Restore LoRa while protecting directions that vary in nearby real source IQ."""

import torch
from torch import nn

from .spectral_block_recovery import head


NEIGHBORS = 12


def damped_coordinates(atoms):
    metric = atoms.mH@atoms
    ridge = .15*metric.diagonal(dim1=-1,dim2=-2).real.mean(-1,keepdim=True).clamp_min(.001)
    eye = torch.eye(atoms.shape[-1],device=atoms.device)
    dual, _ = torch.linalg.solve_ex(metric+ridge[...,None]*eye,atoms.mH,check_errors=False)
    return dual


def preserve_direction(change, atoms, dual, protect):
    shared = atoms@(dual@change.unsqueeze(-1))
    constrained = change-.70*shared.squeeze(-1)
    if protect:
        return constrained
    before = torch.linalg.vector_norm(change,dim=-1,keepdim=True)
    after = torch.linalg.vector_norm(constrained,dim=-1,keepdim=True)
    return change*(after/torch.where(before>0,before,1.)).clamp_max(1.)


class EmpiricalTangent(nn.Module):
    def __init__(self, library, *, protect):
        super().__init__()
        self.protect = protect
        self.register_buffer('library',library,persistent=False)
        frequency = torch.fft.fftfreq(library.shape[-1],device=library.device)
        self.register_buffer('frequency',frequency,persistent=False)
        self.register_buffer('weak',(frequency.abs()>0)&(frequency.abs()<=.125),persistent=False)
        self.register_buffer('outer',frequency.abs()>.25,persistent=False)
        self.register_buffer('identity',frequency==0,persistent=False)
        for name, chirps in (('even',slice(0,None,2)),('odd',slice(1,None,2))):
            key = self.key(library[:,chirps])
            self.register_buffer(f'key_{name}',key,persistent=False)
            self.register_buffer(f'norm_{name}',key.square().mean(-1),persistent=False)
        self.coarse = head(10,2)
        self.refine = head(12,2)
        for module in (self.coarse,self.refine):
            nn.init.normal_(module[-1].weight,std=.008)
            nn.init.zeros_(module[-1].bias)

    def key(self, spectrum):
        return torch.view_as_real(spectrum[...,self.weak].mean(1)).flatten(1)

    @torch.no_grad()
    def query(self, input_spectrum, noise, name, destination):
        query = self.key(input_spectrum)
        key = getattr(self,f'key_{name}')
        bank_power = getattr(self,f'norm_{name}')
        distance = query.square().mean(-1,keepdim=True)+bank_power-2*(query@key.T)/key.shape[-1]
        error, index = distance.topk(NEIGHBORS,largest=False,sorted=True)
        temp = (noise.mean(1)/input_spectrum.shape[1]).clamp_min(.002)
        weight = (-(error-error[:,:1])/temp).softmax(-1)[:,:,None,None]
        samples = self.library[index]
        mean = (weight*samples).sum(1)
        centered = samples-mean[:,None]
        spread = (weight*centered.abs().square()).sum(1)
        donor_change = centered[:,:,destination][...,self.weak]*weight.sqrt()
        atoms = donor_change.flatten(2).transpose(-1,-2).contiguous()
        return mean,spread,atoms,damped_coordinates(atoms)

    @staticmethod
    def estimate(net, values, mixed):
        with torch.autocast('cuda',dtype=torch.bfloat16,enabled=mixed):
            result = net(torch.stack(values,dim=2).flatten(0,1))
        a,b = result.float().unflatten(0,values[0].shape[:2]).unbind(2)
        return torch.complex(a.tanh(),b.tanh())

    def limit(self, edit, contexts):
        edit = edit.masked_fill(self.identity,0.)
        for destination, atoms, dual in contexts:
            group = edit[:,destination].clone()
            part = group[...,self.weak]
            preserved = preserve_direction(part.flatten(1),atoms,dual,self.protect).reshape_as(part)
            group[...,self.weak] = preserved
            edit[:,destination] = group
        power = edit.abs().square().mean((1,2),keepdim=True)
        return edit*(.22*power.clamp_min(1e-8).rsqrt()).clamp_max(1.)

    def forward(self, observed, quality):
        x = torch.fft.fft(observed,norm='ortho')
        level = x[...,self.outer].abs().square().mean(-1,keepdim=True)
        even, odd = slice(0,None,2),slice(1,None,2)
        p_even,v_even,atoms_o,dual_o = self.query(x[:,even],level[:,even],'even',odd)
        p_odd,v_odd,atoms_e,dual_e = self.query(x[:,odd],level[:,odd],'odd',even)
        contexts = ((even,atoms_e,dual_e),(odd,atoms_o,dual_o))
        group = torch.arange(x.shape[1],device=x.device)[None,:,None]%2==0
        prior, variance = torch.where(group,p_odd,p_even),torch.where(group,v_odd,v_even)
        agreement = torch.exp(-(p_even-p_odd).abs().square()/(level+v_even+v_odd+.002))
        strength = level/(level+variance+.018)
        proposal = strength*agreement*(prior-x)
        q = quality.expand_as(x.real)
        features = (x.real,x.imag,prior.real,prior.imag,(prior-x).abs().log1p(),variance.log1p(),
                    level.log1p().expand_as(q),q,agreement,self.frequency.expand_as(q))
        mixed = self.training and x.is_cuda and torch.cuda.is_bf16_supported()
        first = .55*proposal+.18*strength*self.estimate(self.coarse,features,mixed)
        coarse = self.limit(first,contexts)
        fine = self.limit(first+.18*strength*self.estimate(self.refine,(*features,coarse.real,coarse.imag),mixed),contexts)
        return tuple(torch.fft.ifft(x+d,norm='ortho') for d in (coarse,fine))
