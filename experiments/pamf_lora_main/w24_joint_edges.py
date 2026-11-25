"""Two-stage native edge readouts conditioned on the corresponding W24 MAP estimates."""

import torch
from torch import nn


class BoundaryRefiner(nn.Module):
    def __init__(self):
        super().__init__()
        native = torch.stack((torch.arange(24), torch.arange(1000, 1024)))
        # W24 has 384 center samples at offsets 12 + index * 125/48 at 1 MS/s.
        index = (native - 12).float() * (48/125)
        clipped = index.clamp(0, 383)
        lo = clipped.floor().long()
        self.register_buffer('lo', lo, persistent=False)
        self.register_buffer('hi', (lo+1).clamp_max(383), persistent=False)
        self.register_buffer('mix', clipped-lo, persistent=False)
        self.register_buffer('overlap', ((index>=0) & (index<=383)).float(), persistent=False)
        self.register_buffer('side', torch.tensor([-1., 1.])[:,None].expand(2,24), persistent=False)
        self.heads = nn.ModuleList(nn.Sequential(nn.Conv1d(10, 32, 5, padding=2), nn.SiLU(),
            nn.Conv1d(32, 32, 3, padding=1), nn.SiLU(), nn.Conv1d(32, 2, 3, padding=1)) for _ in range(2))
        for head in self.heads:
            nn.init.normal_(head[-1].weight, std=.001)
            nn.init.zeros_(head[-1].bias)

    def gather(self, center):
        lower = center[...,self.lo]
        return lower + self.mix * (center[...,self.hi]-lower)

    def forward(self, edges, first, final, phase, rms, quality, variance):
        rotation = phase[..., [0,-1], None]
        frame = rotation * rms[..., None]
        received = edges.unflatten(-1,(2,24)) / frame
        context = (self.gather(variance.detach()).log1p(), quality[...,None].expand_as(received.real),
                   self.overlap.expand_as(received.real), self.side.expand_as(received.real))
        state = received
        results = []
        for model, center in zip(self.heads,(first,final),strict=True):
            nearby = self.gather(center*phase)*rotation.conj()
            features = torch.stack((received.real,received.imag,state.real,state.imag,
                                    nearby.real,nearby.imag,*context),dim=-2).flatten(0,2)
            mixed = self.training and edges.is_cuda and torch.cuda.is_bf16_supported()
            with torch.autocast('cuda',dtype=torch.bfloat16,enabled=mixed):
                edit = model(features).float().reshape(*received.shape[:-1], 2, 24)
            state = state + torch.complex(edit[...,0,:],edit[...,1,:])
            results.append((state*frame).flatten(-2))
        return tuple(results)


def attach_edge_refiner(make_center):
    model = make_center()
    model.boundary = BoundaryRefiner()
    return model


def edge_objective(received, target, rms, first, final, with_identity):
    def risk(value):
        error = ((value-target)/rms).unflatten(-1,(2,24)).abs().square().mean(-1)
        power = (target/rms).unflatten(-1,(2,24)).abs().square().mean(-1)
        return (error/(.02+power)).log1p().mean()

    half = received + .5*(final-received)
    identity = ((half[-6:]-received[-6:])/rms[-6:]).abs().square().mean() if with_identity else half.real.new_zeros(())
    costs = torch.stack((risk(first),risk(half),identity))
    return costs @ costs.new_tensor((.10,.20,.25)), costs


def restore_joint(model, tracker, original):
    from experiments.pamf_lora_main.dura_repair import complex_iq, real_iq
    from experiments.pamf_lora_main.w24_formal import coordinates

    y, phase, quality, rms = coordinates(tracker, original[..., :384])
    edge = complex_iq(original[..., 384:])
    (_, first, final), _, variance, *_ = model.propose(y, quality, rms, edge)
    _, edge_final = model.boundary(edge, first, final, phase, rms, quality, variance)
    center = original[..., :384] + .5*real_iq((final-y)*phase*rms)
    full = torch.cat((center, original[..., 384:] + .5*real_iq(edge_final-edge)), dim=-1)
    return center, full
