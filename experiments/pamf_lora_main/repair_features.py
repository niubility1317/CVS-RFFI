"""LoRa preamble features and balanced sampling for the repaired Tri recipe."""
import numpy as np
import torch
from torch import nn

from .clean_aperture import ALL_CHIRPS, NATIVE_EDGE_POINTS, native_edge_positions
from .data import BANDWIDTH_HZ
from .physical import ChirpResponse


def complete_labeled_batches(y, rx, rng):
    """Cover every class/receiver cell exactly once, with four RXs per class."""
    cells = [[np.flatnonzero((y == c) & (rx == r)) for r in range(5)] for c in range(10)]
    sizes = {len(cell) for group in cells for cell in group}
    if len(sizes) != 1 or min(sizes) < 16 or min(sizes) % 16:
        raise ValueError('LoRa balanced batches require equal class/RX cells divisible by 16')
    size = sizes.pop()
    cells = [[rng.permutation(cell) for cell in group] for group in cells]
    for used in range(0, size, 16):
        # Five pairs of classes take turns being absent. In four of the five
        # batches each class advances to the next omitted receiver.
        classes = rng.permutation(10).reshape(5, 2)
        orders = np.stack([rng.permutation(5) for _ in range(10)])
        cursor = np.full((10, 5), used, dtype=np.int64)
        turns = np.zeros(10, dtype=np.int64)
        for step in range(25):
            banned = step % 5
            chosen = classes[np.arange(5) != banned].reshape(-1)
            batch = []
            for c in chosen:
                dropped = orders[c, turns[c] % 5]
                turns[c] += 1
                for r in range(5):
                    if r != dropped:
                        batch.append(cells[c][r][cursor[c, r]])
                        cursor[c, r] += 1
            yield np.asarray(batch, dtype=np.int64)


def packet_phase_curvature(gain):
    phase=gain/gain.abs().clamp_min(1e-5)
    # A constant phase and a common linear phase slope both cancel.
    variation=phase[:,2:]*phase[:,:-2]*phase[:,1:-1].conj().square()
    return torch.cat((variation.real,variation.imag),dim=1)


def relative_packet_physics(tone, dc, residual):
    def centered(values):
        return values-values[:,1:-1].mean(1,keepdim=True)

    gain=centered(dc.abs().clamp_min(1e-6).log())
    residual_power=(residual-dc[...,None]).abs().square().mean(-1)
    energy=centered((residual_power/dc.abs().square().clamp_min(1e-12)).clamp_min(1e-7).log())
    tone=centered(tone)
    # Rounded per-position scales measured on the labeled source only.
    scales=gain.new_tensor((.09,.07,.04,.02,.02,.04,.07,.11,
                            20.,5.,4.,3.,2.,3.,6.,11.,
                            .75,.13,.10,.10,.05,.10,.16,.19))
    return torch.asinh(torch.cat((gain,tone,energy),dim=1)/scales)


class AlignedTrace(nn.Module):
    """Remove the common dechirped tone and retain residual amplitude and phase."""

    def __init__(self, *, samples=256):
        super().__init__()
        self.reference = ChirpResponse(samples=samples)
        self.register_buffer('time', torch.arange(samples, dtype=torch.float32), persistent=False)

    def forward(self, x, *, with_drift=False, with_phase=False, with_packet_physics=False):
        z = torch.complex(x[:, :, 0], x[:, :, 1])
        baseband = z * self.reference.chirp.conj()
        length = 8*self.time.numel()
        k = torch.fft.fft(baseband, n=length).abs().argmax(-1)
        k = (k + length // 2) % length - length // 2
        coarse = 2 * torch.pi * k[..., None] * self.time / length
        w = baseband * torch.polar(torch.ones_like(coarse), -coarse)
        lag = 3*self.time.numel()//32
        slope = (w[..., lag:] * w[..., :-lag].conj()).mean(-1).angle() / lag
        slope = slope.clamp(-torch.pi / length, torch.pi / length)
        fine = slope[..., None] * self.time
        w = w * torch.polar(torch.ones_like(fine), -fine)
        dc = w.mean(-1, keepdim=True)
        if with_packet_physics:
            rate=self.reference.sample_rate
            tone=k*(rate/length)+slope*(rate/(2*torch.pi))
            packet_physics=relative_packet_physics(tone,dc.squeeze(-1),w)
        w = w * (dc.conj() / dc.abs().clamp_min(1e-5))
        rms = z.abs().square().mean(-1, keepdim=True).clamp_min(1e-6).sqrt()
        w = w / rms
        trace = torch.stack((w.real, w.imag, z.abs() / rms), dim=2)
        if with_packet_physics:
            return trace,packet_phase_curvature(dc.squeeze(-1)),packet_physics
        if with_phase:
            return trace,packet_phase_curvature(dc.squeeze(-1))
        if with_drift:
            # Only variation between chirps enters the model. The common
            # carrier offset, including receiver oscillator bias, is removed.
            rate = self.reference.sample_rate
            tone = k * (rate / length) + slope * (rate / (2 * torch.pi))
            drift = tone - tone.mean(dim=1, keepdim=True)
            return trace, drift
        return trace


def residual_logit_scale(core_logits,delta_logits,limit):
    core=core_logits.detach()
    core_norm=(core-core.mean(1,keepdim=True)).norm(dim=1,keepdim=True)
    delta_norm=(delta_logits-delta_logits.mean(1,keepdim=True)).norm(dim=1,keepdim=True)
    return (limit*core_norm/delta_norm.clamp_min(1e-8)).clamp(max=1.)


class NativeBoundaryPhase(nn.Module):
    """Represent within-window phase differences and adjacent-chirp boundaries."""

    def __init__(self,channels,*,edge_points=NATIVE_EDGE_POINTS,lag=1):
        super().__init__()
        if not isinstance(lag,int) or not 1<=lag<edge_points:
            raise ValueError('Native phase lag must fit within one edge window')
        self.edge_points=edge_points
        self.lag=lag
        count=len(ALL_CHIRPS)
        width=count*2*(edge_points-lag)+(count-1)*3
        t=torch.tensor(native_edge_positions(edge_points),dtype=torch.float64)*1e-6
        reference=torch.exp(2j*torch.pi*BANDWIDTH_HZ*(t.square()/(2*.001024)-t/2))
        self.register_buffer('edge_reference',reference.to(torch.complex64).reshape(2,edge_points),
                             persistent=False)
        self.project=nn.Sequential(nn.LayerNorm(width),nn.Linear(width,96),nn.SiLU(),
                                   nn.Linear(96,channels,bias=False))
        nn.init.zeros_(self.project[-1].weight)

    def features(self,core,edge,core_reference):
        def unit(z):
            return z/z.abs().clamp_min(1e-8)

        baseband=torch.complex(core[:,:,0],core[:,:,1])*core_reference.conj()
        core_steps=unit(baseband[...,1:]*baseband[...,:-1].conj())
        # One native step lasts 1 us. The core is sampled at 384 kHz.
        angle=.384*core_steps[...,24:-24].mean(-1).angle()
        correction=torch.polar(torch.ones_like(angle),-angle)
        edges=torch.complex(edge[:,:,0],edge[:,:,1]).reshape(len(edge),len(ALL_CHIRPS),2,self.edge_points)
        edges=edges*self.edge_reference.conj()
        lag=self.lag
        lag_correction=correction if lag==1 else torch.polar(torch.ones_like(angle),-lag*angle)
        steps=unit(edges[...,lag:]*edges[...,:-lag].conj())*lag_correction[...,None,None]
        next_start=edges[:,1:,0,0]
        previous_end=edges[:,:-1,1,-1]
        junction=unit(next_start*previous_end.conj())*correction[:,:-1]
        gain=next_start.abs().clamp_min(1e-5).log()-previous_end.abs().clamp_min(1e-5).log()
        return torch.cat((steps.imag.flatten(1),junction.real,junction.imag,gain),dim=1)

    def forward(self,core,edge,core_reference):
        return .25*self.project(self.features(core,edge,core_reference))


class PacketGroupFusion(nn.Module):
    """Fuse chirp positions and relative physics before the existing head."""

    def __init__(self, channels, chirp_count, *, radial=False):
        super().__init__()
        self.compression=nn.Linear(channels,8,bias=False)
        widths=(chirp_count*8,36,channels,channels,channels,channels,2*channels)
        self.groups=nn.ModuleList(
            nn.Sequential(nn.LayerNorm(width),nn.Linear(width,32)) for width in widths
        )
        self.fusion=nn.Linear(32*len(widths),channels,bias=False)
        self.radial=radial
        nn.init.zeros_(self.fusion.weight)

    def forward(self,chirps,physics,raw_packet):
        first,middle,tail=chirps[:,0],chirps[:,1:-1].mean(1),chirps[:,-1]
        edges=torch.cat((first-middle,tail-middle),dim=1)
        values=(self.compression(chirps).flatten(1),physics,raw_packet,first,middle,tail,edges)
        projected=[layer(value) for layer,value in zip(self.groups,values,strict=True)]
        packet=torch.cat(projected,dim=1)
        packet=torch.exp(-.5*packet.square())-.5 if self.radial else torch.nn.functional.silu(packet)
        return .25*self.fusion(packet)


class CoherentPacket(nn.Module):
    """Add a common-tone-aligned residual branch to the existing PAMF packet F."""

    def __init__(self, raw):
        super().__init__()
        self.raw = raw
        self.trace = AlignedTrace()
        self.encoder = nn.Sequential(
            nn.Conv1d(3, 24, 9, stride=2, padding=4), nn.GroupNorm(4, 24), nn.SiLU(),
            nn.Conv1d(24, 40, 7, stride=2, padding=3), nn.GroupNorm(5, 40), nn.SiLU(),
            nn.Conv1d(40, 48, 5, stride=2, padding=2), nn.GroupNorm(6, 48), nn.SiLU(),
            nn.AdaptiveAvgPool1d(8), nn.Flatten(), nn.Linear(384, 160), nn.LayerNorm(160), nn.SiLU(),
        )
        self.head = nn.Linear(160, 10)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x):
        logits, fused = self.raw(x)
        traces = self.trace(x)
        features = self.encoder(traces.flatten(0, 1)).reshape(len(x), x.shape[1], -1).mean(1)
        return logits + self.head(features), fused + .10 * features
