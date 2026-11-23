"""Receiver-only WiSig frequency-block prior with repeated-preamble context."""

import torch
from torch import nn
import torch.nn.functional as F

from experiments.pamf_lora_main.spectral_block_recovery import head
from experiments.pamf_reception_restore.models import Core
from .wisig_uncertain_prior import as_complex, as_iq, perturb


PERIOD = 20
FIRST = 2 * PERIOD
LAST = 8 * PERIOD
SAMPLES = 256


def phase_reference(z):
    """The fixed center of the captured legacy STF is available in the packet."""
    body = z[..., FIRST:LAST].unflatten(-1, (-1, PERIOD))
    earlier, later = body[..., :-1, :], body[..., 1:, :]
    product = (later*earlier.conj()).sum(dim=(-1, -2))
    strength = (earlier.abs().square().sum((-1, -2))*later.abs().square().sum((-1,-2))).sqrt()
    coherence = (product.abs()/strength.clamp_min(1e-8)).clamp(0,1)
    omega = product.angle()/PERIOD
    time = torch.arange(SAMPLES, device=z.device, dtype=z.real.dtype)
    rotation = torch.exp(1j*omega[..., None]*time)
    start = FIRST
    end = LAST
    aligned = z*rotation.conj()
    cycles = aligned[...,start:end].unflatten(-1,(-1,PERIOD))
    consensus = cycles.mean(dim=-2,keepdim=True).expand_as(cycles).flatten(-2)
    taper = torch.arange(PERIOD, device=z.device, dtype=z.real.dtype)/(PERIOD-1)
    window = torch.cat((taper, taper.new_ones(end-start-2*PERIOD), taper.flip(0)))
    repeated = F.pad(consensus*window, (start,SAMPLES-end))*rotation
    difference = cycles[...,1::2,:]-cycles[...,::2,:]
    variance = .5*difference.abs().square().mean((-1,-2))
    body_power = body.abs().square().mean((-1,-2)).clamp_min(1e-8)
    return repeated, coherence, variance/body_power, aligned


def observable(z):
    repeat, coherence, variance, aligned = phase_reference(z)
    # A delayed transition prolongs periodicity beyond the nominal STF body.
    # This is a waveform mismatch indicator, not a scene identifier.
    a, b = aligned[:, 8*PERIOD:10*PERIOD], aligned[:, 9*PERIOD:11*PERIOD]
    tail_power = (a.abs().square()+b.abs().square()).sum(-1).clamp_min(1e-8)
    transition = (a-b).abs().square().sum(-1)/tail_power
    return repeat, coherence, variance, transition


class RepeatBlockRecovery(nn.Module):
    """Predict a masked spectrum, then refine the raw receive observation."""

    def __init__(self, *, masked):
        super().__init__()
        self.masked = masked
        group = torch.arange(SAMPLES).div(2, rounding_mode='floor') % 4
        self.register_buffer('hidden', group == torch.arange(4)[:,None], persistent=False)
        self.register_buffer('frequency', torch.fft.fftshift(torch.fft.fftfreq(SAMPLES)), persistent=False)
        self.prior = head(7, 2)
        self.detail = Core(incoming=9, widths=(24, 32, 48, 64))

    def prior_spectrum(self, spectrum):
        mask = self.hidden[:,None]
        visible = spectrum.unsqueeze(0).expand(len(mask),-1,-1)
        if self.masked:
            visible = visible.masked_fill(mask,0.)
        time = torch.fft.ifft(torch.fft.ifftshift(visible,dim=-1),norm='ortho')
        repeated, *_ = phase_reference(time)
        side = torch.fft.fftshift(torch.fft.fft(repeated,norm='ortho'),dim=-1)
        fields = torch.stack((visible.real, visible.imag, side.real, side.imag, visible.abs().log1p(),
                              self.frequency.expand_as(visible.real), mask.expand_as(visible).to(visible.real.dtype)),dim=-2)
        guess = self.prior(fields.flatten(0,1)).float().unflatten(0,visible.shape[:2])
        return (torch.complex(guess[...,0,:],guess[...,1,:])*mask).sum(0)

    def forward(self, y):
        repeated, quality, variance, _ = observable(y)
        spectrum = torch.fft.fftshift(torch.fft.fft(y,norm='ortho'),dim=-1)
        mixed = self.training and y.is_cuda and torch.cuda.is_bf16_supported()
        with torch.autocast('cuda',dtype=torch.bfloat16,enabled=mixed):
            prior = self.prior_spectrum(spectrum)
        prior = torch.fft.ifft(torch.fft.ifftshift(prior,dim=-1),norm='ortho')
        innovation = prior-y
        confidence = .30+.70*(variance/(variance+.10))
        coarse = y+.5*confidence[:,None]*innovation
        fields = torch.cat((as_iq(y),as_iq(coarse),as_iq(innovation),as_iq(repeated),
                            quality[:,None,None].expand(-1,1,y.shape[-1])),dim=1)
        with torch.autocast('cuda',dtype=torch.bfloat16,enabled=mixed):
            local = self.detail(fields)
        return coarse, coarse+.5*confidence[:,None]*as_complex(local.float()), variance


def generic_observation(source, generator):
    """Generic crop displacement augments the existing independent distortions."""
    noisy = perturb(source,generator)
    shift = torch.randint(-2*PERIOD, 2*PERIOD+1, (len(noisy),1,1), device=noisy.device, generator=generator)
    # Mix nonshifted inputs with shifted ones without relying on scene parameters.
    shift = shift*(torch.rand(shift.shape,device=noisy.device,generator=generator)>.35)
    index = torch.arange(noisy.shape[-1],device=noisy.device)+shift
    reflected = torch.where(index<0,-index,torch.minimum(index,2*(noisy.shape[-1]-1)-index))
    return noisy.gather(-1,reflected.expand_as(noisy.real))


def polar_readouts(raw, proposed):
    a = torch.fft.fft(raw,norm='ortho')
    b = torch.fft.fft(proposed,norm='ortho')
    # Use the other estimate's phase only when the required coefficient is zero.
    unit_a = torch.polar(torch.ones_like(a.real),a.angle())
    unit_b = torch.polar(torch.ones_like(b.real),b.angle())
    phase_a = torch.where(a!=0,unit_a,unit_b)
    phase_b = torch.where(b!=0,unit_b,unit_a)
    magnitude = torch.fft.ifft(b.abs()*phase_a,norm='ortho')
    phase = torch.fft.ifft(a.abs()*phase_b,norm='ortho')
    return magnitude, phase
