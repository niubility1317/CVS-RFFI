"""WiSig joint timing and polar recovery with observed repeat curvature."""

from math import pi

import torch
from torch import nn

from .wisig_anchored_window import AnchoredWindow, cnn, generic_crop, normalized_error
from .wisig_timing import PERIOD


STF_BEGIN, STF_END, BLOCKS = 40, 160, 6


def differences(z):
    first, second = z[:, :-1], z[:, 1:]
    product = (second * first.conj()).sum(-1)
    low, high = first.abs().square().sum(-1), second.abs().square().sum(-1)
    coherence = product.abs() / (low * high).clamp_min(1e-10).sqrt()
    unit = product / product.abs().clamp_min(1e-7)
    angular = unit[:, 1:] * unit[:, :-1].conj()
    log_gain = .5 * ((high + 1e-7).log() - (low + 1e-7).log())
    logarithmic = log_gain.diff(dim=1)
    weight = torch.minimum(coherence[:, 1:], coherence[:, :-1]).clamp(0., 1.)
    return angular, logarithmic, weight


def weighted_curvature(raw):
    matrix = raw[:, STF_BEGIN:STF_END].reshape(-1, BLOCKS, PERIOD)
    a_phase, a_gain, a_quality = differences(matrix[..., ::2])
    b_phase, b_gain, b_quality = differences(matrix[..., 1::2])
    agreement = (a_phase * b_phase.conj()).real.clamp(0., 1.)
    gain_agreement = torch.exp(-4. * (a_gain - b_gain).abs())
    quality = torch.minimum(a_quality, b_quality) * agreement * gain_agreement
    phase = (a_phase + b_phase).angle()
    return torch.stack((.5 * (a_gain + b_gain),phase), dim=-1) * quality[..., None], quality


def polar_edit(base, log_amplitude, phase):
    return base * torch.polar(log_amplitude.exp(), phase)


class RepeatDrift(AnchoredWindow):
    def __init__(self, apply_observed):
        super().__init__(known_waveform=True)
        self.apply_observed = apply_observed
        self.coarse = cnn(10,24,2)
        self.refine = cnn(12,24,2)
        difference = torch.eye(BLOCKS).diff(n=2, dim=0)
        normal = difference @ difference.mT + .12 * torch.eye(BLOCKS-2)
        inverse = torch.linalg.solve(normal, difference)
        self.register_buffer('curvature_inverse', inverse, persistent=False)
        t = torch.arange(len(self.time), dtype=torch.float32)
        nodes = STF_BEGIN + (torch.arange(BLOCKS)+.5)*PERIOD - .5
        interpolation = (1. - ((t[:, None] - nodes)/PERIOD).abs()).clamp_min(0.)
        taper = torch.sin((pi*(t-STF_BEGIN+.5)/(STF_END-STF_BEGIN)).clamp(0.,pi)).square()
        taper[(t<STF_BEGIN) | (t>=STF_END)] = 0.
        coordinate = (t - (t*taper).sum()/taper.sum()) / (STF_END-STF_BEGIN)
        affine = torch.stack((torch.ones_like(t),coordinate),dim=1)
        projector = torch.linalg.solve(affine.mT@(affine*taper[:, None]),affine.mT*taper)
        self.register_buffer('interpolation', interpolation.mT, persistent=False)
        self.register_buffer('edge_taper', taper, persistent=False)
        self.register_buffer('affine', affine, persistent=False)
        self.register_buffer('remove_affine', projector, persistent=False)
        self.register_buffer('maximum_drift', torch.tensor([.35,.80])[None,:,None], persistent=False)

    def estimate_drift(self, raw):
        curvature, confidence = weighted_curvature(raw)
        nodes = curvature.transpose(1,2) @ self.curvature_inverse
        drift = nodes @ self.interpolation
        projected = (drift - (drift @ self.remove_affine.mT) @ self.affine.mT) * self.edge_taper
        # Uniform scaling keeps the projected mean and slope at zero.
        scale = (self.maximum_drift/projected.abs().amax(-1,keepdim=True).clamp_min(1e-7)).clamp_max(1.)
        log_amp, phase = (projected*scale).unbind(1)
        return log_amp,phase,confidence.mean(1)

    def stages(self, analysis, *, use_observed=None):
        raw,level,omega,noise,logits,a,b = analysis
        aligned,support,guide,delta,margin = self.select(raw,omega,logits,a,b)
        log_amp, phase, measured_confidence = self.estimate_drift(raw)
        received_index = (self.time[None]+delta[:, None]).clamp(0,raw.shape[-1]-1)
        log_amp, phase = (field.gather(1,received_index)*support for field in (log_amp,phase))
        apply = self.apply_observed if use_observed is None else use_observed
        coarse_base = polar_edit(aligned, -log_amp, -phase) if apply else aligned
        amplitude = aligned.abs()
        gauge = aligned.conj() / (.05 + amplitude)
        guide_in_gauge = guide * gauge
        shift_signal = torch.cat((aligned[:, :1],aligned[:, :-1]),dim=1)
        adjacent = shift_signal * gauge / (.08+shift_signal.abs())
        measured = noise[:, None].expand_as(support)
        confidence = measured_confidence[:, None].expand_as(support)
        reliability = .20 + .65 * measured / (.06+measured)
        features = (amplitude,guide_in_gauge.real,guide_in_gauge.imag,adjacent.real,adjacent.imag,
                    support,measured.log1p(),log_amp,phase,confidence)
        mixed = self.training and raw.is_cuda and torch.cuda.is_bf16_supported()
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=mixed):
            gain, turn = self.coarse(torch.stack(features,dim=1)).float().tanh().unbind(1)
        coarse = polar_edit(coarse_base,.45*reliability*gain,.95*reliability*turn)
        correction = (coarse-aligned)*gauge
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=mixed):
            gain, turn = self.refine(torch.stack((*features,correction.real,correction.imag),dim=1)).float().tanh().unbind(1)
        fine = polar_edit(coarse,.35*reliability*gain,.75*reliability*turn)
        return tuple(x*level for x in (aligned,coarse_base,coarse,fine)),delta,margin,noise

    def restore(self, analysis):
        (timed,_,coarse,fine),delta,margin,noise = self.stages(analysis)
        return (timed,coarse,fine),delta,margin,noise


def generic_drift(source, rng, model):
    time = model.time / source.shape[-1]
    batch = len(source)
    def rand():
        return torch.rand((batch,2,1),device=source.device,generator=rng)
    basis = (2*pi*(rand()+(.35+2.5*rand())*time)).sin()
    strength = rand() * (rand() > .2)
    phase = (.68*strength*basis).sum(1)
    log_amp = (.38*strength*(2*rand()-1)*basis).sum(1)
    return generic_crop(polar_edit(source, log_amp, phase), rng, model)


def polar_risk(output, reference):
    desired, observed = reference.abs(),output.abs()
    weight = desired / (.08 + desired)
    cosine = (output * reference.conj()).real / (desired*observed).clamp_min(1e-7)
    phase = (weight * (1. - cosine.clamp(-1.,1.))).mean()
    amplitude = ((observed+.04).log()-(desired+.04).log()).abs().mean()
    return phase, amplitude


def objective(model, source, rng, *, identity):
    draws = torch.stack((generic_drift(source,rng,model),generic_drift(source,rng,model)),dim=1)
    if identity:
        draws[-3:] = source[-3:,None]
    reference = source[:,None].expand_as(draws).flatten(0,1)
    (timed,coarse,fine),_,_,_ = model(draws.flatten(0,1))
    pair = fine.unflatten(0,draws.shape[:2])
    consistency = ((pair[:,0]-pair[:,1]).abs().square().mean(-1)
                   /source.abs().square().mean(-1).clamp_min(.01)).mean()
    phase, amplitude = polar_risk(fine, reference)
    truth = torch.fft.fft(reference,norm='ortho')
    error = torch.fft.fft(fine-reference,norm='ortho')
    spectrum = (error.abs().square()/(.025+truth.abs().square())).log1p().mean()
    metrics = torch.stack((normalized_error(timed,reference),normalized_error(coarse,reference),
                          normalized_error(fine,reference),phase,amplitude,consistency,spectrum))
    return (metrics*metrics.new_tensor((.30,.40,1.,.22,.10,.08,.13))).sum(),metrics
