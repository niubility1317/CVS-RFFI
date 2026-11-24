"""Two-stage WiSig DURA with a fitted preamble response in its observation term."""

import numpy as np
import torch
from torch import nn

from experiments.pamf_reception_restore.models import Core
from .wisig_anchored_window import AnchoredWindow, cnn, generic_crop, iq_fields
from .wisig_observed_solve import Action, solve
from .wisig_smooth_preamble import IN_ANCHOR, OBSERVED, SMOOTH, STARTS, TIME, WEIGHT, blend_weights, waveform
from .wisig_timing import NOMINAL, long_template
from .wisig_uncertain_prior import normalized_error


class PreambleDura(nn.Module):
    """Fit and infer in an aligned latent window, with data consistency on measured samples."""

    def __init__(self, anchored):
        super().__init__()
        self.anchored = anchored
        light = (16, 24, 32, 48)
        wide = (24, 40, 60, 80)
        self.frequency = Core(incoming=5, widths=light)
        self.coarse = Core(incoming=12, widths=light)
        self.estimator = cnn(12, 24, 9)
        self.priors = nn.ModuleList(Core(incoming=19, widths=wide) for _ in range(2))
        shifts = STARTS - NOMINAL
        coordinate = TIME[None, :] + shifts[:, None]
        arrays = dict(time=TIME, shifts=shifts, coordinate=coordinate, center=IN_ANCHOR,
                      columns=OBSERVED, window_weight=WEIGHT.astype(np.float32),
                      interpolation=blend_weights(coordinate),
                      canonical=waveform(TIME-NOMINAL).astype(np.complex64),
                      long_extension=long_template(np.arange(80)).astype(np.complex64),
                      mirror=(-TIME) % len(TIME))
        for index, (columns, projector, gram, inverse) in enumerate(SMOOTH):
            arrays.update({f'columns_{index}': columns, f'projection_{index}': projector,
                           f'gram_{index}': gram, f'inverse_{index}': inverse})
        dtypes = {'b': torch.bool, 'i': torch.int64, 'f': torch.float32, 'c': torch.complex64}
        for name, values in arrays.items():
            self.register_buffer(name, torch.tensor(values.tolist(), dtype=dtypes[values.dtype.kind]), persistent=False)

    oscillator = staticmethod(AnchoredWindow.oscillator)

    def preamble_fit(self, dephased):
        selected = dephased[:, self.columns] * self.window_weight
        terms = []
        for index in range(3):
            observed = selected[:, getattr(self, f'columns_{index}')]
            rhs = (observed @ getattr(self, f'projection_{index}')).reshape(len(observed), len(self.shifts), 4)
            estimate = torch.einsum('dij,bdj->bdi', getattr(self, f'inverse_{index}'), rhs)
            terms.append((rhs, estimate, observed.abs().square().sum(-1, keepdim=True)))
        rhs, fitted, power = terms[0]
        scores = (fitted.conj()*rhs).real.sum(-1) / power.clamp_min(1e-8)
        predicted = []
        for fit, test in ((1, 2), (2, 1)):
            rhs = terms[test][0]
            gain = terms[fit][1]
            gram_gain = torch.einsum('dij,bdj->bdi', getattr(self, f'gram_{test}'), gain)
            predicted.append((gain.conj()*(2*rhs-gram_gain)).real.sum(-1))
        held = sum(predicted) / power.clamp_min(1e-8)
        outer, index = scores.masked_fill(self.center, -torch.inf).max(-1)
        peak = scores.masked_fill(~self.center, -torch.inf).amax(-1)
        margin = (outer-peak).clamp_min(0.)
        index = torch.where(margin > 0., index, len(self.shifts)//2)
        row = torch.arange(len(index), device=index.device)
        return index, fitted[row, index], held[row, index].clamp(0., 1.), margin

    def analyze(self, received):
        level = received.abs().square().mean(-1, keepdim=True).clamp_min(1e-8).sqrt()
        raw = received/level
        rotation, omega, variance = self.oscillator(raw, self.time)
        index, fitted, confidence, margin = self.preamble_fit(raw*rotation.conj())
        observed_index = self.coordinate[index]
        support = (observed_index>=0) & (observed_index<len(self.time))
        y = raw.gather(-1, observed_index.clamp(0, len(self.time)-1))
        envelope = (self.interpolation[index]*fitted[:, None]).sum(-1)
        phase = torch.polar(torch.ones_like(raw.real), omega[:, None]*observed_index)
        guide = self.canonical * phase * envelope
        # Anchor to the mean received response over verified short training samples.
        # This leaves the packet-wide complex gain to the learned residual and prior.
        anchor = envelope[:, 64:144].mean(-1, keepdim=True)
        relative = envelope * anchor.conj() / (.03 + anchor.abs().square())
        relative = relative / relative.abs().clamp_min(1e-5) * relative.abs().clamp(.40, 2.0)
        trust = confidence[:, None] * (anchor.abs().square() / (.03 + anchor.abs().square()))
        nominal = 1. + trust*(relative-1.)
        return raw, torch.where(support, y, guide), level, support, guide, nominal, variance, confidence, self.shifts[index], margin

    @staticmethod
    def predict(network, fields):
        tensor = torch.stack(fields, dim=1)
        amp = network.training and tensor.is_cuda and torch.cuda.is_bf16_supported()
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=amp):
            return network(tensor).float()

    @classmethod
    def correction(cls, network, fields):
        return torch.complex(*cls.predict(network, fields).tanh().unbind(1))

    def stages(self, analysis, *, anchor=None, use_noise=True, use_cg=True):
        raw, y, level, support, guide, nominal, noise, confidence, shift, margin = analysis
        sigma = torch.log1p(noise[:, None]).expand_as(y.real)
        reliability = confidence[:, None].expand_as(y.real)
        t = (self.time/255.).expand_as(y.real)
        fields = (*iq_fields(raw, y, guide), support.to(y.real.dtype), sigma, reliability, t)
        spectrum = torch.fft.fft(y, norm='ortho')
        spectral = self.correction(self.frequency, (*iq_fields(spectrum, spectrum[:, self.mirror].conj()), sigma))
        spectral = torch.fft.ifft(spectral, norm='ortho')
        coarse = y + spectral + 1.4*self.correction(self.coarse, (*fields, *iq_fields(spectral)))
        gain_re, gain_im, img_re, img_im, left_re, left_im, right_re, right_im, extra = self.predict(
            self.estimator, (*fields, *iq_fields(coarse))).tanh().unbind(1)
        enabled = self.anchored if anchor is None else anchor
        gain = (nominal if enabled else 1.) + .35*torch.complex(gain_re, gain_im)
        action = Action(gain, .15*torch.complex(img_re, img_im),
                        .20*torch.complex(left_re, left_im), .20*torch.complex(right_re, right_im))
        variance = (.018 + noise[:, None]) * (1.5*extra).exp()
        used = variance if use_noise else torch.full_like(variance, .16)
        weight = support / (.06 + used)
        current, stages = coarse, []
        for network in self.priors:
            residual = action.adjoint(weight*(y-action.apply(current)))
            innovation = residual / (.005 + residual.abs().square().mean(-1, keepdim=True)).sqrt()
            components = (*fields, *iq_fields(coarse, current, innovation, action.gain-1.), used.log().clamp(-7., 3.)/4.)
            prior = coarse + 1.3*self.correction(network, components)
            current = solve(action, y, prior, weight, 2.2, coupled=True) if use_cg else prior
            stages.extend((prior, current))
        return (coarse, *stages), action, variance, level, shift, margin

    def restore(self, analysis):
        stages, _, variance, level, shift, margin = self.stages(analysis)
        return tuple(value*level for value in stages), shift, margin, variance.mean(-1)


def perturb(source, generator, model):
    observed = generic_crop(source, generator, model)
    draw = torch.rand((len(source), 4), device=source.device, generator=generator)
    time = model.time/255.
    angle = (2.*draw[:, 0:1]-1.)*torch.pi + (2.*draw[:, 1:2]-1.)*2.*time
    angle = angle + 1.1*(2.*draw[:, 2:3]-1.)*torch.sin((1. + 8.*draw[:, 3:4])*time)
    return observed * torch.polar(torch.ones_like(angle), angle)


def objective(model, source, generator, *, identity):
    views = torch.stack((perturb(source, generator, model), perturb(source, generator, model)), dim=1)
    if identity:
        views[-3:] = source[-3:, None]
    reference = source[:, None].expand_as(views).flatten(0, 1)
    analysis = model.analyze(views.flatten(0, 1))
    values, action, variance, level, _, _ = model.stages(analysis)
    c, _, first, _, final = (value*level for value in values)
    y, support = analysis[1], analysis[3]
    squared = (y - action.apply(reference/level)).abs().square()
    observed = support.sum().clamp_min(1.)
    forward = (squared*support).sum()/observed
    effective_variance = (support*(squared.detach()/variance + variance.log())).sum()/observed
    paired = final.unflatten(0, views.shape[:2])
    consistency = normalized_error(paired[:, 0], paired[:, 1])
    spectrum = torch.fft.fft(reference, norm='ortho').abs().square()
    difference = torch.fft.fft(final-reference, norm='ortho').abs().square()
    weak = (difference / (.025 + spectrum)).log1p().mean()
    terms = torch.stack((normalized_error(c, reference), normalized_error(first, reference),
                         normalized_error(final, reference), forward, effective_variance, consistency, weak))
    return terms @ terms.new_tensor((.35, .35, 1., .10, .02, .10, .14)), terms
