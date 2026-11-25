"""A source-only, two-stage WiSig restorer with a native-window receive action."""

from math import log

import torch
from torch import nn
from torch.nn import functional as F

from experiments.pamf_action_restore.models import SCALE
from experiments.pamf_action_restore.operators import Action
from experiments.pamf_r65_inference.program import reused_map
from experiments.pamf_reception_restore.models import Core, mirror_spectrum
from experiments.pamf_restoration.models import zero_output

from .wisig_uncertain_prior import as_complex, as_iq, natural_delta, normalized_error, perturb


SAMPLES, RADIUS, KERNEL = 256, 40, 81
WIDTHS = (16, 24, 36, 48)
VARIANCE_LOW, VARIANCE_HIGH = 1e-6, .5


def reflect(index):
    return torch.where(index < 0, -index, torch.minimum(index, 2*(SAMPLES-1)-index))


class WindowAction(Action):
    """Use the original real-linear FIR and adjoint on explicit reflected support."""

    def __init__(self, kernel, main, image, index):
        self.kernel, self.main, self.image = kernel, main, image
        self.count, self.length = main.shape[-1], kernel.shape[-1]
        self.index = index
        self.nfft = 1 << (len(index)+self.length-2).bit_length()
        self.response = torch.fft.fft(kernel, n=self.nfft)


class NativeFIR(nn.Module):
    def __init__(self, centered):
        super().__init__()
        self.center = RADIUS if centered else 0
        self.spectral = Core(incoming=4, outgoing=4, widths=(12, 16, 24, 32))
        self.coarse = Core(incoming=4, widths=WIDTHS)
        self.estimator = Core(incoming=4, outgoing=4, widths=WIDTHS)
        self.global_head = zero_output(nn.Linear(3*WIDTHS[-1], 2*KERNEL+1))
        self.priors = nn.ModuleList(Core(incoming=9, widths=WIDTHS) for _ in range(2))
        self.register_buffer('time', torch.arange(SAMPLES), persistent=False)
        start = self.center-(KERNEL-1)
        self.register_buffer('support', reflect(torch.arange(start, SAMPLES+self.center)), persistent=False)
        impulse = torch.zeros(KERNEL)
        impulse[self.center] = 1.
        self.register_buffer('impulse', impulse, persistent=False)

    def analyze(self, received):
        if not received.is_complex() or received.ndim != 2 or received.shape[-1] != SAMPLES:
            raise ValueError('Only the current native 256-point complex WiSig window is required')
        level = received.abs().square().mean(-1, keepdim=True).clamp_min(1e-8).sqrt()
        value = received/level
        spectrum = torch.fft.fft(value, norm='ortho')
        mirror = mirror_spectrum(spectrum)
        iq = as_iq(value)
        mixed = self.training and value.is_cuda and torch.cuda.is_bf16_supported()
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=mixed):
            correction = self.spectral(torch.cat((as_iq(spectrum), as_iq(mirror)), 1)).float().tanh()
        spectral = (1.+.35*as_complex(correction[:, :2]))*spectrum + .2*as_complex(correction[:, 2:])*mirror
        spectral_iq = as_iq(torch.fft.ifft(spectral, norm='ortho'))
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=mixed):
            base = spectral_iq+.35*self.coarse(torch.cat((iq, spectral_iq), 1)).float().tanh()
            features = self.estimator.encoder(torch.cat((iq, base), 1))
            hidden = features[-1]
            points = hidden.shape[-1]//4
            pooled = torch.cat((hidden.mean(-1), hidden[..., :points].mean(-1), hidden[..., -points:].mean(-1)), 1)
            global_value = self.global_head(pooled).float()
            for block, skip in zip(self.estimator.decode, features[-2::-1], strict=True):
                hidden = block(torch.cat((F.interpolate(hidden, size=skip.shape[-1], mode='linear',
                                                         align_corners=False), skip), 1))
            local = self.estimator.out(hidden).float().tanh()

        kernel = torch.complex(global_value[:, :KERNEL]+self.impulse, global_value[:, KERNEL:2*KERNEL])
        kernel = kernel/kernel.abs().square().sum(-1, keepdim=True).clamp_min(1e-8).sqrt()
        main = torch.polar((.55*local[:, 0]).exp(), 1.20*local[:, 1])
        image = .18*as_complex(local[:, 2:])
        log_variance = log(VARIANCE_LOW)+log(VARIANCE_HIGH/VARIANCE_LOW)*(global_value[:, -1:]+.8).sigmoid()
        return value, level, base, kernel, main, image, log_variance.exp()

    def gate_scores(self, analysis):
        kernel, variance = analysis[3], analysis[-1]
        deviation = (kernel-self.impulse).abs().square().sum(-1)
        return deviation, variance[:, 0]

    def stages(self, analysis):
        value, level, base, kernel, main, image, variance = analysis
        action = WindowAction(kernel, main, image, self.support)
        iq = as_iq(value)
        current = as_complex(base)
        logvar = variance.detach().log().clamp(-16, 4)[:, :, None].expand(-1, -1, SAMPLES)/8
        output = [current*level]
        mixed = self.training and value.is_cuda and torch.cuda.is_bf16_supported()
        for network in self.priors:
            innovation = action.adjoint(value-action(current))
            features = as_iq(innovation)
            features = features/(features.square().mean((1, 2), keepdim=True)+1e-6).sqrt()
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=mixed):
                prior = as_complex(base+SCALE*network(torch.cat((iq, base, as_iq(current), features, logvar), 1)).float())
            current = reused_map(action, prior, current, variance.detach(), innovation)
            output.extend((prior*level, current*level))
        return tuple(output), action

    def restore(self, analysis):
        stages, _ = self.stages(analysis)
        peak = analysis[3].abs().argmax(-1)-self.center
        return stages, peak, *self.gate_scores(analysis)


def observe(source, rng, model):
    """Generate broad nuisance observations without returning any action parameters."""
    wave = perturb(source, rng)
    shape = (len(source), 1, 1)

    def draw():
        return torch.rand(shape, device=source.device, generator=rng)

    t = model.time/float(SAMPLES-1)
    path = (2*torch.pi*((.3+1.7*draw())*t+draw())).sin()
    strength = draw()*(draw() > .35)
    carrier = torch.polar((.38*strength*path).exp(), 1.2*strength*path)
    mirror = .14*strength*torch.polar(torch.ones_like(path), 2*path)
    wave = carrier*wave+mirror*wave.conj()
    offset = torch.randint(-RADIUS, RADIUS+1, shape, device=wave.device, generator=rng)
    index = reflect(model.time+offset*(draw() > .25))
    return wave.gather(-1, index.expand_as(wave))


def disturb(source, rng, model):
    return observe(source[:, None], rng, model)[:, 0]


def objective(model, source, rng, *, identity):
    detail = natural_delta(source)
    targets = torch.stack((source, source, source+detail), 1)
    shared = observe(targets[:, (0, 2)], rng, model)
    fresh = observe(targets[:, 1:2], rng, model)
    views = torch.stack((shared[:, 0], fresh[:, 0], shared[:, 1]), 1)
    if identity:
        views[-3:] = targets[-3:]
    clean = targets.flatten(0, 1)
    analysis = model.analyze(views.flatten(0, 1))
    stages, action = model.stages(analysis)
    measured, level, _, _, _, _, variance = analysis
    residual = (action(clean/level)-measured).abs().square()
    forward = residual.log1p().mean()
    likelihood = (.5*residual.detach()/variance+variance.log()).mean()
    result = stages[-1].unflatten(0, (len(source), 3))
    consistency = normalized_error(result[:, 1], result[:, 0])
    preserved = (result[:, 2]-result[:, 0]-detail).abs().square().mean(-1)
    transfer = (preserved/(.002+detail.abs().square().mean(-1))).log1p().mean()
    error = torch.fft.fft(stages[-1]-clean, norm='ortho').abs().square()
    weak = (error/(.025+torch.fft.fft(clean, norm='ortho').abs().square())).log1p().mean()
    costs = torch.stack((*(normalized_error(stage, clean) for stage in stages),
                         forward, likelihood, consistency, transfer, weak))
    weights = costs.new_tensor((.30, .13, .35, .12, 1., .75, .035, .10, .20, .12))
    return (weights*costs).sum(), costs
