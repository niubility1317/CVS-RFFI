"""Fit a LoRa conditional complex mean and use its uncertainty for readout."""

import torch
from torch import nn

from .band_allocation import fft, ifft
from .channel_blind_source import waveform_risk
from .common_anchor import iq_field
from .cross_frequency import CrossFrequencyPrior
from .dura_repair import complex_iq, real_iq
from .joint_spectrum import limit, split
from .joint_unroll_source import coordinates
from .natural_detail import paired_inputs, transfer_risk, weak_difference
from .phase_dose_bursts_source import burst_observation
from .received_update import ReceivedUpdate
from .variance_transport import posterior_gain


MAX_RMS_FRACTION = .65
FIXED_STRENGTH = .75
VARIANCE_FLOOR = .001


def condition_waveforms(clean, rng):
    """Generic additive and mixed observations with unreported draw parameters."""
    shape = (len(clean),1,1)
    sample = torch.rand(shape,device=clean.device,dtype=clean.real.dtype,generator=rng)
    level = torch.exp(-3.9 + 3.65*sample)
    scale = clean.abs().square().mean((1,2),keepdim=True).sqrt()
    standard = torch.randn(clean.shape,device=clean.device,dtype=clean.dtype,generator=rng)
    gaussian = clean + level*scale*standard
    mixed = burst_observation(clean,rng)
    branch = torch.rand(shape,device=clean.device,dtype=clean.real.dtype,generator=rng)
    return torch.where(branch<.55,gaussian,mixed)


def transport_strength(variance, measured):
    # Under a centered scalar Gaussian with known additive noise, this maps
    # the posterior-mean correction to the correction that preserves variance.
    fraction = (variance / measured.clamp_min(1e-8)).clamp(.002, 1.)
    return (1. + fraction.sqrt()).reciprocal()


def predictor(channels):
    conv = nn.Sequential(nn.Conv1d(channels,32,5,padding=2), nn.SiLU(),
                         nn.Conv1d(32,48,5,padding=4,dilation=2), nn.SiLU(),
                         nn.Conv1d(48,32,3,padding=1), nn.SiLU(), nn.Conv1d(32,3,1))
    nn.init.normal_(conv[-1].weight, std=.005)
    nn.init.zeros_(conv[-1].bias)
    return conv


class PosteriorTransport(nn.Module):
    def __init__(self, *, adaptive):
        super().__init__()
        self.adaptive = adaptive
        self.prior = CrossFrequencyPrior()
        self.coarse = predictor(12)
        self.fine = predictor(17)
        frequency = torch.fft.fftshift(torch.fft.fftfreq(256))
        self.register_buffer('frequency',frequency,persistent=False)
        common = torch.tensor([1.,0.,0.,0.,0.,0.]).view(1,6,1)
        self.register_buffer('common',common,persistent=False)

    def spectral(self,wave):
        return self.prior.basis.to(wave) @ torch.fft.fftshift(fft(wave),dim=-1)

    def waveform(self,spectrum):
        rotated = self.prior.basis.T.to(spectrum) @ spectrum
        return ifft(torch.fft.ifftshift(rotated,dim=-1))

    def analytic_mean(self, spectrum, quadrature_variance):
        coordinates = torch.view_as_real(spectrum).flatten(-2) - self.prior.center
        edits = []
        for component, block in enumerate((coordinates[:,:1],coordinates[:,1:])):
            axis = self.prior.joint_axes[component]
            gain = posterior_gain(self.prior.joint_variance[component],
                                  quadrature_variance[:,component,None,None])
            edit = ((block @ axis)*(gain-1.)) @ axis.T
            edits.append(torch.view_as_complex(edit.unflatten(-1,(255,2)).contiguous()))
        change = torch.cat(edits,1)
        return torch.fft.fftshift(torch.cat((torch.zeros_like(change[:,:,:1]),change),2),dim=-1)

    def observe(self,y,*,use_action=True,use_noise=True):
        unshifted = self.prior.spectrum(y)
        original = self.spectral(y)
        measured = self.prior.noise(unshifted)
        noise = measured if use_noise else torch.full_like(measured,.04)
        action, reference, uncertainty = ReceivedUpdate.estimate(y,noise.mean(-1).reshape(-1,1,1))
        if not use_action:
            action = torch.ones_like(action)
        analytic = self.analytic_mean(unshifted,noise)
        variance = 2.*torch.cat((noise[:,:1], noise[:,1:].expand(-1,5)),1).unsqueeze(-1)
        variance = variance.clamp_min(VARIANCE_FLOOR)
        discrepancy = self.spectral(action.conj()*(y-action*y))
        raw_features = (*iq_field(original,analytic,self.spectral(reference),discrepancy),
                        (12.*variance).log1p(), (4.*uncertainty.mean((1,2))).view(-1,1,1).log1p(),
                        self.frequency, self.common)
        features = tuple(x.expand_as(original.real) for x in raw_features)
        return original,analytic,variance,action,uncertainty,features

    @staticmethod
    def read(module,features):
        values = torch.stack(features,2).flatten(0,1)
        mixed = module.training and values.is_cuda and torch.cuda.is_bf16_supported()
        with torch.autocast('cuda',dtype=torch.bfloat16,enabled=mixed):
            output = module(values)
        return output.float().unflatten(0,features[0].shape[:2]).unbind(2)

    def prediction(self, output, center, variance):
        real, imag, log_var = output
        mean_edit = center + torch.complex(real.tanh(),imag.tanh())*variance.sqrt()
        posterior_variance = (variance * torch.exp(3.5*log_var.tanh())).clamp_min(VARIANCE_FLOOR)
        return mean_edit*(self.frequency!=0.),posterior_variance

    def estimate(self,y,rms,context,*,use_feedback=True):
        _,analytic,variance,action,_,features = context
        first,v0 = self.prediction(self.read(self.coarse,features),analytic,variance)
        bounded = limit(self.waveform(first),y,rms,y.real.new_tensor(MAX_RMS_FRACTION))
        first = self.spectral(bounded)
        predicted = y+bounded
        residual = action.conj()*(y-action*predicted)/(.15+action.abs().square())
        response = self.spectral(residual)
        if not use_feedback:
            response = torch.zeros_like(response)
        update = .25*(v0/(v0+variance))*response
        representation = (*features,*iq_field(first,response),(8.*v0).log1p())
        final,v1 = self.prediction(self.read(self.fine,representation),first+update,variance)
        final = self.spectral(limit(self.waveform(final),y,rms,y.real.new_tensor(MAX_RMS_FRACTION)))
        return (first,v0),(final,v1)

    def readout(self,y,rms,posterior,variance,*,adaptive=None):
        mean,ratio = posterior
        adaptive = self.adaptive if adaptive is None else adaptive
        strength = transport_strength(ratio,variance) if adaptive else FIXED_STRENGTH
        return limit(self.waveform(mean*strength),y,rms,y.real.new_tensor(MAX_RMS_FRACTION))


@torch.no_grad()
def initialize(model,tracker,training,_calibration):
    model.prior.calibrate(coordinates(tracker, batch)[0] for batch in training.split(96))


def objective(model,tracker,iq,rng,*,with_identity,recipe,step):
    ground = complex_iq(iq)
    source,rotation,_,rms_ground = coordinates(tracker,iq)
    source_frame = rotation*rms_ground
    weak = weak_difference(source)
    change = weak*source_frame
    pairs = paired_inputs(ground,recipe.perturb,rng,change,with_identity=with_identity)
    y,rotation,_,rms = coordinates(tracker,real_iq(pairs.flatten(0,1)))
    frame = rotation*rms
    reference = torch.stack((ground,ground,ground+change),1).flatten(0,1)/frame
    context = model.observe(y)
    spectrum,_,noise,action,_,_ = context
    first,last = model.estimate(y,rms,context)
    edits = [model.readout(y,rms,stage,noise) for stage in (first,last)]
    outputs = [y+edit for edit in edits]
    truth = model.spectral(reference)
    active = model.frequency!=0.
    proper = []
    for change,variance in (first,last):
        square_error = (spectrum+change-truth).abs().square()
        proper.append((square_error[...,active]/variance[...,active] + variance[...,active].log()).mean())
    posterior = .5*(proper[0]+proper[1])
    waveform = [waveform_risk(recovered,reference) for recovered in outputs]
    restored = (outputs[1]*frame).unflatten(0,pairs.shape[:2])
    actual = split((restored[:,2]-restored[:,0])/source_frame)
    expected = split(weak)
    preservation = transfer_risk(actual[:,:1],expected[:,:1])+transfer_risk(actual[:,1:],expected[:,1:])
    observed = (y-action*outputs[1]).abs().square().mean((1,2))
    noise_expected = noise.mean((1,2))
    consistency = ((observed/(.01+noise_expected)-1.2).clamp_min(0.).square()).mean()
    identity = ((edits[1]*frame).unflatten(0,pairs.shape[:2])[-2:,:2].abs().square().mean()
                if with_identity else y.real.new_zeros(()))
    values = torch.stack((*waveform,posterior,preservation,consistency,identity))
    a,b,_ = recipe.loss_weights(step)
    return values @ values.new_tensor((a,b,.10,.24,.02,.22)), values
