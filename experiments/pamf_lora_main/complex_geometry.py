"""Joint LoRa recovery with a source-derived weak-spectrum relation loss."""

import torch
from torch import nn

from .dura_repair import complex_iq, real_iq
from .joint_unroll_source import coordinates
from .natural_detail import paired_inputs, transfer_risk, weak_difference
from .spectral_block_recovery import head


class ComplexGeometry(nn.Module):
    def __init__(self, geometry):
        super().__init__()
        self.geometry = geometry
        self.coarse = head(9,2)
        self.refine = head(10,2)
        f = torch.fft.fftshift(torch.fft.fftfreq(256))
        self.register_buffer('frequency',f,persistent=False)
        self.register_buffer('noncarrier',f!=0,persistent=False)
        self.register_buffer('outer',f.abs()>.25,persistent=False)
        self.register_buffer('bins',torch.stack(((f.abs()>0)&(f.abs()<=1/32),
                                                  (f.abs()>1/32)&(f.abs()<=1/8),
                                                  (f.abs()>1/8)&(f.abs()<=1/4),f.abs()>1/4)),
                             persistent=False)
        for net in (self.coarse,self.refine):
            nn.init.normal_(net[-1].weight,std=.01)
            nn.init.zeros_(net[-1].bias)

    @staticmethod
    def spectral(x):
        return torch.fft.fftshift(torch.fft.fft(x,norm='ortho'),dim=-1)

    @staticmethod
    def temporal(x):
        return torch.fft.ifft(torch.fft.ifftshift(x,dim=-1),norm='ortho')

    def observe(self, y):
        spectrum = self.spectral(y)
        mean = spectrum.mean(1,keepdim=True)
        support = (spectrum.sum(1,keepdim=True)-spectrum)/(y.shape[1]-1)
        spread = (spectrum-mean).abs().square().mean(1,keepdim=True)
        power = spectrum[...,self.outer].abs().square().mean(-1,keepdim=True)
        independent = spread[..., self.outer].mean(-1,keepdim=True)*y.shape[1]/(y.shape[1]-1)
        return spectrum,support,spread,power,independent

    @staticmethod
    def residual(net, fields, mixed):
        with torch.autocast('cuda',dtype=torch.bfloat16,enabled=mixed):
            out = net(torch.stack(fields,dim=2).flatten(0,1))
        a,b = out.float().unflatten(0,fields[0].shape[:2]).unbind(2)
        return torch.complex(a,b)

    def bounded(self, edit, bound):
        dose = (bound/edit.abs().clamp_min(1e-6)).clamp_max(1.)
        return edit*dose*self.noncarrier

    def forward(self, y, quality):
        x, support, spread, power, independent = self.observe(y)
        power_feature = power.log1p().expand_as(x.real)
        repeat_feature = independent.log1p().expand_as(x.real)
        q = quality.expand_as(x.real)
        mixed = self.training and y.is_cuda and torch.cuda.is_bf16_supported()
        features = (x.real,x.imag,support.real,support.imag,spread.log1p().expand_as(q),power_feature,
                    repeat_feature,q,self.frequency.expand_as(q))
        activation = power/(power+.025)
        bound = .60*(x.abs()+.15*power.clamp_min(1e-8).sqrt())
        rough = self.bounded(self.residual(self.coarse,features,mixed)*activation,bound)
        coarse = self.temporal(x+rough)
        mean = (coarse.sum(1,keepdim=True)-coarse)/(y.shape[1]-1)
        innovation = y-coarse
        correction = self.residual(self.refine,(y.real,y.imag,coarse.real,coarse.imag,mean.real,mean.imag,
                                   innovation.real,innovation.imag,power_feature,repeat_feature),mixed)
        fine = self.bounded(rough+.60*activation*self.spectral(correction),bound)
        return coarse,self.temporal(x+fine)


def relation(spectrum, band):
    """Normalized complex Gram of repeat residuals, referenced to packet energy."""
    x = spectrum[...,band]
    residual = x-x.mean(-2,keepdim=True)
    # Compute terms directly to avoid TensorFloat32 or AMP on phase products.
    cross = (residual.unsqueeze(-2)*residual.conj().unsqueeze(-3)).sum(-1)
    energy = residual.abs().square().sum((-1,-2),keepdim=True)
    return cross/(.004*band.sum()+energy).clamp_min(1e-8)


def objective(model, tracker, iq, rng, *, with_identity, recipe, step):
    source = complex_iq(iq)
    aligned,source_phase,_,source_rms = coordinates(tracker,iq)
    basis = source_phase*source_rms
    weak = weak_difference(aligned)
    delta = weak*basis
    exposures = paired_inputs(source,recipe.perturb,rng,delta,with_identity=with_identity)
    y, phase, quality, rms = coordinates(tracker,real_iq(exposures.flatten(0,1)))
    frame = phase*rms
    clean = torch.stack((source,source,source+delta),dim=1).flatten(0,1)/frame
    coarse, fine = model(y,quality)
    restored = torch.stack((coarse,fine))
    clean_spec = model.spectral(clean)
    error = model.spectral(restored-clean).abs().square()
    power = clean_spec.abs().square()
    risks = torch.stack([(error[..., b].mean(-1)/(.002+power[..., b].mean(-1))).log1p().mean((1,2))
                         for b in model.bins]).mean(0)
    full = (fine*frame).unflatten(0,exposures.shape[:2])
    transfer = transfer_risk((full[:,2]-full[:,0])/basis,weak)
    # Compare the off-diagonal complex relations as well as the relative
    # energy on the diagonal, without providing any latent identity.
    fine_spec = model.spectral(fine)
    geometry = torch.stack([(relation(fine_spec,b)-relation(clean_spec,b)).abs().square().mean()
                            for b in model.bins[:2]]).mean()
    factor = min(1.,max(0.,(step-4000)/8000.))
    loss = (.55-.20*factor)*risks[0]+(.45+.20*factor)*risks[1]+(.15+.20*factor)*transfer
    if model.geometry:
        loss = loss+(10.+20.*factor)*geometry
    return loss,torch.cat((risks,transfer.view(1),geometry.view(1)))
