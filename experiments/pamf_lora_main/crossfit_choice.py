"""Joint LoRa proposals with receive-side held-out directional evidence."""

import torch
from torch import nn
from torch.nn import functional as F

from .dura_repair import complex_iq, no_tone, real_iq
from .joint_unroll_source import coordinates
from .natural_detail import paired_inputs, transfer_risk, weak_difference
from .spectral_block_recovery import SpectralBlockRecovery
from .weak_signal_probe import observed_noise_rms


PROPOSALS = ('raw','self_coarse','self_fine','cross_coarse','cross_fine')


class CrossfitChoice(nn.Module):
    """Test correction directions on repeats excluded from proposal construction."""

    def __init__(self, signed):
        super().__init__()
        self.signed = signed
        self.restore = SpectralBlockRecovery(masked=True)
        self.selector = nn.Sequential(nn.Linear(27,48),nn.SiLU(),nn.Linear(48,len(PROPOSALS)-1))
        nn.init.zeros_(self.selector[-1].weight)
        nn.init.zeros_(self.selector[-1].bias)
        f = torch.fft.fftfreq(256).abs()*256
        self.register_buffer('bands',torch.stack(((f>0)&(f<=8),(f>=9)&(f<=32),f>32)),persistent=False)

    @staticmethod
    def common_move(observed, other):
        target = other.mean(dim=-2,keepdim=True)
        return .5*no_tone(target-observed.mean(dim=-2,keepdim=True))

    def heldout_score(self, raw, coarse, joint, spectrum):
        measurements = []
        for offset in (1,2):
            support = torch.arange(3,device=raw.device).add(offset).remainder(3)
            validation = torch.arange(3,device=raw.device).sub(offset).remainder(3)
            candidates = torch.stack((coarse-raw,joint-raw,
                self.common_move(raw,coarse[:,support]).expand_as(raw),
                .5*(joint-raw)+self.common_move(raw,joint[:,support])),dim=1)
            predicted = torch.fft.fft(candidates.mean(-2),norm='ortho')
            expected = spectrum[:,validation]-spectrum
            gain = 2*(predicted.conj()*expected[:,None]).real-predicted.abs().square()
            scale = expected.abs().square()[:,None]
            # The validator's own observed chirps never enter these proposals.
            stats = torch.stack([gain[...,band].sum(-1)/(.002+scale[...,band].sum(-1)) for band in self.bands],dim=-1)
            measurements.append(stats)
        network_input = torch.stack(measurements,dim=2).flatten(2,3)
        if not self.signed:
            network_input = network_input.abs()
        return torch.cat((network_input.mean(2).flatten(1),network_input.amin(2).flatten(1)),dim=1).tanh()

    def forward(self, observed, quality):
        # Each fold is a pair of nonadjacent chirps. Folds are never concatenated
        # inside the proposal network before the independent validation step.
        b, c, n = observed.shape
        raw = observed.reshape(b,2,3,n).transpose(1,2)
        q = quality.reshape(b,2,3,1).transpose(1,2)
        coarse, joint = self.restore(raw.flatten(0,1),q.flatten(0,1))
        coarse, joint = [x.unflatten(0,(b,3)) for x in (coarse,joint)]
        support_coarse = (coarse.sum(1,keepdim=True)-coarse)/2
        support_joint = (joint.sum(1,keepdim=True)-joint)/2
        candidates = torch.stack((raw,coarse,joint,raw+self.common_move(raw,support_coarse),
                                  .5*(raw+joint)+self.common_move(raw,support_joint)),dim=1)
        spectrum = torch.fft.fft(raw.mean(-2),norm='ortho')
        with torch.no_grad():
            score = self.heldout_score(raw,coarse,joint,spectrum)
            priors = torch.stack((quality.mean((1,2)),observed_noise_rms(observed).mean((1,2)),
                                  quality.squeeze(-1).amin(1)),dim=1)
            features = torch.cat((score,priors),dim=1)
        gains = self.selector(features)
        candidates = candidates.transpose(2,3).reshape(b,len(PROPOSALS),c,n)
        return candidates, gains


def decision(candidates, gains):
    utility = F.pad(gains,(1,0))
    choice = utility.argmax(1)
    hard = candidates[torch.arange(len(candidates),device=candidates.device),choice]
    soft = (candidates*(utility/.15).softmax(1)[:,:,None,None]).sum(1)
    return hard, soft, choice


def objective(model, tracker, iq, rng, *, with_identity, recipe, step):
    clean = complex_iq(iq)
    framed, rotation, _, rms = coordinates(tracker,iq)
    delta = weak_difference(framed)
    difference = delta*rotation*rms
    pair = paired_inputs(clean,recipe.perturb,rng,difference,with_identity=with_identity)
    observed, phase, quality, scale = coordinates(tracker,real_iq(pair.flatten(0,1)))
    basis = phase*scale
    desired = torch.stack((clean,clean,clean+difference),dim=1).flatten(0,1)/basis
    candidates, gains = model(observed,quality)
    error = torch.fft.fft(candidates-desired[:,None],norm='ortho').abs().square()
    power = torch.fft.fft(desired,norm='ortho').abs().square()[:,None]
    risks = torch.stack([(error[...,band].mean((-1,-2))/(.002+power[...,band].mean((-1,-2)))).log1p()
                        for band in model.bands]).mean(0)
    actual = (risks[:, :1]-risks[:, 1:]).detach().clamp(-2,2)
    forecast = F.smooth_l1_loss(gains,actual)
    expected = ((F.pad(gains,(1,0))/.15).softmax(1)*risks).sum(1).mean()
    health = risks[:,1:].mean()
    prediction = (candidates[:,1:]*basis[:,None]).unflatten(0,pair.shape[:2])
    transfer = transfer_risk((prediction[:,2]-prediction[:,0])/(rotation*rms)[:,None],delta[:,None])
    values = torch.stack((health,expected,forecast,transfer))
    loss = .4*health+.6*expected+.35*forecast+.35*transfer
    return loss,values
