"""Source-moment mixture posterior with an observation-only fusion network."""
import torch
from torch import nn
from .covariance_model import CovarianceMAP
from .multiscale import MultiscaleAttentionRestorer
from .mixture_gate_posterior import mixture_components, combine_components


class MixturePhysical(CovarianceMAP):
    def __init__(self):
        super().__init__()
        self.register_buffer('mixture_within', torch.eye(512))
        self.register_buffer('mixture_offsets', torch.zeros(4, 512))
        self.register_buffer('mixture_weights', torch.full((4,), .25))

    def bind_mixture(self, within, offsets, weights):
        if within.shape != (512, 512) or offsets.shape != (4, 512) or weights.shape != (4,):
            raise ValueError('Four source IQ mixture modes required')
        values = (within, offsets, weights)
        if not all(torch.isfinite(v).all() for v in values) or not (weights > 0).all():
            raise ValueError('Finite positive source mixture weights required')
        w, d, c = weights.double(), offsets.double(), within.double()
        if not torch.allclose(w.sum(), w.new_tensor(1.), rtol=1e-6, atol=1e-6):
            raise ValueError('Source mixture weights must sum to one')
        if (w @ d).abs().max() > 1e-5:
            raise ValueError('Source mixture mean must match original source prior')
        reconstructed = c + (d.T * w) @ d
        if not torch.allclose(reconstructed, self.prior_covariance.double(), rtol=1e-5, atol=1e-6):
            raise ValueError('Source mixture covariance must match original source prior')
        torch.linalg.cholesky(c)
        self.mixture_within.copy_(within.to(self.mixture_within))
        self.mixture_offsets.copy_(offsets.to(self.mixture_offsets))
        self.mixture_weights.copy_(weights.to(self.mixture_weights))


class FusionBlock(nn.Module):
    def __init__(self, dilation):
        super().__init__()
        self.layers = nn.Sequential(nn.GroupNorm(8, 128), nn.SiLU(),
            nn.Conv1d(128, 128, 3, padding=dilation, dilation=dilation),
            nn.GroupNorm(8, 128), nn.SiLU(), nn.Conv1d(128, 128, 1))

    def forward(self, x):
        return x + self.layers(x)


class MixtureFusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Conv1d(15, 128, 5, padding=2)
        self.blocks = nn.Sequential(*(FusionBlock(d) for d in (1, 2, 4, 8, 16, 32)))
        self.output = nn.Linear(128, 4)
        self.refiner = MultiscaleAttentionRestorer(context_channels=15)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, received, components, probability, variance):
        batch = len(received)
        iq = components.transpose(1, 2).reshape(batch, 8, 256).float()
        p = probability.float()[:, :, None].expand(-1, -1, 256)
        noise = variance.float().log()[:, None, None].expand(-1, 1, 256)
        features = torch.cat((received, iq, p, noise), dim=1)
        return self.output(self.blocks(self.stem(features)).mean(-1)), features


class MixtureGateRestorer(nn.Module):
    def __init__(self, seed):
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.physical = MixturePhysical()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed+1)
            self.detail = MixtureFusion()
        self.initialization_report = dict(mode='scratch_single_stage_four_source_mixture_hypotheses',
            model_seed=seed, detail_seed=seed+1, ancestor_checkpoints=[], training_stages=1,
            source_mixture_fit_uses_TX_labels=False, gate_initial_adjustment=0.,
            unconstrained_IQ_refiner=True, refiner_initial_residual=0.)

    def forward(self, received, *, return_aux=False):
        aux = self.physical.estimate(received)
        with torch.no_grad():
            components, evidence, residual = mixture_components(
                aux['action_matrix'].detach(), received.detach().flatten(1).double(),
                self.physical.prior_mean.double(), self.physical.mixture_within.double(),
                self.physical.mixture_offsets.double(), self.physical.mixture_weights.double(),
                aux['variance'].detach().double())
            base, base_probability = combine_components(components, evidence, torch.zeros_like(evidence))
        adjustment, context = self.detail(received.detach(), components, base_probability, aux['variance'].detach())
        combined, probability = combine_components(components, evidence, adjustment)
        combined = combined.float().reshape_as(received)
        restored = self.detail.refiner(combined, context)
        info = dict(**aux, base_iq=base.float().reshape_as(received), relative_solve_residual=residual,
                    mixture_probability=probability.float(), mixture_adjustment=adjustment)
        return (restored, info) if return_aux else restored


def scratch_mixture_gate(seed):
    return MixtureGateRestorer(seed)
