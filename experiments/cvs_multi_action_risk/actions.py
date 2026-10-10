"""State-conditioned polynomial L/T actions used to propose real IQ views.

The shared control uses exactly the same inputs, polynomial coefficients and
heads, sharing only the state trunk also accepted by the R action. No old
generic residual action is substituted for this control.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from .physics import (apply_action, phase_basis, quality_proxies, sample_parameters,
                      phase_process, CFO_LIMIT_HZ, PHASE_NOISE_LIMIT_RAD,
                      CURVATURE_PRESSURE_HZ_S)
from experiments.cvs_multi_disentangle.model import identity_from_intermediate, _encoder
from experiments.cvs_multi_state_action.actions import feature_scales, vector_margin

KINDS = ('linear', 'temporal')
STATE_DIM = 40
PARAMETER_BASIS_DIM = 7
PAIRS = tuple((i, j) for i in range(7) for j in range(i, 7))


def signal_state(x):
    z = torch.complex(x[:, 0], x[:, 1])
    power = z.abs().square()
    psd = torch.fft.fft(z).abs().square().reshape(len(x), 32, 8).mean(2)
    psd = torch.log1p(psd / psd.mean(1, keepdim=True).clamp_min(1e-10))
    normalized = power / power.mean(1, keepdim=True).clamp_min(1e-10)
    moments = torch.stack((normalized.square().mean(1), normalized.pow(3).mean(1),
                           normalized.max(1).values), 1)
    corr = (z[:, 1:] * z[:, :-1].conj()).mean(1) / power.mean(1).clamp_min(1e-10)
    return torch.cat((psd, quality_proxies(x), 8 * torch.tanh(moments / 8),
                      corr.real[:, None], corr.imag[:, None]), 1)


class SharedActionCore(nn.Module):
    """Identical independent/shared state processor for all three action types."""
    def __init__(self, hidden=64):
        super().__init__()
        self.hidden = hidden
        self.h_norm = nn.LayerNorm(349)
        self.state_norm = nn.LayerNorm(40)
        self.net = nn.Sequential(nn.Linear(349 + 40 + 3, hidden), nn.SiLU(),
                                 nn.Linear(hidden, hidden), nn.SiLU())

    def forward(self, h, state, kind):
        kind = {'L':'linear', 'T':'temporal', 'R':'receiver'}.get(kind, kind)
        if kind not in ('linear', 'temporal', 'receiver') or state.shape != (len(h), 40):
            raise ValueError('Shared state core requires registered action and [B,40] state')
        tag = h.new_zeros((len(h), 3))
        tag[:, ('linear', 'temporal', 'receiver').index(kind)] = 1
        return self.net(torch.cat((self.h_norm(h), self.state_norm(state), tag), 1))


def parameter_basis(p, kind, phase_noise=None):
    if p.ndim != 2 or p.shape[1] != 4:
        raise ValueError('Each action has four registered physical parameters')
    if kind == 'linear':
        return F.pad(p, (0, 3))
    if kind != 'temporal':
        raise ValueError('Unknown digital action')
    if phase_noise is None:
        if bool((p[:, 2] != 0).any()):
            raise ValueError('Stochastic temporal prediction needs realized phase process')
        modes = p.new_zeros((len(p), 3))
    else:
        basis = phase_basis(p)
        # Recover actual bounded random-series coefficients, rather than predict a
        # deterministic endpoint from a noise strength without its realization.
        modes = (phase_noise.to(p) @ basis.T) / basis.square().sum(1)
        modes = modes * (p[:, 2:3] / PHASE_NOISE_LIMIT_RAD)
    return torch.cat((p[:, :1].sin(), 1 - p[:, :1].cos(),
                      p[:, 1:2] / CFO_LIMIT_HZ, modes,
                      p[:, 3:4] / CURVATURE_PRESSURE_HZ_S), 1)


def polynomial_terms(q):
    return torch.cat((q, torch.stack([q[:, i] * q[:, j] for i, j in PAIRS], 1)), 1)


def _semantic_groups(kind):
    if kind == 'linear':
        return {'linear': tuple(range(35))}
    groups = {'phase': (0, 1), 'frequency': (2,), 'stochastic': (3, 4, 5), 'curvature_pressure': (6,)}
    terms = {name: list(indices) for name, indices in groups.items()}
    terms['interactions'] = []
    for index, (a, b) in enumerate(PAIRS, 7):
        owner = next((name for name, indices in groups.items() if a in indices and b in indices), 'interactions')
        terms[owner].append(index)
    return {key: tuple(value) for key, value in terms.items()}


class StateAction(nn.Module):
    """J(h,s)q + Q(h,s)vec_sym(qqT) and exact fixed 29D response.

    T heads share the recipient state and separate periodic phase, Hz CFO,
    realized stochastic phase, artificial curvature, and cross effects.
    """
    def __init__(self, kind, hidden=64, shared_core=None):
        super().__init__()
        if kind not in KINDS:
            raise ValueError('Use independent L/T modules sharing an explicit core for shared control')
        self.kind = kind
        self.parameter_dim, self.output_dim, self.nterms = 4, 320, 35
        self.core = shared_core if shared_core is not None else SharedActionCore(hidden)
        hidden = self.core.hidden
        self.term_groups = _semantic_groups(kind)
        self.heads = nn.ModuleDict({name: nn.Linear(hidden, len(indices) * 320)
                                   for name, indices in self.term_groups.items()})
        for head in self.heads.values():
            nn.init.normal_(head.weight, std=.001)
            nn.init.zeros_(head.bias)

    def forward(self, h, x, p, reference_fn=None, kind=None, phase_noise=None,
                cached_reference=None, cached_endpoint_reference=None, endpoint_x=None,
                cached_state=None):
        if kind is not None and kind != self.kind:
            raise ValueError('Action kind mismatch')
        q = parameter_basis(p, self.kind, phase_noise)
        basis = polynomial_terms(q)
        state = signal_state(x) if cached_state is None else cached_state
        hidden = self.core(h, state, self.kind)
        learned = h.new_zeros((len(h), 320))
        for name, indices in self.term_groups.items():
            coefficients = self.heads[name](hidden).reshape(len(h), len(indices), 320)
            learned = learned + (coefficients * basis[:, indices, None]).sum(1)
        if cached_reference is None or cached_endpoint_reference is None:
            if reference_fn is None:
                raise ValueError('Exact29 needs endpoint cache or actual frozen frontend')
            if cached_reference is None:
                cached_reference = reference_fn(x)
            if cached_endpoint_reference is None:
                endpoint_x = apply_action(x, p, self.kind, phase_noise) if endpoint_x is None else endpoint_x
                cached_endpoint_reference = reference_fn(endpoint_x)
        exact = cached_endpoint_reference - cached_reference
        if exact.shape != (len(h), 29):
            raise ValueError('Exact reference response must be [B,29]')
        return torch.cat((learned, exact), 1)

    predict = forward


def action_loss(identity, h, target_delta, pred_delta, y=None, scales=None,
                label_free=False, lambda_z=.1, lambda_margin=.1):
    scales = feature_scales(target_delta) if scales is None else scales
    feature = sum(width * (pred_delta[:, a:b] - target_delta[:, a:b]).square().mean() / scale
                  for width, (a, b), scale in zip((160/349,160/349,29/349),
                    ((0,160),(160,320),(320,349)), scales))
    target_z = identity_from_intermediate(identity, h + target_delta).detach()
    pred_z = identity_from_intermediate(identity, h + pred_delta)
    normalized_z = (F.normalize(pred_z, dim=1) - F.normalize(target_z, dim=1)).square().mean()
    classifier = _encoder(identity).classify_features
    target_logits, pred_logits = classifier(target_z).detach(), classifier(pred_z)
    if label_free:
        target_disc = target_logits - target_logits.mean(1, keepdim=True)
        pred_disc = pred_logits - pred_logits.mean(1, keepdim=True)
    else:
        if y is None:
            raise ValueError('Supervised action margin requires visible source L labels')
        target_disc, pred_disc = vector_margin(target_logits, y), vector_margin(pred_logits, y)
    margins = (pred_disc - target_disc).square().mean()
    return dict(loss=feature + lambda_z * normalized_z + lambda_margin * margins,
                feature=feature, normalized_z=normalized_z, margins=margins)


def conditional_edges(batch, hcache, step=0, enabled=True):
    """Two fit slots; replace one main edge by conditional edge, never add cost."""
    common = dict(phase_noise=batch['phase_noise'])
    if not enabled:
        pairs = [('linear','x00','x10'), ('temporal','x00','x01')]
    elif batch['order'] == 'LT':
        pairs = [('linear','x00','x10'), ('temporal','x10','x11')]
    else:
        pairs = [('temporal','x00','x01'), ('linear','x01','x11')]
    return [dict(kind=kind, x=batch[a], endpoint_x=batch[b], h=hcache[a],
                 target_delta=hcache[b] - hcache[a], p=batch[kind + '_parameters'],
                 cached_reference=hcache[a][:,320:], cached_endpoint_reference=hcache[b][:,320:],
                 conditional=a != 'x00', source=a, destination=b, **common)
            for kind, a, b in pairs]


@torch.no_grad()
def chain_metrics(models, batch, hcache, reference_fn):
    edges = conditional_edges(batch, hcache)
    current = hcache['x00']
    results = {}
    for edge in edges:
        kwargs = dict(phase_noise=edge['phase_noise'], cached_reference=edge['cached_reference'],
                      cached_endpoint_reference=edge['cached_endpoint_reference'])
        actual_delta = models[edge['kind']](edge['h'], edge['x'], edge['p'], reference_fn, **kwargs)
        results[edge['kind'] + '_conditional_mse'] = float((actual_delta - edge['target_delta']).square().mean())
        current = current + models[edge['kind']](current, edge['x'], edge['p'], reference_fn, **kwargs)
    results['predicted_chain_mse'] = float((current - hcache['x11']).square().mean())
    results['order'] = batch['order']
    return results


@torch.no_grad()
def proposal_select(model, identity, h, x, y, generator, *, candidates=4,
                    random_fraction=.25, reliability=1., reference_fn=None):
    """Bounded finite proposal; return true IQ, predicted risk, random coverage.

    Caller computes true E/G/C CE on returned IQ and records verify_proposal.
    Selection and parameters are detached; false proposals never remove real CE.
    """
    if candidates < 2 or not 0 <= random_fraction <= 1:
        raise ValueError('Invalid registered proposal budget')
    if reference_fn is None:
        reference_fn = _encoder(identity).response
    risks, xs, ps, noises = [], [], [], []
    reference = reference_fn(x)
    state = signal_state(x)
    for _ in range(candidates):
        p = sample_parameters(model.kind, len(x), generator, x)
        noise = phase_process(len(x), generator, x) if model.kind == 'temporal' else None
        endpoint = apply_action(x, p, model.kind, noise)
        delta = model(h, x, p, reference_fn, phase_noise=noise, cached_reference=reference,
                      cached_endpoint_reference=reference_fn(endpoint), cached_state=state)
        z = identity_from_intermediate(identity, h + delta)
        risks.append(F.cross_entropy(_encoder(identity).classify_features(z), y, reduction='none'))
        xs.append(endpoint); ps.append(p); noises.append(noise)
    risk = torch.stack(risks)
    hardest = risk.argmax(0)
    random_index = torch.randint(candidates, (len(x),), generator=generator).to(x.device)
    usage = torch.as_tensor(reliability,device=x.device,dtype=x.dtype).clamp(0,1) * (1-random_fraction)
    if usage.ndim and usage.shape != (len(x),):
        raise ValueError('Reliability must be scalar or per-recipient [B]')
    learned = (torch.rand(len(x), generator=generator).to(x.device) < usage)
    selected = torch.where(learned, hardest, random_index)
    index = torch.arange(len(x), device=x.device)
    return dict(x=torch.stack(xs)[selected,index].detach(),
                parameters=torch.stack(ps)[selected,index].detach(),
                phase_noise=torch.stack(noises)[selected,index].detach() if noises[0] is not None else None,
                predicted_ce=risk[selected,index].detach(), learned_mask=learned,
                candidate_index=selected, candidate_count=candidates,
                requested_random_fraction=random_fraction, learned_usage_probability=float(usage.mean()),
                exact29_evaluations=candidates+1)


def verify_proposal(proposal, real_logits, y):
    actual = F.cross_entropy(real_logits.detach(), y, reduction='none')
    predicted = proposal['predicted_ce']
    relative_error = (predicted - actual).abs().mean() / (actual.abs().mean() + .1)
    per_sample_error = (predicted-actual).abs()/(actual.abs()+.1)
    return dict(real_ce=float(actual.mean()), predicted_ce=float(predicted.mean()),
                risk_mae=float((predicted-actual).abs().mean()),
                relative_risk_error=float(relative_error),
                reliability_next=float(torch.exp(-relative_error)),
                learned_used=int(proposal['learned_mask'].sum()),
                random_used=int((~proposal['learned_mask']).sum()), samples=len(y),
                per_sample_relative_error=per_sample_error.cpu().tolist(),
                per_sample_reliability_next=torch.exp(-per_sample_error).cpu().tolist())


def estimate_pair(x0, x1, kind, phase_noise=None):
    """Known-pair analytic coefficients; no class/RX/day metadata."""
    if kind == 'linear':
        from experiments.cvs_multi_state_action.actions import estimate_pair as old_estimate
        return old_estimate(x0, x1, kind)
    z0 = torch.complex(x0[:,0].double(), x0[:,1].double())
    z1 = torch.complex(x1[:,0].double(), x1[:,1].double())
    phase = torch.angle(z1 * z0.conj())
    # Registered phase ranges avoid wrapping for ordinary well-supported pairs.
    t = (torch.arange(256,device=x0.device,dtype=torch.float64)-127.5)/25_000_000.
    noise = torch.zeros_like(phase) if phase_noise is None else phase_noise.double()
    # Normalize columns for conditioning, then return declared physical units.
    basis = torch.stack((torch.ones_like(noise), (2*math.pi*CFO_LIMIT_HZ*t).expand_as(noise),
                         PHASE_NOISE_LIMIT_RAD*noise,
                         (math.pi*CURVATURE_PRESSURE_HZ_S*t.square()).expand_as(noise)),2)
    weight = z0.abs().clamp_min(1e-10)
    design = basis * weight[:,:,None]
    solution = torch.linalg.pinv(design) @ (phase*weight)[:,:,None]
    result = solution[:,:,0] * x0.new_tensor([1.,CFO_LIMIT_HZ,PHASE_NOISE_LIMIT_RAD,CURVATURE_PRESSURE_HZ_S]).double()
    return result.to(x0), dict(rank=torch.linalg.matrix_rank(design),
        residual=(basis @ solution-phase[:,:,None]).square().mean((1,2)),
        interpretation='Known process realization LS; phase wrapping/noisy pair can invalidate estimate')


class NeuralPairEstimator(nn.Module):
    """Known-pair label-free parameter estimator; normalized output supervision."""
    def __init__(self, kind, hidden=64):
        super().__init__(); self.kind=kind
        self.net=nn.Sequential(nn.Linear(1024,hidden),nn.SiLU(),nn.Linear(hidden,4))
        scales = [1.]*4 if kind=='linear' else [.25,CFO_LIMIT_HZ,PHASE_NOISE_LIMIT_RAD,CURVATURE_PRESSURE_HZ_S]
        self.register_buffer('scales',torch.tensor(scales))
    def forward(self,x0,x1):
        base=x0.flatten(1); delta=(x1-x0).flatten(1)
        return (self.net(torch.cat((base,delta),1))-self.net(torch.cat((base,torch.zeros_like(delta)),1))) * self.scales
