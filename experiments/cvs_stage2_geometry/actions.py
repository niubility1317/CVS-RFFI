"""Fixed coverage versus conditional directions at exactly matched radii.

The normalized L-infinity radius is shared sample by sample for equal private
seeds. Actual IQ changes depend on direction and waveform and are measured,
never claimed to be equal. All old physical bounds/operators are retained.
"""
import math
import time

import torch
from torch.nn import functional as F

from experiments.cvs_stage2_decouple import actions as old
from experiments.cvs_stage2_decouple.source_counterfactual_audit import ReadOnlyState
from experiments.cvs_multi_action_risk.physics import phase_process
from .objectives import M_REQ, margin_risk


KINDS = ('linear', 'temporal')
POLICIES = ('coverage', 'conditional')
FORMAT_VERSION = 2
RHO = .5
ACTION_ORDER = ('linear', 'linear', 'temporal', 'temporal')


def action_contract():
    physical = old.action_contract()
    return dict(format_version=FORMAT_VERSION, physical=physical,
                direction_dimensions=4, state_dimensions=40, code_dimensions=8,
                radius='Uniform(0,1), normalized physical L-infinity radius',
                coverage_direction='four independent Uniform(-1,1) cube coordinates, normalize by max absolute coordinate',
                temporal_noise_direction='absolute value of coordinate2; nonnegative phase-noise amplitude',
                conditional_direction='same old48-SiLU-48 network output converted to normalized physical coordinates, then L-infinity normalized',
                conditional_fraction=RHO, B1='coverage only', B2_B3='0.5 coverage + 0.5 conditional direction',
                common_draw_order=['radius', 'cube', 'mixture_mask', 'network_code', 'temporal_phase_process'],
                strength_claim='exact shared normalized radius for equal private RNG; IQ effects are not assumed equal',
                fit=dict(initialization='fresh private-seed network; no inherited old-policy weights',
                         passes=2, batch_size=128, learning_rate=2e-4, weight_decay=1e-4,
                         objective='maximize frozen-parent raw margin drop using the same mixed radial transform as application'),
                identity_training='frozen actions, no auxiliary optimizer or refresh',
                schedule=dict(every_identity_steps=4, samples=32, order=list(ACTION_ORDER)),
                boundary_diagnostics='normalized coordinate/radius concentration and IQ edge-change energy',
                coverage_interpretation='uniform cube-derived direction plus uniform radius, not uniform volume in the physical cube')


def _generator(value):
    if isinstance(value, int):
        return torch.Generator(device='cpu').manual_seed(value)
    if not isinstance(value, torch.Generator) or str(value.device) != 'cpu':
        raise ValueError('An explicit private CPU generator or integer seed is required')
    return value


def _unit_direction(value, kind):
    if value.ndim != 2 or value.shape[1] != 4 or kind not in KINDS:
        raise ValueError('Four normalized physical coordinates are required')
    if kind == 'temporal':
        value = torch.stack((value[:, 0], value[:, 1], value[:, 2].abs(), value[:, 3]), 1)
    length = value.abs().amax(1, keepdim=True)
    normalized = value / length.clamp_min(1e-12)
    fallback = torch.zeros_like(value)
    fallback[:, 0] = 1.
    return torch.where(length > 1e-12, normalized, fallback)


def _network_coordinates(net, state, code, kind):
    bounded = net(state, code)
    if kind == 'linear':
        return bounded
    return torch.stack((bounded[:, 0] / old.PHASE_LIMIT_RAD,
                        bounded[:, 1] / old.CFO_LIMIT_HZ,
                        bounded[:, 2] / old.PHASE_NOISE_LIMIT_RAD,
                        bounded[:, 4] / old.DELAY_LIMIT_SAMPLES), 1)


def _physical(normalized, kind):
    if kind == 'linear':
        # Legacy apply_linear multiplies each tap component by0.06.
        return normalized
    return torch.stack((normalized[:, 0] * old.PHASE_LIMIT_RAD,
                        normalized[:, 1] * old.CFO_LIMIT_HZ,
                        normalized[:, 2] * old.PHASE_NOISE_LIMIT_RAD,
                        torch.zeros_like(normalized[:, 0]),
                        normalized[:, 3] * old.DELAY_LIMIT_SAMPLES), 1)


def _proposal(net, x, kind, generator, policy, state=None):
    if kind not in KINDS or policy not in POLICIES:
        raise ValueError('Unregistered action kind or policy')
    g = _generator(generator)
    n = len(x)
    radius = torch.rand(n, 1, generator=g).to(x)
    cube = (2 * torch.rand(n, 4, generator=g) - 1).to(x)
    selected = (torch.rand(n, generator=g) < RHO).to(x.device)
    code = (2 * torch.rand(n, old.CODE_DIM, generator=g) - 1).to(x)
    noise = phase_process(n, g, x) if kind == 'temporal' else None
    coverage = _unit_direction(cube, kind)
    if policy == 'conditional':
        if net is None:
            raise ValueError('Conditional policy requires its fitted frozen network')
        state = old.signal_state(x) if state is None else state
        conditional = _unit_direction(_network_coordinates(net, state, code, kind), kind)
        direction = torch.where(selected[:, None], conditional, coverage)
    else:
        direction = coverage
        selected = torch.zeros_like(selected)
    normalized = radius * direction
    return dict(kind=kind, policy=policy, parameters=_physical(normalized, kind),
                normalized_parameters=normalized, direction=direction, radius=radius.squeeze(1),
                conditional_mask=selected, noise=noise)


def apply_proposal(x, proposal):
    """Pure differentiable operator; parameters/noise never regenerated from x."""
    return old.apply_physical_action(x, proposal['parameters'], proposal['kind'], proposal['noise'])


def _observations(x, augmented, proposal):
    normal = proposal['normalized_parameters'].detach()
    difference = (augmented.detach() - x.detach()).square().sum(1)
    energy = x.detach().square().sum((1, 2)).clamp_min(1e-12)
    total = difference.sum(1)
    edge = difference[:, :16].sum(1) + difference[:, -16:].sum(1)
    return dict(radius=proposal['radius'].detach(), normalized_linf=normal.abs().amax(1),
                normalized_l2=normal.square().sum(1).sqrt(),
                iq_relative_change=(total / energy).sqrt(),
                iq_rms_ratio=(augmented.detach().square().sum((1, 2)) / energy).sqrt(),
                edge_change_energy_fraction=torch.where(total > 1e-20, edge / total.clamp_min(1e-20), torch.zeros_like(total)),
                conditional=proposal['conditional_mask'].float())


def _stats(values):
    values = values.detach().cpu().float().reshape(-1)
    if not len(values) or not bool(torch.isfinite(values).all()):
        raise ValueError('Empty/nonfinite source action diagnostic')
    return dict(count=len(values), mean=float(values.mean()), rms=float(values.square().mean().sqrt()),
                q05=float(torch.quantile(values, .05)), q50=float(torch.quantile(values, .5)),
                q95=float(torch.quantile(values, .95)), minimum=float(values.min()), maximum=float(values.max()))


def proposal_statistics(x, augmented, proposal):
    stats = old._parameter_summary(proposal['parameters'], proposal['kind'])
    normal = proposal['normalized_parameters'].detach()
    observed = _observations(x, augmented, proposal)
    stats.update(policy=proposal['policy'], conditional_count=int(proposal['conditional_mask'].sum()),
                 coverage_count=len(x) - int(proposal['conditional_mask'].sum()),
                 radius_linf_max_error=float((observed['radius'] - observed['normalized_linf']).abs().max()),
                 normalized_parameter_abs_mean=normal.abs().mean(0).cpu().tolist(),
                 normalized_parameter_boundary_fraction=(normal.abs() >= .95).float().mean(0).cpu().tolist(),
                 radius_boundary_fraction=float((proposal['radius'] >= .95).float().mean()),
                 measurements={key: _stats(value) for key, value in observed.items()})
    return stats


def _source_diagnostics(parent, cache, nets, seed, device, batch_size):
    """Complete L_s risk/coverage/boundary evidence; never validation or target."""
    results = {}
    labels = cache['y'].cpu()
    for index, kind in enumerate(KINDS):
        for policy in POLICIES:
            # Policies restart the same private stream: paired radii/noise are exact.
            generator = _generator(seed + 81313 + index * 101)
            values, parameters, physical_parameters = {}, [], []
            for start in range(0, len(labels), batch_size):
                x = cache['x'][start:start + batch_size].to(device)
                y = labels[start:start + batch_size].to(device)
                p = _proposal(nets[kind], x, kind, generator, policy,
                              cache['state'][start:start + batch_size].to(device))
                augmented = apply_proposal(x, p)
                z = parent(augmented)
                u = old.logit_margin(z, y)
                u0 = cache['parent_margin'][start:start + batch_size].to(device)
                risk = margin_risk(u0, u)
                observations = dict(_observations(x, augmented, p), margin=u, margin_drop=u0 - u,
                                    correct=(z.argmax(1) == y).float(), diagnostic_ce=F.cross_entropy(z, y, reduction='none'),
                                    finite_risk=risk, finite_risk_active=(risk > 0).float())
                for name, value in observations.items():
                    values.setdefault(name, []).append(value.detach().cpu())
                parameters.append(p['normalized_parameters'].detach().cpu())
                actual = p['parameters'].detach().cpu()
                physical_parameters.append(.06 * actual if kind == 'linear' else actual)
            values = {name: torch.cat(parts) for name, parts in values.items()}
            normal, physical = torch.cat(parameters), torch.cat(physical_parameters)
            strata = {}
            for key, tags in [('tx', labels)] + ([('rx', cache['rx'].cpu())] if 'rx' in cache else []):
                strata[key] = {str(tag): {name: _stats(value[tags == tag]) for name, value in values.items()}
                               for tag in sorted(set(tags.tolist()))}
            results[kind + '/' + policy] = dict(source_count=len(labels), target_access=False,
                diagnostic_seed=seed + 81313 + index * 101,
                scalar_statistics={name: _stats(value) for name, value in values.items()},
                normalized_parameter_statistics=[_stats(normal[:, j]) for j in range(4)],
                physical_parameter_statistics=[_stats(physical[:, j]) for j in range(physical.shape[1])],
                normalized_boundary_fraction=(normal.abs() >= .95).float().mean(0).tolist(),
                radius_boundary_fraction=float((values['radius'] >= .95).float().mean()),
                radius_linf_max_error=float((values['radius'] - values['normalized_linf']).abs().max()),
                normalized_radius_histogram=torch.histc(values['radius'], bins=10, min=0., max=1.).long().tolist(),
                direction_sign_coverage=[sorted(set(torch.sign(normal[:, j]).int().tolist())) for j in range(4)],
                strata=strata, diagnostic_ce_is_not_training_loss=True)
    return results


def fit_actions(parent, cache, seed, device, passes=2, batch_size=128):
    """Fresh fit on two complete source-L passes using the final radial policy."""
    if passes != 2 or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError('Exactly two complete passes and positive batch size required')
    x, labels, states = old._validate_cache(cache)
    if cache.get('role') != 'L_s' or 'parent_margin' not in cache:
        raise ValueError('Visible source L and fixed deterministic parent margins required')
    parent_margin = cache['parent_margin'].detach().cpu()
    if parent_margin.shape != (len(x),) or not bool(torch.isfinite(parent_margin).all()):
        raise ValueError('Missing/nonfinite fixed parent margins')
    started = time.perf_counter()
    branches, networks, nets = {}, {}, {}
    preservation = ReadOnlyState(parent)
    with preservation:
        parent.eval()
        for p in parent.parameters():
            p.requires_grad_(False)
        for kind_index, kind in enumerate(KINDS):
            generator = _generator(int(seed) + 17041 * (kind_index + 1))
            net = old._new_net(kind, generator, device)
            optimizer = torch.optim.AdamW(net.parameters(), lr=2e-4, weight_decay=1e-4)
            passes_log = []
            for iteration in range(passes):
                order = torch.randperm(len(x), generator=generator)
                drop_sum, gradient_sum, conditional_count, calls = 0., 0., 0, 0
                for start in range(0, len(order), batch_size):
                    ids = order[start:start + batch_size]
                    batch = x[ids].to(device)
                    y = labels[ids].to(device)
                    optimizer.zero_grad(set_to_none=True)
                    proposal = _proposal(net, batch, kind, generator, 'conditional', states[ids].to(device))
                    augmented = apply_proposal(batch, proposal)
                    u = old.logit_margin(parent(augmented), y)
                    drop = parent_margin[ids].to(device) - u
                    loss = -drop.mean()
                    if not bool(torch.isfinite(loss)):
                        raise FloatingPointError('Nonfinite radial action search loss')
                    loss.backward()
                    gradients = [p.grad for p in net.parameters() if p.grad is not None]
                    norm = sum(g.detach().square().sum() for g in gradients).sqrt()
                    if not bool(torch.isfinite(norm)):
                        raise FloatingPointError('Nonfinite radial action network gradient')
                    optimizer.step()
                    drop_sum += float(drop.detach().sum())
                    gradient_sum += float(norm)
                    conditional_count += int(proposal['conditional_mask'].sum())
                    calls += 1
                passes_log.append(dict(pass_index=iteration + 1, samples=len(order), unique_physical_ids=len(order),
                                       optimizer_steps=calls, conditional_samples=conditional_count,
                                       mean_parent_margin_drop=drop_sum / len(order), mean_action_gradient_norm=gradient_sum / calls))
            net.eval().requires_grad_(False)
            nets[kind] = net
            networks[kind] = {name: value.detach().cpu().clone() for name, value in net.state_dict().items()}
            branches[kind] = dict(passes=passes_log, optimizer_steps=passes * math.ceil(len(x) / batch_size),
                                  sample_occurrences=passes * len(x), parameters=sum(p.numel() for p in net.parameters()))
        with torch.no_grad():
            diagnostics = _source_diagnostics(parent, cache, nets, int(seed), device, batch_size)
    if not all(preservation.report.values()):
        raise RuntimeError('Frozen parent changed during radial action fitting')
    return dict(format_version=FORMAT_VERSION, contract=action_contract(), seed=int(seed), networks=networks,
                source_ids=list(cache['ids']), fit_settings=dict(passes=2, batch_size=batch_size, lr=2e-4, weight_decay=1e-4),
                diagnostics=dict(source_count=len(x), source_role='L_s', target_access=False, branches=branches,
                                 full_source_policy_diagnostics=diagnostics, parent_state_invariance=preservation.report,
                                 preparation_iq_augmented_forwards=4 * len(x) + 4 * len(x),
                                 fit_iq_augmented_forwards=4 * len(x), diagnostic_iq_augmented_forwards=4 * len(x),
                                 elapsed_seconds=time.perf_counter() - started))


class FrozenGeometryActions:
    def __init__(self, payload, device):
        if payload.get('format_version') != FORMAT_VERSION or payload.get('contract') != action_contract():
            raise ValueError('Changed action policy contract or obsolete old-policy artifact')
        self.device = torch.empty(0, device=device).device
        self.networks = torch.nn.ModuleDict()
        for index, kind in enumerate(KINDS):
            net = old._new_net(kind, _generator(payload['seed'] + index), device)
            net.load_state_dict(payload['networks'][kind], strict=True)
            net.eval().requires_grad_(False)
            self.networks[kind] = net
        self.networks.eval()

    @torch.no_grad()
    def propose(self, x, kind, generator, policy='conditional'):
        if x.device != self.device or tuple(x.shape[1:]) != (2, 256) or not len(x):
            raise ValueError('Expected nonempty source IQ[B,2,256] on the action device')
        proposal = _proposal(self.networks[kind], x, kind, generator, policy)
        stats = old._parameter_summary(proposal['parameters'], kind)
        if not stats['finite'] or not stats['within_registered_bounds']:
            raise ValueError('Radial action escaped original physical bounds')
        return proposal

    @torch.no_grad()
    def transform(self, x, kind, generator, policy='conditional'):
        proposal = self.propose(x, kind, generator, policy)
        augmented = apply_proposal(x, proposal)
        return augmented.detach(), proposal_statistics(x, augmented, proposal)


def load_actions(payload, device):
    return FrozenGeometryActions(payload, device)
