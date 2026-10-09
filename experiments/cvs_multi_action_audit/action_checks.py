"""Held-out action diagnostics. No identity fitting, target access or selection."""
import torch

BLOCKS = (('base', 0, 160), ('pa', 160, 320), ('reference', 320, 349))


def block_scales(fit_delta, epsilon=1e-12):
    """All scales are fitted on the auxiliary-fit role only."""
    return torch.stack([fit_delta[:, a:b].square().mean().clamp_min(epsilon)
                        for _, a, b in BLOCKS]).detach()


def block_loss(prediction, target, scales):
    return torch.stack([(prediction[:, a:b] - target[:, a:b]).square().mean() / scale
                        for (_, a, b), scale in zip(BLOCKS, scales)]).mean()


def margin(logits, labels):
    other = logits.clone()
    other.scatter_(1, labels[:, None], -torch.inf)
    return logits.gather(1, labels[:, None]).squeeze(1) - other.max(1).values


def spectrum(delta, energy_epsilon=1e-12):
    """Uncentered action energy spectrum; centered spectrum also recorded."""
    result = {}
    for name, value in (('uncentered', delta), ('centered', delta - delta.mean(0))):
        singular = torch.linalg.svdvals(value.detach().double().cpu())
        energy = singular.square()
        total = float(energy.sum())
        if total <= energy_epsilon:
            result[name] = dict(status='N/A_ZERO_ENERGY', singular_values=singular.tolist(),
                                rank90=None, rank95=None, energy_at_rank16=None)
        else:
            cumulative = energy.cumsum(0) / total
            result[name] = dict(status='MEASURED', singular_values=singular.tolist(),
                rank90=int(torch.searchsorted(cumulative, .9)) + 1,
                rank95=int(torch.searchsorted(cumulative, .95)) + 1,
                energy_at_rank16=float(energy[:16].sum() / total))
    return result


def _error(prediction, target, baseline_mean, energy_epsilon):
    energy = float(target.square().mean())
    mse = float((prediction - target).square().mean())
    mean_mse = float((baseline_mean - target).square().mean())
    sufficient = energy > energy_epsilon
    # A zero/mean predictor cannot pass. No denominator floor creates a score.
    beats = sufficient and mse < min(energy, mean_mse)
    return dict(target_energy=energy, target_rms=energy ** .5, mse=mse,
        zero_mse=energy, training_mean_mse=mean_mse,
        skill_vs_zero=1 - mse / energy if sufficient else None,
        skill_vs_training_mean=1 - mse / mean_mse if mean_mse > energy_epsilon else None,
        reliability=max(0., 1 - mse / min(energy, mean_mse)) if beats else 0.,
        status='MEASURED' if sufficient else 'N/A_ZERO_ENERGY',
        beats_zero_and_training_mean=bool(beats))


@torch.no_grad()
def action_metrics(h, delta, prediction, z0, z1, logits0, logits1, labels,
                   train_mean, g_fn, classifier_fn, energy_epsilon=1e-12):
    """Score one fixed prediction; G is frozen and never fitted on this role."""
    z_pred = g_fn(h + prediction)
    logits_pred = classifier_fn(z_pred)
    z_mean = g_fn(h + train_mean)
    dh = _error(prediction, delta, train_mean, energy_epsilon)
    dz = _error(z_pred - z0, z1 - z0, z_mean - z0, energy_epsilon)
    m0, m1 = margin(logits0, labels), margin(logits1, labels)
    mp = margin(logits_pred, labels)
    blocks = {name: _error(prediction[:, a:b], delta[:, a:b], train_mean[:, a:b], energy_epsilon)
              for name, a, b in BLOCKS}
    return dict(count=len(h), h=dh, blocks=blocks, z=dz,
        reliability=min(dh['reliability'], dz['reliability']),
        reliability_rule='heldout h and z must each beat zero and fit-role mean',
        margin=dict(actual_delta_mean=float((m1 - m0).mean()),
                    actual_delta_abs_mean=float((m1 - m0).abs().mean()),
                    predicted_delta_mean=float((mp - m0).mean()),
                    predicted_delta_abs_mean=float((mp - m0).abs().mean()),
                    delta_mae=float((mp - m1).abs().mean()),
                    delta_rmse=float((mp - m1).square().mean().sqrt())),
        clean_accuracy=float((logits0.argmax(1) == labels).float().mean()),
        actual_accuracy=float((logits1.argmax(1) == labels).float().mean()),
        predicted_accuracy=float((logits_pred.argmax(1) == labels).float().mean()),
        actual_action_norm_mean=float(delta.norm(dim=1).mean()),
        predicted_action_norm_mean=float(prediction.norm(dim=1).mean()),
        classification_flip_rate=float((logits1.argmax(1) != logits0.argmax(1)).float().mean()))


def temporal_observability():
    result = {}
    full = torch.linspace(-1, 1, 256, dtype=torch.float64)
    for name, u in (('full_256', full), ('observed_80_160', full[80:160])):
        basis = torch.stack((.08 * torch.ones_like(u), .10 * u, .06 * u.square()), 1)
        sv = torch.linalg.svdvals(basis)
        fit = torch.linalg.lstsq(basis[:, :2], basis[:, 2]).solution
        residual = basis[:, 2] - basis[:, :2] @ fit
        result[name] = dict(condition_number=float(sv.max() / sv.min()),
            singular_values=sv.tolist(), quadratic_independent_rms=float(residual.square().mean().sqrt()))
    return result


@torch.no_grad()
def physical_checks(x, changed, parameters, kind, response):
    before = response.components(x)
    after = response.components(changed)
    route = before['selected_delay_samples'] != after['selected_delay_samples']
    delta_power = (changed - x).square().sum(1)
    total = delta_power.sum(1)
    # Only first three IQ coordinates are directly affected by causal zero padding.
    edge = delta_power[:, :3].sum(1)
    valid = total > 1e-12
    result = dict(route_change_count=int(route.sum()), count=len(x),
        route_change_rate=float(route.float().mean()),
        mean_delay_jump_samples=float((after['selected_delay_samples'] - before['selected_delay_samples']).abs().mean()),
        intervention_iq_energy=float(delta_power.mean()),
        first_three_samples_energy_fraction=float((edge[valid] / total[valid]).mean()) if bool(valid.any()) else None)
    if kind == 'linear':
        from experiments.cvs_multi_disentangle.physics import LINEAR_TAP_SCALE
        z = torch.complex(x[:, 0], x[:, 1])
        circular = z.clone()
        for delay, offset in ((1, 0), (3, 2)):
            tap = LINEAR_TAP_SCALE * torch.complex(parameters[:, offset], parameters[:, offset + 1])
            circular += tap[:, None] * z.roll(delay, 1)
        zero = torch.complex(changed[:, 0], changed[:, 1])
        result['zero_vs_circular_boundary_mse'] = float((zero - circular).abs().square().mean())
        result['boundary_reference_caveat'] = 'Circular extension is diagnostic only, not true preceding IQ.'
    else:
        result['temporal_observability'] = temporal_observability()
    return result, route


def focused_checks():
    """Small synthetic correctness checks, not evidence of real-data efficacy."""
    import json
    from torch import nn
    from .actions import fit_and_audit, composition_audit

    class Teacher(nn.Module):
        def __init__(self):
            super().__init__()
            self.e = nn.Linear(512, 349)
            self.g = nn.Sequential(nn.LayerNorm(349), nn.Linear(349, 160))
            self.c = nn.Linear(160, 3)

        def features(self, x):
            return self.e(x.flatten(1))

    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        torch.manual_seed(11)
        teacher = Teacher().train()
        original = {key: value.clone() for key, value in teacher.state_dict().items()}
        fit_x, audit_x = torch.randn(12, 2, 256), torch.randn(6, 2, 256)
        fit_y, audit_y = torch.arange(12) % 3, torch.arange(6) % 3
        kwargs = dict(fit_ids=['fit' + str(i) for i in range(12)],
            audit_ids=['audit' + str(i) for i in range(6)], steps=3, batch_size=6,
            feature_fn=teacher.features, g_fn=teacher.g, classifier_fn=teacher.c)
        rng_before = torch.random.get_rng_state().clone()
        result = fit_and_audit(teacher, fit_x, fit_y, audit_x, audit_y,
                              torch.Generator().manual_seed(17), **kwargs)
        assert torch.equal(rng_before, torch.random.get_rng_state())
        assert teacher.training and all(p.requires_grad for p in teacher.parameters())
        assert all(torch.equal(original[key], value) for key, value in teacher.state_dict().items())
        assert all(p.grad is None for p in teacher.parameters())
        assert len(result['step_logs']) == 2 * 4 * 3
        assert all(row['gradient_norm'] > 0 for row in result['step_logs'])
        for kind in result['report']['kinds'].values():
            assert kind['same_intervention_donor_verified']
            for mode in ('legacy_self', 'block_z_self', 'block_z_cross'):
                assert kind['modes'][mode]['zero']['reliability'] == 0
                assert kind['modes'][mode]['training_mean']['reliability'] == 0
        json.dumps(result['report'], allow_nan=False)
        comp = composition_audit(teacher, audit_x, audit_y, torch.Generator().manual_seed(19),
            result['state_dicts'], batch_size=6, feature_fn=teacher.features,
            g_fn=teacher.g, classifier_fn=teacher.c)
        assert torch.equal(rng_before, torch.random.get_rng_state())
        assert len(comp['modes']) == 4
        json.dumps(comp, allow_nan=False)
        try:
            fit_and_audit(teacher, fit_x, fit_y, audit_x, audit_y,
                torch.Generator().manual_seed(17), **dict(kwargs, audit_ids=['fit0'] + kwargs['audit_ids'][1:]))
        except ValueError as error:
            assert 'overlap' in str(error)
        else:
            raise AssertionError('Physical overlap must be rejected')
        zero = torch.zeros(4, 349)
        metric = _error(zero, zero, zero.mean(0, keepdim=True), 1e-12)
        assert metric['reliability'] == 0 and metric['skill_vs_zero'] is None
        # Frozen G must transmit a nonzero gradient into a predicted action.
        for p in teacher.g.parameters():
            p.requires_grad_(False)
        predicted = torch.randn(4, 349, requires_grad=True)
        teacher.g(predicted).square().mean().backward()
        assert predicted.grad.norm() > 0 and all(p.grad is None for p in teacher.g.parameters())
        assert 40 < temporal_observability()['observed_80_160']['condition_number'] < 50
        print(json.dumps(dict(status='VERIFIED', checks=['teacher_unchanged', 'gradient_through_frozen_G',
            'physical_role_overlap_rejected', 'same_p_cross_TX', 'zero_mean_reliability_zero',
            'zero_energy_NA', 'all_fit_modes', 'composition_modes', 'private_rng', 'finite_JSON', 'condition_number']), indent=2))
    finally:
        torch.set_num_threads(previous_threads)


if __name__ == '__main__':
    focused_checks()
