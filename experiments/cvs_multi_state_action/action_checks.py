"""Focused scientific and actual-CVS checks for explicit state action models."""
import argparse
import json
import torch


def circular_delay_distance(a, b, period=20.):
    distance = (a - b).abs().remainder(period)
    return torch.minimum(distance, period - distance)


@torch.no_grad()
def physical_diagnostics(identity, data, kind, cache, batch_size=32):
    from .actions import estimate_pair, apply_action, _encoder
    results = {}
    n = len(data['x']); p = cache['p'][n:2 * n]
    x, changed = data['x'], apply_action(data['x'], p, kind)
    for lo, hi in ((0, 256), (80, 160)):
        errors, conditions, ranks = [], [], []
        for start in range(0, n, batch_size):
            sl = slice(start, start + batch_size)
            estimate, info = estimate_pair(x[sl], changed[sl], kind, (lo, hi))
            errors.append((estimate - p[sl]).square().mean(1))
            conditions.append(info['condition']); ranks.append(info['full_rank'])
        e, c, r = torch.cat(errors), torch.cat(conditions), torch.cat(ranks)
        results[f'analytic_{lo}_{hi}'] = dict(parameter_mse=float(e.mean()), parameter_max_rms=float(e.max().sqrt()),
            full_rank_rate=float(r.float().mean()), condition_mean=float(c.mean()), condition_max=float(c.max()),
            interpretation='Deterministic digital pair; shared packet noise is not independent measurement noise.')
    count = (len(cache['p']) // n - 1) // 2
    positive = cache['delta'][n:(count + 1) * n]
    negative = cache['delta'][(count + 1) * n:]
    results['even_odd'] = dict(even_energy=float(((positive + negative) / 2).square().mean()),
                               odd_energy=float(((positive - negative) / 2).square().mean()),
                               zero_action_energy=float(cache['delta'][:n].square().mean()))
    response = _encoder(identity).response
    distances, changes = [], []
    for start in range(0, n, batch_size):
        a = response.components(x[start:start + batch_size])['selected_delay_samples']
        b = response.components(changed[start:start + batch_size])['selected_delay_samples']
        distances.append(circular_delay_distance(a, b)); changes.append(a != b)
    results['reference_routing'] = dict(circular_delay_mean=float(torch.cat(distances).mean()),
        circular_delay_max=float(torch.cat(distances).max()), changed_fraction=float(torch.cat(changes).float().mean()),
        period=20, interpretation='Periodic fixed-template routing, not physical propagation delay.')
    return results


def checks(device='cpu', actual_cvs=True):
    from .actions import (StateAction, NeuralPairEstimator, estimate_pair, apply_action,
                          feature_scales, donor_indices, validate_roles, fit_and_audit,
                          intermediate, _encoder, composition_audit, fixed_identity, action_loss)
    torch.set_num_threads(2)
    generator = torch.Generator().manual_seed(191)
    x = torch.randn(24, 2, 256, generator=generator).to(device)
    h = torch.randn(24, 349, generator=generator).to(device)
    results = {}
    for kind, width in (('linear', 4), ('temporal', 3)):
        p = (torch.rand(24, width, generator=generator) * 2 - 1).to(device)
        estimate, info = estimate_pair(x, apply_action(x, p, kind), kind)
        assert float((estimate - p).abs().max()) < 2e-5 and bool(info['full_rank'].all())
        estimate80, _ = estimate_pair(x, apply_action(x, p, kind), kind, (80, 160))
        assert float((estimate80 - p).abs().max()) < 2e-4
        zero, zi = estimate_pair(torch.zeros_like(x), torch.zeros_like(x), kind)
        assert bool(torch.isfinite(zero).all()) and not bool(zi['full_rank'].any())
        model = StateAction(kind, exact_reference=False).to(device)
        assert torch.count_nonzero(model(h, x, torch.zeros_like(p))) == 0
        even = (model(h, x, p) + model(h, x, -p)) / 2
        assert float(even.detach().square().mean()) > 0
        first = StateAction(kind, second_order=False, exact_reference=False).to(device)
        assert torch.allclose(first(h, x, p), -first(h, x, -p))
        neural = NeuralPairEstimator(kind).to(device)
        assert torch.count_nonzero(neural(x, x)) == 0
        results[kind] = dict(analytic_full_max_error=float((estimate - p).abs().max()),
                            analytic80_max_error=float((estimate80 - p).abs().max()),
                            rank_deficient_detected=True, zero_identity=True, even_representable=True)
    assert torch.allclose(circular_delay_distance(torch.tensor([19.9375]), torch.tensor([0.])), torch.tensor([.0625]))
    tiny = torch.zeros(5, 349, device=device); tiny[:, :160] = 1
    assert float(feature_scales(tiny)[2]) >= .04
    # Two conditions of each physical packet, three TX, matched RX/day, other day.
    y = torch.arange(24, device=device) % 3
    data = dict(x=x, y=y, rx=[1] * 24, day=[1] * 12 + [2] * 12,
                ids=[str(i) for i in range(24)], condition=['clean'] * 24)
    donors, masks, _ = donor_indices(data)
    assert bool(masks['cross_tx_matched'].all())
    assert bool((y[donors['cross_tx_matched']] != y).all())
    assert bool((y[donors['same_tx_other_packet']] == y).all())
    fit = {k: v[:12] for k, v in data.items()}; audit = {k: v[12:] for k, v in data.items()}
    validate_roles(fit, audit)
    try: validate_roles(fit, fit)
    except ValueError: pass
    else: raise AssertionError('Physical overlap accepted')
    try: validate_roles(fit, audit, heldout_tx=0)
    except ValueError: pass
    else: raise AssertionError('Auxiliary TX leakage accepted')
    if actual_cvs:
        from experiments.cvs_multi_disentangle.checks import identity
        from experiments.cvs_multi_action_audit.design import parent_config, SEEDS
        identity_model = identity(parent_config(SEEDS[0]), device).id_backbone
        saved = {k: v.detach().clone() for k, v in identity_model.state_dict().items()}
        identity_model.eval()
        with fixed_identity(identity_model):
            hi = intermediate(identity_model, x[:4]).detach()
            p = torch.randn(4, 4, device=device) * .3
            exact = StateAction('linear').to(device)
            delta = exact(hi, x[:4], p, _encoder(identity_model).response)
            target = _encoder(identity_model).response(apply_action(x[:4], p, 'linear')) - hi[:, 320:]
            assert torch.allclose(delta[:, 320:], target, atol=1e-6)
            true_delta = intermediate(identity_model, apply_action(x[:4], p, 'linear')).detach() - hi
            losses = action_loss(identity_model, hi, true_delta, delta, y[:4])
            decision_gradient = torch.autograd.grad(losses['normalized_z'] + losses['margins'], delta, retain_graph=True)[0]
            assert float(decision_gradient.norm()) > 0
        # Fresh toy data tests machinery, never scientific source performance.
        rng = torch.random.get_rng_state().clone()
        result = fit_and_audit(identity_model, fit, audit, torch.Generator().manual_seed(7),
                              steps=2, batch_size=6, interventions=1)
        composition = composition_audit(identity_model, audit['x'][:6], torch.Generator().manual_seed(8),
                                        result['state_dicts'], result['model_configs'])
        assert len(result['step_logs']) == 24 and len(composition['modes']) == 6
        assert torch.equal(rng, torch.random.get_rng_state())
        assert composition['temporal_changed_state_inverse_iq_mse'] < 1e-12
        assert composition['linear_negative_not_inverse_iq_mse'] > 1e-8
        assert all(torch.equal(v, identity_model.state_dict()[k]) for k, v in saved.items())
        assert all(torch.isfinite(torch.tensor(row['loss'])) for row in result['step_logs'])
        # One shared model per branch fits two conditions; auxiliary heldout TX
        # is omitted from fit, while audit retains other TX for valid donors.
        from experiments.cvs_multi_action_audit.runner import channel_view
        def joint(role):
            leo = channel_view(dict(role, rx=torch.tensor(role['rx'], device=device)))
            return {k: torch.cat((v, leo[k]), 0) if torch.is_tensor(v) else list(v) + list(role[k])
                    for k, v in role.items() if k != 'condition'} | {'condition': ['clean'] * len(role['x']) + ['source_practical_mid'] * len(role['x'])}
        fit_joint, audit_joint = joint(fit), joint(audit)
        keep = fit_joint['y'] != 0
        fit_joint = {k: v[keep] if torch.is_tensor(v) else [a for a, flag in zip(v, keep.tolist()) if flag]
                     for k, v in fit_joint.items()}
        joint_result = fit_and_audit(identity_model, fit_joint, audit_joint, torch.Generator().manual_seed(9),
                                    heldout_tx=0, steps=1, batch_size=6, interventions=2)
        assert len(joint_result['step_logs']) == 12
        assert len(joint_result['report']['jointly_fitted_conditions']) == 2
        for kr in joint_result['report']['kinds'].values():
            for mr in kr['modes'].values():
                assert '0' in mr['audit']['cross_tx_matched']['tx_strata']
        results['actual_CVS'] = dict(fit_audit_12_models=True, composition=True, exact_reference=True,
                                   identity_unchanged=True, gradients_through_G_C=True,
                                   joint_clean_source_LEO=True, auxiliary_TX_holdout=True, caller_RNG_preserved=True)
    results.update(status='PASS', device=device, circular_delay=True, feature_floor=True,
                   donors=True, physical_overlap_rejected=True, heldout_TX_leakage_rejected=True)
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--device', default='cpu'); parser.add_argument('--output')
    args = parser.parse_args(); result = checks(args.device)
    if args.output:
        from pathlib import Path
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))
