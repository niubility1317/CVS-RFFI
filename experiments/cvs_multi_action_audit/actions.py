"""Source-only fixed-teacher L/T fit and held-out action audit.

New auxiliary fits are diagnostics, not replay of previously trained branches.
All intervention views of a physical packet retain its fit/audit role. Same-p
cross-TX donors always come from the same role as the receiving packet.
"""
from contextlib import contextmanager
import time
import torch
from experiments.cvs_multi_disentangle.model import (
    DisentanglementBranch, BoundedConditionalOperator, intermediate,
    identity_from_intermediate, _encoder,
)
from experiments.cvs_multi_disentangle.physics import apply_linear, apply_temporal
from .action_checks import block_scales, block_loss, action_metrics, physical_checks, spectrum


@contextmanager
def fixed_identity(identity):
    parameters = [(p, p.requires_grad) for p in identity.parameters()]
    modes = [(m, m.training) for m in identity.modules()]
    identity.eval()
    for p, _ in parameters:
        p.requires_grad_(False)
    try:
        yield
    finally:
        for p, enabled in parameters:
            p.requires_grad_(enabled)
        for module, training in modes:
            module.training = training


def _pair_indices(labels, generator):
    labels = labels.detach().cpu()
    indices = []
    for label in labels:
        allowed = torch.where(labels != label)[0]
        if not len(allowed):
            raise ValueError('Cross-TX audit requires at least two TX in each role')
        indices.append(int(allowed[torch.randint(len(allowed), (1,), generator=generator)]))
    return torch.tensor(indices, device=labels.device)


def _batched(function, x, batch_size):
    return torch.cat([function(x[i:i + batch_size]).detach()
                      for i in range(0, len(x), batch_size)])


@torch.no_grad()
def _cache(x, labels, branch, generator, feature_fn, g_fn, classifier_fn, batch_size, base):
    width = branch.parameter_dim
    params = (2 * torch.rand(len(x), width, generator=generator) - 1).to(x)
    donor = _pair_indices(labels, generator).to(x.device)
    apply = apply_linear if branch.kind == 'linear' else apply_temporal
    changed = apply(x, params)
    donor_changed = apply(x[donor], params)  # EXACT same intervention on another TX.
    h0, z0, logits0 = base
    h1 = _batched(feature_fn, changed, batch_size)
    z1 = _batched(g_fn, h1, batch_size)
    logits1 = _batched(classifier_fn, z1, batch_size)
    view0 = _batched(branch.view, x, batch_size)
    view1 = _batched(branch.view, changed, batch_size)
    donor_view1 = _batched(branch.view, donor_changed, batch_size)
    physical, route = physical_checks(x, changed, params, branch.kind, branch.views.response)
    # A non-zero cyclic shift guarantees another parameter draw, preserving q's marginal distribution.
    shuffle = torch.arange(len(x), device=x.device).roll(1)
    return dict(h=h0, delta=h1 - h0, z0=z0, z1=z1, logits0=logits0, logits1=logits1,
        y=labels, params=params, view0=view0, view1=view1,
        donor_view0=view0[donor], donor_view1=donor_view1,
        donor=donor, shuffle=shuffle, physical=physical, route=route)


def _operator_parameters(params):
    return torch.nn.functional.pad(params, (0, 8 - params.shape[1]))


def _fit(branch, cache, mode, g_fn, generator, steps, batch_size, lr, alpha_z, parameter_weight, on_step=None):
    optimizer = torch.optim.AdamW(branch.parameters(), lr=lr, weight_decay=1e-4)
    scales = block_scales(cache['delta'])
    z_scale = (cache['z1'] - cache['z0']).square().mean().clamp_min(1e-12).detach()
    logs = []
    for step in range(steps):
        tick = time.perf_counter()
        index = torch.randperm(len(cache['h']), generator=generator)[:batch_size].to(cache['h'].device)
        h, target = cache['h'][index], cache['delta'][index]
        if mode == 'oracle_parameters':
            prediction = branch(h, _operator_parameters(cache['params'][index]))
            parameter_loss = prediction.new_zeros(())
        else:
            cross = mode == 'block_z_cross'
            v0 = cache['donor_view0' if cross else 'view0'][index]
            v1 = cache['donor_view1' if cross else 'view1'][index]
            result = branch.forward_views(v0, v1, h)
            prediction = result['delta']
            parameter_loss = (result['parameters'] - cache['params'][index]).square().mean()
        global_h = (prediction - target).square().mean() / target.square().mean().clamp_min(1e-4)
        block_h = block_loss(prediction, target, scales)
        # Frozen G remains differentiable w.r.t. prediction. Do not put this in no_grad.
        z_loss = (g_fn(h + prediction) - cache['z1'][index]).square().mean() / z_scale
        loss = (global_h if mode == 'legacy_self' else block_h + alpha_z * z_loss) + parameter_weight * parameter_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradients = [p.grad for p in branch.parameters() if p.grad is not None]
        if not bool(torch.isfinite(loss)) or any(not bool(torch.isfinite(g).all()) for g in gradients):
            raise FloatingPointError('Nonfinite action fit; no retries or heldout feedback')
        grad_norm = float(torch.stack([g.detach().norm() for g in gradients]).norm())
        optimizer.step()
        logs.append(dict(step=step + 1, mode=mode, loss=float(loss.detach()),
            global_h=float(global_h.detach()), block_h=float(block_h.detach()), z=float(z_loss.detach()),
            parameter=float(parameter_loss.detach()), gradient_norm=grad_norm,
            learning_rate=lr, samples=len(index), seconds=time.perf_counter() - tick))
        if on_step is not None:
            on_step(logs[-1])
    return logs


@torch.no_grad()
def _audit(branch, cache, fit_cache, g_fn, classifier_fn, oracle=False):
    h = cache['h']
    mean = fit_cache['delta'].mean(0, keepdim=True)
    if oracle:
        predictions = {'oracle_known_parameters': branch(h, _operator_parameters(cache['params']))}
    else:
        own = branch.encode_pair(cache['view0'], cache['view1'])
        cross = branch.encode_pair(cache['donor_view0'], cache['donor_view1'])
        predictions = {'own_q': branch.operator(h, own), 'cross_tx_same_p_q': branch.operator(h, cross),
                       'shuffled_parameter_q': branch.operator(h, own[cache['shuffle']]),
                       'zero': torch.zeros_like(h), 'training_mean': mean.expand_as(h)}
    reports = {}
    for name, prediction in predictions.items():
        metrics = action_metrics(h, cache['delta'], prediction, cache['z0'], cache['z1'],
            cache['logits0'], cache['logits1'], cache['y'], mean, g_fn, classifier_fn)
        stratified = {}
        for route_name, mask in (('route_unchanged', ~cache['route']), ('route_changed', cache['route'])):
            if not bool(mask.any()):
                stratified[route_name] = dict(count=0, status='N/A_EMPTY')
            else:
                stratified[route_name] = action_metrics(h[mask], cache['delta'][mask], prediction[mask],
                    cache['z0'][mask], cache['z1'][mask], cache['logits0'][mask], cache['logits1'][mask],
                    cache['y'][mask], mean, g_fn, classifier_fn)
        metrics['route_strata'] = stratified
        reports[name] = metrics
    return reports


def fit_and_audit(identity, fit_x, fit_y, audit_x, audit_y, generator, *, fit_ids, audit_ids,
                  steps=200, batch_size=32, alpha_z=1., parameter_weight=1., learning_rate=2e-4,
                  kinds=('linear', 'temporal'), fit_modes=('legacy_self', 'block_z_self', 'block_z_cross'),
                  feature_fn=None, g_fn=None, classifier_fn=None, initial_states=None, on_step=None):
    """Return JSON-safe report/step_logs and CPU state_dicts for archival.

    IDs must identify physical source-L packets, not augmented view identifiers.
    Caller enforces source data and checkpoint provenance before invoking this API.
    initial_states optionally maps kind to legacy DisentanglementBranch state_dict;
    those original weights are audited before fitting separate fresh diagnostics.
    """
    if generator is None or str(generator.device) != 'cpu':
        raise ValueError('Private CPU generator required')
    if len(fit_ids) != len(fit_x) or len(audit_ids) != len(audit_x):
        raise ValueError('Physical ID count differs from role samples')
    fit_keys, audit_keys = list(map(str, fit_ids)), list(map(str, audit_ids))
    if len(set(fit_keys)) != len(fit_keys) or len(set(audit_keys)) != len(audit_keys):
        raise ValueError('Duplicate physical IDs within auxiliary role')
    if set(fit_keys) & set(audit_keys):
        raise ValueError('Auxiliary fit/audit physical packet overlap')
    if min(len(fit_x), len(audit_x)) < 2 or steps < 1 or not 1 <= batch_size <= 32:
        raise ValueError('Positive budget, batch_size <=32 and at least two samples per role required')
    allowed = {'legacy_self', 'block_z_self', 'block_z_cross'}
    if not set(fit_modes) <= allowed or not set(kinds) <= {'linear', 'temporal'}:
        raise ValueError('Unknown preregistered action mode/kind')
    feature_fn = feature_fn or (lambda x: intermediate(identity, x))
    g_fn = g_fn or (lambda h: identity_from_intermediate(identity, h))
    classifier_fn = classifier_fn or _encoder(identity).classify_features
    device = fit_x.device
    fit_y, audit_y = fit_y.long().to(device), audit_y.long().to(device)
    if audit_x.device != device or len(fit_y) != len(fit_x) or len(audit_y) != len(audit_x):
        raise ValueError('Role devices/counts inconsistent')
    report = dict(scope='source-L fixed E/G action diagnostic', identity_updated=False,
        target_access=False, fit_count=len(fit_x), audit_count=len(audit_x), physical_roles_disjoint=True,
        rank=8, qdim=8, steps=steps, batch_size=batch_size, learning_rate=learning_rate,
        alpha_z=alpha_z, parameter_weight=parameter_weight, optimizer='AdamW', weight_decay=1e-4,
        role_semantics='held-out physical packets; TX are shared, donors use a different TX within their own role',
        fresh_fit_caveat='Fresh fixed-teacher fits do not reproduce the original online auxiliary training.',
        early_stopping=False, audit_feedback_to_training=False, kinds={})
    states, logs = {}, []
    with fixed_identity(identity):
        with torch.no_grad():
            bases = []
            for x in (fit_x, audit_x):
                h = _batched(feature_fn, x, batch_size)
                z = _batched(g_fn, h, batch_size)
                bases.append((h, z, _batched(classifier_fn, z, batch_size)))
        for kind in kinds:
            # fork_rng prevents diagnostics changing any caller's model RNG.
            init_seed = int(torch.randint(2**31 - 1, (1,), generator=generator))
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(init_seed)
                template = DisentanglementBranch(kind, rank=8, state_dim=8).to(device)
            fit_cache = _cache(fit_x, fit_y, template, generator, feature_fn, g_fn, classifier_fn, batch_size, bases[0])
            audit_cache = _cache(audit_x, audit_y, template, generator, feature_fn, g_fn, classifier_fn, batch_size, bases[1])
            kind_report = dict(physical=audit_cache['physical'], spectra={
                'h': spectrum(audit_cache['delta']),
                'base': spectrum(audit_cache['delta'][:, :160]),
                'pa': spectrum(audit_cache['delta'][:, 160:320]),
                'reference': spectrum(audit_cache['delta'][:, 320:]),
                'z': spectrum(audit_cache['z1'] - audit_cache['z0'])}, modes={},
                same_intervention_donor_verified=bool((audit_y[audit_cache['donor']] != audit_y).all()),
                initializer_seed=init_seed, fit_block_scales=block_scales(fit_cache['delta']).tolist())
            if initial_states and kind in initial_states:
                import copy
                old = copy.deepcopy(template)
                old.load_state_dict(initial_states[kind], strict=True)
                kind_report['original_branch'] = _audit(old, audit_cache, fit_cache, g_fn, classifier_fn)
            else:
                kind_report['original_branch'] = {'status': 'N/A_NOT_SUPPLIED'}
            fit_seed = int(torch.randint(2**31 - 1, (1,), generator=generator))
            for mode in (*fit_modes, 'oracle_parameters'):
                import copy
                if mode == 'oracle_parameters':
                    branch = copy.deepcopy(template.operator)
                else:
                    branch = copy.deepcopy(template)
                # Identical sample schedule, initial operator, architecture and budget across modes.
                schedule = torch.Generator(device='cpu').manual_seed(fit_seed)
                mode_logs = _fit(branch, fit_cache, mode, g_fn, schedule, steps, batch_size,
                                 learning_rate, alpha_z, parameter_weight,
                                 on_step=(lambda row, k=kind: on_step(dict(kind=k, **row))) if on_step else None)
                logs.extend([dict(kind=kind, **row) for row in mode_logs])
                kind_report['modes'][mode] = _audit(branch, audit_cache, fit_cache, g_fn, classifier_fn,
                                                  oracle=mode == 'oracle_parameters')
                states[kind + '/' + mode] = {key: value.detach().cpu().clone() for key, value in branch.state_dict().items()}
            report['kinds'][kind] = kind_report
    return dict(report=report, step_logs=logs, state_dicts=states)


def composition_audit(identity, x, y, generator, state_dicts, *, batch_size=32,
                      feature_fn=None, g_fn=None, classifier_fn=None):
    """Audit ordered composition of previously fitted L/T on held-out source L.

    No fitting or reliability calibration occurs here. A four-corner residual is
    explicitly not interpreted as a pure physical interaction or noncommutativity.
    """
    from .action_checks import margin
    if generator is None or str(generator.device) != 'cpu':
        raise ValueError('Private CPU generator required')
    feature_fn = feature_fn or (lambda v: intermediate(identity, v))
    g_fn = g_fn or (lambda h: identity_from_intermediate(identity, h))
    classifier_fn = classifier_fn or _encoder(identity).classify_features
    y = y.long().to(x.device)
    with fixed_identity(identity), torch.no_grad():
        p_l = (2 * torch.rand(len(x), 4, generator=generator) - 1).to(x)
        p_t = (2 * torch.rand(len(x), 3, generator=generator) - 1).to(x)
        xl, xt = apply_linear(x, p_l), apply_temporal(x, p_t)
        xlt, xtl = apply_temporal(xl, p_t), apply_linear(xt, p_l)
        h0, hl, ht, hlt, htl = [_batched(feature_fn, view, batch_size)
                               for view in (x, xl, xt, xlt, xtl)]
        z0, zlt, ztl = [g_fn(h) for h in (h0, hlt, htl)]
        logits0, logitslt, logitstl = [classifier_fn(z) for z in (z0, zlt, ztl)]
        pure = hlt - hl - ht + h0
        report = dict(scope='held-out source-L composition diagnostic only', fitting=False,
            pure_lt_norm_mean=float(pure.norm(dim=1).mean()),
            pure_lt_energy=float(pure.square().mean()),
            true_sum_main_norm_mean=float((hl + ht - 2 * h0).norm(dim=1).mean()),
            actual_combined_norm_mean=float((hlt - h0).norm(dim=1).mean()),
            real_order_h_mse=float((hlt - htl).square().mean()),
            real_order_z_mse=float((zlt - ztl).square().mean()),
            real_order_margin_mae=float((margin(logitslt, y) - margin(logitstl, y)).abs().mean()),
            interpretation='Four-corner residual includes operator composition and encoder curvature; not pure physical interaction.',
            modes={})
        modes = sorted({key.split('/', 1)[1] for key in state_dicts if key.startswith('linear/')})
        for mode in modes:
            if 'temporal/' + mode not in state_dicts:
                continue
            with torch.random.fork_rng(devices=[]):
                if mode == 'oracle_parameters':
                    linear = BoundedConditionalOperator(rank=8, state_dim=8).to(x.device)
                    temporal = BoundedConditionalOperator(rank=8, state_dim=8).to(x.device)
                else:
                    linear = DisentanglementBranch('linear', rank=8, state_dim=8).to(x.device)
                    temporal = DisentanglementBranch('temporal', rank=8, state_dim=8).to(x.device)
            linear.load_state_dict(state_dicts['linear/' + mode], strict=True)
            temporal.load_state_dict(state_dicts['temporal/' + mode], strict=True)
            if mode == 'oracle_parameters':
                ql, qt = _operator_parameters(p_l), _operator_parameters(p_t)
                op_l, op_t = linear, temporal
                qt_after = qt
            else:
                ql = linear.encode_pair(linear.view(x), linear.view(xl))
                qt = temporal.encode_pair(temporal.view(x), temporal.view(xt))
                qt_after = temporal.encode_pair(temporal.view(xl), temporal.view(xlt))
                op_l, op_t = linear.operator, temporal.operator
            dl, dt = op_l(h0, ql), op_t(h0, qt)
            additive = h0 + dl + dt
            sequential = h0 + dl + op_t(h0 + dl, qt)
            reestimated = h0 + dl + op_t(h0 + dl, qt_after)
            results = {}
            for name, prediction in (('additive', additive), ('sequential_reused_q', sequential),
                                     ('sequential_reestimated_q', reestimated)):
                zp = g_fn(prediction)
                delta = prediction - h0
                results[name] = dict(h_mse=float((prediction - hlt).square().mean()),
                    z_mse=float((zp - zlt).square().mean()),
                    margin_delta_mae=float((margin(classifier_fn(zp), y) - margin(logitslt, y)).abs().mean()),
                    total_action_norm_mean=float(delta.norm(dim=1).mean()),
                    exceeds_single_branch_radius_rate=float((delta.norm(dim=1) > .25 * h0.norm(dim=1)).float().mean()))
            results['q_state_change_rms'] = float((qt_after - qt).square().mean().sqrt())
            report['modes'][mode] = results
        report['clean_margin_mean'] = float(margin(logits0, y).mean())
    return report
