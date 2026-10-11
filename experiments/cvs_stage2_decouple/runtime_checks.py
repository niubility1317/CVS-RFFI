"""Real identity graph and bounded second-stage integration checks; no target I/O."""
import argparse
import copy
import json
from pathlib import Path
import time

import torch

from . import design as d
from .common import identity_logits, margins, parameter_groups, learning_rate
from .source import objective, SourceStream


def receiver_fixture():
    from .receiver_checks import fixture
    from .receiver import fit_receiver, ReceiverBank
    cache = fixture(per_group=8, rx_count=2)
    payload = fit_receiver(cache, 481, fit_steps=0)
    return cache, ReceiverBank(payload)


def active_sample(bank):
    sample = bank.sample_relations(torch.Generator().manual_seed(53))
    for relation in sample['relations']:
        assert relation['valid']
        relation.update(active=True, c=.8, tau=20., delta=.5, reason='synthetic active path')
    sample['diagnostics']['active_relations'] = 2
    return sample


def objective_checks(bank):
    row = next(r for r in d.rows() if r['arm'] == 'C3')
    c = d.config(row)
    y = torch.zeros(128, dtype=torch.long)
    logits = torch.zeros(128, 6, requires_grad=True)
    parent_margin = torch.full((128,), 6.)
    tau = torch.full((128,), 8.)
    parent_correct = torch.ones(128, dtype=torch.bool)
    aug = torch.zeros(32, 6, requires_grad=True)
    with torch.no_grad():
        logits[:, 0] = 4.
        aug[:, 0] = 1.
    loss, weighted, parts = objective(c, logits, y, parent_margin, parent_correct, tau, 444, aug)
    assert torch.isfinite(loss)
    clean_drop, changed_drop = torch.autograd.grad(parts['drop'], [logits, aug], allow_unused=True, retain_graph=True)
    assert clean_drop is None or not clean_drop.any()
    assert changed_drop is not None and changed_drop.abs().sum() > 0
    positions = torch.arange(96, 128)
    _, _, relation_action = objective(c, logits, y, parent_margin, parent_correct, tau, 444,
                                      aug_logits=aug, action_positions=positions)
    selected_clean, selected_aug = torch.autograd.grad(relation_action['drop'], [logits, aug],
                                                      allow_unused=True, retain_graph=True)
    assert selected_clean is None or not selected_clean.any()
    assert selected_aug is not None and selected_aug.abs().sum() > 0
    # Parent errors must not impose an imitation floor; denominator remains full batch.
    parent_correct[1:] = False
    _, _, one = objective(c, logits, y, parent_margin, parent_correct, tau, 444)
    expected = torch.tensor((6. - 4. - .5 - .5) / 128)
    assert torch.allclose(one['keep'], expected)
    sample = active_sample(bank)
    values = torch.zeros(128, requires_grad=True)
    strong, weak = [], []
    for relation in sample['relations']:
        strong.extend(96 + i for i in relation['strong_positions'])
        weak.extend(96 + i for i in relation['weak_positions'])
    with torch.no_grad():
        values[strong] = 4.
    rloss, diagnostics = bank.identity_loss(values, sample, offset=96)
    gradient, = torch.autograd.grad(rloss, values)
    assert rloss > 0 and gradient[weak].abs().sum() > 0 and not gradient[strong].any()
    assert not gradient[:96].any() and diagnostics['fixed_denominator'] == 2
    # C0 has no auxiliary objective or action forward, and its derivative is plain CE.
    c0 = d.config(next(r for r in d.rows() if r['arm'] == 'C0'))
    total, weighted0, _ = objective(c0, logits, y, parent_margin, parent_correct, tau, 444)
    expected = torch.nn.functional.cross_entropy(logits, y)
    assert torch.equal(total, expected)
    assert all(float(v.detach()) == 0 for k, v in weighted0.items() if k != 'ce')
    return dict(drop_clean_direct_gradient_zero=True, drop_augmented_gradient_nonzero=True,
                receiver_strong_direct_gradient_zero=True, receiver_weak_gradient_nonzero=True,
                keep_full_batch_denominator=True, parent_wrong_no_imitation=True, pure_ce_objective=True,
                relation_action_positions_supported=True)


def schedule_checks(cache, bank):
    reference = None
    counts = {}
    for arm in d.ARMS:
        stream = SourceStream(len(cache['y']), bank, d.SEEDS[0])
        result = []
        count = dict(linear=0, temporal=0, relation=0)
        for step in range(1, 8881):
            indices, sample = stream.step(step)
            assert indices.shape == (128,)
            result.append(indices)
            if sample:
                count['relation'] += 1
                assert sample['indices'].shape == (32,)
                assert all(r['independent_ids'] == 16 for r in sample['relations'])
            if step % 4 == 0:
                kind = ['linear', 'linear', 'temporal', 'temporal'][(step // 4 - 1) % 4]
                count[kind] += 1
        result = torch.stack(result)
        if reference is None:
            reference = result
        else:
            assert torch.equal(reference, result)
        assert count == dict(linear=1110, temporal=1110, relation=1110)
        counts[arm] = count
    for peak in (3e-5, 1e-4):
        assert learning_rate(1, peak) > 1e-6
        assert learning_rate(444, peak) == peak
        assert learning_rate(445, peak) < peak
        assert learning_rate(8880, peak) == 1e-6
        for invalid in (0, 8881):
            try:
                learning_rate(invalid, peak)
            except ValueError:
                pass
            else:
                raise AssertionError('Out-of-budget step accepted')
    return dict(full_8880_stream_equal_across_four_arms=True, counts=counts,
                learning_rate_endpoints=True, shared_sampling_is_not_identity_auxiliary=True)


def actual_graph(device, bank):
    from experiments.cvs_phase1_stack.runtime import installed
    from experiments.cvs_equivariant_identity.precision import numerical_context
    from .actions import PhysicalActionNet, action_contract, load_actions
    c = d.config(next(r for r in d.rows() if r['arm'] == 'C3'))
    torch.manual_seed(993)
    a = d.make_args(c, device)
    a.num_domains = 5
    a.input_len = 256
    with numerical_context(d.FULL_FP32_POLICY), installed(c) as native:
        # The test never calls common.build_model, so no unavailable parent is read.
        model = native.build_baseline_model(a, torch.device(device))
    head = model.id_backbone.encoder.core.id_backbone.cls_head
    assert head.scale == 30. and not torch.is_tensor(head.scale)
    assert not any('scale' in n for n, _ in head.named_parameters())
    groups, names = parameter_groups(model)
    flattened = [p for group in groups for p in group['params']]
    assert len(flattened) == len({id(p) for p in flattened})
    assert {id(p) for p in flattened} == {id(p) for p in model.id_backbone.parameters()}
    assert not any(p.requires_grad for name, p in model.named_parameters() if not name.startswith('id_backbone.'))
    misclassified = [name for name in names['encoder'] if any(token in name for token in ('response_gain', 't_proj', 'f_proj'))]
    x = torch.randn(128, 2, 256, device=device)
    x = x / x.square().sum(1, keepdim=True).mean(2, keepdim=True).sqrt()
    y = torch.arange(128, device=device) % 6
    domain_calls = []
    hook = model.dom_backbone.register_forward_hook(lambda *args: domain_calls.append(1))
    model.eval()
    with torch.no_grad():
        expected = model(x[:4])
        observed = identity_logits(model, x[:4])
    hook.remove()
    assert torch.equal(expected, observed)
    identity_calls = []
    hook = model.dom_backbone.register_forward_hook(lambda *args: identity_calls.append(1))
    with torch.no_grad():
        identity_logits(model, x[:4])
    hook.remove()
    assert not identity_calls
    # A mature response branch usually has nonzero gain. This synthetic value
    # exercises its graph without claiming to be a legitimate training parent.
    with torch.no_grad():
        model.id_backbone.encoder.response_gain.fill_(.1)
        eval_logits_before = identity_logits(model, x)
        eval_margins_before = margins(eval_logits_before, y)
    frozen_before = {n: p.detach().clone() for n, p in model.named_parameters() if not p.requires_grad}
    buffers_before = {n: b.detach().clone() for n, b in model.named_buffers() if not n.startswith('id_backbone.')}
    original_before = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
    payload = dict(format_version=1, seed=391, contract=action_contract(), networks={
        kind: PhysicalActionNet(kind).state_dict() for kind in ('linear', 'temporal')})
    actions = load_actions(payload, device)
    actions_before = copy.deepcopy(actions.networks.state_dict())
    if str(device).startswith('cuda'):
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    started = time.perf_counter()
    optimizer = torch.optim.AdamW(groups, weight_decay=1e-4)
    for group in optimizer.param_groups:
        group['lr'] = learning_rate(444, group['peak'])
    model.train()
    optimizer.zero_grad(set_to_none=True)
    logits = identity_logits(model, x)
    action_positions = torch.arange(96, 128, device=device)
    augmented, stats = actions.transform(x[action_positions], 'linear', 477)
    aug_logits = identity_logits(model, augmented)
    sample = active_sample(bank)
    actual_margins = margins(logits, y).detach()
    for relation in sample['relations']:
        # Synthetic graph probe: ensure a violated relation instead of mistaking
        # an already-satisfied hinge for a detached encoder. Production sampling
        # is untouched and never ranks current student margins this way.
        members = relation['strong_positions'] + relation['weak_positions']
        ranked = sorted(members, key=lambda i: float(actual_margins[96 + i]))
        relation['strong_positions'] = ranked[8:]
        relation['weak_positions'] = ranked[:8]
        relation['delta'] = 0.
    parent_margin = margins(logits, y).detach() + 2.
    parent_correct = torch.ones(128, device=device, dtype=torch.bool)
    tau = torch.full((128,), 20., device=device)
    loss, weighted, _ = objective(c, logits, y, parent_margin, parent_correct, tau, 444,
                                aug_logits=aug_logits, relation=sample, bank=bank, action_positions=action_positions)
    assert torch.isfinite(loss)
    r_grads = torch.autograd.grad(weighted['receiver'], groups[0]['params'], retain_graph=True, allow_unused=True)
    r_encoder_norm = sum(g.detach().square().sum() for g in r_grads if g is not None).sqrt()
    assert torch.isfinite(r_encoder_norm) and r_encoder_norm > 0
    r_logits_gradient, = torch.autograd.grad(weighted['receiver'], logits, retain_graph=True)
    strong_ids = [96 + i for r in sample['relations'] for i in r['strong_positions']]
    weak_ids = [96 + i for r in sample['relations'] for i in r['weak_positions']]
    assert not r_logits_gradient[strong_ids].any()
    assert r_logits_gradient[weak_ids].abs().sum() > 0
    loss.backward()
    missing = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is None]
    assert all(torch.isfinite(p.grad).all() for p in flattened if p.grad is not None)
    nonzero_gradient = [n for n, p in model.named_parameters()
                        if p.requires_grad and p.grad is not None and p.grad.abs().sum() > 0]
    optimizer.step()
    if str(device).startswith('cuda'):
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    assert all(torch.equal(value, dict(model.named_parameters())[name]) for name, value in frozen_before.items())
    assert all(torch.equal(value, dict(model.named_buffers())[name]) for name, value in buffers_before.items())
    assert all(torch.equal(value, actions.networks.state_dict()[name]) for name, value in actions_before.items())
    changed = [n for n, p in model.named_parameters() if n in original_before and not torch.equal(p.detach(), original_before[n])]
    assert changed and nonzero_gradient
    model.eval()
    with torch.no_grad():
        eval_logits_after = identity_logits(model, x)
        eval_margins_after = margins(eval_logits_after, y)
    margin_trajectory = {}
    for name, indices in [('clean', list(range(128))), ('strong', strong_ids), ('weak', weak_ids)]:
        before = eval_margins_before[indices].mean()
        after = eval_margins_after[indices].mean()
        margin_trajectory[name] = dict(before=float(before), after=float(after), difference=float(after - before),
            accuracy_before=float((eval_logits_before[indices].argmax(1) == y[indices]).float().mean()),
            accuracy_after=float((eval_logits_after[indices].argmax(1) == y[indices]).float().mean()))
    result = dict(device=str(device), hardware=torch.cuda.get_device_name(device) if str(device).startswith('cuda') else 'CPU',
        forward_identity_matches_eval=True, identity_only_skips_domain=True, classifier_scale=30.,
        identity_parameters=sum(p.numel() for p in flattened), group_parameters={g['name']: sum(p.numel() for p in g['params']) for g in groups},
        parameter_groups=names, parameters_missing_gradient=missing, fusion_parameters_misclassified=misclassified,
        nonzero_gradient_parameter_count=len(nonzero_gradient), changed_parameter_count=len(changed),
        domain_parameters_and_buffers_unchanged=True, frozen_actions_unchanged=True,
        receiver_encoder_gradient_norm=float(r_encoder_norm),
        receiver_strong_logit_direct_gradient_zero=True, receiver_weak_logit_gradient_nonzero=True,
        actual_same_batch_eval_margin_trajectory=margin_trajectory,
        full_batch_clean=128, action_batch=32, loss=float(loss.detach()),
        weighted_loss={k: float(v.detach()) for k, v in weighted.items()}, step_seconds=elapsed,
        peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated() if str(device).startswith('cuda') else None,
        peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved() if str(device).startswith('cuda') else None,
        scope='Synthetic architecture, one integration update; no source/target samples or legitimate parent loaded')
    del model
    return result


def run(device):
    torch.set_num_threads(2)
    if str(device).startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('Requested CUDA runtime unavailable')
    cache, bank = receiver_fixture()
    result = dict(objective=objective_checks(bank), schedule=schedule_checks(cache, bank))
    result['actual_graph'] = actual_graph(device, bank)
    result['status'] = 'FAIL' if (result['actual_graph']['parameters_missing_gradient'] or result['actual_graph']['fusion_parameters_misclassified']) else 'PASS'
    result['target_access'] = False
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = run(args.device)
    d.write(args.output, result)
    print(json.dumps(dict(status=result['status'], output=str(args.output), actual_graph={
        k: v for k, v in result['actual_graph'].items() if k != 'parameter_groups'})))
    if result['status'] != 'PASS':
        raise SystemExit(1)
