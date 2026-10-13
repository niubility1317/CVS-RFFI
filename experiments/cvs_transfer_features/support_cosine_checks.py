"""Synthetic focused checks for the fixed ordinary support cosine interface."""
import json
import torch
from torch.nn import functional as F
from unittest.mock import patch
from . import support_cosine as new
from experiments.cvs_source_transfer_probe import readout as old


def checks():
    torch.set_num_threads(2)
    torch.manual_seed(1419)
    features = torch.randn(18, 160)
    labels = torch.arange(6).repeat(3)
    query = torch.randn(11, 160)
    source_before = features.clone()
    rng_before = torch.random.get_rng_state().clone()
    normalized = F.normalize(features, dim=1, eps=1e-4)
    weight_old, audit_old = old.fit(normalized, labels, 7153, epochs=20, batch_size=len(labels))
    weight, audit = new.fit(features, labels, 7153, num_classes=6)
    assert torch.equal(weight, weight_old)
    assert all(audit[key] == audit_old[key] for key in ('epochs', 'updates', 'losses', 'feature_updates'))
    output = new.logits(query, weight, num_classes=6)
    output_old = old.logits(F.normalize(query, dim=1, eps=1e-4), weight_old)
    assert torch.equal(output, output_old)
    assert torch.equal(features, source_before) and features.grad is None
    assert torch.equal(rng_before, torch.random.get_rng_state())
    assert not weight.requires_grad and audit['head_updates'] == 20 and audit['full_support_batch']
    # Inspect the actual optimizer updates: only the fresh C x160 matrix is
    # differentiable, while the supplied frozen feature tensor is never changed.
    optimizer_steps = []
    actual_adam = torch.optim.Adam
    class AuditedAdam(actual_adam):
        def step(self, closure=None):
            params = [p for group in self.param_groups for p in group['params']]
            assert len(params) == 1 and params[0].shape == (8, 160)
            assert params[0].requires_grad and params[0].grad is not None
            assert torch.isfinite(params[0].grad).all() and params[0].grad.norm() > 0
            optimizer_steps.append(float(params[0].grad.norm()))
            return super().step(closure)
    support8 = torch.randn(16, 160)
    before8 = support8.clone()
    labels8 = torch.arange(8).repeat(2)
    with patch.object(torch.optim, 'Adam', AuditedAdam):
        weights8, audit8 = new.fit(support8, labels8, 7154, num_classes=8)
    assert len(optimizer_steps) == 20 and support8.grad is None and torch.equal(support8, before8)
    class_cases = {8: dict(shape=list(weights8.shape), head_updates=audit8['head_updates'])}
    for count in (26, 125):
        support = torch.randn(count * 2, 160)
        target = torch.arange(count).repeat(2)
        before = support.clone()
        fitted, info = new.fit(support, target, 8000 + count, num_classes=count)
        values = new.logits(torch.randn(5, 160), fitted, num_classes=count)
        assert fitted.shape == (count, 160) and values.shape == (5, count)
        assert torch.isfinite(fitted).all() and torch.isfinite(values).all()
        assert info['head_updates'] == info['updates'] == 20 and info['feature_updates'] == 0
        assert torch.equal(support, before) and support.grad is None
        assert not fitted.requires_grad
        class_cases[count] = dict(shape=list(fitted.shape), head_updates=info['head_updates'])
    # Fixed weights transform with class permutation; no query batch fitting or
    # statistics are part of the score function.
    permutation = torch.tensor([5, 1, 7, 0, 3, 2, 6, 4])
    q = torch.randn(7, 160)
    before_q = q.clone()
    values = new.logits(q, weights8, num_classes=8)
    relabeled = new.logits(q, weights8[permutation], num_classes=8)
    assert torch.allclose(relabeled, values[:, permutation], atol=2e-6, rtol=2e-6)
    individual = torch.cat([new.logits(row[None], weights8, num_classes=8) for row in q])
    assert torch.allclose(values, individual, atol=2e-5, rtol=2e-5)
    changed = q.clone();changed[1:] = torch.randn_like(changed[1:])
    assert torch.equal(values[0], new.logits(changed, weights8, num_classes=8)[0])
    assert torch.equal(q, before_q) and q.grad is None
    assert torch.isfinite(new.logits(torch.zeros(4, 160), weights8, num_classes=8)).all()
    zero = torch.zeros(12, 160)
    zero_labels = torch.arange(6).repeat(2)
    zero_weight, zero_audit = new.fit(zero, zero_labels, 7161, num_classes=6)
    assert torch.isfinite(zero_weight).all() and all(torch.isfinite(torch.tensor(v)) for v in zero_audit['losses'])
    assert zero_audit['updates'] == 20
    with torch.autocast(device_type='cpu', dtype=torch.bfloat16):
        autocast_weight, autocast_audit = new.fit(features, labels, 7153, num_classes=6)
        autocast_values = new.logits(query, autocast_weight, num_classes=6)
    assert torch.equal(weight, autocast_weight) and audit == autocast_audit
    assert torch.equal(output, autocast_values)
    negative_cases = []
    def reject(name, call):
        try:call()
        except (ValueError, TypeError):negative_cases.append(name)
        else:raise AssertionError('Invalid support readout accepted: ' + name)
    reject('explicit_classes_required', lambda: new.fit(features, labels, 7153))
    for name, bad in [('float_classes', 6.), ('bool_classes', True), ('zero_classes', 0), ('negative_classes', -1)]:
        reject(name, lambda: new.fit(features, labels, 7153, num_classes=bad))
    for name, bad in [('feature_shape', torch.zeros(18, 159)), ('feature_rank', torch.zeros(18, 2, 160)),
                      ('empty_support', torch.zeros(0, 160)), ('double_support', features.double()),
                      ('half_support', features.half()), ('integer_support', features.long()),
                      ('trainable_support', features.clone().requires_grad_(True))]:
        reject(name, lambda: new.fit(bad, labels, 7153, num_classes=6))
    missing = labels.clone();missing[missing == 5] = 4
    high = labels.clone();high[0] = 6
    low = labels.clone();low[0] = -1
    for name, bad in [('missing_class', missing), ('label_high', high), ('label_low', low),
                      ('label_int32', labels.int()), ('label_float', labels.float()),
                      ('label_shape', labels[:, None]), ('label_count', labels[:-1])]:
        reject(name, lambda: new.fit(features, bad, 7153, num_classes=6))
    for value in (float('nan'), float('inf')):
        bad = features.clone();bad[0, 0] = value
        reject('nonfinite_support_' + str(value), lambda: new.fit(bad, labels, 7153, num_classes=6))
    reject('query_class_count', lambda: new.logits(query, weight, num_classes=8))
    reject('query_requires_grad', lambda: new.logits(query.clone().requires_grad_(True), weight, num_classes=6))
    reject('query_dtype', lambda: new.logits(query.double(), weight, num_classes=6))
    reject('query_shape', lambda: new.logits(torch.zeros(11, 159), weight, num_classes=6))
    reject('weight_requires_grad', lambda: new.logits(query, weight.clone().requires_grad_(True), num_classes=6))
    reject('weight_dtype', lambda: new.logits(query, weight.double(), num_classes=6))
    bad = weight.clone();bad[0, 0] = float('nan')
    reject('weight_nonfinite', lambda: new.logits(query, bad, num_classes=6))
    bad = query.clone();bad[0, 0] = float('inf')
    reject('query_nonfinite', lambda: new.logits(bad, weight, num_classes=6))
    return dict(status='PASS', six_class_existing_fit_W_exact=True,
            six_class_existing_fit_logs_exact=True, six_class_existing_logits_exact=True,
            equivalence_input='new raw features; existing fit/logits receive the same pre-normalized rows',
            classes=class_cases, updates=20, optimizer_parameter_tensors=1,
            W_only_gradient_steps=len(optimizer_steps), feature_updates=0,
            class_permutation_fixed_W_equivalent=True, query_packet_independent=True,
            zero_input_finite=True, input_tensors_unchanged=True, private_rng_unchanged=True,
            outer_autocast_FP32_stable=True, negative_cases=negative_cases,
            real_data_or_checkpoint_read=False, experiment_started=False,
            Phase2_validation_claim=False)


if __name__ == '__main__':
    print(json.dumps(checks(), allow_nan=False))
