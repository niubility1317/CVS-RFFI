"""Synthetic mathematical checks; no source/target data or model weights are read."""
import json
import torch
from torch import nn
from torch.nn import functional as F
from . import ridge


def checks():
    torch.set_num_threads(2)
    torch.manual_seed(730)
    support = torch.randn(18, 160, dtype=torch.float64, requires_grad=True)
    query = torch.randn(12, 160, dtype=torch.float64, requires_grad=True)
    sy = torch.arange(6).repeat(3)
    qy = torch.arange(6).repeat(2)
    actual = ridge.ridge_logits(support, sy, query)
    s = F.normalize(support, dim=1, eps=1e-4)
    q = F.normalize(query, dim=1, eps=1e-4)
    targets = F.one_hot(sy, 6).double()
    primal_system = s.T @ s + .1 * torch.eye(160, dtype=torch.float64)
    primal = 10. * q @ torch.linalg.solve(primal_system, s.T @ targets)
    assert torch.allclose(actual, primal, atol=2e-12, rtol=2e-12)
    loss = ridge.episode_loss(support, sy, query, qy)
    assert torch.equal(loss, F.cross_entropy(actual, qy))
    gs, gq = torch.autograd.grad(loss, (support, query))
    assert torch.isfinite(gs).all() and gs.norm() > 0
    assert torch.isfinite(gq).all() and gq.norm() > 0
    # Numerical derivatives verify the whole normalization / dual-solve path,
    # including the support dependence of the fitted temporary coefficients.
    gradient_check = torch.autograd.gradcheck(
        lambda a, b: ridge.episode_loss(a, sy, b, qy), (support, query),
        eps=1e-6, atol=1e-5, rtol=1e-4, fast_mode=True)
    assert gradient_check
    permutation = torch.tensor([3, 5, 0, 4, 1, 2])
    renamed = ridge.ridge_logits(support, permutation[sy], query)
    assert torch.allclose(actual, renamed[:, permutation], atol=2e-12, rtol=2e-12)
    assert torch.allclose(loss, ridge.episode_loss(support, permutation[sy], query, permutation[qy]),
                          atol=2e-12, rtol=2e-12)
    support_order = torch.randperm(len(support))
    reordered = ridge.ridge_logits(support[support_order], sy[support_order], query)
    assert torch.allclose(actual, reordered, atol=2e-12, rtol=2e-12)
    single = torch.cat([ridge.ridge_logits(support, sy, row[None]) for row in query])
    assert torch.allclose(actual, single, atol=2e-12, rtol=2e-12)
    changed_query = query.detach().clone()
    changed_query[1:] = torch.randn_like(changed_query[1:])
    assert torch.equal(actual[0], ridge.ridge_logits(support, sy, changed_query)[0])
    assert not any(isinstance(v, (nn.Parameter, nn.Module, torch.Tensor))
                   for v in vars(ridge).values())
    assert ridge.ridge_logits.__closure__ is None and ridge.episode_loss.__closure__ is None
    # Rank-one repeated supports and the fully zero system remain well-defined.
    duplicate = torch.randn(1, 160).repeat(18, 1).requires_grad_(True)
    duplicate_query = torch.randn(12, 160, requires_grad=True)
    duplicate_loss = ridge.episode_loss(duplicate, sy, duplicate_query, qy)
    duplicate_grads = torch.autograd.grad(duplicate_loss, (duplicate, duplicate_query))
    assert torch.isfinite(duplicate_loss) and all(torch.isfinite(g).all() for g in duplicate_grads)
    zeros = torch.zeros(18, 160, requires_grad=True)
    zero_query = torch.zeros(12, 160, requires_grad=True)
    zero_loss = ridge.episode_loss(zeros, sy, zero_query, qy)
    zero_grads = torch.autograd.grad(zero_loss, (zeros, zero_query))
    assert torch.isfinite(zero_loss) and all(torch.isfinite(g).all() for g in zero_grads)
    fp32_s = support.detach().float().requires_grad_(True)
    fp32_q = query.detach().float().requires_grad_(True)
    fp32_logits = ridge.ridge_logits(fp32_s, sy, fp32_q)
    fp32_loss = ridge.episode_loss(fp32_s, sy, fp32_q, qy)
    fp32_grads = torch.autograd.grad(fp32_loss, (fp32_s, fp32_q))
    assert fp32_logits.dtype == torch.float32 and torch.isfinite(fp32_logits).all()
    assert all(torch.isfinite(g).all() and g.norm() > 0 for g in fp32_grads)
    # Explicitly check an outer mixed-precision context cannot change solve dtype.
    with torch.autocast(device_type='cpu', dtype=torch.bfloat16):
        autocast_logits = ridge.ridge_logits(fp32_s, sy, fp32_q)
    assert autocast_logits.dtype == torch.float32 and torch.equal(fp32_logits, autocast_logits)
    negative_cases = []
    def reject(name, call):
        try:
            call()
        except ValueError:
            negative_cases.append(name)
        else:
            raise AssertionError('Invalid episode accepted: ' + name)
    reject('support_dim', lambda: ridge.ridge_logits(torch.zeros(18, 159), sy, fp32_q))
    reject('query_dim', lambda: ridge.ridge_logits(fp32_s, sy, torch.zeros(12, 161)))
    reject('empty_support', lambda: ridge.ridge_logits(torch.zeros(0, 160), torch.zeros(0, dtype=torch.long), fp32_q))
    reject('empty_query', lambda: ridge.ridge_logits(fp32_s, sy, torch.zeros(0, 160)))
    reject('non_tensor', lambda: ridge.ridge_logits([[0.]], sy, fp32_q))
    reject('half_features', lambda: ridge.ridge_logits(fp32_s.half(), sy, fp32_q.half()))
    reject('integer_features', lambda: ridge.ridge_logits(fp32_s.long(), sy, fp32_q))
    reject('mixed_dtype', lambda: ridge.ridge_logits(fp32_s, sy, query))
    nan_support = fp32_s.detach().clone();nan_support[0, 0] = float('nan')
    inf_query = fp32_q.detach().clone();inf_query[0, 0] = float('inf')
    reject('nonfinite_support', lambda: ridge.ridge_logits(nan_support, sy, fp32_q))
    reject('nonfinite_query', lambda: ridge.ridge_logits(fp32_s, sy, inf_query))
    reject('support_label_dtype', lambda: ridge.ridge_logits(fp32_s, sy.int(), fp32_q))
    reject('support_label_shape', lambda: ridge.ridge_logits(fp32_s, sy[:, None], fp32_q))
    reject('support_label_count', lambda: ridge.ridge_logits(fp32_s, sy[:-1], fp32_q))
    high = sy.clone();high[0] = 6
    low = sy.clone();low[0] = -1
    nan_labels = sy.float();nan_labels[0] = float('nan')
    reject('support_label_high', lambda: ridge.ridge_logits(fp32_s, high, fp32_q))
    reject('support_label_low', lambda: ridge.ridge_logits(fp32_s, low, fp32_q))
    reject('support_label_nonfinite', lambda: ridge.ridge_logits(fp32_s, nan_labels, fp32_q))
    reject('query_label_dtype', lambda: ridge.episode_loss(fp32_s, sy, fp32_q, qy.float()))
    reject('query_label_count', lambda: ridge.episode_loss(fp32_s, sy, fp32_q, qy[:-1]))
    bad_qy = qy.clone();bad_qy[0] = 6
    reject('query_label_range', lambda: ridge.episode_loss(fp32_s, sy, fp32_q, bad_qy))
    return dict(status='PASS', feature_dim=160, classes=6, ridge=.1, logit_scale=10.,
                normalization_epsilon=1e-4, primal_max_abs_error=float((actual-primal).detach().abs().max()),
                support_gradient_norm=float(gs.norm()), query_gradient_norm=float(gq.norm()),
                numerical_gradient_check=gradient_check, class_permutation_equivalent=True,
                support_order_equivalent=True, packet_query_independent=True,
                persistent_parameters=0, duplicate_support_finite=True, zero_support_finite=True,
                fp32_and_outer_autocast_verified=True, negative_cases=negative_cases,
                real_IQ_or_checkpoint_read=False, experiment_started=False)


if __name__ == '__main__':
    print(json.dumps(checks(), allow_nan=False))
