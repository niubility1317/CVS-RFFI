"""Bounded U-only feature adaptation; no data loaders, files, or truth access."""

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


RUN_ID = '20261030-phase3-star-reciprocal-adapter-s2026092701-r01'
MODEL_SEED = 2026092701
STEPS = 200
PARAMETERS = 10432
RESIDUAL_LIMIT = .1


def _raw_tensor(value):
    if (not isinstance(value, torch.Tensor) or value.dtype != torch.float32 or value.ndim != 2
            or value.shape[1] != 160 or not len(value) or not bool(torch.isfinite(value).all())):
        raise ValueError('Raw fused features must be finite nonempty FP32 [N,160]')
    norms = torch.linalg.vector_norm(value, dim=1, keepdim=True)
    if not bool(torch.isfinite(norms).all()) or not bool((norms > 0).all()):
        raise ValueError('Raw fused feature norms must be finite and nonzero')
    return norms


def _raw_numpy(value):
    if (not isinstance(value, np.ndarray) or value.dtype != np.float32 or value.ndim != 2
            or value.shape[1] != 160 or not len(value) or not np.isfinite(value).all()):
        raise ValueError('Raw fused features must be finite nonempty FP32 [N,160]')
    # FP64 validation avoids accepting FP32 norm overflow as a usable feature.
    norms = np.linalg.norm(value.astype(np.float64), axis=1)
    if np.any(norms == 0) or np.any(norms > np.finfo(np.float32).max):
        raise ValueError('Raw fused feature norms must be finite and nonzero in FP32')
    return value


class Adapter(nn.Module):
    """160→32→160 residual on directions, preserving each original raw norm."""

    def __init__(self):
        super().__init__()
        # Both arms start identically without changing the caller's CPU RNG.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(MODEL_SEED)
            self.fc1 = nn.Linear(160, 32, device='cpu', dtype=torch.float32)
            self.fc2 = nn.Linear(32, 160, device='cpu', dtype=torch.float32)
            nn.init.zeros_(self.fc2.weight)
            nn.init.zeros_(self.fc2.bias)

    def _parts(self, raw):
        norm = _raw_tensor(raw)
        unit = raw / norm
        residual = self.fc2(F.relu(self.fc1(unit)))
        length = torch.linalg.vector_norm(residual, dim=1, keepdim=True)
        residual = residual / torch.clamp(length / RESIDUAL_LIMIT, min=1.)
        if not bool(torch.isfinite(residual).all()):
            raise ValueError('Adapter residual must remain finite')
        return norm, unit, residual

    def residual(self, raw):
        """The bounded residual before the final directional normalization."""
        return self._parts(raw)[2]

    def forward(self, raw):
        norm, unit, residual = self._parts(raw)
        return norm * F.normalize(unit + residual, dim=1, eps=1e-12)


def gram_loss(student, original):
    """Off-diagonal unit-Gram MSE with an explicitly detached original teacher."""
    if (student.ndim != 2 or student.shape != original.shape or student.shape[1] != 160
            or not len(student) or not bool(torch.isfinite(student).all())
            or not bool(torch.isfinite(original).all())):
        raise ValueError('Aligned finite student/original [N,160] features are required')
    a = F.normalize(student, dim=1, eps=1e-12)
    b = F.normalize(original.detach(), dim=1, eps=1e-12)
    if len(a) == 1:
        return a.sum() * 0
    difference = a @ a.T - b @ b.T
    mask = ~torch.eye(len(a), dtype=torch.bool, device=a.device)
    return difference[mask].square().mean()


def _logits(classify_fn, values, need_gradient=False):
    logits = classify_fn(values)
    if (not isinstance(logits, torch.Tensor) or logits.dtype != torch.float32
            or tuple(logits.shape) != (len(values), 6) or not bool(torch.isfinite(logits).all())):
        raise ValueError('Frozen default head must return finite FP32 [N,6] logits')
    if need_gradient and not logits.requires_grad:
        raise ValueError('Default head must retain the input-feature gradient')
    return logits


def fit(raw_U, labels, classify_fn, device='cuda:0'):
    """Fit only the adapter using fixed hard pseudo-labels and all-U structure."""
    values = _raw_numpy(raw_U)
    if (not isinstance(labels, np.ndarray) or labels.dtype != np.int64
            or labels.shape != (len(values),) or np.any((labels < -1) | (labels > 5))):
        raise ValueError('Fixed aligned int64 pseudo-labels must be -1 or 0..5')
    owner = classify_fn if isinstance(classify_fn, nn.Module) else getattr(classify_fn, '__self__', None)
    if isinstance(owner, nn.Module) and any(p.requires_grad for p in owner.parameters()):
        raise ValueError('Original default-head/F0 parameters must already be frozen')
    model = Adapter().to(device)
    before = {key: value.detach().clone() for key, value in model.state_dict().items()}
    selected_count = int(np.count_nonzero(labels >= 0))
    summary = dict(status='SKIPPED_NO_PSEUDOLABELS', steps=0, U_count=len(values), selected_count=selected_count,
        model_seed=MODEL_SEED, parameters=PARAMETERS, optimizer=None,
        training_adapter_forward_calls=0, training_adapter_forward_rows=0,
        diagnostic_adapter_forward_calls=0, diagnostic_adapter_forward_rows=0,
        adapter_forward_calls=0, adapter_forward_rows=0,
        classifier_forward_calls=0, classifier_forward_rows=0,
        F0_updates=0, G_updates=0, parameters_changed=False, parameter_changed_count=0,
        max_abs_parameter_change=0., max_abs_feature_change=0., max_pre_normalization_residual=0.,
        finite=True, loss_initial=None, loss_final=None)
    trace = []
    if selected_count:
        x = torch.tensor(values.tolist(), dtype=torch.float32, device=device)
        y = torch.tensor(labels.tolist(), dtype=torch.int64, device=device)
        selected = y >= 0
        original = F.normalize(x, dim=1, eps=1e-12).detach()
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001,
            betas=(.9, .999), eps=1e-8, foreach=False)
        model.train()
        for step in range(1, STEPS + 1):
            optimizer.zero_grad(set_to_none=True)
            # One full-U adapter and default-head forward per update.
            norm, unit, residual = model._parts(x)
            adapted = norm * F.normalize(unit + residual, dim=1, eps=1e-12)
            logits = _logits(classify_fn, adapted, need_gradient=True)
            ce = F.cross_entropy(logits[selected], y[selected])
            structure = gram_loss(adapted, original)
            total = ce + structure
            if not bool(torch.isfinite(total)):
                raise ValueError('Adapter loss became nonfinite')
            total.backward()
            gradients = [p.grad for p in model.parameters() if p.grad is not None]
            if not gradients or any(not bool(torch.isfinite(g).all()) for g in gradients):
                raise ValueError('Adapter gradients must be present and finite')
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
            optimizer.step()
            if any(not bool(torch.isfinite(p).all()) for p in model.parameters()):
                raise ValueError('Adapter parameters became nonfinite')
            trace.append(dict(step=step, full_U_count=len(values), selected_count=selected_count,
                loss_ce=float(ce.detach()), loss_gram=float(structure.detach()), loss_total=float(total.detach()),
                grad_norm=float(grad_norm), max_pre_normalization_residual=float(residual.detach().norm(dim=1).max())))
        with torch.no_grad():
            norm, unit, residual = model._parts(x)
            final = norm * F.normalize(unit + residual, dim=1, eps=1e-12)
            max_feature_change = float((final - x).abs().max())
            final_residual = float(residual.norm(dim=1).max())
        changes = {key: (value.detach() - before[key]).abs() for key, value in model.state_dict().items()}
        summary.update(status='COMPLETE', steps=STEPS,
            optimizer=dict(name='AdamW', lr=.001, weight_decay=.0001, betas=[.9, .999], eps=1e-8, foreach=False, clip_norm=5.),
            training_adapter_forward_calls=STEPS, training_adapter_forward_rows=STEPS * len(values),
            diagnostic_adapter_forward_calls=1, diagnostic_adapter_forward_rows=len(values),
            adapter_forward_calls=STEPS + 1, adapter_forward_rows=(STEPS + 1) * len(values),
            classifier_forward_calls=STEPS, classifier_forward_rows=STEPS * len(values),
            parameters_changed=any(bool(d.any()) for d in changes.values()),
            parameter_changed_count=sum(int(torch.count_nonzero(d)) for d in changes.values()),
            max_abs_parameter_change=max(float(d.max()) for d in changes.values()),
            max_abs_feature_change=max_feature_change,
            max_pre_normalization_residual=max(final_residual, *(r['max_pre_normalization_residual'] for r in trace)),
            loss_initial={k: trace[0][k] for k in ('loss_total', 'loss_ce', 'loss_gram')},
            loss_final={k: trace[-1][k] for k in ('loss_total', 'loss_ce', 'loss_gram')})
    model.eval().requires_grad_(False)
    return model, trace, summary


def predict(model, raw_np, classify_fn, device='cuda:0'):
    """Independent rows with frozen adapter/head; no fitting or query statistics."""
    values = _raw_numpy(raw_np)
    model = model.to(device)
    with torch.no_grad():
        adapted = model(torch.tensor(values.tolist(), dtype=torch.float32, device=device))
        logits = _logits(classify_fn, adapted)
    return (np.asarray(adapted.cpu().tolist(), np.float32),
            np.asarray(logits.cpu().tolist(), np.float32))
