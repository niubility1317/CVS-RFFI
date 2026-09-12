"""Read-only comparisons on the same fixed batch and ordered parameter set."""
import torch


def _vector(values):
    if isinstance(values, torch.Tensor):
        return values.detach().reshape(-1).double()
    values = tuple(values)
    if not values or any(v is None for v in values):
        raise ValueError('provide aligned gradients, filling unused entries with shaped zeros')
    return torch.cat([v.detach().reshape(-1).double() for v in values])


def compare_gradients(left, right, eps=1e-12):
    a, b = _vector(left), _vector(right)
    if a.shape != b.shape:
        raise ValueError('gradient parameter spaces differ')
    na, nb = a.norm(), b.norm()
    valid = bool(torch.isfinite(a).all() and torch.isfinite(b).all() and na > eps and nb > eps)
    return dict(valid=valid, reason='' if valid else 'nonfinite_or_zero_gradient',
                left_norm=float(na), right_norm=float(nb), difference_norm=float((a - b).norm()),
                cosine=float(torch.dot(a, b) / (na * nb)) if valid else None,
                norm_ratio=float(nb / na) if valid else None)


def gradient_diagnostics(g_id, g_adv_online, g_adv_recovered, *, g_nonadv=None):
    """g_adv must already carry the encoder's adversarial sign and scale."""
    result = dict(id_online=compare_gradients(g_id, g_adv_online),
                  id_recovered=compare_gradients(g_id, g_adv_recovered),
                  online_recovered=compare_gradients(g_adv_online, g_adv_recovered))
    if g_nonadv is not None:
        result['nonadv_online'] = compare_gradients(g_nonadv, g_adv_online)
        result['nonadv_recovered'] = compare_gradients(g_nonadv, g_adv_recovered)
    result['valid'] = all(v['valid'] for v in result.values())
    return result
