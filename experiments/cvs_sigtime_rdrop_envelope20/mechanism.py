"""Same-packet evidence synthesis; no cross-packet, RX, U or teacher labels."""
import math
import torch

ARMS = ('FULL', 'MEAN', 'CONFIDENCE', 'ENVELOPE')


def new_audit():
    return dict(calls=0, packets=0, mixed_envelope_packets=0,
                teacher_true_probability_violations=0,
                max_confidence_control_error=0., max_teacher_mass_error=0.,
                teacher_true_probability_sum=0., mean_teacher_true_probability_sum=0.)


def validate(first, second, y, arm):
    if (arm not in ARMS or y.ndim != 1 or y.dtype != torch.long or not len(y)
            or first.shape != (len(y), 6) or second.shape != first.shape
            or first.dtype not in (torch.float32, torch.float64)
            or second.dtype != first.dtype or first.device != second.device
            or y.device != first.device or not bool(((y >= 0) & (y < 6)).all())
            or not bool(torch.isfinite(first).all()) or not bool(torch.isfinite(second).all())):
        raise ValueError('Finite paired six-class logits and actual visible L labels required')


@torch.no_grad()
def teachers(first, second, y):
    # Relative logits are invariant to separate additive offsets in either view.
    first, second = first.detach(), second.detach()
    p, q = first.log_softmax(1), second.log_softmax(1)
    mean = torch.logaddexp(p, q) - math.log(2.)
    a = first - first.gather(1, y[:, None])
    b = second - second.gather(1, y[:, None])
    envelope = torch.minimum(a, b).log_softmax(1)
    other = torch.arange(6, device=y.device)[None, :] != y[:, None]
    # Exact same true-class mass as ENVELOPE. Only the conditional wrong-class
    # distribution differs: use the arithmetic-mean teacher's conditional shape.
    confidence = torch.empty_like(envelope)
    confidence.scatter_(1, y[:, None], envelope.gather(1, y[:, None]))
    log_other_mass = envelope.masked_select(other).reshape(len(y), 5).logsumexp(1)
    mean_other = mean.masked_select(other).reshape(len(y), 5).log_softmax(1)
    confidence[other] = (mean_other + log_other_mass[:, None]).flatten()
    # A mixed packet receives strict stronger evidence from each different view
    # against at least one competing identity. Ties do not count as mixing.
    mixed = ((a < b) & other).any(1) & ((b < a) & other).any(1)
    return dict(MEAN=mean, CONFIDENCE=confidence, ENVELOPE=envelope), mixed


def consistency(first, second, y, arm, audit):
    validate(first, second, y, arm)
    p, q = first.log_softmax(1), second.log_softmax(1)
    native = .5 * ((p.exp() - q.exp()) * (p - q)).sum(1).mean()
    targets, mixed = teachers(first, second, y)
    env, conf, avg = (targets[k] for k in ('ENVELOPE', 'CONFIDENCE', 'MEAN'))
    true = env.gather(1, y[:, None]).exp().flatten()
    strongest = torch.maximum(p.gather(1, y[:, None]).exp(), q.gather(1, y[:, None]).exp()).flatten()
    violations = int((true + 2e-6 < strongest).sum())
    confidence_error = float((env.gather(1, y[:, None]) - conf.gather(1, y[:, None])).abs().max())
    mass_error = max(float(t.exp().sum(1).sub(1).abs().max()) for t in targets.values())
    if violations or confidence_error > 1e-6 or mass_error > 2e-6 or any(t.requires_grad for t in targets.values()):
        raise ValueError('Detached normalized envelope and confidence-matched control required')
    if arm == 'FULL':
        loss = native
    else:
        target = targets[arm]
        loss = .5 * (target.exp() * (2 * target - p - q)).sum(1).mean()
    if not bool(torch.isfinite(loss)):
        raise ValueError('Non-finite consistency loss')
    audit['calls'] += 1
    audit['packets'] += len(y)
    audit['mixed_envelope_packets'] += int(mixed.sum())
    audit['teacher_true_probability_violations'] += violations
    audit['max_confidence_control_error'] = max(audit['max_confidence_control_error'], confidence_error)
    audit['max_teacher_mass_error'] = max(audit['max_teacher_mass_error'], mass_error)
    audit['teacher_true_probability_sum'] += float(true.sum())
    audit['mean_teacher_true_probability_sum'] += float(avg.gather(1, y[:, None]).exp().sum())
    return loss, dict(native_symmetric_KL=native.detach(), consistency_loss=loss.detach(),
                      envelope_true_probability=true.mean(), mean_teacher_true_probability=avg.gather(1, y[:, None]).exp().mean(),
                      mixed_envelope_fraction=mixed.double().mean(), confidence_control_error=confidence_error,
                      teacher_mass_error=mass_error, teacher_true_probability_violations=violations)
