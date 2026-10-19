"""Replace only the CE of the 64 existing practical slots; no new forward."""
from contextlib import contextmanager
import torch
from torch.nn import functional as F

ARMS = {'CE_HALF': {'temperature': 1., 'weight': .5, 'smoothing': .01},
        'CE_T2': {'temperature': 2., 'weight': 1., 'smoothing': .01}}


def validate_objective(arm, config):
    if arm not in ARMS or config != ARMS[arm]:
        raise ValueError('Only fixed satellite CE_HALF or CE_T2 with smoothing .01; no T squared')
    return config


def validate_head(head):
    if (getattr(head, 'scale', None) != 30.
            or not torch.is_tensor(getattr(head, 'weight', None))
            or tuple(head.weight.shape) != (6, 160)):
        raise ValueError('Original default six-class cosine head/scale30 required')


def replacement(first, second, labels, arm, config):
    """Return .25*(new satellite CE64 mean - original satellite CE64 mean).

    The unchanged native objective gives the mixed RD group weight .5. Each
    mixed CE is averaged over 128 slots, of which exactly 64 are satellite.
    Hence the two-pass satellite CE64 mean has coefficient .5*(64/128)=.25.
    CE is evaluated separately for both dropout passes, never on mean logits.
    """
    validate_objective(arm, config)
    if (first.shape != (64, 6) or second.shape != (64, 6)
            or labels.shape != (64,) or labels.dtype != torch.long
            or not first.is_floating_point() or not second.is_floating_point()
            or first.device != second.device or labels.device != first.device
            or not bool(((labels >= 0) & (labels < 6)).all())
            or not bool(torch.isfinite(first).all())
            or not bool(torch.isfinite(second).all())):
        raise ValueError('Only two finite satellite64/default6 logits and visible TX labels')
    temperature, weight, smoothing = (config[k] for k in ('temperature', 'weight', 'smoothing'))
    old0 = F.cross_entropy(first, labels, label_smoothing=smoothing)
    old1 = F.cross_entropy(second, labels, label_smoothing=smoothing)
    new0 = weight * F.cross_entropy(first / temperature, labels, label_smoothing=smoothing)
    new1 = weight * F.cross_entropy(second / temperature, labels, label_smoothing=smoothing)
    old, new = .5 * (old0 + old1), .5 * (new0 + new1)
    difference = new - old
    correction = .25 * difference
    record = dict(arm=arm, temperature=temperature, weight=weight,
                  label_smoothing=smoothing, T_squared=False,
                  old_sat_CE0=float(old0.detach()), old_sat_CE1=float(old1.detach()),
                  old_sat_CE=float(old.detach()),
                  new_sat_CE0=float(new0.detach()), new_sat_CE1=float(new1.detach()),
                  new_sat_CE=float(new.detach()),
                  satellite_CE_delta_unscaled=float(difference.detach()),
                  satellite_CE_correction=float(correction.detach()),
                  mixed_group_weight=.5, satellite_fraction=.5,
                  satellite_objective_coefficient=.25, per_pass_coefficient=.125,
                  satellite_slots=64, mixed_slots=128, passes=2,
                  two_separate_CE_mean=True, all_satellite_CE_retained=True,
                  symmetric_KL_unchanged=True, clean_raw_ordinary_CE_unchanged=True,
                  deployment_head_scale=30., extra_NN=0, extra_parameters=0)
    return correction, record


@contextmanager
def captured_native_views(m):
    """Capture ordinary1/mixed0/mixed1/raw0/raw1 without altering the calls."""
    outputs, observations = [], []
    channel = m._full_channel
    present_forward, present_apply = 'forward_identity_only' in vars(m), 'apply' in vars(channel)
    shadow_forward, shadow_apply = vars(m).get('forward_identity_only'), vars(channel).get('apply')
    forward, apply = m.forward_identity_only, channel.apply

    def record_forward(*args, **kwargs):
        output = forward(*args, **kwargs)
        logits = output.get('tx_logits') if isinstance(output, dict) else None
        if logits is None or logits.shape != (128, 6) or not bool(torch.isfinite(logits).all()):
            raise ValueError('Exactly five existing finite128x6 default-head forwards required')
        outputs.append(logits)
        return output

    def record_channel(*args, **kwargs):
        output = apply(*args, **kwargs)
        if not isinstance(output, (tuple, list)) or len(output) != 2 or len(output[1]) != 64:
            raise ValueError('Exactly64 existing practical records required')
        observations.append(output[1])
        return output

    m.forward_identity_only, channel.apply = record_forward, record_channel
    try:
        yield outputs, observations
    finally:
        if present_forward:
            m.forward_identity_only = shadow_forward
        else:
            delattr(m, 'forward_identity_only')
        if present_apply:
            channel.apply = shadow_apply
        else:
            delattr(channel, 'apply')
