"""The historical one-sided relation penalty; no fitted head or new prediction."""
from contextlib import contextmanager
import math
import torch

URBAN = ('practical_mid_urban', 'practical_low_urban')

def penalty(clean, satellite, labels, records):
    if (clean.shape != (64, 6) or satellite.shape != (64, 6)
            or labels.shape != (64,) or labels.dtype != torch.long
            or clean.device != satellite.device or labels.device != clean.device
            or not bool(((labels >= 0) & (labels < 6)).all())
            or not bool(torch.isfinite(clean).all()) or not bool(torch.isfinite(satellite).all())
            or len(records) != 64):
        raise ValueError('Only finite paired64/default6 logits and visible labels')
    weights, urban = [], []
    for r in records:
        if 'signal_to_added_noise_db' not in r or r.get('scene') not in ('practical_low_suburban',) + URBAN:
            raise ValueError('Actual full-channel added-noise metadata required; no quality proxy fallback')
        db, is_urban = r['signal_to_added_noise_db'], r['scene'] in URBAN
        if db is None:
            value = 0.
        else:
            db = float(db)
            if not math.isfinite(db):
                raise ValueError('Finite added-noise SNR or explicit None required')
            value = 1. / (1. + math.exp(-max(-60., min(60., (db - 5.) / 5.))))
        weights.append(value if is_urban else 0.)
        urban.append(is_urban)
    # Raw teacher relation is stop-gradient, including its correctness predicate.
    reference, student = clean.detach() / 30., satellite / 30.
    y = labels[:, None]
    mc = reference.gather(1, y) - reference
    ms = student.gather(1, y) - student
    rivals = torch.ones_like(mc, dtype=torch.bool).scatter_(1, y, False)
    mc, ms = mc[rivals].reshape(64, 5), ms[rivals].reshape(64, 5)
    q = torch.tensor(weights, device=student.device, dtype=student.dtype)
    correct = reference.argmax(1).eq(labels)
    urban_t = torch.tensor(urban, device=student.device, dtype=torch.bool)
    hinge = torch.relu(.5 * mc.clamp(0., 1.) - ms)
    raw = (q * correct.to(q.dtype) * hinge.mean(1)).mean()
    weighted = .5 * raw
    info = dict(pair_raw=float(raw.detach()), pair_weighted=float(weighted.detach()),
                pair_lambda=.5, rho=.5, m_max=1., head_scale=30., pair_count=64,
                q_sum=float(q.sum()), q_mean=float(q.mean()), q_nonzero=int((q > 0).sum()),
                urban_count=sum(urban), clean_correct_count=int(correct.sum()),
                clean_correct_urban_count=int((correct & urban_t).sum()),
                active_competitor_count=int(((hinge > 0) & (q[:, None] > 0) & correct[:, None]).sum()),
                competitor_count=320, clean_margin_mean=float(mc.mean()),
                satellite_margin_mean=float(ms.detach().mean()),
                clean_reference_stopgrad=True, confidence_threshold=False,
                all_satellite_CE_retained=True, two_pass_mean=True, extra_NN=0)
    return weighted, info

@contextmanager
def captured_native_views(m):
    """Capture five existing hook forwards, preserving instance shadow semantics."""
    outputs, observations = [], []
    channel = m._full_channel
    present_forward, present_apply = 'forward_identity_only' in vars(m), 'apply' in vars(channel)
    shadow_forward, shadow_apply = vars(m).get('forward_identity_only'), vars(channel).get('apply')
    forward, apply = m.forward_identity_only, channel.apply
    def record_forward(*a, **kw):
        out = forward(*a, **kw)
        z = out.get('tx_logits') if isinstance(out, dict) else None
        if z is None or z.shape != (128, 6) or not bool(torch.isfinite(z).all()):
            raise ValueError('Exactly five existing finite128x6 default-head forwards required')
        outputs.append(z)
        return out
    def record_channel(*a, **kw):
        out = apply(*a, **kw)
        if len(out) != 2 or len(out[1]) != 64:
            raise ValueError('Exactly64 existing practical records required')
        observations.append(out[1])
        return out
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
