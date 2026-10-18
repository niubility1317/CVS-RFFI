"""One extra default-head U KL with matched total confidence-quality mass.

Metadata must be in exactly the same sample order as both logits tensors. This
function preserves that order; the view producer/caller owns identity matching.
Observation weights describe added-noise quality, not recoverability truth or
total SNR. They affect only this extra loss, never labeled CE or native U masks.
"""
from collections import defaultdict
from collections.abc import Mapping
import math
from numbers import Real

import torch
from torch.nn import functional as F


ARMS = ('MASS_UNIFORM', 'OBSERVATION_WEIGHTED')
TRAIN_SCENES = ('practical_low_suburban', 'practical_mid_urban', 'practical_low_urban')
STATES = ('LOS', 'PARTIAL', 'BLOCKED')
LOCKS = ('locked', 'degraded', 'unlocked')


def weight(arm, local_epoch, policy):
    """Fixed new E1..10 ramp, corresponding to student total E21..30."""
    if (arm not in ARMS or isinstance(local_epoch, bool)
            or not isinstance(local_epoch, int) or local_epoch not in range(1, 11)):
        raise ValueError('Only registered quality arm and new local E1..10')
    if (not isinstance(policy, Mapping)
            or (policy.get('tau'), policy.get('max_weight'), policy.get('ramp_epochs')) != (.95, .16, 5)):
        raise ValueError('Fixed confidence mass and five-local-epoch ramp required')
    return .16 * min(1., local_epoch / 5.)


def _mass(values):
    return float(values.detach().sum(dtype=torch.float64))


def _statistics(values):
    values = values.detach().double()
    return dict(
        count=len(values), mass=_mass(values), mean=float(values.mean()),
        positive=int((values > 0).sum()),
        quantiles=[float(v) for v in torch.quantile(values, values.new_tensor([0., .1, .5, .9, 1.]))],
        histogram=[int((values == 0).sum()), int(((values > 0) & (values <= .25)).sum()),
                   int(((values > .25) & (values <= .5)).sum()),
                   int(((values > .5) & (values <= .75)).sum()), int((values > .75).sum())])


def _observations(metadata, n, template):
    if not isinstance(metadata, (tuple, list)) or len(metadata) != n:
        raise ValueError('One current practical metadata record per input sample, in input order, required')
    required = ('scene', 'state_end', 'receiver_lock_status', 'signal_to_added_noise_db')
    snrs, missing, strata = [], [], defaultdict(list)
    for i, record in enumerate(metadata):
        if not isinstance(record, Mapping) or any(k not in record for k in required):
            raise ValueError('Current practical scene/state/lock/added-noise metadata required')
        scene, state, lock = (record[k] for k in required[:3])
        if scene not in TRAIN_SCENES or state not in STATES or lock not in LOCKS:
            raise ValueError('Original three training scenes and known practical state/lock roles required')
        value = record['signal_to_added_noise_db']
        none = value is None
        if not none and (isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value)):
            raise ValueError('Added-noise SNR must be a finite real number or explicit None')
        # None carries no physical availability evidence for this extra KL.
        # Keep its slot, native U supervision and all labeled CE unchanged.
        snrs.append(0. if none else float(value))
        missing.append(none)
        strata[(scene, state, lock)].append(i)
    # Compute this bounded mapping in double before the model dtype conversion;
    # finite but extreme physical values must not overflow an intermediate cast.
    noise = torch.tensor(snrs, dtype=torch.float64, device=template.device)
    none_mask = torch.tensor(missing, dtype=torch.bool, device=template.device)
    q_n = torch.sigmoid((noise - 5.) / 5.).masked_fill(none_mask, 0.).to(template.dtype)
    return q_n, snrs, missing, strata


def _stratified(metadata, snrs, missing, strata, q_t, q_n, q_uniform, q_observed, selected):
    result = []
    for (scene, state, lock), indices in sorted(strata.items()):
        index = torch.tensor(indices, device=q_t.device)
        n = len(indices)
        known = [snrs[i] for i in indices if not missing[i]]
        # Finite input values can still overflow a naive sum; incremental mean
        # retains a bounded descriptive average without changing any weights.
        snr_mean = None
        if known:
            snr_mean = sum(v / len(known) for v in known)
        mass = _mass(selected[index])
        result.append(dict(
            scene=scene, state=state, lock=lock, count=n,
            none_snr_count=sum(missing[i] for i in indices),
            q_teacher_mass=_mass(q_t[index]), q_observation_mass=_mass(q_n[index]),
            mass_uniform_mass=_mass(q_uniform[index]), observation_weighted_mass=_mass(q_observed[index]),
            q_mass=mass, q_mean=mass / n, q_teacher_mean=_mass(q_t[index]) / n,
            q_observation_mean=_mass(q_n[index]) / n, positive_q=int((selected[index] > 0).sum()),
            signal_to_added_noise_db_mean=snr_mean))
    return result


def teaching_loss(student, teacher, hidden_y, metadata, arm, local_epoch, policy):
    """Teacher T=1 stop-gradient KL, normalized over the whole original subset.

    Both arms calculate the same qT and qN from one frozen raw-U teacher and
    the existing full practical view. Uniform redistributes the candidate's
    total mass proportionally to qT, rather than reducing the candidate alone.
    There is no urban-student confidence selection or selected-count divisor.
    """
    coefficient = weight(arm, local_epoch, policy)
    if (not isinstance(hidden_y, torch.Tensor) or hidden_y.dtype != torch.long
            or hidden_y.ndim != 1 or not bool((hidden_y == -1).all())):
        raise ValueError('U TX must remain hidden long -1 sentinels')
    n = len(hidden_y)
    if (not isinstance(student, torch.Tensor) or n == 0 or student.shape != (n, 6)
            or not student.is_floating_point() or not bool(torch.isfinite(student).all())):
        raise ValueError('Finite original six-class student logits required')
    if (not isinstance(teacher, torch.Tensor) or teacher.shape != student.shape
            or not teacher.is_floating_point() or teacher.device != student.device
            or teacher.dtype != student.dtype or not bool(torch.isfinite(teacher).all())):
        raise ValueError('One matched finite frozen As20 default-head teacher tensor required')
    p = teacher.detach().softmax(1)
    q_t = ((p.max(1).values - .95) / .05).clamp(0., 1.)
    q_n, snrs, missing, strata = _observations(metadata, n, q_t)
    q_observed = q_t * q_n
    teacher_mass = _mass(q_t)
    observed_mass = _mass(q_observed)
    multiplier = observed_mass / teacher_mass if teacher_mass > 0. else 0.
    q_uniform = q_t * multiplier
    uniform_mass = _mass(q_uniform)
    # Roundoff only: both formulas must represent the identical total mass.
    if not math.isclose(observed_mass, uniform_mass, rel_tol=1e-6, abs_tol=1e-7):
        raise ValueError('Matched observation-weighted and uniform confidence mass required')
    q = q_uniform if arm == 'MASS_UNIFORM' else q_observed
    per_sample = F.kl_div(F.log_softmax(student, dim=1), p, reduction='none').sum(1)
    raw = (q * per_sample).mean()
    loss = coefficient * raw
    details = _statistics(q)
    record = dict(
        arm=arm, selected_teacher='FROZEN_AS20_EMA', weight=coefficient,
        KL=float(raw.detach()), weighted_KL=float(loss.detach()),
        unweighted_KL=float(per_sample.detach().mean()),
        q_mass=details['mass'], q_mean=details['mean'], positive_q=details['positive'],
        none_snr_count=sum(missing), q_teacher=_statistics(q_t),
        q_observation=_statistics(q_n), effective_q=details,
        mass_match=dict(q_teacher_mass=teacher_mass, observation_weighted_mass=observed_mass,
                        mass_uniform_mass=uniform_mass, difference=observed_mass - uniform_mass,
                        absolute_difference=abs(observed_mass - uniform_mass), uniform_multiplier=multiplier),
        strata=_stratified(metadata, snrs, missing, strata, q_t, q_n, q_uniform, q_observed, q),
        U_TX_truth_read=False, urban_confidence_filter=False, old_U_masks_unchanged=True,
        temperature=1., selected_count_normalization=False,
        observation_proxy_is_recoverability_truth=False)
    return loss, record
