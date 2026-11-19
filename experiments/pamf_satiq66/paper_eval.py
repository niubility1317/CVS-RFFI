"""Fixed, score-blind reference panels for supervised SatIQ embeddings."""

import numpy as np
import torch
from scipy.stats import rankdata
from torch.nn import functional as F

from .train import device_array


def normalize_packet(iq, signed=False):
    low = iq.amin(dim=(1, 2), keepdim=True)
    scale = (iq.amax(dim=(1, 2), keepdim=True) - low).clamp_min(1e-8)
    bounded = (iq - low) / scale
    return 2 * bounded - 1 if signed else bounded


def model_input(model, iq):
    return normalize_packet(iq, signed=getattr(model, "signed_input", False))


@torch.no_grad()
def embeddings(model, raw, device, block=128):
    model.eval()
    features = []
    samples = getattr(model, "waveform_samples", raw.shape[-1])
    for start in range(0, len(raw), block):
        batch = model_input(model, device_array(raw[start:start + block, :, :samples], device))
        features.append(np.from_dlpack(model.embed(batch).cpu()))
    return np.concatenate(features)


def panel(labels, classes, seed, reference_size=16, reference_period=None):
    rng = np.random.default_rng(seed)
    eligible, ref, query, targets = [], [], [], []
    for label in range(len(classes)):
        rows = np.flatnonzero(labels == label)
        if len(rows) <= reference_size:
            continue
        rows = rng.permutation(rows)
        if reference_period is None:
            enrolment, evaluation = rows[:reference_size], rows[reference_size:]
        else:
            is_reference = reference_period[rows]
            enrolment, evaluation = rows[is_reference][:reference_size], rows[~is_reference]
        if len(enrolment) < reference_size or not len(evaluation):
            continue
        targets.extend([len(eligible)] * len(evaluation))
        eligible.append(label)
        ref.append(enrolment)
        query.extend(evaluation)
    if len(eligible) < 2:
        raise ValueError("Validation requires at least two identities with a disjoint reference and query")
    satellites = np.asarray(classes)[eligible, 0]
    targets = np.asarray(targets)
    neighbor = []
    foreign = []
    candidates = np.arange(len(ref))
    for label in targets:
        same = candidates[(satellites == satellites[label]) & (candidates != label)]
        neighbor.append(rng.choice(same) if len(same) else -1)
        other = candidates[satellites != satellites[label]]
        foreign.append(rng.choice(other) if len(other) else -1)
    return np.asarray(ref), np.asarray(query), targets, np.asarray(neighbor), np.asarray(foreign)


def binary_scores(genuine, impostor, threshold=None, include_eer_threshold=False):
    if not len(genuine) or not len(impostor):
        return None
    ranks = rankdata(np.concatenate((genuine, impostor)))
    auc = (ranks[:len(genuine)].sum() - len(genuine) * (len(genuine) + 1) / 2) / (len(genuine) * len(impostor))
    thresholds = np.sort(np.unique(np.r_[genuine, impostor, np.inf]))
    miss = np.searchsorted(np.sort(genuine), thresholds) / len(genuine)
    false = 1 - np.searchsorted(np.sort(impostor), thresholds) / len(impostor)
    i = np.argmin(np.abs(miss - false))
    result = dict(auc=float(auc), eer=float((miss[i] + false[i]) / 2), pairs=len(genuine))
    if include_eer_threshold:
        result["eer_threshold"] = float(thresholds[i])
    if threshold is not None:
        result["fixed"] = dict(threshold=float(threshold), far=float(np.mean(impostor >= threshold)),
                               frr=float(np.mean(genuine < threshold)))
    return result


@torch.no_grad()
def metrics(features, data_panel, device, thresholds=None, include_eer_threshold=False):
    ref, query, labels, same_sat, other_sat = data_panel
    vectors = F.normalize(device_array(features, device), dim=1)
    ref = device_array(ref.astype(np.int64, copy=False), device)
    q = vectors[device_array(query.astype(np.int64, copy=False), device)]
    results = {}
    for k in (1, ref.shape[1]):
        means = vectors[ref[:, :k]].mean(dim=1)
        scores = q @ means.T
        p = np.arange(len(labels))
        score = np.from_dlpack(scores.cpu())
        genuine = score[p, labels]
        cutoffs = thresholds[f"K{k}"] if thresholds is not None else {}
        results[f"K{k}"] = dict(top1=float((score.argmax(1) == labels).mean()),
                                same_satellite=binary_scores(genuine[same_sat >= 0],
                                                             score[p[same_sat >= 0], same_sat[same_sat >= 0]],
                                                             cutoffs.get("same_satellite"), include_eer_threshold),
                                different_satellite=binary_scores(genuine[other_sat >= 0],
                                                                  score[p[other_sat >= 0], other_sat[other_sat >= 0]],
                                                                  cutoffs.get("different_satellite"), include_eer_threshold))
    return dict(reference_classes=len(ref), query_records=len(query), **results)
