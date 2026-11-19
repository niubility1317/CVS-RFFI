"""Fixed, score-blind reference panels for supervised SatIQ embeddings."""

import numpy as np
import torch
from scipy.stats import rankdata
from torch.nn import functional as F

from .train import device_array


def normalize_packet(iq):
    low = iq.amin(dim=(1, 2), keepdim=True)
    scale = (iq.amax(dim=(1, 2), keepdim=True) - low).clamp_min(1e-8)
    return (iq - low) / scale


@torch.no_grad()
def embeddings(model, raw, device, block=128):
    model.eval()
    features = []
    for start in range(0, len(raw), block):
        batch = normalize_packet(device_array(raw[start:start + block], device))
        features.append(np.from_dlpack(model.embed(batch).cpu()))
    return np.concatenate(features)


def panel(labels, classes, seed, reference_size=16):
    rng = np.random.default_rng(seed)
    eligible = [k for k in range(len(classes)) if np.count_nonzero(labels == k) > reference_size]
    if len(eligible) < 2:
        raise ValueError("Validation requires at least two identities with a disjoint reference and query")
    satellites = np.asarray(classes)[eligible, 0]
    ref, query, targets = [], [], []
    for col, label in enumerate(eligible):
        rows = rng.permutation(np.flatnonzero(labels == label))
        ref.append(rows[:reference_size])
        query.extend(rows[reference_size:])
        targets.extend([col] * (len(rows) - reference_size))
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


def binary_scores(genuine, impostor):
    if not len(genuine) or not len(impostor):
        return None
    ranks = rankdata(np.concatenate((genuine, impostor)))
    auc = (ranks[:len(genuine)].sum() - len(genuine) * (len(genuine) + 1) / 2) / (len(genuine) * len(impostor))
    thresholds = np.sort(np.unique(np.r_[genuine, impostor, np.inf]))
    miss = np.searchsorted(np.sort(genuine), thresholds) / len(genuine)
    false = 1 - np.searchsorted(np.sort(impostor), thresholds) / len(impostor)
    i = np.argmin(np.abs(miss - false))
    return dict(auc=float(auc), eer=float((miss[i] + false[i]) / 2), pairs=len(genuine))


@torch.no_grad()
def metrics(features, data_panel, device):
    ref, query, labels, same_sat, other_sat = data_panel
    vectors = F.normalize(device_array(features, device), dim=1)
    ref = torch.as_tensor(ref, device=device)
    q = vectors[torch.as_tensor(query, device=device)]
    results = {}
    for k in (1, ref.shape[1]):
        means = vectors[ref[:, :k]].mean(dim=1)
        scores = q @ means.T
        p = np.arange(len(labels))
        score = np.from_dlpack(scores.cpu())
        genuine = score[p, labels]
        results[f"K{k}"] = dict(top1=float((score.argmax(1) == labels).mean()),
                                same_satellite=binary_scores(genuine[same_sat >= 0],
                                                             score[p[same_sat >= 0], same_sat[same_sat >= 0]]),
                                different_satellite=binary_scores(genuine[other_sat >= 0],
                                                                  score[p[other_sat >= 0], other_sat[other_sat >= 0]]))
    return dict(reference_classes=len(ref), query_records=len(query), **results)
