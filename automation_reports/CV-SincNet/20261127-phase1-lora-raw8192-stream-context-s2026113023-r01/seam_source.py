"""Source interpretation of the readout with identical weights and severed chirp seams."""

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from functools import partial
import importlib
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch


project = Path.cwd()
run = sys.argv[1]
release = project / 'releases' / f'{run}-release'
rt = json.loads((release / 'runtime.json').read_bytes())
root = Path(rt['run_root'])
full = root / 'full_native'
stamp = json.loads((full / 'source-predictions.json').read_bytes())
source_scores = json.loads((full / 'source-scores.json').read_bytes())
assert stamp['status'] == 'PREDICTIONS_FROZEN' and not stamp['source_truth_reads']
assert not stamp['target_reads'] and source_scores['source_truth_last']
w24 = Path(rt['w24_release_root'])
os.chdir(w24)
sys.path.insert(0, str(w24))
package = importlib.import_module('experiments.pamf_lora_main')
package.__path__.append(str(release / 'experiments/pamf_lora_main'))
from experiments.pamf_lora_main import w24_joint_pamf as joint
from experiments.pamf_lora_main import w24_native_training as native


def relative_rms(continuous, bounded):
    return ((continuous - bounded).square().mean((1, 2, 3)) /
            continuous.square().mean((1, 2, 3)).clamp_min(1e-9)).sqrt().cpu().tolist()


@torch.no_grad()
def predict():
    device = 'cuda:0'
    reader, models, _, contract = native.load_readers(joint, rt, device)
    mode = native.STREAM_ARMS[-1]
    continuous = models[mode]
    bounded = deepcopy(continuous)
    bounded.core.stream.continuous = False
    source = Path(rt['source_root'])
    rows, rx, ids = joint.panel(source, contract)
    assert ids == stamp['physical_ids'] and len(rows) == 300 and len(set(ids)) == 300
    clean = joint.native(source, 'V', contract['counts']['V'])[rows]
    jobs = [(int(i), joint.SOURCE[int(g)], raw) for i, g, raw in zip(rows, rx, clean, strict=True)]
    predictions, observed_difference = {}, {}
    with ThreadPoolExecutor(max_workers=4) as pool:
        for view in joint.VIEWS:
            received = np.stack(list(pool.map(partial(joint.source_receive, view), jobs)))
            active_classes, bounded_classes, center, edges = [], [], [], []
            for start in range(0, len(rows), joint.PRED_BATCH):
                samples = received[start:start + joint.PRED_BATCH]
                apertures = [
                    joint.received_aperture(native.restored_raw({mode: restoring}, samples, mode, device), device)
                    for restoring in (continuous, bounded)
                ]
                active, cut = apertures
                active_classes.extend(joint.classify_aperture(reader, active))
                bounded_classes.extend(joint.classify_aperture(reader, cut))
                center.extend(relative_rms(active[..., :384], cut[..., :384]))
                edges.extend(relative_rms(active[..., 384:], cut[..., 384:]))
            predictions[view] = (np.asarray(active_classes), np.asarray(bounded_classes))
            observed_difference[view] = dict(center_relative_rms_p50=float(np.median(center)),
                                             edge_relative_rms_p50=float(np.median(edges)))
    return source, rows, rx, mode, predictions, observed_difference


source, rows, rx, mode, predictions, differences = predict()
with np.load(source / 'V_metadata.npz', allow_pickle=False) as metadata:
    truth = metadata['y'][rows]

views = {}
with np.load(full / 'predictions.npz', allow_pickle=False) as expected:
    for view, (continuous, bounded) in predictions.items():
        matched = np.array_equal(continuous, expected[view][stamp['modes'].index(mode)])
        correct, cut_correct = continuous == truth, bounded == truth
        repair = int((correct & ~cut_correct).sum())
        harm = int((cut_correct & ~correct).sum())
        views[view] = dict(registered_match=matched, continuous_correct=int(correct.sum()),
                           bounded_correct=int(cut_correct.sum()), paired=dict(repairs=repair, harms=harm, net=repair-harm),
                           net_by_receiver=[int(correct[rx==group].sum()-cut_correct[rx==group].sum())
                                            for group in np.unique(rx)], **differences[view])

print(json.dumps(dict(n_packets=len(rows), source_truth_after_prediction=True, target_reads=0, views=views),
                 allow_nan=False))
