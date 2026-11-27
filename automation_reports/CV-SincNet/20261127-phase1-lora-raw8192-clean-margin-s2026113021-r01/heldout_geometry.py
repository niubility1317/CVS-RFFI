import importlib
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch
from torch.nn import functional as F


def read(path):
    return json.loads(path.read_bytes())


def summarize(data):
    return dict(mean=float(np.mean(data)), p10_p50_p90=np.quantile(data, [.1, .5, .9]).tolist())


project = Path.cwd()
run = sys.argv[1]
release = project / 'releases' / f'{run}-release'
rt = read(release / 'runtime.json')
root = Path(rt['run_root'])
panel_root = root / 'full_native'
stamp = read(panel_root / 'source-predictions.json')
scores = read(panel_root / 'source-scores.json')
if stamp['status'] != 'PREDICTIONS_FROZEN' or stamp['source_truth_reads'] or not scores['source_truth_last']:
    raise ValueError('The registered same-packet source evaluation must already be frozen and scored')

reference = Path(rt['w24_release_root'])
os.chdir(reference)
sys.path.insert(0, str(reference))
package = importlib.import_module('experiments.pamf_lora_main')
package.__path__.append(str(release / 'experiments/pamf_lora_main'))
from experiments.pamf_lora_main import w24_joint_pamf as joint
from experiments.pamf_lora_main import w24_native_training as native
from experiments.pamf_lora_main.w24_clean_geometry import clean_prototypes


@torch.no_grad()
def evaluate():
    device = 'cuda:0'
    reader, models, frontend, contract = native.load_readers(joint, rt, device)
    parent_name = native.TASK_RUN
    parent_rt = read(project / 'releases' / f'{parent_name}-release' / 'runtime.json')
    _, parent, _, parent_contract = native.load_readers(joint, parent_rt, device)
    if parent_contract != contract:
        raise ValueError('Expected matching registered native parent')
    source = Path(rt['source_root'])
    rows, rx, ids = joint.panel(source, contract)
    if ids != stamp['physical_ids'] or len(rows) != 300:
        raise ValueError('Expected fixed registered physical source panel')
    ls = joint.native(source, 'L_s', contract['counts']['L_s'])
    prototypes = clean_prototypes(joint, reader, frontend, ls,
                                  joint.training_labels(source, contract), device)
    clean = joint.native(source, 'V', contract['counts']['V'])[rows]

    def batches(data):
        return (data[i:i + joint.PRED_BATCH] for i in range(0, len(data), joint.PRED_BATCH))

    def embed(data):
        logits, feature = reader(joint.received_aperture(data, device))
        return logits, F.normalize(feature, dim=-1)

    anchor = torch.cat([embed(batch)[1] for batch in batches(clean)])
    modes = {native.TASK: parent, **{mode: models for mode in native.GEOMETRY_ARMS}}
    frozen_features = {}
    for view in joint.VIEWS:
        observed = np.stack([joint.source_receive(view, (int(row), joint.SOURCE[int(group)], packet))
                             for row, group, packet in zip(rows, rx, clean, strict=True)])
        direct = [embed(batch) for batch in batches(observed)]
        frozen_features[view] = {'direct': tuple(torch.cat(group) for group in zip(*direct, strict=True))}
        for mode, model in modes.items():
            chunks = [embed(native.restored_raw(model, batch, mode, device)) for batch in batches(observed)]
            frozen_features[view][mode] = tuple(torch.cat(group) for group in zip(*chunks, strict=True))

    with np.load(source / 'V_metadata.npz', allow_pickle=False) as metadata:
        y = torch.tensor(metadata['y'][rows].tolist(), device=device)
    valid = F.one_hot(y, num_classes=len(prototypes)).bool()
    rival_idx = (anchor @ prototypes.T).masked_fill(valid, -torch.inf).argmax(dim=-1)
    rival = prototypes[rival_idx]
    row = torch.arange(len(y), device=device)
    base_sep = (anchor * (anchor - rival)).sum(dim=-1)
    results = {}
    with np.load(panel_root / 'predictions.npz', allow_pickle=False) as predicted:
        for view, mode_features in frozen_features.items():
            results[view] = {}
            for mode, (logits, sample) in mode_features.items():
                scores = sample @ prototypes.T
                proto_gap = scores[row, y] - scores.masked_fill(valid, -torch.inf).amax(dim=-1)
                head_gap = logits[row, y] - logits.masked_fill(valid, -torch.inf).amax(dim=-1)
                relative_shortfall = (base_sep - (sample * (anchor - rival)).sum(dim=-1)) / base_sep.clamp_min(1e-4)
                head = logits.argmax(dim=-1).tolist()
                matched = np.array_equal(head, predicted[view][stamp['modes'].index(mode)])
                results[view][mode] = dict(
                    clean_cosine=summarize((sample * anchor).sum(dim=-1).tolist()),
                    relative_shortfall=summarize(relative_shortfall.tolist()),
                    true_tx_prototype_margin=summarize(proto_gap.tolist()),
                    true_tx_head_margin=summarize(head_gap.tolist()),
                    prototype_correct=int((proto_gap > 0).sum()),
                    head_correct=int((logits.argmax(dim=-1) == y).sum()),
                    matched_registered_predictions=bool(matched),
                )
    return dict(n_packets=len(rows), prototype_source='L_s', prototype_count=len(ls),
                source_truth_after_feature_freeze=True, target_reads=0, views=results)


print(json.dumps(evaluate(), allow_nan=False))
