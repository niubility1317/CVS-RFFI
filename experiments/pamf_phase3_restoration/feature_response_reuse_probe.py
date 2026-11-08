"""Source-only CPU probe of frozen-feature cache and shared-autograd equivalence."""
import argparse
import copy
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F

from .cache import check_F0_state, load_F0, tensor
from .covariance_detail import detail_objective
from .feature_response_loss import feature_response_loss
from .main_split_eval import valid_identity
from .objective import PairedObjective
from .response_joint_prior import response_loss, selected_perturbation
from .source_inverse import R07
from .source_pair import read, validate_source
from .variance_trial import load_scale

RUN = '20261108-diagnostic-Restore-FeatureReuse-sourceLU-s2026092701-r01'


def frozen_eval(reader):
    if any(m.training for m in reader.modules()) or any(p.requires_grad for p in reader.parameters()):
        raise ValueError('Original frozen evaluation reader required')


def cache_source_features(reader, clean, indices):
    frozen_eval(reader)
    result = torch.empty((len(clean), 4, 160), device=clean.device, dtype=clean.dtype)
    take = torch.arange(len(clean), device=clean.device)
    with torch.no_grad():
        for epoch in range(4):
            delta, choice = selected_perturbation(clean, indices, epoch)
            result[take, choice] = reader.features(clean + delta)
    return result


def compare_one_epoch(reader, clean, received, indices, scale, target, cache, epoch):
    frozen_eval(reader)
    delta, choice = selected_perturbation(clean, indices, epoch)
    with torch.no_grad():
        online_target = reader.features(clean + delta)
        cache_target = cache[torch.arange(len(clean), device=clean.device), choice]
        torch.testing.assert_close(online_target, cache_target, rtol=1e-5, atol=1e-6)
    torch.manual_seed(193)
    original_model = torch.nn.Conv1d(2, 2, 1)
    shared_model = copy.deepcopy(original_model)
    outcome = []
    for model, shared in ((original_model, False), (shared_model, True)):
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=1e-4)
        base = model(received); changed = model(received + .01*delta)
        base.retain_grad(); changed.retain_grad()
        if shared:
            fbase = reader.features(base)
            paired, _ = PairedObjective().to(clean.device)(base, clean)
            cosine = 1-F.cosine_similarity(fbase, target.detach(), dim=1, eps=1e-8)
            standardized = ((fbase-target.detach()).square()/scale.detach()).mean(1)
            detail = paired + .01*(cosine+standardized)
        else:
            detail, _ = detail_objective(reader, base, clean, target, scale, PairedObjective().to(clean.device))
            fbase = reader.features(base)
        fchanged = reader.features(changed)
        source_changed = cache_target if shared else online_target
        feature = feature_response_loss(fbase, fchanged, target, source_changed, scale)
        loss = (detail + .01*response_loss(base, changed, delta) + .01*feature).mean()
        loss.backward()
        gradients = [p.grad.detach().clone() for p in model.parameters()]
        input_gradients = [base.grad.detach().clone(), changed.grad.detach().clone()]
        optimizer.step()
        outcome.append((loss.detach(), gradients, input_gradients,
                        [p.detach().clone() for p in model.parameters()]))
    a, b = outcome
    for left, right in zip([a[0], *a[1], *a[2], *a[3]], [b[0], *b[1], *b[2], *b[3]]):
        torch.testing.assert_close(left, right, rtol=1e-5, atol=1e-6)
    return dict(epoch=epoch, feature_max_abs=float((online_target-cache_target).abs().max()),
        loss_max_abs=float((a[0]-b[0]).abs()),
        parameter_gradient_max_abs=max(float((x-y).abs().max()) for x,y in zip(a[1], b[1])),
        input_gradient_max_abs=max(float((x-y).abs().max()) for x,y in zip(a[2], b[2])),
        updated_parameter_max_abs=max(float((x-y).abs().max()) for x,y in zip(a[3], b[3])))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--commit', required=True)
    args = parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('CPU-only probe required')
    if (Path.cwd()/'release_commit.txt').read_text().strip() != args.commit:
        raise ValueError('Actual release code commit required')
    torch.set_num_threads(2)
    os.nice(10)
    start = time.perf_counter()
    cfg = validate_source(read(args.inputs))
    reader, identity = load_F0(cfg, 'cpu'); valid_identity(identity); frozen_eval(reader)
    snapshot = {k:v.detach().clone() for k,v in reader.state_dict().items()}
    cache_root = args.root/'runs'/R07/'cache'
    done = read(cache_root/'complete.json')
    if (done['status'], done['train_count'], done['TX_labels_read'], done['target_reads']) != (
            'ACTION_SOURCE_CACHE_COMPLETE', 63000, 0, 0):
        raise ValueError('Original source training cache required')
    indices_np = [0, 6300]
    roles = np.load(cache_root/'train/roles.npy', allow_pickle=False)
    if [str(roles[i]) for i in indices_np] != ['L_s', 'U_s']:
        raise ValueError('Exactly one source L_s and one source U_s required')
    def read_rows(path):
        source = np.load(path, mmap_mode='r', allow_pickle=False)
        try: return tensor(source[indices_np], 'cpu')
        finally: source._mmap.close()
    clean = read_rows(cache_root/'train/clean.npy')
    received = read_rows(cache_root/'epochs/01/received.npy')
    indices = torch.tensor(indices_np, dtype=torch.long)
    scale = load_scale(args.root)
    with torch.no_grad(): target = reader.features(clean).detach()
    cached = cache_source_features(reader, clean, indices)
    results = [compare_one_epoch(reader, clean, received, indices, scale, target, cached, e)
               for e in range(1, 5)]
    check_F0_state(reader, snapshot)
    out = args.root/'runs'/RUN; out.mkdir()
    result = dict(status='PASS', code_commit=args.commit, F0_identity=identity,
        source_rows=2, source_roles=['L_s','U_s'], directions=4, epoch_cases=4,
        toy_model='CPU Conv1d for computation-graph equivalence, not production G',
        toy_AdamW_comparison_updates=8, actual_G_loads=0, actual_G_updates=0,
        TX_labels_read=0, V_reads=0, OE_reads=0, target_inputs=0,
        CPU_threads=2, cache_batch=2, training_equivalence_tolerance='rtol1e-5/atol1e-6',
        production_GPU_FP32_equivalence_verified=False, training_speed_measured=False,
        cases=results, seconds=time.perf_counter()-start)
    with (out/'results.json').open('x',encoding='utf-8') as f:json.dump(result,f,indent=2,allow_nan=False)
    with (out/'complete.json').open('x',encoding='utf-8') as f:json.dump({'status':'ARTIFACTS_COMPLETE'},f)
    print(json.dumps(result,allow_nan=False),flush=True)


if __name__ == '__main__':
    main()
