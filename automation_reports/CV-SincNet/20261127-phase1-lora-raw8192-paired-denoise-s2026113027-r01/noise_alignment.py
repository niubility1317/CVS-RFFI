"""Observe stage-wise noise alignment after freezing; exact source traces remain in diagnostics."""

from concurrent.futures import ThreadPoolExecutor
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
full = Path(rt['run_root']) / 'full_native'
stamp = json.loads((full / 'source-predictions.json').read_bytes())
source_score = json.loads((full / 'source-scores.json').read_bytes())
assert stamp['status'] == 'PREDICTIONS_FROZEN' and not stamp['source_truth_reads'] and not stamp['target_reads']
assert source_score['source_truth_last'] and not source_score['target_reads']
w24 = Path(rt['w24_release_root'])
os.chdir(w24)
sys.path.insert(0, str(w24))
package = importlib.import_module('experiments.pamf_lora_main')
package.__path__.append(str(release / 'experiments/pamf_lora_main'))
from experiments.pamf_lora_main import w24_joint_pamf as joint, w24_native_training as native
from experiments.pamf_lora_main.w24_aperture_noise_audit import native_aperture
from experiments.pamf_lora_main.w24_edit_budget_audit import full_components


def edit_alignment(received, signal, edited):
    projected = [native_aperture(x.astype(np.complex128))[..., :384] for x in (received,signal,edited)]
    y, s, r = (x.reshape(len(x),-1) for x in projected)
    edit, exact = r-y, s-y
    edit_norm = np.linalg.norm(edit,axis=-1)
    exact_norm = np.linalg.norm(exact,axis=-1)
    observed_norm = np.linalg.norm(y,axis=-1)
    product = np.sum((exact.conj()*edit).real,axis=-1)
    cosine = product / np.maximum(exact_norm*edit_norm,1e-12)
    component = product / np.maximum(exact_norm,1e-12)
    orth = np.sqrt(np.maximum(edit_norm**2-component**2,0.))
    residual = np.linalg.norm(exact-edit,axis=-1)
    return np.stack((edit_norm/np.maximum(observed_norm,1e-12), cosine,
                     component/np.maximum(exact_norm,1e-12), orth/np.maximum(exact_norm,1e-12),
                     residual/np.maximum(exact_norm,1e-12)),axis=-1)


@torch.no_grad()
def predictions():
    device = 'cuda:0'
    f, models, _, contract = native.load_readers(joint, rt, device)
    source = Path(rt['source_root'])
    rows, rx, ids = joint.panel(source, contract)
    assert ids == stamp['physical_ids'] and len(rows) == 300
    clean = joint.native(source,'V',contract['counts']['V'])[rows]
    jobs = [(int(i),joint.SOURCE[int(r)],x) for i,r,x in zip(rows,rx,clean,strict=True)]
    output, analysis = {}, {}
    with ThreadPoolExecutor(max_workers=4) as pool:
        for view in joint.VIEWS:
            received, signal = (np.stack(a) for a in zip(*pool.map(partial(full_components,view),jobs),strict=True))
            guesses = {arm:[[],[]] for arm in models}
            metrics = {arm:[[],[]] for arm in models}
            for start in range(0,len(rows),joint.PRED_BATCH):
                a = received[start:start+joint.PRED_BATCH]
                s = signal[start:start+joint.PRED_BATCH]
                iq = joint.tensor(a.view(np.float32),device).view(torch.complex64)
                for arm, model in models.items():
                    outputs = model(iq,return_intermediate=True)
                    for level, edited in enumerate(outputs):
                        wave = np.asarray(edited.cpu().view(torch.float32).tolist(),dtype=np.float32).view(np.complex64)
                        guesses[arm][level].extend(joint.classify_aperture(f,joint.received_aperture(wave,device)))
                        if view != 'clean':
                            metrics[arm][level].append(edit_alignment(a,s,wave))
            output[view] = {arm:np.asarray(groups) for arm,groups in guesses.items()}
            analysis[view] = {} if view=='clean' else {
                arm:np.quantile(np.stack([np.concatenate(v,axis=0) for v in values]),[.1,.5,.9],axis=1).tolist()
                for arm,values in metrics.items()}
    return source, rows, output, analysis


source, rows, blind, analysis = predictions()
with np.load(source/'V_metadata.npz',allow_pickle=False) as metadata:
    truth=metadata['y'][rows]

result = {}
with np.load(full/'predictions.npz',allow_pickle=False) as registered:
    for view,matrix in blind.items():
        result[view] = {}
        for arm,x in matrix.items():
            final_parity = bool(np.array_equal(x[-1],registered[view][stamp['modes'].index(arm)]))
            if not final_parity:
                raise ValueError('Frozen final source decisions differ from registered source replay')
            hi = x==truth
            changes = hi[-1].astype(np.int16)-hi[0]
            result[view][arm] = dict(correct_by_stage=hi.sum(-1).tolist(), final_parity=final_parity,
                repairs=int((changes>0).sum()), harms=int((changes<0).sum()),
                levels=None if view=='clean' else analysis[view][arm])

print(json.dumps(dict(n_packets=len(rows),target_reads=0,metrics=[
    'edit_relative_rms','direction_cosine','parallel_relative_to_noise','orthogonal_relative_to_noise','residual_relative_to_noise'
], stats_axis=['quantile','stage','metric'],views=result),allow_nan=False))
