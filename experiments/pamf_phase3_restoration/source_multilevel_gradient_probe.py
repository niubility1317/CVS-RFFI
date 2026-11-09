"""Source-only F0 multilevel input-gradient diagnostic; no model updates."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F

RUN = '20261109-diagnostic-Restore-MultilevelFGradient-sourceLU-s2026092701-r01'
MODULE = 'experiments.pamf_phase3_restoration.source_multilevel_gradient_probe'
LAYERS = {'shallow': 'core.id_backbone.time_down', 'middle': 'core.id_backbone.t2.pool'}


def captured_features(reader, x):
    """Capture exactly one call per fixed actual layer and preserve input gradients."""
    seen = {key: [] for key in LAYERS}
    handles = []
    try:
        for key, path in LAYERS.items():
            module = reader.get_submodule(path)
            def capture(_module, _inputs, output, key=key):
                seen[key].append(output)
            handles.append(module.register_forward_hook(capture))
        final = reader.features(x)
    finally:
        for handle in handles:
            handle.remove()
    if any(len(values) != 1 for values in seen.values()):
        raise ValueError('Each fixed F0 layer must execute exactly once')
    result = {key: values[0] for key, values in seen.items()}
    result['final'] = final
    if any(not torch.is_tensor(value) or value.ndim < 2 or value.shape[0] != len(x)
           or not bool(torch.isfinite(value).all()) for value in result.values()):
        raise ValueError('Finite actual F0 feature maps required')
    return result


def gradient_pair(a, b):
    a, b = a.detach().flatten().double(), b.detach().flatten().double()
    if a.shape != b.shape or not torch.isfinite(a).all() or not torch.isfinite(b).all():
        raise ValueError('Aligned finite gradients required')
    na, nb = a.norm(), b.norm()
    cosine = float((a @ b) / (na * nb)) if na > 0 and nb > 0 else None
    return dict(first_norm=float(na), second_norm=float(nb),
                ratio=float(nb / na) if na > 0 else None, cosine=cosine,
                first_zero=bool(na == 0), second_zero=bool(nb == 0))


def normalized_map_error(actual, target):
    if actual.shape != target.shape:
        raise ValueError('Matched source feature maps required')
    target = target.detach()
    dims = tuple(range(1, target.ndim))
    energy = target.square().mean(dims).clamp_min(1e-8)
    return (actual-target).square().mean(dims) / energy


def one_record(reader, model, objective, scale, clean, received):
    with torch.no_grad():
        reference = captured_features(reader, clean)
        signal, auxiliary = model(received, return_aux=True)
        residual = float(auxiliary['relative_solve_residual'].max())
        if not torch.isfinite(signal).all() or not np.isfinite(residual) or residual > 1e-8:
            raise ValueError('Invalid source recovery or physical posterior')
    signal = signal.detach().requires_grad_(True)
    actual = captured_features(reader, signal)
    if actual['final'].shape != (len(signal), 160) or scale.shape != (160,):
        raise ValueError('Original fused160 F0 and source variance required')
    iq, iq_metrics = objective(signal, clean)
    final_cos = 1-F.cosine_similarity(actual['final'], reference['final'].detach(), dim=1, eps=1e-8)
    final_scaled = ((actual['final']-reference['final'].detach()).square()/scale).mean(1)
    losses = {'waveform': iq.mean(), 'final': (.01*(final_cos+final_scaled)).mean(),
              'shallow': normalized_map_error(actual['shallow'], reference['shallow']).mean(),
              'middle': normalized_map_error(actual['middle'], reference['middle']).mean()}
    if any(not bool(torch.isfinite(value)) for value in losses.values()):
        raise ValueError('Finite source-only losses required')
    gradients = {}
    for n, (name, loss) in enumerate(losses.items()):
        gradients[name] = torch.autograd.grad(loss, signal, retain_graph=n<len(losses)-1)[0].detach()
    pairs = [('waveform','final'), ('waveform','shallow'), ('waveform','middle'),
             ('final','shallow'), ('final','middle'), ('shallow','middle')]
    return dict(losses={k:float(v.detach()) for k,v in losses.items()},
        iq_nmse=float(iq_metrics['iq_nmse'].detach().mean()),
        shapes={k:list(v.shape[1:]) for k,v in actual.items()},
        input_gradient_pairs={a+'__'+b:gradient_pair(gradients[a],gradients[b]) for a,b in pairs},
        posterior_residual=residual)


def worker(args):
    from . import physical_feature_gradient_probe as original
    from . import multiview_trial as trial
    from .cache import check_F0_state, load_F0, tensor
    from .cost_probe import write
    from .objective import PairedObjective
    from .source_pair import read, validate_source
    from .variance_trial import load_scale

    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = 'cuda:0'
    started = time.perf_counter()
    config = validate_source(read(args.inputs))
    source, roles = trial.source(args)
    reader, identity = load_F0(config, device)
    trial.main_eval.valid_identity(identity)
    reader.eval().requires_grad_(False)
    before_f = {k:v.detach().clone() for k,v in reader.state_dict().items()}
    models, metadata = original.load_models(args.root, device)
    model = models['fixed_source_E30']
    del models
    model.eval().requires_grad_(False)
    before_g = {k:v.detach().clone() for k,v in model.state_dict().items()}
    if metadata['F0_identity'] != identity:
        raise ValueError('Original frozen F0 must match parent recovery model')
    scale = load_scale(args.root).to(device)
    objective = PairedObjective().to(device)
    folder = source/'epochs'/'01'
    done = read(folder/'complete.json')
    if (done['count'], done['TX_labels_read'], done['target_reads']) != (63000, 0, 0):
        raise ValueError('Original completed source draw required')
    with np.load(folder/'groups.npz', allow_pickle=False) as groups:
        selected = original.select_rows(roles, groups['scene'])
    arrays = {name:np.load(path, mmap_mode='r', allow_pickle=False) for name,path in {
        'clean':source/'train/clean.npy', 'received':folder/'received.npy'}.items()}
    out = args.root/'runs'/RUN
    rows = []
    try:
        for index, role, scene in selected:
            clean = tensor(np.asarray(arrays['clean'][index:index+1]).copy(), device)
            received = tensor(np.asarray(arrays['received'][index:index+1]).copy(), device)
            row = one_record(reader, model, objective, scale, clean, received)
            row.update(role=role, scene=scene)
            rows.append(row)
            if len(rows) == 2:
                if not any(row['input_gradient_pairs']['final__shallow']['second_norm'] > 0 for row in rows):
                    raise ValueError('Actual source intermediate gradient must be nonzero')
                write(out/'smoke.json', dict(status='PASS', source_rows=2, actual_source_G=True,
                    original_F0=True, F0_logits_used=False, source_input_gradients_finite=True,
                    TX_labels_read=0, V_reads=0, OE_reads=0, target_inputs=0, optimizer_updates=0))
                print('REAL_SOURCE_MULTILEVEL_GRADIENT_SMOKE_PASS', flush=True)
    finally:
        for array in arrays.values():
            array._mmap.close()
    if len(rows) != 28 or any(not torch.equal(value, model.state_dict()[k]) for k,value in before_g.items()):
        raise ValueError('Expected 28 rows and unchanged source recovery model')
    check_F0_state(reader, before_f)
    write(out/'results.json', dict(status='SOURCE_MULTILEVEL_GRADIENT_COMPLETE',run_id=RUN,
        code_commit=args.commit,source_draw=1,source_records=28,source_roles=['L_s','U_s'],
        source_scenes=list(original.SCENES),source_model=original.PARENT_RUN,
        source_model_commit=original.PARENT_COMMIT,F0_identity=identity,rows=rows,
        gradient_object='restored_IQ_not_G_parameters', intermediate_scaling='per_record_clean_map_mean_square_floor_1e-8',
        optimizer_updates=0,model_state_updates=0,TX_labels_read=0,V_reads=0,OE_reads=0,
        target_inputs=0,F0_logits_used=False,source_panel_representative=False,seconds=time.perf_counter()-started))
    write(out/'complete.json', dict(status='ARTIFACTS_COMPLETE',source_only=True,records=28,optimizer_updates=0))
    print('SOURCE_MULTILEVEL_GRADIENT_COMPLETE', len(rows), flush=True)


def controller(args):
    from . import bounded_projected_resource_trial as resource
    from .cost_probe import write
    from .source_pair import read
    from .variance_trial import recovery_gpu_choice
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('Controller must have empty CUDA visibility')
    out = args.root/'runs'/RUN
    out.mkdir()
    write(out/'launch.json',dict(pid=os.getpid(),command=sys.argv,CWD=os.getcwd(),code_commit=args.commit))
    choices = []
    while not choices:
        choices = recovery_gpu_choice(allowed_gpus=(4,6), maximum_processes=3)
        if not choices:
            print('WAITING_FOR_ADMITTED_GPU', flush=True)
            time.sleep(30)
    admission = choices[0]
    command = [sys.executable,'-u','-m',MODULE,'worker','--root',str(args.root),
        '--commit',args.commit,'--inputs',str(args.inputs),'--cache',str(args.cache)]
    with (out/'worker.log').open('xb') as log:
        child = subprocess.Popen(command,cwd=Path.cwd(),
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=admission['CUDA_ID']),
            stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    write(out/'submitted.json',dict(pid=child.pid,command=command,CWD=os.getcwd(),admission=admission))
    failures = 0
    reason = None
    observed = None
    while child.poll() is None:
        try:
            observed = resource.compute_pids(admission['GPU'], admission['CUDA_ID'])
            failures = 0
            if child.pid in observed and len(observed) > 3:
                reason = 'TOTAL_COMPUTE_PID_CAP_EXCEEDED'
        except (RuntimeError, ValueError) as error:
            failures += 1
            print('GPU_COUNT_OBSERVATION_FAILED',str(error),flush=True)
            if failures >= 3:
                reason = 'GPU_COUNT_UNVERIFIABLE'
        if reason:
            child.terminate()
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
            write(out/'resource_stop.json',dict(pid=child.pid,reason=reason,cap=3,
                observed_compute_PIDs=sorted(observed) if observed is not None else None,
                only_own_child_signaled=True,other_owner_processes_signaled=False))
            break
        time.sleep(2)
    code = child.wait()
    write(out/'terminal.json',dict(pid=child.pid,exit_code=code))
    if code or reason:
        raise RuntimeError('Diagnostic stopped or failed; outputs preserved')
    if read(out/'complete.json')['status'] != 'ARTIFACTS_COMPLETE':
        raise ValueError('Incomplete source diagnostic')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode',choices=['run','worker'])
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--inputs',type=Path,required=True)
    parser.add_argument('--cache',type=Path,required=True)
    parser.add_argument('--commit',required=True)
    args = parser.parse_args()
    if (Path.cwd()/'release_commit.txt').read_text().strip() != args.commit:
        raise ValueError('Actual release commit required')
    controller(args) if args.mode == 'run' else worker(args)


if __name__ == '__main__':
    main()
