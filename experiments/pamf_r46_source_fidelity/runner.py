import argparse
import gc
import json
import os
from pathlib import Path
import sys
import time
import traceback
import numpy as np
import torch
from . import design as d
from .metrics import complex_iq, fidelity, relative_error, summarize
from experiments.pamf_r46_cuda_graph.runtime import load_model, validate_sources
from experiments.pamf_r46_multipoint_fast.tensor_io import array_tensor
from experiments.pamf_restoration.data import write_json
from experiments.pamf_restoration.runner import code_commit, read_json


def select_positions(roles, per_role=d.PER_ROLE):
    if roles.ndim != 1 or set(roles.tolist()) != {'L_s', 'U_s'} or per_role < 1:
        raise ValueError('Original L_s/U_s only; no V/target/labels')
    selected = []
    for role in ('L_s', 'U_s'):
        positions = np.flatnonzero(roles == role)
        if len(positions) < per_role:
            raise ValueError('Enough distinct original source-training positions required')
        selected.extend(positions[np.linspace(0, len(positions)-1, per_role, dtype=np.int64)].tolist())
    return np.asarray(selected, dtype=np.int64)


def load_panel(args):
    if args.parent_action.name != d.frozen.ACTION_RUN:
        raise ValueError('Fixed original R07 source action cache required before IQ IO')
    if args.source_cache.name != 'source_cache':
        raise ValueError('Existing source metadata handle required before IO')
    manifest = read_json(args.source_cache/'manifest.json')
    if manifest.get('status') != 'READY' or manifest.get('native_role_ids') != 'EXACT_MATCH':
        raise ValueError('Reuse READY/EXACT_MATCH source data only')
    cache = args.parent_action/'cache'
    roles = np.load(cache/'train/roles.npy', allow_pickle=False)
    if roles.shape != (63000,) or (roles == 'L_s').sum() != 6300 or (roles == 'U_s').sum() != 56700:
        raise ValueError('Original 6300 L_s plus 56700 U_s required')
    take = select_positions(roles)
    paths = dict(clean=cache/'train/clean.npy', received01=cache/'epochs/01/received.npy',
                 received20=cache/'epochs/20/received.npy')
    panel = {}
    for view, path in paths.items():
        values = np.load(path, mmap_mode='r', allow_pickle=False)
        if values.shape != (63000, 2, 256) or values.dtype != np.dtype('float32'):
            raise ValueError('Unchanged native source IQ cache required')
        panel[view] = array_tensor(values[take], 'cpu')
    return panel, take, roles[take]


def grouped(values):
    return {role:summarize({k:v[sl] for k,v in values.items()}) for role,sl in
            (('all', slice(None)), ('L_s', slice(0, d.PER_ROLE)), ('U_s', slice(d.PER_ROLE, None)))}


@torch.no_grad()
def diagnose(args):
    if args.output.name != d.RUN or args.output.exists():
        raise ValueError('New immutable diagnostic output required')
    if code_commit() != args.expected_commit or os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('Matching committed code and CPU-only environment required')
    validate_sources(args)
    panel, take, roles = load_panel(args)
    args.output.mkdir()
    args._owns_output = True
    write_json(args.output/'launch.json', dict(status='SOURCE_DIAGNOSTIC_RUNNING', commit=args.expected_commit,
        pid=os.getpid(), cwd=os.getcwd(), argv=sys.argv, CPU_threads=2, GPU_workers=0, time=time.time()))
    write_json(args.output/'comparison_plan.json', d.configuration())
    write_json(args.output/'panel.json', dict(opaque_role_positions=take.tolist(), roles=roles.tolist(),
        count=256, L_s=128, U_s=128, TX_labels_read=0, V_reads=0, target_reads=0, F_weight_reads=0))
    started = time.perf_counter()
    reference = panel['clean']
    baseline = {view:grouped(fidelity(value, reference)) for view,value in panel.items()}
    write_json(args.output/'raw_source_metrics.json', baseline)
    records = {}
    for arm in d.ARMS:
        folder = args.output/arm
        folder.mkdir()
        model, payload = load_model(args, arm, 'cpu')
        snapshot = {k:v.clone() for k,v in model.state_dict().items()}
        probe = model(reference[:2])
        if probe.shape != (2,2,256) or not torch.isfinite(probe).all():
            raise ValueError('Actual checkpoint two-L_s no-query smoke failed')
        write_json(folder/'smoke.json', dict(status='PASS', real_L_s_IQ=2, TX_labels_read=0,
            F_weight_reads=0, F_output_reads=0, V_reads=0, target_reads=0, optimizer_updates=0))
        print(json.dumps(dict(event='REAL_SOURCE_CHECKPOINT_SMOKE_PASS', arm=arm)), flush=True)
        views = {}
        for view, value in panel.items():
            parts = {}
            for offset in range(0, len(value), d.BATCH):
                outputs, variance, action, coarse = model.stage_outputs(value[offset:offset+d.BATCH])
                if len(outputs) != 3 or any(not torch.isfinite(v).all() for v in (*outputs, coarse, variance)) or not (variance > 0).all():
                    raise ValueError('Full frozen restorer produced invalid source output')
                clean = reference[offset:offset+d.BATCH]
                measured = fidelity(outputs[-1], clean)
                for name, stage in [('coarse',coarse), *[(f'stage{i+1}',v) for i,v in enumerate(outputs)]]:
                    measured[name+'_iq_nmse'] = relative_error(complex_iq(stage), complex_iq(clean))
                for key, values in measured.items():
                    parts.setdefault(key, []).append(values.detach().cpu())
            per_example = {k:torch.cat(v) for k,v in parts.items()}
            views[view] = grouped(per_example)
            write_json(folder/(view+'_metrics.json'), views[view])
            print(json.dumps(dict(event='SOURCE_VIEW_COMPLETE', arm=arm, view=view, count=256,
                iq_nmse=views[view]['all']['iq_nmse']['mean'], seconds=time.perf_counter()-started)), flush=True)
        if any(not torch.equal(v, snapshot[k]) for k,v in model.state_dict().items()) or any(p.grad is not None or p.requires_grad for p in model.parameters()):
            raise ValueError('Frozen source checkpoint mutated')
        records[arm] = dict(parameters=d.frozen.PARAMETERS[arm], source_final=[payload['epoch'],payload['updates']],
            views=views, checkpoint_unchanged=True, optimizer_updates=0)
        write_json(folder/'complete.json', dict(status='SOURCE_DIAGNOSTIC_COMPLETE', arm=arm,
            views=list(d.VIEWS), count=256, checkpoint_unchanged=True))
        del model, payload, snapshot
        gc.collect()
    result = dict(status='SOURCE_DIAGNOSTIC_COMPLETE', run=d.RUN, method_commit=args.expected_commit,
        configuration=d.configuration(), rows=records, raw_source=baseline, seconds=time.perf_counter()-started,
        F_weight_reads=0, F_output_reads=0, TX_labels_read=0, V_reads=0, target_reads=0,
        optimizer_updates=0, new_network_versions=0, all_checkpoint_states_unchanged=True)
    write_json(args.output/'results.json', result)
    write_json(args.output/'complete.json', dict(status=result['status'], arms=list(records), views=list(d.VIEWS),
        source_examples_per_view=256, method_commit=args.expected_commit, seconds=result['seconds']))
    print(json.dumps(dict(event='SOURCE_DIAGNOSTIC_COMPLETE', seconds=result['seconds'])), flush=True)


def parser():
    value = argparse.ArgumentParser(description='CPU-only source-training IQ fidelity; no F, labels, V or target')
    for flag in ('teacher-run','compact-parent','factor-run','parent-action','source-cache','output'):
        value.add_argument('--'+flag, type=Path, required=True)
    value.add_argument('--expected-commit', required=True)
    return value


def main():
    args = parser().parse_args()
    args._owns_output = False
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    try:
        diagnose(args)
    except Exception as exc:
        if args._owns_output and not (args.output/'failure.json').exists():
            write_json(args.output/'failure.json', dict(status='TECHNICAL_FAILURE', error=repr(exc), traceback=traceback.format_exc()))
        raise


if __name__ == '__main__':
    main()
