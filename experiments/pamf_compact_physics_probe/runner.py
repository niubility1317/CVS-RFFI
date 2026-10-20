"""CPU2 source-only frozen context extraction, then independent oracle scoring."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import numpy as np
import torch
from experiments.pamf_restoration.runner import array_tensor, tensor_array, code_commit, read_json
from experiments.pamf_restoration.data import IQSource, SCENES, write_json
from experiments.pamf_restoration.models import pair
from experiments.pamf_r17_task.runtime import validate_teacher_root, load_original
from experiments.pamf_iq_joint_compact.runtime import load_checkpoint as load_compact
from experiments.pamf_compact_identity_probe.runtime import validate_training
from .analysis import KEYS, validate_estimate, diagnostics, summarize

ROOT = Path(__file__).resolve().parents[2]
RUN = '20261020-diagnostic-restoration-compact-physics-s2026092701-m3-r40'
R17 = '20261020-phase1-restoration-unrolled-selfsup-s2026092701-m2-r17'
R38 = '20261020-phase1-restoration-r17-IQ-joint-compact-s2026092701-m2-r38'
R38_COMMIT = 'e0868f0a0c3cf056a1753a5b4e5ce4f0bff84381'
ACTION = '20261019-phase1-restoration-learned-action-source-s2026092701-r07'
ACTION_COMMIT = 'c8ccdfdbd03ab5ad8828b1eb9210b99a664ebe7b'
ROWS = ('r17_reference', 'dense_three_quarter', 'dense_half')
VIEWS = ('clean', *SCENES)
PROBE_SEED = 2026102040
PATH_ARGS = ('runs_root', 'source_cache', 'output')


def configuration(commit):
    return dict(schema='compact_source_receive_chain_probe_v1', run=RUN, commit=commit,
        rows=list(ROWS), views=list(VIEWS), source_count=1800, CPU_threads=2, GPU_workers=0,
        context_only=True, prior_forwards=0, extra_final_IQ_predictions=0,
        context_estimates=3*7*1800, training_updates=0, state_updates=0,
        F_weights_read=0, F_outputs_read=0, TX_labels_read=0, target_reads=0,
        model_seed=2026092701, probe_seed=PROBE_SEED, probes_per_sample=4,
        compare='Coarse clean NMSE; composite operator on paired clean and common complex Gaussian probes; real-axis native noise variance and residuals',
        noise_units='Absolute per-real-coordinate variance; no division by SCALE**2; clean uses synthetic floor1e-8 and has no actual noise',
        source_selection=False, candidate_ranking=False, new_network=False,
        oracle_after_all_estimates=True, original_R17_source_selected=True,
        stop_rule='Wrong source/model contract, incomplete/nonfinite input or estimate, outputcollision or technical failure:preserve outputs,no automatic retry;healthy peers untouched',
        claim='Known simulated residual operator, not unique kernel factorization, real in-orbit channel or TX-detail authenticity proof')


def inputs(args):
    action_root = args.runs_root/ACTION
    if read_json(action_root/'launch.json').get('commit') != ACTION_COMMIT:
        raise ValueError('Original source action simulation provenance required')
    header = read_json(action_root/'cache/complete.json')
    if (header.get('status') != 'ACTION_SOURCE_CACHE_COMPLETE' or
            header.get('train_count') != 63000 or header.get('source_V_count') != 1800 or
            header.get('TX_labels_read') != 0 or header.get('target_reads') != 0):
        raise ValueError('Existing original source action cache required')
    reference = action_root/'cache/validation'
    ids = np.load(reference/'physical_ids.npy', allow_pickle=False)
    source = IQSource(args.source_cache, 'V')
    lookup = {str(v): i for i, v in enumerate(source.ids)}
    if len(ids) != 1800 or len(set(ids.tolist())) != 1800 or any(str(v) not in lookup for v in ids):
        raise ValueError('Original 1800 source V physical IDs required')
    index = np.asarray([lookup[str(v)] for v in ids])
    return reference, ids, source.rx[index], source.day[index]


def validate_models(args):
    validate_teacher_root(args.runs_root/R17)
    compact = args.runs_root/R38
    if read_json(compact/'launch.json').get('commit') != R38_COMMIT:
        raise ValueError('Original F-free compact source run required')
    for row in ROWS[1:]:
        validate_training(read_json(compact/row/'training_complete.json'), row)


def model_for(args, row):
    if row not in ROWS:
        raise ValueError('Fixed all-three comparison required before checkpoint IO')
    if row == ROWS[0]:
        model, value = load_original(args.runs_root/R17/'unroll_unshared/native_map_prior/final.pt', 'cpu')
        if value['candidate'] != 'unroll_unshared':
            raise ValueError('Original source-selected R17 row required')
    else:
        model, value = load_compact(args.runs_root/R38/row/'native_map_prior/final.pt', 'cpu')
        if value['arm'] != row:
            raise ValueError('Compact checkpoint row mismatch')
    if (value['epoch'], value['updates']) != (20, 4940):
        raise ValueError('Fixed final source checkpoint required')
    return model.eval().requires_grad_(False)


def estimate(model, received, row):
    base, operator, variance = (model.frozen.estimate(received) if row == ROWS[0]
                                else model.estimate_context(received))
    return dict(coarse=tensor_array(base), kernel=tensor_array(pair(operator.kernel)),
                coefficients=tensor_array(torch.stack((operator.main.real, operator.main.imag,
                    operator.image.real, operator.image.imag), dim=1)), variance=tensor_array(variance))


@torch.no_grad()
def smoke(args):
    validate_models(args)
    x = array_tensor(IQSource(args.source_cache, 'L_s').iq[:2], 'cpu')
    for row in ROWS:
        model = model_for(args, row)
        validate_estimate(estimate(model, x, row), 2)
        if any(p.requires_grad for p in model.parameters()):
            raise ValueError('Frozen context only')
        del model
    write_json(args.output/'smoke.json', dict(status='PASS', real_Ls_IQ=2, actual_checkpoints=3,
        V_reads=0, target_reads=0, TX_labels_read=0, F_access=0, state_updates=0, prior_forwards=0))


@torch.no_grad()
def extract(args):
    validate_models(args)
    reference, ids, rx, day = inputs(args)
    np.savez(args.output/'physical_metadata.npz', physical_ids=ids, rx=rx, day=day)
    started = time.perf_counter()
    for row in ROWS:
        model = model_for(args, row)
        for view in VIEWS:
            received = np.load(reference/(view+'.npy'), mmap_mode='r', allow_pickle=False)
            if received.shape != (1800, 2, 256) or not np.isfinite(received).all():
                raise ValueError('Complete fixed source observation required')
            pieces = {key: [] for key in KEYS}
            for begin in range(0, len(ids), 64):
                values = estimate(model, array_tensor(received[begin:begin+64], 'cpu'), row)
                validate_estimate(values, len(values['variance']))
                for key in KEYS:
                    pieces[key].append(values[key])
            values = {key: np.concatenate(pieces[key]) for key in KEYS}
            validate_estimate(values, 1800)
            with (args.output/(row+'__'+view+'.npz')).open('xb') as stream:
                np.savez(stream, **values)
            print(json.dumps(dict(event='SOURCE_CONTEXT_FIXED', row=row, view=view, count=1800)), flush=True)
        del model
    write_json(args.output/'estimates_complete.json', dict(status='ESTIMATES_COMPLETE', rows=list(ROWS),
        views=list(VIEWS), count=1800, cells=21, TX_labels_read=0, F_access=0, target_reads=0,
        clean_oracle_reads=0, state_updates=0, seconds=time.perf_counter()-started))


def load_estimates(args, ids):
    header = read_json(args.output/'estimates_complete.json')
    expected = dict(status='ESTIMATES_COMPLETE', rows=list(ROWS), views=list(VIEWS), count=1800,
        cells=21, TX_labels_read=0, F_access=0, target_reads=0, clean_oracle_reads=0, state_updates=0)
    if any(header.get(k) != v for k, v in expected.items()):
        raise ValueError('All frozen estimates required before oracle access')
    with np.load(args.output/'physical_metadata.npz', allow_pickle=False) as data:
        if not np.array_equal(data['physical_ids'], ids):
            raise ValueError('Frozen context physical IDs differ')
    result = {}
    for row in ROWS:
        for view in VIEWS:
            with np.load(args.output/(row+'__'+view+'.npz'), allow_pickle=False) as data:
                value = {key: data[key] for key in data.files}
            validate_estimate(value, 1800)
            result[(row, view)] = value
    return result


def score(args):
    reference, ids, rx, day = inputs(args)
    estimates = load_estimates(args, ids)
    # No oracle file is opened until all 21 estimate cells pass completeness.
    clean = np.load(reference/'clean.npy', allow_pickle=False)
    generator = np.random.default_rng(PROBE_SEED)
    probes = (generator.normal(size=(4, len(ids), 256))+
              1j*generator.normal(size=(4, len(ids), 256)))/np.sqrt(2)
    records, saved = [], {}
    started = time.perf_counter()
    for view in VIEWS:
        folder = args.runs_root/ACTION/'cache/validation_actions'/view
        oracle = {key: np.load(folder/(key+'.npy'), allow_pickle=False)
                  for key in ('kernel', 'coefficients', 'variance')}
        received = np.load(reference/(view+'.npy'), allow_pickle=False)
        for row in ROWS:
            values = diagnostics(estimates[(row, view)], oracle, clean, received, probes)
            records.extend(summarize(values, row, view, rx, day))
            for key, value in values.items():
                saved[row+'__'+view+'__'+key] = value
    with (args.output/'sample_metrics.npz').open('xb') as stream:
        np.savez(stream, **saved)
    write_json(args.output/'summary.json', dict(status='ANALYZED', run=RUN, role='source_V',
        records=records, configuration=configuration(args.expected_commit), seconds=time.perf_counter()-started,
        selected_candidate=None, TX_labels_read=0, F_access=0, target_reads=0, source_fit_updates=0,
        oracle_after_all_estimates=True, model_or_training_decisions_changed=False, goal_complete=False))


def command(args, mode):
    result = [sys.executable, '-u', '-m', 'experiments.pamf_compact_physics_probe.runner', mode]
    for key in (*PATH_ARGS, 'expected_commit'):
        result += ['--'+key.replace('_', '-'), str(getattr(args, key))]
    return result


def run(args):
    args.output.mkdir()
    (args.output/'logs').mkdir()
    write_json(args.output/'launch.json', dict(pid=os.getpid(), cwd=str(ROOT), argv=sys.argv,
        commit=args.expected_commit, started=time.time()))
    write_json(args.output/'resolved_config.json', configuration(args.expected_commit))
    try:
        smoke(args)
        for mode in ('extract', 'score'):
            with (args.output/'logs'/(mode+'.log')).open('x', encoding='utf-8') as log:
                process = subprocess.Popen(command(args, mode), cwd=ROOT, stdin=subprocess.DEVNULL,
                    stdout=log, stderr=subprocess.STDOUT, env=dict(os.environ, CUDA_VISIBLE_DEVICES=''))
                write_json(args.output/(mode+'_launch.json'), dict(pid=process.pid, cwd=str(ROOT), argv=command(args, mode)))
                if process.wait():
                    raise RuntimeError('Source '+mode+' failed; preserve outputs, no automatic retry')
        write_json(args.output/'complete.json', dict(status='ANALYZED', TX_labels_read=0, F_access=0,
            target_reads=0, GPU_workers=0, state_updates=0, new_network=False, goal_complete=False))
    except BaseException as error:
        write_json(args.output/'failure.json', dict(status='FAILED_TECHNICAL', error=repr(error),
            traceback=traceback.format_exc(), outputs_preserved=True, automatic_retry=False))
        raise


def main():
    parser = argparse.ArgumentParser(description='Fixed CPU source-only compact receive-chain diagnostic; no new network or target')
    parser.add_argument('mode', choices=('run', 'extract', 'score'))
    for key in PATH_ARGS:
        parser.add_argument('--'+key.replace('_', '-'), type=Path, required=True)
    parser.add_argument('--expected-commit', required=True)
    args = parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('This registered source diagnostic is CPU-only')
    if code_commit() != args.expected_commit or args.output.name != RUN:
        raise ValueError('Pinned code and immutable registered output required')
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    globals()[args.mode](args)


if __name__ == '__main__':
    main()
