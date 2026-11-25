"""Score inference-only observation interventions on the original frozen WiSig R69."""

import argparse
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from experiments.cric_dura_comparison.io import read, tensor, write
from experiments.pamf_lora_main.runner import activate
from experiments.pamf_r46_cuda_graph.program import StaticAction, estimate
from experiments.pamf_stage2_inference.program import solve_context

from . import wisig_joint_preamble_source as tri
from . import wisig_overlap_isolation as isolation
from . import wisig_uncertain_source as core
from .wisig_anchored_window_source import classify
from .wisig_short_training import _estimate
from .wisig_tri_learned_replay import BASELINE_COMMIT, parent
from .wisig_uncertain_prior import as_complex


RUN = '20261125-diagnostic-wisig-r69-observation-source-m2-r01'
ROUTES, VIEWS, BATCH = isolation.ROUTES, isolation.VIEWS, isolation.BATCH
MODES = ('direct', 'r69', 'unit_action', 'fir_only', 'pointwise_only', 'constant_variance', 'repeat_variance')
COMPARISONS = tuple((variant, 'r69') for variant in MODES[2:]) + (('fir_only', 'unit_action'),
               ('pointwise_only', 'unit_action'), ('fir_only', 'pointwise_only'))


def action_views(action):
    impulse = torch.ones_like(action.kernel[:, :1])
    one = torch.ones_like(action.main)
    zero = torch.zeros_like(action.image)
    return (StaticAction(impulse, one, zero), StaticAction(action.kernel, one, zero),
            StaticAction(impulse, action.main, action.image))


def repeat_variance(received):
    current = np.asarray(as_complex(received).tolist(), dtype=np.complex64)
    observed = _estimate(current)[0].effective_variance
    # The repeat residual estimates complex noise power. R69 uses variance per real component.
    return tensor(np.clip(.5*observed, 1e-8, 1.).astype(np.float32), received.device)


def head(model, received):
    base = model.frozen.coarse(received)
    action, variance = estimate(model.frozen.estimator, received, base)
    return base, action, variance


def intervene(model, received, constant):
    base, action, variance = head(model, received)
    actions = action_views(action)
    repeated = repeat_variance(received)
    replacement = ((action, variance), *((a, variance) for a in actions),
                   (action, torch.full_like(variance, constant)), (action, repeated))
    options, valid = [], []
    for alternative, noise in replacement:
        wave, _, _, _, finite = solve_context(model, received, base, alternative, noise)
        options.append(wave)
        valid.append(finite)
    if not bool(torch.stack(valid).all()):
        raise FloatingPointError('An R69 observation intervention returned nonfinite IQ')
    return options, torch.stack((variance, repeated))


def calibrate(model, source, calibration):
    values = []
    for chunk in np.array_split(calibration, len(calibration)//BATCH):
        packet = tensor(source[chunk], 'cuda:0')
        values.append(head(model, packet)[2])
    values = torch.cat(values)
    if not bool(torch.isfinite(values).all() & (values > 0).all()):
        raise FloatingPointError('Original R69 returned invalid source U variance')
    return float(values.quantile(.5))


@torch.no_grad()
def smoke(rt):
    source, _, calibration = core.source_iq(Path(rt['source_root']))
    engine = core.baseline.original_r69(Path(rt['parent_root']))
    model = engine.pipeline.model
    root = Path(rt['run_root'])
    constant = calibrate(model, source, calibration)
    write(root/'constant-variance.json', dict(role='U_s', count=len(calibration), variance=constant,
                                             F_reads=0, V_reads=0, target_reads=0))
    reader = tri.reader_factory(rt).eval().requires_grad_(False)
    received = tensor(source[calibration[:4]], 'cuda:0')
    original, = engine(received)
    options, _ = intervene(model, received, constant)
    torch.testing.assert_close(original, options[0], rtol=5e-5, atol=5e-6)
    for wave in (received, *options):
        classify(reader, as_complex(wave))
    write(root/'smoke.json', dict(status='PASS', count=len(received), V_reads=0, target_reads=0, optimizer_updates=0))


@torch.no_grad()
def evaluate(rt, route):
    cache, _, _, _, ids, _, _ = core.baseline.handles(rt)
    baseline = parent(rt, tri.RUN, BASELINE_COMMIT)/'predictions'
    done = read(baseline/'complete.json')
    if (done['status'], done['modes'], done['count'], done['source_truth_reads'], done['target_reads']) != (
            'FROZEN', list(tri.MODES), len(ids), 0, 0):
        raise ValueError('Expected archived matched direct and R69 predictions')
    reader = tri.reader_factory(rt).eval().requires_grad_(False)
    engine = core.baseline.original_r69(Path(rt['parent_root']))
    root = Path(rt['run_root'])
    constant = read(root/'constant-variance.json')['variance']
    summary = {}
    for view in VIEWS:
        started = perf_counter()
        key = f'{route}_{view}'
        wave = np.load(cache/ROUTES[route]/f'{view}.npy', mmap_mode='r', allow_pickle=False)
        with np.load(baseline/f'{key}.npz', allow_pickle=False) as archived:
            if not np.array_equal(ids, archived['ids']) or wave.shape != (len(ids), 2, 256) or wave.dtype != np.float32:
                raise ValueError('Source IQ view or archived physical IDs differ')
            reference = archived['decision'][[tri.MODES.index(name) for name in MODES[:2]]]
        decisions, measurements = [], []
        for begin in range(0, len(wave), BATCH):
            received = tensor(wave[begin:begin+BATCH], 'cuda:0')
            original, = engine(received)
            options, variance = intervene(engine.pipeline.model, received, constant)
            torch.testing.assert_close(original, options[0], rtol=5e-5, atol=5e-6)
            decisions.append([classify(reader, as_complex(x)).argmax(-1).tolist() for x in (received, original, *options[1:])])
            measurements.append(variance.tolist())
        matrix = np.concatenate(decisions, axis=1).astype(np.uint8)
        if matrix.shape != (len(MODES), len(ids)) or not np.array_equal(matrix[:2], reference):
            raise ValueError('The matched direct and R69 decisions must replay exactly')
        with (root/'predictions'/f'{key}.npz').open('xb') as stream:
            np.savez(stream, ids=ids, decision=matrix)
        variance = np.concatenate(measurements, axis=1)
        summary[key] = dict(count=len(ids), seconds=perf_counter()-started,
                            variance_quantiles=np.quantile(variance, [.1, .5, .9], axis=1).T.tolist())
        print('WISIG_R69_OBSERVATION_FROZEN', key, len(ids), flush=True)
    write(root/'predictions'/f'{route}-complete.json', summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest='action', required=True)
    publisher = actions.add_parser('publish')
    for name in ('repo', 'output', 'context', 'ssh-config'):
        publisher.add_argument('--'+name, type=Path, required=True)
    for action in ('dispatch', 'smoke', 'evaluate', 'score'):
        worker = actions.add_parser(action)
        worker.add_argument('runtime', type=Path)
        if action != 'dispatch':
            worker.add_argument('name', choices=ROUTES if action == 'evaluate' else ('source',))
    args = vars(parser.parse_args())
    action = args.pop('action')
    if action == 'publish':
        isolation.publish(**args, run=RUN, runner=__spec__.name)
        return
    rt = read(args['runtime'])
    if action == 'dispatch':
        if rt['run_id'] != RUN:
            raise ValueError('Expected the immutable R69 observation audit root')
        isolation.dispatch(rt, modes=MODES, paired_source_iq_isolated=False)
    elif action == 'score':
        core.baseline.score(rt, MODES, comparisons=COMPARISONS)
    else:
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
        activate(rt)
        smoke(rt) if action == 'smoke' else evaluate(rt, args['name'])


if __name__ == '__main__':
    main()
