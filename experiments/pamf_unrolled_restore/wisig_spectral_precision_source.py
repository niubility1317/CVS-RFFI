"""Train and score fixed-weight and frequency-weighted WiSig restorers on source IQ."""

import argparse
import os
from pathlib import Path, PurePosixPath
import subprocess
from time import perf_counter

from experiments.cric_dura_comparison.io import read, tensor, write


RUN = '20261125-phase1-wisig-spectral-precision-source-m2-r01'
REFERENCE = '20261125-phase1-wisig-native-fir-source-m4-r01'
REFERENCE_COMMIT = 'ef9f4c227ce57c79dc17a3aeb599395cc25d883b'
ARMS = ('fixed_precision', 'spectral_precision')
ROUTES = ('full', 'residual')
VIEWS = ('clean', 'practical_high', 'practical_mid', 'practical_low_suburban',
         'practical_high_urban', 'practical_mid_urban', 'practical_low_urban')
MODES = ('direct', 'r69', *(f'{arm}_{name}' for arm in ARMS for name in ('map_1', 'final')))
COMPARISONS = (('spectral_precision_map_1', 'fixed_precision_map_1'),
               ('spectral_precision_final', 'fixed_precision_final'),
               *((f'{arm}_final', baseline) for arm in ARMS for baseline in ('direct', 'r69')))
SEED, STEPS, BATCH, HELD = 2026092701, 40_000, 24, 512
CHECKS = (1800, 6000, 12000, 20000, 30000, STEPS)


def reference_runtime(rt):
    parent = read(Path(rt['reference']) / 'runtime.json')
    if (parent['run_id'], parent['code_commit'], parent['project']) != (REFERENCE, REFERENCE_COMMIT, rt['project']):
        raise ValueError('Expected the registered native FIR source release for code and blinded comparators')
    return parent


def proof(arm):
    return dict(status='SOURCE_FROZEN', arm=arm, steps=STEPS, seed=SEED, batch=BATCH,
                fit_count=52092, held_count=HELD, TX_truth_reads=0, F_reads=0, V_reads=0,
                target_reads=0, trace_reads=0, parent_checkpoints=[], input_gate=False)


def train(config, arm, *, smoke):
    import numpy as np
    import torch
    from experiments.pamf_unrolled_restore.wisig_uncertain_prior import as_complex
    from experiments.pamf_unrolled_restore.wisig_uncertain_source import source_iq
    from wisig_spectral_precision import PrecisionFIR, check_spectral_map, source_objective

    if set(config) != {'run_root', 'source_root'}:
        raise ValueError('Training may receive only the source IQ and output paths')
    device = 'cuda:0'
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if smoke:
        check_spectral_map(device)
    source, ids, _ = source_iq(Path(config['source_root']))
    training, held = ids[HELD:], ids[:HELD]
    if len(training) != proof(arm)['fit_count']:
        raise ValueError('Unexpected TX-hidden source training population')
    torch.manual_seed(SEED)
    model = PrecisionFIR(arm).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(opt, STEPS, eta_min=3e-5)
    rng = np.random.default_rng(SEED + 2)
    generator = torch.Generator(device=device).manual_seed(SEED + 1)

    def batch(indices):
        return as_complex(tensor(source[indices], device))

    @torch.no_grad()
    def held_costs():
        fixed = torch.Generator(device=device).manual_seed(SEED + 5)
        costs = [source_objective(model, batch(chunk), fixed)[1] * len(chunk)
                 for chunk in np.array_split(held, HELD // BATCH)]
        reading = torch.stack(costs).sum(0) / HELD
        if not torch.isfinite(reading).all():
            raise FloatingPointError('Nonfinite held source costs')
        return reading.tolist()

    torch.cuda.reset_peak_memory_stats()
    started = perf_counter()
    history = {}
    for step in range(1, (3 if smoke else STEPS) + 1):
        train_iq = batch(rng.choice(training, BATCH, replace=False))
        loss, costs = source_objective(model, train_iq, generator)
        if not bool(torch.isfinite(costs).all() & torch.isfinite(loss)):
            raise FloatingPointError('Nonfinite source IQ objective')
        opt.zero_grad(set_to_none=True)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        if step == 3:
            heads = [model.spectral.out, model.coarse.out, model.estimator.out, model.global_head,
                     *(prior.out for prior in model.priors)]
            if model.spectral_weights is not None:
                heads.extend(block.output for block in model.spectral_weights)
            if any(x.weight.grad is None or not bool(x.weight.grad.norm() > 0) for x in heads):
                raise ValueError('Each declared restoration stage must receive a source loss gradient')
        opt.step()
        schedule.step()
        if step in (1, 3) or step % 100 == 0:
            print('WISIG_SPECTRAL_FIT', arm, step, *costs.detach().tolist(), float(norm), flush=True)
        if not smoke and step in CHECKS:
            model.eval()
            history[str(step)] = held_costs()
            model.train()
            print('WISIG_SPECTRAL_HELD', arm, step, *history[str(step)], flush=True)

    torch.cuda.synchronize()
    root = Path(config['run_root'])
    if smoke:
        write(root / f'{arm}-train-smoke.json', dict(status='PASS', steps=3, target_reads=0, F_reads=0))
        return
    evidence = dict(proof(arm), optimizer_seconds=perf_counter()-started,
                    held_trajectory=history, parameters=sum(p.numel() for p in model.parameters()),
                    peak_cuda=torch.cuda.max_memory_allocated())
    with (root / f'{arm}.pt').open('xb') as stream:
        torch.save(dict(weights=model.state_dict(), evidence=evidence), stream)
    write(root / f'{arm}-complete.json', evidence)


def models_from(current, arms=ARMS):
    import torch
    from wisig_spectral_precision import PrecisionFIR

    root = Path(current['run_root'])
    for arm in arms:
        if arm not in ARMS:
            raise ValueError('Expected a registered frozen source arm')
        meta = read(root / f'{arm}-complete.json')
        ckpt = torch.load(root / f'{arm}.pt', map_location='cpu', weights_only=True)
        if ckpt['evidence'] != meta or any(meta[k] != v for k, v in proof(arm).items()):
            raise ValueError('Expected the complete, fresh, gate-free source checkpoint')
        model = PrecisionFIR(arm).cuda().eval().requires_grad_(False)
        model.load_state_dict(ckpt['weights'], strict=True)
        yield model, None


def proposals(model, _unused, received):
    import torch

    stages, _ = model.stages(model.analyze(received))
    active = torch.ones(len(received), device=received.device, dtype=torch.bool)
    offsets = torch.zeros(len(received), device=received.device, dtype=torch.int64)
    return (stages[2], stages[4]), active, offsets


def reader_smoke(current):
    import torch
    from experiments.pamf_unrolled_restore.wisig_anchored_window_source import classify, core, tri
    from experiments.pamf_unrolled_restore.wisig_native_fir import disturb
    from experiments.pamf_unrolled_restore.wisig_uncertain_prior import as_complex

    timings = {}
    with torch.no_grad():
        source, _, indices = core.source_iq(Path(current['source_root']))
        received = as_complex(tensor(source[indices[:8]], 'cuda:0'))
        reader = tri.reader_factory(current).eval().requires_grad_(False)
        for arm, (model, state) in zip(ARMS, models_from(current), strict=True):
            disturbed = disturb(received, torch.Generator(device='cuda:0').manual_seed(SEED + 9), model)
            for view in (received, disturbed):
                stages, active, _ = proposals(model, state, view)
                for value in stages:
                    classify(reader, value)
                if not bool(active.all()) or torch.equal(stages[-1], view):
                    raise ValueError('Expected a real all-packet waveform readout from the frozen model')
            for _ in range(8):
                proposals(model, state, disturbed[:1])
            begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            begin.record()
            for _ in range(32):
                proposals(model, state, disturbed[:1])
            end.record()
            end.synchronize()
            timings[arm] = begin.elapsed_time(end) / 32
    write(Path(current['run_root']) / 'smoke.json', dict(status='PASS', source_count_per_view=8,
          source_V_reads=0, target_reads=0, restore_only_mean_ms_B1=timings))


def freeze_predictions(root, views):
    summaries = {}
    for route in ROUTES:
        by_view = read(root / f'{route}-complete.json')
        if set(by_view) != {f'{route}_{name}' for name in views}:
            raise ValueError('Missing the full registered source prediction matrix')
        if any(row['count'] != 1800 or row['active'] != dict.fromkeys(ARMS, 1800) for row in by_view.values()):
            raise ValueError('Both candidates must recover every source packet in each view')
        summaries.update(by_view)
    write(root / 'complete.json', dict(status='FROZEN', count=1800, modes=list(MODES), views=list(views),
          source_truth_reads=0, target_reads=0, summaries=summaries))


def evaluate(rt, action, name):
    from experiments.pamf_lora_main.runner import activate
    from experiments.pamf_unrolled_restore.wisig_anchored_window_source import core, evaluate as predict

    original = reference_runtime(rt)
    current = dict(original, run_root=rt['run_root'])
    if action == 'score':
        core.baseline.score(current, MODES, comparisons=COMPARISONS)
        return
    activate(original)
    if action == 'reader_smoke':
        reader_smoke(current)
    else:
        predict(current, name, arms=ARMS, models_from=models_from, variants=proposals)


def dispatch(runtime):
    from experiments.pamf_lora_main.packet_runner import schedule

    rt = read(runtime)
    parent = reference_runtime(rt)
    root = Path(rt['run_root'])
    root.mkdir()
    source_config = root / 'fit-inputs.json'
    write(source_config, dict(run_root=rt['run_root'], source_root=parent['source_root']))

    def status(value, **fields):
        write(root / 'state.json', dict(status=value, **fields), exclusive=False)

    def launch(_plan, action, name, gpu=None):
        archive = rt['reference']
        restricted = action in ('train_smoke', 'train')
        cfg = source_config if restricted else runtime
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='' if gpu is None else str(gpu),
                   PYTHONPATH=archive, OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1')
        cmd = [rt['python'], '-u', str(Path(__file__).resolve()), action, str(cfg), name]
        with (root / f'{action}-{name}.log').open('x', encoding='utf-8') as stream:
            child = subprocess.Popen(cmd, env=env, cwd=archive, stdout=stream, stderr=subprocess.STDOUT)
        print('WISIG_SPECTRAL_WORKER', action, name, child.pid, gpu, flush=True)
        return child

    try:
        rows = subprocess.check_output(['nvidia-smi', '--query-gpu=index,utilization.gpu,memory.free',
                                        '--format=csv,noheader,nounits'], text=True).splitlines()
        devices = [tuple(map(int, row.split(','))) for row in rows]
        rt['allowed_gpus'] = [i for _, _, i in sorted((util, -free, i) for i, util, free in devices)]
        status('SOURCE_TRAIN_SMOKE')
        schedule(rt, [('train_smoke', arm) for arm in ARMS], launch)
        status('SOURCE_TRAINING')
        schedule(rt, [('train', arm) for arm in ARMS], launch)
        status('SOURCE_READER_SMOKE')
        schedule(rt, [('reader_smoke', 'frozen')], launch)
        (root / 'predictions').mkdir()
        status('SOURCE_PREDICTING')
        schedule(rt, [('predict', route) for route in ROUTES], launch)
        freeze_predictions(root / 'predictions', VIEWS)
        status('PREDICTIONS_FROZEN')
        if launch(rt, 'score', 'independent').wait():
            raise ValueError('Independent source scorer failed')
        status('SOURCE_SCORED')
    except Exception as exc:
        status('FAILED', error=str(exc))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='action', required=True)
    commands.add_parser('dispatch').add_argument('runtime', type=Path)
    for action in ('train_smoke', 'train', 'reader_smoke', 'predict', 'score'):
        cmd = commands.add_parser(action)
        cmd.add_argument('runtime', type=Path)
        cmd.add_argument('name')
    pub = commands.add_parser('publish')
    for key in ('repo', 'output', 'context', 'ssh-config'):
        pub.add_argument('--' + key, type=Path, required=True)
    args = parser.parse_args()
    if args.action == 'publish':
        from experiments.pamf_lora_main.publish import release, release_commit

        ctx = read(args.context)
        rt = {key: ctx[key] for key in ('project', 'python', 'frozen_sources')}
        project = PurePosixPath(rt['project'])
        reference = str(project / 'releases' / f'{REFERENCE}-release')
        rt.update(run_id=RUN, runner=__spec__.name, code_commit=release_commit(args.repo), reference=reference,
                  release_root=str(project / 'releases' / f'{RUN}-release'), run_root=str(project / 'runs' / RUN),
                  external_releases=[reference], max_workers=2, max_compute_processes=8)
        release(args.repo, args.output, rt, args.ssh_config)
    elif args.action == 'dispatch':
        dispatch(args.runtime)
    elif args.action in ('train_smoke', 'train'):
        train(read(args.runtime), args.name, smoke=args.action == 'train_smoke')
    else:
        evaluate(read(args.runtime), args.action, args.name)


if __name__ == '__main__':
    main()
