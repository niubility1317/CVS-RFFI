"""Train the native W24 restorer with separate disturbed and clean outer-chirp losses."""

import argparse
from dataclasses import replace
import importlib
import os
from pathlib import Path, PurePosixPath
import subprocess
from time import monotonic, sleep

from experiments.cric_dura_comparison.io import read, tensor, write


RUN = '20261125-phase1-lora-w24-outer-objective-source-s2026112991-r01'
NAME = 'w24_outer_objective'
STEPS = 40_000
SEED = 2026112991
MODES = ('direct', 'all_half', 'outer_objective_half')
CONTRASTS = (('outer_objective_half', 'all_half'),)


def recipe_for(training):
    def no_gate(model, _tracker, _unused):
        return None, dict(parameters=sum(p.numel() for p in model.parameters()), input_gate=False)

    return replace(training.RECIPE, name=NAME, finalize_calibration=no_gate)


def objective_for(archived):
    # This reuses the original module's coordinates, source perturbation and loss primitives.
    import torch

    risk = archived.waveform_risk
    window = archived.WIDE

    def calculate(model, tracker, raw, rng, *, with_identity, recipe, step):
        raw = raw / raw.abs().square().mean(-1, keepdim=True).clamp_min(1e-12).sqrt()
        clean = tracker.aperture(raw)
        center = clean[..., :window]
        framed, phase, _, rms = archived.coordinates(tracker, archived.real_iq(center))
        clean_frame = phase * rms
        delta = archived.weak_difference(framed)
        change = delta * clean_frame
        a, b = (tracker.aperture(archived.condition_waveforms(raw.unflatten(-1, (8, 1024)), rng).flatten(-2))
                for _ in range(2))
        if with_identity:
            a[-2:] = b[-2:] = clean[-2:]
        data = torch.stack((a, b, torch.cat((a[..., :window] + change, a[..., window:]), dim=-1)), dim=1)
        observed = data.flatten(0, 1)
        y, phase, quality, rms = archived.coordinates(tracker, archived.real_iq(observed[..., :window]))
        frame = phase * rms
        ground = torch.stack((center, center, center + change), dim=1).flatten(0, 1) / frame
        (coarse, first, final), action, variance, _, _ = model.propose(y, quality, rms, observed[..., window:])
        half = y + .5 * (final - y)
        prediction = (half * frame).unflatten(0, data.shape[:2])
        corruption = (y - action(ground)).abs().square()
        signal = archived.split((prediction[:, 2] - prediction[:, 0]) / clean_frame)
        expected = archived.split(delta)
        preserve = (archived.transfer_risk(signal[:, :1], expected[:, :1])
                    + archived.transfer_risk(signal[:, 1:], expected[:, 1:]))
        consistency = ((prediction[:, 0] - prediction[:, 1]) / clean_frame).abs().square().mean()
        identity = y.real.new_zeros(())
        outer_identity = y.real.new_zeros(())
        n_disturbed = len(half)
        if with_identity:
            identity = (prediction[-2:, :2] - data[-2:, :2, :, :window]).abs().square().mean()
            plain = half.unflatten(0, data.shape[:2])[-2:, :2, [0, 7]]
            received = y.unflatten(0, data.shape[:2])[-2:, :2, [0, 7]]
            outer_identity = risk(plain, received)
            n_disturbed -= 6
        half_risk = risk(half, ground)
        outer_disturbed = risk(half[:n_disturbed, [0, 7]], ground[:n_disturbed, [0, 7]])
        inner_disturbed = risk(half[:n_disturbed, 1:-1], ground[:n_disturbed, 1:-1])
        # Per-chirp averaging gives 1/4 to the two outer chirps. Transfer 1/4 of
        # the disturbed part of that loss from the six inner chirps to the outer two.
        balanced_risk = half_risk + .25 * (n_disturbed / len(half)) * (outer_disturbed - inner_disturbed)
        a_weight, b_weight, c_weight = recipe.loss_weights(step)
        values = torch.stack((risk(coarse, ground), risk(first, ground), risk(final, ground), balanced_risk,
                              preserve, consistency, (corruption / (.01 + variance.detach())).log1p().mean(),
                              (corruption.detach() / variance + variance.log()).mean(), identity, outer_identity))
        weights = values.new_tensor((.6 * a_weight, .4 * a_weight, .35 * b_weight, .65 * b_weight,
                                     .30, c_weight, .07, .015, .35, .35))
        return values @ weights, torch.cat((values, outer_disturbed[None], inner_disturbed[None]))

    return calculate


def fit_source(runtime, action):
    import torch
    from experiments.cric_dura_comparison.bootstrap import numerical_policy
    from experiments.pamf_lora_main import channel_blind_source, dura_w24_model, dura_w24_train

    rt = read(runtime)
    if set(rt) != {'source_root', 'run_root'}:
        raise ValueError('Fit may receive only the source IQ and its output root')

    def prepare(config, device, *, smoke, fit_per_rx, bank_per_rx):
        source = Path(config['source_root'])
        contract = dura_w24_train.source_contract(source)
        iq, rx, _, _ = dura_w24_train.source_role(source, contract, 'U_s')
        if smoke:
            training, _, held = channel_blind_source.partition(rx, fit=5, cal=4, held=3)
        else:
            training, _, held = channel_blind_source.partition(rx, fit=fit_per_rx)
        start, end = (tensor(iq[index].view('float32'), device).view(torch.complex64) for index in (training, held))
        return (start, torch.empty(0, device=device), end), contract

    numerical_policy()
    channel_blind_source.fit(rt, smoke=(action == 'fit-smoke'), recipe=recipe_for(dura_w24_train),
                             calculate=objective_for(dura_w24_model), receiver_batches=True, prepare=prepare,
                             make_tracker=dura_w24_train.WideTracker)


def load_candidate(root, contract, device):
    import torch

    training = importlib.import_module('lora_wide_dura_archive.dura_w24_train')
    source = importlib.import_module('lora_wide_dura_archive.channel_blind_source')
    recipe = recipe_for(training)
    summary = read(root / f'{NAME}-complete.json')
    expected = source.proof(contract, recipe)
    public = {k: v for k, v in expected.items() if k != 'contract'}
    if (any(summary.get(k) != v for k, v in public.items()) or expected['steps'] != STEPS
            or expected['seed'] != SEED or summary.get('input_gate') is not False
            or summary.get('threshold') is not None or summary.get('n_calibration') != 0):
        raise ValueError('Expected the source-frozen no-gate outer-chirp candidate')
    checkpoint = torch.load(root / f'{NAME}-model.pt', map_location=device, weights_only=True)
    if checkpoint['proof'] != expected or checkpoint['threshold'] is not None:
        raise ValueError('Candidate weights do not satisfy the registered source contract')
    model = recipe.model().to(device).eval().requires_grad_(False)
    model.load_state_dict(checkpoint['weights'], strict=True)
    return model, summary


def dispatch(runtime):
    from experiments.pamf_lora_main.runner import available
    from experiments.pamf_lora_main.w24_native_full import W24_COMMIT
    from experiments.pamf_lora_main.w24_wide_transfer import training_runtime

    rt = read(runtime)
    archive, _ = training_runtime(rt)
    reference = Path(rt['w24_release_root'])
    if read(reference / 'submit.json')['code_commit'] != W24_COMMIT:
        raise ValueError('Expected the registered W24 reference release')
    root = Path(rt['run_root'])
    root.mkdir()
    cfg = root / 'fit-inputs.json'
    write(cfg, {k: rt[k] for k in ('source_root', 'run_root')})
    state = root / 'state.json'

    def eligible_gpu():
        deadline = monotonic() + 1800
        while not (cards := available(rt['allowed_gpus'], max_compute_processes=8)):
            if monotonic() >= deadline:
                raise ValueError('GPU wait expired')
            sleep(20)
        return next(i for i in rt['allowed_gpus'] if i in cards)

    def launch(action, cwd, script, config, gpu):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='' if gpu is None else str(gpu), PYTHONPATH=str(cwd),
                   OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
        with (root / f'{action}.log').open('x', encoding='utf-8') as stream:
            job = subprocess.Popen([rt['python'], '-u', str(script), action, str(config)], cwd=cwd,
                                   env=env, stdout=stream, stderr=subprocess.STDOUT)
            write(state, dict(status='RUNNING', phase=action, pid=job.pid, gpu=gpu), exclusive=False)
            if job.wait():
                raise ValueError(f'W24 outer-chirp {action} failed')

    own = Path(__file__).resolve()
    evaluator = own.with_name('w24_native_residual_source.py')
    try:
        write(state, dict(status='WAIT_GPU'))
        gpu = eligible_gpu()
        launch('fit-smoke', archive, own, cfg, gpu)
        launch('fit', archive, own, cfg, eligible_gpu())
        launch('predict-source', reference, evaluator, runtime, eligible_gpu())
        launch('score-source', reference, evaluator, runtime, None)
        write(state, dict(status='SOURCE_SCORED'), exclusive=False)
    except Exception as error:
        write(state, dict(status='FAILED', error=str(error)), exclusive=False)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    for action in ('dispatch', 'fit-smoke', 'fit'):
        sub.add_parser(action).add_argument('runtime', type=Path)
    pub = sub.add_parser('publish')
    for key in ('repo', 'output', 'context', 'ssh-config'):
        pub.add_argument('--' + key, type=Path, required=True)
    args = parser.parse_args()
    if args.action == 'publish':
        from experiments.pamf_lora_main.publish import release, release_commit
        from experiments.pamf_lora_main.w24_factor_audit import AUG_RUN
        from experiments.pamf_lora_main.w24_native_full import DATA_RUN, W24_RUN
        from experiments.pamf_lora_main.w24_native_residual_source import SWEEP_RUN
        from experiments.pamf_lora_main.w24_wide_transfer import TRAIN_RUN

        ctx = read(args.context)
        rt = {key: ctx[key] for key in ('project', 'python', 'frozen_sources')}
        project = PurePosixPath(rt['project'])
        releases, runs = project / 'releases', project / 'runs'
        rt.update(run_id=RUN, runner=__spec__.name, code_commit=release_commit(args.repo),
                  release_root=str(releases / f'{RUN}-release'), run_root=str(runs / RUN),
                  source_root=str(releases / f'{DATA_RUN}-release' / 'data/source'),
                  classifier_root=str(runs / AUG_RUN), w24_release_root=str(releases / f'{W24_RUN}-release'),
                  training_release_root=str(releases / f'{TRAIN_RUN}-release'), outer_objective=True,
                  allowed_gpus=[6, 3, 2, 7, 4, 1, 0, 5], cpu_workers=8)
        rt['external_releases'] = [rt[k] for k in ('source_root', 'classifier_root', 'w24_release_root',
                                                  'training_release_root')] + [str(runs / SWEEP_RUN)]
        release(args.repo, args.output, rt, args.ssh_config)
    elif args.action == 'dispatch':
        dispatch(args.runtime)
    else:
        fit_source(args.runtime, args.action)


if __name__ == '__main__':
    main()
