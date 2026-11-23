"""Compare a fitted WiSig observation action with a capacity-matched identity arm."""

import argparse
from functools import partial
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from experiments.cric_dura_comparison.io import read, tensor, write
from experiments.pamf_lora_main.packet_runner import schedule
from . import receiver_pilot_source as panel
from .dura_baseline import SEEDS, historical, model_pair
from .observable_action import ObservableAction, generic_observation, objective
from .receiver_pilot import estimate
from .receiver_timing import Timing


RUN = '20261123-phase1-wisig-observable-action-source-m4-r01'
SEED = 2026112321
BATCH, STEPS = 48, 1200
COUNT, CALIBRATION, HOLD = 56700, 4096, 768
QUANTILE = .999
ARMS = {'identity': False, 'fitted': True}
READOUTS = ('identity', 'identity_guarded', 'fitted', 'fitted_guarded')


def source_u(rt):
    return panel.iq_array(panel.source_root(rt) / 'U_s_x.npy', COUNT)


def partition():
    selected = np.random.default_rng(SEED).permutation(COUNT)
    return selected[CALIBRATION + HOLD:], selected[:CALIBRATION], selected[CALIBRATION:CALIBRATION + HOLD]


def received(values, device):
    iq = tensor(np.ascontiguousarray(values), device)
    return torch.complex(iq[:, 0], iq[:, 1])


def proof(name, run=RUN):
    return dict(run_id=run, arm=name, steps=STEPS, batch=BATCH, seed=SEED,
                target_reads=0, source_truth_reads=0, recognizer_reads=0,
                source_V_reads=0, practical_training_reads=0)


@torch.no_grad()
def checkpoint_smoke(rt):
    cfg = historical(rt, 'predict', SEEDS[0])
    reader, baseline = model_pair(rt, cfg, 'cuda:0')
    source = received(source_u(rt)[:4], 'cuda:0')
    logits, _ = panel.read_logits(reader, baseline, panel.as_iq(source), {})
    if any(value.shape != (4, 6) or not torch.isfinite(value).all() for value in logits.values()):
        raise ValueError('Frozen WiSig checkpoint smoke failed on source U')
    write(Path(rt['run_root']) / 'checkpoint_smoke.json', dict(status='PASS', source_U=4, target_reads=0))
    print('WISIG_ACTION_REAL_CHECKPOINT_PASS', flush=True)


def fit(rt, name, *, smoke=False, model_factory=None, run=RUN, criterion=objective,
        mixed_precision=True):
    device = 'cuda:0'
    torch.manual_seed(SEED)
    model = (ObservableAction(ARMS[name]) if model_factory is None else model_factory()).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    schedule_lr = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, STEPS, eta_min=3e-5)
    rd = np.random.default_rng(SEED + 1)
    generator = torch.Generator(device=device).manual_seed(SEED + 2)
    source = source_u(rt)
    fit_ids, _, hold = partition()
    start = perf_counter()
    for step in range(1, 2 if smoke else STEPS + 1):
        sample = received(source[rd.choice(fit_ids, BATCH, replace=False)], device)
        optimizer.zero_grad(set_to_none=True)
        loss, parts = criterion(model, sample, generator)
        if not torch.isfinite(parts).all():
            raise FloatingPointError('Unstable source-only WiSig action objective')
        loss.backward()
        if step == 1 and any(head.out.weight.grad.norm() <= 0 for head in (model.coarse, model.fine)):
            raise ValueError('Expected source gradient in both restoration stages')
        grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        schedule_lr.step()
        if step == 1 or step % 50 == 0:
            print('WISIG_ACTION_FIT', name, step, *parts.detach().tolist(), float(grad), flush=True)
    torch.cuda.synchronize()
    seconds = perf_counter() - start
    root = Path(rt['run_root'])
    if smoke:
        write(root / f'smoke-{name}.json', dict(status='PASS', source_U=BATCH))
        return
    model.eval()
    with torch.no_grad():
        hold_rng = torch.Generator(device=device).manual_seed(SEED + 3)
        held = torch.stack([criterion(model, received(source[ids], device), hold_rng)[1]
                            for ids in np.split(hold, HOLD // BATCH)]).mean(dim=0)
    report = dict(proof(name, run), status='SOURCE_FROZEN', held=held.tolist(), worker_seconds=seconds,
                  parameter_count=sum(p.numel() for p in model.parameters()),
                  n_fit=len(fit_ids), n_hold=HOLD, bf16_training=mixed_precision and torch.cuda.is_bf16_supported())
    with (root / f'{name}.pt').open('xb') as f:
        torch.save(dict(meta=report, weights=model.state_dict()), f)
    write(root / f'{name}-complete.json', report)


class Guard:
    def __init__(self, device):
        self.timing = Timing(device)

    def measures(self, y):
        obs = estimate(y)
        _, margin = self.timing.choose(self.timing.boundary_score(obs.aligned))
        return torch.stack((margin, obs.disturbance), dim=1)


@torch.no_grad()
def calibrate(rt):
    source = source_u(rt)
    _, ids, _ = partition()
    guard = Guard('cpu')
    measured = torch.cat([guard.measures(received(source[part], 'cpu'))
                          for part in np.split(ids, CALIBRATION // 64)])
    if not torch.isfinite(measured).all():
        raise ValueError('Nonfinite source U guard')
    limits = measured.quantile(QUANTILE, dim=0).tolist()
    write(Path(rt['run_root']) / 'guard.json', dict(source_U=CALIBRATION, seed=SEED,
          quantile=QUANTILE, limits=limits))
    print('WISIG_ACTION_SOURCE_GUARD', limits, flush=True)


def load_arm(root, name, device, *, model_factory=None, run=RUN):
    state = torch.load(root / f'{name}.pt', map_location=device, weights_only=True)
    if state['meta'] != read(root / f'{name}-complete.json') or any(
            state['meta'][k] != v for k, v in proof(name, run).items()):
        raise ValueError('Expected immutable WiSig source fit')
    model = (ObservableAction(ARMS[name]) if model_factory is None else model_factory())
    model = model.to(device).eval().requires_grad_(False)
    model.load_state_dict(state['weights'], strict=True)
    return model


def guard_for(root, device):
    frozen = read(root / 'guard.json')
    if (frozen['source_U'], frozen['seed'], frozen['quantile']) != (CALIBRATION, SEED, QUANTILE):
        raise ValueError('Expected WiSig source U calibration')
    return Guard(device), torch.tensor(frozen['limits'], device=device)


def spectral_evidence(waves, clean):
    from .receiver_pilot import power

    def spectrum(iq):
        return torch.fft.fft(torch.complex(iq[:, 0], iq[:, 1]), norm='ortho')

    reference = spectrum(clean)
    frequency = torch.fft.fftfreq(reference.shape[-1], device=reference.device).abs()
    result = {}
    for name, waveform in waves.items():
        value = spectrum(waveform)
        for band, region in (('low', frequency <= .125),
                             ('mid', (frequency > .125) & (frequency <= .25)),
                             ('high', frequency > .25)):
            p, x = value[:, region], reference[:, region]
            target_power = power(x).sum(1)
            magnitude = (p.abs() - x.abs()).square().sum(1) / target_power.clamp_min(1e-8)
            coherence = (p * x.conj()).real.sum(1) / (power(p).sum(1) * target_power).clamp_min(1e-12).sqrt()
            result[f'{name}_{band}_magnitude_nmse'] = magnitude
            result[f'{name}_{band}_real_correlation'] = coherence
    return result


def evaluate(rt, route):
    device = 'cuda:0'
    root = Path(rt['run_root'])
    models = {name: load_arm(root, name, device) for name in ARMS}
    guard, limit = guard_for(root, device)

    def restore(y):
        measures = guard.measures(y)
        active = (measures > limit).any(dim=1)
        outputs = {}
        diag = dict(boundary_margin=measures[:, 0], repeat_disturbance=measures[:, 1])
        for name, model in models.items():
            _, refined, info = model(y)
            outputs[name] = refined
            outputs[f'{name}_guarded'] = torch.where(active[:, None], refined, y)
            diag.update({f'{name}_{k}': v for k, v in info.items()})
        return outputs, diag

    panel.predict(rt, route, alternatives=READOUTS, restore=restore, diagnose=spectral_evidence)


@torch.no_grad()
def measure(rt, *, name='fitted', model_factory=None, run=RUN, guard_root=None):
    device = 'cuda:0'
    root = Path(rt['run_root'])
    model = load_arm(root, name, device, model_factory=model_factory, run=run)
    guard, limit = guard_for(root if guard_root is None else guard_root, device)
    source = received(source_u(rt)[:32], device)
    perturbed = generic_observation(source[:, None], torch.Generator(device=device).manual_seed(SEED + 8))[:, 0]
    margin = (guard.measures(perturbed) - limit).amax(dim=1)
    metrics = {}
    for mode, sample in (('clean', source[:1]), ('generic_disturbed', perturbed[margin.argmax().view(1)])):
        times = []
        for i in range(32):
            torch.cuda.synchronize()
            start = perf_counter()
            active = bool((guard.measures(sample) > limit).any())
            output = model(sample)[1] if active else sample
            torch.cuda.synchronize()
            if i >= 8:
                times.append((perf_counter() - start) * 1000)
        if not torch.isfinite(output).all():
            raise FloatingPointError('Invalid frozen timing output')
        metrics[mode] = dict(median_ms=float(np.median(times)), active=active)
    write(root / 'b1.json', dict(timing=metrics, gpu=torch.cuda.get_device_name(),
          n_measurements=24, parameters=sum(p.numel() for p in model.parameters())))


def dispatch(rt):
    root = Path(rt['run_root'])
    root.mkdir()
    (root / 'logs').mkdir()
    launch = partial(panel.launch, module=rt['runner'])
    try:
        schedule(rt, [('checkpoint_smoke', 'source')], launch)
        if launch(rt, 'calibrate', 'source').wait():
            raise RuntimeError('WiSig U-only guard calibration failed')
        for kind, names in (('smoke', ARMS), ('fit', ARMS), ('timing', ('source',)), ('predict', panel.ROUTES)):
            schedule(rt, [(kind, name) for name in names], launch)
        if launch(rt, 'cpu_score', 'source').wait():
            raise RuntimeError('WiSig source truth-last scorer failed')
        write(root / 'state.json', dict(status='SOURCE_SCORED'))
    except Exception as exc:
        write(root / 'state.json', dict(status='FAILED', error=str(exc)))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='action', required=True)
    pub = commands.add_parser('publish')
    for name in ('repo', 'output', 'local-context', 'ssh-config'):
        pub.add_argument('--' + name, type=Path, required=True)
    for action in ('dispatch', 'checkpoint_smoke', 'calibrate', 'smoke', 'fit', 'timing', 'predict', 'cpu_score'):
        command = commands.add_parser(action)
        command.add_argument('runtime', type=Path)
        if action != 'dispatch':
            choices = ARMS if action in ('smoke', 'fit') else panel.ROUTES if action == 'predict' else ('source',)
            command.add_argument('name', choices=choices)
    args = vars(parser.parse_args())
    action = args.pop('action')
    if action == 'publish':
        panel.publish(**args, run_id=RUN, runner=__spec__.name)
        return
    rt = read(args['runtime'])
    if action == 'dispatch':
        dispatch(rt)
    elif action == 'cpu_score':
        info = read(Path(rt['run_root']) / 'fitted-complete.json')
        panel.cpu_score(rt, READOUTS, pairs=(('fitted', 'identity'), ('fitted_guarded', 'identity_guarded')),
                        trainable_parameters=info['parameter_count'])
    elif action == 'calibrate':
        calibrate(rt)
    elif action in ('smoke', 'fit'):
        fit(rt, args['name'], smoke=action == 'smoke')
    elif action == 'timing':
        measure(rt)
    else:
        panel.activate(rt)
        if action == 'checkpoint_smoke':
            checkpoint_smoke(rt)
        else:
            evaluate(rt, args['name'])


if __name__ == '__main__':
    main()
