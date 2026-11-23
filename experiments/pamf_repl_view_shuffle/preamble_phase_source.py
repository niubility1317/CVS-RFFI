"""WiSig source-only phase calibration and matched scratch two-stage recovery."""

import argparse
from functools import partial
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from experiments.cric_dura_comparison.io import read, write
from experiments.pamf_lora_main.packet_runner import schedule
from . import observable_action_source as action
from . import receiver_pilot_source as panel
from .preamble_phase import MIN_AGREEMENT, MIN_CORRELATION, QUANTILE, ScratchUnroll, ShortFieldPhase
from .receiver_pilot import phase
from .observable_action import generic_observation


RUN = '20261124-phase1-wisig-preamble-phase-source-m4-r01'
ARMS = ('scratch', 'scratch_phase')
MODES = ('phase', 'phase_guarded', 'scratch', 'scratch_guarded',
         'scratch_phase', 'scratch_phase_guarded')
PAIRS = (('scratch_phase', 'scratch'), ('scratch_phase_guarded', 'scratch_guarded'),
         ('scratch_phase_guarded', 'phase_guarded'))


@torch.no_grad()
def calibrate(rt):
    panel.activate(rt)
    source = action.source_u(rt)
    _, ids, _ = action.partition()
    gauge = ShortFieldPhase()
    batches = [gauge.measure(action.received(source[ix], 'cpu')) for ix in np.split(ids, 64)]
    angles, fit, agree, _ = [torch.cat(column) for column in zip(*batches)]
    good = (fit >= MIN_CORRELATION) & (agree >= MIN_AGREEMENT)
    unit = phase(angles[good]).mean()
    if int(good.sum()) < len(ids) // 2 or not torch.isfinite(unit) or unit.abs() < .5:
        raise ValueError('Unstable source U STF reference')
    center = unit.angle()
    errors = phase(angles[good] - center).angle().abs()
    limit = errors.quantile(QUANTILE)
    write(Path(rt['run_root']) / 'phase-calibration.json', dict(
        source_U=len(ids), seed=action.SEED, quantile=QUANTILE, sufficient=int(good.sum()),
        circular_resultant=unit.abs().item(), center_rad=center.item(), limit_rad=limit.item(),
        absolute_error_rad=errors.quantile(torch.tensor([.1,.5,.9,.99])).tolist()))
    print('WISIG_PHASE_U_FROZEN', float(center), float(limit), int(good.sum()), flush=True)


def settings(root):
    proof = read(root / 'phase-calibration.json')
    if (proof['source_U'], proof['seed'], proof['quantile']) != (action.CALIBRATION, action.SEED, QUANTILE):
        raise ValueError('Expected frozen source U phase calibration')
    return proof['center_rad'], proof['limit_rad']


def factory(root, name):
    center, limit = settings(root)
    return partial(ScratchUnroll, canonical=name == 'scratch_phase', center=center, limit=limit)


def prior_guard(rt, device):
    root = Path(rt['run_root']).parent / action.RUN
    if read(root / 'state.json')['status'] != 'SOURCE_SCORED':
        raise ValueError('Expected frozen source engineering bypass')
    return action.guard_for(root, device)


def evaluate(rt, route):
    panel.activate(rt)
    root, device = Path(rt['run_root']), 'cuda:0'
    guard, threshold = prior_guard(rt, device)
    gauge = ShortFieldPhase().to(device)
    center, limit = settings(root)
    models = {name: action.load_arm(root, name, device, model_factory=factory(root, name), run=RUN)
              for name in ARMS}

    def restore(y):
        rotation, rotate, telemetry = gauge.rotation(y, center, limit)
        correction = y * rotation[:, None]
        readouts = {'phase': correction, 'phase_guarded': torch.where(rotate[:, None], correction, y)}
        active = rotate | (guard.measures(y) > threshold).any(dim=1)
        telemetry['model_active'] = active.float()
        for name, model in models.items():
            _, value, info = model(y)
            readouts[name] = value
            readouts[f'{name}_guarded'] = torch.where(active[:, None], value, y)
            telemetry[f'{name}_map_variance'] = info['map_variance']
        return readouts, telemetry

    panel.predict(rt, route, alternatives=MODES, restore=restore, diagnose=action.spectral_evidence)


@torch.no_grad()
def timing(rt, name):
    device = 'cuda:0'
    root = Path(rt['run_root'])
    guard, thresholds = prior_guard(rt, device)
    center, limit = settings(root)
    model = action.load_arm(root, name, device, model_factory=factory(root, name), run=RUN)
    source = action.received(action.source_u(rt)[:32], device)
    generator = torch.Generator(device=device).manual_seed(action.SEED + 8)
    perturbed = generic_observation(source[:, None], generator)[:, 0]
    margin = (guard.measures(perturbed) - thresholds).amax(dim=1)
    metrics = {}
    for kind, sample in (('clean', source[:1]), ('generic_disturbed', perturbed[margin.argmax().view(1)])):
        times = []
        for n in range(32):
            torch.cuda.synchronize()
            start = perf_counter()
            _, rotate, _ = model.gauge.rotation(sample, center, limit)
            active = bool(rotate.any() | (guard.measures(sample) > thresholds).any())
            value = model(sample)[1] if active else sample
            torch.cuda.synchronize()
            if n >= 8:
                times.append((perf_counter() - start) * 1000)
        if not torch.isfinite(value).all():
            raise FloatingPointError('Invalid WiSig phase timing output')
        metrics[kind] = dict(median_ms=float(np.median(times)), active=active)
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
            raise RuntimeError('Source U phase calibration failed')
        schedule(rt, [('smoke', name) for name in ARMS], launch)
        schedule(rt, [('fit', name) for name in ARMS], launch)
        schedule(rt, [('timing', 'scratch_phase')], launch)
        schedule(rt, [('predict', route) for route in panel.ROUTES], launch)
        if launch(rt, 'cpu_score', 'source').wait():
            raise RuntimeError('WiSig phase source scoring failed')
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
    for stage in ('dispatch', 'checkpoint_smoke', 'calibrate', 'smoke', 'fit',
                  'timing', 'predict', 'cpu_score'):
        cmd = commands.add_parser(stage)
        cmd.add_argument('runtime', type=Path)
        if stage != 'dispatch':
            choices = ARMS if stage in ('smoke', 'fit', 'timing') else panel.ROUTES if stage == 'predict' else ('source',)
            cmd.add_argument('name', choices=choices)
    args = vars(parser.parse_args())
    stage = args.pop('action')
    if stage == 'publish':
        panel.publish(**args, run_id=RUN, runner=__spec__.name)
        return
    rt = read(args['runtime'])
    root = Path(rt['run_root'])
    if stage == 'dispatch':
        dispatch(rt)
    elif stage == 'checkpoint_smoke':
        panel.activate(rt)
        action.checkpoint_smoke(rt)
    elif stage == 'calibrate':
        calibrate(rt)
    elif stage == 'cpu_score':
        meta = read(root / 'scratch_phase-complete.json')
        panel.cpu_score(rt, MODES, pairs=PAIRS, trainable_parameters=meta['parameter_count'])
    elif stage in ('smoke', 'fit'):
        name = args['name']
        action.fit(rt, name, smoke=stage == 'smoke', model_factory=factory(root, name), run=RUN,
                   mixed_precision=False)
    elif stage == 'timing':
        timing(rt, args['name'])
    else:
        evaluate(rt, args['name'])


if __name__ == '__main__':
    main()
