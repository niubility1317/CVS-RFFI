"""Score observation-anchored Fourier readouts and separate paired-reference diagnostics."""

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
from .phase_magnitude import candidates, paired_reference_diagnostic, weighted_phase
from .receiver_pilot import estimate


RUN = '20261123-phase1-wisig-phase-magnitude-source-m4-r01'
MODES = ('fitted', 'fitted_guarded', 'fit_rms', 'fit_phase', 'fit_magnitude',
         'weighted_phase', 'fit_phase_guarded', 'weighted_phase_guarded')
DIAGNOSTICS = ('reference_phase', 'reference_magnitude', 'paired_clean')
PAIRS = (('fit_rms', 'fitted'), ('fit_phase', 'fitted'), ('fit_magnitude', 'fitted'),
         ('weighted_phase', 'fit_phase'), ('fit_phase_guarded', 'fitted_guarded'),
         ('weighted_phase_guarded', 'fitted_guarded'))


def parent(rt):
    original = Path(rt['run_root']).parent / action.RUN
    if read(original / 'state.json')['status'] != 'SOURCE_SCORED':
        raise ValueError('Expected closed source-only parent')
    return original


def oracle_rt(rt):
    return dict(rt, run_root=str(Path(rt['run_root']) / 'paired_reference_diagnostic'))


def evaluate(rt, route):
    root, device = parent(rt), 'cuda:0'
    model = action.load_arm(root, 'fitted', device)
    guard, limit = action.guard_for(root, device)

    def restore(y):
        noise = estimate(y).noise_power
        active = (guard.measures(y) > limit).any(dim=1)
        _, fitted, _ = model(y)
        restored = dict(fitted=fitted, **candidates(y, fitted, noise))
        for name in ('fitted', 'fit_phase', 'weighted_phase'):
            restored[f'{name}_guarded'] = torch.where(active[:, None], restored[name], y)
        return restored, {'effective_noise': noise, 'gate_active': active.float()}

    panel.predict(rt, route, alternatives=MODES, restore=restore, diagnose=action.spectral_evidence)


@torch.no_grad()
def measure(rt):
    device = 'cuda:0'
    root = parent(rt)
    model = action.load_arm(root, 'fitted', device)
    guard, limit = action.guard_for(root, device)
    sample = action.received(action.source_u(rt)[:32], device)
    generator = torch.Generator(device=device).manual_seed(action.SEED + 11)
    corrupted = action.generic_observation(sample[:, None], generator)[:, 0]
    priority = (guard.measures(corrupted) - limit).amax(dim=1)
    measurements = {}
    for label, y in (('source_clean', sample[:1]), ('generic_disturbed', corrupted[priority.argmax().view(1)])):
        elapsed = []
        for i in range(32):
            torch.cuda.synchronize()
            start = perf_counter()
            active = bool((guard.measures(y) > limit).any())
            if active:
                _, fitted, _ = model(y)
                result = weighted_phase(y, fitted, estimate(y).noise_power)
            else:
                result = y
            torch.cuda.synchronize()
            if i >= 8:
                elapsed.append(1000 * (perf_counter() - start))
        if not torch.isfinite(result).all():
            raise FloatingPointError('Invalid guarded phase readout')
        measurements[label] = dict(median_ms=float(np.median(elapsed)), active=active)
    write(Path(rt['run_root']) / 'b1.json', dict(timing=measurements, n_measurements=24,
          gpu=torch.cuda.get_device_name(), parameters=sum(p.numel() for p in model.parameters())))


def score(rt):
    dimensions = (('phase_magnitude_interaction',
                  (('fitted', 1), ('raw', 1), ('fit_phase', -1), ('fit_magnitude', -1))),)
    isolated = oracle_rt(rt)
    for route in panel.ROUTES:
        ids = []
        for runtime, modes in ((rt, MODES), (isolated, DIAGNOSTICS)):
            actual, _, _ = panel.load_predictions(runtime, route, ('raw', 'R69', *modes))
            ids.append(actual)
        if not np.array_equal(*ids):
            raise ValueError('Candidate and isolated diagnostic packets differ')
        for runtime, expected in ((rt, False), (isolated, True)):
            meta = read(Path(runtime['run_root']) / route / 'complete.json')
            if meta['diagnostic_reference_input'] != expected:
                raise ValueError('Unexpected source reference attribution')
    meta = read(parent(rt) / 'fitted-complete.json')
    panel.cpu_score(rt, MODES, pairs=PAIRS, trainable_parameters=meta['parameter_count'], contrasts=dimensions)
    panel.cpu_score(isolated, DIAGNOSTICS)


def dispatch(rt):
    root = Path(rt['run_root'])
    root.mkdir()
    (root / 'logs').mkdir()
    Path(oracle_rt(rt)['run_root']).mkdir()
    launch = partial(panel.launch, module=rt['runner'])
    try:
        for stage, names in (('checkpoint_smoke', ('source',)), ('timing', ('source',)),
                             ('predict', panel.ROUTES), ('diagnostic_predict', panel.ROUTES)):
            schedule(rt, [(stage, name) for name in names], launch)
        if launch(rt, 'cpu_score', 'source').wait():
            raise RuntimeError('WiSig spectral split scoring failed')
        write(root / 'state.json', dict(status='SOURCE_SCORED'))
    except Exception as exc:
        write(root / 'state.json', dict(status='FAILED', error=str(exc)))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    pub = sub.add_parser('publish')
    for k in ('repo', 'output', 'local-context', 'ssh-config'):
        pub.add_argument('--' + k, type=Path, required=True)
    for name in ('dispatch', 'checkpoint_smoke', 'timing', 'predict', 'diagnostic_predict', 'cpu_score'):
        stage = sub.add_parser(name)
        stage.add_argument('runtime', type=Path)
        if name != 'dispatch':
            stage.add_argument('name', choices=panel.ROUTES if name.endswith('predict') else ('source',))
    args = vars(parser.parse_args())
    mode = args.pop('action')
    if mode == 'publish':
        panel.publish(**args, run_id=RUN, runner=__spec__.name)
        return
    rt = read(args['runtime'])
    if mode == 'dispatch':
        dispatch(rt)
    elif mode == 'cpu_score':
        score(rt)
    elif mode == 'timing':
        measure(rt)
    else:
        panel.activate(rt)
        if mode == 'checkpoint_smoke':
            action.checkpoint_smoke(rt)
        elif mode == 'predict':
            evaluate(rt, args['name'])
        else:
            panel.predict(oracle_rt(rt), args['name'], alternatives=DIAGNOSTICS,
                          diagnostic_restore=paired_reference_diagnostic, diagnose=action.spectral_evidence)


if __name__ == '__main__':
    main()
