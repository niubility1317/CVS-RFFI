"""A matched WiSig source comparison with received-amplitude protection in both stages."""

import argparse
from functools import partial
from pathlib import Path

import torch

from experiments.cric_dura_comparison.io import read, write
from experiments.pamf_lora_main.packet_runner import schedule
from . import observable_action_source as action
from . import receiver_pilot_source as panel
from .observable_action import ObservableAction


RUN = '20261123-phase1-wisig-observed-amplitude-source-m3-r01'
MODES = ('fitted', 'fitted_guarded', 'protected', 'protected_guarded')
PAIRS = (('protected', 'fitted'), ('protected_guarded', 'fitted_guarded'))
MODEL = partial(ObservableAction, fitted=True, protect_amplitude=True)


def parent(rt):
    path = Path(rt['run_root']).parent / action.RUN
    if read(path / 'state.json')['status'] != 'SOURCE_SCORED':
        raise ValueError('Expected completed frozen source waveform control')
    return path


def evaluate(rt, route):
    root, device = Path(rt['run_root']), 'cuda:0'
    frozen = parent(rt)
    models = dict(fitted=action.load_arm(frozen, 'fitted', device),
                  protected=action.load_arm(root, 'protected', device, model_factory=MODEL, run=RUN))
    guard, limit = action.guard_for(frozen, device)

    def restore(y):
        active = (guard.measures(y) > limit).any(dim=1)
        variants = {}
        diag = {'gate_active': active.float()}
        for name, model in models.items():
            _, output, info = model(y)
            variants[name] = output
            variants[f'{name}_guarded'] = torch.where(active[:, None], output, y)
            diag.update({f'{name}_{k}': v for k, v in info.items()})
        return variants, diag

    panel.predict(rt, route, alternatives=MODES, restore=restore, diagnose=action.spectral_evidence)


def dispatch(rt):
    root = Path(rt['run_root'])
    root.mkdir()
    (root / 'logs').mkdir()
    launch = partial(panel.launch, module=rt['runner'])
    try:
        for stage, names in (('checkpoint_smoke', ('source',)), ('smoke', ('protected',)),
                             ('fit', ('protected',)), ('timing', ('source',)), ('predict', panel.ROUTES)):
            schedule(rt, [(stage, name) for name in names], launch)
        if launch(rt, 'cpu_score', 'source').wait():
            raise RuntimeError('WiSig amplitude protection scoring failed')
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
    for stage in ('dispatch', 'checkpoint_smoke', 'smoke', 'fit', 'timing', 'predict', 'cpu_score'):
        cmd = commands.add_parser(stage)
        cmd.add_argument('runtime', type=Path)
        if stage != 'dispatch':
            choices = ('protected',) if stage in ('smoke', 'fit') else panel.ROUTES if stage == 'predict' else ('source',)
            cmd.add_argument('name', choices=choices)
    args = vars(parser.parse_args())
    stage = args.pop('action')
    if stage == 'publish':
        panel.publish(**args, run_id=RUN, runner=__spec__.name)
        return
    rt = read(args['runtime'])
    if stage == 'dispatch':
        dispatch(rt)
    elif stage == 'cpu_score':
        meta = read(Path(rt['run_root']) / 'protected-complete.json')
        panel.cpu_score(rt, MODES, pairs=PAIRS, trainable_parameters=meta['parameter_count'])
    elif stage in ('smoke', 'fit'):
        action.fit(rt, 'protected', smoke=stage == 'smoke', model_factory=MODEL, run=RUN)
    elif stage == 'timing':
        action.measure(rt, name='protected', model_factory=MODEL, run=RUN, guard_root=parent(rt))
    else:
        panel.activate(rt)
        if stage == 'checkpoint_smoke':
            action.checkpoint_smoke(rt)
        else:
            evaluate(rt, args['name'])


if __name__ == '__main__':
    main()
