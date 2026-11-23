"""WiSig Main source-only matched evaluation of measured packet geometry."""

import argparse
from functools import partial
from pathlib import Path

import torch

from experiments.cric_dura_comparison.io import read, write
from experiments.pamf_lora_main.packet_runner import schedule
from . import observable_action_source as action
from . import observed_amplitude_source as control
from . import receiver_pilot_source as panel
from .instance_geometry import distances, pair_loss, relation_loss
from .observable_action import generic_observation, objective


RUN = '20261124-phase1-wisig-instance-geometry-source-m3-r01'
MODES = ('control', 'control_guarded', 'geometry', 'geometry_guarded')
PAIRS = (('geometry', 'control'), ('geometry_guarded', 'control_guarded'))
FIT = partial(action.fit, model_factory=control.MODEL, run=RUN,
              criterion=partial(objective, relation=relation_loss))


def models(rt, device):
    root = Path(rt['run_root'])
    parent = root.parent / control.RUN
    if read(parent / 'state.json')['status'] != 'SOURCE_SCORED':
        raise ValueError('Expected completed matched received-amplitude control')
    return {
        'control': action.load_arm(parent, 'protected', device, model_factory=control.MODEL,
                                   run=control.RUN),
        'geometry': action.load_arm(root, 'geometry', device, model_factory=control.MODEL, run=RUN),
    }


@torch.no_grad()
def held_metrics(rt):
    panel.activate(rt)
    device = 'cuda:0'
    frozen = models(rt, device)
    source = action.source_u(rt)
    _, _, held_ids = action.partition()
    generator = torch.Generator(device=device).manual_seed(action.SEED + 403)
    totals = {name: dict(correct=0, count=0, point=0., loss=0.)
              for name in ('received', *frozen)}
    for start in range(0, len(held_ids), 48):
        clean = action.received(source[held_ids[start:start + 48]], device)
        observed = torch.cat(tuple(generic_observation(clean[:, None], generator)
                                   for _ in range(2)), dim=1)
        outputs = {'received': observed}
        for name, model in frozen.items():
            _, restored, _ = model(observed.flatten(0, 1))
            outputs[name] = restored.unflatten(0, observed.shape[:2])
        for name, output in outputs.items():
            predicted, baseline = distances(output, clean)
            row = totals[name]
            count = 2 * len(clean)
            row['count'] += count
            row['correct'] += (predicted.argmin(dim=-1) == torch.arange(len(clean), device=device)).sum().item()
            row['point'] += predicted.diagonal(dim1=-2, dim2=-1).sum().item()
            row['loss'] += count * pair_loss(predicted, baseline).item()
    for row in totals.values():
        row['point'] /= row['count']
        row['loss'] /= row['count']
    write(Path(rt['run_root']) / 'held_geometry.json', dict(packets=len(held_ids), metrics=totals))


def evaluate(rt, route):
    panel.activate(rt)
    device = 'cuda:0'
    frozen = models(rt, device)
    guard, limit = action.guard_for(control.parent(rt), device)

    def restore(y):
        active = (guard.measures(y) > limit).any(dim=1)
        variants = {}
        diag = {'gate_active': active.float()}
        for name, model in frozen.items():
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
        schedule(rt, [('checkpoint_smoke', 'source')], launch)
        schedule(rt, [('smoke', 'geometry')], launch)
        schedule(rt, [('fit', 'geometry')], launch)
        schedule(rt, [('held_metrics', 'source'), ('timing', 'source')], launch)
        schedule(rt, [('predict', route) for route in panel.ROUTES], launch)
        if launch(rt, 'cpu_score', 'source').wait():
            raise RuntimeError('WiSig packet geometry scoring failed')
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
    for stage in ('dispatch', 'checkpoint_smoke', 'smoke', 'fit', 'held_metrics',
                  'timing', 'predict', 'cpu_score'):
        cmd = commands.add_parser(stage)
        cmd.add_argument('runtime', type=Path)
        if stage != 'dispatch':
            choices = ('geometry',) if stage in ('smoke', 'fit') else panel.ROUTES if stage == 'predict' else ('source',)
            cmd.add_argument('name', choices=choices)
    args = vars(parser.parse_args())
    stage = args.pop('action')
    if stage == 'publish':
        panel.publish(**args, run_id=RUN, runner=__spec__.name)
        return
    rt = read(args['runtime'])
    if stage == 'dispatch':
        dispatch(rt)
    elif stage == 'checkpoint_smoke':
        panel.activate(rt)
        action.checkpoint_smoke(rt)
    elif stage == 'cpu_score':
        meta = read(Path(rt['run_root']) / 'geometry-complete.json')
        panel.cpu_score(rt, MODES, pairs=PAIRS, trainable_parameters=meta['parameter_count'])
    elif stage in ('smoke', 'fit'):
        FIT(rt, 'geometry', smoke=stage == 'smoke')
    elif stage == 'held_metrics':
        held_metrics(rt)
    elif stage == 'timing':
        action.measure(rt, name='geometry', model_factory=control.MODEL, run=RUN,
                       guard_root=control.parent(rt))
    else:
        evaluate(rt, args['name'])


if __name__ == '__main__':
    main()
