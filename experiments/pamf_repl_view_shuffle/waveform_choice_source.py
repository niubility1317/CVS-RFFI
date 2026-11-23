"""Train and test a WiSig selector without access to device labels or recognition outputs."""

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
from .observable_action import generic_observation
from .waveform_choice import PROPOSALS, WaveformChoice, objective, select, waveform_risk


RUN = '20261124-phase1-wisig-waveform-choice-source-m3-r01'
MODES = (*PROPOSALS[1:], 'static', 'routed', 'routed_guarded')
COUNTS = tuple(f'guarded_{name}' for name in PROPOSALS)
QUANTILE = .999


def model_at(rt):
    return action.load_arm(Path(rt['run_root']), 'choice', 'cuda:0', model_factory=WaveformChoice, run=RUN)


def calibration(rt):
    evidence = read(Path(rt['run_root']) / 'selection.json')
    if (evidence['n'], evidence['seed'], evidence['quantile']) != (action.CALIBRATION, action.SEED, QUANTILE):
        raise ValueError('Missing source U only choice calibration')
    return evidence


@torch.no_grad()
def calibrate(rt):
    model = model_at(rt)
    _, indices, _ = action.partition()
    source = action.source_u(rt)
    generator = torch.Generator(device='cuda:0').manual_seed(action.SEED + 14)
    risks, chosen_risks, margins = [], [], []
    for part in np.split(indices, action.CALIBRATION // 64):
        clean = action.received(source[part], 'cuda:0')
        candidates, logits = model(clean)
        margins.append(select(candidates, logits)[2])
        observed = generic_observation(clean[:, None], generator)[:, 0]
        candidates, logits = model(observed)
        risk = waveform_risk(candidates, clean)
        chosen = select(candidates, logits)[1]
        chosen_risks.append(risk.gather(1, chosen[:, None]).squeeze(1))
        risks.append(risk)
    cost = torch.cat(risks)
    selected = torch.cat(chosen_risks)
    clean_margin = torch.cat(margins)
    if not all(torch.isfinite(v).all() for v in (cost, clean_margin, selected)):
        raise FloatingPointError('Nonfinite source U selection evidence')
    mean = cost.mean(dim=0)
    static = int(mean.argmin())
    threshold = max(0., float(clean_margin.quantile(QUANTILE)))
    evidence = dict(n=len(indices), seed=action.SEED, quantile=QUANTILE, minimum_margin=threshold,
                    static=PROPOSALS[static], average_proposal_risk=dict(zip(PROPOSALS, mean.tolist())),
                    routed_waveform_risk=float(selected.mean()), best_per_packet_waveform_risk=float(cost.min(1).values.mean()),
                    perturbation_seed=action.SEED + 14,
                    target_reads=0, recognizer_reads=0, source_truth_reads=0)
    write(Path(rt['run_root']) / 'selection.json', evidence)
    print('WISIG_CHOICE_CALIBRATED', evidence['static'], threshold, flush=True)


def evaluate(rt, route):
    model = model_at(rt)
    config = calibration(rt)
    fixed = PROPOSALS.index(config['static'])

    def restore(y):
        candidates, logits = model(y)
        routed, _, margin = select(candidates, logits)
        guarded, chosen, _ = select(candidates, logits, config['minimum_margin'])
        outputs = {name: candidates[:, at] for at, name in enumerate(PROPOSALS) if at}
        outputs.update(static=candidates[:, fixed], routed=routed, routed_guarded=guarded)
        counts = {f'guarded_{name}': (chosen == at).float() for at, name in enumerate(PROPOSALS)}
        return outputs, dict(routing_margin=margin, **counts)

    panel.predict(rt, route, alternatives=MODES, restore=restore, diagnose=action.spectral_evidence,
                  count_diagnostics=COUNTS)


@torch.no_grad()
def timing(rt):
    model = model_at(rt)
    floor = calibration(rt)['minimum_margin']
    clean = action.received(action.source_u(rt)[:1], 'cuda:0')
    generator = torch.Generator(device='cuda:0').manual_seed(action.SEED + 16)
    disturbed = generic_observation(clean[:, None], generator)[:, 0]
    measured = {}
    for name, wave in (('clean', clean), ('generic_disturbed', disturbed)):
        samples = []
        for i in range(32):
            torch.cuda.synchronize()
            start = perf_counter()
            candidates, logits = model(wave)
            result, choice, _ = select(candidates, logits, floor)
            torch.cuda.synchronize()
            if i >= 8:
                samples.append(1000 * (perf_counter() - start))
        if not torch.isfinite(result).all():
            raise FloatingPointError('Invalid timed WiSig recovery')
        measured[name] = dict(median_ms=float(np.median(samples)), output=PROPOSALS[int(choice[0])])
    write(Path(rt['run_root']) / 'b1.json', dict(n_measurements=24, timing=measured,
          gpu=torch.cuda.get_device_name(), parameters=sum(p.numel() for p in model.parameters())))


def dispatch(rt):
    root = Path(rt['run_root'])
    root.mkdir()
    (root / 'logs').mkdir()
    launch = partial(panel.launch, module=rt['runner'])
    try:
        for stage, names in (('checkpoint_smoke', ('source',)), ('smoke', ('choice',)), ('fit', ('choice',)),
                             ('calibrate', ('source',)), ('timing', ('source',)), ('predict', panel.ROUTES)):
            schedule(rt, [(stage, name) for name in names], launch)
        if launch(rt, 'cpu_score', 'source').wait():
            raise RuntimeError('WiSig waveform choice scoring failed')
        write(root / 'state.json', dict(status='SOURCE_SCORED'))
    except Exception as error:
        write(root / 'state.json', dict(status='FAILED', error=str(error)))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='action', required=True)
    pub = commands.add_parser('publish')
    for key in ('repo', 'output', 'local-context', 'ssh-config'):
        pub.add_argument('--' + key, type=Path, required=True)
    for stage in ('dispatch', 'checkpoint_smoke', 'smoke', 'fit', 'calibrate', 'timing', 'predict', 'cpu_score'):
        cmd = commands.add_parser(stage)
        cmd.add_argument('runtime', type=Path)
        if stage != 'dispatch':
            choices = ('choice',) if stage in ('smoke', 'fit') else panel.ROUTES if stage == 'predict' else ('source',)
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
        meta = read(Path(rt['run_root']) / 'choice-complete.json')
        pairs = (('routed', 'static'), ('routed_guarded', 'static'), ('routed_guarded', 'routed'))
        panel.cpu_score(rt, MODES, pairs=pairs, trainable_parameters=meta['parameter_count'])
    elif stage in ('smoke', 'fit'):
        action.fit(rt, 'choice', smoke=stage == 'smoke', model_factory=WaveformChoice,
                   run=RUN, criterion=objective)
    elif stage == 'calibrate':
        calibrate(rt)
    elif stage == 'timing':
        timing(rt)
    else:
        panel.activate(rt)
        if stage == 'checkpoint_smoke':
            action.checkpoint_smoke(rt)
        else:
            evaluate(rt, args['name'])


if __name__ == '__main__':
    main()
