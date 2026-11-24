"""Isolate which reconstructed waveform components explain known-action LoRa gains."""
import argparse
from functools import partial
from itertools import product
from pathlib import Path

import numpy as np
import torch

from experiments.cric_dura_comparison.io import read, write
from .channel_blind_source import panel
from .chirp_spectrum_source import read_frozen
from .common_failure_source import MODES as HISTORICAL_MODES
from .common_failure_source import RUN as HISTORICAL_RUN
from .common_failure_source import diagnostic_capture, evaluate
from .complex_vector_source import publish
from .data import SOURCE, VIEWS
from .denoise_source import summarize
from .dura_repair import complex_iq, real_iq
from .full_packet_views import source_contract
from .packet_runner import schedule
from .physical_inverse_source import source_trace_implementation
from .refine_runner import start
from .runner import activate
from .spectral_unroll import ROUTES


RUN = '20261124-phase1-lora-carrier-factorial-source-m2-r01'
FACTORS = ('trajectory', 'carrier_amplitude', 'detail')
LEVELS = tuple(product((0, 1), repeat=len(FACTORS)))
BASE = 'noise100'
EXACT = 'noise100_action'
ARM_NAMES = (BASE, 'detail', 'carrier_amplitude', 'amplitude_detail',
             'trajectory', 'trajectory_detail', 'trajectory_amplitude', EXACT)
MODES = ('direct', 'reference', *ARM_NAMES, 'fixed_received_carrier')
PAIRS = tuple((arm, BASE) for arm in MODES[3:])
CAPTURE = partial(diagnostic_capture, direct_doses=(0., 1.), inverse_doses=(1.,))


def normalize_to(value, reference):
    power = value.abs().square().mean(dim=(-2, -1), keepdim=True)
    wanted = reference.abs().square().mean(dim=(-2, -1), keepdim=True)
    return value * (wanted / power.clamp_min(1e-12)).sqrt()


def carrier_terms(wave, rotation):
    tracked = wave * rotation.conj()
    mean = tracked.mean(-1, keepdim=True)
    return mean, tracked-mean


def factorial_readouts(base, recovered, rotations):
    """Cross the source of the trajectory, mean tone and centered chirp shape."""
    first_rotation, second_rotation, received_rotation = rotations
    carrier, detail = zip(*(carrier_terms(x, r) for x,r in
                            ((base, first_rotation), (recovered, second_rotation))), strict=True)
    trajectories = (first_rotation, second_rotation)
    cross = [base]
    for t, a, d in LEVELS[1:-1]:
        waveform = trajectories[t] * (carrier[a] + detail[d])
        cross.append(normalize_to(waveform, base))
    cross.append(recovered)
    delta = recovered-base
    _, noncarrier = carrier_terms(delta, received_rotation)
    # Only a common packet gain may change the retained carrier projection.
    protected = normalize_to(base + noncarrier*received_rotation, base)
    return (*cross, protected)


@torch.no_grad()
def render(block, reference, tracker):
    received, denoised, exact = block
    if torch.equal(denoised, exact):
        controls = (denoised,) * (len(MODES)-2)
    else:
        rotations = tuple(tracker.track(value)[1] for value in (denoised, exact, received))
        controls = tuple(real_iq(value) for value in factorial_readouts(
            complex_iq(denoised), complex_iq(exact), rotations))
        if not all(torch.isfinite(v).all() for v in controls):
            raise FloatingPointError('Nonfinite carrier-factorial waveform')
    return torch.stack((received, reference, *controls))


def mean_effect(correct, factor):
    levels = np.asarray(LEVELS)[:,factor].astype(bool)
    return correct[levels].mean(axis=0) - correct[~levels].mean(axis=0)


def scoped_effects(predictions, labels, batch):
    all_correct = np.stack([predictions[v]==labels for v in VIEWS[1:]], axis=1)
    cube = all_correct[2:2+len(LEVELS)]
    baseline = cube[0]
    improved = cube[-1] & ~baseline

    def interval(values):
        return dict(net=float(values.sum()), ci=np.quantile(values[batch].sum(1),(.025,.975)).tolist())

    effects = {}
    for i,name in enumerate(FACTORS):
        effect = mean_effect(cube, i)
        effects[name] = dict(six=interval(effect.sum(0)), low=interval(effect[-1]))
    outcomes = {}
    for i, name in enumerate(MODES):
        retained = (all_correct[i] & improved)
        outcomes[name] = dict(six=int(retained.sum()),low=int(retained[-1].sum()))
    return dict(effects=effects,
                oracle_gains=dict(six=int(improved.sum()),low=int(improved[-1].sum())),
                retained=outcomes)


def score(rt):
    root, source = Path(rt['run_root']), Path(rt['source_root'])
    history = root.parent/HISTORICAL_RUN
    if read(history/'state.json')['status'] != 'SOURCE_SCORED':
        raise ValueError('Known-action source reference must be complete')
    frozen = {r: read_frozen(root,r,MODES) for r in ROUTES}
    old = {r: read_frozen(history,r,HISTORICAL_MODES) for r in ROUTES}
    order = frozen['full'][0]
    if any(v[0] != order for v in (*frozen.values(),*old.values())):
        raise ValueError('Carrier diagnostic requires identical physical source IDs and order')
    controls = ('direct','reference',BASE,EXACT)
    for route in ROUTES:
        for view in VIEWS:
            for arm in controls:
                new = frozen[route][1][view][MODES.index(arm)]
                previous = old[route][1][view][HISTORICAL_MODES.index(arm)]
                if not np.array_equal(new,previous):
                    raise ValueError(f'Known-action endpoint differs for {route}/{view}/{arm}')
        clean = frozen[route][1]['clean']
        if not np.all(clean[2:] == clean[:1]):
            raise ValueError('Carrier diagnostics must leave clean classifications unchanged')

    ids = np.asarray(source_contract(source)['role_ids']['V'])
    lookup = {v:i for i,v in enumerate(ids)}
    chosen = np.asarray([lookup[v] for v in order])
    with np.load(source/'V_metadata.npz',allow_pickle=False) as metadata:
        if not np.array_equal(panel(np.asarray(SOURCE)[metadata['rx_i']]),chosen):
            raise ValueError('Unexpected fixed source V panel')
        labels = metadata['y'][chosen]
    rng = np.random.default_rng(2026113761)
    per_rx = len(chosen)//len(SOURCE)
    batches = np.concatenate([rng.integers(j*per_rx,(j+1)*per_rx,(3000,per_rx))
                              for j in range(len(SOURCE))],axis=-1)
    findings = {r:scoped_effects(frozen[r][1],labels,batches) for r in ROUTES}
    write(root/'carrier-factorial-scores.json',dict(status='SOURCE_DIAGNOSTIC_SCORED',n=len(chosen),
          arms=list(MODES), historical_endpoint_differences=0,target_reads=0,training_updates=0,
          truth_after_predictions=True, routes=findings))
    for r,(_,readouts,metrics) in frozen.items():
        summarize(root,r,source,chosen,ids,readouts,metrics,MODES,
                  predictions_written=True,comparisons=PAIRS,per_rx=per_rx)
    print('LORA_CARRIER_FACTORIAL_SOURCE_SCORED',flush=True)


def dispatch(rt):
    root = Path(rt['run_root'])
    root.mkdir()
    launch = partial(start,module=rt['runner'])
    try:
        for phase in ('smoke','evaluate'):
            schedule(rt,[(phase,r) for r in ROUTES],launch)
        if launch(rt,'score','source').wait():
            raise RuntimeError('Independent carrier-factorial source scoring failed')
        write(root/'state.json',dict(status='SOURCE_SCORED'))
    except Exception as exc:
        write(root/'state.json',dict(status='FAILED',error=str(exc)))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest='action',required=True)
    pub = actions.add_parser('publish')
    for key in ('repo','output','context','ssh-config','parent-submit'):
        pub.add_argument('--'+key,type=Path,required=True)
    for action in ('dispatch','smoke','evaluate','score'):
        sub=actions.add_parser(action)
        sub.add_argument('runtime',type=Path)
        if action != 'dispatch':
            sub.add_argument('row',choices=('source',) if action == 'score' else ROUTES)
    options = vars(parser.parse_args())
    action = options.pop('action')
    if action == 'publish':
        return publish(**options,run=RUN,runner=__spec__.name)
    rt = read(options['runtime'])
    if action == 'dispatch':
        return dispatch(rt)
    if action == 'score':
        return score(rt)
    activate(rt)
    source_trace_implementation()
    return evaluate(rt,options['row'],smoke=action=='smoke',capture=CAPTURE,render=render,modes=MODES)


if __name__ == '__main__':
    main()
