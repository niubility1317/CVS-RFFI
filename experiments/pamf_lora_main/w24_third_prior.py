"""Matched two- and three-prior W24 recovery on unlabeled source IQ."""
import argparse
from copy import deepcopy
from dataclasses import dataclass, replace
from functools import partial
import importlib
import os
import re
from pathlib import Path, PurePosixPath
import subprocess
from time import monotonic, sleep

if __package__ is None:
    from experiments import pamf_lora_main
    pamf_lora_main.__path__.append(str(Path(__file__).resolve().parent))


RUN = '20261126-phase1-lora-w24-third-prior-s2026112991-r01'
ARMS = ('two', 'three')
STEPS = 1800
MODES = ('direct', 'all_half', 'two_half', 'three_half', 'two_full', 'three_full')
CONTRASTS = (('three_half', 'two_half'), ('three_full', 'two_full'), ('three_half', 'direct'))
ROUTES = ('full_native', 'native_residual_w24_v1')
ISOLATION = ('F_reads', 'TX_reads', 'V_reads', 'target_reads', 'practical_reads', 'trace_reads')
DIRECTION_RUN = '20261126-phase1-lora-w24-first-direction-s2026112991-r01'
DIRECTION_ARMS = ('direction_control', 'direction_first')
READOUT_RUN = '20261126-phase1-lora-w24-trained-readout-s2026112991-r01'
READOUT_ARMS = ('readout_control', 'readout_adaptive')
ROBUST_RUN = '20261126-phase1-lora-w24-robust-map-s2026112991-r01'
ROBUST_ARMS = ('robust_control', 'robust_map')
RADIUS_RUN = '20261126-phase1-lora-w24-final-radius-s2026112991-r01'
RADIUS_ARMS = ('radius_control', 'radius_adaptive')
PEER_RUN = '20261126-phase1-lora-w24-peer-innovation-s2026112991-r01'
PEER_ARMS = ('innovation_local', 'innovation_peer')
BIAS_RUN = '20261126-phase1-lora-w24-coherent-error-s2026112991-r01'
BIAS_ARMS = ('coherent_control', 'coherent_error')
BAND_TUNE_RUN = '20261126-phase1-lora-w24-band-direction-tune-s2026112991-r01'
BAND_TUNE_ARMS = ('band_tune_control', 'band_tune_direction')
BAND_ADAPTER_RUN = '20261126-phase1-lora-w24-band-adapter-s2026112991-r01'
BAND_ADAPTER_ARMS = ('band_adapter_time', 'band_adapter_frequency')
BAND_REFINE_RUN = '20261126-phase1-lora-w24-band-polar-refine-s2026112991-r02'
BAND_REFINE_ARMS = ('band_cartesian', 'band_polar')
WHITE_RUN = '20261126-phase1-lora-w24-white-corruption-s2026112991-r01'
WHITE_ARMS = ('archive_noise', 'stationary_noise')
BAND_PRECISION_RUN = '20261126-phase1-lora-w24-band-precision-s2026112991-r01'
BAND_PRECISION_ARMS = ('precision_shared', 'precision_resolved')
SEVERITY_RUN = '20261126-phase1-lora-w24-severity-weight-s2026112991-r01'
SEVERITY_ARMS = ('severity_uniform', 'severity_weighted')
HALF_PRIMARY_RUN = '20261126-phase1-lora-w24-half-primary-s2026112991-r01'
HALF_PRIMARY_ARMS = ('half_aux_control', 'half_primary')
OUTER_TUNE_ARMS = (*BAND_TUNE_ARMS, *BAND_ADAPTER_ARMS, *BAND_REFINE_ARMS,
                   *WHITE_ARMS, *BAND_PRECISION_ARMS, *SEVERITY_ARMS, *HALF_PRIMARY_ARMS)
OUTER_BASELINE = 'outer_parent_half'
OUTER_PARENT = '20261125-phase1-lora-w24-outer-objective-source-s2026112991-r01'
VARIANTS = {
    'direction': (DIRECTION_RUN, DIRECTION_ARMS),
    'readout': (READOUT_RUN, READOUT_ARMS),
    'robust': (ROBUST_RUN, ROBUST_ARMS),
    'final-radius': (RADIUS_RUN, RADIUS_ARMS),
    'peer': (PEER_RUN, PEER_ARMS),
    'coherent-error': (BIAS_RUN, BIAS_ARMS),
    'band-tune': (BAND_TUNE_RUN, BAND_TUNE_ARMS),
    'band-adapter': (BAND_ADAPTER_RUN, BAND_ADAPTER_ARMS),
    'band-refine': (BAND_REFINE_RUN, BAND_REFINE_ARMS),
    'white-corruption': (WHITE_RUN, WHITE_ARMS),
    'band-precision': (BAND_PRECISION_RUN, BAND_PRECISION_ARMS),
    'half-primary': (HALF_PRIMARY_RUN, HALF_PRIMARY_ARMS),
    'severity-weight': (SEVERITY_RUN, SEVERITY_ARMS),
}
ARM_TRIAL = dict.fromkeys(ARMS, True)
ARM_TRIAL.update({arm: kind for kind, (_, arms) in VARIANTS.items() for arm in arms})


@dataclass(frozen=True)
class Trial:
    run: str
    arms: tuple
    steps: int
    source_modes: tuple
    contrasts: tuple
    routes: tuple = ROUTES


def trial(kind):
    if kind is True:
        return Trial(RUN, ARMS, STEPS, MODES, CONTRASTS)
    if kind not in VARIANTS:
        raise ValueError(f'Unknown matched W24 trial: {kind}')
    run, arms = VARIANTS[kind]
    modes = ('direct', 'all_half', *(f'{arm}_{alpha}' for alpha in ('half', 'full') for arm in arms))
    contrasts = ((modes[3], modes[2]), (modes[5], modes[4]), (modes[3], modes[0]))
    if arms[0] in OUTER_TUNE_ARMS:
        steps = 1200 if kind == 'band-refine' else 1800
        return Trial(run, arms, steps, (*modes, OUTER_BASELINE), (*contrasts, (modes[3], OUTER_BASELINE)), ROUTES[:1])
    return Trial(run, arms, 3600, modes, contrasts, ROUTES[:1])


def restorer(training, arm):
    if arm not in ARM_TRIAL:
        raise ValueError(f'Unknown prior arm: {arm}')
    if arm == 'readout_adaptive':
        from experiments.pamf_lora_main.w24_readout_strength import adaptive_final

        class AdaptiveReadout(training.DuraW24):
            def propose(self, y, quality, rms, edges):
                (coarse, first, final), action, variance, noise, support = super().propose(y, quality, rms, edges)
                final = adaptive_final(y, final, noise, rms)
                return (coarse, first, final), action, variance, noise, support

        return AdaptiveReadout()
    if arm == 'robust_map':
        class RobustMap(training.DuraW24):
            def stages(self, y, quality, rms, **options):
                return super().stages(y, quality, rms, **dict(options, robust=True))

        return RobustMap()
    if arm == 'radius_adaptive':
        from experiments.pamf_lora_main.w24_final_radius import projected_stages

        archive_package = training.DuraW24.__module__.rpartition('.')[0]
        old = importlib.import_module('.dura_capacity', archive_package)

        class FinalRadius(training.DuraW24):
            def stages(self, y, quality, rms, *, condition=()):
                return projected_stages(self, old, y, quality, rms, condition=condition)

        return FinalRadius()
    if arm in PEER_ARMS:
        from experiments.pamf_lora_main.w24_peer_innovation import restorer as innovation_restorer

        return innovation_restorer(training, use_peers=arm == 'innovation_peer')
    if arm in BAND_ADAPTER_ARMS:
        from experiments.pamf_lora_main.w24_band_adapter import restorer as adapter_restorer

        return adapter_restorer(training, frequency=arm == BAND_ADAPTER_ARMS[1])
    if arm in BAND_REFINE_ARMS:
        from experiments.pamf_lora_main.w24_band_refine import BandRefiner

        return BandRefiner(training, polar=arm == BAND_REFINE_ARMS[1])
    if arm in BAND_PRECISION_ARMS:
        from experiments.pamf_lora_main.w24_band_precision import restorer as precision_restorer

        return precision_restorer(training, resolved=arm == BAND_PRECISION_ARMS[1])
    if arm != 'three':
        return training.DuraW24()

    class ThreePrior(training.DuraW24):
        def __init__(self):
            super().__init__()
            # Leave the two matched initial prior states intact. The third has
            # separately trainable parameters, initialized from prior two.
            self.priors.append(deepcopy(self.priors[-1]))

        def propose(self, *args, **kwargs):
            stages, *physical = super().propose(*args, **kwargs)
            return (stages[0], stages[1], stages[-1]), *physical

    return ThreePrior()


def fit_inputs(rt, arm):
    if arm not in ARM_TRIAL:
        raise ValueError(f'Unknown prior arm: {arm}')
    inputs = dict(source_root=rt['source_root'], run_root=str(PurePosixPath(rt['run_root']) / arm))
    if arm in OUTER_TUNE_ARMS:
        inputs['source_parent'] = rt['source_parent']
    return inputs


def restore_center(original, delta, fraction):
    restored = original.clone()
    restored[..., :384] += fraction * delta
    return restored


def promotion(routes, kind=True):
    spec = trial(kind)
    direct, mature, control, candidate = spec.source_modes[:4]
    controls = (control, direct, mature, OUTER_BASELINE) if OUTER_BASELINE in spec.source_modes else (control, direct, mature)
    return len(routes) == len(spec.routes) and all(
        row[candidate]['by_scene']['clean'] >= row[direct]['by_scene']['clean']
        and row[candidate]['six_correct'] > max(row[k]['six_correct'] for k in controls)
        and row[candidate]['low_urban_correct'] >= max(row[k]['low_urban_correct'] for k in controls)
        for row in routes)


def qualified_source(root, kind=True):
    from experiments.cric_dura_comparison.io import read
    from experiments.pamf_lora_main.w24_native_full import VIEWS

    spec = trial(kind)
    if read(root / 'state.json')['status'] not in ('SOURCE_SCORED', 'FORMAL_PREDICTING'):
        raise ValueError('Expected frozen and scored source rows')
    results, physical_ids = [], None
    for route in spec.routes:
        predictions = read(root / route / 'source-predictions.json')
        scores = read(root / route / 'source-scores.json')
        ids, waveform = predictions['physical_ids'], predictions['waveform']
        if (predictions['status'] != 'PREDICTIONS_FROZEN' or predictions['target_reads'] != 0
                or predictions['source_truth_reads'] != 0 or predictions['modes'] != list(spec.source_modes)
                or predictions['views'] != list(VIEWS) or len(ids) != 300 or len(set(ids)) != 300
                or (physical_ids is not None and ids != physical_ids)
                or waveform['candidate_steps'] != spec.steps or waveform['candidate_run'] != spec.run
                or waveform['route'] != route or waveform['ungated'] is not True
                or waveform['amplitudes'] != [.5, 1.]
                or waveform['activation'] != dict.fromkeys(VIEWS, [300]*(len(spec.source_modes)-1))
                or scores['status'] != 'ANALYZED' or scores['source_truth_last'] is not True
                or scores['target_reads'] != 0 or scores['n_packets'] != 300 or scores['n_practical'] != 1800
                or scores['waveform'] != waveform):
            raise ValueError(f'Invalid source evaluation for {route}')
        physical_ids = ids
        results.append(scores['scores'])
    if not promotion(results, kind):
        raise ValueError('Source comparison did not qualify')


def source_recipe(training, arm, *, initialize=None):
    def finalize(model, _tracker, _calibration):
        if arm in BAND_ADAPTER_ARMS:
            trained, key = (model.band_adapter,), 'adapter_gradient_norms'
        elif arm in BAND_PRECISION_ARMS:
            trained, key = (model.band_precision,), 'precision_gradient_norms'
            if any(p.requires_grad or p.grad is not None for name, p in model.named_parameters()
                   if not name.startswith('band_precision.')):
                raise ValueError('Expected a frozen source backbone for band precision')
        elif arm in BAND_REFINE_ARMS:
            trained, key = (model.head,), 'head_gradient_norms'
            if any(p.requires_grad or p.grad is not None for p in model.backbone.parameters()):
                raise ValueError('Expected a frozen source backbone for band refinement')
        else:
            trained, key = model.priors, 'prior_gradient_norms'
        grads = [sum(float(p.grad.square().sum()) for p in module.parameters() if p.grad is not None)**.5
                 for module in trained]
        if not all(0 < value < float('inf') for value in grads):
            raise FloatingPointError(f'Invalid gradient in a trainable W24 prior or adapter: {grads}')
        info = dict(input_gate=False, prior_count=len(model.priors), **{key: grads},
                    parameters=sum(p.numel() for p in model.parameters()))
        if arm in OUTER_TUNE_ARMS:
            info['source_parent'] = OUTER_PARENT
        if arm in BAND_ADAPTER_ARMS:
            info['trainable_parameters'] = sum(p.numel() for p in model.band_adapter.parameters())
        if arm in BAND_REFINE_ARMS:
            info.update(trainable_parameters=sum(p.numel() for p in model.head.parameters()), backbone_frozen=True)
        if arm in BAND_PRECISION_ARMS:
            info.update(trainable_parameters=sum(p.numel() for p in model.band_precision.parameters()),
                        backbone_frozen=True)
        if arm in WHITE_ARMS:
            info['source_perturbation'] = 'equal_center_energy_white' if arm == WHITE_ARMS[1] else 'archived'
        if arm in SEVERITY_ARMS:
            info['severity_weighted'] = arm == SEVERITY_ARMS[1]
        return None, info

    spec = trial(ARM_TRIAL[arm])
    name = f'w24_{arm}_prior' if arm in ARMS else f'w24_{arm}'
    return replace(training.RECIPE, name=name, steps=spec.steps,
                   validation_steps=(spec.steps // 3, 2 * spec.steps // 3),
                   model=partial(restorer, training, arm), finalize_calibration=finalize,
                   initialize=initialize, coarse_first=training.RECIPE.coarse_first and arm not in OUTER_TUNE_ARMS)


def fit(config, arm, *, smoke):
    import torch
    from experiments.cric_dura_comparison.bootstrap import numerical_policy
    from experiments.cric_dura_comparison.io import read, tensor
    from experiments.pamf_lora_main import channel_blind_source, dura_w24_model, dura_w24_train

    rt = read(config)
    paths = {'source_root', 'run_root'}
    if arm in OUTER_TUNE_ARMS:
        paths.add('source_parent')
    if set(rt) != paths:
        raise ValueError('Fit may receive only registered source paths and its own output root')

    def prepare(roles, device, *, smoke, fit_per_rx, bank_per_rx):
        source = Path(roles['source_root'])
        contract = dura_w24_train.source_contract(source)
        iq, rx, _, _ = dura_w24_train.source_role(source, contract, 'U_s')
        if smoke:
            train, _, held = channel_blind_source.partition(rx, fit=5, cal=4, held=3)
        else:
            train, _, held = channel_blind_source.partition(rx, fit=fit_per_rx)
        train, held = (tensor(iq[index].view('float32'), device).view(torch.complex64) for index in (train, held))
        return (train, torch.empty(0, device=device), held), contract

    calculate, initialize = dura_w24_model.objective, None
    if arm in ('direction_first', 'coherent_error'):
        from experiments.pamf_lora_main.w24_first_direction import objective_for

        calculate = objective_for(dura_w24_model, output_bias=arm == 'coherent_error')
    if arm in OUTER_TUNE_ARMS:
        from experiments.pamf_lora_main.w24_band_tune import WEIGHT
        from experiments.pamf_lora_main.w24_outer_objective_source import objective_for, read_outer_checkpoint

        parent = Path(rt['source_parent'])
        if parent != Path(rt['run_root']).parent.parent / OUTER_PARENT:
            raise ValueError('Expected the registered source-only outer-loss parent')
        contract = dura_w24_train.source_contract(Path(rt['source_root']))
        _, weights, _ = read_outer_checkpoint(parent, contract, 'cpu', dura_w24_train, channel_blind_source)

        def initialize(model, _tracker, _training, _calibration):
            if arm in BAND_ADAPTER_ARMS:
                from experiments.pamf_lora_main.w24_band_adapter import load_parent

                load_parent(model, weights)
            elif arm in BAND_PRECISION_ARMS:
                from experiments.pamf_lora_main.w24_band_adapter import load_parent

                load_parent(model, weights, new_module='band_precision')
            elif arm in BAND_REFINE_ARMS:
                model.backbone.load_state_dict(weights, strict=True)
            else:
                model.load_state_dict(weights, strict=True)

        weight = WEIGHT if arm == BAND_TUNE_ARMS[1] else 0.
        if arm in BAND_REFINE_ARMS:
            calculate = objective_for(dura_w24_model, refine_band=True)
        elif arm in WHITE_ARMS:
            calculate = objective_for(dura_w24_model, white_corruption=arm == WHITE_ARMS[1])
        elif arm in SEVERITY_ARMS:
            calculate = objective_for(dura_w24_model, severity_weighted=arm == SEVERITY_ARMS[1])
        elif arm in HALF_PRIMARY_ARMS:
            calculate = objective_for(dura_w24_model, half_primary=arm == HALF_PRIMARY_ARMS[1])
        else:
            calculate = objective_for(dura_w24_model, band_direction_weight=weight)
    numerical_policy()
    channel_blind_source.fit(rt, smoke=smoke, recipe=source_recipe(dura_w24_train, arm, initialize=initialize),
                             calculate=calculate, receiver_batches=True, prepare=prepare,
                             make_tracker=dura_w24_train.WideTracker)


def load_candidate(root, contract, device, arm):
    import torch
    from experiments.cric_dura_comparison.io import read

    training = importlib.import_module('lora_wide_dura_archive.dura_w24_train')
    archived = importlib.import_module('lora_wide_dura_archive.channel_blind_source')
    recipe = source_recipe(training, arm)
    row = root / arm
    summary = read(row / f'{recipe.name}-complete.json')
    expected = archived.proof(contract, recipe)
    if (any(summary.get(k) != value for k, value in expected.items() if k != 'contract')
            or any(expected.get(key) != 0 for key in ISOLATION)
            or summary.get('n_calibration') != 0 or summary.get('input_gate') is not False
            or summary.get('threshold') is not None or summary.get('prior_count') != (3 if arm == 'three' else 2)):
        raise ValueError(f'Unverified source freeze for {arm}')
    if arm in OUTER_TUNE_ARMS and summary.get('source_parent') != OUTER_PARENT:
        raise ValueError(f'Unverified warm start for {arm}')
    if arm in (*BAND_REFINE_ARMS, *BAND_PRECISION_ARMS) and summary.get('backbone_frozen') is not True:
        raise ValueError(f'Unverified frozen source backbone for {arm}')
    checkpoint = torch.load(row / f'{recipe.name}-model.pt', map_location=device, weights_only=True)
    if checkpoint['proof'] != expected or checkpoint['threshold'] is not None:
        raise ValueError(f'Invalid {arm} checkpoint source binding')
    model = recipe.model().to(device).eval().requires_grad_(False)
    model.load_state_dict(checkpoint['weights'], strict=True)
    return model, summary


def dispatch(runtime):
    from experiments.cric_dura_comparison.io import read, write
    from experiments.pamf_lora_main.packet_runner import schedule
    from experiments.pamf_lora_main.runner import available
    from experiments.pamf_lora_main.w24_native_full import W24_COMMIT
    from experiments.pamf_lora_main.w24_wide_transfer import training_runtime

    rt = read(runtime)
    kind = rt['third_prior']
    spec = trial(kind)
    archive, _ = training_runtime(rt)
    reference = Path(rt['w24_release_root'])
    root = Path(rt['run_root'])
    root.mkdir()
    state = root / 'state.json'
    own = Path(__file__).resolve()
    evaluator = own.with_name('w24_native_residual_source.py')

    def status(**fields):
        write(state, fields, exclusive=False)

    def start(action, label, gpu, cwd, entry, config, *extra):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='' if gpu is None else str(gpu),
                   PYTHONPATH=str(cwd), OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
        with (root / f'{label}.log').open('x', encoding='utf-8') as log:
            job = subprocess.Popen([rt['python'], '-u', str(entry), action, str(config), *extra],
                                   cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT)
        print('W24_PRIOR_STARTED', label, job.pid, gpu, flush=True)
        return job

    def training_job(_plan, _phase, arm, gpu):
        return start('fit-arm', f'{arm}-fit', gpu, archive, own, root / arm / 'fit-inputs.json', arm)

    try:
        if read(reference / 'submit.json')['code_commit'] != W24_COMMIT:
            raise ValueError('Expected the registered W24 evaluator archive')
        for arm in spec.arms:
            (root / arm).mkdir()
            write(root / arm / 'fit-inputs.json', fit_inputs(rt, arm))
        cards = available(rt['allowed_gpus'], rt['max_compute_processes'])
        if len(cards) < len(spec.arms):
            raise RuntimeError('The matched pair requires two GPUs with sufficient free memory')
        status(status='SOURCE_TRAINING', arms=list(spec.arms), steps=spec.steps, available_gpus=cards)
        schedule(rt, [('fit', arm) for arm in spec.arms], training_job)
        status(status='SOURCE_FROZEN', arms=list(spec.arms), steps=spec.steps)
        deadline = monotonic() + 600
        while not (cards := available(rt['allowed_gpus'], rt['max_compute_processes'])):
            if monotonic() >= deadline:
                raise RuntimeError('No GPU available for frozen source evaluation')
            sleep(15)
        for phase, gpu in (('predict-source', cards[0]), ('score-source', None)):
            proc = start(phase, phase, gpu, reference, evaluator, runtime)
            status(status='RUNNING', phase=phase, pid=proc.pid, gpu=gpu)
            if proc.wait():
                raise RuntimeError(f'{phase} failed')
        results = [read(root / route / 'source-scores.json')['scores'] for route in spec.routes]
        promote = promotion(results, kind)
        status(status='SOURCE_SCORED', promote=promote, arms=list(spec.arms), steps=spec.steps)
        if promote:
            qualified_source(root, kind)
            config = root / 'formal-runtime.json'
            write(config, dict(rt, candidate_root=str(root), run_root=str(root/'formal'), ungated=True))
            job = start('dispatch', 'formal-dispatch', None, reference, own.with_name('w24_formal.py'), config)
            status(status='FORMAL_PREDICTING', pid=job.pid)
            if job.wait() or read(root/'formal'/'state.json')['status'] != 'PREDICTIONS_FROZEN':
                raise RuntimeError('Qualified candidate formal predictions did not freeze')
            status(status='PREDICTIONS_FROZEN', target_truth_read=False)
    except Exception as error:
        status(status='FAILED', error=str(error))
        raise


def publish(args):
    from experiments.cric_dura_comparison.io import read
    from experiments.pamf_lora_main.publish import release, release_commit
    from experiments.pamf_lora_main.w24_factor_audit import AUG_RUN
    from experiments.pamf_lora_main.w24_formal import RUN_UNGATED
    from experiments.pamf_lora_main.w24_native_full import DATA_RUN, W24_RUN
    from experiments.pamf_lora_main.w24_native_residual_source import SWEEP_RUN
    from experiments.pamf_lora_main.w24_wide_transfer import TRAIN_RUN

    ctx = read(args.context)
    kind = args.kind
    run = trial(kind).run
    rt = {key: ctx[key] for key in ('project', 'python', 'frozen_sources')}
    project = PurePosixPath(rt['project'])
    releases, runs = project / 'releases', project / 'runs'
    pools = {'readout': [0, 3, 7], 'robust': [5, 6, 1, 7], 'final-radius': [0, 5, 6, 7],
             'peer': [3, 7, 6, 5], 'coherent-error': [0, 3, 5, 7], 'band-tune': [7, 6, 3, 0, 2],
             'band-adapter': [0, 3, 5, 7], 'band-refine': [6, 7, 2, 0, 3],
             'white-corruption': [0, 3, 6, 7, 2], 'band-precision': [0, 3, 6, 7],
             'half-primary': [0, 3, 5, 7],
             'severity-weight': [6, 5, 3, 0, 7, 2]}
    rt.update(run_id=run, runner=__spec__.name, code_commit=release_commit(args.repo),
              release_root=str(releases / f'{run}-release'), run_root=str(runs / run),
              source_root=str(releases / f'{DATA_RUN}-release' / 'data/source'),
              native_target_root=str(releases / f'{DATA_RUN}-release' / 'data/target'),
              classifier_root=str(runs / AUG_RUN), w24_release_root=str(releases / f'{W24_RUN}-release'),
              training_release_root=str(releases / f'{TRAIN_RUN}-release'), third_prior=kind,
              cpu_workers=8, allowed_gpus=pools.get(kind, [5, 6, 1, 0]),
              max_workers=2, max_compute_processes=8)
    rt['external_releases'] = [rt[k] for k in ('source_root', 'classifier_root', 'w24_release_root',
                                               'training_release_root', 'native_target_root')] + [str(runs / name) for name in (SWEEP_RUN, RUN_UNGATED)]
    if OUTER_BASELINE in trial(kind).source_modes:
        rt['source_parent'] = str(runs / OUTER_PARENT)
        rt['external_releases'].append(rt['source_parent'])
    release(args.repo, args.output, rt, args.ssh_config)


def collect_score(args):
    from experiments.cric_dura_comparison.io import read
    from experiments.pamf_lora_main.w24_formal_score import score
    import json

    root = PurePosixPath(read(args.submit)['run_root'])
    kind = args.kind
    if not re.fullmatch(r'/[A-Za-z0-9_./-]+', str(root)) or root.name != trial(kind).run:
        raise ValueError('Unexpected immutable N607 run root')
    connection = ['-F', str(args.ssh_config), '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10']
    deadline = monotonic() + 12*60*60
    last = None
    while True:
        state = None
        try:
            remote = subprocess.run(['ssh', *connection, '-T', 'N607', 'cat ' + str(root/'state.json')],
                                    timeout=25, capture_output=True)
            if remote.returncode == 0:
                state = json.loads(remote.stdout)
        except (subprocess.TimeoutExpired, json.JSONDecodeError):
            # A short SSH failure or concurrent state write can interrupt a read.
            # Reobserve the same run without resubmitting it.
            pass
        if state is not None:
            if last != state['status']:
                print(state, flush=True)
                last = state['status']
            if state['status'] == 'FAILED':
                raise RuntimeError(state['error'])
            if state['status'] == 'SOURCE_SCORED' and not state['promote']:
                return
            if state['status'] == 'PREDICTIONS_FROZEN':
                break
        if monotonic() >= deadline:
            raise TimeoutError('No terminal N607 state in scoring window')
        sleep(40)
    args.output.mkdir(parents=True)
    subprocess.run(['scp', *connection, '-r', 'N607:' + str(root/'formal'/'predictions'), str(args.output)],
                   check=True, timeout=180)
    score(args.output/'predictions', args.dataset, args.output/'scores', third_prior=kind)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest='action', required=True)
    actions.add_parser('dispatch').add_argument('runtime', type=Path)
    for phase in ('fit-smoke', 'fit', 'fit-arm'):
        cmd = actions.add_parser(phase)
        cmd.add_argument('runtime', type=Path)
        cmd.add_argument('arm', choices=ARM_TRIAL)
    pub = actions.add_parser('publish')
    for key in ('repo', 'output', 'context', 'ssh-config'):
        pub.add_argument('--' + key, type=Path, required=True)
    collector = actions.add_parser('collect-score')
    for command in (pub, collector):
        variants = command.add_mutually_exclusive_group()
        variants.set_defaults(kind=True)
        for kind in VARIANTS:
            variants.add_argument('--' + kind, dest='kind', action='store_const', const=kind)
    for key in ('submit', 'dataset', 'output', 'ssh-config'):
        collector.add_argument('--' + key, type=Path, required=True)
    args = parser.parse_args()
    if args.action == 'publish':
        publish(args)
    elif args.action == 'dispatch':
        dispatch(args.runtime)
    elif args.action == 'collect-score':
        collect_score(args)
    elif args.action == 'fit-arm':
        import sys

        # Run the startup check in an independent process with a fresh optimizer.
        for action in ('fit-smoke', 'fit'):
            subprocess.run([sys.executable, '-u', __file__, action, str(args.runtime), args.arm], check=True)
    else:
        fit(args.runtime, args.arm, smoke=args.action == 'fit-smoke')
