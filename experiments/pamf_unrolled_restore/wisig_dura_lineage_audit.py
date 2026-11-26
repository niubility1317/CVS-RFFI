"""Frozen DURA stage readouts on paired source IQ and a fixed Main split sample."""

import argparse
from copy import deepcopy
from importlib import import_module
from pathlib import Path, PurePosixPath
from time import perf_counter

import numpy as np
import torch

from experiments.cric_dura_comparison import g_lineage
from experiments.cric_dura_comparison.evaluation import handle
from experiments.cric_dura_comparison.io import read, tensor, write
from experiments.pamf_action_restore.models import SCALE
from experiments.pamf_action_restore.operators import soft_map
from experiments.pamf_lora_main.packet_runner import schedule
from experiments.pamf_lora_main.publish import release, release_commit
from experiments.pamf_repl_view_shuffle.runner import activate
from experiments.pamf_residual_restore.runtime import load_coarse
from experiments.pamf_restoration.models import complex_iq, pair
from experiments.pamf_stage2_inference.engine import InferenceEngine

from . import wisig_dura_estimated as workers
from . import wisig_dura_stress as main_eval
from . import wisig_original_dura as replay
from .wisig_preamble_dura_target import historical


RUN = '20261126-diagnostic-wisig-dura-frozen-lineage-m4-r01'
OLD = {
    'action': ('pamf_action_restore', '20261019-phase1-restoration-learned-action-source-s2026092701-r07',
               'learned_operator/final.pt'),
    'r17': ('pamf_unrolled_restore', '20261020-phase1-restoration-unrolled-selfsup-s2026092701-m2-r17',
            'unroll_unshared/native_map_prior/final.pt'),
    'r38': ('pamf_iq_joint_compact', '20261020-phase1-restoration-r17-IQ-joint-compact-s2026092701-m2-r38',
            'dense_half/native_map_prior/final.pt'),
    'r44': ('pamf_r38_compression_methods', '20261020-phase1-restoration-R38-compression-methods-s2026092701-m2-r44',
            'group_pruned/native_map_prior/final.pt'),
    'r46': ('pamf_r44_pruned_refine', '20261020-phase1-restoration-R44-pruned-refine-s2026092701-m2-r46',
            'prune_refine/native_map_prior/final.pt'),
    'r65': ('pamf_r46_delayed_prune', '20261021-phase1-restoration-R46-delayed-prune-s2026092701-m2-r65',
            'prune_after10/native_map_prior/final.pt'),
    'r69': ('pamf_r65_four_route', '20261021-phase1-restoration-R65-four-route-s2026092701-m4-r69',
            'stage2/native_map_prior/final.pt'),
}
ARMS = {'original': None, 'waveform': replay.RUN, 'adaptive': replay.OPERATOR_RUN, 'fast': replay.FAST_RUN}
SPLITS = {'cross_rx': 1536, 'cross_both': 512}
PROFILES = ('default', 'strong')
VIEWS, SEEDS = main_eval.VIEWS, main_eval.SEEDS
BATCH = 96
SWAPS = ('r07_adaptive_action_waveform_variance', 'r07_waveform_action_adaptive_variance',
         'r17_adaptive_context_waveform_prior', 'r17_waveform_context_adaptive_prior')
MODES = ('raw', 'r02', *(f'{arm}_{stage}' for stage in OLD for arm in ARMS), *SWAPS)


def stage_model(rt, arm, stage, device):
    runs = Path(rt['project']) / 'runs'
    if arm == 'original':
        package, run, relative = OLD[stage]
        loader = import_module(f'experiments.{package}.runtime').load_checkpoint
        model, metadata = loader(runs / run / relative, device)
    else:
        base = runs / ARMS[arm]
        frozen = base / 'G'
        if read(frozen / 'complete.json')['status'] != 'SOURCE_FROZEN':
            raise ValueError('Complete estimate-supervised lineage required')
        contract = read(base / 'inputs/source_contract.json')
        if contract['historical_received_epochs'] != str(Path(rt['action_iq']) / 'epochs'):
            raise ValueError('Original R07 source observations differ')
        model, metadata = g_lineage.load(frozen / stage / 'final.pt', contract, 'LU', device)
        parents = [str(frozen / s / 'final.pt') for s in g_lineage.STAGES[:g_lineage.STAGES.index(stage)]]
        if metadata['stage'] != stage or metadata['ancestors'] != parents:
            raise ValueError('Wrong frozen stage lineage')
    if (metadata['epoch'], metadata['updates']) != (20, 4940):
        raise ValueError('Only fixed E20 readouts are allowed')
    return model.eval().requires_grad_(False)


def action_swap(received, action_model, variance_model):
    base = action_model.coarse(received)
    action, _ = action_model.estimator(received, base)
    _, variance = variance_model.estimator(received, base)
    return pair(soft_map(action, complex_iq(received), complex_iq(base), variance, SCALE**2, steps=6))


def readouts(rt, *, target):
    yield 'raw', lambda iq: iq
    coarse, _ = load_coarse(Path(rt['coarse_run']), 'cuda:0')
    yield 'r02', coarse
    del coarse
    retained = {}
    for stage in OLD:
        if target and stage == 'r69':
            continue
        for arm in ARMS:
            model = stage_model(rt, arm, stage, 'cuda:0')
            if stage in ('action', 'r17') and arm in ('adaptive', 'waveform'):
                retained[stage, arm] = model
            if stage == 'r69':
                engine = InferenceEngine(model, reuse=True, graph=True, native=False)
                yield f'{arm}_{stage}', lambda iq: engine(iq)[0]
                del engine
            else:
                yield f'{arm}_{stage}', model
            del model
    a, w = [retained['action', arm] for arm in ('adaptive', 'waveform')]
    yield SWAPS[0], lambda iq: action_swap(iq, a, w)
    yield SWAPS[1], lambda iq: action_swap(iq, w, a)
    a, w = [retained['r17', arm] for arm in ('adaptive', 'waveform')]
    for key, prior, context in ((SWAPS[2], w, a), (SWAPS[3], a, w)):
        model = deepcopy(prior)
        model.frozen = context.frozen
        yield key, model


@torch.no_grad()
def smoke(rt):
    replay.setup_torch()
    source = Path(rt['action_iq'])
    iq = tensor(replay.targets.received(source, 1)[:2], 'cuda:0')
    count = 0
    for name, fn in readouts(rt, target=False):
        wave = fn(iq)
        if wave.shape != iq.shape or not torch.isfinite(wave).all():
            raise FloatingPointError(f'Invalid real-checkpoint smoke output {name}')
        count += 1
    if count != len(MODES):
        raise ValueError('Missing frozen stage readout')
    write(Path(rt['run_root']) / 'smoke.json', dict(status='PASS', query_reads=0, readouts=count))


def source_metrics(wave, clean):
    reference = complex_iq(clean)
    predicted = complex_iq(wave)
    norm = reference.abs().square().mean(-1).clamp_min(1e-4)
    nmse = (predicted-reference).abs().square().mean(-1)/norm
    ref_s = torch.fft.fft(reference, norm='ortho')
    pred_s = torch.fft.fft(predicted, norm='ortho')
    angle = (pred_s*ref_s.conj()).angle()
    power = ref_s.abs().square()
    phase = ((2.-2.*angle.cos())*power).sum(-1)/power.sum(-1).clamp_min(1e-8)
    magnitude = (pred_s.abs()-ref_s.abs()).square().mean(-1)/norm
    return torch.stack((nmse, phase, magnitude), dim=1).cpu().tolist()


@torch.no_grad()
def source(rt):
    replay.setup_torch()
    cache = Path(rt['action_iq'])
    roles = replay.targets.load(cache/'validation', 'roles')
    ids = np.sort(np.random.default_rng(271397).choice(len(roles), 128, replace=False))
    if not np.all(roles[ids] == 'V'):
        raise ValueError('Use the same fixed source validation panel as the R07 probe audit')
    clean = tensor(replay.targets.load(cache/'validation', 'clean')[ids], 'cuda:0')
    views = {k: tensor(replay.targets.load(cache/'validation_actions'/k, 'received')[ids], 'cuda:0') for k in VIEWS}
    results = {}
    for name, fn in readouts(rt, target=False):
        results[name] = {}
        for view, iq in views.items():
            parts = [source_metrics(fn(inp), ref) for inp, ref in zip(iq.split(BATCH), clean.split(BATCH))]
            values = np.concatenate(parts)
            if not np.isfinite(values).all():
                raise FloatingPointError(f'Nonfinite source stage metric {name}')
            results[name][view] = dict(mean=values.mean(axis=0).tolist(), p90=np.quantile(values,.9,axis=0).tolist())
        print('FROZEN_STAGE_SOURCE', name, flush=True)
    write(Path(rt['run_root']) / 'source.json', dict(status='SOURCE_ANALYZED', packets=128, target_reads=0,
          metrics=['IQ_NMSE', 'energy_weighted_phase_chord_error', 'spectral_magnitude_NMSE'], results=results))


def packet_selection(rt, reader, split):
    manifest, ids, positions = handle(main_eval.view_root(rt, reader, 'full', split), reader['classes'])
    if len(ids) != main_eval.COUNTS[split] or manifest['channel'] != 'full/post_sync/noeq' or positions is None:
        raise ValueError('Expected the exact admitted Full Main split views')
    seed = 2026112603 + tuple(SPLITS).index(split)
    take = np.sort(np.random.default_rng(seed).choice(len(ids), SPLITS[split], replace=False))
    return manifest, ids, positions, take


def frozen_decisions(rt, arm, profile, split, ids, take):
    run = replay.RUN if arm == 'original' else ARMS[arm]
    folder = Path(rt['project']) / 'runs' / run / 'predictions' / f'{profile}-full-{split}'
    marker = read(folder / 'complete.json')
    expected = ('PREDICTIONS_FROZEN', profile, 'full', split, len(ids), list(SEEDS), list(VIEWS), False)
    if tuple(marker[k] for k in ('status','profile','route','split','count','seeds','views','truth_read')) != expected:
        raise ValueError('Expected the exact completed WiSig Main comparison rows')
    arm_keys = dict(original='R69', waveform='trained_adaptive', adaptive='trained_adaptive_operator',
                    fast='trained_fast_mixture_operator')
    with np.load(folder / 'decisions.npz', allow_pickle=False) as data:
        if not np.array_equal(data['ids'], ids):
            raise ValueError('Packet order differs from the completed comparison')
        own = np.stack([data[f'{arm_keys[arm]}__{view}'][:,take] for view in VIEWS], axis=1)
        raw = np.stack([data[f'raw__{view}'][:,take] for view in VIEWS], axis=1)
    return own, raw


@torch.no_grad()
def predict(rt, profile):
    activate(rt)
    replay.setup_torch()
    originals, heads = main_eval.readers(rt)
    reader = originals[SEEDS[0]]
    strong = Path(rt['strong_view_root'])
    if profile == 'strong' and read(strong/'complete.json') != dict(status='IQ_COMPLETE', count=168000, truth_read=False):
        raise ValueError('Missing frozen strong IQ observations')
    inputs, decisions, ids_by_split = {}, {}, {}
    for split in SPLITS:
        manifest, ids, positions, take = packet_selection(rt, reader, split)
        ids_by_split[split] = ids[take]
        rows = {}
        for view in VIEWS:
            if view == 'clean':
                path = Path(manifest['clean_ref'])
            elif profile == 'strong':
                path = strong/'full'/f'{view}.npy'
            else:
                path = Path(manifest['observations_root'])/f'{view}.npy'
            iq = np.load(path, mmap_mode='r', allow_pickle=False)
            if iq.dtype != np.float32 or iq.shape[1:] != (2,256):
                raise ValueError('IQ representation differs from the completed comparison')
            rows[view] = tensor(iq[positions[take]], 'cuda:0')
        inputs[split] = rows
        decisions[split] = {}
        for arm in ARMS:
            final, raw = frozen_decisions(rt, arm, profile, split, ids, take)
            decisions[split][f'{arm}_r69'] = final
            if 'raw' in decisions[split] and not np.array_equal(raw, decisions[split]['raw']):
                raise ValueError('Raw controls disagree between frozen comparison runs')
            decisions[split]['raw'] = raw
    for name, fn in readouts(rt, target=True):
        if name == 'raw':
            continue
        started = perf_counter()
        for split, views in inputs.items():
            results = []
            for iq in views.values():
                pieces = [np.asarray(main_eval.classify(heads, fn(batch)), dtype=np.uint8) for batch in iq.split(BATCH)]
                results.append(np.concatenate(pieces, axis=1))
            decisions[split][name] = np.stack(results, axis=1)
        print('FROZEN_STAGE_TARGET', profile, name, round(perf_counter()-started,2), flush=True)
    folder = Path(rt['run_root']) / 'predictions' / profile
    folder.mkdir(parents=True)
    for split, rows in decisions.items():
        if set(rows) != set(MODES) or any(q.shape != (len(SEEDS),len(VIEWS),SPLITS[split]) for q in rows.values()):
            raise ValueError('The complete lineage prediction matrix is required')
        with (folder/f'{split}.npz').open('xb') as out:
            np.savez(out, ids=ids_by_split[split], **rows)
    write(folder/'complete.json', dict(status='PREDICTIONS_FROZEN', profile=profile, splits=SPLITS,
          modes=list(MODES), seeds=list(SEEDS), views=list(VIEWS), truth_read=False))


def interval(first, second, rng):
    boot = np.zeros(1600)
    size = sum(len(x) for x in first)
    for a,b in zip(first,second,strict=True):
        gap = b-a
        rows = rng.integers(len(gap), size=(len(boot),len(gap)))
        boot += gap[rows].sum(axis=1)/size
    return (100*np.quantile(boot,[.025,.975])).tolist()


def score(rt):
    root = Path(rt['run_root'])
    original = historical(Path(rt['recognizer_run']), 'predict', SEEDS[0])
    all_rows = {}
    for profile in PROFILES:
        folder = root/'predictions'/profile
        expected = dict(status='PREDICTIONS_FROZEN', profile=profile, splits=SPLITS,
                        modes=list(MODES), seeds=list(SEEDS), views=list(VIEWS), truth_read=False)
        if read(folder/'complete.json') != expected:
            raise ValueError('Independent scorer requires both fully frozen profile matrices')
        all_rows[profile] = {}
        for split in SPLITS:
            _, ids, _, take = packet_selection(rt, original, split)
            with np.load(folder/f'{split}.npz',allow_pickle=False) as data:
                if not np.array_equal(data['ids'],ids[take]) or set(data.files) != {'ids',*MODES}:
                    raise ValueError('Frozen predictions do not match this diagnostic sample')
                all_rows[profile][split] = {k:data[k] for k in data.files}
    truth = historical(Path(rt['recognizer_run']), 'score', SEEDS[0])
    labels = {}
    for split in SPLITS:
        entries = read(truth['tests'][split]['truth'])
        labels[split] = np.asarray([entries[str(i)]['label'] for i in all_rows['default'][split]['ids']])
    rng = np.random.default_rng(2026112604)
    results = {}
    for profile, parts in all_rows.items():
        good = {name:[parts[split][name] == labels[split] for split in SPLITS] for name in MODES}
        results[profile] = dict(accuracy={name:(100*np.concatenate(v,axis=-1).mean(axis=(0,2))).tolist()
                                           for name,v in good.items()}, paired={})
        for stage in OLD:
            comparisons = [(f'{arm}_{stage}',f'{other}_{stage}') for arm,other in (
                ('waveform','adaptive'),('original','adaptive'),('original','fast'),('original','waveform'))]
            if stage == 'action':
                comparisons.extend(('waveform_action',k) for k in SWAPS[:2])
            elif stage == 'r17':
                comparisons.extend(('waveform_r17',k) for k in SWAPS[2:])
            for a,b in comparisons:
                row = {}
                for label, take in [('practical_mean',slice(1,None)),('low_urban', [VIEWS.index('practical_low_urban')])]:
                    x,y = ([part[:,take].mean(axis=(0,1)) for part in good[k]] for k in (a,b))
                    row[label] = dict(gap_pp=100*float((np.concatenate(y)-np.concatenate(x)).mean()),
                                      ci95_pp=interval(x,y,rng))
                results[profile]['paired'][f'{a}_to_{b}'] = row
    write(root/'scores.json',dict(status='SCORED', splits=SPLITS, views=list(VIEWS), seeds=list(SEEDS),
          truth_last=True, diagnostic_only=True, results=results))


def dispatch(rt):
    root = Path(rt['run_root'])
    root.mkdir()
    (root/'logs').mkdir()
    state = root/'state.json'
    try:
        allowed,_ = workers.gpu_selection(minimum=0)
        workers_rt = dict(rt, allowed_gpus=allowed, max_workers=2, max_compute_processes=None)
        write(state,dict(status='SOURCE_SMOKE'))
        schedule(workers_rt,[('smoke','lineage')],workers.launch)
        write(state,dict(status='DIAGNOSTICS_RUNNING'),exclusive=False)
        schedule(workers_rt,[('predict',p) for p in PROFILES]+[('source','lineage')],workers.launch)
        workers.complete(workers.launch(rt,'score','cpu'),'independent sample truth-last scoring')
        write(state,dict(status='ANALYZED'),exclusive=False)
    except Exception as e:
        write(state,dict(status='FAILED',error=str(e)),exclusive=False)
        raise


def publish(repo, output, context, ssh_config):
    rt = read(context)
    rt = {key:rt[key] for key in ('project','python','frozen_sources')}
    project = PurePosixPath(rt['project'])
    runs = project/'runs'
    paths = dict(action_iq=str(runs/OLD['action'][1]/'cache'), coarse_run=str(runs/replay.coarse_design.PARENT_RUN),
                 baseline_root=str(runs/main_eval.BASELINE), recognizer_run=str(runs/main_eval.F_RUN),
                 strong_view_root=str(runs/main_eval.RUN/'strong_target_iq'))
    input_paths = [*paths.values(), *(str(runs/run/path) for _,run,path in OLD.values()),
                   *(str(runs/r/'G/complete.json') for r in ARMS.values() if r)]
    rt.update(run_id=RUN, run_root=str(runs/RUN), release_root=str(project/'releases'/f'{RUN}-release'),
              runner=__spec__.name, code_commit=release_commit(repo), external_releases=input_paths, **paths)
    release(repo,output,rt,ssh_config,archive_extras=(
        'tools/wisig_paired_main_uncertainty.py','tools/r17_activation_factorization.py'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='mode',required=True)
    pub = commands.add_parser('publish')
    for field in ('repo','output','context','ssh-config'):
        pub.add_argument('--'+field,type=Path,required=True)
    for mode in ('dispatch','smoke','source','predict','score'):
        job = commands.add_parser(mode)
        job.add_argument('runtime',type=Path)
        job.add_argument('name',nargs='?')
    args = parser.parse_args()
    if args.mode == 'publish':
        return publish(args.repo,args.output,args.context,args.ssh_config)
    rt = read(args.runtime)
    if args.mode == 'predict':
        return predict(rt,args.name)
    return dict(dispatch=dispatch,smoke=smoke,source=source,score=score)[args.mode](rt)


if __name__ == '__main__':
    main()
