"""Fixed R17 and probe real-gain mixture: preregistered target receiver screen."""

import argparse
from pathlib import Path, PurePosixPath
import os
import subprocess
import sys
from time import monotonic

import numpy as np
import torch

from experiments.cric_dura_comparison.io import read, write
from experiments.pamf_lora_main.dura_real_gain_fit import DEVICE, as_complex, wisig_models, wisig_real_prior
from experiments.pamf_lora_main.dura_real_gain_fusion import RealSpectralGain
from experiments.pamf_lora_main.dura_real_gain_amplitude import R17AmplitudeGain
from experiments.pamf_lora_main.dura_real_gain_packet2d import PacketR17Gain
from experiments.pamf_lora_main.dura_real_gain_unroll import fused_unroll
from experiments.pamf_lora_main.dura_real_gain_wisig_dura_screen import load_dura, R69_RUN, R69_RELEASE
from experiments.pamf_restoration.models import complex_iq, pair
from experiments.pamf_lora_main.dura_real_gain_wisig_joint import original_student
from experiments.pamf_lora_main.dura_real_gain_run import free_gpus
from experiments.pamf_lora_main.dura_real_gain_screen import load_gain, packet_scale, paired


RUN = '20261128-phase1-wisig-r17-probe-target-s2026113137-r01'
DURA_MIX_SCREEN = '20261128-phase1-wisig-dura-probe-mix-s2026113121-r01'
MULTISCENE_SCREEN = '20261128-phase1-wisig-r07-multiscene-probe-s2026113105-r01'
CONTINUED_SCREEN = '20261128-phase1-wisig-r07-probe-continue-s2026113097-r01'
PROBE_PARENT = '20261128-phase1-wisig-r07-probe-action-s2026113073-r01'
PACKET2D_SCREEN = '20261128-phase1-wisig-r07-packet2d-s2026113065-r01'
PAIRED_SCREEN = '20261128-phase1-wisig-r07-paired-residual-s2026113049-r01'
ACTION_CACHE_RUN = '20261019-phase1-restoration-learned-action-source-s2026092701-r07'
CONTINUE_SCREEN = '20261128-phase1-wisig-r07-action-continue-s2026112945-r01'
COARSE_SCREEN = '20261128-phase1-wisig-r02-coarse-joint-s2026112953-r01'
CONTRAST_SCREEN = '20261128-phase1-wisig-r07-packet-contrast-s2026112969-r01'
GATE_SCREEN = '20261128-phase1-wisig-r17-packet-gate-s2026112985-r01'
CONSENSUS_SCREEN = '20261128-phase1-wisig-r17-consensus-s2026113009-r01'
GROUP_SCREEN = '20261128-phase1-wisig-r07-group-robust-s2026113025-r01'
STRONG_SCREEN = '20261128-phase1-wisig-r07-action-strong-s2026112937-r01'
ACTION_PARENT = '20261128-phase1-wisig-r07-action-joint-s2026112925-r01'
JOINT_SCREEN = '20261128-phase1-wisig-r17-internal-joint-s2026112917-r01'
INTERNAL_SCREEN = '20261128-phase1-wisig-r17-internal-gain-s2026112909-r01'
AMPLITUDE_SCREEN = '20261128-phase1-wisig-r17-amplitude-gain-s2026112905-r01'
URBAN_PARENT = '20261128-phase1-wisig-r17-urban-source-s2026112893-r01'
DOSE_SCREEN = '20261128-phase1-wisig-r17-gain-dose-s2026112901-r01'
PREVIOUS_SCREEN = '20261128-phase1-wisig-r17-full-source-s2026112889-r01'
PARENT = '20261128-phase1-dura-realgain-crossdata-s2026112845-r02'
SEED, STEPS, BATCH, RX = 2026113137, 1500, 14, '1-1'
VIEWS = ('clean', 'practical_low_urban')
MODES = ('direct', 'wisig_r17', 'dura_v1', 'real_gain', 'prior_fusion', 'urban_fusion', 'action_fusion', 'probe_fusion', 'fusion')


def spectral_fusion(model, observed, prior):
    return prior + model(observed.unflatten(-1, (8, 32)), prior.unflatten(-1, (8, 32))).flatten(-2)


def setup(rt):
    from experiments.cric_dura_comparison.bootstrap import numerical_policy
    numerical_policy()
    project = Path(rt['project'])
    source, ids, r17 = wisig_models(project, rt['frozen_sources'])
    prior = wisig_real_prior(project)
    control = load_gain(project / 'runs' / PARENT, 'wisig', real=True)
    old_fusion = load_gain(project / 'runs' / PARENT, 'wisig')

    def real_gain(y):
        observed, estimate, scale = prior(y)
        return ((estimate + control(observed, estimate)) * scale).flatten(-2)

    return source, ids, r17, real_gain, old_fusion


def select_fresh_rows(ids, rx, day, previous_ids):
    result = []
    for value in (1, 2, 3):
        eligible = np.flatnonzero((rx == RX) & (day == value) & ~np.isin(ids, previous_ids))
        if len(eligible) < 100:
            raise ValueError('Insufficient source V packets outside previous screens')
        result.extend(np.random.default_rng(SEED + value).choice(eligible, 100, replace=False).tolist())
    return np.asarray(sorted(result), dtype=np.int64)


def urban_training_indices(unlabeled, quantile, scenes):
    if quantile.shape != scenes.shape:
        raise ValueError('Source observation metadata shapes differ')
    return np.intersect1d(unlabeled, np.flatnonzero((quantile >= .15) & (scenes == 5)))


def configure_action(model):
    model.requires_grad_(False)
    model.frozen.estimator.requires_grad_(True)
    model.eval()
    return model


@torch.no_grad()
def predict(rt):
    import importlib
    project, root = Path(rt['project']), Path(rt['run_root'])
    pipeline = load_dura(project)
    source, source_ids, r17, real_gain, old_fusion = setup(rt)
    from leo_practical.channel import normalize_input
    from experiments.pamf_reception_restore.data import shared_pair
    original = importlib.import_module('experiments.pamf_unrolled_restore.wisig_anchored_window_source')
    reference = project / 'releases/20261125-phase1-wisig-native-fir-source-m4-r01-release'
    config = read(reference / 'runtime.json')
    reader = original.tri.reader_factory(config).eval().requires_grad_(False)
    parent_root = project / 'runs' / URBAN_PARENT
    parent = torch.load(parent_root / 'gain.pt', map_location=DEVICE, weights_only=True)
    parent_proof = read(parent_root / 'fit.json')
    if parent['proof'] != parent_proof or parent_proof['status'] != 'SOURCE_FROZEN' or any(
            parent_proof[k] for k in ('F_reads', 'TX_reads', 'V_reads', 'target_reads')):
        raise ValueError('Expected source-only urban real-gain control')
    urban = RealSpectralGain(32).to(DEVICE).eval().requires_grad_(False)
    urban.load_state_dict(parent['weights'], strict=True)
    action_root = project / 'runs' / ACTION_PARENT
    action_parent = torch.load(action_root / 'gain.pt', map_location=DEVICE, weights_only=True)
    action_proof = read(action_root / 'fit.json')
    if action_parent['proof'] != action_proof or action_proof['status'] != 'SOURCE_FROZEN' or any(
            action_proof[k] for k in ('F_reads', 'TX_reads', 'V_reads', 'target_reads')):
        raise ValueError('Expected lawful source-only action fusion control')
    old_r17 = original_student(project).eval().requires_grad_(False)
    old_r17.frozen.estimator.load_state_dict(action_parent['estimator'], strict=True)
    old_gain = R17AmplitudeGain(32).to(DEVICE).eval().requires_grad_(False)
    old_gain.load_state_dict(action_parent['weights'], strict=True)
    probe_root = project / 'runs' / PROBE_PARENT
    probe_parent = torch.load(probe_root / 'gain.pt', map_location=DEVICE, weights_only=True)
    probe_proof = read(probe_root / 'fit.json')
    if (probe_parent['proof'] != probe_proof or (probe_proof['status'], probe_proof['seed'], probe_proof['updates']) !=
            ('SOURCE_FROZEN', 2026113073, 1500) or any(probe_proof[k] for k in ('F_reads', 'TX_reads', 'V_reads', 'target_reads'))):
        raise ValueError('Expected lawful source-only probe parent')
    probe_r17 = original_student(project).eval().requires_grad_(False)
    probe_r17.frozen.estimator.load_state_dict(probe_parent['estimator'], strict=True)
    probe_gain = PacketR17Gain(32).to(DEVICE).eval().requires_grad_(False)
    probe_gain.load_state_dict(probe_parent['weights'], strict=True)

    def classify(z):
        return np.asarray(original.classify(reader, z).argmax(1).cpu().tolist(), dtype=np.int16)

    def decisions(batch):
        y = as_complex(batch)
        scale = packet_scale(y)
        n = y / scale
        p, action = r17(n)
        r = real_gain(n)
        dura, = pipeline(pair(y))
        return np.stack([classify(z) for z in (y, r17(y)[0], complex_iq(dura), r * scale,
                         old_fusion(n, p, action) * scale, spectral_fusion(urban, n, p) * scale,
                         fused_unroll(old_r17, old_gain, n, r) * scale,
                         fused_unroll(probe_r17, probe_gain, n, r) * scale,
                         .5 * (r17(y)[0] + fused_unroll(probe_r17, probe_gain, n, r) * scale))])

    if decisions(source[source_ids[:2]]).shape != (len(MODES), 2):
        raise ValueError('No-query frozen checkpoint smoke failed')
    reference = project / 'runs/20261019-phase1-restoration-formal-frozen-residual-s2026092701-r05/reference'
    manifest = read(reference / 'manifest.json')
    if manifest['status'] != 'FORMAL_REFERENCE_READY' or manifest['count'] != 168000 or not manifest['original_IQ_bytes_reused']:
        raise ValueError('Expected original registered received observations')
    all_ids = np.load(reference / 'physical_ids.npy', allow_pickle=False)
    # Receiver metadata is stored alongside labels; this preparatory access extracts only receiver.
    # No TX labels, IQ, predictions or performance statistics influence sampling.
    truth = read(project / 'runs/20260927-phase1-baselines-final-clean-satellite-m5-r01/data/truth.json')
    eligible = np.asarray([i for i, key in enumerate(all_ids) if str(truth[str(key)]['receiver']) == RX])
    del truth
    if len(eligible) != 24000:
        raise ValueError('Registered receiver coverage differs')
    selected = np.sort(np.random.default_rng(SEED + 2).choice(eligible, 300, replace=False))
    write(root / 'panel.json', dict(status='FIXED_BEFORE_PREDICTION', receiver=RX,
          count=300, seed=SEED+2, physical_ids=all_ids[selected].tolist(),
          selection='uniform among receiver IDs, no TX label or signal use',
          receiver_metadata='receiver field only extracted from historical truth index',
          historic_formal_exposure=True))
    outputs = {}
    for view in VIEWS:
        received = np.load(reference / (view + '.npy'), mmap_mode='r', allow_pickle=False)
        if received.shape != (len(all_ids), 2, 256):
            raise ValueError('Registered received observations differ')
        values = received[selected]
        outputs[view] = np.concatenate([decisions(values[i:i+48]) for i in range(0, 300, 48)], axis=1)
        print('WISIG_TARGET_SCREEN', view, 300, flush=True)
    with (root / 'screen.npz').open('xb') as stream:
        np.savez(stream, ids=all_ids[selected], **outputs)
    write(root / 'screen-frozen.json', dict(status='PREDICTIONS_FROZEN', modes=list(MODES), views=list(VIEWS),
          count=len(selected), target_RX=RX, TX_truth_reads=0, target_signal_reads=600, receiver_metadata_from_truth_index=True))


def score(rt):
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('Independent CPU scorer required')
    root, project = Path(rt['run_root']), Path(rt['project'])
    frozen = read(root / 'screen-frozen.json')
    if (frozen['status'], frozen['modes'], frozen['views'], frozen['count'], frozen['TX_truth_reads']) != (
            'PREDICTIONS_FROZEN', list(MODES), list(VIEWS), 300, 0):
        raise ValueError('Both views must freeze before source truth')
    with np.load(root / 'screen.npz', allow_pickle=False) as archive:
        ids = archive['ids'].copy()
        predictions = {view: archive[view].copy() for view in VIEWS}
    if len(ids) != 300 or any(value.shape != (len(MODES), 300) for value in predictions.values()):
        raise ValueError('Incomplete source predictions')
    truth = read(project / 'runs/20260927-phase1-baselines-final-clean-satellite-m5-r01/data/truth.json')
    labels = np.asarray([truth[str(x)]['label'] for x in ids], dtype=np.int64)
    permutation = np.random.default_rng(SEED + 3).permutation(len(ids))
    hard, clean = predictions[VIEWS[-1]], predictions['clean']
    pairs = {name: paired(hard[-1], hard[i], labels, permutation, 2) for i, name in enumerate(MODES[:-1])}
    clean_net = {name: int(((clean[-1] == labels).astype(np.int16) - (clean[i] == labels).astype(np.int16)).sum())
                 for i, name in enumerate(MODES[:-1])}
    promote = all(v['passes'] for v in pairs.values()) and min(clean_net.values()) >= -2
    write(root / 'screen-score.json', dict(status='TARGET_SCREEN_SCORED', truth_last=True, count=len(ids),
          modes=list(MODES), correct={v: (x == labels).sum(1).tolist() for v, x in predictions.items()},
          low_urban_pairs=pairs, clean_net=clean_net, promote=promote,
          missing_controls=[], formal_acceptance_complete=False,
          exposure='Historical full target benchmark exposed; target receiver metadata accessed only for panel selection; TX truth independently connected after both views and all modes froze. Screening IDs must be excluded from any held-out subset analysis.'))


def dispatch(runtime):
    rt = read(runtime)
    root = Path(rt['run_root'])
    root.mkdir(exist_ok=False)
    state = root / 'state.json'
    try:
        available = free_gpus()
        if not available:
            raise RuntimeError('No GPU with 8 GiB available')
        gpu = next(value for value in (7, 2, 3, 6, 1, 4, 0, 5) if value in available)
        for action in ('predict', 'score'):
            device = None if action == 'score' else gpu
            env = dict(os.environ, CUDA_VISIBLE_DEVICES='' if device is None else str(device),
                       OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='2')
            args = [sys.executable, '-u', '-m', __spec__.name, action, str(runtime)]
            with (root / f'{action}.log').open('x', encoding='utf-8') as log:
                proc = subprocess.Popen(args, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
            write(state, dict(status=action.upper(), pid=proc.pid, gpu=device), exclusive=False)
            print('WISIG_R17_PROBE_MIX_WORKER', action, proc.pid, device, flush=True)
            if proc.wait():
                raise RuntimeError(f'{action} failed; preserve partial artifacts')
        write(state, dict(status='TARGET_SCREEN_SCORED', gpu=gpu, promote=read(root / 'screen-score.json')['promote']),
              exclusive=False)
    except Exception as error:
        write(state, dict(status='FAILED', error=str(error)), exclusive=False)
        raise


def publish(repo, output, context, ssh_config):
    from experiments.pamf_lora_main.publish import release, release_commit
    source = read(context)
    rt = {key: source[key] for key in ('project', 'python', 'frozen_sources')}
    project = PurePosixPath(rt['project'])
    runs, releases = project / 'runs', project / 'releases'
    rt.update(run_id=RUN, runner=__spec__.name, code_commit=release_commit(repo),
              release_root=str(releases / f'{RUN}-release'), run_root=str(runs / RUN),
              external_releases=[str(runs / PARENT), str(runs / PREVIOUS_SCREEN), str(runs / URBAN_PARENT), str(runs / DOSE_SCREEN), str(runs / AMPLITUDE_SCREEN), str(runs / INTERNAL_SCREEN), str(runs / JOINT_SCREEN), str(runs / ACTION_PARENT), str(runs / STRONG_SCREEN), str(runs / CONTINUE_SCREEN), str(runs / COARSE_SCREEN), str(runs / CONTRAST_SCREEN), str(runs / GATE_SCREEN), str(runs / CONSENSUS_SCREEN), str(runs / GROUP_SCREEN), str(runs / PAIRED_SCREEN), str(runs / PACKET2D_SCREEN), str(runs / PROBE_PARENT), str(runs / CONTINUED_SCREEN), str(runs / MULTISCENE_SCREEN), str(runs / DURA_MIX_SCREEN), str(runs / '20261019-phase1-restoration-formal-frozen-residual-s2026092701-r05/reference'), str(runs / '20260927-phase1-baselines-final-clean-satellite-m5-r01/data/truth.json'), str(runs / ACTION_CACHE_RUN), str(runs / R69_RUN), str(releases / R69_RELEASE),
                 str(releases / '20261125-phase1-wisig-native-fir-source-m4-r01-release')])
    release(repo, output, rt, ssh_config, archive_extras=('tools/r17_activation_factorization.py',))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='action', required=True)
    for action in ('dispatch', 'predict', 'score'):
        commands.add_parser(action).add_argument('runtime', type=Path)
    pub = commands.add_parser('publish')
    for field in ('repo', 'output', 'context', 'ssh-config'):
        pub.add_argument('--' + field, type=Path, required=True)
    args = vars(parser.parse_args())
    action = args.pop('action')
    if action == 'publish':
        publish(**args)
    elif action == 'dispatch':
        dispatch(**args)
    else:
        {'predict': predict, 'score': score}[action](read(args['runtime']))
