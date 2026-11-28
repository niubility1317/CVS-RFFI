"""Truth-last target pilot of one frozen common physical-solve candidate."""
import argparse
import importlib
import os
from pathlib import Path, PurePosixPath
import subprocess
import sys
import time
import zipfile

import numpy as np
import torch

from experiments.cric_dura_comparison.io import read, write
from experiments.pamf_lora_main.dura_real_gain_fit import DEVICE, LORA_GAIN, WISIG_PARENT, as_complex
from experiments.pamf_lora_main.shared_dura_realgain_run import REFERENCE
from experiments.pamf_lora_main.dura_real_gain_run import free_gpus
from experiments.pamf_lora_main import shared_twoprior_run as shared
from experiments.pamf_lora_main import shared_physicalsolve_run as candidate
from experiments.pamf_lora_main.shared_practical_urban_run import source_teacher
from experiments.pamf_restoration.models import complex_iq, pair

RUN = '20261128-phase1-shared-physicalsolve-target-pilot-s2026112957-r01'
PARENT = '20261128-phase1-shared-physicalsolve-gain-s2026112957-r01'
VIEWS = ('clean', 'practical_low_urban')
MODES = ('direct', 'r17', 'real_gain', 'dura', 'fusion')
LORA_RX, LORA_N = 'b200_2', 200
WISIG_RX, WISIG_N = 0, 2000
SEED = 2026112852


def modes_for(dataset):
    return ('direct', 'r17', 'real_gain', 'lora_r69_adapt' if dataset == 'lora' else 'dura_v1', 'fusion')


def bind(rt, dataset):
    source, ids, original, gain = shared.experts(rt, dataset)
    teacher = source_teacher(rt, dataset)
    project = Path(rt['project'])
    folder = project/'runs'/PARENT/dataset
    proof = read(folder/'fit.json')
    frozen = torch.load(folder/'gain.pt', map_location=DEVICE, weights_only=True)
    if (frozen['proof'] != proof or proof['status'] != 'SOURCE_FROZEN' or
            (proof['dataset'], proof['seed'], proof['updates']) != (dataset, candidate.SEED, candidate.STEPS) or
            any(proof[k] for k in ('TX_reads', 'V_reads', 'target_reads'))):
        raise ValueError('Expected the preregistered source-frozen physical-solve candidate')
    dura_model, last, trained = candidate.load_trainable_r17(project, dataset)
    last.load_state_dict(torch.load(folder/'prior_last.pt', map_location=DEVICE, weights_only=True), strict=True)
    dura_model.eval().requires_grad_(False)
    fusion = candidate.SharedPhysicalSolveGain(shared.WIDTH[dataset], anchor_real=candidate.ANCHOR_REAL[dataset], correction_cap=.05).to(DEVICE)
    fusion.load_state_dict(frozen['weights'], strict=True)
    fusion.eval().requires_grad_(False)
    if dataset == 'lora':
        from experiments.pamf_lora_main.lora_r69_w24_source_control import boot, restore_packets
        _, _, _, _, gate, model = boot(rt)

        def dura(z):
            cpu = np.asarray(z.detach().contiguous().view(torch.float32).tolist(), dtype=np.float32).view(np.complex64)
            restored, _ = restore_packets(cpu, gate, model)
            return as_complex(restored)
    else:
        module = importlib.import_module('experiments.pamf_unrolled_restore.wisig_short_training_source')
        engine = module.original_r69(project/'runs'/WISIG_PARENT)

        def dura(z):
            recovered, = engine(pair(z))
            return complex_iq(recovered)

    def decisions(raw):
        z = as_complex(raw)
        scale = z.abs().square().mean(-1, keepdim=True).clamp_min(1e-12).sqrt()
        normalized = z/scale
        real_normalized = gain(normalized)
        real = gain(z) if dataset == 'lora' else real_normalized*scale
        r17, _ = original(z)
        physical, action = trained(normalized)
        restored = fusion(normalized, real_normalized, physical, action)*scale
        signals = (z, r17, real, dura(z), restored)
        outputs = [np.asarray(teacher(x).argmax(1).detach().cpu().tolist(), dtype=np.int16) for x in signals]
        if any(a.shape != (len(z),) for a in outputs):
            raise ValueError('Invalid frozen default-head output')
        return np.stack(outputs)
    return source, ids, decisions


@torch.no_grad()
def predict(rt, dataset):
    from experiments.cric_dura_comparison.bootstrap import numerical_policy
    numerical_policy()
    root, project = Path(rt['run_root']), Path(rt['project'])
    source, ids, decide = bind(rt, dataset)
    sample = as_complex(source[np.asarray(ids[:2])])
    if decide(sample).shape != (len(MODES), 2):
        raise ValueError('Real source-U no-query checkpoint smoke failed')
    print('TARGET_PILOT_REAL_CHECKPOINT_SMOKE', dataset, 'PASS', flush=True)
    values = {}
    if dataset == 'lora':
        from experiments.pamf_lora_main import w24_joint_pamf as joint
        target = project/'releases'/'20261122-phase1-lora-full-aligned-data-r01-release'/'data'/'target'
        joint.target_contract(target)
        raw = joint.native(target, LORA_RX, joint.LIMIT)[:LORA_N]
        selected_ids = np.arange(LORA_N, dtype=np.int64)
        for view in VIEWS:
            observations = np.stack([joint.channel_job(view, LORA_RX, (i, packet))
                                     for i, packet in enumerate(raw)])
            values[view] = np.concatenate([decide(observations[i:i+4]) for i in range(0, LORA_N, 4)], axis=1)
            print('TARGET_PILOT_FROZEN_VIEW', dataset, view, LORA_N, flush=True)
    else:
        from experiments.cric_dura_comparison.evaluation import handle
        parent = project/'runs'/'20261123-phase1-wisig-r69-paired-main-m6-r01'/'full_views'/'cross_rx'
        cfg = read(project/'runs'/'20261027-phase1-pamf-repl-shuffled-views-m16-r02'/'configs'/
                   'predict__main_split-repl-shuffled_time-3view-s2026092701.json')
        manifest, all_ids, positions = handle(parent, cfg['classes'])
        if manifest['channel'] != 'full/post_sync/noeq' or WISIG_RX not in manifest['target_rxs']:
            raise ValueError('Wrong target receiver or full observation contract')
        selected_ids = all_ids[:WISIG_N].copy()
        selection = np.arange(WISIG_N) if positions is None else positions[:WISIG_N]
        for view in VIEWS:
            path = Path(manifest['clean_ref']) if view == 'clean' else Path(manifest['observations_root'])/(view+'.npy')
            observed = np.load(path, mmap_mode='r', allow_pickle=False)
            if observed.shape[1:] != (2, 256) or observed.dtype != np.float32:
                raise ValueError('Wrong registered target IQ')
            values[view] = np.concatenate([decide(observed[selection[i:i+32]]) for i in range(0, WISIG_N, 32)], axis=1)
            print('TARGET_PILOT_FROZEN_VIEW', dataset, view, WISIG_N, flush=True)
    folder = root/dataset
    with (folder/'predictions.npz').open('xb') as stream:
        np.savez(stream, ids=selected_ids, **values)
    write(folder/'frozen.json', dict(status='PREDICTIONS_FROZEN', dataset=dataset, views=list(VIEWS),
        modes=list(modes_for(dataset)), count=len(selected_ids), receiver=LORA_RX if dataset == 'lora' else WISIG_RX,
        receiver_selection='fixed first 200 blind indices' if dataset == 'lora' else
                           'RX0 among fixed first 2000 blind-index positions, group only after freeze',
        truth_read=False, target_fit=False, candidate=PARENT))


def frozen_pair(root):
    prepared = {}
    for dataset, n in (('lora', LORA_N), ('wisig', WISIG_N)):
        folder = root/dataset
        mark = read(folder/'frozen.json')
        if (mark['status'], mark['dataset'], mark['views'], mark['modes'], mark['count'],
                mark['truth_read'], mark['target_fit'], mark['candidate']) != (
                'PREDICTIONS_FROZEN', dataset, list(VIEWS), list(modes_for(dataset)), n, False, False, PARENT):
            raise ValueError('Both complete two-view target prediction sets must freeze first')
        with np.load(folder/'predictions.npz', allow_pickle=False) as file:
            ids = file['ids'].copy()
            values = {view: file[view].copy() for view in VIEWS}
        if len(set(ids.tolist())) != n or any(x.shape != (len(MODES), n) for x in values.values()):
            raise ValueError('Invalid frozen target matrix')
        if (mark.get('receiver') != (LORA_RX if dataset == 'lora' else WISIG_RX) or
                (dataset == 'lora' and not np.array_equal(ids, np.arange(LORA_N)))):
            raise ValueError('Frozen target physical selection differs')
        prepared[dataset] = (ids, values)
    return prepared


def score_values(ids, values, labels, rx, days, dataset):
    if len(ids) != len(labels) or not len(ids):
        raise ValueError('Invalid target truth coverage')
    rng = np.random.default_rng(SEED)
    draws = rng.integers(0, len(ids), (2000, len(ids)))
    halves = np.array_split(rng.permutation(len(ids)), 2)
    results = {}
    for view, matrix in values.items():
        hit = matrix == labels
        pairs = {}
        for i, name in enumerate(modes_for(dataset)[:-1]):
            delta = hit[-1].astype(np.int8)-hit[i].astype(np.int8)
            pairs[name] = dict(repaired=int((hit[-1] & ~hit[i]).sum()), harmed=int((~hit[-1] & hit[i]).sum()),
                net=int(delta.sum()), half_nets=[int(delta[x].sum()) for x in halves],
                ci95=np.quantile(delta[draws].sum(1), (.025, .975)).tolist())
        results[view] = dict(correct=hit.sum(1).tolist(), accuracy=hit.mean(1).tolist(), pairs=pairs,
            by_tx={str(t): dict(n=int((labels==t).sum()), correct=hit[:, labels==t].sum(1).tolist()) for t in np.unique(labels)},
            by_day={str(d): dict(n=int((days==d).sum()), correct=hit[:, days==d].sum(1).tolist()) for d in np.unique(days)})
    hard, clean = results[VIEWS[-1]], results['clean']
    clean_net = {}
    for i, name in enumerate(modes_for(dataset)[:-1]):
        clean_net[name] = int(clean['correct'][-1]-clean['correct'][i])
    passed = all(min(p['half_nets']) >= 2 for p in hard['pairs'].values()) and min(clean_net.values()) >= -2
    return dict(status='TARGET_PILOT_SCORED', truth_last=True, dataset=dataset, receiver=str(rx), n=len(ids),
        modes=list(modes_for(dataset)), views=results, clean_net=clean_net, available_control_screen_pass=passed,
        unavailable_exact_checkpoint_reference=['original fixed DURA v1 under LoRa contract'] if dataset == 'lora' else [],
        exposure='development target pilot; excluded from any later reserved acceptance set',
        target_used_for_training_or_weight_selection=False, formal_acceptance_complete=False)


def score_wisig(rt):
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('Independent CPU scorer only')
    root, project = Path(rt['run_root']), Path(rt['project'])
    ids, values = frozen_pair(root)['wisig']
    cfg = read(project/'runs'/'20261027-phase1-pamf-repl-shuffled-views-m16-r02'/'configs'/
               'score__main_split-repl-shuffled_time-3view-s2026092701.json')
    truth = read(Path(cfg['tests']['cross_rx']['truth']))
    rows = [truth[str(x)] for x in ids]
    mask = np.asarray([str(x['receiver']) == str(WISIG_RX) for x in rows])
    labels = np.asarray([x['label'] for x in rows], dtype=np.int64)[mask]
    days = np.asarray([str(x['day']) for x in rows])[mask]
    if mask.sum() < 40:
        raise ValueError('Insufficient predetermined RX0 target packets')
    score = score_values(ids[mask], {v:x[:, mask] for v, x in values.items()}, labels, WISIG_RX, days, 'wisig')
    score['predicted_blind_pool'] = len(ids)
    write(root/'wisig'/'target-score.json', score)


def score_lora(root, dataset):
    ids, values = frozen_pair(root)['lora']
    from experiments.pamf_lora_main.data import label_ids, packet_indices, recording
    scratch = root/'lora'/'score_scratch'
    scratch.mkdir(exist_ok=False)
    with zipfile.ZipFile(dataset/'multiple_receiver_test.zip') as archive:
        with recording(archive, f'{LORA_RX}_test.h5', scratch) as capture:
            labels = label_ids(capture, 2000)[packet_indices(2000, 0)][ids]
    score = score_values(ids, values, labels, LORA_RX, np.full(len(ids), 'single_capture'), 'lora')
    write(root/'lora'/'target-score.json', score)


def worker(runtime, dataset):
    predict(read(runtime), dataset)


def launch(runtime):
    rt = read(runtime)
    root = Path(rt['run_root'])
    root.mkdir(exist_ok=False)
    for name in ('lora', 'wisig'):
        (root/name).mkdir()
    try:
        gpus = [i for i in free_gpus() if i in rt['allowed_gpus']]
        if len(gpus) < 2:
            raise RuntimeError('Two distinct GPUs with 8 GiB free required')
        jobs = {}
        for name, gpu in zip(('lora', 'wisig'), gpus[:2], strict=True):
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='2',
                       OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='2')
            with (root/f'{name}-worker.log').open('x', encoding='utf-8') as out:
                jobs[name] = subprocess.Popen([sys.executable, '-u', '-m', __spec__.name, 'worker',
                    str(runtime), name], env=env, stdin=subprocess.DEVNULL, stdout=out, stderr=subprocess.STDOUT)
        write(root/'state.json', dict(status='TARGET_PREDICTING', workers={k: p.pid for k,p in jobs.items()},
                                     gpus=dict(zip(('lora','wisig'), gpus[:2], strict=True))))
        failures = {name: proc.wait() for name, proc in jobs.items()}
        if any(failures.values()):
            raise RuntimeError(f'Target prediction workers failed: {failures}')
        frozen_pair(root)
        write(root/'state.json', dict(status='TARGET_PREDICTIONS_FROZEN', truth_read=False), exclusive=False)
        result = subprocess.run([sys.executable, '-u', '-m', __spec__.name, 'score-wisig', str(runtime)],
                                env=dict(os.environ, CUDA_VISIBLE_DEVICES=''), stdin=subprocess.DEVNULL, check=False)
        if result.returncode:
            raise RuntimeError('Independent WiSig target scoring failed')
        write(root/'state.json', dict(status='TARGET_PREDICTIONS_FROZEN_WISIG_SCORED',
            lora_score='pending independent local truth', truth_last=True), exclusive=False)
    except Exception as error:
        write(root/'state.json', dict(status='FAILED', error=str(error)), exclusive=False)
        raise


def publish(repo, output, context, ssh_config):
    from experiments.pamf_lora_main.publish import release, release_commit
    source = read(context)
    rt = {k: source[k] for k in ('project', 'python', 'frozen_sources')}
    project = PurePosixPath(rt['project'])
    runs, releases = project/'runs', project/'releases'
    rt.update(run_id=RUN, runner=__spec__.name, code_commit=release_commit(repo),
        release_root=str(releases/f'{RUN}-release'), run_root=str(runs/RUN),
        lora_frozen_root=str(runs/candidate.PARENT), allowed_gpus=[2, 3],
        external_releases=[str(runs/PARENT), str(releases/REFERENCE),
                           str(runs/'20261123-phase1-wisig-r69-paired-main-m6-r01')])
    release(repo, output, rt, ssh_config, archive_extras=('tools/r17_activation_factorization.py',))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest='action', required=True)
    for action in ('dispatch', 'worker', 'score-wisig'):
        child = actions.add_parser(action)
        child.add_argument('runtime', type=Path)
        if action == 'worker':
            child.add_argument('dataset', choices=('lora', 'wisig'))
    local = actions.add_parser('score-lora')
    local.add_argument('root', type=Path)
    local.add_argument('dataset', type=Path)
    pub = actions.add_parser('publish')
    for field in ('repo', 'output', 'context', 'ssh-config'):
        pub.add_argument('--'+field, type=Path, required=True)
    args = vars(parser.parse_args())
    action = args.pop('action')
    if action == 'dispatch':
        launch(args['runtime'])
    elif action == 'worker':
        worker(**args)
    elif action == 'score-wisig':
        score_wisig(read(args['runtime']))
    elif action == 'score-lora':
        score_lora(**args)
    else:
        publish(**args)
