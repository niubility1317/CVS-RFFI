"""LoRa R07 action and gain trained on source Full low-urban pairs."""

import argparse
import os
from pathlib import Path, PurePosixPath
import subprocess
import sys
from time import monotonic

import numpy as np
import torch
from torch.nn import functional as F

from experiments.cric_dura_comparison.io import read, write
from experiments.pamf_lora_main.dura_real_gain_fit import DEVICE, LORA_ACTION, LORA_GAIN, as_complex
from experiments.pamf_lora_main.dura_real_gain_amplitude import R17AmplitudeGain
from experiments.pamf_lora_main.dura_real_gain_unroll import fused_anchor_unroll
from experiments.pamf_lora_main.dura_real_gain_r17_fit import boot
from experiments.pamf_lora_main.dura_real_gain_run import free_gpus
from experiments.pamf_lora_main.dura_real_gain_screen import packet_scale, paired


RUN = '20261128-phase1-lora-r07-action-joint-s2026112929-r01'
FULL_PREVIOUS = '20261128-phase1-lora-r17-full-source-s2026112921-r01'
ANCHOR_PREVIOUS = '20261128-phase1-lora-r17-anchor-unroll-s2026112913-r01'
AMPLITUDE_PREVIOUS = '20261128-phase1-lora-r17-amplitude-gain-s2026112897-r01'
RETENTION_PREVIOUS = '20261128-phase1-lora-r17-recognition-retain-s2026112885-r01'
PREVIOUS = '20261128-phase1-lora-r17-realgain-e104-s2026112861-r01'
JOINT_PREVIOUS = '20261128-phase1-lora-joint-r17-realgain-s2026112873-r01'
SEED, STEPS, ROLE_BATCH, COUNT = 2026112929, 1500, 2, 300
VIEWS = ('clean', 'practical_low_urban')
MODES = ('direct', 'lora_native_r17', 'real_gain', 'joint_fusion')


def recognition_retention(logits, baseline_logits, targets, active):
    ce = F.cross_entropy(logits, targets, reduction='none')
    base_ce = F.cross_entropy(baseline_logits, targets, reduction='none')
    kl = F.kl_div(F.log_softmax(logits, dim=-1), F.softmax(baseline_logits, dim=-1), reduction='none').sum(-1)
    weights = active.to(ce.dtype)
    return ((ce + 2. * (ce - base_ce).relu() + .5 * kl) * weights).sum() / weights.sum().clamp_min(1)


def normalize_pair(clean, observed):
    scale = packet_scale(observed)
    return clean / scale, observed / scale


def source_training_pairs(joint, source, contract, root):
    from concurrent.futures import ThreadPoolExecutor
    from leo_practical.channel import normalize_input
    from experiments.pamf_lora_main.w24_native_full import native_full_view
    rng = np.random.default_rng(SEED + 4)
    labels = joint.training_labels(source, contract)
    records = []
    for role in ('L_s', 'U_s'):
        packets, receivers, ids, _ = joint.source_role(source, contract, role)
        selected = np.sort(rng.choice(len(packets), 512, replace=False))
        records.append((np.asarray(packets[selected]), receivers[selected], ids[selected], selected))
    clean_raw = np.concatenate((records[0][0], records[1][0]))
    receivers = np.concatenate((records[0][1], records[1][1]))
    ids = np.concatenate((records[0][2], records[1][2]))
    if len(np.unique(ids)) != len(ids):
        raise ValueError('Source L/U physical packets must be distinct')
    normalized = np.stack([normalize_input(value)[0] for value in clean_raw]).astype(np.complex64)
    def generate(i):
        return native_full_view(clean_raw[i], 'practical_low_urban', str(receivers[i]), str(ids[i]),
                                'dura-fusion-source-train-2026112929')
    with ThreadPoolExecutor(max_workers=4) as pool:
        observed = []
        for i, value in enumerate(pool.map(generate, range(len(clean_raw))), 1):
            observed.append(value)
            if i % 128 == 0:
                print('LORA_SOURCE_FULL_GENERATE', i, len(clean_raw), flush=True)
    observed = np.stack(observed)
    with (root / 'train-source-ids.npz').open('xb') as stream:
        np.savez(stream, ids=ids, receivers=receivers, L_count=512, U_count=512)
    return normalized, observed, np.asarray(labels)[records[0][3]]


def fit(rt):
    joint, native, old, contract, reader, tracker, baseline, r17 = boot(rt)
    source = Path(old['source_root'])
    r17.estimator.requires_grad_(True)
    preparation_started = monotonic()
    clean_source, observed_source, labels = source_training_pairs(joint, source, contract, Path(rt['run_root']))
    preparation_seconds = monotonic() - preparation_started
    torch.manual_seed(SEED)
    gain = R17AmplitudeGain(1024).to(DEVICE).train()
    rng = np.random.default_rng(SEED)
    opt = torch.optim.AdamW([{'params': r17.estimator.parameters(), 'lr': 1e-5},
                             {'params': gain.parameters(), 'lr': 1e-4}], weight_decay=1e-4)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(opt, STEPS, eta_min=1e-6)
    accepted = 0
    started = monotonic()
    for step in range(1, STEPS + 1):
        li = rng.choice(512, ROLE_BATCH, replace=False)
        ui = rng.choice(512, ROLE_BATCH, replace=False)
        indices = np.concatenate((li, 512 + ui))
        clean, observed = normalize_pair(as_complex(clean_source[indices]), as_complex(observed_source[indices]))
        with torch.no_grad():
            p, _ = baseline(observed)
            pc, _ = baseline(clean)
            clean_ap, _ = joint.normalize_aperture(tracker.aperture(clean))
            confidence, pseudo = reader(clean_ap)[0][ROLE_BATCH:].softmax(-1).max(-1)
            active_u = confidence >= .95
            base_ap, _ = joint.normalize_aperture(tracker.aperture(p))
            base_logits = reader(base_ap)[0]
            clean_base_ap, _ = joint.normalize_aperture(tracker.aperture(pc))
            clean_base_logits = reader(clean_base_ap)[0]
        output, action = fused_anchor_unroll(r17, gain, observed, p, train_action=True, return_action=True)
        preserved = fused_anchor_unroll(r17, gain, clean, pc, train_action=True)
        if step == 1:
            torch.testing.assert_close(output, p, rtol=1e-5, atol=1e-6)
        aperture, _ = joint.normalize_aperture(tracker.aperture(output))
        logits = reader(aperture)[0]
        preserved_ap, _ = joint.normalize_aperture(tracker.aperture(preserved))
        preserved_logits = reader(preserved_ap)[0]
        targets = torch.cat((torch.as_tensor(labels[li], device=DEVICE, dtype=torch.long), pseudo))
        active = torch.cat((torch.ones(ROLE_BATCH, device=DEVICE, dtype=torch.bool), active_u))
        recognition = recognition_retention(logits, base_logits, targets, active)
        clean_kl = F.kl_div(F.log_softmax(preserved_logits, -1), F.softmax(clean_base_logits, -1), reduction='batchmean')
        wave = (output - clean).abs().square().mean()
        clean_keep = (preserved - pc).abs().square().mean()
        physical = (action(clean) - observed).abs().square().mean()
        loss = recognition + 2. * clean_kl + .3 * wave + 1.5 * clean_keep + .1 * physical
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if step == 1:
            head = gain.net[-1].weight.grad.flatten(1).norm(dim=1)
            action_grads = [p.grad.norm() for p in r17.estimator.parameters() if p.grad is not None]
            if (not torch.isfinite(head).all() or not bool((head > 0).all()) or not action_grads
                    or not any(bool(torch.isfinite(g) and g > 0) for g in action_grads)
                    or any(p.grad is not None for p in r17.coarse.parameters())
                    or any(p.grad is not None for p in r17.priors.parameters())):
                raise ValueError('Gain and R07 estimator must train; R02/R17 priors stay frozen')
        norm = torch.nn.utils.clip_grad_norm_([*gain.parameters(), *r17.estimator.parameters()], 1., error_if_nonfinite=True)
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite source-only recognition retention loss')
        opt.step()
        schedule.step()
        accepted += int(active_u.sum())
        if step == 1 or step % 50 == 0:
            print('LORA_ACTION_JOINT_FIT', step, float(loss.detach()), float(norm), flush=True)
    torch.cuda.synchronize()
    proof = dict(status='SOURCE_FROZEN', dataset='lora', seed=SEED, updates=STEPS,
                 labeled_exposures=STEPS * ROLE_BATCH, unlabeled_exposures=STEPS * ROLE_BATCH,
                 accepted_pseudolabels=accepted, pseudo_threshold=.95, F_frozen=True,
                 U_truth_reads=0, V_reads=0, target_reads=0, source_L_pool=512, source_U_pool=512,
                 channel_preparation_seconds=preparation_seconds, channel='full practical_low_urban',
                 R02_trainable=False, R07_trainable=True, R17_priors_trainable=False,
                 params=sum(p.numel() for p in gain.parameters()), seconds=monotonic()-started,
                 peak_cuda_bytes=torch.cuda.max_memory_allocated())
    root = Path(rt['run_root'])
    with (root / 'gain.pt').open('xb') as stream:
        torch.save(dict(weights=gain.state_dict(), estimator=r17.estimator.state_dict(), proof=proof), stream)
    write(root / 'fit.json', proof)


def select_rows(ids, rx, earlier_rows, prior_ids):
    eligible = np.flatnonzero((rx == 0) & ~np.isin(ids, prior_ids))
    eligible = np.setdiff1d(eligible, earlier_rows)
    if len(eligible) < COUNT:
        raise ValueError('Insufficient unexposed source V packets for fixed RX')
    return np.sort(np.random.default_rng(SEED + 2).choice(eligible, COUNT, replace=False))


@torch.no_grad()
def predict(rt):
    joint, native, old, contract, reader, _, baseline, r17 = boot(rt)
    root, source = Path(rt['run_root']), Path(old['source_root'])
    checkpoint = torch.load(root / 'gain.pt', map_location=DEVICE, weights_only=True)
    proof = read(root / 'fit.json')
    if (checkpoint['proof'] != proof or (proof['status'], proof['seed'], proof['updates']) !=
            ('SOURCE_FROZEN', SEED, STEPS) or any(proof[k] for k in ('U_truth_reads', 'V_reads', 'target_reads'))):
        raise ValueError('Expected fixed lawful source-trained joint fusion')
    gain = R17AmplitudeGain(1024).to(DEVICE).eval().requires_grad_(False)
    gain.load_state_dict(checkpoint['weights'], strict=True)
    candidate_r17 = native.dura_model(Path(rt['project']) / 'runs' / LORA_ACTION, contract, DEVICE).eval().requires_grad_(False)
    candidate_r17.estimator.load_state_dict(checkpoint['estimator'], strict=True)

    def classify(z):
        cpu = np.asarray(z.contiguous().view(torch.float32).tolist(), dtype=np.float32).view(np.complex64)
        return joint.classify_aperture(reader, joint.received_aperture(cpu, DEVICE))

    def decisions(raw):
        y = as_complex(raw)
        scale = packet_scale(y)
        normalized = y / scale
        p, _ = baseline(normalized)
        values = (y, r17(y), baseline(y)[0], fused_anchor_unroll(candidate_r17, gain, normalized, p) * scale)
        return np.stack([classify(z) for z in values])

    if decisions(joint.source_role(source, contract, 'U_s')[0][:2]).shape != (len(MODES), 2):
        raise ValueError('No-query checkpoint smoke failed')
    earlier_rows, _, _ = joint.panel(source, contract)
    with np.load(source / 'V_metadata.npz', allow_pickle=False) as metadata:
        ids, rx = metadata['ids'].copy(), metadata['rx_i'].copy()
    if ids.tolist() != contract['role_ids']['V']:
        raise ValueError('Source V physical order differs')
    with np.load(Path(rt['project']) / 'runs' / PREVIOUS / 'screen.npz', allow_pickle=False) as prior:
        prior_ids = prior['ids'].copy()
    with np.load(Path(rt['project']) / 'runs' / JOINT_PREVIOUS / 'screen.npz', allow_pickle=False) as prior:
        prior_ids = np.concatenate((prior_ids, prior['ids'].copy()))
    with np.load(Path(rt['project']) / 'runs' / RETENTION_PREVIOUS / 'screen.npz', allow_pickle=False) as prior:
        prior_ids = np.concatenate((prior_ids, prior['ids'].copy()))
    with np.load(Path(rt['project']) / 'runs' / AMPLITUDE_PREVIOUS / 'screen.npz', allow_pickle=False) as prior:
        prior_ids = np.concatenate((prior_ids, prior['ids'].copy()))
    with np.load(Path(rt['project']) / 'runs' / ANCHOR_PREVIOUS / 'screen.npz', allow_pickle=False) as prior:
        prior_ids = np.concatenate((prior_ids, prior['ids'].copy()))
    with np.load(Path(rt['project']) / 'runs' / FULL_PREVIOUS / 'screen.npz', allow_pickle=False) as prior:
        prior_ids = np.concatenate((prior_ids, prior['ids'].copy()))
    rows = select_rows(ids, rx, earlier_rows, prior_ids)
    raw = joint.native(source, 'V', contract['counts']['V'])[rows]
    matrix = {}
    for view in VIEWS:
        signals = np.stack([joint.source_receive(view, (int(i), joint.SOURCE[0], value))
                            for i, value in zip(rows, raw, strict=True)])
        matrix[view] = np.concatenate([decisions(signals[i:i+8]) for i in range(0, COUNT, 8)], axis=1)
        print('LORA_ACTION_JOINT_SCREEN', view, COUNT, flush=True)
    with (root / 'screen.npz').open('xb') as stream:
        np.savez(stream, ids=ids[rows], **matrix)
    write(root / 'screen-frozen.json', dict(status='PREDICTIONS_FROZEN', modes=list(MODES), views=list(VIEWS),
          count=COUNT, source_RX='b200_1', prior_screen_overlap=0, source_truth_reads=0, target_reads=0))


def score(rt):
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('Independent CPU scorer required')
    root = Path(rt['run_root'])
    frozen = read(root / 'screen-frozen.json')
    if (frozen['status'], frozen['modes'], frozen['views'], frozen['count'], frozen['source_truth_reads']) != (
            'PREDICTIONS_FROZEN', list(MODES), list(VIEWS), COUNT, 0):
        raise ValueError('All source predictions must freeze before truth')
    with np.load(root / 'screen.npz', allow_pickle=False) as archive:
        ids = archive['ids'].copy()
        outputs = {view: archive[view].copy() for view in VIEWS}
    if len(ids) != COUNT or any(x.shape != (len(MODES), COUNT) for x in outputs.values()):
        raise ValueError('Incomplete prediction matrix')
    old = read(Path(rt['project']) / 'runs' / LORA_GAIN / 'frequency_inphase_native' / 'fit-inputs.json')
    with np.load(Path(old['source_root']) / 'V_metadata.npz', allow_pickle=False) as metadata:
        truth = dict(zip(metadata['ids'].tolist(), metadata['y'].tolist(), strict=True))
    labels = np.asarray([truth[x] for x in ids], dtype=np.int64)
    permutation = np.random.default_rng(SEED + 3).permutation(COUNT)
    hard, clean = outputs[VIEWS[-1]], outputs['clean']
    pairs = {name: paired(hard[-1], hard[i], labels, permutation, 2) for i, name in enumerate(MODES[:-1])}
    clean_net = {name: int(((clean[-1] == labels).astype(np.int16) - (clean[i] == labels).astype(np.int16)).sum())
                 for i, name in enumerate(MODES[:-1])}
    promote = all(value['passes'] for value in pairs.values()) and min(clean_net.values()) >= -2
    write(root / 'screen-score.json', dict(status='SOURCE_SCREEN_SCORED', truth_last=True, count=COUNT,
          modes=list(MODES), correct={v: (x == labels).sum(1).tolist() for v, x in outputs.items()},
          low_urban_pairs=pairs, clean_net=clean_net, promote=promote,
          missing_controls=['same-contract LoRa DURA v1'], formal_acceptance_complete=False,
          exposure='Fresh source V packets; E104 selection historically saw source V.'))


def dispatch(runtime):
    rt = read(runtime)
    root = Path(rt['run_root'])
    root.mkdir(exist_ok=False)
    try:
        devices = free_gpus()
        if not devices:
            raise RuntimeError('No GPU with 8 GiB free')
        gpu = next(value for value in (5, 7, 2, 3, 6, 1, 4, 0) if value in devices)
        for action in ('fit', 'predict', 'score'):
            device = None if action == 'score' else gpu
            env = dict(os.environ, CUDA_VISIBLE_DEVICES='' if device is None else str(device),
                       OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='2')
            with (root / f'{action}.log').open('x', encoding='utf-8') as log:
                proc = subprocess.Popen([sys.executable, '-u', '-m', __spec__.name, action, str(runtime)],
                                        env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
            write(root / 'state.json', dict(status=action.upper(), pid=proc.pid, gpu=device), exclusive=False)
            if proc.wait():
                raise RuntimeError(f'{action} failed; preserve partial artifacts')
        write(root / 'state.json', dict(status='SOURCE_SCREEN_SCORED', gpu=gpu,
              promote=read(root / 'screen-score.json')['promote']), exclusive=False)
    except Exception as error:
        write(root / 'state.json', dict(status='FAILED', error=str(error)), exclusive=False)
        raise


def publish(repo, output, context, ssh_config):
    from experiments.pamf_lora_main.publish import release, release_commit
    source = read(context)
    rt = {key: source[key] for key in ('project', 'python', 'frozen_sources')}
    project = PurePosixPath(rt['project'])
    runs, releases = project / 'runs', project / 'releases'
    rt.update(run_id=RUN, runner=__spec__.name, code_commit=release_commit(repo),
              release_root=str(releases / f'{RUN}-release'), run_root=str(runs / RUN),
              external_releases=[str(runs / LORA_ACTION), str(runs / LORA_GAIN), str(runs / PREVIOUS), str(runs / JOINT_PREVIOUS), str(runs / RETENTION_PREVIOUS), str(runs / AMPLITUDE_PREVIOUS), str(runs / ANCHOR_PREVIOUS), str(runs / FULL_PREVIOUS),
                                 str(releases / f'{LORA_GAIN}-release')])
    release(repo, output, rt, ssh_config)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='action', required=True)
    for action in ('dispatch', 'fit', 'predict', 'score'):
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
        {'fit': fit, 'predict': predict, 'score': score}[action](read(args['runtime']))
