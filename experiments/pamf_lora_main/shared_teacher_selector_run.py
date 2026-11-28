"""Proposed source-U frozen-classifier-guided selector; do not train without authorization."""
import argparse
import importlib
import os
from pathlib import Path, PurePosixPath
import subprocess
import sys
from time import monotonic

import numpy as np
import torch
from torch.nn import functional as F

from experiments.cric_dura_comparison.io import read, write
from experiments.pamf_lora_main.dura_real_gain_fit import DEVICE, LORA_ACTION, LORA_GAIN, WISIG_CACHE, WISIG_PACKET, WISIG_PARENT, as_complex
from experiments.pamf_lora_main.shared_dura_realgain import SharedPacketSelector
from experiments.pamf_lora_main.shared_dura_realgain_run import REFERENCE, dispatch, score
from experiments.pamf_lora_main import shared_twoprior_run as shared

RUN = '20261128-phase1-shared-teacher-selector-s2026112937-r01'
SEED, STEPS = 2026112937, 1000
PARENT = '20261128-phase1-dura-realgain-crossdata-s2026112845-r02'


def preferred_expert(clean_logits, real_logits, r17_logits):
    if (clean_logits.ndim != 2 or clean_logits.shape != real_logits.shape or
            clean_logits.shape != r17_logits.shape or not torch.isfinite(clean_logits).all() or
            not torch.isfinite(real_logits).all() or not torch.isfinite(r17_logits).all()):
        raise ValueError('Expected finite matching frozen teacher logits')
    clean_logp = F.log_softmax(clean_logits, dim=-1)
    clean_p = clean_logp.exp()
    real_kl = (clean_p*(clean_logp-F.log_softmax(real_logits, dim=-1))).sum(-1)
    r17_kl = (clean_p*(clean_logp-F.log_softmax(r17_logits, dim=-1))).sum(-1)
    return (r17_kl < real_kl).to(clean_logits.dtype)


def source_teacher(rt, dataset):
    """Load the matching frozen original classifier; called only during authorized fit."""
    project = Path(rt['project'])
    if dataset == 'lora':
        from experiments.pamf_lora_main.dura_real_gain_fit import activate_wisig_archive
        release = project / 'releases' / f'{LORA_GAIN}-release'
        parent = read(release / 'runtime.json')
        activate_wisig_archive(release)
        activate_wisig_archive(Path(parent['w24_release_root']))
        from experiments.pamf_lora_main import w24_joint_pamf as joint
        old = read(project / 'runs' / LORA_GAIN / 'frequency_inphase_native' / 'fit-inputs.json')
        _, reader, _, _ = joint.boot(old, DEVICE)
        reader.eval().requires_grad_(False)

        def teacher(z):
            cpu = np.asarray(z.contiguous().view(torch.float32).tolist(), dtype=np.float32).view(np.complex64)
            return reader(joint.received_aperture(cpu, DEVICE))[0]
    else:
        original = importlib.import_module('experiments.pamf_unrolled_restore.wisig_anchored_window_source')
        config = read(project / 'releases' / REFERENCE / 'runtime.json')
        reader = original.tri.reader_factory(config).eval().requires_grad_(False)

        def teacher(z):
            return original.classify(reader, z)
    return teacher


def fit(rt, dataset):
    from experiments.pamf_lora_main.w24_native_dura import source_pair
    source, ids, r17, gain = shared.experts(rt, dataset)
    teacher = source_teacher(rt, dataset)
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    generator = torch.Generator(device=DEVICE).manual_seed(SEED + 1)
    model = SharedPacketSelector(shared.WIDTH[dataset]).to(DEVICE).train()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(opt, STEPS, eta_min=3e-5)
    started = monotonic()
    correct = total = 0

    def decision_data(received, clean):
        scale = received.abs().square().mean(-1, keepdim=True).clamp_min(1e-12).sqrt()
        normalized = received / scale
        with torch.no_grad():
            a = gain(normalized)
            b, action = r17(normalized)
            target = preferred_expert(teacher(clean), teacher(a*scale), teacher(b*scale))
        return normalized, a, b, action, target

    for step in range(1, STEPS + 1):
        raw = as_complex(source[rng.choice(ids, shared.BATCH[dataset], replace=False)])
        clean, observed, _, _ = source_pair(raw, generator)
        n, a, b, action, target = decision_data(observed, clean)
        nc, ac, bc, actionc, targetc = decision_data(clean, clean)
        logits = model.logits(n, a, b, action)
        clean_logits = model.logits(nc, ac, bc, actionc)
        if step == 1:
            torch.testing.assert_close(model(n, a, b, action), b, rtol=0, atol=0)
        loss = F.binary_cross_entropy_with_logits(logits, target) + .5*F.binary_cross_entropy_with_logits(clean_logits, targetc)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if step == 1 and (model.head.weight.grad is None or not torch.isfinite(model.head.weight.grad).all()
                          or not bool(model.head.weight.grad.norm() > 0)):
            raise ValueError('Real-checkpoint smoke requires finite nonzero selector gradient')
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite source selector loss')
        opt.step()
        schedule.step()
        correct += int(((logits >= 0) == target.bool()).sum())
        total += len(target)
        if step == 1 or step % 50 == 0:
            print('SHARED_TEACHER_SELECTOR_FIT', dataset, step, float(loss.detach()), float(norm), flush=True)
    torch.cuda.synchronize()
    proof = dict(status='SOURCE_FROZEN', dataset=dataset, seed=SEED, updates=STEPS,
                 batch=shared.BATCH[dataset], U_pool=len(ids), U_exposures=STEPS*shared.BATCH[dataset],
                 F_reads=6*STEPS*shared.BATCH[dataset], teacher='LoRa W24 E104' if dataset == 'lora' else 'WiSig Tri-SIG-TIME SSL F0 default head',
                 TX_reads=0, V_reads=0, target_reads=0, training_selector_correct=correct,
                 training_selector_total=total, parameters=sum(p.numel() for p in model.parameters()),
                 seconds=monotonic()-started, peak_cuda_bytes=torch.cuda.max_memory_allocated())
    folder = Path(rt['run_root']) / dataset
    with (folder / 'gain.pt').open('xb') as stream:
        torch.save(dict(weights=model.state_dict(), proof=proof), stream)
    write(folder / 'fit.json', proof)


def worker(runtime, dataset):
    shared.SharedTwoPriorGain = SharedPacketSelector
    shared.SEED = SEED
    shared.STEPS = STEPS
    rt = read(runtime)
    fit(rt, dataset)
    shared.predict(rt, dataset, allow_source_f_guidance=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='')
    result = subprocess.run([sys.executable, '-u', '-m', __spec__.name, 'score', str(runtime), dataset],
                            env=env, stdin=subprocess.DEVNULL, check=False)
    if result.returncode:
        raise RuntimeError('Independent source scoring failed')


def publish(repo, output, context, ssh_config):
    from experiments.pamf_lora_main.publish import release, release_commit
    source = read(context)
    rt = {k: source[k] for k in ('project', 'python', 'frozen_sources')}
    project = PurePosixPath(rt['project'])
    runs, releases = project / 'runs', project / 'releases'
    rt.update(run_id=RUN, runner=__spec__.name, code_commit=release_commit(repo),
              release_root=str(releases / f'{RUN}-release'), run_root=str(runs / RUN),
              lora_frozen_root=str(runs / PARENT), allowed_gpus=[0, 5, 3, 6, 1, 4],
              external_releases=[str(runs / n) for n in (LORA_ACTION, LORA_GAIN, WISIG_CACHE,
                  WISIG_PACKET, WISIG_PARENT, PARENT,
                  '20261128-diagnostic-lora-r69-full-w24-s2026112895-r01')] + [str(releases / REFERENCE)])
    release(repo, output, rt, ssh_config, archive_extras=('tools/r17_activation_factorization.py',))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='action', required=True)
    for phase in ('dispatch', 'worker', 'score'):
        child = commands.add_parser(phase)
        child.add_argument('runtime', type=Path)
        if phase != 'dispatch':
            child.add_argument('dataset', choices=('lora', 'wisig'))
    pub = commands.add_parser('publish')
    for field in ('repo', 'output', 'context', 'ssh-config'):
        pub.add_argument('--' + field, type=Path, required=True)
    args = vars(parser.parse_args())
    action = args.pop('action')
    if action == 'dispatch':
        dispatch(args['runtime'], worker_module=__spec__.name)
    elif action == 'score':
        score(read(args['runtime']), args['dataset'])
    elif action == 'worker':
        worker(**args)
    else:
        publish(**args)
