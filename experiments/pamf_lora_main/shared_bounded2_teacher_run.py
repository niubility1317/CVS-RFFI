"""Shared 2%-bounded real-frequency correction with source-selected anchors."""
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
from experiments.pamf_lora_main.dura_real_gain_fit import (DEVICE, LORA_ACTION, LORA_GAIN, WISIG_CACHE,
    WISIG_PACKET, WISIG_PARENT, activate_wisig_archive, as_complex)
from experiments.pamf_lora_main.shared_dura_realgain import SharedBoundedGain
from experiments.pamf_lora_main.shared_dura_realgain_run import REFERENCE, dispatch, score
from experiments.pamf_lora_main import shared_twoprior_run as shared

RUN = '20261128-phase1-shared-bounded2-teacher-s2026112957-r01'
SEED, STEPS = 2026112957, 1000
ANCHOR_REAL = {'lora': True, 'wisig': False}
PARENT = '20261128-phase1-dura-realgain-crossdata-s2026112845-r02'


def exact_forward_surrogate(exact, surrogate):
    if exact.shape != surrogate.shape or not torch.isfinite(exact).all() or not torch.isfinite(surrogate).all():
        raise ValueError('Exact and surrogate teacher inputs must be finite and match')
    return exact.detach() + (surrogate-surrogate.detach())


def teacher_kl(clean_logits, restored_logits):
    if (clean_logits.shape != restored_logits.shape or clean_logits.ndim != 2 or
            not torch.isfinite(clean_logits).all() or not torch.isfinite(restored_logits).all()):
        raise ValueError('Expected finite matching teacher logits')
    clean_logp = F.log_softmax(clean_logits.detach(), dim=-1)
    return (clean_logp.exp()*(clean_logp-F.log_softmax(restored_logits, dim=-1))).sum(-1).mean()


def source_teacher(rt, dataset):
    project = Path(rt['project'])
    if dataset == 'lora':
        release = project / 'releases' / f'{LORA_GAIN}-release'
        parent = read(release / 'runtime.json')
        activate_wisig_archive(release)
        activate_wisig_archive(Path(parent['w24_release_root']))
        from experiments.pamf_lora_main import w24_joint_pamf as joint
        old = read(project / 'runs' / LORA_GAIN / 'frequency_inphase_native' / 'fit-inputs.json')
        _, reader, _, tracker = joint.boot(old, DEVICE)
        reader.eval().requires_grad_(False)
        tracker.eval().requires_grad_(False)

        def teacher(z):
            cpu = np.asarray(z.detach().contiguous().view(torch.float32).tolist(), dtype=np.float32).view(np.complex64)
            exact = joint.received_aperture(cpu, DEVICE)
            if z.requires_grad:
                surrogate, _ = joint.normalize_aperture(tracker.aperture(z))
                if not surrogate.requires_grad:
                    raise ValueError('LoRa aperture surrogate must propagate input gradients')
                aperture = exact_forward_surrogate(exact, surrogate)
            else:
                aperture = exact
            return reader(aperture)[0]
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
    model = SharedBoundedGain(shared.WIDTH[dataset], anchor_real=ANCHOR_REAL[dataset], correction_cap=.02).to(DEVICE).train()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(opt, STEPS, eta_min=3e-5)
    started = monotonic()
    for step in range(1, STEPS + 1):
        raw = as_complex(source[rng.choice(ids, shared.BATCH[dataset], replace=False)])
        clean, observed, _, _ = source_pair(raw, generator)
        output, prior = shared.restore_scaled(r17, gain, model, observed)
        preserved, clean_prior = shared.restore_scaled(r17, gain, model, clean)
        if not ANCHOR_REAL[dataset]:
            clean_scale = clean.abs().square().mean(-1, keepdim=True).clamp_min(1e-12).sqrt()
            with torch.no_grad():
                clean_r17, _ = r17(clean/clean_scale)
                clean_prior = clean_r17*clean_scale
        if step == 1:
            if not ANCHOR_REAL[dataset]:
                obs_scale = observed.abs().square().mean(-1, keepdim=True).clamp_min(1e-12).sqrt()
                with torch.no_grad():
                    physical, _ = r17(observed/obs_scale)
                    prior = physical*obs_scale
            torch.testing.assert_close(output, prior, rtol=0, atol=0)
        with torch.no_grad():
            clean_logits = teacher(clean)
        output_logits = teacher(output)
        scale = clean.abs().square().mean(-1, keepdim=True).clamp_min(1e-12).sqrt()
        wave = ((output-clean)/scale).abs().square().mean()
        preserve = ((preserved-clean_prior)/scale).abs().square().mean()
        teacher_loss = teacher_kl(clean_logits, output_logits)
        if step == 1:
            teacher_grad, = torch.autograd.grad(teacher_loss, model.net[-1].weight, retain_graph=True)
            if not torch.isfinite(teacher_grad).all() or not bool(teacher_grad.norm() > 0):
                raise ValueError('Real-checkpoint teacher loss must produce a finite nonzero restorer gradient')
        loss = teacher_loss + .2*wave + 1.5*preserve
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if step == 1 and (model.net[-1].weight.grad is None or not torch.isfinite(model.net[-1].weight.grad).all()
                          or not bool(model.net[-1].weight.grad.norm() > 0)):
            raise ValueError('Real-checkpoint teacher smoke requires finite nonzero restorer gradient')
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite shared source teacher objective')
        opt.step()
        schedule.step()
        if step == 1 or step % 50 == 0:
            print('SHARED_BOUNDED2_TEACHER_FIT', dataset, step, float(loss.detach()), float(norm), flush=True)
    torch.cuda.synchronize()
    proof = dict(status='SOURCE_FROZEN', dataset=dataset, seed=SEED, updates=STEPS,
                 batch=shared.BATCH[dataset], U_pool=len(ids), U_exposures=STEPS*shared.BATCH[dataset],
                 F_reads=2*STEPS*shared.BATCH[dataset],
                 teacher='LoRa W24 E104' if dataset == 'lora' else 'WiSig Tri-SIG-TIME SSL F0 default head',
                 lora_backward='exact scipy frontend forward; WideTracker surrogate gradient' if dataset == 'lora' else 'not applicable',
                 anchor='real_gain' if ANCHOR_REAL[dataset] else 'r17', correction_rms_fraction_cap=.02,
                 anchor_selection='previously exposed source V low-urban strongest of real-gain and R17',
                 TX_reads=0, V_reads=0, target_reads=0, parameters=sum(p.numel() for p in model.parameters()),
                 seconds=monotonic()-started, peak_cuda_bytes=torch.cuda.max_memory_allocated())
    folder = Path(rt['run_root']) / dataset
    with (folder / 'gain.pt').open('xb') as stream:
        torch.save(dict(weights=model.state_dict(), proof=proof), stream)
    write(folder / 'fit.json', proof)


def worker(runtime, dataset):
    shared.SharedTwoPriorGain = lambda width: SharedBoundedGain(width, anchor_real=ANCHOR_REAL[dataset], correction_cap=.02)
    shared.SEED, shared.STEPS = SEED, STEPS
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
              lora_frozen_root=str(runs / PARENT), allowed_gpus=[7, 1, 2, 4],
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
