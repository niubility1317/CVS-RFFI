"""Shared source-pseudolabel risk-aware routing between two frozen recovery experts."""
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

RUN = '20261128-phase1-shared-confidence-router-s2026112957-r01'
SEED, STEPS = 2026112957, 1000
ANCHOR_REAL = {'lora': True, 'wisig': False}
PARENT = '20261128-phase1-dura-realgain-crossdata-s2026112845-r02'


class SharedConfidenceRouter(SharedPacketSelector):
    """Select the anchor or alternative using current-packet classifier confidence."""

    def __init__(self, width, *, anchor_real, teacher):
        super().__init__(width)
        self.register_buffer('anchor_real', torch.tensor(bool(anchor_real)))
        self.teacher = teacher
        self.combiner = torch.nn.Linear(8, 1)
        torch.nn.init.zeros_(self.combiner.weight)
        torch.nn.init.constant_(self.combiner.bias, -2.)

    @staticmethod
    def confidence(logits):
        probabilities = logits.softmax(-1)
        top = probabilities.topk(2, dim=-1).values
        entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum(-1)
        return torch.stack((top[:, 0], top[:, 0] - top[:, 1], entropy), dim=-1)

    def scores(self, received, real_gain, r17, action):
        anchor = torch.where(self.anchor_real, real_gain, r17)
        other = torch.where(self.anchor_real, r17, real_gain)
        with torch.no_grad():
            anchor_logits, other_logits = self.teacher(anchor), self.teacher(other)
            same = (anchor_logits.argmax(-1) == other_logits.argmax(-1)).to(anchor_logits.dtype).unsqueeze(-1)
            features = torch.cat((self.confidence(anchor_logits), self.confidence(other_logits), same), dim=-1)
        physical = super().logits(received, real_gain, r17, action).unsqueeze(-1)
        return self.combiner(torch.cat((physical, features), dim=-1)).squeeze(-1), anchor_logits, other_logits

    def forward(self, received, real_gain, r17, action):
        score, _, _ = self.scores(received, real_gain, r17, action)
        anchor = torch.where(self.anchor_real, real_gain, r17)
        other = torch.where(self.anchor_real, r17, real_gain)
        return torch.where((score >= 0)[:, None], other, anchor)


def switch_targets(clean_logits, anchor_logits, other_logits):
    if (clean_logits.ndim != 2 or clean_logits.shape != anchor_logits.shape or
            clean_logits.shape != other_logits.shape or not torch.isfinite(clean_logits).all() or
            not torch.isfinite(anchor_logits).all() or not torch.isfinite(other_logits).all()):
        raise ValueError('Expected finite matching frozen teacher logits')
    label = clean_logits.argmax(-1)
    anchor_correct = anchor_logits.argmax(-1) == label
    other_correct = other_logits.argmax(-1) == label
    informative = anchor_correct | other_correct
    return (~anchor_correct & other_correct).to(clean_logits.dtype), informative


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


def fit(rt, dataset, teacher):
    from experiments.pamf_lora_main.w24_native_dura import source_pair
    source, ids, r17, gain = shared.experts(rt, dataset)
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    generator = torch.Generator(device=DEVICE).manual_seed(SEED + 1)
    model = SharedConfidenceRouter(shared.WIDTH[dataset], anchor_real=ANCHOR_REAL[dataset], teacher=teacher).to(DEVICE).train()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(opt, STEPS, eta_min=3e-5)
    started = monotonic()
    positives = informative_count = 0

    def experts(received):
        scale = received.abs().square().mean(-1, keepdim=True).clamp_min(1e-12).sqrt()
        normalized = received / scale
        with torch.no_grad():
            a = gain(normalized)
            b, action = r17(normalized)
        return normalized, a, b, action, scale

    for step in range(1, STEPS + 1):
        raw = as_complex(source[rng.choice(ids, shared.BATCH[dataset], replace=False)])
        clean, observed, _, _ = source_pair(raw, generator)
        n, a, b, action, scale = experts(observed)
        nc, ac, bc, actionc, _ = experts(clean)
        anchor, other = (a, b) if ANCHOR_REAL[dataset] else (b, a)
        logits, anchor_logits, other_logits = model.scores(n, a, b, action)
        clean_logits, _, _ = model.scores(nc, ac, bc, actionc)
        with torch.no_grad():
            target, informative = switch_targets(teacher(nc), anchor_logits, other_logits)
        if step == 1:
            torch.testing.assert_close(model(n, a, b, action), anchor, rtol=0, atol=0)
        per_example = F.binary_cross_entropy_with_logits(logits, target, reduction='none')
        weights = informative.to(logits.dtype) * torch.where(target.bool(), 1., 2.)
        route_loss = (per_example*weights).sum() / weights.sum().clamp_min(1.)
        clean_loss = F.binary_cross_entropy_with_logits(clean_logits, torch.zeros_like(clean_logits))
        loss = route_loss + .5*clean_loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if step == 1 and (model.combiner.weight.grad is None or not torch.isfinite(model.combiner.weight.grad).all()
                          or not bool(model.combiner.weight.grad.norm() > 0)):
            raise ValueError('Real-checkpoint smoke requires finite nonzero router gradient')
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite source router loss')
        opt.step()
        schedule.step()
        positives += int(target.sum())
        informative_count += int(informative.sum())
        if step == 1 or step % 50 == 0:
            print('SHARED_CONFIDENCE_ROUTER_FIT', dataset, step, float(loss.detach()), float(norm), positives, informative_count, flush=True)
    torch.cuda.synchronize()
    proof = dict(status='SOURCE_FROZEN', dataset=dataset, seed=SEED, updates=STEPS,
                 batch=shared.BATCH[dataset], U_pool=len(ids), U_exposures=STEPS*shared.BATCH[dataset],
                 F_reads=(5*STEPS+2)*shared.BATCH[dataset],
                 teacher='LoRa W24 E104' if dataset == 'lora' else 'WiSig Tri-SIG-TIME SSL F0 default head',
                 anchor='real_gain' if ANCHOR_REAL[dataset] else 'r17',
                 anchor_selection='previously exposed source V low-urban strongest of real-gain and R17', F_inference=True,
                 pseudo_switch_positives=positives, informative=informative_count,
                 TX_reads=0, V_reads=0, target_reads=0, parameters=sum(p.numel() for p in model.parameters()),
                 seconds=monotonic()-started, peak_cuda_bytes=torch.cuda.max_memory_allocated())
    folder = Path(rt['run_root']) / dataset
    with (folder / 'gain.pt').open('xb') as stream:
        torch.save(dict(weights=model.state_dict(), proof=proof), stream)
    write(folder / 'fit.json', proof)


def worker(runtime, dataset):
    shared.SEED = SEED
    shared.STEPS = STEPS
    rt = read(runtime)
    if dataset == 'wisig':
        from experiments.cric_dura_comparison.bootstrap import activate
        from experiments.pamf_lora_main.dura_real_gain_fit import activate_wisig_archive
        activate(rt)
        activate_wisig_archive(Path(rt['project']) / 'releases' / REFERENCE)
    teacher = source_teacher(rt, dataset)
    shared.SharedTwoPriorGain = lambda width: SharedConfidenceRouter(width, anchor_real=ANCHOR_REAL[dataset], teacher=teacher)
    fit(rt, dataset, teacher)
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
              lora_frozen_root=str(runs / PARENT), allowed_gpus=[7, 1, 4, 6],
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
