"""Shared selector trained with a source-only two-view contrastive embedding."""
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
from experiments.pamf_lora_main.dura_real_gain_fit import DEVICE, LORA_ACTION, LORA_GAIN, WISIG_CACHE, WISIG_PACKET, WISIG_PARENT, as_complex
from experiments.pamf_lora_main.shared_dura_realgain import SharedPacketSelector
from experiments.pamf_lora_main.shared_dura_realgain_run import REFERENCE, dispatch, score
from experiments.pamf_lora_main import shared_twoprior_run as shared

RUN = '20261128-phase1-shared-contrastive-selector-s2026112921-r01'
SEED = 2026112921
STEPS, EMBED_STEPS = 1000, 400
PARENT = '20261128-phase1-dura-realgain-crossdata-s2026112845-r02'


def packet_channels(x):
    if x.ndim != 2 or not x.is_complex() or x.shape[-1] < 32:
        raise ValueError('Expected a batch of complex packets with at least 32 samples')
    z = x / x.abs().square().mean(-1, keepdim=True).clamp_min(1e-12).sqrt()
    phase = z[:, 1:] * z[:, :-1].conj()
    phase = phase / phase.abs().clamp_min(1e-12)
    return torch.stack((z.abs(), F.pad(phase.real, (1, 0)), F.pad(phase.imag, (1, 0))), dim=1)


class PacketEmbedding(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(torch.nn.Conv1d(3, 32, 7, stride=2, padding=3), torch.nn.SiLU(),
            torch.nn.Conv1d(32, 64, 5, stride=2, padding=2), torch.nn.SiLU(),
            torch.nn.Conv1d(64, 64, 5, stride=2, padding=2), torch.nn.SiLU(),
            torch.nn.AdaptiveAvgPool1d(1), torch.nn.Flatten())

    def forward(self, x):
        pooled = F.adaptive_avg_pool1d(packet_channels(x), 256)
        return F.normalize(self.net(pooled), dim=-1, eps=1e-8)


def contrastive_loss(first, second):
    if first.ndim != 2 or first.shape != second.shape or len(first) < 2:
        raise ValueError('At least two matching embeddings required')
    logits = F.normalize(first, dim=-1) @ F.normalize(second, dim=-1).T / .1
    target = torch.arange(len(first), device=first.device)
    return .5 * (F.cross_entropy(logits, target) + F.cross_entropy(logits.T, target))


def fit(rt, dataset):
    from experiments.pamf_lora_main.w24_native_dura import source_pair
    source, ids, r17, gain = shared.experts(rt, dataset)
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    generator = torch.Generator(device=DEVICE).manual_seed(SEED + 1)
    model = SharedPacketSelector(shared.WIDTH[dataset]).to(DEVICE).train()
    encoder = PacketEmbedding().to(DEVICE).train()
    started = monotonic()
    encoder_opt = torch.optim.AdamW(encoder.parameters(), lr=3e-4, weight_decay=1e-4)
    # Validate both real source experts before the first optimization step.
    raw = as_complex(source[rng.choice(ids, shared.BATCH[dataset], replace=False)])
    _, smoke_view, _, _ = source_pair(raw, generator)
    normalized = smoke_view / smoke_view.abs().square().mean(-1, keepdim=True).clamp_min(1e-12).sqrt()
    with torch.no_grad():
        a, (b, action) = gain(normalized), r17(normalized)
        torch.testing.assert_close(model(normalized, a, b, action), b, rtol=0, atol=0)
    for step in range(1, EMBED_STEPS + 1):
        raw = as_complex(source[rng.choice(ids, shared.BATCH[dataset], replace=False)])
        _, one, _, _ = source_pair(raw, generator)
        _, two, _, _ = source_pair(raw, generator)
        embedding_loss = contrastive_loss(encoder(one), encoder(two))
        encoder_opt.zero_grad(set_to_none=True)
        embedding_loss.backward()
        torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1., error_if_nonfinite=True)
        if not torch.isfinite(embedding_loss):
            raise FloatingPointError('Nonfinite source embedding loss')
        encoder_opt.step()
        if step == 1 or step % 50 == 0:
            print('SHARED_CONTRASTIVE_EMBED', dataset, step, float(embedding_loss.detach()), flush=True)
    encoder.eval().requires_grad_(False)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(opt, STEPS, eta_min=3e-5)
    correct = total = 0

    def decision_data(received, clean):
        scale = received.abs().square().mean(-1, keepdim=True).clamp_min(1e-12).sqrt()
        normalized = received / scale
        with torch.no_grad():
            a = gain(normalized)
            b, action = r17(normalized)
            reference = encoder(clean)
            err_a = (encoder(a)-reference).square().mean(-1)
            err_b = (encoder(b)-reference).square().mean(-1)
            target = (err_b < err_a).to(normalized.real.dtype)
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
            print('SHARED_CONTRASTIVE_SELECTOR_FIT', dataset, step, float(loss.detach()), float(norm), flush=True)
    torch.cuda.synchronize()
    proof = dict(status='SOURCE_FROZEN', dataset=dataset, seed=SEED, updates=STEPS,
                 batch=shared.BATCH[dataset], U_pool=len(ids), U_exposures=STEPS*shared.BATCH[dataset],
                 embedding_updates=EMBED_STEPS, embedding_U_exposures=EMBED_STEPS*shared.BATCH[dataset],
                 embedding_parameters=sum(p.numel() for p in encoder.parameters()),
                 negative_pairing='different_packet_may_share_TX',
                 F_reads=0, TX_reads=0, V_reads=0, target_reads=0, training_selector_correct=correct,
                 training_selector_total=total, parameters=sum(p.numel() for p in model.parameters()),
                 seconds=monotonic()-started, peak_cuda_bytes=torch.cuda.max_memory_allocated())
    folder = Path(rt['run_root']) / dataset
    with (folder / 'gain.pt').open('xb') as stream:
        torch.save(dict(weights=model.state_dict(), proof=proof), stream)
    with (folder / 'source_encoder.pt').open('xb') as stream:
        torch.save(encoder.state_dict(), stream)
    write(folder / 'fit.json', proof)


def worker(runtime, dataset):
    shared.SharedTwoPriorGain = SharedPacketSelector
    shared.SEED = SEED
    shared.STEPS = STEPS
    rt = read(runtime)
    fit(rt, dataset)
    shared.predict(rt, dataset)
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
              lora_frozen_root=str(runs / PARENT), allowed_gpus=[2, 3, 6, 4],
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
