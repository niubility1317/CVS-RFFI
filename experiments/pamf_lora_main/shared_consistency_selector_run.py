"""Shared selector trained on same-packet two-view consistency, without F or TX."""
import argparse
from pathlib import Path, PurePosixPath
from time import monotonic

import numpy as np
import torch
from torch.nn import functional as F

from experiments.cric_dura_comparison.io import read, write
from experiments.pamf_lora_main.dura_real_gain_fit import DEVICE, LORA_ACTION, LORA_GAIN, WISIG_CACHE, WISIG_PACKET, WISIG_PARENT, as_complex
from experiments.pamf_lora_main.shared_dura_realgain import SharedPacketSelector
from experiments.pamf_lora_main.shared_dura_realgain_run import REFERENCE, dispatch, score
from experiments.pamf_lora_main import shared_selector_run as selector
from experiments.pamf_lora_main import shared_twoprior_run as shared

RUN = '20261128-phase1-shared-consistency-selector-s2026112913-r01'
SEED, STEPS = 2026112913, 1000
PARENT = '20261128-phase1-dura-realgain-crossdata-s2026112845-r02'


def pair_discrimination(first, second):
    """Same-record correlation minus a cyclic different-record comparison."""
    if first.ndim != 2 or first.shape != second.shape or len(first) < 2 or not first.is_complex():
        raise ValueError('At least two matching complex packet pairs required')

    def correlation(a, b):
        inner = (a.conj()*b).sum(-1).abs().square()
        power = a.abs().square().sum(-1)*b.abs().square().sum(-1)
        return inner/power.clamp_min(1e-12)

    return correlation(first, second) - correlation(first, second.roll(1, dims=0))


def fit(rt, dataset):
    from experiments.pamf_lora_main.w24_native_dura import source_pair
    source, ids, r17, gain = shared.experts(rt, dataset)
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    generator = torch.Generator(device=DEVICE).manual_seed(SEED + 1)
    model = SharedPacketSelector(shared.WIDTH[dataset]).to(DEVICE).train()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(opt, STEPS, eta_min=3e-5)
    started = monotonic()
    preference = total = 0

    def experts(received):
        scale = received.abs().square().mean(-1, keepdim=True).clamp_min(1e-12).sqrt()
        normalized = received / scale
        with torch.no_grad():
            a = gain(normalized)
            b, action = r17(normalized)
        return normalized, a, b, action

    for step in range(1, STEPS + 1):
        raw = as_complex(source[rng.choice(ids, shared.BATCH[dataset], replace=False)])
        clean, first, _, _ = source_pair(raw, generator)
        _, second, _, _ = source_pair(raw, generator)
        n1, a1, b1, action1 = experts(first)
        n2, a2, b2, action2 = experts(second)
        nc, ac, bc, actionc = experts(clean)
        with torch.no_grad():
            quality_a = pair_discrimination(a1, a2)
            quality_b = pair_discrimination(b1, b2)
            target = (quality_b >= quality_a).to(n1.real.dtype)
        p1, p2, pc = (model.logits(*row) for row in
                       ((n1, a1, b1, action1), (n2, a2, b2, action2), (nc, ac, bc, actionc)))
        if step == 1:
            torch.testing.assert_close(model(n1, a1, b1, action1), b1, rtol=0, atol=0)
        loss = .4*F.binary_cross_entropy_with_logits(p1, target) + .4*F.binary_cross_entropy_with_logits(p2, target) + .2*F.binary_cross_entropy_with_logits(pc, target)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if step == 1 and (model.head.weight.grad is None or not torch.isfinite(model.head.weight.grad).all()
                          or not bool(model.head.weight.grad.norm() > 0)):
            raise ValueError('Real-checkpoint smoke requires finite nonzero selector gradients')
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite source-only consistency loss')
        opt.step()
        schedule.step()
        preference += int(target.sum())
        total += len(target)
        if step == 1 or step % 50 == 0:
            print('SHARED_CONSISTENCY_SELECTOR_FIT', dataset, step, float(loss.detach()), float(norm), flush=True)
    torch.cuda.synchronize()
    proof = dict(status='SOURCE_FROZEN', dataset=dataset, seed=SEED, updates=STEPS,
                 batch=shared.BATCH[dataset], U_pool=len(ids), U_exposures=STEPS*shared.BATCH[dataset],
                 simulated_views=2*STEPS*shared.BATCH[dataset], clean_exposures=STEPS*shared.BATCH[dataset],
                 F_reads=0, TX_reads=0, V_reads=0, target_reads=0,
                 source_r17_preferred=preference, source_preference_total=total,
                 negative_pairing='cyclic_other_packet_may_share_TX', parameters=sum(p.numel() for p in model.parameters()),
                 seconds=monotonic()-started, peak_cuda_bytes=torch.cuda.max_memory_allocated())
    folder = Path(rt['run_root']) / dataset
    with (folder / 'gain.pt').open('xb') as stream:
        torch.save(dict(weights=model.state_dict(), proof=proof), stream)
    write(folder / 'fit.json', proof)


def worker(runtime, dataset):
    selector.SEED = SEED
    selector.fit = fit
    shared.STEPS = STEPS
    selector.worker(runtime, dataset)


def publish(repo, output, context, ssh_config):
    from experiments.pamf_lora_main.publish import release, release_commit
    source = read(context)
    rt = {k: source[k] for k in ('project', 'python', 'frozen_sources')}
    project = PurePosixPath(rt['project'])
    runs, releases = project / 'runs', project / 'releases'
    rt.update(run_id=RUN, runner=__spec__.name, code_commit=release_commit(repo),
              release_root=str(releases / f'{RUN}-release'), run_root=str(runs / RUN),
              lora_frozen_root=str(runs / PARENT), allowed_gpus=[7, 2, 3, 6, 1, 4],
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
