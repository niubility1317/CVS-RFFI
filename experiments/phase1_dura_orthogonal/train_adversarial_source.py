"""Train one conditional source-IQ adversarial DURA correction; no F or TX."""
import argparse
import os
from pathlib import Path
import time

import torch

from experiments.phase1_dura_orthogonal import train_source as base
from experiments.phase1_dura_orthogonal.conditional_iq_critic import ConditionalIQCritic, update_critic, WEIGHT, SEED, OBJECTIVE, CRITIC_KIND
from experiments.phase1_dura_orthogonal.fused_gradient_guard import checked_backward_and_clip


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--commit', required=True)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--prepare-batch', type=int, default=32)
    p.add_argument('--source-context-cache', type=Path)
    p.add_argument('--deterministic', action='store_true')
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--smoke-fit', type=int, default=64)
    p.add_argument('--objective', choices=(OBJECTIVE,), required=True)
    p.add_argument('--architecture', choices=('dura_multiscale',), required=True)
    args = p.parse_args()
    base.training_environments(args.device, os.environ)
    device = torch.device(args.device)
    if device.type not in ('cpu', 'cuda') or (device.type == 'cuda' and not torch.cuda.is_available()):
        raise ValueError('Available explicitly selected device required')
    if not args.deterministic or not 1 <= args.prepare_batch <= 128 or not 1 <= args.smoke_fit <= 8192:
        raise ValueError('Deterministic training and bounded preparation required')
    if not args.smoke and not args.source_context_cache:
        raise ValueError('Full training requires fixed source context')
    if args.out.exists():
        raise FileExistsError('Unique source output required')
    args.out.mkdir(parents=True)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(2026110612)
    ids, rx, scenes, _, received, clean_pilot = base.source_inputs(args.root)
    _, fresh, clean, fit = base.load_source_training(args.root, ids, fit_size=base.WIDE_FIT_SIZE)
    if args.smoke:
        fit = fit[:args.smoke_fit]
    dura = base.frozen_dura(args.root, str(device))
    if args.source_context_cache and not args.smoke:
        data, counts = base.load_cache(args.source_context_cache, fit, args.prepare_batch, str(device), deterministic=True)
    else:
        data, counts = base.prepare(args.root, dura, clean, fresh, fit, str(device), args.prepare_batch, 'dura_multiscale')
    data = tuple(value.to(device) for value in data)
    model = base.DuraMultiscaleSpectral(width=32, taps=9).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0003, weight_decay=.001)
    torch.manual_seed(SEED)
    critic = ConditionalIQCritic().to(device)
    critic_optimizer = torch.optim.AdamW(critic.parameters(), lr=.0001, betas=(.5, .9), weight_decay=0.)
    generator = torch.Generator().manual_seed(2026110613)
    with torch.no_grad():
        if not torch.equal(model(*(v[:8] for v in data[:4])), data[1][:8]):
            raise ValueError('New model must initially reproduce frozen DURA')
    epochs = 1 if args.smoke else 5
    losses, critic_losses, adversarial_losses = [], [], []
    started = time.monotonic()
    for epoch in range(epochs):
        order = torch.randperm(len(data[0]), generator=generator).to(device)
        totals = torch.zeros(3, device=device)
        model.train()
        for start in range(0, len(order), 32):
            index = order[start:start+32]
            x, anchor, direction, variance, y = (value[index] for value in data)
            restored = model(x, anchor, direction, variance, validate=False)
            critic_loss, adversarial = update_critic(critic, critic_optimizer, x, anchor, restored, variance, y)
            loss = base.source_loss(restored, anchor, y, objective='dura_guard')+WEIGHT*adversarial
            optimizer.zero_grad(set_to_none=True)
            checked_backward_and_clip(loss, model, (model.wave.weight, model.filter_head.weight))
            optimizer.step()
            totals += torch.stack((loss.detach(), critic_loss, adversarial.detach()))*len(index)
        values = (totals/len(order)).cpu().tolist()
        losses.append(values[0]); critic_losses.append(values[1]); adversarial_losses.append(values[2])
        print('CONDITIONAL_ADVERSARY_EPOCH', epoch+1, 'losses', values, 'seconds', time.monotonic()-started, flush=True)
    if any(p.requires_grad or p.grad is not None for p in dura.parameters()):
        raise ValueError('DURA must remain frozen')
    metadata = dict(commit=args.commit, epochs=epochs, architecture='dura_multiscale', objective=OBJECTIVE,
        candidate_dura_trainable=False, spectral_freq_bins=17, adaptive_filter_taps=9,
        noise_gate_low=base.GATE_LOG10_LOW, noise_gate_high=base.GATE_LOG10_HIGH,
        source_fit_size=base.WIDE_FIT_SIZE, DURA_training_reads=1, DURA_frozen=True,
        TX_labels_read=0, F_weights_read=0, F_outputs_read=0, V_reads=0, target_reads=0,
        source_adversarial_weight=WEIGHT, source_critic_seed=SEED, source_critic_kind=CRITIC_KIND,
        source_critic_parameters=sum(p.numel() for p in critic.parameters()), critic_inference_reads=0,
        source_critic_updates_per_generator=1, losses=losses, source_critic_losses=critic_losses,
        source_adversarial_losses=adversarial_losses, deterministic_algorithms=True,
        source_context_cache=str(args.source_context_cache) if args.source_context_cache else None,
        device=str(device), prepare_batch=args.prepare_batch, training_seconds=time.monotonic()-started)
    if args.smoke:
        base.core.write(args.out/'smoke.json', dict(status='SOURCE_TRAINING_SMOKE_PASS', fit_ids=len(fit), **metadata))
        return
    with (args.out/'final.pt').open('xb') as stream:
        torch.save(dict(schema=base.SCHEMA, model=model.state_dict(), width=32, fit_ids=fit, counts=counts,
                        source_critic_state=critic.state_dict(), **metadata), stream)
    _, groups = base.evaluate_source(model.eval(), dura, ids, rx, scenes, received, clean_pilot, str(device), args.prepare_batch)
    base.core.write(args.out/'source_summary.json', dict(status='SOURCE_TRAINED', fit_ids=len(fit),
        counts=counts, exposure=len(data[0]), source_groups=groups, source_IQ_metrics_not_identity_accuracy=True, **metadata))
    print('CONDITIONAL_ADVERSARY_SOURCE_TRAINED', flush=True)


if __name__ == '__main__':
    main()
