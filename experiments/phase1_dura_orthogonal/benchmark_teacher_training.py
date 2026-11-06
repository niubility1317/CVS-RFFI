"""Compare original and cached frozen-teacher source training on one GPU."""
import argparse
import json
import os
import time
from pathlib import Path

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
import torch

from experiments.phase1_dura_orthogonal.benchmark_masked_training_full import load_inputs
from experiments.phase1_dura_orthogonal.benchmark_teacher_precompute import targets
from experiments.phase1_dura_orthogonal.fused_gradient_guard import checked_backward_and_clip
from experiments.phase1_dura_orthogonal.masked_prior_constraint import masked_prior_loss, visible_pair, MASK_SEED, WEIGHT
from experiments.phase1_dura_orthogonal.multiscale_spectral import DuraMultiscaleSpectral
from experiments.phase1_dura_orthogonal.source_masked_prior import SPAN
from experiments.phase1_dura_orthogonal.train_source import source_loss


def cached_teacher_loss(prior, output, clean, starts, target):
    observed, scale, _, _, mask = visible_pair(output, clean, starts)
    prediction = prior(observed, mask, scale)
    energy = clean.square().mean((1, 2)).clamp_min(1e-8)
    return (((prediction-target).square()*mask[:, None]).sum((1, 2))/(2*SPAN*energy)).mean()


def schedule(count, epochs):
    order_gen = torch.Generator().manual_seed(2026110613)
    mask_gen = torch.Generator().manual_seed(MASK_SEED)
    batches = []
    for _ in range(epochs):
        order = torch.randperm(count, generator=order_gen)
        starts = torch.cat([torch.randint(1, 224, (min(32, count-i),), generator=mask_gen)
                            for i in range(0, count, 32)])
        batches.append((order.to('cuda:0'), starts.to('cuda:0')))
    return batches


def train(fields, prior, initial, batches, precompute):
    torch.manual_seed(2026110612)
    model = DuraMultiscaleSpectral(width=32, taps=9).to('cuda:0')
    model.load_state_dict(initial, strict=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0003, weight_decay=.001)
    losses = []
    torch.cuda.synchronize()
    started = time.monotonic()
    for order, starts in batches:
        cache = targets(prior, fields[4][order], starts, 256) if precompute else None
        for start in range(0, len(order), 32):
            index = order[start:start+32]
            x, anchor, direction, variance, clean = (v[index] for v in fields)
            output = model(x, anchor, direction, variance, validate=False)
            masks = starts[start:start+32]
            term = (cached_teacher_loss(prior, output, clean, masks, cache[start:start+32])
                    if precompute else masked_prior_loss(prior, output, clean, masks))
            loss = source_loss(output, anchor, clean, objective='dura_guard') + WEIGHT*term
            optimizer.zero_grad(set_to_none=True)
            checked_backward_and_clip(loss, model, (model.wave.weight, model.filter_head.weight))
            optimizer.step()
            losses.append(loss.detach())
    torch.cuda.synchronize()
    return model, torch.stack(losses).cpu(), time.monotonic()-started


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--teacher', type=Path, required=True)
    parser.add_argument('--initial-state', type=Path, required=True)
    parser.add_argument('--initial-commit', required=True)
    parser.add_argument('--commit', required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists() or not torch.cuda.is_available():
        raise ValueError('Unique output and available GPU required')
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    data, prior, initial = load_inputs(args.cache,args.teacher,args.initial_state,args.initial_commit)
    fields = tuple(torch.cat([v[i:i+512] for i in (0,32768,65536)]) for v in data)
    del data
    small = tuple(v[:256] for v in fields)
    for use_cache in (False,True):
        model, _, _ = train(small, prior, initial, schedule(len(small),1), use_cache)
        del model
    batches = schedule(len(fields[0]), 5)
    pairs = []
    for order in ((False,True),(True,False)):
        outcomes = {}
        for use_cache in order:
            outcomes[use_cache] = train(fields, prior, initial, batches, use_cache)
        a, b = outcomes[False], outcomes[True]
        loss_exact = torch.equal(a[1],b[1])
        parameter_exact = all(torch.equal(a[0].state_dict()[k],b[0].state_dict()[k]) for k in a[0].state_dict())
        pairs.append(dict(original_seconds=a[2],precompute_seconds=b[2],steps=len(a[1]),
                          loss_exact=loss_exact,parameters_exact=parameter_exact,
                          max_loss_abs=float((a[1]-b[1]).abs().max())))
        del outcomes, a, b
    args.out.mkdir(parents=True)
    result = dict(status='COMPLETE',commit=args.commit,source_ids=512,views=3,epochs=5,
                  batch=32,teacher_precompute_batch=256,pairs=pairs,gpu=torch.cuda.get_device_name(),
                  TX_labels_read=0,F_weights_read=0,F_outputs_read=0,V_reads=0,target_reads=0,
                  scope='Local paired source diagnostic; scheduling generated before both timers; no model saved')
    (args.out/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf8')
    print(json.dumps(result),flush=True)


if __name__ == '__main__':
    main()
