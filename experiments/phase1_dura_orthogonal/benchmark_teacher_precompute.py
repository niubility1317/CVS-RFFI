"""Compare fixed clean-teacher outputs under training and larger batch sizes."""
import argparse
import json
import os
import time
from pathlib import Path

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
import torch

from experiments.phase1_dura_orthogonal.benchmark_masked_training_full import load_inputs
from experiments.phase1_dura_orthogonal.masked_prior_constraint import MASK_SEED
from experiments.phase1_dura_orthogonal.source_masked_prior import SPAN


def targets(prior, clean, starts, batch_size):
    result = []
    with torch.no_grad():
        for offset in range(0, len(clean), batch_size):
            y = clean[offset:offset+batch_size]
            start = starts[offset:offset+batch_size]
            positions = torch.arange(256, device=y.device)[None, :]
            mask = (positions >= start[:, None]) & (positions < start[:, None]+SPAN)
            observed = y.masked_fill(mask[:, None], 0.)
            scale = (observed.square().sum((1, 2), keepdim=True)/(2*(256-SPAN))).clamp_min(1e-12).sqrt()
            result.append(prior(observed, mask, scale))
    return torch.cat(result)


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
    data, prior, _ = load_inputs(args.cache,args.teacher,args.initial_state,args.initial_commit)
    clean = torch.cat([data[4][i:i+512] for i in (0,32768,65536)])
    del data
    order_gen = torch.Generator().manual_seed(2026110613)
    mask_gen = torch.Generator().manual_seed(MASK_SEED)
    orders, masks = [], []
    for _ in range(5):
        orders.append(torch.randperm(len(clean),generator=order_gen))
        masks.append(torch.cat([torch.randint(1,224,(32,),generator=mask_gen) for _ in range(len(clean)//32)]))
    clean = clean[torch.cat(orders).to('cuda:0')]
    starts = torch.cat(masks).to('cuda:0')
    for size in (32,256):
        targets(prior,clean[:256],starts[:256],size)
    timings = []
    for sizes in ((32,256),(256,32)):
        outputs, elapsed = {}, {}
        for size in sizes:
            torch.cuda.synchronize()
            before = time.monotonic()
            outputs[size] = targets(prior,clean,starts,size)
            torch.cuda.synchronize()
            elapsed[size] = time.monotonic()-before
        difference = (outputs[32]-outputs[256]).abs()
        timings.append(dict(batch32_seconds=elapsed[32],batch256_seconds=elapsed[256],
            exact=torch.equal(outputs[32],outputs[256]), max_abs=float(difference.max()),
            changed_values=int(torch.count_nonzero(difference))))
        del outputs, difference
    args.out.mkdir(parents=True)
    result = dict(status='COMPLETE',commit=args.commit,source_ids=512,views=3,epochs=5,
        rows=len(clean),batch_sizes=[32,256],pairs=timings,gpu=torch.cuda.get_device_name(),
        TX_labels_read=0,F_weights_read=0,F_outputs_read=0,V_reads=0,target_reads=0,
        scope='Fixed teacher outputs only, not model trajectory or N607 performance')
    (args.out/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf8')
    print(json.dumps(result),flush=True)


if __name__ == '__main__':
    main()
