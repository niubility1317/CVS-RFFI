"""Benchmark a source-checkpoint RadioNet step without target/query/truth."""
import argparse
import copy
import math
from pathlib import Path
import statistics
import time

import numpy as np
import torch

from .io import activate, read, require, write
from .group1 import source_scope
from .group1_source import load_source, source_data
from .group2_c_backbone_adapter import CrossAdaptationAdapter
from .vendor_native.comparison_suite import adaptation


def run(rt):
    activate(rt)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    source = source_scope(rt)
    source['run_root'] = rt['c_source_roots']['mopc_hr']
    bundle = load_source(source, 'mopc_hr', rt['model_seeds'][0], 'cuda:0')
    data = source_data(source, 'L_s')
    labels = np.asarray(data.metadata['y'])
    ix = [int(i) for c in range(6) for i in np.flatnonzero(labels == c)[:27]]
    require(len(ix) == 162, 'Source benchmark needs 27 examples per class')
    x = data.x[ix[:128]].to('cuda:0')
    y = torch.tensor(labels[ix[:128]], device='cuda:0', dtype=torch.long)
    support = data.x[ix[128:158]].to('cuda:0')
    torch.manual_seed(9947)
    model0 = CrossAdaptationAdapter(bundle['model'], 'mopc_hr', 'radionet_ada_knn', 256).cuda()
    disc0 = adaptation.RadioNetADAHead(1).cuda()
    original = adaptation._grad_norm
    candidate = adaptation._radionet_grad_norm
    checks = []

    def compare(model):
        old = original(model)
        cpu, gpu = torch.get_rng_state(), torch.cuda.get_rng_state()
        new = candidate(model)
        checks.append(old == new and torch.equal(cpu, torch.get_rng_state()) and
                      torch.equal(gpu, torch.cuda.get_rng_state()))
        require(checks[-1], 'Same-gradient/RNG check failed')
        return new

    def trial(norm, seed):
        model, disc = copy.deepcopy(model0), copy.deepcopy(disc0)
        optimizer = torch.optim.Adam(model.parameters(), lr=.00001)
        optimizer_d = torch.optim.Adam(disc.parameters(), lr=.00001)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        adaptation._radionet_grad_norm = norm
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(10):
            adaptation.radionet_ada_batch_step(model, disc, x, y, support, optimizer, optimizer_d)
        torch.cuda.synchronize()
        return time.perf_counter() - start

    try:
        trial(compare, 500)
        require(len(checks) == 20 and all(checks), 'Incomplete same-gradient check')
        # Warm both paths and alternate order on the same source batch/model.
        trial(original, 501); trial(candidate, 501)
        durations = {'old': [], 'new': []}
        for seed, order in enumerate((('old','new'),('new','old'),('old','new')), 601):
            for name in order:
                durations[name].append(trial(original if name == 'old' else candidate, seed))
        old, new = statistics.median(durations['old']), statistics.median(durations['new'])
        require(math.isfinite(old) and math.isfinite(new), 'Nonfinite benchmark duration')
        return dict(status='PASS', same_gradient=True, rng_unchanged=True, source_only=True,
                    query_read=False, truth_read=False, source_checkpoint=bundle['source_checkpoint'],
                    source_batch=128, support_batch=30, steps_per_trial=10, raw_seconds=durations,
                    old_seconds=old, new_seconds=new, device=torch.cuda.get_device_name(0))
    finally:
        adaptation._radionet_grad_norm = candidate


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--runtime', required=True)
    args = parser.parse_args()
    runtime = read(args.runtime)
    write(Path(runtime['run_root'])/'radionet_benchmark.json', run(runtime))
