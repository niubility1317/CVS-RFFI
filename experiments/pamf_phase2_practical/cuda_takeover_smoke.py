"""Real-checkpoint, source-only smoke; no target or query data is opened."""
from pathlib import Path
import numpy as np
import torch

from .features import load_reader, source_smoke
from .group1_target import _features
from .io import activate, read, require, write
from .milkr.cuda_local_ridge import CudaLocalRidge
from .milkr.d92_branch_local_ridge import fit_branch_local_ridge


def run(rt, output):
    output = Path(output)
    activate(rt)
    source_smoke(rt, output.with_name('source_smoke.json'))
    from experiments.cric_dura_comparison.f_data import SourceDataset
    contract = read(rt['source_contract'])
    original = SourceDataset(rt['source_cache'], 'L_s', contract).x[:2]
    require(len(original) == 2, 'Insufficient real source examples for smoke')
    samples = torch.cat([torch.roll(original[i % 2:i % 2+1], 3*i, dims=-1) for i in range(6)])
    raw = np.asarray(samples.tolist(), dtype=np.float32)
    reader = load_reader(rt, rt['model_seeds'][0], 'cuda:0')
    features, _ = _features(reader, raw, 'cuda:0', batch=6)
    state = fit_branch_local_ridge(**features, support_labels=np.arange(6),
        support_ids=[f'source_smoke_{i}' for i in range(6)], classes=[f'c{i}' for i in range(6)])
    cpu = state.score(**features)
    scorer = CudaLocalRidge(state)
    gpu = scorer.score(**features)
    require(np.allclose(cpu, gpu, rtol=1e-10, atol=1e-10), 'CUDA score differs from CPU on source smoke')
    require(np.array_equal(cpu.argmax(1), gpu.argmax(1)), 'CUDA source prediction differs')
    write(output, dict(status='PASS', actual_checkpoint=True, source_examples=2,
        deterministic_source_transforms=6, target_rows=0, query_rows=0, truth_read=False,
        max_score_difference=float(np.max(np.abs(cpu-gpu))), cpu_tie_rows=scorer.cpu_tie_rows,
        CUDA_device=torch.cuda.get_device_name(0)))
