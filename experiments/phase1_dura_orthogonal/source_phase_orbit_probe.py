"""Source-only phase-orbit consistency of the unchanged DURA restorer."""
import argparse
import os
from pathlib import Path
import sys

from experiments.phase1_dura_orthogonal.benchmark_flow_n607_sync import wait_for_idle


def rotate(x, quarter_turns):
    import torch
    if x.ndim != 3 or x.shape[1:] != (2, 256) or quarter_turns not in (0, 1, 2, 3):
        raise ValueError('IQ windows and a fixed quarter turn required')
    i, q = x[:, 0], x[:, 1]
    return (x, torch.stack((-q, i), 1), torch.stack((-i, -q), 1), torch.stack((q, -i), 1))[quarter_turns]


def evaluate_batch(dura, x, clean):
    import torch
    with torch.no_grad():
        paths = torch.stack([rotate(dura(rotate(x, k)), (-k) % 4) for k in range(4)], 1)
        averaged = paths.mean(1)
        energy = clean.square().mean((1, 2)).clamp_min(1e-8)
        baseline_error = (paths[:, 0]-clean).square().mean((1, 2))/energy
        average_error = (averaged-clean).square().mean((1, 2))/energy
        path_disagreement = (paths-averaged[:, None]).square().mean((1, 2, 3))/energy
        result = torch.stack((baseline_error, average_error, path_disagreement), 1)
        if not torch.isfinite(result).all():
            raise FloatingPointError('Finite source-only phase-orbit outputs required')
        return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--commit', required=True)
    p.add_argument('--gpu', type=int, required=True)
    args = p.parse_args()
    uuid = wait_for_idle(args.out, args.gpu, allow_shared=True)
    if 'torch' in sys.modules:
        raise RuntimeError('GPU must be selected before importing torch')
    os.environ.update(CUDA_VISIBLE_DEVICES=uuid, OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
                      OPENBLAS_NUM_THREADS='2', CUBLAS_WORKSPACE_CONFIG=':4096:8')
    import numpy as np
    import torch
    from experiments.pamf_phase1_flow import experiment as core
    from experiments.pamf_phase1_flow.anchor_source_smoke import frozen_dura
    from experiments.phase1_dura_innovation_probe.source_probe import source_inputs

    if not torch.cuda.is_available():
        raise ValueError('Registered GPU required')
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    ids, rx, scene, _, received, clean = source_inputs(args.root)
    dura = frozen_dura(args.root, 'cuda:0')
    x = core.tensor(received[ids[:8]], 'cuda:0')
    y = core.tensor(clean[ids[:8]], 'cuda:0')
    smoke = evaluate_batch(dura, x, y)
    if smoke.shape != (8, 3) or any(p.requires_grad for p in dura.parameters()):
        raise ValueError('Frozen real-checkpoint source smoke required')
    print('SOURCE_PHASE_ORBIT_SMOKE_PASS', len(x), flush=True)
    results = []
    for start in range(0, len(ids), 16):
        chosen = ids[start:start+16]
        results.append(evaluate_batch(dura, core.tensor(received[chosen], 'cuda:0'),
                                      core.tensor(clean[chosen], 'cuda:0')).cpu())
    values = torch.cat(results)
    groups = {}
    for name, receivers in (('RX4_6', (4, 6)), ('RX8', (8,))):
        for view in core.VIEWS:
            mask = torch.as_tensor((np.isin(rx, receivers) & (scene == view)).tolist(), dtype=torch.bool)
            cell = values[mask]
            if len(cell) != 48:
                raise ValueError('48 records per source receiver/view group required')
            groups[name+'/'+view] = dict(count=len(cell), mean_metrics=cell.mean(0).tolist(),
                                         average_beats_baseline=int((cell[:, 1] < cell[:, 0]).sum()))
    args.out.mkdir(parents=True)
    core.write(args.out/'result.json', dict(status='SOURCE_PHASE_ORBIT_COMPLETE', commit=args.commit,
        records=len(ids), quarter_turns=[0, 1, 2, 3], groups=groups,
        metric_order=['baseline_nmse', 'orbit_average_nmse', 'phase_path_disagreement_nmse'],
        TX_labels_read=0, F_weights_read=0, F_outputs_read=0, V_reads=0,
        target_reads=0, model_updates=0, physical_gpu=args.gpu, physical_gpu_uuid=uuid,
        caveat='Source waveforms only. Orbit averaging is not yet a frozen candidate or identity accuracy; DURA may have seen these records.'))
    print('SOURCE_PHASE_ORBIT_COMPLETE', len(ids), flush=True)


if __name__ == '__main__':
    main()
