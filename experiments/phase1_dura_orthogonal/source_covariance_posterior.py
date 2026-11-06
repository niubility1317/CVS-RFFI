"""Source-only Gaussian covariance posterior; true action is an oracle diagnostic."""
import argparse
import os
from pathlib import Path
import time

import numpy as np
import torch

from experiments.pamf_phase1_flow import experiment as core
from experiments.pamf_phase1_flow.anchor_source_smoke import frozen_dura
from experiments.phase1_dura_innovation_probe.source_probe import source_inputs
from experiments.phase1_dura_orthogonal.controller import training_environments
from experiments.phase1_dura_orthogonal.oracle_action_source_probe import iq_nmse, true_action
from experiments.phase1_dura_orthogonal.train_source import load_source_training


def shrunk_second_moment(moment, *, shrinkage=.1):
    if moment.ndim != 2 or moment.shape[0] != moment.shape[1] or not 0 < shrinkage <= 1:
        raise ValueError('Square second moment and positive shrinkage required')
    symmetric = (moment+moment.T)*.5
    scale = symmetric.trace()/len(symmetric)
    if not torch.isfinite(symmetric).all() or scale <= 0:
        raise ValueError('Finite positive source second moment required')
    return (1-shrinkage)*symmetric + shrinkage*scale*torch.eye(len(symmetric), device=moment.device, dtype=moment.dtype)


def posterior_cg(action, observed, covariance, variance, *, steps=40, rtol=1e-4):
    """Zero-mean Gaussian posterior using CG on A C A* + variance I."""
    if observed.ndim != 3 or observed.shape[1] != 2 or variance.shape != (len(observed),):
        raise ValueError('Paired real IQ and per-record variance required')
    length = observed.shape[-1]
    if covariance.shape != (2*length, 2*length) or (variance <= 0).any():
        raise ValueError('Matching covariance and positive variance required')
    def to_complex(real):
        return torch.complex(real[:, :length], real[:, length:])
    def to_real(complex_value):
        return torch.cat((complex_value.real, complex_value.imag), dim=1)
    def operator(value):
        back = to_real(action.adjoint(to_complex(value)))
        return to_real(action(to_complex(back @ covariance))) + variance[:, None]*value
    rhs = observed.reshape(len(observed), 2*length)
    result = torch.zeros_like(rhs)
    residual = rhs.clone()
    direction = residual.clone()
    squared = residual.square().sum(1)
    threshold = squared.clamp_min(1e-30)*(rtol**2)
    for _ in range(steps):
        active = squared > threshold
        normal = operator(direction)
        denom = (direction*normal).sum(1).clamp_min(1e-30)
        alpha = torch.where(active, squared/denom, torch.zeros_like(squared))
        result = result + alpha[:, None]*direction
        residual = residual - alpha[:, None]*normal
        following = residual.square().sum(1)
        beta = torch.where(active, following/squared.clamp_min(1e-30), torch.zeros_like(squared))
        direction = torch.where((following > threshold)[:, None], residual+beta[:, None]*direction,
                                torch.zeros_like(direction))
        squared = following
    output = to_real(action.adjoint(to_complex(result))) @ covariance
    relative = (rhs-operator(result)).norm(dim=1)/rhs.norm(dim=1).clamp_min(1e-15)
    return output.reshape_as(observed), relative


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--commit', required=True)
    args = parser.parse_args()
    training_environments('cuda:0', os.environ)
    if args.out.exists():
        raise FileExistsError('Unique source covariance diagnostic output required')
    args.out.mkdir(parents=True)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    ids, rx, scene, _, received, clean = source_inputs(args.root)
    _, _, _, fit = load_source_training(args.root, ids)
    moment = torch.zeros((512, 512), dtype=torch.float32, device='cuda:0')
    for offset in range(0, len(fit), 256):
        values = core.tensor(clean[fit[offset:offset+256]], 'cuda:0').reshape(-1, 512)
        moment += values.T @ values
    moment /= len(fit)
    covariance = shrunk_second_moment(moment, shrinkage=.1)
    isotropic = torch.eye(512, device='cuda:0')*(covariance.trace()/512)
    folder = args.root/'runs/20261019-phase1-restoration-learned-action-source-s2026092701-r07/cache/epochs/19'
    meta = core.read(folder/'complete.json')
    if (meta.get('count') != core.COUNT or meta.get('namespace') != 'learned-action/train/epoch19' or
            any(meta.get(k) != 0 for k in ('TX_labels_read', 'F_outputs_read', 'target_reads'))):
        raise ValueError('Approved source simulator action cache required')
    kernel = np.load(folder/'kernel.npy', mmap_mode='r', allow_pickle=False)
    coefficients = np.load(folder/'coefficients.npy', mmap_mode='r', allow_pickle=False)
    variance = np.load(folder/'variance.npy', mmap_mode='r', allow_pickle=False)
    if (kernel.shape != (core.COUNT, 2, 64) or coefficients.shape != (core.COUNT, 4, 256) or
            variance.shape != (core.COUNT,) or any(v.dtype != np.float32 for v in (kernel, coefficients, variance))):
        raise ValueError('Complete matching source simulator action required')
    dura = frozen_dura(args.root, 'cuda:0')
    errors, residuals = [], []
    started = time.monotonic()
    for offset in range(0, len(ids), 16):
        chosen = ids[offset:offset+16]
        x, y = core.tensor(received[chosen], 'cuda:0'), core.tensor(clean[chosen], 'cuda:0')
        h, c, v = (core.tensor(array[chosen], 'cuda:0') for array in (kernel, coefficients, variance))
        base, estimated, estimated_v = dura.estimate_context(x)
        oracle = true_action(type(estimated), h, c)
        original = dura.stage_outputs(x, (base, estimated, estimated_v))[0][-1]
        outputs, residual = [], []
        for action, noise in ((oracle, v), (estimated, estimated_v)):
            for prior in (isotropic, covariance):
                value, rel = posterior_cg(action, x, prior, noise, steps=40, rtol=1e-4)
                outputs.append(value)
                residual.append(rel)
        if offset == 0:
            torch.testing.assert_close(original, dura(x), rtol=1e-5, atol=3e-6)
            print('SOURCE_COVARIANCE_POSTERIOR_SMOKE_PASS', flush=True)
        values = torch.stack([iq_nmse(original, y), *[iq_nmse(z, y) for z in outputs]], dim=1)
        norms = torch.stack(residual, dim=1)
        if not torch.isfinite(values).all() or not torch.isfinite(norms).all():
            raise FloatingPointError('Finite source posterior outputs required')
        errors.append(values.cpu())
        residuals.append(norms.cpu())
    errors, residuals = torch.cat(errors), torch.cat(residuals)
    groups = {}
    for name, receivers in (('RX4_6', (4, 6)), ('RX8', (8,))):
        for view in core.VIEWS:
            selected = np.flatnonzero(np.isin(rx, receivers) & (scene == view)).tolist()
            if len(selected) != 48:
                raise ValueError('Complete source receiver and view cells required')
            cell, cg = errors[selected], residuals[selected]
            groups[name+'/'+view] = dict(count=len(cell), mean_nmse=cell.mean(0).tolist(),
                better_than_dura=(cell[:, 1:] < cell[:, :1]).sum(0).tolist(),
                cg_mean_relative_residual=cg.mean(0).tolist(), cg_max_relative_residual=cg.max(0).values.tolist())
    torch.save(dict(covariance=covariance.cpu(), fit_ids=fit.tolist(), shrinkage=.1), args.out/'source_covariance.pt')
    core.write(args.out/'result.json', dict(status='SOURCE_COVARIANCE_POSTERIOR_COMPLETE', commit=args.commit,
        records=len(ids), covariance_fit_ids=len(fit), covariance_shrinkage=.1, cg_steps=40, cg_rtol=1e-4,
        order=['dura', 'true_action_true_noise_isotropic', 'true_action_true_noise_covariance',
               'estimated_action_estimated_noise_isotropic', 'estimated_action_estimated_noise_covariance'],
        groups=groups, elapsed_seconds=time.monotonic()-started, TX_labels_read=0, F_weights_read=0,
        F_outputs_read=0, V_reads=0, target_IQ_reads=0, target_truth_reads=0, model_updates=0,
        caveat='True simulator action/noise are privileged source-only diagnostics, not deployable or theoretical upper bounds; IQ NMSE is not TX accuracy. Covariance fit and source pilot share previously explored source domains.'))
    print('SOURCE_COVARIANCE_POSTERIOR_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
