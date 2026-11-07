"""Source-statistic geometry of diagonal versus coupled covariance mode updates."""
import argparse
import os
from pathlib import Path
import time

import numpy as np

from .cost_probe import write
from .covariance_trial import load_prior
from .mismatched_prior_training import RECEIVERS,STATISTICS_CODE,STATISTICS_RUN
from .source_pair import read,validate_source

RUN = '20261107-analysis-Restore-source-covariance-capacity-s2026110707-r01'
RANK = 64


def capacity(base,target,rank=RANK):
    base,target = np.asarray(base,dtype=np.float64),np.asarray(target,dtype=np.float64)
    if (base.ndim != 2 or base.shape[0] != base.shape[1] or target.shape != base.shape
            or not 1 <= rank <= len(base) or not np.isfinite(base).all() or not np.isfinite(target).all()
            or not np.allclose(base,base.T,rtol=0,atol=1e-8) or not np.allclose(target,target.T,rtol=0,atol=1e-8)):
        raise ValueError('Finite matched symmetric covariances and a valid rank required')
    np.linalg.cholesky(base)
    np.linalg.cholesky(target)
    eigenvalues,eigenvectors = np.linalg.eigh(base)
    delta = target-base
    total = float(np.square(delta).sum())
    leading = eigenvectors[:,-rank:]
    projected = leading.T @ delta @ leading
    diagonal = np.diag(projected)
    full_energy = float(np.square(projected).sum())
    diagonal_energy = float(np.square(diagonal).sum())
    all_diagonal = np.einsum('ij,ij->j',eigenvectors,delta @ eigenvectors)
    all_diagonal_energy = float(np.square(all_diagonal).sum())
    changes = np.clip(diagonal/eigenvalues[-rank:],-.9,.9)
    bounded_update = eigenvalues[-rank:]*changes
    bounded_error = total-2*float(diagonal @ bounded_update)+float(bounded_update @ bounded_update)
    errors = dict(diagonal_rank=total-diagonal_energy,coupled_rank=total-full_energy,
                  diagonal_all=total-all_diagonal_energy,bounded_signed_rank=bounded_error)
    if min(errors.values()) < -1e-8*max(1.,total):
        raise ValueError('Projection residual violates orthogonality identity')
    errors = {key:max(0.,value) for key,value in errors.items()}
    explained = {key:(None if total==0 else 1-value/total) for key,value in errors.items()}
    relative = np.diag(projected+np.diag(eigenvalues[-rank:]))/eigenvalues[-rank:]
    # A coupled replacement of this block is a valid SPD covariance: outside
    # modes retain base eigenvalues, while this block is the target projection.
    target_block = leading.T @ target @ leading
    return dict(rank=rank,dimension=len(base),difference_squared_Frobenius=total,
        difference_Frobenius_over_base=float(np.sqrt(total)/np.linalg.norm(base)),
        source_top_rank_variance_fraction=float(eigenvalues[-rank:].sum()/eigenvalues.sum()),
        explained_difference_energy=explained,residual_squared_Frobenius=errors,
        coupled_target_block_min_eigenvalue=float(np.linalg.eigvalsh(target_block).min()),
        ideal_diagonal_variance_ratio_min=float(relative.min()),ideal_diagonal_variance_ratio_max=float(relative.max()),
        bounded_signed_coefficients_at_limit=int((np.abs(changes)>=.9).sum()),
        zero_difference=(total==0))


def worker(args):
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or (Path.cwd()/'release_commit.txt').read_text().strip() != args.commit:
        raise ValueError('Pinned CPU-only source-statistics analysis required')
    os.nice(10)
    validate_source(read(args.inputs))
    folder = args.root/'runs'/STATISTICS_RUN
    metadata = read(folder/'results.json')
    if (metadata['status'] != 'SOURCE_RECEIVER_PRIOR_PREDICTIONS_COMPLETE'
            or metadata['code_commit'] != STATISTICS_CODE or metadata['prior_fit_roles'] != ['L_s','U_s']
            or (metadata['prior_samples'],metadata['prior_L_s'],metadata['prior_U_s']) != (12600,1260,11340)
            or metadata['prior_fit_seed'] != 2026110707 or metadata['V_fit_rows'] != 0
            or metadata['TX_labels_read'] != 0 or metadata['target_inputs'] != 0):
        raise ValueError('Existing pre-V source-only receiver moments required')
    _,global_covariance = load_prior(args.root)
    # Match the actual deployed covariance storage, then analyze in FP64.
    global_covariance = global_covariance.astype(np.float32).astype(np.float64)
    with np.load(folder/'fitted_priors.npz',allow_pickle=False) as source:
        banks = {rx:{name:source[f'{rx}__{name}__covariance'].astype(np.float32).astype(np.float64)
                     for name in ('same','other')} for rx in RECEIVERS}
    out = args.root/'runs'/RUN
    out.mkdir()
    started = time.perf_counter()
    cells = {}
    for rx in RECEIVERS:
        cells[str(rx)] = {}
        for kind,base in (('global',global_covariance),('other_receiver',banks[rx]['other'])):
            if base.shape != (512,512) or banks[rx]['same'].shape != (512,512):
                raise ValueError('Original real512 source covariance required')
            cells[str(rx)][kind] = capacity(base,banks[rx]['same'])
            if not (out/'smoke.json').exists():
                write(out/'smoke.json',dict(status='PASS',actual_source_covariance_dimension=512,
                    rank=RANK,neural_checkpoints_used=0,IQ_reads=0,TX_labels_read=0,target_reads=0))
                print('SOURCE_COVARIANCE_CAPACITY_ACTUAL_MATRIX_PASS',flush=True)
            print('SOURCE_COVARIANCE_CAPACITY_CELL',rx,kind,flush=True)
    write(out/'results.json',dict(status='SOURCE_COVARIANCE_CAPACITY_ANALYZED',code_commit=args.commit,
        source_moment_run=STATISTICS_RUN,source_moment_code=STATISTICS_CODE,cells=cells,
        rank=RANK,cell_count=10,reference='Existing same-RX L_s/U_s covariance, post-fit source proxy only.',
        arithmetic='FP64 eigenspaces of actual FP32 covariance storage; ideal representational projection, no neural fitting.',
        neural_checkpoints_used=0,neural_forwards=0,IQ_reads=0,TX_labels_read=0,V_reads=0,OE_reads=0,target_reads=0,
        optimizer_updates=0,new_IQ_or_statistic_fits=0,seconds=time.perf_counter()-started,
        claim_scope='Squared-Frobenius covariance-difference geometry only; not recoverable IQ energy, identification accuracy, or Main ceiling.'))
    write(out/'complete.json',dict(status='COMPLETE',code_commit=args.commit,cell_count=10,rank=RANK))
    print('SOURCE_COVARIANCE_CAPACITY_COMPLETE',flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--inputs',type=Path,required=True)
    parser.add_argument('--commit',required=True)
    worker(parser.parse_args())
