"""Expose a conditional prior to source receiver mismatch without RX inputs."""

import numpy as np
import torch

from .cache import tensor
from .projected_prior_detail import shrinkage_training_readout
from .source_inverse import R07
from .source_pair import read


STATISTICS_RUN = '20261107-diagnostic-Restore-receiver-prior-sourceV-s2026092701-r02'
STATISTICS_CODE = '1a9aafda7cf9cf5ee1d293a5e782ef57e27eb5ad'
RECEIVERS = (1, 3, 4, 6, 8)


def load_training_priors(root, device):
    folder = root/'runs'/STATISTICS_RUN
    meta = read(folder/'results.json')
    if (meta['status'] != 'SOURCE_RECEIVER_PRIOR_PREDICTIONS_COMPLETE'
            or meta['code_commit'] != STATISTICS_CODE or meta['prior_fit_roles'] != ['L_s','U_s']
            or (meta['prior_samples'],meta['prior_L_s'],meta['prior_U_s']) != (12600,1260,11340)
            or meta['prior_fit_seed'] != 2026110707 or meta['V_fit_rows'] != 0
            or meta['TX_labels_read'] != 0 or meta['target_inputs'] != 0):
        raise ValueError('Original source-only fitted receiver priors required')
    with np.load(folder/'fitted_priors.npz',allow_pickle=False) as file:
        mean = np.stack([file[f'{rx}__other__mean'] for rx in RECEIVERS])
        covariance = np.stack([file[f'{rx}__other__covariance'] for rx in RECEIVERS])
    if (mean.shape != (5,512) or covariance.shape != (5,512,512)
            or mean.dtype != np.float32 or covariance.dtype != np.float32
            or not np.isfinite(mean).all() or not np.isfinite(covariance).all()):
        raise ValueError('Exact finite source IQ moment bank required')
    np.linalg.cholesky(covariance.astype(np.float64))
    cache = root/'runs'/R07/'cache'
    rx = np.load(cache/'train/rx.npy',allow_pickle=False)
    if rx.shape != (63000,) or set(rx.tolist()) != set(RECEIVERS):
        raise ValueError('Original five source RX assignment required')
    assignment = np.searchsorted(RECEIVERS,rx)
    return (tensor(mean,device),tensor(covariance,device)),torch.tensor(assignment.tolist(),device=device,dtype=torch.long)


def choose_prior(priors,assignment,source_indices,epoch):
    if source_indices is None or not 0 <= epoch <= 30:
        raise ValueError('Explicit source indices and fixed single30 epoch required')
    if epoch % 2:
        return None
    index = assignment[source_indices]
    return tuple(value[index] for value in priors)


def mismatch_readout(model,reader,received,clean,features,scale,paired_objective,*,
                     source_indices,epoch,priors,assignment):
    prior = choose_prior(priors,assignment,source_indices,epoch)
    restored,aux,loss,metrics = shrinkage_training_readout(model,reader,received,clean,features,scale,
        paired_objective,prior=prior)
    metrics['receiver_mismatch_prior_fraction'] = torch.full_like(loss,float(prior is not None))
    return restored,aux,loss,metrics
