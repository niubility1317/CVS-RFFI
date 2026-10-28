"""Head-aligned branch evidence summaries; fixed geometry and row-local inputs."""
from __future__ import annotations

import numpy as np

from . import source_open_readout as core


DESCRIPTOR_SCHEMA = 'head_aligned_branch_summary_v1'
DESCRIPTOR_DIM = 100


def _summary(values, probabilities):
    ordered = np.sort(values, axis=1)
    maximum = ordered[:, -1]
    # Sorting products fixes accumulation order under class-column permutation,
    # including equal logits; no winning class is chosen to break a head tie.
    weighted = np.sort(values * probabilities, axis=1).sum(axis=1)
    return np.column_stack((maximum, ordered[:, -2], ordered.mean(axis=1),
                            ordered.std(axis=1, ddof=0), weighted, maximum - weighted))


def descriptor(branches, logits6, geometry):
    values = core._branches(branches)
    fitted = core._geometry(geometry)
    logits = core._matrix(logits6, 'complete six-class logits', 6, fp32=True)
    if len(logits) != len(values['fused160']):
        raise ValueError('Branch/head row alignment differs')
    top = logits.max(axis=1)
    exponentials = np.exp(logits - top[:, None])
    denominator = np.sort(exponentials, axis=1).sum(axis=1)
    probabilities = exponentials / denominator[:, None]
    columns = []
    for name in core.DIMENSIONS:
        branch = fitted[name]
        unit = core._unit((values[name] - branch['mean']) / branch['scale'])
        centers = branch['prototypes']
        cosine = np.clip(np.einsum('nd,kd->nk', unit, centers, optimize=False), -1., 1.)
        white = unit @ branch['whitener'].T
        white_centers = np.einsum('nd,kd->nk', centers, branch['whitener'], optimize=False)
        distances = (np.einsum('nd,nd->n', white, white)[:, None]
            + np.einsum('kd,kd->k', white_centers, white_centers)[None, :]
            - 2. * np.einsum('nd,kd->nk', white, white_centers, optimize=False))
        columns.extend((_summary(cosine, probabilities), _summary(-np.maximum(distances, 0.), probabilities)))
    ordered_logits = np.sort(logits, axis=1)
    columns.append(np.column_stack((1. / denominator, top, ordered_logits[:, -1] - ordered_logits[:, -2],
                                    top + np.log(denominator))))
    result = core._fp32(np.concatenate(columns, axis=1), 'head-aligned branch descriptor')
    if result.shape != (len(logits), DESCRIPTOR_DIM):
        raise ValueError('Head-aligned descriptor dimension differs')
    return result
