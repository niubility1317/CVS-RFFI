"""Persist the actual B model and its MoPC anchor for a split C-method run.

These files are private experiment outputs, never repository artifacts. Load
only files produced by this run: torch serialization contains model objects.
"""
from pathlib import Path

import torch

from .io import require


def _binding(job, k, split_id, b_seed):
    return dict(key=job['key'], profile=job['profile'], model_seed=int(job['seed']),
                capsule=job['capsule'], k=int(k), old_split_id=split_id, b_seed=int(b_seed))


def load_b_state(path, job, k, split_id, b_seed, device):
    # Outputs are created internally by save_b_state, not accepted from users.
    payload = torch.load(Path(path), map_location=device, weights_only=False)
    require(payload['version'] == 1 and payload['binding'] == _binding(job, k, split_id, b_seed),
            'Saved B state binding differs')
    require(payload['anchor'].method == 'mopc_hr' and payload['anchor'].model is not None and
            payload['anchor'].prototypes is not None, 'Saved MoPC B anchor is incomplete')
    return payload


def save_b_state(path, job, k, split_id, b_seed, model, anchor, old_prototypes=None):
    path = Path(path)
    require(job['profile'] != 'protonet_cda' or old_prototypes is not None,
            'ProtoNet B prototypes are missing')
    payload = dict(version=1, binding=_binding(job, k, split_id, b_seed), model=model, anchor=anchor,
                   old_prototypes=old_prototypes)
    # Exclusive creation protects all earlier results and prevents accidental
    # reuse of a half-written B state after interruption.
    with path.open('xb') as stream:
        torch.save(payload, stream)
    loaded = load_b_state(path, job, k, split_id, b_seed, next(model.parameters()).device)
    for old, restored in ((model.state_dict(), loaded['model'].state_dict()),
                          (anchor.model.state_dict(), loaded['anchor'].model.state_dict())):
        require(old.keys() == restored.keys() and all(torch.equal(old[n], restored[n]) for n in old),
                'Saved B parameters or buffers differ')
    require(torch.equal(anchor.prototypes, loaded['anchor'].prototypes), 'Saved B anchor prototypes differ')
    require(old_prototypes is None or torch.equal(old_prototypes, loaded['old_prototypes']),
            'Saved native B prototypes differ')
    return loaded['binding']
