"""B-objective transplant onto a source C model, followed by its C core.

These are project integrations. No target query, truth, or result-dependent
selection is accepted by the fitting interfaces in this module.
"""
from __future__ import annotations

import torch

from .external_execution import seed_for
from .external_native import _adaptation_config, _registration_config
from .group2_c_backbone_adapter import C_SOURCE_SPECS, CrossAdaptationAdapter, lora_source_centroid_head
from .group2_c_backbone_state import prepare_c_state
from .group2_lora_bridge import register_lora
from .vendor_native.comparison_suite.adaptation import fit_adaptation
from .vendor_native.comparison_suite.registration import register_new_classes


BOUND_B = frozenset({'protonet_cda', 'mrior_sda', 'dadda_sda', 'twostage_sda', 'radionet_ada_knn'})
REPLAY_B = frozenset({'mrior_sda', 'dadda_sda', 'twostage_sda', 'radionet_ada_knn'})


def fit_b_from_c_source(rt, bundle, b_method, old_x, old_y, *, seed, model_seed=None,
                        source_x=None, source_y=None, spec=None, logger=None):
    """Fit one old-only B state; caller reuses it for all N at fixed K."""
    provenance = bundle.get('provenance') or {}
    if provenance.get('source_only') is not True:
        raise ValueError('C source checkpoint provenance must be source-only')
    c_method = bundle['method']
    if b_method not in BOUND_B or c_method not in C_SOURCE_SPECS:
        raise ValueError('unbound B/C transplant')
    if b_method in REPLAY_B and (source_x is None or source_y is None):
        raise ValueError('this native B core requires authorized L_s replay')
    selected = dict(C_SOURCE_SPECS[c_method] if spec is None else spec)
    dim = int(selected['embedding_dim'])
    taps = tuple(selected['tap_names']) if b_method in {'dadda_sda', 'twostage_sda'} else ()
    aux = {}
    if c_method == 'lora_triplet_knn':
        native = bundle['native_state']
        aux['source_head'], aux['source_head_metadata'] = lora_source_centroid_head(
            native['source_enrollment_features'], native['source_enrollment_y'],
            classes=6, provenance=provenance)
    init_seed = seed_for(int(seed if model_seed is None else model_seed), c_method, b_method, 'B_aux_source')
    adapter = CrossAdaptationAdapter(bundle['model'], c_method, b_method, dim,
                                     tap_names=taps, init_seed=init_seed, **aux)
    counts = torch.bincount(old_y, minlength=6)
    if old_y.ndim != 1 or len(old_x) != len(old_y) or not bool((counts == counts[0]).all()):
        raise ValueError('B requires balanced old-only support')
    cfg = _adaptation_config(rt, b_method, seed, old_x.device, int(counts[0]), logger)
    cfg['source_classes'] = list(range(6))
    b_state = fit_adaptation(b_method, adapter, old_x, old_y, tuple(range(6)), cfg,
                             source_x=source_x, source_y=source_y,
                             metadata=dict(provenance, cross_method_transplant=True))
    c_state = prepare_c_state(b_state, old_x, old_y, source_provenance=provenance)
    metadata = dict(c_state['metadata'], source_checkpoint=bundle.get('source_checkpoint'),
                    B_auxiliary_initialization_seed=init_seed,
                    B_multiscale_taps=list(taps), B_source_replay=b_method in REPLAY_B,
                    B_auxiliary_source_head=aux.get('source_head_metadata'),
                    B_source_backbone=c_method, B_method=b_method, B_resources=dict(b_state.resources))
    return {'b_state': b_state, 'c_state': c_state, 'metadata': metadata}


def register_native_c(rt, prepared, new_x, new_y, new_ids, *, seed):
    """Fork a completed B state into the original destination C core."""
    method = prepared['method']
    if method == 'lora_triplet_knn':
        if tuple(new_ids) != tuple(torch.unique(new_y, sorted=True).tolist()):
            raise ValueError('new class IDs do not match LoRa support')
        return register_lora(prepared['anchor'], new_x, new_y, k=15)
    if method not in {'mopc_hr', 'orthogonal'}:
        raise ValueError('unbound native C registration')
    cfg = _registration_config(rt, method, seed)
    device = new_x.device
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == 'cuda' else []
    with torch.random.fork_rng(devices=devices):
        torch.random.default_generator.manual_seed(int(seed))
        if device.type == 'cuda':
            with torch.cuda.device(device):
                torch.cuda.manual_seed(int(seed))
        state = register_new_classes(prepared['anchor'], new_x, new_y, new_ids, config=cfg)
    state.protocol_metadata.update(prepared['metadata'])
    return state
