"""Fit CIL-core transplants from one already fitted, old-only B state.

This module never receives query data or a scorer. Each registration forks the
same B state; callers must name transplants separately from native CIL methods.
"""
import copy
import torch
from . import group2_cil_bridge as cil
from . import group2_lora_bridge as lora


def prepare_b(model, profile, extractor, old_x, old_y, *, old_count=6, old_prototypes=None, b_scores=None):
    result = {'lora': lora.build_gallery(model, extractor, old_x, old_y, old_count=old_count)}
    if profile in ('dadda_sda', 'mrior_sda', 'twostage', 'poster_ft', 'feature_separation_hu2024', 'protonet_cda', 'radionet_ada_knn', 'tweak_m1'):
        if profile == 'protonet_cda':
            if old_prototypes is None or len(old_prototypes) != old_count:
                raise ValueError('ProtoNet B requires its actual old-class prototypes')
            adapter = cil.from_protonet(model, old_prototypes)
        elif profile == 'tweak_m1':
            if b_scores is None:raise ValueError('Tweak M1 requires its actual B centroids and radii')
            adapter=cil.from_tweak(model,b_scores,old_count=old_count)
        elif profile == 'radionet_ada_knn':
            adapter = cil.from_radionet(model, b_scores, old_count=old_count)
        else:
            adapter = cil.from_adapted_model(model, profile, old_count=old_count)
        result['mopc_hr'] = cil.make_anchor(adapter, old_x, old_y, method='mopc_hr')
        if profile=='tweak_m1':
            result['mopc_hr'].protocol_metadata.update(B_decision='Tweak_M1_centroid_radius',
                score_coordinates='finite_rank_preserving_replacement_for_native_minus_1e30_sentinel')
        orthogonal = copy.deepcopy(result['mopc_hr'])
        orthogonal.method = 'orthogonal'
        orthogonal.model.classifier_weight = torch.nn.Parameter(
            orthogonal.model.fc.weight.detach().clone(), requires_grad=False)
        result['orthogonal'] = orthogonal
    return result


def register_c(prepared, new_x, new_y, *, seed, methods, mopc_max_grad_norm=None):
    if len(methods) != len(set(methods)):
        raise ValueError('duplicate registration method')
    if mopc_max_grad_norm is not None and 'mopc_hr' not in methods:
        raise ValueError('MoPC diagnostic clipping requires the MoPC method')
    missing = set(methods) - set(prepared)
    if missing:
        raise NotImplementedError('No compatible B-to-C state bridge for: ' + ', '.join(sorted(missing)))
    result = {}
    for method in methods:
        if method == 'lora':
            result[method] = lora.register_lora(prepared[method], new_x, new_y)
        else:
            if method=='mopc_hr' and mopc_max_grad_norm is not None:
                result[method] = cil.register(prepared[method], new_x, new_y, seed=seed,
                                              mopc_max_grad_norm=mopc_max_grad_norm)
            else:
                result[method] = cil.register(prepared[method], new_x, new_y, seed=seed)
    return result
