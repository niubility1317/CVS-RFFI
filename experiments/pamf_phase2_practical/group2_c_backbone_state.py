"""Transfer a fitted B encoder into its original C registration mechanism."""
from __future__ import annotations

import copy

import torch

from .group2_c_backbone_adapter import CrossAdaptationAdapter
from .group2_lora_bridge import build_gallery
from .vendor_native.comparison_suite.registration import RegistrationState
from .vendor_native.paper_reproduction.mopc_hr_non_exemplar_cil_sei.algorithm import compute_class_prototypes


def c_source_features(model, x):
    output = model(x)
    return output['features'] if isinstance(output, dict) else output


def prepare_c_state(b_state, old_x, old_y, *, source_provenance):
    """Use the adapted C backbone and the same labeled old support as B.

    The MoPC old prototypes or LoRa old gallery are re-extracted with this
    adapted model, avoiding stale pre-B source embeddings. Orthogonal retains
    the B-adapted native classifier weights. RadioNet's kNN B decision and any
    LoRa auxiliary B classifier are replaced by the destination C decision.
    """
    if source_provenance.get('source_only') is not True:
        raise ValueError('C source checkpoint provenance must be source-only')
    wrapper = b_state.model
    if (not hasattr(wrapper, 'base') or not hasattr(wrapper, 'columns') or
            not isinstance(wrapper.base, CrossAdaptationAdapter)):
        raise ValueError('B state is not a verified C-backbone transplant')
    adapter = wrapper.base
    classes = tuple(range(adapter.classes))
    if tuple(b_state.classes) != classes or wrapper.columns.tolist() != list(classes):
        raise ValueError('B old class order differs from C')
    if (old_y.dtype != torch.long or old_y.ndim != 1 or len(old_x) != len(old_y) or not len(old_y) or
            tuple(torch.unique(old_y, sorted=True).tolist()) != classes or not bool(torch.isfinite(old_x).all())):
        raise ValueError('invalid target old support')
    counts = torch.bincount(old_y, minlength=len(classes))
    if not bool((counts == counts[0]).all()):
        raise ValueError('old support must have equal K')
    model = copy.deepcopy(adapter.backbone).eval()
    metadata = dict(C_source_method=adapter.c_method, B_method=adapter.b_method,
                    C_inherits_adapted_encoder=True, source_only_checkpoint=True,
                    target_old_labels=len(old_y), query_used_for_fit=False,
                    cross_method_transplant=True)
    if adapter.b_method == 'radionet_ada_knn':
        metadata['B_decision_handoff'] = 'B_kNN_replaced_by_native_C_decision_adapted_encoder_retained'
    elif adapter.b_method == 'protonet_cda':
        if b_state.decision != 'protonet' or b_state.weight is None:
            raise ValueError('ProtoNet B requires its actual target old prototypes')
        metadata['B_decision_handoff'] = 'B_prototypes_translated_into_native_C_state'
    else:
        metadata['B_decision_handoff'] = 'native_C_decision_with_adapted_source_model'
    if adapter.c_method == 'lora_triplet_knn':
        anchor = build_gallery(model, c_source_features, old_x, old_y, old_count=len(classes))
        metadata.update(old_gallery='B_encoder_target_old_support',
                        B_auxiliary_head='source_centroid_head_discarded_after_B')
        if adapter.b_method == 'protonet_cda':
            metadata['B_decision_handoff'] = 'B_prototype_decision_replaced_by_native_LoRa_kNN_old_support_retained'
    else:
        anchor = RegistrationState(adapter.c_method, model, classes, len(classes),
                                   protocol_metadata=dict(metadata, phase='B', B_executed=True,
                                                          C_inherits_B_state=True))
        if adapter.c_method == 'mopc_hr':
            with torch.no_grad():
                z = c_source_features(model, old_x)
                prototypes, ids = compute_class_prototypes(z, old_y)
            if ids.tolist() != list(classes) or not bool(torch.isfinite(prototypes).all()):
                raise ValueError('invalid B-adapted MoPC old prototypes')
            if adapter.b_method == 'protonet_cda' and not torch.allclose(prototypes, b_state.weight):
                raise ValueError('B prototypes do not match transferred old support')
            anchor.prototypes = prototypes.detach().clone()
            metadata['old_prototypes'] = 'B_encoder_target_old_support'
            if adapter.b_method == 'protonet_cda':
                metadata['B_decision_handoff'] = 'B_prototypes_in_C_augmentation_native_MoPC_classifier_restored'
            anchor.protocol_metadata['old_prototypes'] = metadata['old_prototypes']
        elif adapter.c_method == 'orthogonal':
            if adapter.b_method == 'protonet_cda':
                if b_state.weight.shape != model.classifier_weight.shape:
                    raise ValueError('B prototypes do not match orthogonal classifier dimension')
                with torch.no_grad():
                    model.classifier_weight.copy_(b_state.weight)
                metadata['B_decision_handoff'] = 'B_prototypes_as_cosine_old_weights_squared_distance_bias_discarded'
            metadata['old_classifier'] = 'B_adapted_native_orthogonal_head'
            anchor.protocol_metadata['old_classifier'] = metadata['old_classifier']
        else:
            raise ValueError('unsupported native C method')
    if isinstance(anchor, RegistrationState):
        anchor.protocol_metadata['B_decision_handoff'] = metadata['B_decision_handoff']
    return {'anchor': anchor, 'method': adapter.c_method, 'metadata': metadata}
