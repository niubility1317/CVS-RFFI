"""POSTER dense-reset B core transplanted onto an existing C source model.

The source encoder is frozen during B; two freshly initialized dense layers
map its embedding back to the same dimension, and a reset six-class head is
trained on old support. The learned projection stays inside the C model.
This declared project transplant is not the original POSTER architecture.
"""
from __future__ import annotations

import copy
import time
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F

from .group2_c_backbone_adapter import C_SOURCE_SPECS
from .group2_c_backbone_state import c_source_features
from .group2_lora_bridge import build_gallery
from .vendor_extra.li_backbones import keras_initialize
from .vendor_native.comparison_suite.registration import RegistrationState
from .vendor_native.paper_reproduction.mopc_hr_non_exemplar_cil_sei.algorithm import compute_class_prototypes


class PosterProjectedC(nn.Module):
    """Keep the C source encoder and POSTER B projection in native C state."""
    def __init__(self, backbone, projection, head, c_method):
        super().__init__()
        self.backbone = copy.deepcopy(backbone)
        self.projection = copy.deepcopy(projection)
        self.c_method = c_method
        if c_method == 'mopc_hr':
            self.fc = copy.deepcopy(head)
        elif c_method == 'orthogonal':
            self.classifier_weight = nn.Parameter(head.weight.detach().clone())
        elif c_method != 'lora_triplet_knn':
            raise ValueError('unbound native C model')

    def forward(self, x):
        z = self.projection(c_source_features(self.backbone, x))
        if self.c_method == 'mopc_hr':
            return {'features': z, 'tx_logits': self.fc(z)}
        if self.c_method == 'orthogonal':
            return {'features': z, 'tx_logits': F.normalize(z, dim=1) @
                    F.normalize(self.classifier_weight, dim=1).T}
        return {'features': z}


def fit_poster_b(bundle, old_x, old_y, *, seed, epochs=10, embedding_dim=None):
    """Run the fixed POSTER target-support dense reset; no query is accepted."""
    if bundle.get('provenance', {}).get('source_only') is not True:
        raise ValueError('C source checkpoint provenance must be source-only')
    c = bundle['method']
    if c not in C_SOURCE_SPECS:
        raise ValueError('unbound native C model')
    if (old_y.dtype != torch.long or old_y.ndim != 1 or len(old_x) != len(old_y) or not len(old_y) or
            tuple(torch.unique(old_y, sorted=True).tolist()) != tuple(range(6)) or
            not bool(torch.isfinite(old_x).all())):
        raise ValueError('invalid old-only support')
    counts = torch.bincount(old_y, minlength=6)
    if not bool((counts == counts[0]).all()) or epochs < 0:
        raise ValueError('invalid balanced support or B budget')
    dim = int(C_SOURCE_SPECS[c]['embedding_dim'] if embedding_dim is None else embedding_dim)
    backbone = copy.deepcopy(bundle['model']).eval().requires_grad_(False)
    started = time.perf_counter()
    with torch.no_grad():
        z = c_source_features(backbone, old_x).detach()
    if z.ndim != 2 or z.shape != (len(old_x), dim) or not bool(torch.isfinite(z).all()):
        raise ValueError('C source embedding shape or values invalid')
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(int(seed))
        projection = nn.Sequential(nn.Linear(dim, 256), nn.ReLU(), nn.Linear(256, dim), nn.ReLU())
        head = nn.Linear(dim, 6)
        projection.apply(keras_initialize); head.apply(keras_initialize)
    projection, head = projection.to(z), head.to(z)
    optimizer = torch.optim.Adam(list(projection.parameters()) + list(head.parameters()), lr=.001, eps=1e-7)
    generator = torch.Generator().manual_seed(int(seed))
    steps = 0
    for _ in range(int(epochs)):
        for indices in torch.randperm(len(z), generator=generator).split(256):
            indices = indices.to(z.device)
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(head(projection(z[indices])), old_y[indices])
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError('nonfinite POSTER transplant loss')
            loss.backward()
            if any(p.grad is not None and not bool(torch.isfinite(p.grad).all())
                   for group in optimizer.param_groups for p in group['params']):
                raise FloatingPointError('nonfinite POSTER transplant gradient')
            optimizer.step(); steps += 1
    model = PosterProjectedC(backbone, projection, head, c).eval()
    metadata = dict(C_source_method=c, B_method='poster_ft', source_only_checkpoint=True,
        C_inherits_adapted_encoder=True, B_dense_projector_preserved_in_C=True,
        source_backbone_frozen_in_B=True, B_epochs=int(epochs), B_batch=256, B_lr=.001,
        B_optimizer='Adam_eps1e-7', B_optimizer_steps=steps, target_old_labels=len(old_y),
        B_training_seconds=time.perf_counter()-started, query_used_for_fit=False,
        cross_method_transplant=True, source_checkpoint=bundle.get('source_checkpoint'),
        B_projector='reset_dense_dim_to_256_to_dim_plus_reset_6class_head',
        claim='POSTER_dense_reset_core_C_encoder_project_transplant_not_native_POSTER')
    classes = tuple(range(6))
    if c == 'lora_triplet_knn':
        anchor = build_gallery(model, c_source_features, old_x, old_y, old_count=6)
        metadata['B_decision_handoff'] = 'reset_B_classifier_replaced_by_native_LoRa_kNN_projection_retained'
    else:
        anchor = RegistrationState(c, model, classes, 6, protocol_metadata=dict(metadata, phase='B'))
        if c == 'mopc_hr':
            with torch.no_grad():
                prototypes, ids = compute_class_prototypes(c_source_features(model, old_x), old_y)
            if ids.tolist() != list(classes) or not bool(torch.isfinite(prototypes).all()):
                raise ValueError('invalid POSTER-adapted old prototypes')
            anchor.prototypes = prototypes.detach().clone()
            metadata['B_decision_handoff'] = 'reset_B_classifier_and_projection_inherited_by_MoPC'
        else:
            metadata['B_decision_handoff'] = 'reset_B_classifier_weights_as_cosine_old_weights_bias_discarded'
        anchor.protocol_metadata.update(metadata)
    return {'b_state': SimpleNamespace(model=model), 'c_state': {'anchor': anchor, 'method': c,
            'metadata': metadata}, 'metadata': metadata}
