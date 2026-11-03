"""Source-trained Hu feature-separation core on a native C source backbone.

This project transplant keeps the C encoder and resizes Hu's TX/RX branches
to the C embedding dimension. Source-only auxiliary training is required before
target B; target B freezes the RX branch and adapts the C encoder, shared/TX
projection and TX classifier. No query or target truth is accepted here.
"""
from __future__ import annotations

import copy
import time

import torch
from torch import nn
from torch.nn import functional as F

from .group2_c_backbone_adapter import C_SOURCE_SPECS
from .group2_c_backbone_poster import PosterProjectedC
from .group2_c_backbone_state import c_source_features
from .group2_lora_bridge import build_gallery
from .vendor_extra.hu.losses import feature_separation_loss
from .vendor_extra.hu.model import ChannelAttention
from .vendor_native.comparison_suite.registration import RegistrationState
from .vendor_native.paper_reproduction.mopc_hr_non_exemplar_cil_sei.algorithm import compute_class_prototypes


class HuSourceAux(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.shared = nn.Sequential(nn.Linear(dim, 512), nn.LeakyReLU(), ChannelAttention(512))
        self.tx_branch = nn.Sequential(nn.Linear(512, dim), nn.BatchNorm1d(dim), nn.LeakyReLU())
        self.rx_branch = nn.Sequential(nn.Linear(512, dim), nn.BatchNorm1d(dim), nn.LeakyReLU())
        self.tx_classifier, self.rx_classifier = nn.Linear(dim, 6), nn.Linear(dim, 5)

    def forward(self, z):
        shared = self.shared(z)
        tx, rx = self.tx_branch(shared), self.rx_branch(shared)
        return {'tx_features': tx, 'rx_features': rx,
                'tx_logits': self.tx_classifier(tx), 'rx_logits': self.rx_classifier(rx)}


def _source_guard(bundle):
    if (bundle.get('provenance', {}).get('source_only') is not True or
            not bundle.get('source_checkpoint') or bundle.get('method') not in C_SOURCE_SPECS or
            not isinstance(bundle.get('source_model_seed'), int)):
        raise ValueError('C source checkpoint provenance must be source-only and explicit')


def train_hu_source_aux(bundle, source_x, source_y, source_rx, *, model_seed,
                        epochs=200, batch_size=256, embedding_dim=None, logger=None):
    """Train auxiliary Hu TX/RX state only from explicitly supplied L_s."""
    _source_guard(bundle)
    if bundle['source_model_seed'] != int(model_seed):
        raise ValueError('Hu auxiliary source seed differs from C source seed')
    if (source_y.dtype != torch.long or source_rx.dtype != torch.long or source_y.ndim != 1 or
            source_rx.ndim != 1 or len(source_x) != len(source_y) or len(source_y) != len(source_rx) or
            not len(source_y) or not bool(torch.isfinite(source_x).all()) or
            tuple(torch.unique(source_y, sorted=True).tolist()) != tuple(range(6)) or
            bool((source_rx < 0).any()) or bool((source_rx >= 5).any()) or epochs <= 0 or batch_size < 2):
        raise ValueError('invalid labeled source L_s inputs')
    dim = int(C_SOURCE_SPECS[bundle['method']]['embedding_dim'] if embedding_dim is None else embedding_dim)
    backbone = copy.deepcopy(bundle['model']).eval().requires_grad_(False)
    device = next(backbone.parameters()).device
    started = time.perf_counter()
    with torch.no_grad():
        z = torch.cat([c_source_features(backbone, part.to(device)) for part in source_x.split(128)])
    if z.shape != (len(source_x), dim) or not bool(torch.isfinite(z).all()):
        raise ValueError('invalid C source features')
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(int(model_seed))
        aux = HuSourceAux(dim)
    aux = aux.to(z)
    optimizer = torch.optim.Adam(aux.parameters(), lr=.005)
    gen = torch.Generator().manual_seed(int(model_seed))
    source_y, source_rx = source_y.to(device), source_rx.to(device)
    steps = 0
    for epoch in range(int(epochs)):
        aux.train()
        for indices in torch.randperm(len(z), generator=gen).split(int(batch_size)):
            if len(indices) < 2:
                raise ValueError('source batch has fewer than two samples for Hu batch normalization')
            indices = indices.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss, _ = feature_separation_loss(aux(z[indices]), source_y[indices], source_rx[indices])
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError('nonfinite Hu source auxiliary loss')
            loss.backward()
            if any(p.grad is not None and not bool(torch.isfinite(p.grad).all()) for p in aux.parameters()):
                raise FloatingPointError('nonfinite Hu source auxiliary gradient')
            optimizer.step(); steps += 1
        if logger is not None:
            logger(dict(stage='source_aux', epoch=epoch+1, steps=steps, last_loss=float(loss.detach()),
                        target_used=False, query_used=False))
    metadata = dict(source_role='L_s', target_contact=False, source_only_checkpoint=True,
        source_checkpoint=bundle['source_checkpoint'], C_source_method=bundle['method'], model_seed=int(model_seed),
        source_epochs=int(epochs), source_batch=int(batch_size), source_lr=.005, source_optimizer='Adam',
        source_steps=steps, source_count=len(source_y), embedding_dim=dim,
        source_seconds=time.perf_counter()-started, source_views='clean_L_s_only',
        claim='Hu_feature_separation_aux_on_frozen_C_source_project_transplant_not_native_Hu_source_architecture')
    return {'state': {key: value.detach().cpu().clone() for key, value in aux.state_dict().items()}, 'metadata': metadata}


class HuTargetAdapter(nn.Module):
    def __init__(self, backbone, aux):
        super().__init__()
        self.backbone, self.aux = backbone, aux

    def forward(self, x):
        shared = self.aux.shared(c_source_features(self.backbone, x))
        return self.aux.tx_classifier(self.aux.tx_branch(shared))


def fit_hu_b(bundle, source_aux, old_x, old_y, *, seed, model_seed, epochs=100, embedding_dim=None,
             required_source_epochs=200):
    """Adapt the C encoder/shared/TX state with target-old support only."""
    _source_guard(bundle)
    meta = source_aux.get('metadata') or {}
    if (meta.get('source_role') != 'L_s' or meta.get('target_contact') is not False or
            meta.get('source_checkpoint') != bundle['source_checkpoint'] or
            meta.get('C_source_method') != bundle['method'] or meta.get('source_epochs') != required_source_epochs or
            meta.get('model_seed') != int(model_seed) or bundle['source_model_seed'] != int(model_seed)):
        raise ValueError('Hu auxiliary source state does not match C source checkpoint, seed and budget')
    if (old_y.dtype != torch.long or old_y.ndim != 1 or len(old_x) != len(old_y) or not len(old_y) or
            tuple(torch.unique(old_y, sorted=True).tolist()) != tuple(range(6)) or not bool(torch.isfinite(old_x).all())):
        raise ValueError('invalid old-only support')
    counts = torch.bincount(old_y, minlength=6)
    if not bool((counts == counts[0]).all()) or epochs <= 0:
        raise ValueError('invalid balanced support or B budget')
    dim = int(C_SOURCE_SPECS[bundle['method']]['embedding_dim'] if embedding_dim is None else embedding_dim)
    if meta.get('embedding_dim') != dim:
        raise ValueError('Hu source embedding dimension mismatch')
    aux = HuSourceAux(dim).to(old_x.device)
    aux.load_state_dict(source_aux['state'], strict=True)
    aux.rx_branch.requires_grad_(False).eval(); aux.rx_classifier.requires_grad_(False).eval()
    model = HuTargetAdapter(copy.deepcopy(bundle['model']).requires_grad_(True), aux).to(old_x.device)
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=.005)
    gen = torch.Generator().manual_seed(int(seed))
    started = time.perf_counter(); steps = 0
    for _ in range(int(epochs)):
        model.train(); model.aux.rx_branch.eval(); model.aux.rx_classifier.eval()
        for indices in torch.randperm(len(old_y), generator=gen).split(256):
            indices = indices.to(old_x.device)
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(old_x[indices]), old_y[indices])
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError('nonfinite Hu B adaptation loss')
            loss.backward()
            if any(p.grad is not None and not bool(torch.isfinite(p.grad).all())
                   for group in optimizer.param_groups for p in group['params']):
                raise FloatingPointError('nonfinite Hu B adaptation gradient')
            optimizer.step(); steps += 1
    projection = nn.Sequential(model.aux.shared, model.aux.tx_branch)
    c = bundle['method']
    adapted = PosterProjectedC(model.backbone, projection, model.aux.tx_classifier, c).eval()
    metadata = dict(C_source_method=c, B_method='feature_separation_hu2024', source_only_checkpoint=True,
        C_inherits_adapted_encoder=True, B_tx_projection_preserved_in_C=True, B_rx_branch_frozen=True,
        B_epochs=int(epochs), B_batch=256, B_lr=.005, B_optimizer='Adam', B_optimizer_steps=steps,
        B_training_seconds=time.perf_counter()-started, target_old_labels=len(old_y), query_used_for_fit=False,
        source_auxiliary=dict(meta), cross_method_transplant=True,
        source_checkpoint=bundle['source_checkpoint'], claim='Hu_source_TX_RX_core_on_C_backbone_project_transplant')
    classes = tuple(range(6))
    if c == 'lora_triplet_knn':
        anchor = build_gallery(adapted, c_source_features, old_x, old_y, old_count=6)
        metadata['B_decision_handoff'] = 'Hu_TX_classifier_replaced_by_native_LoRa_kNN_TX_projection_retained'
    else:
        anchor = RegistrationState(c, adapted, classes, 6, protocol_metadata=dict(metadata, phase='B'))
        if c == 'mopc_hr':
            with torch.no_grad():
                prototypes, ids = compute_class_prototypes(c_source_features(adapted, old_x), old_y)
            if ids.tolist() != list(classes) or not bool(torch.isfinite(prototypes).all()):
                raise ValueError('invalid Hu-adapted old prototypes')
            anchor.prototypes = prototypes.detach().clone()
            metadata['B_decision_handoff'] = 'Hu_TX_classifier_and_projection_inherited_by_MoPC'
        else:
            metadata['B_decision_handoff'] = 'Hu_TX_weights_as_cosine_old_weights_bias_discarded'
        anchor.protocol_metadata.update(metadata)
    return {'c_state': {'anchor': anchor, 'method': c, 'metadata': metadata}, 'metadata': metadata}
