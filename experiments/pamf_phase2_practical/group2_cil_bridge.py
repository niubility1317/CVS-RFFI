"""Explicit DA-backbone adaptations of native MoPC and Orthogonal C cores.

These are transplanted C mechanisms, not author architectures or pretrained
orthogonal-source models. B logits are preserved before the C-specific bridge.
"""
from __future__ import annotations

import copy
import torch
from torch import nn
from torch.nn import functional as F

from .vendor_native.comparison_suite.registration import RegistrationConfig, RegistrationState, register_new_classes
from .vendor_native.paper_reproduction.mopc_hr_non_exemplar_cil_sei.algorithm import compute_class_prototypes


class AdaptedClassifier(nn.Module):
    def __init__(self, base: nn.Module, fc: nn.Linear, output_key: str):
        super().__init__()
        self.base, self.fc, self.output_key = base, fc, output_key

    def forward(self, x):
        out = self.base(x)
        z = out if self.output_key is None else out[self.output_key]
        return {'features': z, 'tx_logits': self.fc(z)}


class ProtoClassifier(nn.Module):
    """Represent squared-Euclidean old B scores with an exact affine head."""
    def __init__(self, base, prototypes):
        super().__init__()
        self.base = copy.deepcopy(base).eval()
        rows, dim = prototypes.shape
        self.fc = nn.Linear(dim + 1, rows).to(prototypes)
        with torch.no_grad():
            self.fc.weight[:, :dim].copy_(2 * prototypes)
            self.fc.weight[:, dim].fill_(-1)
            self.fc.bias.copy_(-prototypes.square().sum(1))

    def forward(self, x):
        z = self.base(x)['features']
        mapped = torch.cat((z, z.square().sum(1, keepdim=True)), 1)
        return {'features': mapped, 'tx_logits': self.fc(mapped)}


def from_protonet(model: nn.Module, prototypes: torch.Tensor) -> ProtoClassifier:
    if prototypes.ndim != 2 or len(prototypes) == 0 or not bool(torch.isfinite(prototypes).all()):
        raise ValueError('B prototypes must be finite and nonempty')
    return ProtoClassifier(model, prototypes.detach()).eval()


class RadioNetClassifier(nn.Module):
    """Retain B kNN votes as explicit features plus the B embedding."""
    def __init__(self, model, scores, old_count):
        super().__init__()
        self.base = copy.deepcopy(model).eval()
        self.register_buffer('bank', scores.support_features.detach().clone())
        self.register_buffer('labels', scores.support_labels.detach().clone())
        self.k, self.old_count = int(scores.knn_k), old_count
        self.fc = nn.Linear(self.bank.shape[1] + old_count, old_count).to(self.bank)
        with torch.no_grad():
            self.fc.weight.zero_()
            self.fc.weight[:, :old_count].copy_(torch.eye(old_count, device=self.bank.device, dtype=self.bank.dtype))
            self.fc.bias.zero_()

    def forward(self, x):
        z = self.base.embedding(x)
        distances = (1 - F.normalize(z, dim=1) @ F.normalize(self.bank, dim=1).T).clamp_min(0)
        d, indices = distances.topk(self.k, dim=1, largest=False, sorted=True)
        zero = d == 0
        weights = torch.where(zero.any(1, keepdim=True), zero.to(d.dtype), d.clamp_min(1e-12).reciprocal())
        votes = z.new_zeros((len(z), self.old_count))
        votes.scatter_add_(1, self.labels[indices], weights)
        mapped = torch.cat((votes, z), 1)
        return {'features': mapped, 'tx_logits': self.fc(mapped)}


def from_radionet(model, scores, *, old_count):
    if not hasattr(model, 'embedding') or getattr(scores, 'decision', None) != 'knn':
        raise ValueError('RadioNet requires its actual B kNN state and embedding')
    bank, labels = scores.support_features, scores.support_labels
    if bank is None or labels is None or bank.ndim != 2 or len(bank) != len(labels):
        raise ValueError('RadioNet B gallery is missing or malformed')
    if tuple(torch.unique(labels, sorted=True).tolist()) != tuple(range(old_count)):
        raise ValueError('RadioNet B gallery does not cover old classes')
    if not 0 < scores.knn_k <= len(bank) or not bool(torch.isfinite(bank).all()):
        raise ValueError('RadioNet B gallery or k is invalid')
    return RadioNetClassifier(model, scores, old_count).eval()


def from_adapted_model(model: nn.Module, profile: str, *, old_count: int) -> AdaptedClassifier:
    # First-group DA models wrap base in _ActiveClasses; only the exact six
    # ordered classes can be transferred without changing decision columns.
    if hasattr(model, 'columns') and hasattr(model, 'base'):
        if model.columns.tolist() != list(range(old_count)):
            raise ValueError('B active-class order is not the compact old-class order')
        model = model.base
    base = copy.deepcopy(model).eval()
    if profile in ('poster_ft', 'feature_separation_hu2024'):
        name = 'head' if profile == 'poster_ft' else 'tx_classifier'
        head = getattr(base.network, name, None)
        if not isinstance(head, nn.Linear) or head.out_features != old_count:
            raise ValueError('B native classifier must be a matching linear head')
        fc = copy.deepcopy(head)
        setattr(base.network, name, nn.Identity())
        return AdaptedClassifier(base, fc, None).eval()
    if profile == 'dadda_sda':
        classifier, key = base.classifier.net, 'logits'
    elif profile == 'mrior_sda':
        classifier, key = base.classifier.net, 'tx_logits'
    elif profile == 'twostage':
        classifier, key = base.tx_classifier.net, 'tx_logits'
    else:
        raise ValueError('B model has no verified affine-head bridge for this profile')
    if not isinstance(classifier, nn.Sequential) or not isinstance(classifier[-1], nn.Linear):
        raise ValueError('B classifier must end in a linear layer')
    fc = copy.deepcopy(classifier[-1])
    if fc.out_features != old_count:
        raise ValueError('B head width differs from registered old classes')
    classifier[-1] = nn.Identity()
    return AdaptedClassifier(base, fc, key).eval()


def make_anchor(model: AdaptedClassifier, old_x: torch.Tensor, old_y: torch.Tensor, *, method: str) -> RegistrationState:
    if method not in ('mopc_hr', 'orthogonal'):
        raise ValueError('only the named implemented C cores are supported')
    count = model.fc.out_features
    ids = tuple(range(count))
    if old_y.ndim != 1 or len(old_x) != len(old_y) or old_y.dtype != torch.long or tuple(torch.unique(old_y, sorted=True).tolist()) != ids:
        raise ValueError('old support does not cover the compact old classes')
    counts = torch.bincount(old_y, minlength=count)
    if not bool((counts == counts[0]).all()) or not bool(torch.isfinite(old_x).all()):
        raise ValueError('invalid old support or unequal K')
    copied = copy.deepcopy(model).eval()
    with torch.no_grad():
        old_z = copied(old_x)['features']
        if old_z.ndim != 2 or not bool(torch.isfinite(old_z).all()):
            raise ValueError('invalid old B features')
        prototypes, class_ids = compute_class_prototypes(old_z, old_y)
    if class_ids.tolist() != list(ids):
        raise ValueError('old prototype class order mismatch')
    if method == 'orthogonal':
        copied.classifier_weight = nn.Parameter(copied.fc.weight.detach().clone(), requires_grad=False)
    return RegistrationState(method, copied, ids, count, prototypes=prototypes.detach().clone(),
        protocol_metadata={'phase': 'B', 'query_fitting': False, 'B_state_inherited': True,
            'old_support_k': int(counts[0]), 'old_prototypes': 'B_encoder_old_target_support',
            'claim_boundary': 'DA_backbone_with_CIL_core_transplant_not_native_source_architecture'})


def register(anchor: RegistrationState, new_x: torch.Tensor, new_y: torch.Tensor, *,
             seed: int, epochs: int | None = None) -> RegistrationState:
    old = len(anchor.classes)
    if new_y.ndim != 1 or len(new_y) == 0 or new_y.dtype != torch.long:
        raise ValueError('new support labels are invalid')
    ids = tuple(torch.unique(new_y, sorted=True).tolist())
    if ids != tuple(range(old, old + len(ids))):
        raise ValueError('new support IDs must be compact and disjoint from old IDs')
    counts = torch.bincount(new_y, minlength=old + len(ids))[old:]
    if not bool((counts == anchor.protocol_metadata['old_support_k']).all()):
        raise ValueError('new support K must match old support K')
    cfg = RegistrationConfig(seed=seed, old_adaptation_epochs=0,
        mopc_epochs=20 if epochs is None else epochs,
        orthogonal_epochs=50 if epochs is None else epochs)
    device = new_x.device
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == 'cuda' else []
    with torch.random.fork_rng(devices=devices):
        torch.random.default_generator.manual_seed(int(seed))
        if device.type == 'cuda':
            with torch.cuda.device(device):
                torch.cuda.manual_seed(int(seed))
        state = register_new_classes(anchor, new_x, new_y, ids, config=cfg)
    state.protocol_metadata.update(claim_boundary='DA_backbone_with_CIL_core_transplant_not_native_source_architecture',
        source_pseudo_targets_preserved=False, old_bias_retained=state.method != 'orthogonal',
        C_entry_decision='cosine_weight_normalization_discards_B_affine_bias_and_scale' if state.method=='orthogonal'
                         else 'B_affine_old_logits_before_incremental_training')
    return state
