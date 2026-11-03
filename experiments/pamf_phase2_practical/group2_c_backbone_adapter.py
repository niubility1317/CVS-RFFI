"""Interfaces for transplanting B objectives onto a C method's source backbone.

This is project-specific integration, not a claim of author architecture
parity. It does not load datasets, fit support, register classes or read query.
"""
from __future__ import annotations

import copy

import torch
from torch import nn

from .vendor_native.baselines.common.grl import gradient_reverse
from .vendor_native.paper_reproduction.mitigating_receiver_impact_da.model import ThreeLayerFCNet
from .vendor_native.paper_reproduction.receiver_agnostic_twostage_uda.model import DenseReLUHead


# Fixed from each source model's architecture, without consulting target data.
# Multiscale outputs are spatial/temporal means of two distinct native convs.
C_SOURCE_SPECS = {
    'mopc_hr': {'embedding_dim': 256, 'tap_names': ('layers.1.1.conv2', 'layers.4.1.conv2')},
    'orthogonal': {'embedding_dim': 128, 'tap_names': ('encoder.features.8', 'encoder.features.20')},
    'lora_triplet_knn': {'embedding_dim': 512,
                         'tap_names': ('network.resblock2.conv2.conv', 'network.resblock4.conv2.conv')},
}


class CrossAdaptationAdapter(nn.Module):
    """Keep the C source encoder while exposing features consumed by B cores.

    For LoRa, a separately source-fitted old-class head is mandatory: a random
    target-time head is neither the C source model nor a valid B initialization.
    Distinct explicitly named convolution taps are required for the multiscale
    objectives; the caller must preregister their actual names for each model.
    """

    _C = {'mopc_hr', 'orthogonal', 'lora_triplet_knn'}
    _B = {'protonet_cda', 'mrior_sda', 'dadda_sda', 'twostage_sda', 'radionet_ada_knn'}

    def __init__(self, backbone, c_method, b_method, embedding_dim, *, source_head=None,
                 source_head_metadata=None, tap_names=(), classes=6, init_seed=0):
        super().__init__()
        if c_method not in self._C or b_method not in self._B:
            raise ValueError('unbound C source or B objective')
        if embedding_dim <= 0 or classes <= 0:
            raise ValueError('invalid source embedding or class count')
        self.backbone = copy.deepcopy(backbone)
        self.c_method, self.b_method = c_method, b_method
        self.embedding_dim, self.classes = int(embedding_dim), int(classes)
        self.init_seed = int(init_seed)
        if c_method == 'orthogonal':
            weight = getattr(self.backbone, 'classifier_weight', None)
            if not isinstance(weight, torch.Tensor) or weight.shape != (classes, embedding_dim):
                raise ValueError('orthogonal C source head has an incompatible shape')
            # The native orthogonal classifier is normally a frozen buffer.
            # B must be able to update it while retaining its initial values;
            # the copied C source checkpoint remains unchanged.
            delattr(self.backbone, 'classifier_weight')
            self.backbone.register_parameter('classifier_weight', nn.Parameter(weight.detach().clone()))
        self.tap_names = tuple(tap_names)
        modules = dict(self.backbone.named_modules())
        if b_method in {'dadda_sda', 'twostage_sda'}:
            if (len(self.tap_names) != 2 or len(set(self.tap_names)) != 2 or
                    any(name not in modules or not isinstance(modules[name], (nn.Conv1d, nn.Conv2d))
                        for name in self.tap_names)):
                raise ValueError('two distinct source convolution tap names are required')
        elif self.tap_names:
            raise ValueError('tap names are only applicable to multiscale B objectives')
        self.source_head = None
        if c_method == 'lora_triplet_knn':
            meta = source_head_metadata or {}
            if (not isinstance(source_head, nn.Linear) or source_head.in_features != embedding_dim or
                    source_head.out_features != classes or meta.get('source_role') != 'L_s' or
                    meta.get('target_contact') is not False or meta.get('status') != 'SOURCE_FITTED'):
                raise ValueError('LoRa requires a matching source-fitted L_s classifier and provenance')
            self.source_head = copy.deepcopy(source_head)
        elif source_head is not None or source_head_metadata is not None:
            raise ValueError('native classifier methods cannot substitute an auxiliary source head')
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(self.init_seed)
            self.estimate_network = ThreeLayerFCNet(embedding_dim, 1) if b_method == 'mrior_sda' else None
            self.domain_classifier = DenseReLUHead(embedding_dim, 2, 128) if b_method == 'twostage_sda' else None

    def forward(self, x, *, return_activations=False, grl_lambda=1.):
        captured = {}
        hooks = []
        try:
            modules = dict(self.backbone.named_modules())
            for name in self.tap_names:
                def collect(_module, _inputs, output, key=name):
                    if key in captured or not isinstance(output, torch.Tensor) or output.ndim < 2:
                        raise ValueError('source tap did not produce one tensor per forward')
                    captured[key] = output.flatten(2).mean(-1) if output.ndim > 2 else output
                hooks.append(modules[name].register_forward_hook(collect))
            out = self.backbone(x)
        finally:
            for hook in hooks:
                hook.remove()
        if self.c_method == 'lora_triplet_knn':
            z, tx_logits = out, self.source_head(out)
        else:
            if not isinstance(out, dict) or 'features' not in out or 'tx_logits' not in out:
                raise ValueError('C source model must expose its native features and logits')
            z, tx_logits = out['features'], out['tx_logits']
        if (z.ndim != 2 or z.shape[1] != self.embedding_dim or
                tx_logits.shape != (len(z), self.classes)):
            raise ValueError('C source representation or old class head shape mismatch')
        result = {'features': z, 'tx_logits': tx_logits, 'logits': tx_logits}
        if self.tap_names:
            if set(captured) != set(self.tap_names):
                raise ValueError('not all declared source taps executed')
            activations = [captured[name] for name in self.tap_names]
            result.update(global_features=activations[0], local_features=activations[1])
            if return_activations:
                result['activations'] = activations
        if self.domain_classifier is not None:
            result['domain_logits'] = self.domain_classifier(gradient_reverse(z, float(grl_lambda)))
        return result

    def embedding(self, x):
        return self(x)['features']


def lora_source_centroid_head(features, labels, *, classes, provenance):
    """Build a disclosed centroid-linear B head from LoRa's frozen L_s gallery.

    Its softmax equals squared-Euclidean source-centroid softmax; this is an
    auxiliary B classifier, not LoRa's native kNN or an author-trained head.
    """
    if (not isinstance(features, torch.Tensor) or features.ndim != 2 or len(features) == 0 or
            not isinstance(labels, torch.Tensor) or labels.ndim != 1 or len(labels) != len(features) or
            labels.dtype != torch.long or not bool(torch.isfinite(features).all()) or
            provenance.get('source_only') is not True or classes <= 0):
        raise ValueError('invalid source-only LoRa gallery or provenance')
    labels = labels.to(features.device)
    if tuple(torch.unique(labels, sorted=True).tolist()) != tuple(range(classes)):
        raise ValueError('source gallery must cover exactly the old classes')
    centroids = torch.stack([features[labels == c].mean(0) for c in range(classes)])
    with torch.random.fork_rng(devices=[]):
        head = nn.Linear(features.shape[1], classes).to(features)
    with torch.no_grad():
        head.weight.copy_(2 * centroids)
        head.bias.copy_(-centroids.square().sum(1))
    return head, {'source_role': 'L_s', 'target_contact': False, 'status': 'SOURCE_FITTED',
                  'construction': 'analytic_squared_euclidean_source_centroids',
                  'source_samples': len(features), 'optimizer_steps': 0,
                  'claim': 'project_auxiliary_not_native_LoRa_kNN'}
