"""Explicit model interfaces; no historical checkpoint is loaded by this module.

CSIL's ADS-B residual-image encoder and MoPC's original signal dataset do not
match 256-sample WiSig. Their IQ encoder is an explicitly declared CVS extension;
their native classifier and incremental mechanisms remain distinct.
"""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

METHODS = ("protonet", "feature_separation", "dadda", "mrior", "twostage", "csil", "mopc_hr", "orthogonal", "radionet_ada")


class ProtoNet(nn.Module):
    """Author-derived CDA Conv2D encoder; episodic loss trains the encoder.

    Fixed source prototypes are exported after training for Phase1/DG inference.
    They are not a replacement for episodic learning.
    """
    def __init__(self, num_classes: int, embedding_dim: int = 256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 256, (1, 7)), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 80, (2, 7)), nn.BatchNorm2d(80), nn.ReLU(),
            nn.Flatten(), nn.Linear(80 * 244, 256), nn.BatchNorm1d(256), nn.ReLU(),
            nn.Linear(256, embedding_dim), nn.BatchNorm1d(embedding_dim), nn.ReLU())
        self.register_buffer("source_prototypes", torch.zeros(num_classes, embedding_dim))
        self.register_buffer("prototypes_ready", torch.tensor(False))

    def features(self, x):
        return self.encoder(x.unsqueeze(1))

    def forward(self, x):
        z = self.features(x)
        return {"features": z, "tx_logits": -torch.cdist(z, self.source_prototypes)}


class IncrementalIQNet(nn.Module):
    """CVS IQ-input extension with native CSIL/MoPC/orthogonal classifiers."""
    def __init__(self, method: str, num_classes: int, embedding_dim: int = 128, **kwargs):
        super().__init__()
        self.method = method
        self.encoder = PrecisionSixBlockEncoder(embedding_dim=embedding_dim)
        if method == "csil":
            self.fc_bf_fp = nn.Linear(embedding_dim, num_classes)
            self.fingerprints = nn.Parameter(torch.randn(num_classes, num_classes) * .01)
        elif method == "mopc_hr":
            self.fc = nn.Linear(embedding_dim, num_classes)
        else:
            from paper_reproduction.orthogonal_incremental_sei.pseudo_targets import make_simplex_pseudo_targets
            count = int(kwargs.get("num_pseudo_targets", max(num_classes + 20, 32)))
            targets = make_simplex_pseudo_targets(num_targets=count, feature_dim=embedding_dim)
            self.register_buffer("pseudo_targets", targets)
            self.register_buffer("classifier_weight", targets[:num_classes].clone())

    def features(self, x):
        z = self.encoder(x)
        return self.fc_bf_fp(z) if self.method == "csil" else z

    def forward(self, x):
        z = self.features(x)
        if self.method == "csil":
            from paper_reproduction.cvs_aligned.adv3b02_official_repo_ci import zero_bias_logits
            logits = zero_bias_logits(z, self.fingerprints)
        elif self.method == "mopc_hr":
            logits = self.fc(z)
        else:
            logits = F.normalize(z, dim=1) @ F.normalize(self.classifier_weight, dim=1).T
        return {"features": z, "tx_logits": logits}


from paper_reproduction.orthogonal_incremental_sei.model import SixBlockConv1DEncoder


class PrecisionSixBlockEncoder(SixBlockConv1DEncoder):
    """Preserve source dtype, including FP64 official Fisher computation.

    The inherited module shapes and FP32 forward operations are unchanged.
    """
    def forward(self, x):
        if x.ndim != 3 or x.shape[1] != self.input_channels or x.shape[-1] < 64:
            raise ValueError("Expected IQ [batch,2,length>=64]")
        dtype = next(self.parameters()).dtype
        return self.projection(self.pool(self.features(x.to(dtype=dtype))).squeeze(-1))


def factory(method: str, num_classes: int, num_receivers: int, **kwargs) -> nn.Module:
    if method == "radionet_ada":
        from comparison_suite.adaptation import RadioNetADANet
        return RadioNetADANet(num_classes=num_classes)
    if method == "protonet":
        return ProtoNet(num_classes, int(kwargs.get("embedding_dim", 256)))
    if method == "feature_separation":
        from paper_reproduction.feature_separation_crossrx.model import FeatureSeparationNet
        return FeatureSeparationNet(num_tx=num_classes, num_rx=num_receivers)
    if method == "dadda":
        from paper_reproduction.DADDA.model import DADDANet
        return DADDANet(num_classes=num_classes, model_variant="conv2d_paper", base_channels=int(kwargs.get("base_channels", 64)))
    if method == "mrior":
        from paper_reproduction.mitigating_receiver_impact_da.model import ReceiverImpactGADNet
        return ReceiverImpactGADNet(num_tx=num_classes, model_profile="standard_resnet18")
    if method == "twostage":
        from paper_reproduction.receiver_agnostic_twostage_uda.model import ReceiverAgnosticUDANet
        return ReceiverAgnosticUDANet(num_tx=num_classes)
    if method in ("csil", "mopc_hr", "orthogonal"):
        return IncrementalIQNet(method, num_classes, **kwargs)
    raise ValueError(f"Unknown source method {method!r}")


def embedding(model: nn.Module, method: str, x: torch.Tensor) -> torch.Tensor:
    out = model(x)
    key = "tx_features" if method == "feature_separation" else "local_features" if method == "dadda" else "features"
    return out[key]


def classifier_logits(model: nn.Module, method: str, x: torch.Tensor) -> torch.Tensor:
    out = model(x)
    return out["logits" if method == "dadda" else "tx_logits"]


def architecture_metadata(method: str) -> dict:
    return {"method": method, "implementation": "comparison_suite.models.factory",
            "input": "equalized1_center256_unit_rms",
            "paper_original_dataset_reproduction": False,
            "claim": "matched WiSig/LEO extension",
            "encoder_extension": method in ("csil", "mopc_hr"),
            "encoder_gap": "native ADS-B/image or original signal encoder replaced by declared six-block IQ encoder" if method in ("csil", "mopc_hr") else None}
