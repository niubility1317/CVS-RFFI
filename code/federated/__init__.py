"""Federated learning utilities for CVS-RFFI.

The package is intentionally separate from the centralized trainer so FedAvg
and FedProx remain optional and reversible.
"""

from .client_split import build_client_loaders, build_client_splits, infer_client_id
from .fed_aggregate import aggregate_state_dicts, resolve_exclude_keys
from .fedprox import compute_fedprox_loss
from .proto_evidence_bank import ProtoEvidence, ProtoEvidenceBank
from .reliability_fusion import (
    collaborative_probability_fusion,
    collaborative_reliability_from_probabilities,
    conservative_probability_fusion,
    harm_rescue_report,
)
from .rf_style_extractor import RFStyleExtractor
from .style_bank import FederatedStyleBank
from .style_packet import StyleDomainBatch, StylePacket
from .virtual_domain_sampler import VirtualDomainSampler, VirtualStyleView

__all__ = [
    "aggregate_state_dicts",
    "build_client_loaders",
    "build_client_splits",
    "collaborative_probability_fusion",
    "collaborative_reliability_from_probabilities",
    "conservative_probability_fusion",
    "compute_fedprox_loss",
    "FederatedStyleBank",
    "harm_rescue_report",
    "infer_client_id",
    "ProtoEvidence",
    "ProtoEvidenceBank",
    "resolve_exclude_keys",
    "RFStyleExtractor",
    "StyleDomainBatch",
    "StylePacket",
    "VirtualDomainSampler",
    "VirtualStyleView",
]
