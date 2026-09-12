from __future__ import annotations

import torch
from torch import nn

from cvsrffi.phase1_fcr_canonicalizer import ConservativeCanonicalizer
from cvsrffi.phase1_fcr_decoder import PhysicsOrderedDecoder
from cvsrffi.phase1_fcr_factors import ContentFactorEncoder, excitation_features
from cvsrffi.phase1_fcr_fingerprint import (
    ExcitationConditionedFingerprintOperator,
    FingerprintFactorEncoder,
)
from cvsrffi.phase1_fcr_nuisance import StructuredNuisanceEncoder
from cvsrffi.phase1_fcr_types import (
    FCRAggregateOutput,
    FCRConfig,
    FCRFactorOutput,
)

class ADV3B02FactorizedCrossReconstruction(nn.Module):
    """Compose the committed FCR factors without adding a waveform bypass."""

    def __init__(self, config: FCRConfig) -> None:
        super().__init__()
        self.config = config
        self.canonicalizer = ConservativeCanonicalizer(config)
        self.content = ContentFactorEncoder(config)
        self.fingerprint = FingerprintFactorEncoder(config)
        self.fingerprint_operator = ExcitationConditionedFingerprintOperator(config)
        self.nuisance = StructuredNuisanceEncoder(config)
        self.decoder = PhysicsOrderedDecoder(config)

    def identity_only(self, x: torch.Tensor, id_feature_raw: torch.Tensor) -> torch.Tensor:
        """Compute formal R3 identity without nuisance, response, or decoding."""
        with torch.autocast(device_type=x.device.type, enabled=False):
            canonical = self.canonicalizer(x.float())
            content = self.content(canonical.canonical_iq)
            fingerprint = self.fingerprint(
                id_feature_raw.float(),
                canonical.canonical_iq,
                canonical.residual_iq,
                excitation_features(content.s_hat.detach()),
            )
            return fingerprint.z_f_id

    def forward(
        self,
        x: torch.Tensor,
        id_feature_raw: torch.Tensor,
        *,
        pair_context=None,
    ) -> FCRAggregateOutput:
        # CUDA ComplexHalf has incomplete operator coverage (for example,
        # gather). Keep the compact FCR physics branch in FP32/complex64 while
        # allowing the surrounding ADV3B02 backbone to remain under AMP.
        with torch.autocast(device_type=x.device.type, enabled=False):
            return self._forward_fp32(
                x.float(), id_feature_raw.float(), pair_context=pair_context
            )

    def _forward_fp32(
        self,
        x: torch.Tensor,
        id_feature_raw: torch.Tensor,
        *,
        pair_context=None,
    ) -> FCRAggregateOutput:
        # Pair context is intentionally optional. Task10 may consume it for
        # paired losses; single-view factorization never requires a companion.
        del pair_context
        canonical = self.canonicalizer(x)
        content = self.content(canonical.canonical_iq)
        fingerprint_excitation = content.s_hat.detach()
        fingerprint = self.fingerprint(
            id_feature_raw,
            canonical.canonical_iq,
            canonical.residual_iq,
            excitation_features(fingerprint_excitation),
        )
        response = self.fingerprint_operator(fingerprint_excitation, fingerprint)
        nuisance = self.nuisance(x, canonical.eta_hat)
        decoded = self.decoder(content.s_hat, response.delta_f, nuisance)
        z_n_parts = {
            "channel": nuisance.z_ch,
            "receiver": nuisance.z_rx,
            "sync": nuisance.z_sync,
            "gain": nuisance.z_gain,
            "eta_pred": nuisance.eta_pred,
        }
        factors = FCRFactorOutput(
            z_s=content.z_s,
            z_f_id=fingerprint.z_f_id,
            z_tx_state=fingerprint.z_tx_state,
            z_n_parts=z_n_parts,
            s_hat=content.s_hat,
            content_confidence=content.content_confidence,
            response_coef=response.response_coef,
            response_quality=response.response_quality,
        )
        nuisance_vector = torch.cat(
            (nuisance.z_ch, nuisance.z_rx, nuisance.z_sync, nuisance.z_gain), dim=1
        )
        quality = {
            **{
                f"canonical_{name}": value
                for name, value in canonical.quality.items()
            },
            "content_confidence": content.content_confidence,
            **{
                f"fingerprint_{name}": value
                for name, value in response.response_quality.items()
            },
            "nuisance_norm": nuisance_vector.norm(dim=1),
            "decode_variance_mean": decoded.log_variance.exp().mean(dim=1),
        }
        return FCRAggregateOutput(
            canonical=canonical,
            content=content,
            fingerprint=fingerprint,
            response=response,
            nuisance=nuisance,
            factors=factors,
            decode=decoded,
            quality=quality,
        )
