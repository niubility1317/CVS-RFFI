"""PAMF-SUP width variants using only the existing five common-word paths."""

import torch
from torch import nn
from torch.nn import functional as F

from .model import build_model, capture
from .paper_pamf import packet_rms
from .wide_model import FEATURES


NATIVE_PATHS = ("time", "frequency", "nonlinear", "sequence", "reference")
EXTRA_WIDTH = {"native160": 0, "fused256": 96, "fused320": 160, "branches320": 160, "prefuse320": 160}


class PaperNativePAMF(nn.Module):
    waveform_samples = 8000

    def __init__(self, runtime, seed, classes, expansion, satiq_sinc_init=False, center_fusion=False):
        super().__init__()
        self.expansion = expansion
        self.feature_dimensions = FEATURES + EXTRA_WIDTH[expansion]
        self.native = build_model(runtime, seed, 1)
        self.native.core.id_backbone.cls_head.requires_grad_(False)
        if satiq_sinc_init:
            sinc = self.native.core.id_backbone.sinc
            # At 800 kS/s, use roughly two 79-tap resolution bins per band.
            # Overlapping 20 kHz bands cover the occupied baseband and its shoulders.
            low_hz = torch.arange(sinc.out_channels, dtype=sinc.low_origin_hz.dtype) * 8000
            sinc.low_origin_hz.copy_((low_hz[:, None] - sinc.min_low_hz).clamp_min(0))
            sinc.band_origin_hz.fill_(20_000 - sinc.min_band_hz)
        self.expander = None
        if expansion == "branches320":
            self.expander = nn.ModuleList(
                nn.Sequential(nn.Linear(FEATURES, 32), nn.SiLU(), nn.LayerNorm(32))
                for _ in NATIVE_PATHS
            )
        elif EXTRA_WIDTH[expansion]:
            # Reuse the existing fusion input to test the 160-coordinate bottleneck.
            input_width = FEATURES
            hidden_width = 256
            if expansion == "prefuse320":
                input_width += self.native.core.id_backbone.fuse[0].in_features
                hidden_width = 128
            self.expander = nn.Sequential(
                nn.Linear(input_width, hidden_width), nn.SiLU(), nn.Linear(hidden_width, EXTRA_WIDTH[expansion]),
                nn.LayerNorm(EXTRA_WIDTH[expansion]),
            )
        self.head = nn.Linear(self.feature_dimensions, classes, bias=False)
        nn.init.xavier_uniform_(self.head.weight)
        self.register_buffer("fusion_center", torch.zeros(self.feature_dimensions) if center_fusion else None)
        bins = torch.arange(129).float()
        cutoff = torch.cos(torch.pi * ((bins - 60) / 20).clamp(0, 1)) * .5 + .5
        self.register_buffer("cutoff", cutoff, persistent=False)

    def _center_fusion(self, feature):
        if self.fusion_center is None:
            return feature
        if self.training:
            self.fusion_center.lerp_(feature.detach().mean(dim=0), .05)
        return F.normalize(feature - self.fusion_center, dim=1, eps=1e-6)

    def _encode(self, iq, retain_branches):
        if iq.ndim != 3 or iq.shape[1] != 2 or iq.shape[-1] < self.waveform_samples:
            raise ValueError("Native PAMF requires two IQ channels and the complete common word")
        common = packet_rms(iq[:, :, :self.waveform_samples])
        spectrum = torch.fft.rfft(common, dim=-1)[..., :129]
        coarse = torch.fft.irfft(spectrum * self.cutoff, n=256, dim=-1) * (256 / self.waveform_samples)
        branches = None
        fusion_inputs = []
        handle = None
        if self.expansion == "prefuse320":
            handle = self.native.core.id_backbone.fuse.register_forward_pre_hook(
                lambda _module, args: fusion_inputs.append(args[0]))
        try:
            if retain_branches or self.expansion == "branches320":
                _, native, branches = capture(self.native, coarse)
            else:
                native = self.native.features(coarse)
        finally:
            if handle is not None:
                handle.remove()
        native = F.normalize(native, dim=1)
        if self.expander is None:
            return self._center_fusion(native), branches
        if self.expansion == "branches320":
            parts = [layer(F.normalize(x, dim=1)) for layer, x in zip(self.expander, branches, strict=True)]
            extra = torch.cat(parts, dim=1)
        elif self.expansion == "prefuse320":
            (fuse_input,) = fusion_inputs
            extra = self.expander(torch.cat((native, F.normalize(fuse_input, dim=1)), dim=1))
        else:
            extra = self.expander(native)
        feature = F.normalize(torch.cat((native, .5 * F.normalize(extra, dim=1)), dim=1), dim=1)
        return self._center_fusion(feature), branches

    def embed(self, iq):
        return self._encode(iq, False)[0]

    def encode_paths(self, iq):
        return (self.embed(iq),)

    def forward(self, iq, view_logits=False):
        feature, branches = self._encode(iq, view_logits)
        weight = F.normalize(self.head.weight, dim=1)
        logits = 30 * F.linear(feature, weight)
        if view_logits:
            branch_weight = F.normalize(self.head.weight[:, :FEATURES], dim=1)
            branch_logits = 30 * F.linear(F.normalize(branches, dim=-1), branch_weight)
            return feature, logits, branch_logits, branches
        return feature, logits


def native_classification_terms(logits, branch_logits, target, satellites):
    ce = F.cross_entropy(logits, target)
    same = satellites[None, :] == satellites[target, None]
    within = F.cross_entropy(logits.masked_fill(~same, -torch.inf), target)
    branch_ce = F.cross_entropy(branch_logits.flatten(0, 1), target.repeat(len(branch_logits)),
                                 reduction="none").reshape(len(branch_logits), -1).mean(dim=1)
    teacher = logits.detach().softmax(dim=1).expand_as(branch_logits)
    agreement = F.kl_div(branch_logits.log_softmax(dim=-1), teacher, reduction="sum")
    agreement = agreement / (len(branch_logits) * len(target))
    return ce, within, branch_ce, agreement


def native_supervised_loss(logits, branch_logits, target, satellites, weights):
    ce, within, branch_ce, agreement = native_classification_terms(logits, branch_logits, target, satellites)
    loss = (ce + weights["same_satellite_ce"] * within + weights["view_ce"] * branch_ce.mean()
            + weights["view_kl"] * agreement)
    stats = dict(ce=ce.detach(), within_satellite_ce=within.detach(),
                 native_branch_ce=branch_ce.mean().detach(), view_kl=agreement.detach())
    hard_fraction = weights.get("hard_rank_ce_fraction", 0.0)
    if hard_fraction:
        positive = logits.gather(1, target[:, None])
        negatives = logits.scatter(1, target[:, None], -torch.inf)
        hardest = negatives.topk(min(8, logits.shape[1] - 1), dim=1).values
        hard_ce = F.cross_entropy(torch.cat((positive, hardest), dim=1), torch.zeros_like(target))
        loss = loss + hard_fraction * (hard_ce - ce)
        stats["hard_rank_ce"] = hard_ce.detach()
    for name, value in zip(NATIVE_PATHS, branch_ce.detach(), strict=True):
        stats[f"{name}_branch_ce"] = value
    return loss, stats
