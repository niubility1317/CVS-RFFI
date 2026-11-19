"""Matched supervision on a full 11,000-point SatIQ waveform."""

import torch
from torch import nn
from torch.nn import functional as F

from .model import build_model, capture
from .wide_model import FEATURES, SAMPLES_PER_SYMBOL, RawSymbolEncoder


HEADER_SYMBOLS = 11
RECONSTRUCTION_BINS = 32
RECONSTRUCTION_WIDTH = 2 * HEADER_SYMBOLS * RECONSTRUCTION_BINS


def packet_rms(iq):
    centered = iq - iq.mean(dim=(1, 2), keepdim=True)
    rms = centered.square().sum(dim=1, keepdim=True).mean(dim=-1, keepdim=True).clamp_min(1e-8).sqrt()
    return centered / rms


def reconstruction_target(iq):
    symbols = packet_rms(iq).reshape(len(iq), 2 * HEADER_SYMBOLS, SAMPLES_PER_SYMBOL)
    return F.adaptive_avg_pool1d(symbols, RECONSTRUCTION_BINS).flatten(1)


class PaperPAMF(nn.Module):
    def __init__(self, runtime, seed, classes, raw_time_bins, feature_batch_norm=False, full_reconstruction=False):
        super().__init__()
        self.native = build_model(runtime, seed, 1)
        self.native.core.id_backbone.cls_head.requires_grad_(False)
        self.original_rate = RawSymbolEncoder(symbol_count=HEADER_SYMBOLS, bounded=False, time_bins=raw_time_bins)
        self.common_norm = nn.BatchNorm1d(FEATURES, affine=False) if feature_batch_norm else nn.Identity()
        self.full_norm = nn.BatchNorm1d(FEATURES, affine=False) if feature_batch_norm else nn.Identity()
        self.head = nn.Linear(2 * FEATURES, classes, bias=False)
        nn.init.xavier_uniform_(self.head.weight)
        self.full_decoder = None
        if full_reconstruction:
            self.full_decoder = nn.Sequential(
                nn.Linear(FEATURES, 512), nn.SiLU(), nn.Linear(512, RECONSTRUCTION_WIDTH)
            )
        bins = torch.arange(129).float()
        cutoff = torch.cos(torch.pi * ((bins - 60) / 20).clamp(0, 1)) * .5 + .5
        self.register_buffer("cutoff", cutoff, persistent=False)

    def _encode(self, iq, capture_branches=False):
        # Keep both paths comparable in norm. The older residual sum became
        # dominated by the original-rate path at later checkpoints.
        raw = packet_rms(iq)
        common = raw[:, :, :8000]
        spectrum = torch.fft.rfft(common, dim=-1)[..., :129]
        coarse = torch.fft.irfft(spectrum * self.cutoff, n=256, dim=-1) * (256 / 8000)
        branches = None
        if capture_branches:
            _, native, branches = capture(self.native, coarse)
        else:
            native = self.native.features(coarse)
        native = F.normalize(self.common_norm(native), dim=1)
        feature = F.normalize(self.full_norm(self.original_rate(raw)), dim=1)
        return native, feature, branches

    def encode_paths(self, iq):
        native, feature, _ = self._encode(iq)
        return native, feature

    def embed(self, iq):
        return F.normalize(torch.cat(self.encode_paths(iq), dim=1), dim=1)

    def forward(self, iq, view_logits=False, reconstruct=False, capture_branches=False):
        common, full, branches = self._encode(iq, capture_branches)
        embedding = F.normalize(torch.cat((common, full), dim=1), dim=1)
        logits = 30.0 * F.linear(embedding, F.normalize(self.head.weight, dim=1))
        output = (embedding, logits)
        if view_logits:
            common_weight, full_weight = self.head.weight.chunk(2, dim=1)
            output += (30.0 * F.linear(common, F.normalize(common_weight, dim=1)),
                       30.0 * F.linear(full, F.normalize(full_weight, dim=1)))
        if reconstruct:
            output += (self.full_decoder(full),)
        if capture_branches:
            output += (torch.cat((branches, full.unsqueeze(0)), dim=0),)
        return output


def supervised_loss(logits, common, full, target, satellites, weights, class_log_prior=None):
    unadjusted = logits
    if class_log_prior is not None:
        logits = logits + class_log_prior
        common = common + class_log_prior
        full = full + class_log_prior
    ce = F.cross_entropy(logits, target)
    same_satellite = satellites[None, :] == satellites[target, None]
    within = F.cross_entropy(logits.masked_fill(~same_satellite, -torch.inf), target)
    common_ce = F.cross_entropy(common, target)
    full_ce = F.cross_entropy(full, target)
    fused_p = logits.detach().softmax(1)
    agreement = .5 * (F.kl_div(common.log_softmax(1), fused_p, reduction="batchmean")
                      + F.kl_div(full.log_softmax(1), fused_p, reduction="batchmean"))
    loss = (ce + weights["same_satellite_ce"] * within
            + weights["view_ce"] * .5 * (common_ce + full_ce) + weights["view_kl"] * agreement)
    stats = dict(ce=ce.detach(), within_satellite_ce=within.detach(), common_ce=common_ce.detach(),
                 full_ce=full_ce.detach(), view_kl=agreement.detach())
    if class_log_prior is not None:
        stats["prior_adjusted_ce"] = stats["ce"]
        stats["ce"] = F.cross_entropy(unadjusted.detach(), target)
    return loss, stats


def branch_prototype_loss(branches):
    """Classify centered branch features against the source batch's leave-one-out prototypes."""
    vectors = F.normalize(branches, dim=-1)
    centered = vectors - vectors.mean(dim=1, keepdim=True)

    def bounded_direction(value):
        # A floor prevents tiny centered residuals from dominating the supervised gradient.
        return value / (value.square().sum(dim=-1, keepdim=True) + .1 ** 2).sqrt()

    anchors = bounded_direction(centered)
    sums = centered.reshape(len(centered), 8, 4, -1).sum(dim=2)
    prototypes = bounded_direction(sums / 4)
    scores = anchors @ prototypes.transpose(1, 2)
    own_prototypes = bounded_direction((sums.repeat_interleave(4, dim=1) - centered) / 3)
    own_scores = (anchors * own_prototypes).sum(dim=-1, keepdim=True)
    labels = torch.arange(8, device=vectors.device).repeat_interleave(4).expand(len(vectors), -1)
    scores = scores.scatter(2, labels.unsqueeze(-1), own_scores)
    return F.cross_entropy((30 * scores).flatten(0, 1), labels.flatten())
