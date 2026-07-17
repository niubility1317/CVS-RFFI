"""Bounded-memory two-pass Phase1 domain-class geometry aggregation.

The first pass accumulates only per-cell sums and counts from normalized
``z_id`` batches.  After centroids are frozen, the second pass builds a fixed
cosine-distance histogram for every authorized domain-class cell.  No
sample-level feature is retained and this module has no filesystem or
serialization API.

The returned object is an offline, full-precision in-memory handoff.  Its cell
counts are useful for offline aggregation checks but must not be persisted by
a deployment codec.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch


_INTEGER_DTYPES = {
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
    torch.uint8,
}


class Phase1GeometryStreamingError(ValueError):
    """Raised when the two-pass aggregate-only contract is violated."""


@dataclass(frozen=True)
class Phase1GeometryInMemory:
    """Full-precision aggregate-only handoff for an offline deployment codec."""

    domain_class_centroids: torch.Tensor
    radius_p90_cosine_distance: torch.Tensor
    active_cell_mask: torch.Tensor
    domain_class_counts: torch.Tensor
    radius_histogram_bins: int
    radius_resolution_upper_bound: float

    def __post_init__(self) -> None:
        centroids = self.domain_class_centroids
        radius = self.radius_p90_cosine_distance
        active = self.active_cell_mask
        counts = self.domain_class_counts
        if centroids.dtype != torch.float32 or centroids.ndim != 3:
            raise Phase1GeometryStreamingError(
                "domain_class_centroids must be float32[D,C,P]"
            )
        if radius.dtype != torch.float32 or radius.shape != centroids.shape[:2]:
            raise Phase1GeometryStreamingError(
                "radius_p90_cosine_distance must be float32[D,C]"
            )
        if active.dtype != torch.bool or active.shape != centroids.shape[:2]:
            raise Phase1GeometryStreamingError("active_cell_mask must be bool[D,C]")
        if counts.dtype != torch.int64 or counts.shape != centroids.shape[:2]:
            raise Phase1GeometryStreamingError("domain_class_counts must be int64[D,C]")
        if not bool(torch.isfinite(centroids).all()) or not bool(torch.isfinite(radius).all()):
            raise Phase1GeometryStreamingError("final geometry must be finite")
        if bool(torch.any(radius < 0.0)) or bool(torch.any(radius > 2.0)):
            raise Phase1GeometryStreamingError("cosine-distance radius must be in [0,2]")
        if bool(torch.any(counts[active] <= 0)) or bool(torch.any(counts[~active] != 0)):
            raise Phase1GeometryStreamingError("final cell counts do not match active mask")
        if bool(torch.any(centroids[~active] != 0.0)) or bool(torch.any(radius[~active] != 0.0)):
            raise Phase1GeometryStreamingError("inactive cells must remain zero")
        if int(self.radius_histogram_bins) < 16:
            raise Phase1GeometryStreamingError("radius_histogram_bins must be >=16")
        expected_resolution = 2.0 / float(self.radius_histogram_bins)
        if not math.isclose(
            float(self.radius_resolution_upper_bound),
            expected_resolution,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            raise Phase1GeometryStreamingError("radius resolution metadata mismatch")


class Phase1GeometryStreaming:
    """Strict two-pass accumulator with fixed memory independent of sample count."""

    def __init__(
        self,
        *,
        num_domains: int,
        num_classes: int,
        feature_dim: int,
        required_cell_mask: torch.Tensor | None = None,
        min_samples_per_cell: int = 2,
        radius_histogram_bins: int = 4096,
        unit_norm_tolerance: float = 1.0e-4,
        second_pass_sum_tolerance: float = 1.0e-8,
    ) -> None:
        if int(num_domains) <= 0 or int(num_classes) <= 0 or int(feature_dim) <= 0:
            raise Phase1GeometryStreamingError(
                "num_domains, num_classes, and feature_dim must be positive"
            )
        if int(min_samples_per_cell) < 2:
            raise Phase1GeometryStreamingError("min_samples_per_cell must be >=2")
        if int(radius_histogram_bins) < 16:
            raise Phase1GeometryStreamingError("radius_histogram_bins must be >=16")
        if not 0.0 < float(unit_norm_tolerance) <= 1.0e-2:
            raise Phase1GeometryStreamingError("unit_norm_tolerance must be in (0,1e-2]")
        if not 0.0 < float(second_pass_sum_tolerance) <= 1.0e-4:
            raise Phase1GeometryStreamingError(
                "second_pass_sum_tolerance must be in (0,1e-4]"
            )

        self.num_domains = int(num_domains)
        self.num_classes = int(num_classes)
        self.feature_dim = int(feature_dim)
        self.min_samples_per_cell = int(min_samples_per_cell)
        self.radius_histogram_bins = int(radius_histogram_bins)
        self.unit_norm_tolerance = float(unit_norm_tolerance)
        self.second_pass_sum_tolerance = float(second_pass_sum_tolerance)

        shape = (self.num_domains, self.num_classes)
        if required_cell_mask is None:
            active = torch.ones(shape, dtype=torch.bool)
        else:
            if not torch.is_tensor(required_cell_mask):
                raise Phase1GeometryStreamingError("required_cell_mask must be a tensor")
            if required_cell_mask.dtype != torch.bool or tuple(required_cell_mask.shape) != shape:
                raise Phase1GeometryStreamingError(
                    "required_cell_mask must be bool[num_domains,num_classes]"
                )
            active = required_cell_mask.detach().cpu().clone()
        if not bool(active.any()):
            raise Phase1GeometryStreamingError("required_cell_mask must authorize at least one cell")

        self._active = active
        self._pass1_sums = torch.zeros(
            self.num_domains, self.num_classes, self.feature_dim, dtype=torch.float64
        )
        self._pass1_counts = torch.zeros(shape, dtype=torch.int64)
        self._centroids: torch.Tensor | None = None
        self._radius_histogram: torch.Tensor | None = None
        self._pass2_sums: torch.Tensor | None = None
        self._pass2_counts: torch.Tensor | None = None
        self._state = "pass1"

    @property
    def state(self) -> str:
        return self._state

    @property
    def bounded_accumulator_bytes(self) -> int:
        """Return fixed accumulator bytes; the value never depends on sample count."""

        tensors = [self._pass1_sums, self._pass1_counts, self._active]
        for value in (
            self._centroids,
            self._radius_histogram,
            self._pass2_sums,
            self._pass2_counts,
        ):
            if value is not None:
                tensors.append(value)
        return int(sum(item.numel() * item.element_size() for item in tensors))

    def _validate_batch(
        self,
        normalized_z_id: torch.Tensor,
        class_index: torch.Tensor,
        domain_index: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not torch.is_tensor(normalized_z_id):
            raise Phase1GeometryStreamingError("normalized_z_id must be a tensor")
        if normalized_z_id.ndim != 2 or normalized_z_id.shape[1] != self.feature_dim:
            raise Phase1GeometryStreamingError(
                "normalized_z_id must have shape [N,feature_dim]"
            )
        if normalized_z_id.shape[0] <= 0:
            raise Phase1GeometryStreamingError("empty batches are not allowed")
        if not normalized_z_id.dtype.is_floating_point:
            raise Phase1GeometryStreamingError("normalized_z_id must use a floating dtype")
        if not torch.is_tensor(class_index) or not torch.is_tensor(domain_index):
            raise Phase1GeometryStreamingError("class_index and domain_index must be tensors")
        if class_index.dtype not in _INTEGER_DTYPES or domain_index.dtype not in _INTEGER_DTYPES:
            raise Phase1GeometryStreamingError(
                "class_index and domain_index must use integer dtypes"
            )
        expected = (int(normalized_z_id.shape[0]),)
        if tuple(class_index.shape) != expected or tuple(domain_index.shape) != expected:
            raise Phase1GeometryStreamingError(
                "class_index and domain_index must have shape [N]"
            )

        features = normalized_z_id.detach().to(device="cpu", dtype=torch.float64)
        classes = class_index.detach().to(device="cpu", dtype=torch.int64)
        domains = domain_index.detach().to(device="cpu", dtype=torch.int64)
        if not bool(torch.isfinite(features).all()):
            raise Phase1GeometryStreamingError("normalized_z_id must be finite")
        norms = torch.linalg.vector_norm(features, dim=1)
        if not bool(
            torch.all(
                torch.abs(norms - 1.0) <= self.unit_norm_tolerance
            )
        ):
            raise Phase1GeometryStreamingError(
                "normalized_z_id rows must have unit L2 norm within tolerance"
            )
        if bool(torch.any(classes < 0)) or bool(torch.any(classes >= self.num_classes)):
            raise Phase1GeometryStreamingError("class_index is outside the opaque registry")
        if bool(torch.any(domains < 0)) or bool(torch.any(domains >= self.num_domains)):
            raise Phase1GeometryStreamingError("domain_index is outside the opaque registry")
        if bool(torch.any(~self._active[domains, classes])):
            raise Phase1GeometryStreamingError("batch contains a non-authorized domain-class cell")
        return features, classes, domains

    def update_first_pass(
        self,
        normalized_z_id: torch.Tensor,
        class_index: torch.Tensor,
        domain_index: torch.Tensor,
    ) -> None:
        """Accumulate only per-cell sums and counts for centroid construction."""

        if self._state != "pass1":
            raise Phase1GeometryStreamingError("first-pass updates are no longer allowed")
        features, classes, domains = self._validate_batch(
            normalized_z_id, class_index, domain_index
        )
        flat = domains * self.num_classes + classes
        for value in torch.unique(flat, sorted=True):
            mask = flat == value
            domain = int(value.item()) // self.num_classes
            cls = int(value.item()) % self.num_classes
            self._pass1_sums[domain, cls].add_(features[mask].sum(dim=0))
            self._pass1_counts[domain, cls] += int(mask.sum().item())

    def begin_second_pass(self) -> torch.Tensor:
        """Freeze centroids and initialize the bounded P90 accumulator."""

        if self._state != "pass1":
            raise Phase1GeometryStreamingError("second pass can only begin once")
        missing = self._active & (self._pass1_counts < self.min_samples_per_cell)
        if bool(missing.any()):
            cells = torch.nonzero(missing, as_tuple=False).tolist()
            raise Phase1GeometryStreamingError(
                f"required domain-class cells lack minimum coverage: {cells}"
            )
        if bool(torch.any(self._pass1_counts[~self._active] != 0)):
            raise Phase1GeometryStreamingError("inactive cells received first-pass samples")

        means = torch.zeros_like(self._pass1_sums)
        means[self._active] = self._pass1_sums[self._active] / self._pass1_counts[
            self._active
        ].to(torch.float64).unsqueeze(1)
        norms = torch.linalg.vector_norm(means, dim=2)
        if not bool(torch.isfinite(means[self._active]).all()) or bool(
            torch.any(norms[self._active] <= 1.0e-12)
        ):
            raise Phase1GeometryStreamingError(
                "a required cell has a non-finite or zero centroid"
            )
        means[self._active] = means[self._active] / norms[self._active].unsqueeze(1)
        self._centroids = means
        self._radius_histogram = torch.zeros(
            self.num_domains,
            self.num_classes,
            self.radius_histogram_bins,
            dtype=torch.int64,
        )
        self._pass2_sums = torch.zeros_like(self._pass1_sums)
        self._pass2_counts = torch.zeros_like(self._pass1_counts)
        self._state = "pass2"
        return self._centroids.to(torch.float32).clone()

    def update_second_pass(
        self,
        normalized_z_id: torch.Tensor,
        class_index: torch.Tensor,
        domain_index: torch.Tensor,
    ) -> None:
        """Accumulate bounded cosine-distance histograms against frozen centroids."""

        if self._state != "pass2":
            raise Phase1GeometryStreamingError("second-pass updates require frozen centroids")
        if (
            self._centroids is None
            or self._radius_histogram is None
            or self._pass2_sums is None
            or self._pass2_counts is None
        ):
            raise Phase1GeometryStreamingError("second-pass state is incomplete")
        features, classes, domains = self._validate_batch(
            normalized_z_id, class_index, domain_index
        )
        flat = domains * self.num_classes + classes
        for value in torch.unique(flat, sorted=True):
            mask = flat == value
            domain = int(value.item()) // self.num_classes
            cls = int(value.item()) % self.num_classes
            rows = features[mask]
            cosine = torch.sum(rows * self._centroids[domain, cls].unsqueeze(0), dim=1)
            distance = (1.0 - torch.clamp(cosine, -1.0, 1.0)).clamp(0.0, 2.0)
            bins = torch.floor(
                distance * (float(self.radius_histogram_bins) / 2.0)
            ).to(torch.int64)
            bins.clamp_(min=0, max=self.radius_histogram_bins - 1)
            self._radius_histogram[domain, cls].add_(
                torch.bincount(bins, minlength=self.radius_histogram_bins)
            )
            self._pass2_sums[domain, cls].add_(rows.sum(dim=0))
            self._pass2_counts[domain, cls] += int(mask.sum().item())

    def finalize(self) -> Phase1GeometryInMemory:
        """Validate both passes and return an aggregate-only in-memory object."""

        if self._state != "pass2":
            raise Phase1GeometryStreamingError("finalize requires a completed second pass")
        if (
            self._centroids is None
            or self._radius_histogram is None
            or self._pass2_sums is None
            or self._pass2_counts is None
        ):
            raise Phase1GeometryStreamingError("second-pass state is incomplete")
        if not torch.equal(self._pass2_counts, self._pass1_counts):
            raise Phase1GeometryStreamingError(
                "second-pass cell counts do not match first-pass coverage"
            )
        delta = torch.abs(self._pass2_sums - self._pass1_sums)
        if bool(torch.any(delta[self._active] > self.second_pass_sum_tolerance)):
            raise Phase1GeometryStreamingError(
                "second-pass aggregate sums do not match first-pass samples"
            )
        if not torch.equal(
            self._radius_histogram.sum(dim=2), self._pass1_counts
        ):
            raise Phase1GeometryStreamingError(
                "radius histogram counts do not match first-pass coverage"
            )

        radius = torch.zeros(
            self.num_domains, self.num_classes, dtype=torch.float32
        )
        bin_width = 2.0 / float(self.radius_histogram_bins)
        for domain, cls in torch.nonzero(self._active, as_tuple=False).tolist():
            count = int(self._pass1_counts[domain, cls].item())
            target = int(math.ceil(0.90 * count))
            cumulative = torch.cumsum(
                self._radius_histogram[domain, cls], dim=0
            )
            index = int(torch.searchsorted(cumulative, target, right=False).item())
            radius[domain, cls] = min(2.0, float(index + 1) * bin_width)

        result = Phase1GeometryInMemory(
            domain_class_centroids=self._centroids.to(torch.float32).clone(),
            radius_p90_cosine_distance=radius,
            active_cell_mask=self._active.clone(),
            domain_class_counts=self._pass1_counts.clone(),
            radius_histogram_bins=self.radius_histogram_bins,
            radius_resolution_upper_bound=bin_width,
        )
        self._state = "finalized"
        return result


__all__ = [
    "Phase1GeometryInMemory",
    "Phase1GeometryStreaming",
    "Phase1GeometryStreamingError",
]

