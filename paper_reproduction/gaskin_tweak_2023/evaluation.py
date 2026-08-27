from __future__ import annotations

import torch

from .calibration import CalibrationState, closed_set_predict, open_set_admit


def _binary_auroc(known_scores: torch.Tensor, unknown_scores: torch.Tensor) -> float:
    """Mann-Whitney AUROC with higher score meaning a known device."""
    comparisons = known_scores[:, None] - unknown_scores[None, :]
    return float((comparisons.gt(0).float() + 0.5 * comparisons.eq(0).float()).mean())


@torch.no_grad()
def open_set_trial_metrics(
    *,
    known_points: torch.Tensor,
    known_labels: torch.Tensor,
    unknown_points: torch.Tensor,
    state: CalibrationState,
) -> dict[str, float]:
    """Compute the paper's open-set score and Algorithm-3 operating metrics."""
    if known_points.ndim != 2 or unknown_points.ndim != 2 or known_labels.shape != (known_points.shape[0],):
        raise ValueError("known points, unknown points and known labels must align")
    known_distances = torch.cdist(known_points, state.centroids, p=2).amin(dim=1)
    unknown_distances = torch.cdist(unknown_points, state.centroids, p=2).amin(dim=1)
    known_admit = open_set_admit(known_points, state)
    unknown_admit = open_set_admit(unknown_points, state)
    known_prediction = closed_set_predict(known_points, state)
    accepted = known_admit
    accepted_accuracy = (
        float(known_prediction[accepted].eq(known_labels[accepted]).float().mean()) if bool(accepted.any()) else 0.0
    )
    return {
        "auroc": _binary_auroc(-known_distances, -unknown_distances),
        "tpr": float(known_admit.float().mean()),
        "fpr": float(unknown_admit.float().mean()),
        "accepted_known_accuracy": accepted_accuracy,
    }
