from __future__ import annotations

from collections.abc import Mapping, Sequence


def average_trials(trials: Sequence[Mapping[str, float]]) -> dict[str, float]:
    """Average the five random known/unknown trials reported by the paper."""
    if not trials:
        raise ValueError("at least one trial is required")
    required = ("auroc", "tpr", "fpr")
    if any(any(key not in trial for key in required) for trial in trials):
        raise ValueError("each trial must contain auroc, tpr, and fpr")
    return {key: sum(float(trial[key]) for trial in trials) / len(trials) for key in required}

