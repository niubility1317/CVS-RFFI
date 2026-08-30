"""Source-only diagnostics and same-row selection metrics for HCF-DG.

The functions in this module are deliberately stateless.  A caller supplies
frozen source embeddings/logits and explicit train/validation roles; no target
receiver, Phase2 capsule, query or truth-side object is accepted or needed.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, fields as dataclass_fields
from typing import Any
import json

import numpy as np


_EPS = 1e-12


def _to_numpy(value: Any, *, name: str) -> np.ndarray:
    """Convert NumPy-like and torch-like values without changing their data."""

    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def _numeric_matrix(value: Any, *, name: str) -> np.ndarray:
    array = _numeric_array(value, name=name)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a rank-2 matrix")
    return array


def _numeric_array(value: Any, *, name: str) -> np.ndarray:
    array = _to_numpy(value, name=name)
    if np.iscomplexobj(array):
        raise ValueError(f"{name} must contain real values")
    try:
        result = np.asarray(array, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain numeric values") from exc
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must contain only finite values")
    return result


def _vector(value: Any, *, name: str, expected_length: int | None = None) -> np.ndarray:
    array = _to_numpy(value, name=name)
    if array.ndim == 0:
        array = array.reshape(1)
    elif array.ndim == 2 and 1 in array.shape:
        array = array.reshape(-1)
    elif array.ndim != 1:
        raise ValueError(f"{name} must be a vector")
    if expected_length is not None and len(array) != expected_length:
        raise ValueError(f"{name} must have length {expected_length}")
    return np.asarray([_native_scalar(item) for item in array], dtype=object)


def _native_scalar(value: Any) -> Any:
    return value.item() if isinstance(value, np.generic) else value


def _ordered_unique(values: Iterable[Any]) -> list[Any]:
    unique: list[Any] = []
    seen: set[Any] = set()
    for value in values:
        value = _native_scalar(value)
        try:
            key = value
            already_seen = key in seen
        except TypeError:
            key = repr(value)
            already_seen = key in seen
        if not already_seen:
            seen.add(key)
            unique.append(value)
    try:
        return sorted(unique)
    except TypeError:
        return sorted(unique, key=lambda item: (type(item).__name__, repr(item)))


def _mask(value: Any | None, *, expected_length: int, name: str) -> np.ndarray | None:
    if value is None:
        return None
    array = _to_numpy(value, name=name)
    if array.ndim != 1 or len(array) != expected_length:
        raise ValueError(f"{name} must be a boolean vector of length {expected_length}")
    if array.dtype != np.bool_:
        if not np.all(np.isin(array, [0, 1, False, True])):
            raise ValueError(f"{name} must contain boolean values")
        array = array.astype(bool)
    return array


def _apply_mask(array: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    return array if mask is None else array[mask]


def _resolve_alias(
    value: Any | None,
    kwargs: dict[str, Any],
    *,
    name: str,
    aliases: tuple[str, ...],
) -> Any | None:
    for alias in aliases:
        if alias not in kwargs:
            continue
        if value is not None:
            raise TypeError(f"pass only one of {name} and {alias}")
        value = kwargs.pop(alias)
    return value


def _reject_kwargs(kwargs: dict[str, Any]) -> None:
    if kwargs:
        names = ", ".join(sorted(kwargs))
        raise TypeError(f"unexpected keyword argument(s): {names}")


def _prepare_split(
    train_features: Any | None,
    train_labels: tuple[Any | None, ...],
    validation_features: Any | None,
    validation_labels: tuple[Any | None, ...],
    *,
    train_mask: Any | None,
    validation_mask: Any | None,
    feature_name: str,
    label_names: tuple[str, ...],
) -> tuple[np.ndarray, tuple[np.ndarray, ...], np.ndarray, tuple[np.ndarray, ...]]:
    if train_features is None:
        raise TypeError(f"{feature_name} is required")
    train_x = _numeric_matrix(train_features, name=f"train_{feature_name}")
    train_y = tuple(
        _vector(label, name=f"train_{label_name}", expected_length=len(train_x))
        if label is not None
        else None
        for label, label_name in zip(train_labels, label_names)
    )
    if any(label is None for label in train_y):
        missing = label_names[[label is None for label in train_y].index(True)]
        raise TypeError(f"train_{missing} is required")

    if validation_features is None:
        train_selection = _mask(train_mask, expected_length=len(train_x), name="train_mask")
        validation_selection = _mask(
            validation_mask,
            expected_length=len(train_x),
            name="validation_mask",
        )
        if train_selection is None or validation_selection is None:
            raise ValueError(
                "validation features are required unless both train_mask and validation_mask are provided"
            )
        validation_x = train_x[validation_selection]
        validation_y = tuple(label[validation_selection] for label in train_y if label is not None)
        train_x = train_x[train_selection]
        train_y = tuple(label[train_selection] for label in train_y if label is not None)
    else:
        train_selection = _mask(train_mask, expected_length=len(train_x), name="train_mask")
        train_x = _apply_mask(train_x, train_selection)
        train_y = tuple(_apply_mask(label, train_selection) for label in train_y if label is not None)
        validation_x = _numeric_matrix(
            validation_features,
            name=f"validation_{feature_name}",
        )
        validation_y = tuple(
            _vector(label, name=f"validation_{label_name}", expected_length=len(validation_x))
            if label is not None
            else None
            for label, label_name in zip(validation_labels, label_names)
        )
        if any(label is None for label in validation_y):
            missing = label_names[[label is None for label in validation_y].index(True)]
            raise TypeError(f"validation_{missing} is required")
        validation_selection = _mask(
            validation_mask,
            expected_length=len(validation_x),
            name="validation_mask",
        )
        validation_x = _apply_mask(validation_x, validation_selection)
        validation_y = tuple(
            _apply_mask(label, validation_selection) for label in validation_y if label is not None
        )

    if len(train_x) == 0 or len(validation_x) == 0:
        raise ValueError("train and validation splits must both be non-empty")
    if train_x.shape[1] != validation_x.shape[1]:
        raise ValueError(f"train_{feature_name} and validation_{feature_name} must have the same width")
    return train_x, train_y, validation_x, validation_y


def _ridge_fit(features: np.ndarray, labels: np.ndarray, *, alpha: float) -> tuple[list[Any], np.ndarray]:
    if not np.isfinite(alpha) or alpha < 0:
        raise ValueError("alpha must be a finite non-negative number")
    classes = _ordered_unique(labels)
    if not classes:
        raise ValueError("a probe cannot be fitted on an empty label set")
    index = {label: position for position, label in enumerate(classes)}
    target = np.zeros((len(labels), len(classes)), dtype=np.float64)
    for row, label in enumerate(labels):
        target[row, index[label]] = 1.0

    design = np.concatenate(
        [features, np.ones((len(features), 1), dtype=np.float64)],
        axis=1,
    )
    regularizer = np.eye(design.shape[1], dtype=np.float64) * float(alpha)
    regularizer[-1, -1] = 0.0
    gram = design.T @ design + regularizer
    rhs = design.T @ target
    try:
        coefficients = np.linalg.solve(gram, rhs)
    except np.linalg.LinAlgError:
        coefficients = np.linalg.pinv(gram) @ rhs
    if not np.isfinite(coefficients).all():
        raise ValueError("regularized probe produced non-finite coefficients")
    return classes, coefficients


def _ridge_predict(features: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    design = np.concatenate(
        [features, np.ones((len(features), 1), dtype=np.float64)],
        axis=1,
    )
    scores = design @ coefficients
    return np.argmax(scores, axis=1)


class ProbeMetric(float):
    """A scalar macro accuracy with mapping-like access to per-label values."""

    def __new__(
        cls,
        macro_accuracy: float,
        *,
        per_tx: Mapping[Any, float],
        metric_name: str,
    ) -> "ProbeMetric":
        result = float.__new__(cls, float(macro_accuracy))
        result.per_tx = {key: float(value) for key, value in per_tx.items()}
        result.metric_name = str(metric_name)
        return result

    @property
    def macro_accuracy(self) -> float:
        return float(self)

    @property
    def accuracy(self) -> float:
        return float(self)

    @property
    def per_label(self) -> dict[Any, float]:
        return dict(self.per_tx)

    @property
    def per_tx_leakage(self) -> dict[Any, float]:
        return dict(self.per_tx)

    def to_dict(self) -> dict[str, Any]:
        return {"macro_accuracy": float(self), "per_tx": dict(self.per_tx)}

    def __getitem__(self, key: str) -> Any:
        values = self.to_dict()
        if key in {"per_label", "per_tx_leakage"}:
            return values["per_tx"]
        if key in {"accuracy", self.metric_name}:
            return values["macro_accuracy"]
        return values[key]

    def keys(self):
        return self.to_dict().keys()

    def items(self):
        return self.to_dict().items()

    def get(self, key: str, default: Any = None) -> Any:
        return self.to_dict().get(key, default)


def conditional_receiver_leakage(
    train_embeddings: Any | None = None,
    train_tx_labels: Any | None = None,
    train_receiver_labels: Any | None = None,
    validation_embeddings: Any | None = None,
    validation_tx_labels: Any | None = None,
    validation_receiver_labels: Any | None = None,
    *,
    alpha: float = 1.0,
    train_mask: Any | None = None,
    validation_mask: Any | None = None,
    **kwargs: Any,
) -> ProbeMetric:
    """Measure ``A_RX|TX`` with one ridge receiver probe per TX.

    The probes are fitted on source train embeddings only.  Validation rows
    are never used to fit a probe, and the returned macro value is the mean of
    per-TX validation accuracies rather than a sample-count-weighted score.
    When one combined source pool is supplied, both explicit masks are
    required to make the train/validation boundary visible.
    """

    train_embeddings = _resolve_alias(
        train_embeddings,
        kwargs,
        name="train_embeddings",
        aliases=("train_features", "source_train_embeddings", "embeddings"),
    )
    train_tx_labels = _resolve_alias(
        train_tx_labels,
        kwargs,
        name="train_tx_labels",
        aliases=("train_tx", "tx_train", "source_train_tx_labels", "tx_labels"),
    )
    train_receiver_labels = _resolve_alias(
        train_receiver_labels,
        kwargs,
        name="train_receiver_labels",
        aliases=("train_receiver", "receiver_train", "receiver_labels"),
    )
    validation_embeddings = _resolve_alias(
        validation_embeddings,
        kwargs,
        name="validation_embeddings",
        aliases=("val_embeddings", "validation_features", "val_features"),
    )
    validation_tx_labels = _resolve_alias(
        validation_tx_labels,
        kwargs,
        name="validation_tx_labels",
        aliases=("val_tx", "tx_validation", "tx_val", "validation_tx"),
    )
    validation_receiver_labels = _resolve_alias(
        validation_receiver_labels,
        kwargs,
        name="validation_receiver_labels",
        aliases=("val_receiver", "receiver_validation", "receiver_val", "validation_receiver"),
    )
    ridge = kwargs.pop("ridge", None)
    if ridge is not None:
        if alpha != 1.0:
            raise TypeError("pass only one of alpha and ridge")
        alpha = ridge
    _reject_kwargs(kwargs)

    train_x, (train_tx, train_receiver), validation_x, (validation_tx, validation_receiver) = _prepare_split(
        train_embeddings,
        (train_tx_labels, train_receiver_labels),
        validation_embeddings,
        (validation_tx_labels, validation_receiver_labels),
        train_mask=train_mask,
        validation_mask=validation_mask,
        feature_name="embeddings",
        label_names=("tx_labels", "receiver_labels"),
    )

    per_tx: dict[Any, float] = {}
    for tx in _ordered_unique(validation_tx):
        tx_train_mask = train_tx == tx
        tx_validation_mask = validation_tx == tx
        if not np.any(tx_train_mask):
            raise ValueError(f"TX {tx!r} has validation rows but no train rows")
        classes, coefficients = _ridge_fit(
            train_x[tx_train_mask],
            train_receiver[tx_train_mask],
            alpha=alpha,
        )
        predictions = _ridge_predict(validation_x[tx_validation_mask], coefficients)
        class_index = {label: index for index, label in enumerate(classes)}
        expected = np.asarray(
            [class_index.get(label, -1) for label in validation_receiver[tx_validation_mask]],
            dtype=np.int64,
        )
        per_tx[tx] = float(np.mean(predictions == expected))
    return ProbeMetric(
        float(np.mean(list(per_tx.values()))),
        per_tx=per_tx,
        metric_name="conditional_receiver_leakage",
    )


def environment_tx_leakage(
    train_environment: Any | None = None,
    train_tx_labels: Any | None = None,
    validation_environment: Any | None = None,
    validation_tx_labels: Any | None = None,
    *,
    alpha: float = 1.0,
    train_mask: Any | None = None,
    validation_mask: Any | None = None,
    **kwargs: Any,
) -> ProbeMetric:
    """Measure ``A_TX(z_env)`` with a source-only ridge TX probe."""

    train_environment = _resolve_alias(
        train_environment,
        kwargs,
        name="train_environment",
        aliases=("train_embeddings", "train_features", "z_env_train", "environment_train", "embeddings"),
    )
    train_tx_labels = _resolve_alias(
        train_tx_labels,
        kwargs,
        name="train_tx_labels",
        aliases=("train_tx", "tx_train", "tx_labels"),
    )
    validation_environment = _resolve_alias(
        validation_environment,
        kwargs,
        name="validation_environment",
        aliases=("validation_embeddings", "val_environment", "validation_features", "val_embeddings"),
    )
    validation_tx_labels = _resolve_alias(
        validation_tx_labels,
        kwargs,
        name="validation_tx_labels",
        aliases=("val_tx", "tx_validation", "tx_val", "validation_tx"),
    )
    ridge = kwargs.pop("ridge", None)
    if ridge is not None:
        if alpha != 1.0:
            raise TypeError("pass only one of alpha and ridge")
        alpha = ridge
    _reject_kwargs(kwargs)

    train_x, (train_tx,), validation_x, (validation_tx,) = _prepare_split(
        train_environment,
        (train_tx_labels,),
        validation_environment,
        (validation_tx_labels,),
        train_mask=train_mask,
        validation_mask=validation_mask,
        feature_name="environment",
        label_names=("tx_labels",),
    )
    for tx in _ordered_unique(validation_tx):
        if not np.any(train_tx == tx):
            raise ValueError(f"TX {tx!r} has validation rows but no train rows")
    classes, coefficients = _ridge_fit(train_x, train_tx, alpha=alpha)
    predictions = _ridge_predict(validation_x, coefficients)
    class_index = {label: index for index, label in enumerate(classes)}
    expected = np.asarray([class_index.get(label, -1) for label in validation_tx], dtype=np.int64)
    correct = predictions == expected
    per_tx = {
        tx: float(np.mean(correct[validation_tx == tx]))
        for tx in _ordered_unique(validation_tx)
    }
    return ProbeMetric(
        float(np.mean(list(per_tx.values()))),
        per_tx=per_tx,
        metric_name="environment_tx_leakage",
    )


def _class_indices(labels: np.ndarray, class_count: int) -> np.ndarray:
    values = _ordered_unique(labels)
    if all(isinstance(value, (int, np.integer)) for value in values) and all(
        0 <= int(value) < class_count for value in values
    ):
        return np.asarray([int(value) for value in labels], dtype=np.int64)
    if len(values) > class_count:
        raise ValueError("labels contain more classes than the supplied logits")
    index = {value: position for position, value in enumerate(values)}
    return np.asarray([index.get(value, -1) for value in labels], dtype=np.int64)


def _logit_accuracy(logits: Any, labels: Any, *, name: str) -> float:
    scores = _numeric_matrix(logits, name=name)
    target = _vector(labels, name="labels", expected_length=len(scores))
    indices = _class_indices(target, scores.shape[1])
    return float(np.mean(np.argmax(scores, axis=1) == indices))


def specific_gap(
    common_logits: Any | None = None,
    specific_logits: Any | None = None,
    labels: Any | None = None,
    **kwargs: Any,
) -> float:
    """Return ``Delta_spec = A(W_e) - A(W_0)``.

    The normal form accepts common logits, specific logits and TX labels.  Two
    scalar accuracies are also accepted for report aggregation.
    """

    common_logits = _resolve_alias(
        common_logits,
        kwargs,
        name="common_logits",
        aliases=("common", "public_logits", "w0_logits"),
    )
    specific_logits = _resolve_alias(
        specific_logits,
        kwargs,
        name="specific_logits",
        aliases=("specific", "we_logits", "domain_specific_logits"),
    )
    labels = _resolve_alias(labels, kwargs, name="labels", aliases=("tx_labels", "targets"))
    common_accuracy = kwargs.pop("common_accuracy", None)
    specific_accuracy = kwargs.pop("specific_accuracy", None)
    _reject_kwargs(kwargs)

    if common_accuracy is not None or specific_accuracy is not None:
        if common_accuracy is None or specific_accuracy is None:
            raise TypeError("common_accuracy and specific_accuracy must be provided together")
        return float(specific_accuracy) - float(common_accuracy)
    if common_logits is None or specific_logits is None:
        raise TypeError("common_logits and specific_logits are required")
    if labels is None and np.ndim(common_logits) == 0 and np.ndim(specific_logits) == 0:
        return float(specific_logits) - float(common_logits)
    if labels is None:
        raise TypeError("labels are required when common_logits and specific_logits are not scalars")
    common = _numeric_matrix(common_logits, name="common_logits")
    specific = _numeric_matrix(specific_logits, name="specific_logits")
    if common.shape != specific.shape:
        raise ValueError("common_logits and specific_logits must have matching shapes")
    return _logit_accuracy(specific, labels, name="specific_logits") - _logit_accuracy(
        common,
        labels,
        name="common_logits",
    )


@dataclass(frozen=True)
class CounterfactualMetrics:
    """The two report-level counterfactual effectiveness measurements."""

    identity_retention: float
    environment_switch: float

    @property
    def cf_identity_retention(self) -> float:
        return self.identity_retention

    @property
    def cf_environment_switch(self) -> float:
        return self.environment_switch

    def to_dict(self) -> dict[str, float]:
        return {
            "identity_retention": float(self.identity_retention),
            "environment_switch": float(self.environment_switch),
        }

    def __getitem__(self, key: str) -> float:
        if key in {"cf_identity_retention", "identity_retention"}:
            return float(self.identity_retention)
        if key in {"cf_environment_switch", "environment_switch"}:
            return float(self.environment_switch)
        raise KeyError(key)

    def keys(self):
        return self.to_dict().keys()

    def items(self):
        return self.to_dict().items()


def _prediction_vector(value: Any, *, name: str) -> np.ndarray:
    array = _to_numpy(value, name=name)
    if array.ndim == 2:
        if array.shape[1] == 0:
            raise ValueError(f"{name} must contain at least one class")
        return np.argmax(np.asarray(array, dtype=np.float64), axis=1)
    return _vector(array, name=name)


def counterfactual_effectiveness(
    original_logits: Any | None = None,
    counterfactual_logits: Any | None = None,
    labels: Any | None = None,
    counterfactual_environment_logits: Any | None = None,
    target_environment_labels: Any | None = None,
    *,
    counterfactual_environment_predictions: Any | None = None,
    target_environment_logits: Any | None = None,
    **kwargs: Any,
) -> CounterfactualMetrics:
    """Return CF identity retention and target-environment switch accuracy.

    Identity retention is CF TX accuracy when TX labels are supplied; without
    labels it is prediction consistency with the original logits.  Environment
    switch is accuracy of the CF environment prediction against explicit
    target-environment labels.  Both terms are test-time comparisons only and
    do not update model state.
    """

    original_logits = _resolve_alias(
        original_logits,
        kwargs,
        name="original_logits",
        aliases=("original", "identity_logits"),
    )
    counterfactual_logits = _resolve_alias(
        counterfactual_logits,
        kwargs,
        name="counterfactual_logits",
        aliases=("cf_logits", "counterfactual_identity_logits"),
    )
    labels = _resolve_alias(labels, kwargs, name="labels", aliases=("tx_labels", "identity_labels"))
    counterfactual_environment_logits = _resolve_alias(
        counterfactual_environment_logits,
        kwargs,
        name="counterfactual_environment_logits",
        aliases=("cf_environment_logits", "cf_env_logits"),
    )
    target_environment_labels = _resolve_alias(
        target_environment_labels,
        kwargs,
        name="target_environment_labels",
        aliases=("target_env_labels", "environment_labels"),
    )
    _reject_kwargs(kwargs)

    if original_logits is None or counterfactual_logits is None:
        raise TypeError("original_logits and counterfactual_logits are required")
    original = _numeric_matrix(original_logits, name="original_logits")
    counterfactual = _numeric_matrix(counterfactual_logits, name="counterfactual_logits")
    if original.shape != counterfactual.shape:
        raise ValueError("original_logits and counterfactual_logits must have matching shapes")
    if labels is None:
        identity_retention = float(
            np.mean(np.argmax(original, axis=1) == np.argmax(counterfactual, axis=1))
        )
    else:
        identity_retention = _logit_accuracy(counterfactual, labels, name="counterfactual_logits")

    if target_environment_labels is None and target_environment_logits is not None:
        target_environment_labels = _prediction_vector(
            target_environment_logits,
            name="target_environment_logits",
        )
    if target_environment_labels is None:
        raise TypeError("target_environment_labels are required for environment switch")
    target = _vector(
        target_environment_labels,
        name="target_environment_labels",
        expected_length=len(counterfactual),
    )
    if counterfactual_environment_predictions is not None and counterfactual_environment_logits is not None:
        raise TypeError(
            "pass only one of counterfactual_environment_logits and counterfactual_environment_predictions"
        )
    if counterfactual_environment_predictions is not None:
        prediction = _vector(
            counterfactual_environment_predictions,
            name="counterfactual_environment_predictions",
            expected_length=len(counterfactual),
        )
        environment_switch = float(np.mean(prediction == target))
    elif counterfactual_environment_logits is not None:
        environment_switch = _logit_accuracy(
            counterfactual_environment_logits,
            target,
            name="counterfactual_environment_logits",
        )
    else:
        raise TypeError(
            "counterfactual environment logits or predictions are required for environment switch"
        )
    return CounterfactualMetrics(identity_retention, environment_switch)


def minimum_class_margin(class_centers: Any) -> float:
    """Return the minimum Euclidean distance between distinct class centers."""

    centers = _numeric_matrix(class_centers, name="class_centers")
    if len(centers) < 2:
        return float("inf")
    distances = np.linalg.norm(centers[:, None, :] - centers[None, :, :], axis=-1)
    upper = distances[np.triu_indices(len(centers), k=1)]
    return float(np.min(upper))


def domain_drift_ratio(class_domain_centers: Any, class_centers: Any) -> float:
    """Return the squared cross-domain drift ratio from the HCF-DG report.

    ``class_domain_centers`` has shape ``[classes, domains, features]`` and
    ``class_centers`` has shape ``[classes, features]``.  The numerator is
    ``E||mu[y,r] - mu[y]||^2``.  The denominator is the mean, over classes, of
    each class center's nearest other-class squared distance, plus epsilon.
    """

    domain = _numeric_array(class_domain_centers, name="class_domain_centers")
    centers = _numeric_matrix(class_centers, name="class_centers")
    if domain.ndim == 2 and domain.shape == centers.shape:
        domain = domain[:, None, :]
    elif (
        domain.ndim == 2
        and domain.shape[1] == centers.shape[1]
        and domain.shape[0] % centers.shape[0] == 0
    ):
        domain = domain.reshape(centers.shape[0], -1, centers.shape[1])
    elif domain.ndim == 3 and domain.shape[0] == centers.shape[0] and domain.shape[2] == centers.shape[1]:
        pass
    else:
        raise ValueError(
            "class_domain_centers must have shape [classes, domains, features] aligned with class_centers"
        )
    if len(centers) < 2:
        raise ValueError("at least two class centers are required for domain_drift_ratio")
    drift = float(np.mean(np.sum((domain - centers[:, None, :]) ** 2, axis=-1)))
    pairwise_squared = np.sum(
        (centers[:, None, :] - centers[None, :, :]) ** 2,
        axis=-1,
    )
    np.fill_diagonal(pairwise_squared, np.inf)
    denominator = float(np.mean(np.min(pairwise_squared, axis=1))) + _EPS
    return drift / denominator


def harmonic_selection_score(clean: float, leo_mean: float) -> float:
    """Return the harmonic mean used for same-row source selection."""

    clean = float(clean)
    leo_mean = float(leo_mean)
    if not np.isfinite(clean) or not np.isfinite(leo_mean):
        raise ValueError("clean and leo_mean must be finite")
    if clean <= 0.0 or leo_mean <= 0.0:
        return 0.0
    return 2.0 * clean * leo_mean / (clean + leo_mean)


def _json_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


@dataclass(frozen=True)
class SameRowMetrics:
    """All source-selection values belonging to one candidate/fold/seed row."""

    row_id: str
    candidate_id: str
    heldout_receiver: int
    seed: int
    clean: float
    leo_mean: float | None = None
    leo_floor: float | None = None
    lodo_mean: float | None = None
    lodo_floor: float | None = None
    conditional_receiver_leakage: float | None = None
    environment_tx_leakage: float | None = None
    delta_spec: float | None = None
    cf_identity_retention: float | None = None
    cf_environment_switch: float | None = None
    r_drift: float | None = None
    min_class_margin: float | None = None
    harmonic_score: float | None = None
    resources: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.row_id, str) or not self.row_id.strip():
            raise ValueError("row_id must be a non-empty string")
        if not isinstance(self.candidate_id, str) or not self.candidate_id.strip():
            raise ValueError("candidate_id must be a non-empty string")
        if (
            isinstance(self.heldout_receiver, (bool, np.bool_))
            or not isinstance(self.heldout_receiver, (int, np.integer))
            or self.heldout_receiver <= 0
        ):
            raise ValueError("heldout_receiver must be a positive integer")
        if (
            isinstance(self.seed, (bool, np.bool_))
            or not isinstance(self.seed, (int, np.integer))
            or self.seed < 0
        ):
            raise ValueError("seed must be a non-negative integer")
        object.__setattr__(self, "heldout_receiver", int(self.heldout_receiver))
        object.__setattr__(self, "seed", int(self.seed))
        numeric_fields = (
            "clean",
            "leo_mean",
            "leo_floor",
            "lodo_mean",
            "lodo_floor",
            "conditional_receiver_leakage",
            "environment_tx_leakage",
            "delta_spec",
            "cf_identity_retention",
            "cf_environment_switch",
            "r_drift",
            "min_class_margin",
            "harmonic_score",
        )
        for field_name in numeric_fields:
            value = getattr(self, field_name)
            if value is None:
                continue
            value = float(value)
            if not np.isfinite(value):
                raise ValueError(f"{field_name} must be finite when provided")
            object.__setattr__(self, field_name, value)
        if self.harmonic_score is None and self.leo_mean is not None:
            object.__setattr__(
                self,
                "harmonic_score",
                harmonic_selection_score(self.clean, self.leo_mean),
            )
        if self.resources is not None:
            object.__setattr__(self, "resources", dict(self.resources))

    @property
    def selection_score(self) -> float | None:
        return self.harmonic_score

    @property
    def clean_accuracy(self) -> float:
        return self.clean

    @property
    def leo_mean_accuracy(self) -> float | None:
        return self.leo_mean

    @property
    def leo_floor_accuracy(self) -> float | None:
        return self.leo_floor

    @property
    def domain_drift_ratio(self) -> float | None:
        return self.r_drift

    def to_dict(self) -> dict[str, Any]:
        return {
            "row_id": self.row_id,
            "candidate_id": self.candidate_id,
            "heldout_receiver": self.heldout_receiver,
            "seed": self.seed,
            "clean": self.clean,
            "leo_mean": self.leo_mean,
            "leo_floor": self.leo_floor,
            "lodo_mean": self.lodo_mean,
            "lodo_floor": self.lodo_floor,
            "conditional_receiver_leakage": self.conditional_receiver_leakage,
            "environment_tx_leakage": self.environment_tx_leakage,
            "delta_spec": self.delta_spec,
            "cf_identity_retention": self.cf_identity_retention,
            "cf_environment_switch": self.cf_environment_switch,
            "r_drift": self.r_drift,
            "min_class_margin": self.min_class_margin,
            "harmonic_score": self.harmonic_score,
            "resources": _json_value(self.resources),
        }

    def to_json(self, *, sort_keys: bool = True) -> str:
        return json.dumps(self.to_dict(), sort_keys=sort_keys)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SameRowMetrics":
        if not isinstance(payload, Mapping):
            raise TypeError("SameRowMetrics.from_dict expects a mapping")
        values = dict(payload)
        aliases = {
            "clean_accuracy": "clean",
            "leo_mean_accuracy": "leo_mean",
            "leo_floor_accuracy": "leo_floor",
            "domain_drift_ratio": "r_drift",
            "selection_score": "harmonic_score",
        }
        for source, target in aliases.items():
            if target not in values and source in values:
                values[target] = values[source]
        required_fields = {
            "row_id",
            "candidate_id",
            "heldout_receiver",
            "seed",
            "clean",
        }
        missing = sorted(required_fields.difference(values))
        if missing:
            raise ValueError(
                "missing required SameRowMetrics field(s): " + ", ".join(missing)
            )
        field_names = {field.name for field in dataclass_fields(cls)}
        return cls(**{name: values[name] for name in field_names if name in values})

    @classmethod
    def from_json(cls, value: str) -> "SameRowMetrics":
        return cls.from_dict(json.loads(value))


def _coerce_row(value: Any) -> SameRowMetrics:
    if isinstance(value, SameRowMetrics):
        return value
    if isinstance(value, Mapping):
        return SameRowMetrics.from_dict(value)
    if hasattr(value, "to_dict"):
        return SameRowMetrics.from_dict(value.to_dict())
    values = {
        name: getattr(value, name)
        for name in (field.name for field in dataclass_fields(SameRowMetrics))
        if hasattr(value, name)
    }
    if not values:
        raise TypeError("rows must contain SameRowMetrics-compatible values")
    return SameRowMetrics.from_dict(values)


def rank_source_rows(rows: Iterable[SameRowMetrics] | Mapping[Any, Any]) -> list[SameRowMetrics]:
    """Rank complete rows without mixing fields from different rows.

    Primary order is the row-local harmonic clean/LEO mean score, followed by
    the row-local LEO floor, clean accuracy, minimum class margin and row ID.
    """

    if isinstance(rows, Mapping):
        values = [rows] if "row_id" in rows else list(rows.values())
    else:
        values = list(rows)
    resolved = [_coerce_row(value) for value in values]
    row_ids = [row.row_id for row in resolved]
    if len(set(row_ids)) != len(row_ids):
        raise ValueError("source rows must have unique row_id values")

    def descending(value: float | None) -> float:
        if value is None or not np.isfinite(value):
            return float("inf")
        return -float(value)

    return sorted(
        resolved,
        key=lambda row: (
            descending(row.harmonic_score),
            descending(row.leo_floor),
            descending(row.clean),
            descending(row.min_class_margin),
            row.row_id,
        ),
    )


__all__ = [
    "CounterfactualMetrics",
    "ProbeMetric",
    "SameRowMetrics",
    "conditional_receiver_leakage",
    "counterfactual_effectiveness",
    "domain_drift_ratio",
    "environment_tx_leakage",
    "harmonic_selection_score",
    "minimum_class_margin",
    "rank_source_rows",
    "specific_gap",
]
