"""Task6 integration of the BiCAD-XR training mechanisms.

The dual backbone remains an inference-only feature producer.  This module
owns the training heads, sparse mechanisms, stage routing and audit/runtime
state so enabling BiCAD-XR cannot add parameters to the legacy model.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
import math
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .adversarial_game import DynamicGRLDoseController
from .config import BiCADXRConfig, BiCADXRStage, candidate_config, stage_for_update
from .gradients import (
    DEFAULT_LOCAL_PROJECTION_ALLOWLIST,
    measure_bounded_gradient_ratio,
    project_conflicting_gradient,
    scale_explicit_gradients,
)
from .heads import FactorizedAdversarialHeads, FactorizedDomainProjector
from .losses import (
    DetachedEMA,
    apply_margin_tail,
    classification_margin,
    conditional_cross_covariance,
    group_margin_cvar,
    paired_satellite_loss,
)
from .pair import pair_delta_objectives, pair_identity_hinge, vicreg_pair_loss
from .sampler import StructuredEpisode, build_structured_episode
from .tailguard import (
    bounded_hard_group_weights,
    margin_group_risks,
    margin_rex_cvar_loss,
)
from .tangent import (
    ReceiverTangentBank,
    factual_tangent,
    one_step_tangent_worst_direction,
)
from .xdc import XDCLossOutput, xdc_losses


_LEGACY_RUNTIME_CANDIDATES = frozenset(
    {
        "D0",
        "D1",
        "D2",
        "D3",
        "D4",
        "D5",
        "D6",
        "E0",
        "E1",
        "E2",
        "E3",
        "E4",
        "F0",
        "F1",
        "F2",
        "F3",
        "ADV3B02-BICAD-XDC-V1",
    }
)
_PAIR_RUNTIME_DEFAULTS: dict[str, Any] = {
    "strict_pair_concat": False,
    "pair_identity": False,
    "pair_vicreg": False,
    "pair_delta": False,
    "dynamic_adversarial_dose": False,
    "satellite_supervision_mode": "ce_only",
    "pair_projector_dim": 128,
    "factor_interaction_dim": 24,
    "lambda_sat_cls_start": 0.68,
    "lambda_sat_cls_end": 0.68,
}
_CV2_PAIR_IDENTITY_CANDIDATES = frozenset({"CV2-T1", "CV2-T3"})
_CV2_MARGIN_REX_CVAR_CANDIDATES = frozenset({"CV2-T2", "CV2-T3"})
_CV2_PAIR_IDENTITY_EPSILON = 0.05
_CV2_PAIR_IDENTITY_WEIGHT = 0.02
_LEGACY_PAIR_IDENTITY_WEIGHT = 0.08
_CV2_MARGIN_REX_LAMBDA = 0.02
_CV2_MARGIN_CVAR_LAMBDA = 0.05
_CV2_MARGIN_TAIL_FRACTION = 0.2
_CV2_HARD_GROUP_FRACTION = 0.2
_CV2_HARD_GROUP_CAP = 0.30
_CV2_ADVERSARIAL_HEADS = frozenset(
    {"id_receiver", "id_day", "id_channel", "dom_tx"}
)


@dataclass(frozen=True)
class BiCADXRBatch:
    """One source-only batch consumed by :class:`BiCADXRTrainer`.

    ``x``, ``tx``, ``receiver``, ``day`` and ``channel`` are the ordinary
    source rows.  The four optional pair fields are the already-computed
    clean/satellite outputs of the existing concat augmenter; the trainer
    never creates a second pair forward.
    """

    x: Tensor
    tx: Tensor | None
    receiver: Tensor
    day: Tensor
    channel: Tensor
    physical_indices: Tensor | Sequence[int] | None = None
    labeled_mask: Tensor | None = None
    clean_z_id: Tensor | None = None
    satellite_z_id: Tensor | None = None
    clean_logits: Tensor | None = None
    satellite_logits: Tensor | None = None
    pair_tx: Tensor | None = None
    concat_pair: Mapping[str, Tensor] | None = None
    epoch: int | None = None
    source_loro_risk: Tensor | float | None = None
    source_loro_window: bool | None = None
    labeled_tx: Tensor | None = None

    def pair_payload(self) -> dict[str, Tensor]:
        payload: dict[str, Tensor] = {}
        if self.concat_pair is not None:
            payload.update(dict(self.concat_pair))
        direct = {
            "clean_z_id": self.clean_z_id,
            "satellite_z_id": self.satellite_z_id,
            "clean_logits": self.clean_logits,
            "satellite_logits": self.satellite_logits,
            "tx": self.pair_tx,
        }
        for key, value in direct.items():
            if value is not None:
                payload[key] = value
        return payload


@dataclass(frozen=True)
class BiCADXRTrainOutput:
    """Differentiable training output plus complete per-step audit state."""

    total: Tensor
    logits: Tensor
    features: Mapping[str, Tensor]
    audit: dict[str, Any]
    checkpoint_runtime: dict[str, Any]
    backward_plan: "BiCADXRBackwardPlan"

    @property
    def loss(self) -> Tensor:
        return self.total

    @property
    def tx_logits(self) -> Tensor:
        return self.logits

    @property
    def model_output(self) -> Mapping[str, Tensor]:
        return self.features


@dataclass(frozen=True)
class BiCADXRBackwardPlan:
    """Loss decomposition used by explicit pre-optimizer gradient controls."""

    total: Tensor
    domain_forward: Tensor
    adversarial: Tensor
    task_reference: Tensor
    stage: BiCADXRStage
    update: int
    firewall_enabled: bool
    projection_enabled: bool
    conditional_adversarial: Tensor | None = None
    zdom_tx_adversarial: Tensor | None = None


@dataclass(frozen=True)
class _StructuredEpisodeSelection:
    episode: StructuredEpisode
    batch_indices: Tensor
    global_tx: Tensor
    global_receiver: Tensor


def _as_long_labels(value: Any, name: str, *, size: int, device: torch.device) -> Tensor | None:
    if value is None:
        return None
    try:
        labels = torch.as_tensor(value, device=device)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise ValueError(f"{name} must contain integer labels") from exc
    if labels.ndim != 1 or labels.numel() != size:
        raise ValueError(f"{name} must match the batch")
    if labels.dtype == torch.bool or labels.is_complex():
        raise ValueError(f"{name} must contain integer labels")
    if labels.is_floating_point():
        if not bool(torch.isfinite(labels).all()):
            raise ValueError(f"{name} must contain finite integer labels")
        converted = labels.to(dtype=torch.long)
        if not torch.equal(labels, converted.to(dtype=labels.dtype)):
            raise ValueError(f"{name} must contain integer labels")
        labels = converted
    else:
        labels = labels.to(dtype=torch.long)
    if labels.numel() and int(labels.min().item()) < 0:
        raise ValueError(f"{name} must contain non-negative labels")
    return labels


def _as_mask(value: Any, *, size: int, device: torch.device) -> Tensor | None:
    if value is None:
        return None
    mask = torch.as_tensor(value, device=device)
    if mask.ndim != 1 or mask.numel() != size or mask.dtype != torch.bool:
        raise ValueError("labeled_mask must be a boolean vector matching the batch")
    return mask


def _finite_feature(value: Any, name: str, *, batch_size: int | None = None) -> Tensor:
    if not torch.is_tensor(value) or value.ndim != 2:
        raise ValueError(f"{name} must have shape [batch, feature]")
    if not value.is_floating_point() or value.size(1) < 1:
        raise ValueError(f"{name} must be a non-empty floating-point matrix")
    if batch_size is not None and value.size(0) != batch_size:
        raise ValueError(f"{name} must match the batch")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must contain only finite values")
    return value


def _module_device(module: nn.Module) -> torch.device:
    try:
        return next(module.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _infer_positive_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        resolved = int(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return resolved if resolved > 0 else None


def _infer_feature_dim(model: nn.Module) -> int:
    for owner in (model, getattr(model, "id_backbone", None)):
        if owner is None:
            continue
        for name in ("emb_dim", "embed_dim", "feature_dim"):
            value = _infer_positive_int(getattr(owner, name, None))
            if value is not None:
                return value
        classifier = getattr(owner, "classifier", None)
        value = _infer_positive_int(getattr(classifier, "in_features", None))
        if value is not None:
            return value
        cls_head = getattr(owner, "cls_head", None)
        head = getattr(cls_head, "head", None)
        value = _infer_positive_int(getattr(head, "in_features", None))
        if value is not None:
            return value
    raise ValueError("cannot infer BiCAD-XR feature dimension from model")


def _infer_num_classes(model: nn.Module) -> int:
    for owner in (model, getattr(model, "id_backbone", None)):
        if owner is None:
            continue
        value = _infer_positive_int(getattr(owner, "num_classes", None))
        if value is not None:
            return value
        for classifier_name in ("classifier", "cls_head", "head"):
            classifier = getattr(owner, classifier_name, None)
            value = _infer_positive_int(getattr(classifier, "out_features", None))
            if value is not None:
                return value
            head = getattr(classifier, "head", None)
            value = _infer_positive_int(getattr(head, "out_features", None))
            if value is not None:
                return value
    raise ValueError("cannot infer BiCAD-XR class count from model")


def _combine_group_ids(*labels: Tensor) -> Tensor:
    if not labels:
        raise ValueError("at least one group label is required")
    result = labels[0].to(dtype=torch.long)
    for label in labels[1:]:
        label = label.to(dtype=torch.long)
        extent = int(label.max().item()) + 1 if label.numel() else 1
        result = result * max(1, extent) + label
    return result


def _scalar(value: Tensor | float | int) -> float:
    if torch.is_tensor(value):
        if value.numel() != 1:
            raise ValueError("audit values must be scalar tensors")
        resolved = float(value.detach().cpu().item())
    else:
        resolved = float(value)
    if not math.isfinite(resolved):
        raise ValueError("audit scalar must be finite")
    return resolved


def _finite_gradient_vector(
    values: Sequence[Tensor | None],
    name: str,
    *,
    parameters: Sequence[Tensor] | None = None,
) -> Tensor | None:
    if parameters is not None and len(values) != len(parameters):
        raise ValueError(f"{name} must align with its parameter sequence")
    finite: list[Tensor] = []
    for index, value in enumerate(values):
        if value is None:
            if parameters is not None:
                finite.append(torch.zeros_like(parameters[index]).reshape(-1))
            continue
        if not torch.is_tensor(value) or not value.is_floating_point():
            raise ValueError(f"{name}[{index}] must be a floating-point tensor")
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"{name}[{index}] must contain only finite values")
        finite.append(value.detach().reshape(-1))
    if not finite:
        return None
    result = torch.cat(finite)
    if not bool(torch.isfinite(result).all()):
        raise ValueError(f"{name} must contain only finite values")
    return result


class BiCADXRTrainer(nn.Module):
    """Route the Task1–5 APIs through the frozen Stage0–4 schedule."""

    def __init__(
        self,
        model: nn.Module,
        config: BiCADXRConfig | str,
        *,
        num_receivers: int | None = None,
        num_days: int = 3,
        num_channels: int = 4,
        generator: torch.Generator | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(model, nn.Module):
            raise ValueError("model must be a torch.nn.Module")
        if isinstance(config, str):
            config = candidate_config(config)
        if not isinstance(config, BiCADXRConfig):
            raise ValueError("config must be a BiCADXRConfig or candidate ID")
        self.model = model
        self.config = config
        self.feature_dim = _infer_feature_dim(model)
        self.num_classes = _infer_num_classes(model)
        self.num_receivers = (
            int(num_receivers)
            if num_receivers is not None
            else int(config.xdc_microepisode_receivers)
        )
        self.num_days = int(num_days)
        self.num_channels = int(num_channels)
        if min(self.num_receivers, self.num_days, self.num_channels) < 1:
            raise ValueError("environment class counts must be positive")
        if config.sparse_xdc and (
            config.xdc_microepisode_tx,
            config.xdc_microepisode_receivers,
            config.xdc_samples_per_cell,
        ) != (6, 4, 2):
            raise ValueError("BiCAD-XR XDC requires a fixed 6x4x2 structured episode")
        if generator is None:
            generator = torch.Generator(device="cpu").manual_seed(0)
        if not isinstance(generator, torch.Generator):
            raise ValueError("generator must be a torch.Generator or None")
        self._generator = generator

        self.factorized_heads: FactorizedAdversarialHeads | None = None
        if config.factorized_domains or config.conditional_cdan or config.zdom_tx_adversary:
            self.factorized_heads = FactorizedAdversarialHeads(
                self.feature_dim,
                self.num_classes,
                self.num_receivers,
                self.num_days,
                self.num_channels,
            ).to(_module_device(model))

        self.factorized_projector: FactorizedDomainProjector | None = None
        if config.strict_pair_concat and config.factorized_domains:
            self.factorized_projector = FactorizedDomainProjector(
                self.feature_dim,
                self.feature_dim,
                config.factor_interaction_dim,
            ).to(_module_device(model))
        self.pair_projector: nn.Linear | None = None
        if config.strict_pair_concat and (config.pair_identity or config.pair_vicreg or config.pair_delta):
            self.pair_projector = nn.Linear(
                self.feature_dim,
                config.pair_projector_dim,
            ).to(_module_device(model))

        self.tangent_bank: ReceiverTangentBank | None = None
        if config.receiver_tangent != "off":
            self.tangent_bank = ReceiverTangentBank(
                self.feature_dim,
                rank=config.receiver_tangent_rank,
                source_receivers=range(self.num_receivers),
            )
        self._tail_emas: tuple[DetachedEMA, DetachedEMA, DetachedEMA] | None = None
        if config.margin_tail:
            self._tail_emas = tuple(
                DetachedEMA(decay=config.margin_tail_ema) for _ in range(3)
            )  # type: ignore[assignment]
        self._swad_state: dict[str, Any] | None = None
        self._backward_control_state: dict[str, Any] = {
            "firewall_applications": 0,
            "projection_applications": 0,
            "projection_triggers": 0,
            "last_update": None,
        }
        self._last_update: int | None = None
        self._last_total_updates: int | None = None
        self._pairbicad_runtime_state: dict[str, Any] | None = None
        self._dynamic_grl_controller: DynamicGRLDoseController | None = (
            DynamicGRLDoseController()
            if config.dynamic_adversarial_dose
            else None
        )

    @property
    def swad_state(self) -> dict[str, Any] | None:
        return self._swad_state

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Delegate inference to the candidate model loaded by this trainer."""

        return self.model(*args, **kwargs)

    @staticmethod
    def _component(
        raw: Tensor | float = 0.0,
        weighted: Tensor | float = 0.0,
        *,
        called: bool = False,
        effective_count: int = 0,
        skip_reason: str | None = None,
    ) -> dict[str, Any]:
        return {
            "raw": _scalar(raw),
            "weighted": _scalar(weighted),
            "called": bool(called),
            "effective_count": int(effective_count),
            "skip_reason": skip_reason,
        }

    def _pair_satellite_weight(self, update: int, total_updates: int) -> float:
        if not self.config.strict_pair_concat:
            return float(self.config.lambda_sat_cls)
        progress = (float(update) - 1.0) / max(1.0, float(total_updates) - 1.0)
        progress = min(1.0, max(0.0, progress))
        return float(
            self.config.lambda_sat_cls_start
            + progress
            * (self.config.lambda_sat_cls_end - self.config.lambda_sat_cls_start)
        )

    def _pair_adversarial_dose(self, stage: BiCADXRStage) -> float:
        if not self.config.dynamic_adversarial_dose:
            return 0.0
        return {
            BiCADXRStage.stage0: 0.20,
            BiCADXRStage.stage1: 0.20,
            BiCADXRStage.stage2: 0.20,
            BiCADXRStage.stage3: 0.22,
            BiCADXRStage.stage4: 0.25,
        }[stage]

    @staticmethod
    def _prediction_js(clean_logits: Tensor, satellite_logits: Tensor) -> Tensor:
        if clean_logits.shape != satellite_logits.shape:
            raise ValueError("clean and satellite prediction logits must have the same shape")
        if clean_logits.ndim != 2 or clean_logits.size(0) < 1:
            raise ValueError("pair prediction logits must be non-empty matrices")
        clean_log_prob = F.log_softmax(clean_logits, dim=1)
        satellite_log_prob = F.log_softmax(satellite_logits, dim=1)
        clean_prob = clean_log_prob.exp()
        satellite_prob = satellite_log_prob.exp()
        mean_log_prob = (0.5 * (clean_prob + satellite_prob)).clamp_min(1e-8).log()
        return 0.5 * (
            F.kl_div(mean_log_prob, clean_prob, reduction="batchmean")
            + F.kl_div(mean_log_prob, satellite_prob, reduction="batchmean")
        )

    @staticmethod
    def _discriminator_accuracy(
        factorized_output: Mapping[str, Tensor],
        labeled_targets: Mapping[str, Tensor],
    ) -> float | None:
        accuracies: list[Tensor] = []
        for name, labels in labeled_targets.items():
            prediction = factorized_output.get(name)
            if prediction is None or labels.numel() == 0:
                continue
            if prediction.ndim != 2 or prediction.size(0) != labels.numel():
                raise ValueError("discriminator output and labels must align")
            accuracies.append(
                (prediction.detach().argmax(dim=1) == labels.detach())
                .to(dtype=torch.float32)
                .mean()
            )
        if not accuracies:
            return None
        result = float(torch.stack(accuracies).mean().item())
        if not math.isfinite(result):
            raise ValueError("discriminator accuracy must be finite")
        return result

    @staticmethod
    def _coerce_batch(batch: BiCADXRBatch | Mapping[str, Any] | Sequence[Any]) -> BiCADXRBatch:
        if isinstance(batch, BiCADXRBatch):
            return batch
        if isinstance(batch, Mapping):
            x = batch.get("x", batch.get("inputs", batch.get("iq")))
            tx = batch.get("tx", batch.get("y_tx", batch.get("labels")))
            labeled_tx = batch.get("labeled_tx", batch.get("tx_labeled"))
            receiver = batch.get("receiver", batch.get("rx"))
            day = batch.get("day", batch.get("day_id"))
            channel = batch.get("channel", batch.get("channel_id"))
            if x is None:
                raise ValueError("batch mapping must contain x/inputs/iq")
            size = int(torch.as_tensor(x).size(0))
            zeros = torch.zeros(size, dtype=torch.long)
            return BiCADXRBatch(
                x=x,
                tx=tx,
                labeled_tx=labeled_tx,
                receiver=zeros if receiver is None else receiver,
                day=zeros if day is None else day,
                channel=zeros if channel is None else channel,
                physical_indices=batch.get("physical_indices"),
                labeled_mask=batch.get("labeled_mask"),
                clean_z_id=batch.get("clean_z_id"),
                satellite_z_id=batch.get("satellite_z_id"),
                clean_logits=batch.get("clean_logits"),
                satellite_logits=batch.get("satellite_logits"),
                pair_tx=batch.get(
                    "pair_tx",
                    batch.get("satellite_tx", batch.get("concat_pair_tx")),
                ),
                concat_pair=batch.get("concat_pair"),
                epoch=batch.get("epoch"),
                source_loro_risk=batch.get("source_loro_risk"),
                source_loro_window=batch.get("source_loro_window"),
            )
        if isinstance(batch, Sequence) and not isinstance(batch, (str, bytes)):
            if len(batch) < 5:
                raise ValueError("batch sequence must contain x, tx, receiver, day and channel")
            return BiCADXRBatch(
                x=batch[0],
                tx=batch[1],
                receiver=batch[2],
                day=batch[3],
                channel=batch[4],
            )
        raise ValueError("batch must be BiCADXRBatch, mapping or sequence")

    def _prepare_batch(self, batch: BiCADXRBatch | Mapping[str, Any] | Sequence[Any]) -> BiCADXRBatch:
        raw = self._coerce_batch(batch)
        device = _module_device(self.model)
        x = torch.as_tensor(raw.x, device=device)
        if not x.is_floating_point():
            x = x.float()
        if x.ndim < 2 or x.size(0) < 1:
            raise ValueError("batch x must have a non-empty batch dimension")
        size = int(x.size(0))
        tx = _as_long_labels(raw.tx, "tx", size=size, device=device)
        receiver = _as_long_labels(raw.receiver, "receiver", size=size, device=device)
        day = _as_long_labels(raw.day, "day", size=size, device=device)
        channel = _as_long_labels(raw.channel, "channel", size=size, device=device)
        assert receiver is not None and day is not None and channel is not None
        mask = _as_mask(raw.labeled_mask, size=size, device=device)
        labeled_size = size if mask is None else int(mask.sum().item())
        labeled_tx = _as_long_labels(
            raw.labeled_tx,
            "labeled_tx",
            size=labeled_size,
            device=device,
        )

        physical = raw.physical_indices
        if physical is not None:
            physical = torch.as_tensor(physical, dtype=torch.long, device="cpu")
            if physical.ndim != 1 or physical.numel() != size:
                raise ValueError("physical_indices must match the batch")
            if physical.numel() and int(physical.min().item()) < 0:
                raise ValueError("physical_indices must be non-negative")
            if torch.unique(physical).numel() != physical.numel():
                raise ValueError("physical_indices must be unique")

        pair_values: dict[str, Tensor | None] = {}
        for key in ("clean_z_id", "satellite_z_id", "clean_logits", "satellite_logits"):
            value = getattr(raw, key)
            pair_values[key] = None if value is None else torch.as_tensor(value, device=device)
        pair_tx = raw.pair_tx
        if pair_tx is not None:
            pair_tx = torch.as_tensor(pair_tx, device=device)
            if pair_tx.ndim != 1 or pair_tx.dtype == torch.bool or pair_tx.is_complex():
                raise ValueError("pair_tx must be a one-dimensional integer label vector")
            if pair_tx.is_floating_point():
                converted_pair_tx = pair_tx.to(dtype=torch.long)
                if not bool(torch.isfinite(pair_tx).all()) or not torch.equal(
                    pair_tx, converted_pair_tx.to(dtype=pair_tx.dtype)
                ):
                    raise ValueError("pair_tx must contain finite integer labels")
                pair_tx = converted_pair_tx
            else:
                pair_tx = pair_tx.to(dtype=torch.long)
            if pair_tx.numel() and int(pair_tx.min().item()) < 0:
                raise ValueError("pair_tx must contain non-negative labels")
        pair_mapping = None
        if raw.concat_pair is not None:
            pair_mapping = {
                key: torch.as_tensor(value, device=device)
                for key, value in raw.concat_pair.items()
            }
        epoch = raw.epoch
        if epoch is not None:
            if not isinstance(epoch, int) or isinstance(epoch, bool) or epoch < 1:
                raise ValueError("epoch must be a positive integer or None")
        source_loro_risk = raw.source_loro_risk
        if source_loro_risk is not None:
            source_loro_risk = _scalar(source_loro_risk)
        source_loro_window = raw.source_loro_window
        if source_loro_window is not None and not isinstance(source_loro_window, bool):
            raise ValueError("source_loro_window must be a bool or None")
        return BiCADXRBatch(
            x=x,
            tx=tx,
            labeled_tx=labeled_tx,
            receiver=receiver,
            day=day,
            channel=channel,
            physical_indices=physical,
            labeled_mask=mask,
            pair_tx=pair_tx,
            concat_pair=pair_mapping,
            epoch=epoch,
            source_loro_risk=source_loro_risk,
            source_loro_window=source_loro_window,
            **pair_values,
        )

    def _forward(self, batch: BiCADXRBatch) -> Mapping[str, Any]:
        model_tx = batch.tx
        if batch.labeled_mask is not None and not bool(batch.labeled_mask.all()):
            model_tx = None
        try:
            result = self.model(
                batch.x,
                y_tx=model_tx,
                return_aux=True,
                domain_labels=batch.receiver,
            )
        except TypeError as first_error:
            try:
                result = self.model(batch.x, y_tx=model_tx, return_aux=True)
            except TypeError:
                raise first_error
        if not isinstance(result, Mapping):
            raise ValueError("BiCAD-XR model must return a mapping in return_aux=True mode")
        return result

    @staticmethod
    def _output_tensor(output: Mapping[str, Any], *names: str) -> Tensor:
        for name in names:
            value = output.get(name)
            if torch.is_tensor(value):
                return value
        raise ValueError(f"model output is missing one of: {', '.join(names)}")

    def _tangent_logits(self, features: Tensor, labels: Tensor) -> Tensor:
        classifiers = (
            getattr(self.model, "classify_identity_features", None),
            getattr(self.model, "tangent_classifier", None),
            getattr(self.model, "classifier", None),
        )
        for classifier in classifiers:
            if not callable(classifier):
                continue
            try:
                result = classifier(features, labels=labels)
            except TypeError:
                try:
                    result = classifier(features, labels)
                except TypeError:
                    result = classifier(features)
            if (
                torch.is_tensor(result)
                and result.ndim == 2
                and result.size(0) == features.size(0)
                and result.size(1) == self.num_classes
                and bool(torch.isfinite(result).all())
            ):
                return result
        raise RuntimeError(
            "public TX classifier is unavailable for tangent identity features"
        )

    def _structured_episode(
        self,
        batch: BiCADXRBatch,
        labeled_indices: Tensor,
    ) -> _StructuredEpisodeSelection:
        if batch.tx is None:
            raise ValueError("structured XDC requires source TX labels")
        labeled_cpu = labeled_indices.detach().cpu()
        labeled_tx = batch.tx[labeled_indices].detach().cpu()
        labeled_receiver = batch.receiver[labeled_indices].detach().cpu()
        selected_tx = torch.unique(labeled_tx, sorted=True)[
            : self.config.xdc_microepisode_tx
        ]
        selected_receiver = torch.unique(labeled_receiver, sorted=True)[
            : self.config.xdc_microepisode_receivers
        ]
        tx_to_local = {
            int(value): index for index, value in enumerate(selected_tx.tolist())
        }
        receiver_to_local = {
            int(value): index
            for index, value in enumerate(selected_receiver.tolist())
        }
        eligible = torch.tensor(
            [
                int(tx) in tx_to_local and int(receiver) in receiver_to_local
                for tx, receiver in zip(labeled_tx.tolist(), labeled_receiver.tolist())
            ],
            dtype=torch.bool,
        )
        candidate_indices = labeled_cpu[eligible]
        candidate_tx = labeled_tx[eligible]
        candidate_receiver = labeled_receiver[eligible]
        local_tx = torch.tensor(
            [tx_to_local[int(value)] for value in candidate_tx.tolist()],
            dtype=torch.long,
        )
        local_receiver = torch.tensor(
            [receiver_to_local[int(value)] for value in candidate_receiver.tolist()],
            dtype=torch.long,
        )
        episode = build_structured_episode(
            local_tx,
            local_receiver,
            batch.day[labeled_indices][eligible.to(device=labeled_indices.device)].detach().cpu(),
            self.config.xdc_samples_per_cell,
            generator=self._generator,
            num_classes=self.config.xdc_microepisode_tx,
            num_receivers=self.config.xdc_microepisode_receivers,
            physical_indices=candidate_indices,
        )
        if not episode.indices:
            empty = torch.empty(0, dtype=torch.long, device=batch.tx.device)
            return _StructuredEpisodeSelection(
                episode=episode,
                batch_indices=empty,
                global_tx=empty,
                global_receiver=empty,
            )
        local = torch.tensor(
            episode.indices,
            dtype=torch.long,
            device=batch.tx.device,
        )
        if local.numel() and (
            int(local.min().item()) < 0 or int(local.max().item()) >= batch.tx.numel()
        ):
            raise ValueError("structured episode indices are outside the batch")
        global_tx = batch.tx[local]
        global_receiver = batch.receiver[local]
        remapped_tx = torch.tensor(
            [tx_to_local[int(value)] for value in global_tx.detach().cpu().tolist()],
            dtype=torch.long,
        )
        remapped_receiver = torch.tensor(
            [
                receiver_to_local[int(value)]
                for value in global_receiver.detach().cpu().tolist()
            ],
            dtype=torch.long,
        )
        if not torch.equal(remapped_tx, episode.tx):
            raise ValueError("structured episode TX labels lost batch-index alignment")
        if not torch.equal(remapped_receiver, episode.receiver):
            raise ValueError("structured episode receiver labels lost batch-index alignment")
        return _StructuredEpisodeSelection(
            episode=episode,
            batch_indices=local,
            global_tx=global_tx,
            global_receiver=global_receiver,
        )

    def _pair_payload(self, batch: BiCADXRBatch, model_output: Mapping[str, Any]) -> dict[str, Tensor]:
        payload = batch.pair_payload()
        output_pair = model_output.get("concat_pair")
        if isinstance(output_pair, Mapping):
            payload = {**dict(output_pair), **payload}
        return payload

    def _update_swad(self, risk: Tensor | float) -> bool:
        if self._swad_state is None:
            self._swad_state = {
                "candidate_id": self.config.candidate_id,
                "source_loro": True,
                "updates": 0,
                "window_risks": [],
                "average": {},
            }
        resolved_risk = _scalar(risk)
        with torch.no_grad():
            average = self._swad_state["average"]
            count = int(self._swad_state["updates"])
            for name, parameter in self.named_optimizer_parameters():
                value = parameter.detach().clone()
                if name not in average:
                    average[name] = value
                else:
                    average[name] = (average[name] * count + value) / float(count + 1)
        self._swad_state["window_risks"].append(resolved_risk)
        self._swad_state["updates"] = int(self._swad_state["updates"]) + 1
        return True

    @staticmethod
    def _clone_tensor_state(state: Mapping[str, Tensor]) -> dict[str, Tensor]:
        return {name: value.detach().cpu().clone() for name, value in state.items()}

    def _serialized_swad_state(self) -> dict[str, Any] | None:
        if self._swad_state is None:
            return None
        return {
            "candidate_id": self._swad_state["candidate_id"],
            "source_loro": True,
            "updates": int(self._swad_state["updates"]),
            "window_risks": list(self._swad_state["window_risks"]),
            "average": self._clone_tensor_state(self._swad_state["average"]),
        }

    def checkpoint_runtime(
        self,
        update: int | None = None,
        total_updates: int | None = None,
        *,
        stage: BiCADXRStage | None = None,
    ) -> dict[str, Any]:
        if update is None:
            update = self._last_update
        if total_updates is None:
            total_updates = self._last_total_updates
        if stage is None and update is not None and total_updates is not None:
            stage = stage_for_update(update, total_updates)
        config_values = asdict(self.config)
        config_values["sat_train_scenarios"] = list(self.config.sat_train_scenarios)
        factorized_state = (
            None
            if self.factorized_heads is None
            else self._clone_tensor_state(self.factorized_heads.state_dict())
        )
        factorized_projector_state = (
            None
            if self.factorized_projector is None
            else self._clone_tensor_state(self.factorized_projector.state_dict())
        )
        pair_projector_state = (
            None
            if self.pair_projector is None
            else self._clone_tensor_state(self.pair_projector.state_dict())
        )
        serialized_swad = self._serialized_swad_state()
        runtime = {
            "runtime_version": 2,
            "phase1_method": "bicad_xr",
            "candidate_id": self.config.candidate_id,
            "stage": None if stage is None else stage.name,
            "optimizer_update": update,
            "total_updates": total_updates,
            "feature_dim": self.feature_dim,
            "num_classes": self.num_classes,
            "num_receivers": self.num_receivers,
            "num_days": self.num_days,
            "num_channels": self.num_channels,
            "source_only": True,
            "target_access": False,
            "phase2_access": False,
            "support_access": False,
            "query_access": False,
            "truth_access": False,
            "return_aux_false_is_deploy_fast_path": True,
            "protocol": {
                "concat_sat_ce_only": self.config.concat_sat_ce_only,
                "lambda_sat_cls": self.config.lambda_sat_cls,
                "lambda_sat_cons": self.config.lambda_sat_cons,
                "concat_sat_start_epoch": self.config.concat_sat_start_epoch,
                "sat_train_scenarios": list(self.config.sat_train_scenarios),
            },
            "schedule": {
                "xdc_interval": self.config.xdc_interval,
                "pair_interval": self.config.pair_interval,
                "stage4_domain_scale": self.config.stage4_domain_scale,
                "stage4_shared_stem_lr_scale": self.config.stage4_shared_stem_lr_scale,
            },
            "candidate_config": config_values,
            "swad": {
                "enabled": bool(self.config.swad),
                "active": serialized_swad is not None,
                "source_loro": bool(self.config.swad),
                "updates": 0 if serialized_swad is None else serialized_swad["updates"],
                "state": serialized_swad,
            },
            "training_state": {
                "factorized_heads": factorized_state,
                "factorized_projector": factorized_projector_state,
                "pair_projector": pair_projector_state,
                "backward_controls": {
                    "firewall_scale": self.config.gradient_firewall_scale,
                    "stage4_shared_stem_lr_scale": self.config.stage4_shared_stem_lr_scale,
                    **dict(self._backward_control_state),
                },
            },
        }
        if self.config.strict_pair_concat:
            runtime["pairbicad_runtime"] = (
                self._pairbicad_runtime_state
                if self._pairbicad_runtime_state is not None
                else {
                    "runtime_version": 1,
                    "candidate_id": self.config.candidate_id,
                    "active": True,
                    "components": {},
                    "effective_counts": {},
                    "skip_reasons": {},
                    "one_forward_count": 0,
                    "finite": {"checked": False, "loss": None},
                    "gradient": {"checked": False, "finite": None, "rho_adv": 0.0},
                }
            )
        return runtime

    runtime_dict = checkpoint_runtime

    def load_checkpoint_runtime(
        self,
        runtime: Mapping[str, Any],
        *,
        strict: bool = True,
    ) -> None:
        """Restore Task6 training heads, controls and explicit SWAD state."""

        if not isinstance(runtime, Mapping):
            raise ValueError("checkpoint runtime must be a mapping")
        expected = {
            "runtime_version": 2,
            "phase1_method": "bicad_xr",
            "candidate_id": self.config.candidate_id,
            "feature_dim": self.feature_dim,
            "num_classes": self.num_classes,
            "num_receivers": self.num_receivers,
            "num_days": self.num_days,
            "num_channels": self.num_channels,
        }
        if strict:
            mismatches = [
                name for name, value in expected.items() if runtime.get(name) != value
            ]
            if mismatches:
                raise ValueError(
                    "checkpoint runtime mismatch: " + ", ".join(sorted(mismatches))
                )
            runtime_config = runtime.get("candidate_config")
            current_config = asdict(self.config)
            current_config["sat_train_scenarios"] = list(self.config.sat_train_scenarios)
            if not isinstance(runtime_config, Mapping):
                raise ValueError("checkpoint runtime mismatch: candidate_config")
            comparable_config = dict(runtime_config)
            candidate_key = self.config.candidate_id.strip().upper()
            if candidate_key in _LEGACY_RUNTIME_CANDIDATES:
                for name, default in _PAIR_RUNTIME_DEFAULTS.items():
                    comparable_config.setdefault(name, default)
            if comparable_config != current_config:
                raise ValueError("checkpoint runtime mismatch: candidate_config")

        training_state = runtime.get("training_state", {})
        if not isinstance(training_state, Mapping):
            raise ValueError("checkpoint training_state must be a mapping")
        head_state = training_state.get("factorized_heads")
        if head_state is None:
            if strict and self.factorized_heads is not None:
                raise ValueError("checkpoint is missing factorized head state")
        elif self.factorized_heads is None:
            if strict:
                raise ValueError("checkpoint contains unexpected factorized head state")
        else:
            self.factorized_heads.load_state_dict(head_state, strict=strict)

        projector_state = training_state.get("factorized_projector")
        if projector_state is None:
            if strict and self.factorized_projector is not None:
                raise ValueError("checkpoint is missing factorized projector state")
        elif self.factorized_projector is None:
            if strict:
                raise ValueError("checkpoint contains unexpected factorized projector state")
        else:
            self.factorized_projector.load_state_dict(projector_state, strict=strict)

        pair_projector_state = training_state.get("pair_projector")
        if pair_projector_state is None:
            if strict and self.pair_projector is not None:
                raise ValueError("checkpoint is missing pair projector state")
        elif self.pair_projector is None:
            if strict:
                raise ValueError("checkpoint contains unexpected pair projector state")
        else:
            self.pair_projector.load_state_dict(pair_projector_state, strict=strict)

        controls = training_state.get("backward_controls", {})
        if not isinstance(controls, Mapping):
            raise ValueError("checkpoint backward_controls must be a mapping")
        if strict and float(controls.get("firewall_scale", -1.0)) != float(
            self.config.gradient_firewall_scale
        ):
            raise ValueError("checkpoint firewall scale mismatch")
        for key in self._backward_control_state:
            if key in controls:
                self._backward_control_state[key] = controls[key]

        swad_payload = runtime.get("swad", {})
        if not isinstance(swad_payload, Mapping):
            raise ValueError("checkpoint swad state must be a mapping")
        swad_state = swad_payload.get("state")
        if swad_state is None:
            self._swad_state = None
        else:
            if not (
                self.config.swad and self.config.candidate_id.strip().upper() == "F3"
            ):
                if strict:
                    raise ValueError("SWAD state is only valid for F3")
                self._swad_state = None
            else:
                average = swad_state.get("average", {})
                if not isinstance(average, Mapping):
                    raise ValueError("SWAD average must be a mapping")
                parameters = dict(self.named_optimizer_parameters())
                if strict and set(average) != set(parameters):
                    raise ValueError("SWAD average parameter set mismatch")
                restored_average: dict[str, Tensor] = {}
                for name, value in average.items():
                    if name not in parameters:
                        if strict:
                            raise ValueError(f"unknown SWAD parameter: {name}")
                        continue
                    tensor = torch.as_tensor(
                        value,
                        device=parameters[name].device,
                        dtype=parameters[name].dtype,
                    )
                    if tensor.shape != parameters[name].shape:
                        raise ValueError(f"SWAD parameter shape mismatch: {name}")
                    restored_average[name] = tensor.detach().clone()
                self._swad_state = {
                    "candidate_id": self.config.candidate_id,
                    "source_loro": True,
                    "updates": int(swad_state.get("updates", 0)),
                    "window_risks": [
                        float(value) for value in swad_state.get("window_risks", [])
                    ],
                    "average": restored_average,
                }
        self._last_update = runtime.get("optimizer_update")
        self._last_total_updates = runtime.get("total_updates")
        pair_runtime = runtime.get("pairbicad_runtime")
        if pair_runtime is not None:
            if not isinstance(pair_runtime, Mapping):
                raise ValueError("checkpoint pairbicad_runtime must be a mapping")
            self._pairbicad_runtime_state = dict(pair_runtime)

    def named_optimizer_parameters(self) -> list[tuple[str, nn.Parameter]]:
        """Return every trainable model and Task6-head parameter exactly once."""

        return [
            (name, parameter)
            for name, parameter in self.named_parameters(remove_duplicate=True)
            if parameter.requires_grad
        ]

    def optimizer_parameters(self) -> list[nn.Parameter]:
        return [parameter for _, parameter in self.named_optimizer_parameters()]

    def adversarial_parameter_groups(self) -> dict[str, Any]:
        """Return disjoint outer-loop parameter groups and CV2 protections.

        Only active factorized adversarial heads are assigned to the
        discriminator group.  All other trainable parameters remain in the
        encoder group; constructing these groups never performs a forward.
        """

        named_parameters = self.named_optimizer_parameters()
        active_heads: set[str] = set()
        if self.factorized_heads is not None:
            if self.config.conditional_cdan:
                active_heads.update(
                    {"id_receiver", "id_day", "id_channel"}
                )
            if self.config.zdom_tx_adversary:
                active_heads.add("dom_tx")
        discriminator_prefixes = tuple(
            f"factorized_heads.{head}" for head in sorted(active_heads)
        )

        def is_discriminator_parameter(name: str) -> bool:
            return any(
                name == prefix or name.startswith(prefix + ".")
                for prefix in discriminator_prefixes
            )

        discriminator = tuple(
            parameter
            for name, parameter in named_parameters
            if is_discriminator_parameter(name)
        )
        discriminator_ids = {id(parameter) for parameter in discriminator}
        encoder = tuple(
            parameter
            for name, parameter in named_parameters
            if not is_discriminator_parameter(name)
        )
        encoder_ids = {id(parameter) for parameter in encoder}
        all_ids = {id(parameter) for _, parameter in named_parameters}
        if not encoder_ids.isdisjoint(discriminator_ids):
            raise RuntimeError("encoder and discriminator parameters must be disjoint")
        if encoder_ids | discriminator_ids != all_ids:
            raise RuntimeError("encoder and discriminator groups must cover trainable parameters")
        if not active_heads.issubset(_CV2_ADVERSARIAL_HEADS):
            raise RuntimeError("unexpected factorized adversarial head")
        return {
            "encoder": encoder,
            "discriminator": discriminator,
            "local_protection_allowlist": tuple(DEFAULT_LOCAL_PROJECTION_ALLOWLIST),
        }

    cv2_parameter_groups = adversarial_parameter_groups

    def cv2_local_protection_parameters(self) -> list[nn.Parameter]:
        """Return only the identity tail/fusion/projection protection scope."""

        protected: list[nn.Parameter] = []
        for name, parameter in self.named_optimizer_parameters():
            normalized = name.lower()
            if normalized.startswith("factorized_heads.") or "model.domain" in normalized:
                continue
            if any(
                token in normalized
                for token in (
                    "model.identity.",
                    "id_backbone.fuse.",
                    "id_backbone.con_proj.",
                    "id_backbone.cls_head.",
                    "pair_projector.",
                )
            ):
                protected.append(parameter)
        return protected

    def shared_stem_parameters(self) -> list[Tensor]:
        """Return only explicitly shared Sinc/HF parameters for LR control."""

        parameters: list[Tensor] = []
        seen: set[int] = set()
        for backbone in (
            getattr(self.model, "id_backbone", None),
            getattr(self.model, "dom_backbone", None),
        ):
            if backbone is None:
                continue
            for stem_name in ("sinc", "hf"):
                stem = getattr(backbone, stem_name, None)
                if stem is None:
                    continue
                for parameter in stem.parameters():
                    if id(parameter) not in seen:
                        seen.add(id(parameter))
                        parameters.append(parameter)
        return parameters

    def scale_shared_stem_gradients(self, scale: float | None = None) -> int:
        """Apply the configured Stage4 scale to an explicit stem parameter list."""

        if scale is None:
            scale = self.config.stage4_shared_stem_lr_scale
        return scale_explicit_gradients(self.shared_stem_parameters(), scale)

    @staticmethod
    def _accumulate_gradients(
        parameters: Sequence[nn.Parameter],
        gradients: Sequence[Tensor | None],
        *,
        scales: Mapping[int, float] | None = None,
    ) -> int:
        applied = 0
        with torch.no_grad():
            for parameter, gradient in zip(parameters, gradients):
                if gradient is None:
                    continue
                scale = 1.0 if scales is None else float(scales.get(id(parameter), 1.0))
                value = gradient.detach() * scale
                if parameter.grad is None:
                    parameter.grad = value.clone()
                else:
                    parameter.grad.add_(value)
                applied += 1
        return applied

    def apply_backward_controls(self, output: Any) -> dict[str, Any]:
        """Backpropagate one step with firewall, projection and Stage4 LR control.

        The caller must zero gradients first and invoke this method instead of
        ``output.total.backward()`` whenever Task6 controls are enabled.  The
        method does not perform an optimizer step.
        """

        plan = getattr(output, "backward_plan", None)
        if not isinstance(plan, BiCADXRBackwardPlan):
            raise ValueError("output must contain a BiCADXRBackwardPlan")
        parameters = self.optimizer_parameters()
        shared = self.shared_stem_parameters()
        shared_ids = {id(parameter) for parameter in shared}
        cv2_candidate = self.config.candidate_id.strip().upper().startswith("CV2-")
        protected = (
            self.cv2_local_protection_parameters() if cv2_candidate else shared
        )
        needs_decomposition = bool(plan.firewall_enabled or plan.projection_enabled)
        base_loss = plan.total
        if plan.firewall_enabled:
            base_loss = base_loss - plan.domain_forward
        if plan.projection_enabled:
            base_loss = base_loss - plan.adversarial
        base_loss.backward(retain_graph=needs_decomposition)

        firewall_applied = False
        if plan.firewall_enabled:
            domain_gradients = torch.autograd.grad(
                plan.domain_forward,
                parameters,
                retain_graph=plan.projection_enabled,
                allow_unused=True,
            )
            scales = {
                parameter_id: self.config.gradient_firewall_scale
                for parameter_id in shared_ids
            }
            self._accumulate_gradients(parameters, domain_gradients, scales=scales)
            firewall_applied = any(
                gradient is not None and id(parameter) in shared_ids
                for parameter, gradient in zip(parameters, domain_gradients)
            )
            if firewall_applied:
                self._backward_control_state["firewall_applications"] += 1

        projection_applied = False
        projection_triggered = False
        if plan.projection_enabled:
            task_gradients = torch.autograd.grad(
                plan.task_reference,
                protected,
                retain_graph=True,
                allow_unused=True,
            ) if protected else tuple()
            adversarial_gradients = list(
                torch.autograd.grad(
                    plan.adversarial,
                    parameters,
                    retain_graph=False,
                    allow_unused=True,
                )
            )
            task_by_id = {
                id(parameter): gradient
                for parameter, gradient in zip(protected, task_gradients)
            }
            for index, (parameter, gradient) in enumerate(
                zip(parameters, adversarial_gradients)
            ):
                reference = task_by_id.get(id(parameter))
                if gradient is None or reference is None:
                    continue
                projected = project_conflicting_gradient(gradient, reference)
                assert torch.is_tensor(projected)
                projection_triggered = projection_triggered or not torch.equal(
                    projected, gradient
                )
                adversarial_gradients[index] = projected
                projection_applied = True
            self._accumulate_gradients(parameters, adversarial_gradients)
            if projection_applied:
                self._backward_control_state["projection_applications"] += 1
            if projection_triggered:
                self._backward_control_state["projection_triggers"] += 1

        stage4_scaled_count = 0
        if plan.stage is BiCADXRStage.stage4:
            stage4_scaled_count = scale_explicit_gradients(
                shared, self.config.stage4_shared_stem_lr_scale
            )
        self._backward_control_state["last_update"] = int(plan.update)
        result = {
            "gradient_firewall_applied": firewall_applied,
            "task_projection_applied": projection_applied,
            "projection_triggered": projection_triggered,
            "protected_parameter_count": len(protected),
            "stage4_scaled_parameter_count": stage4_scaled_count,
        }
        audit = getattr(output, "audit", None)
        if isinstance(audit, dict):
            audit.update(result)
            pair_runtime = audit.get("pairbicad_runtime")
            if isinstance(pair_runtime, dict):
                gradient = pair_runtime.setdefault("gradient", {})
                gradient.update(
                    {
                        "checked": True,
                        "finite": all(
                            parameter.grad is None
                            or bool(torch.isfinite(parameter.grad).all())
                            for parameter in parameters
                        ),
                        "firewall_applied": firewall_applied,
                        "projection_applied": projection_applied,
                    }
                )
            components = audit.get("components")
            if isinstance(components, dict):
                if "gradient_firewall" in components and firewall_applied:
                    components["gradient_firewall"] = self._component(
                        called=True,
                        effective_count=len(shared),
                    )
                if "task_protected_gradient" in components and projection_applied:
                    components["task_protected_gradient"] = self._component(
                        called=True,
                        effective_count=len(protected),
                    )
        return result

    def compute_step(
        self,
        batch: BiCADXRBatch | Mapping[str, Any] | Sequence[Any],
        update: int,
        total_updates: int,
        *,
        epoch: int | None = None,
        source_loro_risk: Tensor | float | None = None,
        source_loro_window: bool | None = None,
    ) -> BiCADXRTrainOutput:
        """Compute one source-only BiCAD-XR step without performing optimizer.step."""

        stage = stage_for_update(update, total_updates)
        prepared = self._prepare_batch(batch)
        if epoch is None:
            epoch = prepared.epoch
        if epoch is not None and (
            not isinstance(epoch, int) or isinstance(epoch, bool) or epoch < 1
        ):
            raise ValueError("epoch must be a positive integer or None")
        if source_loro_risk is None:
            source_loro_risk = prepared.source_loro_risk
        if source_loro_risk is not None:
            source_loro_risk = _scalar(source_loro_risk)
        if source_loro_window is None:
            source_loro_window = prepared.source_loro_window
        if source_loro_window is not None and not isinstance(source_loro_window, bool):
            raise ValueError("source_loro_window must be a bool or None")
        model_output = self._forward(prepared)
        logits = self._output_tensor(model_output, "tx_logits", "logits")
        z_id = _finite_feature(
            model_output.get("z_id", model_output.get("identity_features")),
            "z_id",
            batch_size=prepared.x.size(0),
        )
        z_dom = _finite_feature(
            model_output.get("z_dom", model_output.get("domain_features")),
            "z_dom",
            batch_size=prepared.x.size(0),
        )
        if logits.ndim != 2 or logits.size(0) != prepared.x.size(0) or logits.size(1) != self.num_classes:
            raise ValueError("tx logits must have shape [batch,num_classes]")
        if not bool(torch.isfinite(logits).all()):
            raise ValueError("tx logits must contain only finite values")

        tx = prepared.tx
        labeled_mask = prepared.labeled_mask
        if labeled_mask is None and tx is not None:
            labeled_mask = torch.ones(tx.size(0), dtype=torch.bool, device=tx.device)
        labeled_indices = (
            torch.nonzero(labeled_mask, as_tuple=False).squeeze(1)
            if labeled_mask is not None
            else torch.empty(0, dtype=torch.long, device=z_id.device)
        )
        tx_labeled = (
            prepared.labeled_tx
            if prepared.labeled_tx is not None
            else None if tx is None else tx[labeled_indices]
        )
        receiver_labeled = prepared.receiver[labeled_indices]
        day_labeled = prepared.day[labeled_indices]
        channel_labeled = prepared.channel[labeled_indices]
        if (
            prepared.labeled_mask is not None
            and not bool(prepared.labeled_mask.all())
            and tx_labeled is not None
            and tx_labeled.numel() > 0
        ):
            conditioned_logits = self._tangent_logits(
                z_id[labeled_indices], tx_labeled
            )
            logits = logits.clone()
            logits[labeled_indices] = conditioned_logits

        zero = z_id.reshape(-1)[:1].sum() * 0.0
        total = zero
        components: dict[str, dict[str, Any]] = {}
        component_names = (
            "tx_ce",
            "satellite_tx_ce",
            "satellite_consistency",
            "domain_forward",
            "conditional_dann",
            "zdom_tx_adversary",
            "conditional_xcov",
            "xdc_cross_entropy",
            "xdc_knowledge_distillation",
            "paired_satellite",
            "pair_identity_hinge",
            "pair_prediction_js",
            "pair_vicreg",
            "pair_delta_identity_adversary",
            "pair_delta_channel_prediction",
            "pair_delta_stability",
            "pair_delta_channel_equivariance",
            "pair_delta_norm_hinge",
            "receiver_tangent",
            "margin_tail",
            "task_protected_gradient",
            "gradient_firewall",
        )
        for name in component_names:
            components[name] = self._component(skip_reason="not_evaluated")

        tx_loss = None
        if tx_labeled is None:
            components["tx_ce"] = self._component(skip_reason="missing_tx_labels")
        elif tx_labeled.numel() == 0:
            components["tx_ce"] = self._component(skip_reason="no_labeled_rows")
        else:
            tx_loss = F.cross_entropy(logits[labeled_indices], tx_labeled)
            total = total + tx_loss
            components["tx_ce"] = self._component(
                tx_loss,
                tx_loss,
                called=True,
                effective_count=int(tx_labeled.numel()),
            )

        pair = self._pair_payload(prepared, model_output)
        satellite_logits = pair.get("satellite_logits")
        pair_tx = pair.get("tx", pair.get("pair_tx", pair.get("satellite_tx")))
        pair_labeled_mask: Tensor | None = None
        if self.config.strict_pair_concat:
            if prepared.labeled_mask is None:
                raise ValueError("strict PairBiCAD concat requires labeled_mask")
            if prepared.x.size(0) % 2 != 0:
                raise ValueError("strict PairBiCAD concat batch must contain clean/satellite pairs")
            physical_count = prepared.x.size(0) // 2
            if prepared.labeled_mask.numel() != prepared.x.size(0):
                raise ValueError("strict PairBiCAD labeled_mask must match network batch")
            pair_labeled_mask = prepared.labeled_mask[:physical_count]
        if epoch is None:
            components["satellite_tx_ce"] = self._component(skip_reason="missing_epoch")
        elif epoch < self.config.concat_sat_start_epoch:
            components["satellite_tx_ce"] = self._component(
                skip_reason=(
                    "before_pair_start"
                    if self.config.strict_pair_concat
                    else "before_epoch80"
                )
            )
        elif not torch.is_tensor(satellite_logits):
            components["satellite_tx_ce"] = self._component(
                skip_reason="concat_satellite_logits_unavailable"
            )
        elif pair_tx is None:
            components["satellite_tx_ce"] = self._component(
                skip_reason="concat_satellite_tx_unavailable"
            )
        else:
            pair_tx = torch.as_tensor(pair_tx, device=satellite_logits.device, dtype=torch.long)
            if satellite_logits.ndim != 2 or satellite_logits.size(1) != self.num_classes:
                raise ValueError("concat satellite logits and TX labels must align")
            if pair_tx.ndim != 1:
                raise ValueError("concat satellite logits and TX labels must align")
            if pair_labeled_mask is not None:
                if pair_labeled_mask.numel() != satellite_logits.size(0):
                    raise ValueError("strict PairBiCAD pair labels must match physical pairs")
                labeled_pairs = int(pair_labeled_mask.sum().item())
                if pair_tx.numel() == satellite_logits.size(0):
                    pair_tx = pair_tx[pair_labeled_mask]
                elif pair_tx.numel() != labeled_pairs:
                    raise ValueError("concat satellite logits and TX labels must align")
                satellite_logits = satellite_logits[pair_labeled_mask]
            elif pair_tx.numel() != satellite_logits.size(0):
                raise ValueError("concat satellite logits and TX labels must align")
            satellite_ce = F.cross_entropy(satellite_logits, pair_tx)
            satellite_weight = (
                self._pair_satellite_weight(update, total_updates)
                if self.config.strict_pair_concat
                else self.config.lambda_sat_cls
            )
            weighted_satellite_ce = satellite_weight * satellite_ce
            total = total + weighted_satellite_ce
            components["satellite_tx_ce"] = self._component(
                satellite_ce,
                weighted_satellite_ce,
                called=True,
                effective_count=int(pair_tx.numel()),
            )
        components["satellite_consistency"] = self._component(
            skip_reason="lambda_sat_cons_zero"
        )

        peak_scale = 0.0 if stage is BiCADXRStage.stage0 else 1.0
        domain_scale = (
            self.config.stage4_domain_scale
            if stage is BiCADXRStage.stage4
            else peak_scale
        )
        if self._dynamic_grl_controller is not None and stage is not BiCADXRStage.stage0:
            dynamic_doses = self._dynamic_grl_controller.doses
            grl_identity = dynamic_doses["identity"]
            grl_zdom = dynamic_doses["zdom"]
        else:
            grl_identity = float(domain_scale)
            grl_zdom = float(domain_scale)
        dynamic_grl_audit: dict[str, Any] | None = None
        if self._dynamic_grl_controller is not None:
            dynamic_grl_audit = {
                "enabled": True,
                "updated": False,
                "feedback": None,
                "doses": {
                    "identity": float(grl_identity),
                    "zdom": float(grl_zdom),
                },
                "next_doses": self._dynamic_grl_controller.doses,
                "identity_dose_bounds": self._dynamic_grl_controller.identity_dose_bounds,
                "zdom_dose_bounds": self._dynamic_grl_controller.zdom_dose_bounds,
                "conditional_ratio_bounds": self._dynamic_grl_controller.conditional_ratio_bounds,
                "zdom_tx_ratio_bounds": self._dynamic_grl_controller.zdom_tx_ratio_bounds,
                "conditional_target_ratio": self._dynamic_grl_controller.conditional_target_ratio,
                "zdom_tx_target_ratio": self._dynamic_grl_controller.zdom_tx_target_ratio,
            }
        factorized_output: dict[str, Tensor] = {}
        factorized_domain_features = z_dom
        factorized_values = None
        if self.factorized_projector is not None:
            factorized_values = self.factorized_projector(z_dom)
            factorized_domain_features = (
                factorized_values.z_r
                + factorized_values.z_d
                + factorized_values.z_c
            ) / 3.0
        domain_loss = zero
        conditional_loss = zero
        zdom_tx_loss = zero
        if self.factorized_heads is not None:
            need_conditional = bool(
                self.config.conditional_cdan or self.config.zdom_tx_adversary
            )
            if need_conditional and tx_labeled is not None and tx_labeled.numel() > 0:
                factorized_output = self.factorized_heads(
                    z_id[labeled_indices],
                    factorized_domain_features[labeled_indices],
                    tx_labeled,
                    grl_identity=grl_identity if self.config.conditional_cdan else 0.0,
                    grl_tx=grl_zdom if self.config.zdom_tx_adversary else 0.0,
                )
            if self.config.factorized_domains:
                domain_terms: list[Tensor] = []
                domain_inputs = (
                    (
                        "dom_receiver",
                        prepared.receiver,
                        None if factorized_values is None else factorized_values.z_r,
                    ),
                    (
                        "dom_day",
                        prepared.day,
                        None if factorized_values is None else factorized_values.z_d,
                    ),
                    (
                        "dom_channel",
                        prepared.channel,
                        None if factorized_values is None else factorized_values.z_c,
                    ),
                )
                for key, labels, factor in domain_inputs:
                    prediction = getattr(self.factorized_heads, key)(
                        z_dom if factor is None else factor
                    )
                    domain_terms.append(F.cross_entropy(prediction, labels))
                if domain_terms:
                    domain_loss = torch.stack(domain_terms).mean()
                    total = total + domain_loss
                    components["domain_forward"] = self._component(
                        domain_loss,
                        domain_loss,
                        called=True,
                        effective_count=int(prepared.x.size(0)),
                    )
            if self.config.conditional_cdan and factorized_output:
                cond_terms = [
                    F.cross_entropy(factorized_output[key], labels)
                    for key, labels in (
                        ("id_receiver", receiver_labeled),
                        ("id_day", day_labeled),
                        ("id_channel", channel_labeled),
                    )
                ]
                conditional_loss = torch.stack(cond_terms).mean() * grl_identity
                total = total + conditional_loss
                components["conditional_dann"] = self._component(
                    conditional_loss / max(grl_identity, 1e-12)
                    if grl_identity > 0.0
                    else zero,
                    conditional_loss,
                    called=True,
                    effective_count=int(tx_labeled.numel()) if tx_labeled is not None else 0,
                )
            elif self.config.conditional_cdan:
                components["conditional_dann"] = self._component(
                    skip_reason="missing_tx_labels"
                    if tx_labeled is None
                    else "no_labeled_rows"
                )
            if self.config.zdom_tx_adversary and factorized_output:
                zdom_tx_loss = F.cross_entropy(factorized_output["dom_tx"], tx_labeled) * grl_zdom
                total = total + zdom_tx_loss
                components["zdom_tx_adversary"] = self._component(
                    zdom_tx_loss / max(grl_zdom, 1e-12)
                    if grl_zdom > 0.0
                    else zero,
                    zdom_tx_loss,
                    called=True,
                    effective_count=int(tx_labeled.numel()) if tx_labeled is not None else 0,
                )
            elif self.config.zdom_tx_adversary:
                components["zdom_tx_adversary"] = self._component(
                    skip_reason="missing_tx_labels"
                    if tx_labeled is None
                    else "no_labeled_rows"
                )
        else:
            legacy_domain = model_output.get("dom_logits")
            if torch.is_tensor(legacy_domain) and legacy_domain.ndim == 2:
                domain_loss = F.cross_entropy(legacy_domain, prepared.receiver)
                total = total + domain_loss
                components["domain_forward"] = self._component(
                    domain_loss,
                    domain_loss,
                    called=True,
                    effective_count=int(prepared.x.size(0)),
                )

        if (
            self.config.conditional_xcov
            and stage is not BiCADXRStage.stage0
            and tx_labeled is not None
            and tx_labeled.numel() > 0
        ):
            xcov_loss = conditional_cross_covariance(
                z_id[labeled_indices], z_dom[labeled_indices], tx_labeled
            )
            weighted_xcov = self.config.lambda_cond_xcov * xcov_loss
            total = total + weighted_xcov
            effective_groups = sum(
                int((tx_labeled == class_id).sum().item()) >= 2
                for class_id in torch.unique(tx_labeled)
            )
            components["conditional_xcov"] = self._component(
                xcov_loss,
                weighted_xcov,
                called=True,
                effective_count=effective_groups,
            )
        elif self.config.conditional_xcov:
            components["conditional_xcov"] = self._component(
                skip_reason=(
                    "missing_tx_labels"
                    if tx_labeled is None
                    else "no_labeled_rows"
                    if tx_labeled.numel() == 0
                    else "stage0_disabled"
                )
            )
        else:
            components["conditional_xcov"] = self._component(skip_reason="candidate_disabled")

        xdc_output: XDCLossOutput | None = None
        xdc_selection: _StructuredEpisodeSelection | None = None
        xdc_tx: Tensor | None = None
        xdc_receiver: Tensor | None = None
        xdc_called = False
        xdc_reason: str | None = None
        if not self.config.sparse_xdc:
            xdc_reason = "candidate_disabled"
        elif stage.value in {"stage0", "stage1"}:
            xdc_reason = "stage_before_stage2"
        elif update % self.config.xdc_interval != 0:
            xdc_reason = "interval"
        elif tx_labeled is None or tx_labeled.numel() == 0:
            xdc_reason = "missing_tx_labels" if tx_labeled is None else "no_labeled_rows"
        else:
            xdc_selection = self._structured_episode(prepared, labeled_indices)
            local = xdc_selection.batch_indices
            if local.numel() == 0:
                xdc_reason = "no_valid_structured_cells"
            else:
                xdc_tx = xdc_selection.global_tx.to(device=z_id.device)
                xdc_receiver = xdc_selection.global_receiver.to(device=z_id.device)
                xdc_output = xdc_losses(
                    z_id[local],
                    xdc_tx,
                    xdc_receiver,
                    logits[local],
                    num_classes=self.num_classes,
                    temperature=self.config.xdc_temperature,
                    ridge=self.config.xdc_ridge,
                    min_support_accuracy=self.config.xdc_min_support_accuracy,
                    num_receivers=self.num_receivers,
                    kd_weight=1.0 if self.config.xdc_kd else 0.0,
                    physical_indices=xdc_selection.episode.indices,
                )
                xdc_called = True
                xdc_reason = xdc_output.skip_reason
                total = total + xdc_output.total
                donor_count = int(xdc_output.donor_bank.valid_receivers.numel())
                components["xdc_cross_entropy"] = self._component(
                    xdc_output.xdc_cross_entropy,
                    xdc_output.xdc_cross_entropy,
                    called=True,
                    effective_count=donor_count,
                    skip_reason=xdc_reason,
                )
                components["xdc_knowledge_distillation"] = self._component(
                    xdc_output.knowledge_distillation,
                    xdc_output.knowledge_distillation if self.config.xdc_kd else zero,
                    called=bool(self.config.xdc_kd),
                    effective_count=donor_count if self.config.xdc_kd else 0,
                    skip_reason=(
                        xdc_reason
                        if xdc_reason is not None
                        else "candidate_disabled"
                        if not self.config.xdc_kd
                        else None
                    ),
                )
        if not xdc_called:
            components["xdc_cross_entropy"] = self._component(skip_reason=xdc_reason)
            components["xdc_knowledge_distillation"] = self._component(
                skip_reason=xdc_reason if xdc_reason is not None else "not_called"
            )

        pair_called = False
        pair_reason: str | None = None
        pair_loss = zero
        pair_adversarial_loss = zero
        pair_identity_gradient_audit: dict[str, Any] | None = None
        pair_enabled = bool(
            self.config.paired_satellite
            and self.config.candidate_id.strip().upper() == "E3"
        )
        if self.config.strict_pair_concat:
            required = ("clean_z_id", "satellite_z_id")
            if not all(key in pair for key in required):
                pair_reason = "concat_pair_unavailable"
            else:
                clean_pair_id = _finite_feature(pair["clean_z_id"], "clean_z_id")
                satellite_pair_id = _finite_feature(
                    pair["satellite_z_id"], "satellite_z_id"
                )
                if clean_pair_id.shape != satellite_pair_id.shape:
                    raise ValueError("clean and satellite pair identity features must align")
                pair_count = int(clean_pair_id.size(0))
                if pair_labeled_mask is None or pair_labeled_mask.numel() != pair_count:
                    raise ValueError("strict PairBiCAD pair mask must match physical pairs")
                unlabeled_pair_mask = ~pair_labeled_mask
                if self.pair_projector is not None:
                    projected_clean_id = self.pair_projector(clean_pair_id)
                    projected_satellite_id = self.pair_projector(satellite_pair_id)
                else:
                    projected_clean_id = clean_pair_id
                    projected_satellite_id = satellite_pair_id

                if self.config.pair_identity or self.config.pair_vicreg:
                    candidate_key = self.config.candidate_id.strip().upper()
                    identity_epsilon = _CV2_PAIR_IDENTITY_EPSILON
                    identity_weight = (
                        _CV2_PAIR_IDENTITY_WEIGHT
                        if candidate_key in _CV2_PAIR_IDENTITY_CANDIDATES
                        else _LEGACY_PAIR_IDENTITY_WEIGHT
                    )
                    identity_raw = pair_identity_hinge(
                        projected_clean_id,
                        projected_satellite_id,
                        epsilon=identity_epsilon,
                    )
                    effective_identity_weight = identity_weight
                    if candidate_key in _CV2_PAIR_IDENTITY_CANDIDATES:
                        task_vector: Tensor | None = None
                        pair_vector: Tensor | None = None
                        parameters = self.optimizer_parameters()
                        if tx_loss is not None and tx_loss.requires_grad:
                            task_gradients = torch.autograd.grad(
                                tx_loss,
                                parameters,
                                retain_graph=True,
                                allow_unused=True,
                            )
                            task_vector = _finite_gradient_vector(
                                task_gradients, "TX task gradients"
                            )
                        if identity_raw.requires_grad:
                            pair_gradients = torch.autograd.grad(
                                identity_raw,
                                parameters,
                                retain_graph=True,
                                allow_unused=True,
                            )
                            pair_vector = _finite_gradient_vector(
                                pair_gradients, "pair identity gradients"
                            )
                        if task_vector is None:
                            task_vector = identity_raw.detach().new_zeros(1)
                        if pair_vector is None:
                            pair_vector = identity_raw.detach().new_zeros(1)
                        gradient_audit = measure_bounded_gradient_ratio(
                            task_vector,
                            pair_vector,
                            initial_weight=identity_weight,
                            max_ratio=0.05,
                        )
                        effective_identity_weight = gradient_audit.effective_weight
                        pair_identity_gradient_audit = asdict(gradient_audit)
                        pair_identity_gradient_audit.update(
                            {
                                "raw_ratio": gradient_audit.raw_ratio,
                                "effective_ratio": gradient_audit.effective_ratio,
                                "raw_gradient_ratio": gradient_audit.raw_ratio,
                                "effective_gradient_ratio": gradient_audit.effective_ratio,
                                "gradient_scale": gradient_audit.scale,
                            }
                        )
                    identity_weighted = effective_identity_weight * identity_raw
                    total = total + identity_weighted
                    components["pair_identity_hinge"] = self._component(
                        identity_raw,
                        identity_weighted,
                        called=True,
                        effective_count=pair_count,
                    )
                    components["pair_identity_hinge"].update(
                        {
                            "epsilon": identity_epsilon,
                            "weight": identity_weight,
                            "effective_weight": effective_identity_weight,
                        }
                    )
                    if pair_identity_gradient_audit is not None:
                        components["pair_identity_hinge"].update(
                            pair_identity_gradient_audit
                        )
                    pair_called = True

                if self.config.pair_vicreg:
                    vicreg = vicreg_pair_loss(
                        projected_clean_id,
                        projected_satellite_id,
                        gamma=1.0,
                    )
                    vicreg_weighted = 0.03 * vicreg["total"]
                    total = total + vicreg_weighted
                    components["pair_vicreg"] = self._component(
                        vicreg["total"],
                        vicreg_weighted,
                        called=True,
                        effective_count=pair_count,
                    )
                    pair_called = True

                if self.config.pair_identity or self.config.pair_vicreg:
                    clean_pair_logits = pair.get("clean_logits")
                    satellite_pair_logits = pair.get("satellite_logits")
                    if (
                        torch.is_tensor(clean_pair_logits)
                        and torch.is_tensor(satellite_pair_logits)
                        and clean_pair_logits.ndim == 2
                        and satellite_pair_logits.ndim == 2
                        and clean_pair_logits.shape == satellite_pair_logits.shape
                        and clean_pair_logits.size(0) == pair_count
                        and bool(unlabeled_pair_mask.any())
                    ):
                        prediction_js = self._prediction_js(
                            clean_pair_logits[unlabeled_pair_mask],
                            satellite_pair_logits[unlabeled_pair_mask],
                        )
                        prediction_weighted = 0.05 * prediction_js
                        total = total + prediction_weighted
                        components["pair_prediction_js"] = self._component(
                            prediction_js,
                            prediction_weighted,
                            called=True,
                            effective_count=int(unlabeled_pair_mask.sum().item()),
                        )
                    else:
                        components["pair_prediction_js"] = self._component(
                            skip_reason="unlabeled_pair_logits_unavailable"
                        )

                if self.config.pair_delta:
                    clean_pair_dom = pair.get("clean_z_dom")
                    satellite_pair_dom = pair.get("satellite_z_dom")
                    if not (
                        torch.is_tensor(clean_pair_dom)
                        and torch.is_tensor(satellite_pair_dom)
                    ):
                        clean_pair_dom = z_dom[:pair_count]
                        satellite_pair_dom = z_dom[pair_count : 2 * pair_count]
                    clean_pair_dom = _finite_feature(
                        clean_pair_dom, "clean_z_dom", batch_size=pair_count
                    )
                    satellite_pair_dom = _finite_feature(
                        satellite_pair_dom, "satellite_z_dom", batch_size=pair_count
                    )
                    if factorized_values is not None:
                        clean_pair_c = factorized_values.z_c[:pair_count]
                        satellite_pair_c = factorized_values.z_c[
                            pair_count : 2 * pair_count
                        ]
                    else:
                        clean_pair_c = clean_pair_dom
                        satellite_pair_c = satellite_pair_dom
                    rho_adv = self._pair_adversarial_dose(stage)
                    delta = pair_delta_objectives(
                        projected_clean_id,
                        projected_satellite_id,
                        clean_pair_c,
                        satellite_pair_c,
                        torch.ones(pair_count, dtype=torch.long, device=z_id.device),
                        epsilon=0.05,
                        delta_radius=0.25,
                        grl_scale=rho_adv,
                        include_delta_norm_hinge=True,
                    )
                    delta_weights = {
                        "identity_channel_adversary": 0.08,
                        "channel_prediction": 0.15,
                        "pair_stability": 0.05,
                        "channel_equivariance": 0.05,
                        "delta_norm_hinge": 0.05,
                    }
                    delta_names = {
                        "identity_channel_adversary": "pair_delta_identity_adversary",
                        "channel_prediction": "pair_delta_channel_prediction",
                        "pair_stability": "pair_delta_stability",
                        "channel_equivariance": "pair_delta_channel_equivariance",
                        "delta_norm_hinge": "pair_delta_norm_hinge",
                    }
                    for key, raw_value in delta.items():
                        weighted_value = delta_weights[key] * raw_value
                        total = total + weighted_value
                        components[delta_names[key]] = self._component(
                            raw_value,
                            weighted_value,
                            called=True,
                            effective_count=pair_count,
                        )
                        if key == "identity_channel_adversary":
                            pair_adversarial_loss = weighted_value
                    pair_called = True
        elif not pair_enabled:
            pair_reason = "candidate_disabled"
        elif stage.value in {"stage0", "stage1"}:
            pair_reason = "stage_before_stage2"
        elif update % self.config.pair_interval != 0:
            pair_reason = "interval"
        else:
            required = ("clean_z_id", "satellite_z_id")
            if not all(key in pair for key in required):
                pair_reason = "concat_pair_unavailable"
            else:
                pair_loss = paired_satellite_loss(
                    pair["clean_z_id"],
                    pair["satellite_z_id"],
                    pair.get("clean_logits"),
                    pair.get("satellite_logits"),
                )
                total = total + pair_loss
                pair_called = True
                components["paired_satellite"] = self._component(
                    pair_loss,
                    pair_loss,
                    called=True,
                    effective_count=int(pair["clean_z_id"].size(0)),
                )
        if self.config.strict_pair_concat:
            for name in (
                "pair_identity_hinge",
                "pair_prediction_js",
                "pair_vicreg",
                "pair_delta_identity_adversary",
                "pair_delta_channel_prediction",
                "pair_delta_stability",
                "pair_delta_channel_equivariance",
                "pair_delta_norm_hinge",
            ):
                if components[name]["skip_reason"] == "not_evaluated":
                    components[name] = self._component(skip_reason="candidate_disabled")
            if not pair_called and pair_reason is None:
                components["paired_satellite"] = self._component(
                    skip_reason="candidate_pair_contract"
                )
        if not pair_called and not self.config.strict_pair_concat:
            components["paired_satellite"] = self._component(skip_reason=pair_reason)
        elif self.config.strict_pair_concat and pair_reason is not None:
            for name in (
                "pair_identity_hinge",
                "pair_prediction_js",
                "pair_vicreg",
                "pair_delta_identity_adversary",
                "pair_delta_channel_prediction",
                "pair_delta_stability",
                "pair_delta_channel_equivariance",
                "pair_delta_norm_hinge",
            ):
                components[name] = self._component(skip_reason=pair_reason)
            components["paired_satellite"] = self._component(
                skip_reason="candidate_pair_contract"
            )

        tangent_called = False
        tangent_reason: str | None = None
        tangent_loss = zero
        tangent_logits: Tensor | None = None
        if self.config.receiver_tangent == "off":
            tangent_reason = "candidate_disabled"
        elif stage.value in {"stage0", "stage1", "stage2"}:
            tangent_reason = "stage_before_stage3"
        elif tx_labeled is None or tx_labeled.numel() == 0:
            tangent_reason = "missing_tx_labels" if tx_labeled is None else "no_labeled_rows"
        else:
            assert self.tangent_bank is not None
            self.tangent_bank.update(z_id[labeled_indices], tx_labeled, receiver_labeled)
            coefficients = self.tangent_bank.coefficients(z_id[labeled_indices], receiver_labeled)
            if self.config.receiver_tangent == "factual":
                tangent_features = factual_tangent(
                    self.tangent_bank,
                    z_id[labeled_indices],
                    receiver_labeled,
                    coefficients,
                )
            else:
                factual_features = factual_tangent(
                    self.tangent_bank,
                    z_id[labeled_indices],
                    receiver_labeled,
                    coefficients,
                )
                factual_logits = self._tangent_logits(factual_features, tx_labeled)
                attack_loss = F.softplus(
                    -classification_margin(factual_logits, tx_labeled)
                ).mean()
                direction = one_step_tangent_worst_direction(
                    attack_loss,
                    coefficients,
                    radius=0.1,
                )
                tangent_features = factual_tangent(
                    self.tangent_bank,
                    z_id[labeled_indices],
                    receiver_labeled,
                    coefficients + direction,
                )
            tangent_logits = self._tangent_logits(tangent_features, tx_labeled)
            tangent_loss = F.cross_entropy(tangent_logits, tx_labeled)
            total = total + tangent_loss
            tangent_called = True
            components["receiver_tangent"] = self._component(
                tangent_loss,
                tangent_loss,
                called=True,
                effective_count=int(tx_labeled.numel()),
            )
        if not tangent_called:
            components["receiver_tangent"] = self._component(skip_reason=tangent_reason)

        tail_called = False
        tail_reason: str | None = None
        tail_mode: str | None = None
        hard_group_cap: float | None = None
        hard_group_weight_mass: float | None = None
        hard_group_weights: list[float] | None = None
        hard_group_indices: list[int] | None = None
        if not self.config.margin_tail:
            tail_reason = "candidate_disabled"
        elif stage.value in {"stage0", "stage1", "stage2"}:
            tail_reason = "stage_before_stage3"
        elif tx_labeled is None or tx_labeled.numel() == 0:
            tail_reason = "missing_tx_labels" if tx_labeled is None else "no_labeled_rows"
        else:
            assert self._tail_emas is not None
            tx_margins = classification_margin(logits[labeled_indices], tx_labeled)
            candidate_key = self.config.candidate_id.strip().upper()
            if candidate_key in _CV2_MARGIN_REX_CVAR_CANDIDATES:
                tx_groups = _combine_group_ids(
                    tx_labeled, receiver_labeled, channel_labeled
                )
                group_risks = margin_group_risks(
                    tx_margins,
                    tx_groups,
                    tail_fraction=_CV2_MARGIN_TAIL_FRACTION,
                )
                hard_weights = bounded_hard_group_weights(
                    group_risks,
                    hard_fraction=_CV2_HARD_GROUP_FRACTION,
                    max_hard_fraction=_CV2_HARD_GROUP_CAP,
                )
                hard_count = min(
                    group_risks.numel(),
                    max(1, math.ceil(_CV2_HARD_GROUP_FRACTION * group_risks.numel())),
                )
                hard_order = torch.argsort(
                    group_risks.detach(), descending=True
                )
                hard_group_indices = [
                    int(value)
                    for value in hard_order[:hard_count].detach().cpu().tolist()
                ]
                hard_group_weight_mass = float(
                    hard_weights[hard_order[:hard_count]].sum().detach().item()
                )
                if not math.isfinite(hard_group_weight_mass):
                    raise ValueError("hard-group weight mass must be finite")
                hard_group_weights = [
                    float(value)
                    for value in hard_weights.detach().cpu().tolist()
                ]
                tail_loss = margin_rex_cvar_loss(
                    tx_margins,
                    tx_groups,
                    tail_fraction=_CV2_MARGIN_TAIL_FRACTION,
                    lambda_rex=_CV2_MARGIN_REX_LAMBDA,
                    lambda_cvar=_CV2_MARGIN_CVAR_LAMBDA,
                    hard_fraction=_CV2_HARD_GROUP_FRACTION,
                    max_hard_fraction=_CV2_HARD_GROUP_CAP,
                )
                total = total + tail_loss
                tail_called = True
                tail_mode = "margin_rex_cvar"
                hard_group_cap = _CV2_HARD_GROUP_CAP
                components["margin_tail"] = self._component(
                    tail_loss,
                    tail_loss,
                    called=True,
                    effective_count=int(torch.unique(tx_groups).numel()),
                )
                components["margin_tail"].update(
                    {
                        "hard_group_weights": hard_group_weights,
                        "hard_group_indices": hard_group_indices,
                        "hard_group_weight_mass": hard_group_weight_mass,
                        "hard_group_cap": _CV2_HARD_GROUP_CAP,
                    }
                )
            else:
                tx_groups = _combine_group_ids(tx_labeled, receiver_labeled)
                tx_risk = group_margin_cvar(
                    tx_margins,
                    tx_groups,
                    tail_fraction=self.config.margin_tail_cvar_fraction,
                    ema=self._tail_emas[0],
                )
                if (
                    xdc_output is not None
                    and xdc_output.skip_reason is None
                    and xdc_tx is not None
                    and xdc_receiver is not None
                ):
                    xdc_groups = _combine_group_ids(xdc_tx, xdc_receiver)
                    xdc_risk = group_margin_cvar(
                        classification_margin(xdc_output.ensemble_logits, xdc_tx),
                        xdc_groups,
                        tail_fraction=self.config.margin_tail_cvar_fraction,
                        ema=self._tail_emas[1],
                    )
                else:
                    xdc_risk = zero
                if tangent_logits is not None:
                    tangent_risk = group_margin_cvar(
                        classification_margin(tangent_logits, tx_labeled),
                        tx_groups,
                        tail_fraction=self.config.margin_tail_cvar_fraction,
                        ema=self._tail_emas[2],
                    )
                else:
                    tangent_risk = zero
                tail_loss = apply_margin_tail(
                    zero,
                    tx_risk,
                    xdc_risk,
                    tangent_risk,
                    weights=self.config.margin_tail_weights,
                )
                total = total + tail_loss
                tail_called = True
                tail_mode = "legacy_margin_tail"
                components["margin_tail"] = self._component(
                    tail_loss,
                    tail_loss,
                    called=True,
                    effective_count=int(torch.unique(tx_groups).numel()),
                )
        if not tail_called:
            components["margin_tail"] = self._component(skip_reason=tail_reason)

        adversarial_loss = conditional_loss + zdom_tx_loss + pair_adversarial_loss
        if (
            dynamic_grl_audit is not None
            and stage is not BiCADXRStage.stage0
            and tx_loss is not None
            and tx_loss.requires_grad
            and adversarial_loss.requires_grad
            and factorized_output
        ):
            discriminator_accuracy = self._discriminator_accuracy(
                factorized_output,
                {
                    **(
                        {
                            "id_receiver": receiver_labeled,
                            "id_day": day_labeled,
                            "id_channel": channel_labeled,
                        }
                        if self.config.conditional_cdan
                        else {}
                    ),
                    **(
                        {"dom_tx": tx_labeled}
                        if self.config.zdom_tx_adversary and tx_labeled is not None
                        else {}
                    ),
                },
            )
            encoder_parameters = tuple(self.adversarial_parameter_groups()["encoder"])
            task_gradients = torch.autograd.grad(
                tx_loss,
                encoder_parameters,
                retain_graph=True,
                allow_unused=True,
            )
            adversarial_gradients = torch.autograd.grad(
                adversarial_loss,
                encoder_parameters,
                retain_graph=True,
                allow_unused=True,
            )
            task_vector = _finite_gradient_vector(
                task_gradients,
                "TX task gradients",
                parameters=encoder_parameters,
            )
            adversarial_vector = _finite_gradient_vector(
                adversarial_gradients,
                "adversarial gradients",
                parameters=encoder_parameters,
            )
            tx_margin_value = _scalar(
                classification_margin(logits[labeled_indices], tx_labeled).mean()
            )
            if (
                discriminator_accuracy is not None
                and task_vector is not None
                and adversarial_vector is not None
            ):
                task_norm = float(torch.linalg.vector_norm(task_vector).item())
                adversarial_norm = float(
                    torch.linalg.vector_norm(adversarial_vector).item()
                )
                denominator = max(task_norm, 1.0e-12)
                adversarial_gradient_ratio = adversarial_norm / denominator
                cosine = float(
                    torch.sum(task_vector * adversarial_vector).item()
                    / max(task_norm * max(adversarial_norm, 1.0e-12), 1.0e-12)
                )
                conflict_signal = max(0.0, min(1.0, -cosine))
                if not (
                    math.isfinite(adversarial_gradient_ratio)
                    and math.isfinite(conflict_signal)
                ):
                    raise ValueError("dynamic GRL gradient feedback must be finite")
                feedback = {
                    "discriminator_accuracy": discriminator_accuracy,
                    "tx_margin": tx_margin_value,
                    "adversarial_gradient_ratio": adversarial_gradient_ratio,
                    "conflict_signal": conflict_signal,
                }
                next_doses = self._dynamic_grl_controller.update(**feedback)
                dynamic_grl_audit.update(
                    {
                        "updated": True,
                        "feedback": feedback,
                        "next_doses": next_doses,
                    }
                )
        projection_called = False
        projection_triggered = False
        projection_reason: str | None = None
        if not self.config.task_protected_gradient:
            projection_reason = "candidate_disabled"
        elif stage is BiCADXRStage.stage0:
            projection_reason = "stage0_disabled"
        elif update % 4 != 0:
            projection_reason = "interval"
        elif tx_loss is None:
            projection_reason = "missing_tx_labels"
        else:
            if not adversarial_loss.requires_grad:
                projection_reason = "no_adversarial_gradient"
            else:
                projection_reason = "awaiting_backward_controls"
        components["task_protected_gradient"] = self._component(
            0.0,
            0.0,
            called=projection_called,
            effective_count=int(z_id.numel()) if projection_called else 0,
            skip_reason=projection_reason,
        )

        swad_updated = False
        swad_candidate = bool(
            self.config.swad and self.config.candidate_id.strip().upper() == "F3"
        )
        if (
            swad_candidate
            and stage is BiCADXRStage.stage4
            and source_loro_risk is not None
            and source_loro_window is True
        ):
            swad_updated = self._update_swad(source_loro_risk)
        swad_reason = None if swad_updated else (
            "candidate_disabled"
            if not swad_candidate
            else "stage_before_stage4"
            if stage is not BiCADXRStage.stage4
            else "missing_source_loro_risk"
            if source_loro_risk is None
            else "not_low_risk_window"
        )
        firewall_scheduled = bool(
            self.config.gradient_firewall
            and stage is not BiCADXRStage.stage0
            and domain_loss.requires_grad
        )
        components["gradient_firewall"] = self._component(
            0.0,
            0.0,
            called=False,
            effective_count=0,
            skip_reason=(
                "awaiting_backward_controls"
                if firewall_scheduled
                else "candidate_disabled"
                if not self.config.gradient_firewall
                else "stage0_disabled"
            ),
        )

        backward_plan = BiCADXRBackwardPlan(
            total=total,
            domain_forward=domain_loss,
            adversarial=adversarial_loss,
            conditional_adversarial=conditional_loss,
            zdom_tx_adversarial=zdom_tx_loss,
            task_reference=zero if tx_loss is None else tx_loss,
            stage=stage,
            update=int(update),
            firewall_enabled=firewall_scheduled,
            projection_enabled=projection_reason == "awaiting_backward_controls",
        )
        runtime = self.checkpoint_runtime(update, total_updates, stage=stage)
        pair_runtime = None
        if self.config.strict_pair_concat:
            assert pair_labeled_mask is not None
            physical_count = int(prepared.x.size(0) // 2)
            pair_runtime = {
                "runtime_version": 1,
                "candidate_id": self.config.candidate_id,
                "active": True,
                "components": {
                    name: dict(components[name])
                    for name in (
                        "satellite_tx_ce",
                        "pair_identity_hinge",
                        "pair_prediction_js",
                        "pair_vicreg",
                        "pair_delta_identity_adversary",
                        "pair_delta_channel_prediction",
                        "pair_delta_stability",
                        "pair_delta_channel_equivariance",
                        "pair_delta_norm_hinge",
                    )
                },
                "effective_counts": {
                    "physical": physical_count,
                    "network": int(prepared.x.size(0)),
                    "labeled": int(tx_labeled.numel()) if tx_labeled is not None else 0,
                    "unlabeled": physical_count - int(pair_labeled_mask.sum().item()),
                    "pair": physical_count,
                },
                "skip_reasons": {
                    name: components[name]["skip_reason"]
                    for name in (
                        "satellite_tx_ce",
                        "pair_identity_hinge",
                        "pair_prediction_js",
                        "pair_vicreg",
                        "pair_delta_identity_adversary",
                        "pair_delta_channel_prediction",
                        "pair_delta_stability",
                        "pair_delta_channel_equivariance",
                        "pair_delta_norm_hinge",
                    )
                },
                "one_forward_count": 1,
                "model_forward_count": 1,
                "extra_forward_count": 0,
                "finite": {
                    "inputs": bool(torch.isfinite(prepared.x).all()),
                    "features": True,
                    "loss": bool(torch.isfinite(total).all()),
                },
                "gradient": {
                    "checked": False,
                    "finite": None,
                    "rho_adv": self._pair_adversarial_dose(stage),
                    "pair_identity": pair_identity_gradient_audit,
                },
            }
            self._pairbicad_runtime_state = pair_runtime
            runtime["pairbicad_runtime"] = pair_runtime
        episode_indices = (
            []
            if xdc_selection is None
            else [int(value) for value in xdc_selection.batch_indices.detach().cpu().tolist()]
        )
        episode_tx = (
            []
            if xdc_selection is None
            else [int(value) for value in xdc_selection.global_tx.detach().cpu().tolist()]
        )
        episode_receiver = (
            []
            if xdc_selection is None
            else [
                int(value)
                for value in xdc_selection.global_receiver.detach().cpu().tolist()
            ]
        )
        episode_local_tx = (
            []
            if xdc_selection is None
            else [int(value) for value in xdc_selection.episode.tx.tolist()]
        )
        episode_local_receiver = (
            []
            if xdc_selection is None
            else [int(value) for value in xdc_selection.episode.receiver.tolist()]
        )
        audit: dict[str, Any] = {
            "stage": stage.name,
            "update": int(update),
            "total_updates": int(total_updates),
            "candidate_id": self.config.candidate_id,
            "epoch": epoch,
            "grl_identity": float(grl_identity if stage is not BiCADXRStage.stage0 else 0.0),
            "grl_tx": float(grl_zdom if stage is not BiCADXRStage.stage0 else 0.0),
            "domain_dann_scale": float(domain_scale),
            "domain_scale": float(domain_scale),
            "shared_stem_lr_scale": float(
                self.config.stage4_shared_stem_lr_scale
                if stage is BiCADXRStage.stage4
                else 1.0
            ),
            "xdc_called": xdc_called,
            "xdc_kd_called": bool(xdc_called and self.config.xdc_kd),
            "xdc_skip_reason": xdc_reason,
            "xdc_effective_donors": 0
            if xdc_output is None
            else int(xdc_output.donor_bank.valid_receivers.numel()),
            "xdc_donor_query_matrix": None
            if xdc_output is None
            else xdc_output.donor_query_matrix,
            "xdc_episode_batch_indices": episode_indices,
            "xdc_episode_tx": episode_tx,
            "xdc_episode_receiver": episode_receiver,
            "xdc_episode_local_tx": episode_local_tx,
            "xdc_episode_local_receiver": episode_local_receiver,
            "xdc_tail_query_tx": episode_tx,
            "pair_called": pair_called,
            "pair_source": "concat_satellite" if pair_called else None,
            "pair_skip_reason": pair_reason,
            "tangent_called": tangent_called,
            "tangent_mode": self.config.receiver_tangent,
            "tangent_skip_reason": tangent_reason,
            "tail_called": tail_called,
            "tail_skip_reason": tail_reason,
            "margin_tail_mode": tail_mode,
            "task_protected_gradient_called": projection_called,
            "projection_triggered": projection_triggered,
            "projection_skip_reason": projection_reason,
            "swad_updated": swad_updated,
            "swad_skip_reason": swad_reason,
            "source_loro_risk": source_loro_risk,
            "source_loro_window": source_loro_window,
            "extra_forward_count": 0,
            "model_forward_count": 1,
            "backbone_forward_count": 1,
            "components": components,
            "checkpoint_runtime": runtime,
        }
        if dynamic_grl_audit is not None:
            audit["dynamic_grl"] = dynamic_grl_audit
        if pair_runtime is not None:
            audit["pairbicad_runtime"] = pair_runtime
            pair_runtime["backbone_forward_count"] = 1
        if pair_identity_gradient_audit is not None:
            audit["pair_identity_gradient"] = pair_identity_gradient_audit
        if hard_group_cap is not None:
            audit["hard_group_cap"] = hard_group_cap
        if hard_group_weight_mass is not None:
            audit["hard_group_weight_mass"] = hard_group_weight_mass
        if hard_group_weights is not None:
            audit["hard_group_weights"] = hard_group_weights
        if hard_group_indices is not None:
            audit["hard_group_indices"] = hard_group_indices
        audit["raw_losses"] = {name: value["raw"] for name, value in components.items()}
        audit["weighted_losses"] = {
            name: value["weighted"] for name, value in components.items()
        }
        audit["component_calls"] = {name: value["called"] for name, value in components.items()}
        audit["effective_counts"] = {
            name: value["effective_count"] for name, value in components.items()
        }
        audit["skip_reasons"] = {
            name: value["skip_reason"] for name, value in components.items()
        }
        if not torch.isfinite(total).all():
            raise ValueError("BiCAD-XR total loss must remain finite")
        self._last_update = int(update)
        self._last_total_updates = int(total_updates)
        output_features: Mapping[str, Any] = model_output
        if factorized_values is not None:
            output_features = dict(model_output)
            output_features["factor_outputs"] = {
                "z_r": factorized_values.z_r,
                "z_d": factorized_values.z_d,
                "z_c": factorized_values.z_c,
                "z_int": factorized_values.z_int,
            }
        return BiCADXRTrainOutput(
            total=total,
            logits=logits,
            features=output_features,
            audit=audit,
            checkpoint_runtime=runtime,
            backward_plan=backward_plan,
        )


__all__ = [
    "BiCADXRBatch",
    "BiCADXRBackwardPlan",
    "BiCADXRTrainOutput",
    "BiCADXRTrainer",
]
