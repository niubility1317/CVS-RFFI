from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional


def _as_float(value: Any, default: float) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    return out if math.isfinite(out) else float(default)


def _as_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _clamp(value: float, low: float, high: float) -> float:
    if high < low:
        low, high = high, low
    return float(max(low, min(high, value)))


def _ema(old: float, new: float, momentum: float) -> float:
    if not math.isfinite(old):
        return float(new)
    if not math.isfinite(new):
        return float(old)
    alpha = _clamp(momentum, 0.0, 1.0)
    return float((1.0 - alpha) * old + alpha * new)


def _percentile(values: list[float], q: float) -> float:
    finite = sorted(float(v) for v in values if math.isfinite(float(v)))
    if not finite:
        return float("nan")
    if len(finite) == 1:
        return float(finite[0])
    pos = _clamp(float(q), 0.0, 100.0) / 100.0 * float(len(finite) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(finite[lo])
    frac = pos - float(lo)
    return float(finite[lo] * (1.0 - frac) + finite[hi] * frac)


def _clean_choice(value: Any, default: str, allowed: set[str]) -> str:
    choice = str(value or default).strip().lower()
    return choice if choice in allowed else default


@dataclass(frozen=True)
class FedCGRLDecision:
    lambda_rx_adv: float
    base_lambda: float
    warmup_gate: float
    leak_gate: float
    tx_gate: float
    conflict_gate: float
    unclamped_lambda: float
    leak_reference_acc: float
    leak_stat: str
    tx_guard_release: float

    def as_metrics(self) -> Dict[str, Any]:
        return {
            "fed_cgrl_lambda_rx_adv": float(self.lambda_rx_adv),
            "fed_cgrl_base_lambda": float(self.base_lambda),
            "fed_cgrl_warmup_gate": float(self.warmup_gate),
            "fed_cgrl_leak_gate": float(self.leak_gate),
            "fed_cgrl_leak_reference_acc": float(self.leak_reference_acc),
            "fed_cgrl_leak_stat": str(self.leak_stat),
            "fed_cgrl_tx_gate": float(self.tx_gate),
            "fed_cgrl_tx_guard_release": float(self.tx_guard_release),
            "fed_cgrl_conflict_gate": float(self.conflict_gate),
            "fed_cgrl_unclamped_lambda": float(self.unclamped_lambda),
        }


@dataclass
class FedCGRLClientState:
    grl_target_acc_ema: float = float("nan")
    loss_cls_ema: float = float("nan")
    lambda_rx_adv: float = float("nan")
    seen: int = 0


class FedCGRLController:
    """Calibrate receiver-adversarial GRL strength from federated diagnostics."""

    def __init__(
        self,
        *,
        enabled: bool,
        base_lambda: float,
        min_lambda: float = 0.0,
        max_lambda: float = 2.0,
        warmup_rounds: int = 0,
        leak_target_acc: float = 20.0,
        leak_gain: float = 0.5,
        leak_gate_min: float = 0.75,
        leak_gate_max: float = 2.0,
        tx_loss_guard: float = 0.0,
        tx_loss_gate_min: float = 0.35,
        tx_guard_release_rounds: int = 0,
        conflict_threshold: float = -0.10,
        conflict_gate_min: float = 0.35,
        conflict_source: str = "auto",
        ema: float = 0.35,
        leak_stat: str = "p90",
    ):
        self.enabled = bool(enabled)
        self.base_lambda = float(base_lambda)
        self.min_lambda = float(min_lambda)
        self.max_lambda = float(max_lambda)
        self.warmup_rounds = max(0, int(warmup_rounds))
        self.leak_target_acc = max(1e-6, float(leak_target_acc))
        self.leak_gain = float(leak_gain)
        self.leak_gate_min = float(leak_gate_min)
        self.leak_gate_max = float(leak_gate_max)
        self.tx_loss_guard = max(0.0, float(tx_loss_guard))
        self.tx_loss_gate_min = float(tx_loss_gate_min)
        self.tx_guard_release_rounds = max(0, int(tx_guard_release_rounds))
        self.conflict_threshold = float(conflict_threshold)
        self.conflict_gate_min = float(conflict_gate_min)
        self.conflict_source = _clean_choice(conflict_source, "auto", {"auto", "none", "client_delta", "vmb"})
        self.leak_stat = _clean_choice(leak_stat, "p90", {"client", "mean", "p90", "max", "worst"})
        self.ema = _clamp(float(ema), 0.0, 1.0)
        self.client_states: Dict[str, FedCGRLClientState] = {}
        self.last_conflict_cos_min = float("nan")
        self.last_conflict_source = "none"
        self.last_conflict_signal_available = 0.0
        self.last_round = 0

    @classmethod
    def from_config(cls, cfg) -> "FedCGRLController":
        fallback_base = _as_float(getattr(cfg, "lambda_rx_adv", 1.0), 1.0)
        base_lambda = _as_float(getattr(cfg, "fed_cgrl_base_lambda", None), fallback_base)
        if base_lambda < 0.0:
            base_lambda = fallback_base
        return cls(
            enabled=bool(getattr(cfg, "use_fed_cgrl", False)),
            base_lambda=base_lambda,
            min_lambda=_as_float(getattr(cfg, "fed_cgrl_min_lambda", 0.0), 0.0),
            max_lambda=_as_float(getattr(cfg, "fed_cgrl_max_lambda", 2.0), 2.0),
            warmup_rounds=_as_int(getattr(cfg, "fed_cgrl_warmup_rounds", 0), 0),
            leak_target_acc=_as_float(getattr(cfg, "fed_cgrl_leak_target_acc", 20.0), 20.0),
            leak_gain=_as_float(getattr(cfg, "fed_cgrl_leak_gain", 0.5), 0.5),
            leak_gate_min=_as_float(getattr(cfg, "fed_cgrl_leak_gate_min", 0.75), 0.75),
            leak_gate_max=_as_float(getattr(cfg, "fed_cgrl_leak_gate_max", 2.0), 2.0),
            tx_loss_guard=_as_float(getattr(cfg, "fed_cgrl_tx_loss_guard", 0.0), 0.0),
            tx_loss_gate_min=_as_float(getattr(cfg, "fed_cgrl_tx_loss_gate_min", 0.35), 0.35),
            tx_guard_release_rounds=_as_int(getattr(cfg, "fed_cgrl_tx_guard_release_rounds", 0), 0),
            conflict_threshold=_as_float(getattr(cfg, "fed_cgrl_conflict_threshold", -0.10), -0.10),
            conflict_gate_min=_as_float(getattr(cfg, "fed_cgrl_conflict_gate_min", 0.35), 0.35),
            conflict_source=getattr(cfg, "fed_cgrl_conflict_source", "auto"),
            ema=_as_float(getattr(cfg, "fed_cgrl_ema", 0.35), 0.35),
            leak_stat=getattr(cfg, "fed_cgrl_leak_stat", "p90"),
        )

    def config_snapshot(self) -> Dict[str, Any]:
        return {
            "enabled": bool(self.enabled),
            "base_lambda": float(self.base_lambda),
            "min_lambda": float(self.min_lambda),
            "max_lambda": float(self.max_lambda),
            "warmup_rounds": int(self.warmup_rounds),
            "leak_target_acc": float(self.leak_target_acc),
            "leak_gain": float(self.leak_gain),
            "leak_gate_min": float(self.leak_gate_min),
            "leak_gate_max": float(self.leak_gate_max),
            "leak_stat": str(self.leak_stat),
            "tx_loss_guard": float(self.tx_loss_guard),
            "tx_loss_gate_min": float(self.tx_loss_gate_min),
            "tx_guard_release_rounds": int(self.tx_guard_release_rounds),
            "conflict_threshold": float(self.conflict_threshold),
            "conflict_gate_min": float(self.conflict_gate_min),
            "conflict_source": str(self.conflict_source),
            "ema": float(self.ema),
        }

    def _state(self, client_id: str) -> FedCGRLClientState:
        key = str(client_id)
        if key not in self.client_states:
            self.client_states[key] = FedCGRLClientState()
        return self.client_states[key]

    def _warmup_gate(self, round_idx: int) -> float:
        if not self.enabled or self.warmup_rounds <= 0:
            return 1.0
        return _clamp(float(max(1, int(round_idx))) / float(self.warmup_rounds), 0.0, 1.0)

    def _leak_reference_acc(self, state: FedCGRLClientState) -> float:
        if not self.enabled:
            return float("nan")
        own = float(state.grl_target_acc_ema)
        if self.leak_stat == "client":
            return own if math.isfinite(own) else float("nan")
        values = [
            float(client_state.grl_target_acc_ema)
            for client_state in self.client_states.values()
            if math.isfinite(float(client_state.grl_target_acc_ema))
        ]
        if not values:
            return own if math.isfinite(own) else float("nan")
        if self.leak_stat == "mean":
            return float(sum(values) / len(values))
        if self.leak_stat in {"max", "worst"}:
            return float(max(values))
        return _percentile(values, 90.0)

    def _leak_gate(self, reference_acc: float) -> float:
        if not self.enabled or not math.isfinite(reference_acc):
            return 1.0
        if reference_acc >= self.leak_target_acc:
            ratio = (reference_acc - self.leak_target_acc) / self.leak_target_acc
            return _clamp(1.0 + self.leak_gain * ratio, self.leak_gate_min, self.leak_gate_max)
        ratio = (self.leak_target_acc - reference_acc) / self.leak_target_acc
        return _clamp(1.0 - 0.5 * ratio, self.leak_gate_min, self.leak_gate_max)

    def _tx_guard_release(self, round_idx: int) -> float:
        if not self.enabled or self.tx_guard_release_rounds <= 0:
            return 0.0
        start_round = max(1, int(self.warmup_rounds))
        progress = (float(max(1, int(round_idx))) - float(start_round)) / float(max(1, self.tx_guard_release_rounds))
        return _clamp(progress, 0.0, 1.0)

    def _tx_gate(self, state: FedCGRLClientState, round_idx: int) -> tuple[float, float]:
        if not self.enabled or self.tx_loss_guard <= 0.0 or not math.isfinite(state.loss_cls_ema):
            return 1.0, 0.0
        if state.loss_cls_ema <= self.tx_loss_guard:
            return 1.0, 0.0
        base_gate = _clamp(self.tx_loss_guard / max(state.loss_cls_ema, 1e-12), self.tx_loss_gate_min, 1.0)
        release = self._tx_guard_release(round_idx)
        return _clamp(base_gate + (1.0 - base_gate) * release, self.tx_loss_gate_min, 1.0), release

    def _conflict_gate(self) -> float:
        if not self.enabled or not math.isfinite(self.last_conflict_cos_min):
            return 1.0
        if self.last_conflict_cos_min >= self.conflict_threshold:
            return 1.0
        denom = max(abs(self.conflict_threshold), 1e-6)
        severity = (self.conflict_threshold - self.last_conflict_cos_min) / denom
        return _clamp(1.0 - severity, self.conflict_gate_min, 1.0)

    def lambda_for_client(self, client_id: str, round_idx: int) -> FedCGRLDecision:
        if not self.enabled:
            return FedCGRLDecision(
                lambda_rx_adv=float(self.base_lambda),
                base_lambda=float(self.base_lambda),
                warmup_gate=1.0,
                leak_gate=1.0,
                tx_gate=1.0,
                conflict_gate=1.0,
                unclamped_lambda=float(self.base_lambda),
                leak_reference_acc=float("nan"),
                leak_stat=str(self.leak_stat),
                tx_guard_release=0.0,
            )
        state = self._state(str(client_id))
        warmup_gate = self._warmup_gate(round_idx)
        leak_reference_acc = self._leak_reference_acc(state)
        leak_gate = self._leak_gate(leak_reference_acc)
        tx_gate, tx_guard_release = self._tx_gate(state, round_idx)
        conflict_gate = self._conflict_gate()
        raw = float(self.base_lambda) * warmup_gate * leak_gate * tx_gate * conflict_gate
        value = _clamp(raw, self.min_lambda, self.max_lambda)
        state.lambda_rx_adv = value
        return FedCGRLDecision(
            lambda_rx_adv=value,
            base_lambda=float(self.base_lambda),
            warmup_gate=warmup_gate,
            leak_gate=leak_gate,
            tx_gate=tx_gate,
            conflict_gate=conflict_gate,
            unclamped_lambda=raw,
            leak_reference_acc=leak_reference_acc,
            leak_stat=str(self.leak_stat),
            tx_guard_release=tx_guard_release,
        )

    def update_after_round(
        self,
        client_results: Mapping[str, Mapping[str, Any]],
        *,
        round_idx: int,
        conflict_summary: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self.last_round = int(round_idx)
        if conflict_summary is not None:
            self.last_conflict_source = str(conflict_summary.get("source", "unknown") or "unknown")
            conflict_value = _as_float(
                conflict_summary.get("grad_cos_min_before", conflict_summary.get("min", float("nan"))),
                float("nan"),
            )
            self.last_conflict_cos_min = conflict_value
            self.last_conflict_signal_available = 1.0 if math.isfinite(conflict_value) else 0.0
        for client_id, metrics in (client_results or {}).items():
            state = self._state(str(client_id))
            seen = _as_int(metrics.get("seen", 0), 0)
            state.seen = max(0, seen)
            if "grl_target_acc" in metrics:
                state.grl_target_acc_ema = _ema(
                    state.grl_target_acc_ema,
                    _as_float(metrics.get("grl_target_acc"), float("nan")),
                    self.ema,
                )
            if "loss_cls" in metrics:
                state.loss_cls_ema = _ema(
                    state.loss_cls_ema,
                    _as_float(metrics.get("loss_cls"), float("nan")),
                    self.ema,
                )
            if "fed_cgrl_lambda_rx_adv" in metrics:
                state.lambda_rx_adv = _as_float(metrics.get("fed_cgrl_lambda_rx_adv"), state.lambda_rx_adv)

    def round_summary(self) -> Dict[str, Any]:
        clients = sorted(self.client_states)
        weights = {cid: max(0, int(self.client_states[cid].seen)) for cid in clients}
        total_seen = sum(weights.values())

        def weighted(name: str) -> float:
            total = 0.0
            denom = 0
            for cid in clients:
                value = getattr(self.client_states[cid], name)
                if not math.isfinite(value):
                    continue
                seen = weights[cid] if total_seen > 0 else 1
                total += float(value) * float(seen)
                denom += int(seen)
            return float(total / max(1, denom)) if denom > 0 else float("nan")

        def values(name: str) -> list[tuple[str, float]]:
            out = []
            for cid in clients:
                value = getattr(self.client_states[cid], name)
                if math.isfinite(value):
                    out.append((cid, float(value)))
            return out

        def stat(name: str, kind: str) -> float:
            vals = [value for _, value in values(name)]
            if not vals:
                return float("nan")
            if kind == "min":
                return float(min(vals))
            if kind == "max":
                return float(max(vals))
            if kind == "p90":
                return _percentile(vals, 90.0)
            return float("nan")

        def worst_client(name: str) -> str:
            vals = values(name)
            if not vals:
                return ""
            return max(vals, key=lambda item: item[1])[0]

        return {
            "enabled": bool(self.enabled),
            "last_round": int(self.last_round),
            "client_count": int(len(clients)),
            "clients": clients,
            "lambda_rx_adv_avg": weighted("lambda_rx_adv"),
            "lambda_rx_adv_min": stat("lambda_rx_adv", "min"),
            "lambda_rx_adv_max": stat("lambda_rx_adv", "max"),
            "lambda_rx_adv_p90": stat("lambda_rx_adv", "p90"),
            "grl_target_acc_avg": weighted("grl_target_acc_ema"),
            "grl_target_acc_min": stat("grl_target_acc_ema", "min"),
            "grl_target_acc_max": stat("grl_target_acc_ema", "max"),
            "grl_target_acc_p90": stat("grl_target_acc_ema", "p90"),
            "grl_target_acc_worst_client": worst_client("grl_target_acc_ema"),
            "loss_cls_avg": weighted("loss_cls_ema"),
            "loss_cls_min": stat("loss_cls_ema", "min"),
            "loss_cls_max": stat("loss_cls_ema", "max"),
            "loss_cls_p90": stat("loss_cls_ema", "p90"),
            "loss_cls_worst_client": worst_client("loss_cls_ema"),
            "conflict_cos_min": float(self.last_conflict_cos_min),
            "conflict_source": str(self.last_conflict_source),
            "conflict_signal_available": float(self.last_conflict_signal_available),
            "leak_stat": str(self.leak_stat),
        }
