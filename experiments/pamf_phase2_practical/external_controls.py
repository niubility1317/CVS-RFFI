"""Declared Phase2 combination-C controls on an actual, immutable B state.

These controls are project extensions, not native incremental-learning claims.
The fitting API accepts only current old/new support.  Query tensors appear
only in the returned score function and never update parameters or buffers.
Native output expansion is explicitly delegated to the source-profile bridge;
an unsupported native head is an interface error, never a replacement readout.
"""
from __future__ import annotations

import copy
import numbers
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F


NATURE = "PROJECT_COMBINATION_EXTENSION"
KINDS = ("ncm", "joint_ce100", "native_output_expand_ft100")
_STEPS = 100
_LR = 0.001
_EPS = 1e-12


def _classes(values: Sequence[int]) -> tuple[int, ...]:
    result = tuple(values)
    if (not result or any(not isinstance(v, numbers.Integral) or isinstance(v, bool)
                          or v < 0 for v in result) or len(set(result)) != len(result)):
        raise ValueError("classes must be unique nonnegative integer class IDs")
    return tuple(int(v) for v in result)


def _support(x: torch.Tensor, y: torch.Tensor, classes: tuple[int, ...]) -> torch.Tensor:
    if (not isinstance(x, torch.Tensor) or not isinstance(y, torch.Tensor)
            or x.ndim < 2 or y.ndim != 1 or not len(x) or len(x) != len(y)
            or y.dtype != torch.long or x.device != y.device):
        raise ValueError("support needs nonempty same-device tensors and one long label per sample")
    if not torch.isfinite(x).all():
        raise ValueError("nonfinite support input")
    if set(y.detach().cpu().tolist()) != set(classes):
        raise ValueError("support must cover exactly all current registered classes")
    ids = torch.tensor(classes, device=y.device, dtype=torch.long)
    return (y[:, None] == ids[None, :]).long().argmax(1)


def _config(value: Mapping[str, Any] | None) -> dict[str, Any]:
    """Accept the design's fixed budget, never a silent shortened test budget."""
    cfg = dict(value or {})
    if "phase2_linkage_design" in cfg:
        cfg = dict(cfg["phase2_linkage_design"])
    return cfg


def _budget(cfg: Mapping[str, Any], kind: str) -> None:
    section = "JointCE" if kind == "joint_ce100" else "native_head_expansion"
    chosen = cfg.get(section, cfg)
    if not isinstance(chosen, Mapping):
        raise ValueError(f"{section} configuration must be a mapping")
    for key, expected in (("steps", _STEPS), ("lr", _LR), ("weight_decay", 0.0),
                          ("optimizer", "Adam")):
        if key in chosen and chosen[key] != expected:
            raise ValueError(f"{kind} requires fixed {key}={expected!r}")
    if "batch_size" in chosen or "epochs" in chosen:
        raise ValueError("combination C uses all support in each of exactly 100 updates")


def _normalize(z: torch.Tensor) -> torch.Tensor:
    return F.normalize(z, p=2, dim=1, eps=_EPS)


def _features(fn: Callable, model: nn.Module, x: torch.Tensor,
              batch_size: int) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        chunks = [fn(model, x[start:start + batch_size]).detach()
                  for start in range(0, len(x), batch_size)]
        if any(z.ndim != 2 or len(z) != min(batch_size, len(x) - i * batch_size)
               for i, z in enumerate(chunks)):
            raise ValueError("features_fn must return one vector per support sample")
        z = torch.cat(chunks)
    if not z.is_floating_point() or not torch.isfinite(z).all():
        raise ValueError("B features must be finite real floating-point vectors")
    return z


def _means(u: torch.Tensor, y: torch.Tensor, count: int) -> torch.Tensor:
    return _normalize(torch.stack([u[y == i].mean(0) for i in range(count)]))


class _Readout(nn.Module):
    def __init__(self, backbone: nn.Module, features_fn: Callable,
                 means: torch.Tensor, linear: bool) -> None:
        super().__init__()
        self.backbone = backbone
        self.features_callback = features_fn
        if linear:
            # Every random initialization value is overwritten. Restore the CPU
            # RNG so this deterministic control does not perturb another fork.
            with torch.random.fork_rng(devices=[]):
                self.head = nn.Linear(means.shape[1], len(means)).to(means)
            with torch.no_grad():
                self.head.weight.copy_(means)
                self.head.bias.zero_()
        else:
            self.register_buffer("prototypes", means.detach().clone())

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.features_callback(self.backbone, x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = _normalize(self.encode(x))
        return self.head(u) if hasattr(self, "head") else u @ self.prototypes.t()


@torch.no_grad()
def _readout_scores(model: _Readout, x: torch.Tensor) -> torch.Tensor:
    model.eval()
    return model(x)


@torch.no_grad()
def _readout_features(model: _Readout, x: torch.Tensor) -> torch.Tensor:
    model.eval()
    return model.encode(x)


class _NativeScores:
    def __init__(self, logits_fn: Callable) -> None:
        self.logits_fn = logits_fn

    @torch.no_grad()
    def __call__(self, model: nn.Module, x: torch.Tensor) -> torch.Tensor:
        model.eval()
        return self.logits_fn(model, x)


def _check_logits(logits: torch.Tensor, rows: int, count: int) -> None:
    if logits.ndim != 2 or logits.shape != (rows, count):
        raise ValueError("C logits must face every registered class in declared column order")
    if not torch.isfinite(logits).all():
        raise FloatingPointError("nonfinite combination-C logits")


def _optimize(model: nn.Module, logits_fn: Callable, x: torch.Tensor, y: torch.Tensor,
              parameters: list[nn.Parameter], count: int, logger: Callable | None,
              kind: str) -> list[dict[str, Any]]:
    optimizer = torch.optim.Adam(parameters, lr=_LR, weight_decay=0.0)
    trace = []
    model.eval()  # Fixed BN statistics, dropout and native temperature definition.
    for step in range(1, _STEPS + 1):
        optimizer.zero_grad(set_to_none=True)
        logits = logits_fn(model, x)
        _check_logits(logits, len(x), count)
        loss = F.cross_entropy(logits, y, reduction="mean")
        if not torch.isfinite(loss):
            raise FloatingPointError("nonfinite combination-C cross entropy")
        loss.backward()
        optimizer.step()
        row = dict(phase="C", control=kind, step=step, loss=float(loss.detach()),
                   learning_rate=_LR, support_rows=len(x), full_batch=True)
        trace.append(row)
        if logger is not None:
            logger(dict(row))
    return trace


def _native(state: dict[str, Any], x: torch.Tensor, y: torch.Tensor, count: int,
            old_count: int, seed: int) -> tuple[nn.Module, Callable, list[nn.Parameter]]:
    callback = state.get("expand_head")
    if not callable(callback):
        raise ValueError("native output expansion requires an explicit compatible expand_head callback")
    from .vendor_native.comparison_suite.adaptation import _ActiveClasses

    def mapping_paths(model):
        return {f"{name}.columns" if name else "columns"
                for name, module in model.named_modules() if isinstance(module, _ActiveClasses)}

    before = {key: value.detach().clone() for key, value in state["model"].state_dict().items()}
    before_mapping_paths = mapping_paths(state["model"])
    expanded = callback(state, x, y, count, seed)
    if not isinstance(expanded, Mapping) or not isinstance(expanded.get("model"), nn.Module):
        raise ValueError("expand_head must return an expanded native model")
    model, logits_fn = expanded["model"], expanded.get("logits_fn")
    parameters = list(expanded.get("trainable_parameters", ()))
    if not callable(logits_fn) or not parameters:
        raise ValueError("expand_head must declare native logits and final output parameters")
    by_id = {id(p): name for name, p in model.named_parameters()}
    if (len({id(p) for p in parameters}) != len(parameters)
            or any(not isinstance(p, nn.Parameter) or id(p) not in by_id for p in parameters)):
        raise ValueError("native trainable parameters must be distinct members of the expanded model")
    weights = [p for p in parameters if p.ndim == 2 and p.shape[0] == count]
    biases = [p for p in parameters if p.ndim == 1 and p.shape[0] == count]
    if len(weights) != 1 or len(biases) > 1 or len(weights) + len(biases) != len(parameters):
        raise ValueError("only the expanded final output weight and optional bias may train")
    names = {by_id[id(p)] for p in parameters}
    after = model.state_dict()
    if set(before) != set(after):
        raise ValueError("native expansion must retain the actual B state names and preceding layers")
    # The actual DA wrapper selects native output columns. Its one declared
    # class-index buffer must grow with registration, unlike fixed BN/prehead
    # buffers. This is deliberately not an arbitrary buffer exception list.
    declared = expanded.get("declared_class_mapping_keys", ())
    if (not isinstance(declared, (list, tuple))
            or any(not isinstance(key, str) for key in declared)
            or len(set(declared)) != len(declared)):
        raise ValueError("declared_class_mapping_keys must list distinct actual wrapper columns paths")
    mapping_names = set(declared)
    if (not mapping_names <= before_mapping_paths
            or not mapping_names <= mapping_paths(model)
            or not mapping_names <= set(before)):
        raise ValueError("only actual B _ActiveClasses columns buffers may be declared class mappings")
    for name, previous in before.items():
        current = after[name]
        if name in names:
            if (previous.shape[0] != old_count or current.shape[0] != count
                    or previous.shape[1:] != current.shape[1:]
                    or not torch.equal(current[:old_count], previous)):
                raise ValueError("native expansion did not inherit actual B old output rows/bias")
        elif name in mapping_names:
            if (previous.dtype != torch.long or current.dtype != torch.long
                    or previous.shape != (old_count,) or current.shape != (count,)
                    or not torch.equal(current[:old_count], previous)
                    or not torch.equal(current[old_count:], torch.arange(
                        old_count, count, device=current.device, dtype=torch.long))):
                raise ValueError("native class mapping must retain its old prefix and append correct class indices")
        elif not torch.equal(current, previous):
            raise ValueError(f"native expansion changed frozen B state {name}")
    if biases and torch.count_nonzero(biases[0].detach()[old_count:]):
        raise ValueError("native new-class output bias must initialize to zero")
    trainable_ids = {id(p) for p in parameters}
    for parameter in model.parameters():
        parameter.requires_grad_(id(parameter) in trainable_ids)
    return model.eval(), logits_fn, parameters


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def build_control(kind: str, state: Mapping[str, Any], all_support_x: torch.Tensor,
                  all_support_y: torch.Tensor, classes: Sequence[int], seed: int,
                  config: Mapping[str, Any] | None = None,
                  logger: Callable | None = None) -> dict[str, Any]:
    """Fork one C prefix from B using only currently permitted target support.

    Labels are numeric class IDs in ``classes``; output columns follow that
    exact order. The native bridge receives compact labels in this order.
    ``features_fn(model, x)`` must use its passed model, not a captured B model.
    ``expand_head`` receives a private copied state and must preserve native
    state names, old output rows, prehead layers, metric and fixed buffers.
    Its optional ``declared_class_mapping_keys`` permits only actual
    ``_ActiveClasses.columns`` buffers to retain their prefix and append new
    compact class indices; no other buffer can be excepted.
    """
    if kind not in KINDS:
        raise ValueError(f"unsupported combination-C control {kind!r}")
    class_ids = _classes(classes)
    compact_y = _support(all_support_x, all_support_y, class_ids)
    if not isinstance(state.get("model"), nn.Module):
        raise ValueError("B state must contain its actual torch model")
    old_classes = _classes(state.get("classes", ()))
    if class_ids[:len(old_classes)] != old_classes:
        raise ValueError("C classes must preserve the actual B class prefix and row order")
    cfg = _config(config)
    batch_size = int(cfg.get("extraction_batch_size", 128))
    if batch_size <= 0:
        raise ValueError("extraction_batch_size must be positive")
    if kind != "ncm":
        _budget(cfg, kind)
    device = all_support_x.device
    _sync(device)
    started = time.perf_counter()
    result = copy.deepcopy(dict(state))
    backbone = result["model"].to(device).eval()
    for parameter in backbone.parameters():
        parameter.requires_grad_(False)
    features_fn = result.get("features_fn")
    trace: list[dict[str, Any]] = []
    if kind == "native_output_expand_ft100":
        model, logits_fn, parameters = _native(result, all_support_x, compact_y,
                                               len(class_ids), len(old_classes), int(seed))
        trace = _optimize(model, logits_fn, all_support_x, compact_y, parameters,
                          len(class_ids), logger, kind)
        score_fn = _NativeScores(logits_fn)
        initialization = "actual_B_old_rows_bias; new_support_mean_direction_times_median_old_weight_norm"
    else:
        if not callable(features_fn):
            raise ValueError("NCM/JointCE require the declared actual B features_fn")
        z = _features(features_fn, backbone, all_support_x, batch_size)
        u = _normalize(z)
        means = _means(u, compact_y, len(class_ids))
        model = _Readout(backbone, features_fn, means, kind == "joint_ce100").eval()
        parameters = list(model.head.parameters()) if kind == "joint_ce100" else []
        if parameters:
            # Cache only support representations; the frozen B module remains
            # resident and is used unchanged for future sample-wise inference.
            trace = _optimize(model, lambda current, values: current.head(values), u,
                              compact_y, parameters, len(class_ids), logger, kind)
        score_fn, features_fn = _readout_scores, _readout_features
        initialization = "all_rows_normalized_current_support_means; zero_bias"
    model.eval()
    _sync(device)
    tensors = list(model.state_dict().values())
    result.update(model=model, score_fn=score_fn, features_fn=features_fn,
                  classes=class_ids, nature=NATURE, phase="C", control=kind,
                  loss_trace=trace, query_fit=False, source_in_C=False,
                  target_U_in_C=False, B_native_decision_inherited=(kind == KINDS[2]),
                  initialization=initialization,
                  resources=dict(inherited_B=copy.deepcopy(state.get("resources", {})),
                      phase="C", optimizer_steps=len(trace), support_rows=len(all_support_x),
                      support_training_exposures=len(all_support_x) * len(trace),
                      source_rows=0, query_rows_fitted=0, target_U_rows=0,
                      trainable_parameters=sum(p.numel() for p in parameters),
                      total_parameters=sum(p.numel() for p in model.parameters()),
                      resident_state_bytes=sum(t.numel() * t.element_size() for t in tensors),
                      training_seconds=time.perf_counter() - started, hardware=str(device),
                      optimizer=("Adam" if parameters else None),
                      learning_rate=(_LR if parameters else None), weight_decay=0.0,
                      full_batch=True, equal_physical_sample_weights=True))
    return result
