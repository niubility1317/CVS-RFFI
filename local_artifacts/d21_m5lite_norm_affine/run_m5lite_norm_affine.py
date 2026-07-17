"""M5-lite: exact-whitelist norm-affine Stage2-C adaptation.

The predictor sees labelled support and sealed LEO_weak IQ only. Query truth is
opened exclusively by the separate scorer after the prediction NPZ is sealed.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[2]
BASE_PATH = REPO_ROOT / "local_artifacts" / "d21_floor_explore" / "run_floor_aware_diag.py"
SPEC = importlib.util.spec_from_file_location("d21_floor_base", BASE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("cannot load fixed FFT96 representation helper")
BASE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BASE)

SCENARIOS = BASE.SCENARIOS
ARMS = ("Z0", "M5lite_norm_only")
EPOCHS = 5
MAX_STEPS = 50
LR = 0.01
FFT_WEIGHT = 8.0
ID_DIM = 160
FFT_DIM = 96
REP_DIM = ID_DIM + FFT_DIM
NORM_BLOCKS = ("t1", "t2", "t3", "f1", "f2", "f3", "pa_b1", "pa_b2", "pa_b3")


def _allowed_names() -> set[str]:
    names = {
        "model.id_backbone.time_fuse.1.weight",
        "model.id_backbone.time_fuse.1.bias",
    }
    for block in NORM_BLOCKS:
        names.add(f"model.id_backbone.{block}.norm.weight")
        names.add(f"model.id_backbone.{block}.norm.bias")
    return names


ALLOWED_NAMES = _allowed_names()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _load_runtime(path: Path) -> tuple[torch.jit.ScriptModule, dict[str, torch.Tensor], list[torch.nn.Parameter]]:
    runtime = torch.jit.load(str(path), map_location="cuda").cuda().train()
    base: dict[str, torch.Tensor] = {}
    trainable: list[torch.nn.Parameter] = []
    seen = set()
    for name, parameter in runtime.named_parameters():
        allowed = name in ALLOWED_NAMES
        parameter.requires_grad_(allowed)
        if allowed:
            seen.add(name)
            trainable.append(parameter)
            base[name] = parameter.detach().clone()
    if seen != ALLOWED_NAMES:
        raise RuntimeError(f"exact whitelist drift: missing={sorted(ALLOWED_NAMES-seen)}, extra={sorted(seen-ALLOWED_NAMES)}")
    if sum(parameter.numel() for parameter in trainable) != 1136:
        raise RuntimeError("M5-lite parameter count drift")
    return runtime, base, trainable


def _torch_fft(iq: np.ndarray) -> torch.Tensor:
    fft = BASE.spectral_logmag_sketch(iq, dim=FFT_DIM).astype(np.float32)
    return F.normalize(torch.from_numpy(fft).cuda(), dim=1)


def _torch_representation(z_id: torch.Tensor, fft: torch.Tensor) -> torch.Tensor:
    return F.normalize(torch.cat([F.normalize(z_id.float(), dim=1), FFT_WEIGHT * fft], dim=1), dim=1)


def _loo_scores(rows: torch.Tensor, labels: torch.Tensor, class_count: int) -> torch.Tensor:
    similarities = rows @ rows.T
    similarities = similarities.masked_fill(torch.eye(rows.shape[0], device=rows.device, dtype=torch.bool), -1e4)
    return torch.stack([similarities[:, labels == index].max(dim=1).values for index in range(class_count)], dim=1)


def _support_metrics(scores: torch.Tensor, labels: torch.Tensor, old_count: int) -> dict[str, float]:
    pred = scores.argmax(dim=1)
    per_class = [float((pred[labels == c] == c).float().mean().item()) for c in range(scores.shape[1])]
    return {
        "support_old_acc": float((pred[labels < old_count] == labels[labels < old_count]).float().mean().item()),
        "support_new_acc": float((pred[labels >= old_count] == labels[labels >= old_count]).float().mean().item()),
        "support_old_floor": min(per_class[:old_count]),
        "support_new_floor": min(per_class[old_count:]),
    }


def _fit(
    runtime: torch.jit.ScriptModule,
    base_state: dict[str, torch.Tensor],
    trainable: list[torch.nn.Parameter],
    iq: np.ndarray,
    labels_np: np.ndarray,
    old_count: int,
    scenario: str,
) -> tuple[list[dict[str, Any]], float]:
    rows = torch.from_numpy(np.asarray(iq, dtype=np.float32)).cuda()
    labels = torch.from_numpy(labels_np.astype(np.int64)).long().cuda()
    fft = _torch_fft(iq)
    old_mask = labels < old_count
    with torch.no_grad():
        base_z, _ = runtime(rows)
        base_rep = _torch_representation(base_z, fft)
        base_old_pairwise = base_rep[old_mask] @ base_rep[old_mask].T
    optimiser = torch.optim.SGD(trainable, lr=LR, momentum=0.0)
    trace: list[dict[str, Any]] = []
    torch.cuda.synchronize()
    start = time.perf_counter()
    for epoch in range(1, EPOCHS + 1):
        optimiser.zero_grad(set_to_none=True)
        z_id, _ = runtime(rows)
        rep = _torch_representation(z_id, fft)
        scores = _loo_scores(rep, labels, int(labels.max().item()) + 1)
        ce_rows = F.cross_entropy(20.0 * scores, labels, reduction="none")
        class_losses = torch.stack([ce_rows[labels == c].mean() for c in range(scores.shape[1])])
        class_cvar = torch.topk(class_losses, k=max(1, int(math.ceil(0.4 * scores.shape[1])))).values.mean()
        old_pairwise = F.smooth_l1_loss(rep[old_mask] @ rep[old_mask].T, base_old_pairwise)
        delta_l2 = torch.stack([
            (dict(runtime.named_parameters())[name] - reference).square().mean()
            for name, reference in base_state.items()
        ]).mean()
        total = ce_rows.mean() + 0.5 * class_cvar + 2.0 * old_pairwise + 0.01 * delta_l2
        total.backward()
        torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
        optimiser.step()
        with torch.no_grad():
            metrics = _support_metrics(scores, labels, old_count)
        row = {
            "candidate": "M5lite_norm_only",
            "scenario": scenario,
            "epoch": epoch,
            "step": epoch,
            "total_loss": float(total.detach().item()),
            "support_ce": float(ce_rows.mean().detach().item()),
            "class_cvar": float(class_cvar.detach().item()),
            "old_pairwise_retention_loss": float(old_pairwise.detach().item()),
            "delta_l2": float(delta_l2.detach().item()),
            **metrics,
        }
        trace.append(row)
        print("[M5LITE-LOSS] " + json.dumps(row, sort_keys=True), flush=True)
    torch.cuda.synchronize()
    duration = time.perf_counter() - start
    if len(trace) > MAX_STEPS:
        raise RuntimeError("adaptation step cap exceeded")
    return trace, duration


def _fp16_patch_and_reload(
    runtime_path: Path,
    trained: torch.jit.ScriptModule,
    base_state: dict[str, torch.Tensor],
) -> tuple[torch.jit.ScriptModule, dict[str, np.ndarray]]:
    trained_params = dict(trained.named_parameters())
    patch = {
        name: (trained_params[name].detach() - base_state[name]).cpu().numpy().astype(np.float16)
        for name in sorted(ALLOWED_NAMES)
    }
    reloaded, reloaded_base, _ = _load_runtime(runtime_path)
    with torch.no_grad():
        reloaded_params = dict(reloaded.named_parameters())
        for name, delta in patch.items():
            reloaded_params[name].copy_(reloaded_base[name] + torch.from_numpy(delta.astype(np.float32)).cuda())
    reloaded.eval()
    return reloaded, patch


def _extract(runtime: torch.jit.ScriptModule, iq: np.ndarray, fft: np.ndarray) -> tuple[np.ndarray, float]:
    rows = torch.from_numpy(np.asarray(iq, dtype=np.float32)).cuda()
    fft_t = F.normalize(torch.from_numpy(fft.astype(np.float32)).cuda(), dim=1)
    torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.inference_mode():
        z_id, _ = runtime(rows)
        rep = _torch_representation(z_id, fft_t)
    torch.cuda.synchronize()
    return rep.cpu().numpy().astype(np.float32), time.perf_counter() - start


def _predict_rows(query: np.ndarray, support: np.ndarray, labels: np.ndarray, class_count: int) -> tuple[np.ndarray, float]:
    start = time.perf_counter_ns()
    scores = BASE._top1_scores(query, support, labels, class_count)
    elapsed = (time.perf_counter_ns() - start) / 1e6
    return scores.argmax(axis=1).astype(np.int64), elapsed / query.shape[0]


def predict(capsule: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError(output)
    after_enrollment = capsule / "predictor" / "after" / "enrollment_only"
    after_apply = capsule / "predictor" / "after" / "apply_only_staging"
    before_apply = capsule / "predictor" / "before" / "apply_only_staging"
    before_manifest = json.loads((capsule / "predictor" / "before" / "enrollment_only" / "package_manifest.json").read_text(encoding="utf-8"))
    after_manifest = json.loads((after_enrollment / "package_manifest.json").read_text(encoding="utf-8"))
    old_count = int(before_manifest["registered_class_count"])
    class_count = int(after_manifest["registered_class_count"])
    if (old_count, class_count) != (6, 11):
        raise RuntimeError("M5-lite requires fixed K10/new5 6->11 capsule")
    runtime_path = after_enrollment / "sealed_feature_runtime.pt"
    arrays: dict[str, np.ndarray] = {}
    traces: list[dict[str, Any]] = []
    patch_arrays: dict[str, np.ndarray] = {}
    resources: list[dict[str, Any]] = []
    torch.cuda.reset_peak_memory_stats()
    for scenario in SCENARIOS:
        with np.load(after_enrollment / f"support_{scenario}.npz", allow_pickle=False) as sf:
            support_iq = sf["support_leo_weak_iq"].astype(np.float32)
            labels = sf["support_class_indices"].astype(np.int64)
        with np.load(after_apply / f"query_{scenario}.npz", allow_pickle=False) as qf:
            query_iq = qf["query_leo_weak_iq"].astype(np.float32)
            query_tokens = qf["query_tokens"].astype(str)
        with np.load(before_apply / f"query_{scenario}.npz", allow_pickle=False) as bf:
            before_tokens = bf["query_tokens"].astype(str)
        token_to_row = {token: i for i, token in enumerate(query_tokens.tolist())}
        before_indices = np.asarray([token_to_row[token] for token in before_tokens], dtype=np.int64)
        support_fft = BASE.spectral_logmag_sketch(support_iq, dim=FFT_DIM).astype(np.float32)
        query_fft = BASE.spectral_logmag_sketch(query_iq, dim=FFT_DIM).astype(np.float32)

        z0, _, _ = _load_runtime(runtime_path)
        z0.eval()
        z0_support, z0_support_s = _extract(z0, support_iq, support_fft)
        z0_query, z0_query_s = _extract(z0, query_iq, query_fft)
        z0_support_q, _, _ = BASE._int8_support_roundtrip(z0_support)
        z0_after, z0_ms = _predict_rows(z0_query, z0_support_q, labels, class_count)
        old_mask = labels < old_count
        z0_before, z0_before_ms = _predict_rows(z0_query[before_indices], z0_support_q[old_mask], labels[old_mask], old_count)

        trained, base_state, trainable = _load_runtime(runtime_path)
        trace, adapt_s = _fit(trained, base_state, trainable, support_iq, labels, old_count, scenario)
        traces.extend(trace)
        merged, patch = _fp16_patch_and_reload(runtime_path, trained, base_state)
        for name, value in patch.items():
            patch_arrays[f"{scenario}__{name.replace('.', '__')}"] = value
        m5_support, m5_support_s = _extract(merged, support_iq, support_fft)
        m5_query, m5_query_s = _extract(merged, query_iq, query_fft)
        m5_support_q, _, _ = BASE._int8_support_roundtrip(m5_support)
        m5_after, m5_ms = _predict_rows(m5_query, m5_support_q, labels, class_count)

        for arm, after_pred, before_pred in (
            ("Z0", z0_after, z0_before),
            ("M5lite_norm_only", m5_after, z0_before),
        ):
            arrays[f"{arm}__{scenario}__after_predictions"] = after_pred
            arrays[f"{arm}__{scenario}__after_tokens"] = query_tokens
            arrays[f"{arm}__{scenario}__before_predictions"] = before_pred
            arrays[f"{arm}__{scenario}__before_tokens"] = before_tokens
        resources.append({
            "scenario": scenario,
            "adaptation_seconds": adapt_s,
            "adaptation_epochs": EPOCHS,
            "adaptation_steps": EPOCHS,
            "z0_feature_seconds_support": z0_support_s,
            "z0_feature_seconds_query": z0_query_s,
            "m5_feature_seconds_support": m5_support_s,
            "m5_feature_seconds_query": m5_query_s,
            "z0_classifier_ms_per_after_query": z0_ms,
            "z0_classifier_ms_per_before_query": z0_before_ms,
            "m5_classifier_ms_per_after_query": m5_ms,
        })
        del z0, trained, merged
        torch.cuda.empty_cache()

    output.parent.mkdir(parents=True, exist_ok=True)
    arrays["schema_json"] = np.asarray(json.dumps({
        "schema": "cvs.phase2.d21_m5lite_predictions.v1",
        "arms": ARMS,
        "phase": "Stage2-C-only",
        "support_only_fit": True,
        "query_fit": False,
        "query_role_oracle": False,
        "all_registered_classes_per_sample": True,
        "m5_before_state": "shared Z0 old-only state; M5 is fitted only after new-class registration",
    }, sort_keys=True))
    np.savez_compressed(output, **arrays)
    trace_path = output.parent / "loss_trace.jsonl"
    with trace_path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in traces:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    patch_path = output.parent / "m5lite_fp16_delta.npz"
    np.savez_compressed(patch_path, **patch_arrays)
    whitelist = []
    probe, _, _ = _load_runtime(runtime_path)
    for name, parameter in probe.named_parameters():
        if name in ALLOWED_NAMES:
            whitelist.append({"name": name, "shape": list(parameter.shape), "parameters": parameter.numel()})
    del probe
    receipt = {
        "schema": "cvs.phase2.d21_m5lite_receipt.v1",
        "prediction_sha256": _sha256(output),
        "loss_trace_sha256": _sha256(trace_path),
        "patch_sha256": _sha256(patch_path),
        "exact_whitelist": whitelist,
        "trainable_parameters": sum(row["parameters"] for row in whitelist),
        "optimizer": "SGD(lr=0.01,momentum=0)",
        "optimizer_state_persisted": False,
        "patch_dtype": "float16",
        "patch_payload_bytes_per_scenario": 1136 * 2,
        "patch_file_bytes": patch_path.stat().st_size,
        "merged_inference_added_MAC": 0,
        "fixed_fft96_knn_classifier_MAC_per_query": class_count * 10 * REP_DIM,
        "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "scenario_resources": resources,
        "query_truth_opened": False,
        "query_role_oracle_access": False,
        "query_true_batch_class_count_access": False,
        "query_class_quota_access": False,
        "query_batch_global_assignment": False,
    }
    _json_dump(output.with_suffix(".receipt.json"), receipt)


def _metrics(pred: np.ndarray, truth: np.ndarray, old_count: int) -> dict[str, Any]:
    per_class = {}
    for class_index in sorted(set(truth.tolist())):
        mask = truth == class_index
        per_class[str(class_index)] = {"count": int(mask.sum()), "accuracy": float(np.mean(pred[mask] == truth[mask]))}
    old = [row["accuracy"] for key, row in per_class.items() if int(key) < old_count]
    new = [row["accuracy"] for key, row in per_class.items() if int(key) >= old_count]
    old_acc = float(np.mean(pred[truth < old_count] == truth[truth < old_count]))
    new_acc = float(np.mean(pred[truth >= old_count] == truth[truth >= old_count]))
    return {
        "old_acc": old_acc,
        "seen_new_acc": new_acc,
        "min_old_class_acc": min(old),
        "min_seen_new_class_acc": min(new),
        "H_old_new": 2 * old_acc * new_acc / max(old_acc + new_acc, 1e-12),
        "per_class": per_class,
    }


def score(prediction: Path, truth_path: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError(output)
    truth_doc = json.loads(truth_path.read_text(encoding="utf-8"))
    truth_by_token = {row["query_token"]: row for row in truth_doc["rows"]}
    result: dict[str, Any] = {
        "schema": "cvs.phase2.d21_m5lite_score.v1",
        "prediction_sha256": _sha256(prediction),
        "truth_sidecar_sha256": _sha256(truth_path),
        "query_truth_joined_only_after_immutable_predictions": True,
        "scorer_feedback_to_predictor": False,
        "arms": {},
    }
    with np.load(prediction, allow_pickle=False) as data:
        for arm in ARMS:
            scene_rows = []
            pooled_before_pred, pooled_before_truth = [], []
            pooled_after_pred, pooled_after_truth = [], []
            for scenario in SCENARIOS:
                btokens = data[f"{arm}__{scenario}__before_tokens"].astype(str)
                atokens = data[f"{arm}__{scenario}__after_tokens"].astype(str)
                bp = data[f"{arm}__{scenario}__before_predictions"]
                ap = data[f"{arm}__{scenario}__after_predictions"]
                bt = np.asarray([truth_by_token[token]["true_class_index"] for token in btokens], dtype=np.int64)
                at = np.asarray([truth_by_token[token]["true_class_index"] for token in atokens], dtype=np.int64)
                old_count = len(set(bt.tolist()))
                row = _metrics(ap, at, old_count)
                before_acc = float(np.mean(bp == bt))
                row.update({"scenario": scenario, "old_acc_before_increment": before_acc, "average_forgetting": before_acc - row["old_acc"]})
                scene_rows.append(row)
                pooled_before_pred.append(bp); pooled_before_truth.append(bt)
                pooled_after_pred.append(ap); pooled_after_truth.append(at)
            bp = np.concatenate(pooled_before_pred); bt = np.concatenate(pooled_before_truth)
            ap = np.concatenate(pooled_after_pred); at = np.concatenate(pooled_after_truth)
            aggregate = _metrics(ap, at, len(set(bt.tolist())))
            before_acc = float(np.mean(bp == bt))
            aggregate.update({"old_acc_before_increment": before_acc, "average_forgetting": before_acc - aggregate["old_acc"]})
            result["arms"][arm] = {"scenario_rows": scene_rows, "aggregate": aggregate}
    _json_dump(output, result)


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    pred = sub.add_parser("predict")
    pred.add_argument("--capsule", type=Path, required=True)
    pred.add_argument("--output", type=Path, required=True)
    scorer = sub.add_parser("score")
    scorer.add_argument("--prediction", type=Path, required=True)
    scorer.add_argument("--truth", type=Path, required=True)
    scorer.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "predict":
        predict(args.capsule.resolve(), args.output.resolve())
    else:
        score(args.prediction.resolve(), args.truth.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
