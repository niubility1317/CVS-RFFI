from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


SCENARIOS = ("leo_clear_weak", "leo_low_elev_weak", "leo_rain_weak")
OLD_COUNT = 6
CLASS_COUNT = 11
K_SHOT = 10
EPOCHS = 5
LR = 1.0e-3
FEATURE_DIM = 256
STATE_CAP = 256 * 1024
WHITELISTS = {
    "A_tail_idproj": (
        "model.id_backbone.cls_head.id_proj.0.weight",
        "model.id_backbone.cls_head.id_proj.0.bias",
    ),
    "B_input_proj": (
        "model.id_backbone.t_proj.weight",
        "model.id_backbone.t_proj.bias",
        "model.id_backbone.f_proj.weight",
        "model.id_backbone.f_proj.bias",
        "model.id_backbone.freq_stats_proj.0.weight",
        "model.id_backbone.freq_stats_proj.0.bias",
        "model.id_backbone.pa_stats_proj.0.weight",
        "model.id_backbone.pa_stats_proj.0.bias",
    ),
    "C_tail_gate": (
        "model.id_backbone.cls_head.id_gate.0.weight",
        "model.id_backbone.cls_head.id_gate.0.bias",
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize(rows: np.ndarray) -> np.ndarray:
    value = np.asarray(rows, dtype=np.float32)
    return value / np.maximum(np.linalg.norm(value, axis=1, keepdims=True), 1.0e-8)


def fft96(rows: np.ndarray) -> np.ndarray:
    output = []
    target_x = np.linspace(0.0, 1.0, 96, dtype=np.float64)
    for row in np.asarray(rows, dtype=np.float32):
        value = row[0].astype(np.float64) + 1j * row[1].astype(np.float64)
        value -= np.mean(value)
        rms = float(np.sqrt(np.mean(np.abs(value) ** 2)))
        value = value / max(rms, 1.0e-8)
        logmag = np.log1p(np.abs(np.fft.fftshift(np.fft.fft(value * np.hanning(value.size)))))
        sketch = np.interp(target_x, np.linspace(0.0, 1.0, len(logmag)), logmag).astype(np.float32)
        sketch -= np.mean(sketch)
        output.append(sketch)
    return normalize(np.stack(output))


def fused(z: np.ndarray, iq: np.ndarray) -> np.ndarray:
    return normalize(np.concatenate((normalize(z), 8.0 * fft96(iq)), axis=1))


def runtime_z(model: torch.jit.ScriptModule, iq: np.ndarray, device: torch.device) -> np.ndarray:
    chunks = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(iq), 128):
            x = torch.from_numpy(np.asarray(iq[start : start + 128], dtype=np.float32)).to(device)
            chunks.append(model(x)[0].detach().float().cpu().numpy())
    return np.concatenate(chunks).astype(np.float32)


def configure_model(runtime_path: Path, whitelist: tuple[str, ...], device: torch.device):
    model = torch.jit.load(str(runtime_path), map_location=device)
    named = dict(model.named_parameters())
    missing = [name for name in whitelist if name not in named]
    if missing:
        raise RuntimeError(f"exact whitelist missing: {missing}")
    for parameter in named.values():
        parameter.requires_grad_(False)
    for name in whitelist:
        named[name].requires_grad_(True)
    actual = tuple(name for name, parameter in model.named_parameters() if parameter.requires_grad)
    if actual != whitelist:
        raise RuntimeError(f"trainable set drift: {actual} != {whitelist}")
    original = {name: named[name].detach().clone() for name in whitelist}
    return model, named, original


def differentiable_loss(
    z: torch.Tensor,
    truth: torch.Tensor,
    z0_old_similarity: torch.Tensor,
    old_count: int,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    z = F.normalize(z, dim=1)
    class_count = int(torch.max(truth).item()) + 1
    sums = torch.stack([z[truth == index].sum(dim=0) for index in range(class_count)])
    counts = torch.stack([(truth == index).sum() for index in range(class_count)]).to(z.dtype)
    prototypes = F.normalize(sums / counts[:, None], dim=1)
    logits = z @ prototypes.T
    own_sum = sums[truth] - z
    own_count = counts[truth] - 1.0
    own_proto = F.normalize(own_sum / own_count[:, None], dim=1)
    logits[torch.arange(len(z), device=z.device), truth] = torch.sum(z * own_proto, dim=1)
    ce = F.cross_entropy(logits * 12.0, truth)
    old_z = z[truth < old_count]
    retention = F.mse_loss(old_z @ old_z.T, z0_old_similarity)
    if class_count > old_count:
        sim = prototypes @ prototypes.T
        eye = torch.eye(class_count, dtype=torch.bool, device=z.device)
        involves_new = (
            (torch.arange(class_count, device=z.device)[:, None] >= old_count)
            | (torch.arange(class_count, device=z.device)[None, :] >= old_count)
        ) & ~eye
        separation = torch.mean(torch.relu(sim[involves_new] - 0.20) ** 2)
    else:
        separation = z.new_zeros(())
    total = ce + retention + 0.25 * separation
    return total, {"ce": ce, "old_pairwise_retention": retention, "new_separation": separation}


def train_patch(
    runtime_path: Path,
    whitelist: tuple[str, ...],
    iq: np.ndarray,
    truth_np: np.ndarray,
    device: torch.device,
    log_handle,
    tag: dict[str, str],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    model, named, original = configure_model(runtime_path, whitelist, device)
    model.train()
    x = torch.from_numpy(np.asarray(iq, dtype=np.float32)).to(device)
    truth = torch.from_numpy(np.asarray(truth_np, dtype=np.int64)).to(device)
    with torch.no_grad():
        z0 = F.normalize(model(x)[0], dim=1)
        old0 = z0[truth < OLD_COUNT]
        old_similarity = old0 @ old0.T
    optimizer = torch.optim.SGD([named[name] for name in whitelist], lr=LR, momentum=0.0)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    for epoch in range(EPOCHS):
        optimizer.zero_grad(set_to_none=True)
        z = model(x)[0]
        loss, parts = differentiable_loss(z, truth, old_similarity, OLD_COUNT)
        loss.backward()
        grad_norm = float(torch.sqrt(sum(torch.sum(named[name].grad ** 2) for name in whitelist)).detach().cpu())
        optimizer.step()
        entry = {
            **tag,
            "epoch": epoch + 1,
            "optimizer_step": epoch + 1,
            "loss": float(loss.detach().cpu()),
            **{name: float(value.detach().cpu()) for name, value in parts.items()},
            "grad_norm": grad_norm,
            "lr": LR,
        }
        log_handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
        log_handle.flush()
    elapsed = time.perf_counter() - started
    patch = {
        name: (named[name].detach() - original[name]).half().cpu().numpy()
        for name in whitelist
    }
    payload_bytes = int(sum(value.nbytes for value in patch.values()))
    updated = int(sum(named[name].numel() for name in whitelist))
    train_macs = int(3 * updated * len(iq) * EPOCHS)
    audit = {
        "updated_original_parameters": updated,
        "epochs": EPOCHS,
        "optimizer_steps": EPOCHS,
        "optimizer": "SGD",
        "momentum": 0.0,
        "optimizer_state_persisted": False,
        "fp16_patch_payload_bytes": payload_bytes,
        "training_updated_layer_macs_estimate": train_macs,
        "training_seconds": elapsed,
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
        "deployment_added_macs_after_merge": 0,
    }
    del optimizer, x, truth, z0, old0, old_similarity
    return patch, audit


def apply_patch(runtime_path: Path, whitelist: tuple[str, ...], patch: dict[str, np.ndarray], device: torch.device):
    model, named, _ = configure_model(runtime_path, whitelist, device)
    with torch.no_grad():
        for name in whitelist:
            named[name].add_(torch.from_numpy(patch[name].astype(np.float32)).to(device))
    model.eval()
    return model


def quantized_head(support: np.ndarray, owners: np.ndarray) -> dict[str, Any]:
    support = normalize(support)
    scale = np.maximum(np.max(np.abs(support), axis=1) / 127.0, 1.0e-8).astype(np.float16)
    q = np.clip(np.rint(support / scale[:, None]), -127, 127).astype(np.int8)
    vectors = normalize(q.astype(np.float32) * scale.astype(np.float32)[:, None])
    owner_u16 = np.asarray(owners, dtype=np.uint16)
    return {
        "vectors": vectors,
        "owners": owner_u16.astype(np.int64),
        "state_bytes": int(q.nbytes + scale.nbytes + owner_u16.nbytes),
        "q": q,
        "scale": scale,
        "owner_u16": owner_u16,
    }


def predict(head: dict[str, Any], query: np.ndarray, class_count: int) -> np.ndarray:
    """Per-sample all-registered-class prediction; no truth, role, count or quota."""
    similarity = normalize(query) @ head["vectors"].T
    scores = np.full((len(query), class_count), -np.inf, dtype=np.float32)
    for index in range(class_count):
        scores[:, index] = np.max(similarity[:, head["owners"] == index], axis=1)
    return np.argmax(scores, axis=1).astype(np.int64)


def loo(head: dict[str, Any], truth: np.ndarray, old_count: int) -> dict[str, Any]:
    similarity = head["vectors"] @ head["vectors"].T
    np.fill_diagonal(similarity, -np.inf)
    class_count = int(np.max(truth)) + 1
    scores = np.full((len(truth), class_count), -np.inf, dtype=np.float32)
    for index in range(class_count):
        scores[:, index] = np.max(similarity[:, truth == index], axis=1)
    return metrics(np.argmax(scores, axis=1), truth, old_count)


def metrics(pred: np.ndarray, truth: np.ndarray, old_count: int) -> dict[str, Any]:
    old = truth < old_count
    new = ~old
    old_acc = float(np.mean(pred[old] == truth[old]))
    new_acc = float(np.mean(pred[new] == truth[new])) if np.any(new) else 0.0
    harmonic = 2 * old_acc * new_acc / (old_acc + new_acc) if old_acc + new_acc else 0.0
    per_class = {str(i): float(np.mean(pred[truth == i] == i)) for i in sorted(set(truth.tolist()))}
    return {
        "old_accuracy": old_acc,
        "seen_new_accuracy": new_acc,
        "H_old_new": float(harmonic),
        "old_floor": min(per_class[str(i)] for i in range(old_count)),
        "seen_new_floor": min((per_class[str(i)] for i in range(old_count, int(np.max(truth)) + 1)), default=0.0),
        "per_class_index": per_class,
    }


def pooled_from_rates(scene_metrics: list[dict[str, Any]]) -> dict[str, Any]:
    pred, truth = [], []
    for row in scene_metrics:
        for key, accuracy in row["per_class_index"].items():
            label = int(key)
            correct = int(round(accuracy * K_SHOT))
            pred.extend([label] * correct + [999] * (K_SHOT - correct))
            truth.extend([label] * K_SHOT)
    return metrics(np.asarray(pred), np.asarray(truth), OLD_COUNT)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capsule-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    capsule = Path(args.capsule_root).resolve()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    patch_dir = output / "fp16_delta_patches"
    patch_dir.mkdir(exist_ok=True)
    device = torch.device(args.device)
    runtime_path = capsule / "predictor" / "after" / "enrollment_only" / "sealed_feature_runtime.pt"
    whitelist_audit = {}
    probe = torch.jit.load(str(runtime_path), map_location=device)
    probe_named = dict(probe.named_parameters())
    for candidate, names in WHITELISTS.items():
        whitelist_audit[candidate] = {"exact_layer_names": list(names), "updated_original_parameters": sum(probe_named[name].numel() for name in names)}
        if whitelist_audit[candidate]["updated_original_parameters"] >= 50000:
            raise RuntimeError("parameter cap violated")
    del probe
    support_data = {}
    for scenario in SCENARIOS:
        path = capsule / "predictor" / "after" / "enrollment_only" / f"support_{scenario}.npz"
        with np.load(path, allow_pickle=False) as archive:
            iq = np.asarray(archive["support_leo_weak_iq"], dtype=np.float32)
            truth = np.asarray(archive["support_class_indices"], dtype=np.int64)
        if len(iq) != CLASS_COUNT * K_SHOT or not np.array_equal(np.bincount(truth), np.full(CLASS_COUNT, K_SHOT)):
            raise RuntimeError("formal K10/new5 support drift")
        support_data[scenario] = {"iq": iq, "truth": truth, "path": str(path), "sha256": sha256(path)}

    training_log_path = output / "training_log.jsonl"
    patches: dict[tuple[str, str, str], dict[str, np.ndarray]] = {}
    support_results: dict[str, Any] = {}
    with training_log_path.open("w", encoding="utf-8", newline="\n") as log_handle:
        for candidate, names in WHITELISTS.items():
            scenes = []
            after_metrics = []
            max_total_state = 0
            for scenario in SCENARIOS:
                data = support_data[scenario]
                state_metrics = {}
                state_audits = {}
                for state, count in (("before", OLD_COUNT), ("after", CLASS_COUNT)):
                    mask = data["truth"] < count
                    patch, audit = train_patch(
                        runtime_path, names, data["iq"][mask], data["truth"][mask], device,
                        log_handle, {"candidate": candidate, "scenario": scenario, "registration_state": state},
                    )
                    patches[(candidate, scenario, state)] = patch
                    patch_path = patch_dir / f"{candidate}__{scenario}__{state}.npz"
                    np.savez_compressed(patch_path, **{name.replace(".", "__"): value for name, value in patch.items()})
                    audit["fp16_patch_file_bytes"] = patch_path.stat().st_size
                    model = apply_patch(runtime_path, names, patch, device)
                    z = runtime_z(model, data["iq"][mask], device)
                    feature = fused(z, data["iq"][mask])
                    head = quantized_head(feature, data["truth"][mask])
                    scored = loo(head, data["truth"][mask], OLD_COUNT)
                    audit["int8_head_bytes"] = head["state_bytes"]
                    audit["patch_plus_head_bytes"] = audit["fp16_patch_payload_bytes"] + head["state_bytes"]
                    audit["state_within_256KB"] = audit["patch_plus_head_bytes"] <= STATE_CAP
                    audit["knn_dot_macs_per_query"] = int(len(feature) * FEATURE_DIM)
                    if not audit["state_within_256KB"] or audit["optimizer_steps"] > 50 or audit["epochs"] > 5:
                        raise RuntimeError("resource contract violation")
                    state_metrics[state] = scored
                    state_audits[state] = audit
                    max_total_state = max(max_total_state, audit["patch_plus_head_bytes"])
                    del model
                    torch.cuda.empty_cache()
                forgetting = state_metrics["before"]["old_accuracy"] - state_metrics["after"]["old_accuracy"]
                scenes.append({"scenario": scenario, "before": state_metrics["before"], "after": state_metrics["after"], "forgetting": forgetting, "resource": state_audits})
                after_metrics.append(state_metrics["after"])
            pooled_after = pooled_from_rates(after_metrics)
            pooled_before_old = float(np.mean([row["before"]["old_accuracy"] for row in scenes]))
            support_results[candidate] = {
                "scenes": scenes,
                "pooled_after": pooled_after,
                "pooled_before_old_accuracy": pooled_before_old,
                "pooled_forgetting": pooled_before_old - pooled_after["old_accuracy"],
                "max_patch_plus_head_bytes": max_total_state,
            }
    ranking = sorted(
        WHITELISTS,
        key=lambda name: (
            min(support_results[name]["pooled_after"]["old_floor"], support_results[name]["pooled_after"]["seen_new_floor"]),
            support_results[name]["pooled_after"]["H_old_new"],
            -support_results[name]["pooled_forgetting"],
            -support_results[name]["max_patch_plus_head_bytes"],
            -list(WHITELISTS).index(name),
        ),
        reverse=True,
    )
    selector = {
        "selected_candidate": ranking[0],
        "ranking": ranking,
        "evidence": "three_scenario_support_only_post_adaptation_loo",
        "selection_key": ["max_min_old_new_floor", "max_H", "min_forgetting", "min_state", "fixed_order"],
        "query_opened_for_selection": False,
        "same_whitelist_and_hyperparameters_across_scenes": True,
    }
    (output / "selector_lock.json").write_text(json.dumps(selector, indent=2), encoding="utf-8")

    selected = ranking[0]
    prediction_dir = output / "sealed_predictions"
    prediction_dir.mkdir(exist_ok=True)
    # Predictor phase: the scorer sidecar is not opened until all three artifacts close.
    for scenario in SCENARIOS:
        support = support_data[scenario]
        query_path = capsule / "predictor" / "after" / "apply_only_staging" / f"query_{scenario}.npz"
        with np.load(query_path, allow_pickle=False) as archive:
            query_iq = np.asarray(archive["query_leo_weak_iq"], dtype=np.float32)
            query_tokens = np.asarray(archive["query_tokens"]).astype(str)
        before_mask = support["truth"] < OLD_COUNT
        before_model = apply_patch(runtime_path, WHITELISTS[selected], patches[(selected, scenario, "before")], device)
        before_support_feature = fused(runtime_z(before_model, support["iq"][before_mask], device), support["iq"][before_mask])
        before_query_feature = fused(runtime_z(before_model, query_iq, device), query_iq)
        before_head = quantized_head(before_support_feature, support["truth"][before_mask])
        before_pred = predict(before_head, before_query_feature, OLD_COUNT)
        after_model = apply_patch(runtime_path, WHITELISTS[selected], patches[(selected, scenario, "after")], device)
        after_support_feature = fused(runtime_z(after_model, support["iq"], device), support["iq"])
        after_query_feature = fused(runtime_z(after_model, query_iq, device), query_iq)
        after_head = quantized_head(after_support_feature, support["truth"])
        after_pred = predict(after_head, after_query_feature, CLASS_COUNT)
        np.savez_compressed(prediction_dir / f"{scenario}.npz", query_token=query_tokens, before_predicted_class_index=before_pred, after_predicted_class_index=after_pred)
        del before_model, after_model
        torch.cuda.empty_cache()

    # Scorer phase: only now may query truth be opened and joined to sealed predictions.
    query_scene_rows = []
    truth_path = capsule / "scorer" / "truth_sidecar.json"
    truth_document = json.loads(truth_path.read_text(encoding="utf-8"))
    truth_map = {row["query_token"]: row for row in truth_document["rows"]}
    pooled_before_pred, pooled_before_truth, pooled_after_pred, pooled_after_truth = [], [], [], []
    for scenario in SCENARIOS:
        with np.load(prediction_dir / f"{scenario}.npz", allow_pickle=False) as artifact:
            query_tokens = np.asarray(artifact["query_token"]).astype(str)
            before_pred = np.asarray(artifact["before_predicted_class_index"], dtype=np.int64)
            after_pred = np.asarray(artifact["after_predicted_class_index"], dtype=np.int64)
        truth = np.asarray([int(truth_map[token]["true_class_index"]) for token in query_tokens], dtype=np.int64)
        old_mask = truth < OLD_COUNT
        before_old = float(np.mean(before_pred[old_mask] == truth[old_mask]))
        after_scored = metrics(after_pred, truth, OLD_COUNT)
        query_scene_rows.append({"scenario": scenario, "before_old_accuracy": before_old, "after": after_scored, "forgetting": before_old - after_scored["old_accuracy"]})
        pooled_before_pred.append(before_pred[old_mask]); pooled_before_truth.append(truth[old_mask])
        pooled_after_pred.append(after_pred); pooled_after_truth.append(truth)
    pooled_after = metrics(np.concatenate(pooled_after_pred), np.concatenate(pooled_after_truth), OLD_COUNT)
    pooled_before_old = float(np.mean(np.concatenate(pooled_before_pred) == np.concatenate(pooled_before_truth)))
    query_result = {
        "selected_candidate": selected,
        "scenes": query_scene_rows,
        "pooled_before_old_accuracy": pooled_before_old,
        "pooled_after": pooled_after,
        "pooled_forgetting": pooled_before_old - pooled_after["old_accuracy"],
    }
    payload = {
        "status": "COMPLETE_STAGE2C_DEVELOPMENT_SCREEN",
        "method": "support-only sparse key-layer delta",
        "receiver": "20-1",
        "seed": 713101,
        "k_shot": K_SHOT,
        "seen_new_count": 5,
        "runtime": {"path": str(runtime_path), "sha256": sha256(runtime_path)},
        "whitelist_audit": whitelist_audit,
        "fixed_hyperparameters": {"epochs": EPOCHS, "optimizer_steps": EPOCHS, "optimizer": "SGD", "lr": LR, "momentum": 0.0, "prototype_ce_temperature_scale": 12.0, "old_pairwise_retention_weight": 1.0, "new_separation_weight": 0.25, "new_separation_cosine_margin": 0.20},
        "support_selection": support_results,
        "selector_lock": selector,
        "query_score": query_result,
        "truth_sidecar": {"path": str(truth_path), "sha256": sha256(truth_path), "opened_after_prediction": True},
        "protocol": {"stage": "stage2c", "phase2_sample_view_policy": "leo_weak_only_no_clean_access", "support_only_calibration": True, "query_decision_policy": "per_sample_all_registered_classes", "query_role_oracle_access": False, "query_true_batch_class_count_access": False, "query_class_quota_access": False, "query_batch_global_assignment": False},
    }
    (output / "results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(query_result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
