"""Matched D92 true-256 registration; no query truth is accepted by this module.

Native checkpoint extraction must run in a separate process using the native
Phase1 code tree. This process imports the frozen root D92 implementation.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
import threading

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
sys.path.insert(0, str(ROOT / "tools"))
import numpy as np
import torch

ARM = "P2-256-FULL"
_FIT_LOCK = threading.RLock()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write(path, data):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def registered_features(iq, identity160):
    """Historical 288-column transport; D92 projects to real 256 internally."""
    from cvsrffi.stage2_diag_cosine_exploration import registered_feature
    identity160 = np.asarray(identity160, dtype=np.float32)
    if identity160.ndim != 2 or identity160.shape[1] != 160:
        raise ValueError("Expected identity160 features")
    return registered_feature(np.asarray(iq, dtype=np.float32), identity160)


@contextmanager
def _old_only_registry(executor, old_only):
    """Skip only the empty-new input assembly, keeping all numerical choices.

The frozen executor already has an old-only assembly path. Temporarily change
only this arm's stage descriptor under the process lock; retain ARM, feature,
metric, D92/D81 fallback, D46/D62 and F3 choices. Restore even on failure.
"""
    original = executor.get_stage2_arm
    from scripts import probe_d92_registration_balanced_covariance as probe
    from cvs_d92_covariance_matched import build_registration_balanced_equal_lda
    original_covariance = probe.build_registration_balanced_equal_lda
    probe.build_registration_balanced_equal_lda = build_registration_balanced_equal_lda
    if old_only:
        executor.get_stage2_arm = lambda arm: (
            replace(original(arm), stage="stage2b") if arm == ARM else original(arm)
        )
    try:
        yield
    finally:
        executor.get_stage2_arm = original
        probe.build_registration_balanced_equal_lda = original_covariance


def fit_d92(*, support_features, support_labels, classes, ground,
            seed, old_class_count=6, device="cpu"):
    """Fit from support only; labels are integer indices into old-first classes."""
    from cvsrffi import stage2_ablation_executors as executor
    classes = tuple(str(value) for value in classes)
    rows = np.asarray(support_features, dtype=np.float32)
    labels = np.asarray(support_labels)
    if (len(classes) < 6 or old_class_count != 6 or len(set(classes)) != len(classes)
            or rows.ndim != 2 or rows.shape[1] != 288
            or not np.isfinite(rows).all() or labels.shape != (len(rows),)
            or not np.issubdtype(labels.dtype, np.integer)
            or set(labels.tolist()) != set(range(len(classes)))):
        raise ValueError("D92 support/registry/288-column transport contract mismatch")
    counts = np.bincount(labels, minlength=len(classes))
    if np.any(counts != counts[0]) or counts[0] < 1:
        raise ValueError("D92 requires equal positive K for every registered class")
    old = labels < old_class_count
    names = np.asarray(classes)[labels]
    with _FIT_LOCK, _old_only_registry(executor, len(classes) == 6):
        state = executor.fit_stage2_ablation(
            ablation_id=ARM,
            old_support_features=rows[old], old_support_labels=names[old],
            old_classes=classes[:6],
            new_support_features=rows[~old] if len(classes) > 6 else None,
            new_support_labels=names[~old] if len(classes) > 6 else None,
            new_classes=classes[6:], seed=int(seed), device=device,
            ground_basis=ground["ground_basis"],
            ground_spectral_weights=ground["ground_spectral_weights"],
            ground_audit=ground["ground_audit"],
        )
    if (state.active_feature_dim != 256 or state.classes != classes
            or state.compiled_affine_state is None
            or state.score_kind != "compiled_affine"):
        raise ValueError("D92 true-256/F3 state contract drift")
    return state


def predict_scores(state, query_features):
    """Independent rows, all registered classes; no fitting or global decision."""
    rows = np.asarray(query_features, dtype=np.float32)
    if rows.ndim != 2 or rows.shape[1] != 288 or not np.isfinite(rows).all():
        raise ValueError("Expected finite query features [N,288]")
    scores = state.score(rows)
    if scores.shape != (len(rows), len(state.classes)) or not np.isfinite(scores).all():
        raise ValueError("D92 query score shape/value drift")
    return scores


def build_ground(*, features_path, source_run, output_dir):
    """Offline L-only aggregate export. U/V IDs or old checkpoints fail closed.

NPZ: identity160, labels (0..5), domains (contiguous), ids (physical IDs).
    Existing source run records establish scratch/final/source-only provenance.
"""
    from cvsrffi.phase1_geometry_streaming import Phase1GeometryStreaming
    from cvsrffi import phase1_center_lowrank_prototype_bundle as codec
    from scripts.export_adv3b02_center_lowrank_radius_component import build_v1_aggregate_payload
    source_run = Path(source_run)
    contract = read(source_run / "source_contract.json")
    init = read(source_run / "initialization.json")
    completion = read(source_run / "completion.json")
    checkpoint_path = source_run / "final_ssdg.pth"
    if (init.get("scratch_only") is not True or init.get("checkpoint_sources") != []
            or init.get("target_contact") is not False
            or init.get("target_training_contact") is not False
            or init.get("source_roles") != "EXACT_MATCH"
            or contract.get("native_role_comparison") != "EXACT_MATCH"
            or completion.get("epochs") != 200 or completion.get("target_evaluated") is not False
            or completion.get("status") != "TRAINING_COMPLETE"
            or completion.get("checkpoint") != "final_ssdg.pth"):
        raise ValueError("Ground checkpoint/data provenance mismatch")
    classes = tuple(contract.get("classes", ()))
    if len(classes) != 6 or len(set(classes)) != 6:
        raise ValueError("Ground requires six bound source TX classes")
    with np.load(features_path, allow_pickle=False) as data:
        if set(data.files) != {"identity160", "labels", "domains", "ids"}:
            raise ValueError("Ground source feature member drift")
        x = np.asarray(data["identity160"], dtype=np.float32)
        y, domains, ids = data["labels"], data["domains"], data["ids"].astype(str)
    expected_ids = contract["role_ids"]["L_s"]
    if (len(ids) != len(set(ids.tolist())) or set(ids.tolist()) != set(expected_ids)
            or x.shape != (len(ids), 160) or not np.isfinite(x).all()
            or y.shape != (len(ids),) or domains.shape != (len(ids),)
            or not np.issubdtype(y.dtype, np.integer) or not np.issubdtype(domains.dtype, np.integer)
            or set(y.tolist()) != set(range(6))
            or set(domains.tolist()) != set(range(int(domains.max()) + 1))):
        raise ValueError("Ground must contain exactly the source L_s physical records")
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    if np.any(norm < 1e-6):
        raise ValueError("Degenerate identity feature")
    x = x / norm
    domain_count = int(domains.max()) + 1
    stream = Phase1GeometryStreaming(num_domains=domain_count, num_classes=6, feature_dim=160,
                                    radius_histogram_bins=4096)
    tx = torch.tensor(x.tolist(), dtype=torch.float32)
    ty = torch.tensor(y.tolist(), dtype=torch.long)
    td = torch.tensor(domains.tolist(), dtype=torch.long)
    stream.update_first_pass(tx, ty, td)
    stream.begin_second_pass()
    stream.update_second_pass(tx, ty, td)
    geometry = stream.finalize()
    # The same v2 encoder and 4096-bin P90 aggregation used by the frozen method.
    # Dense v1 is only an in-memory codec input; it is not a Phase2 artifact.
    dense = build_v1_aggregate_payload(geometry, domain_registry=[str(i) for i in range(domain_count)],
                                      class_registry=classes)
    payload, audit = codec.compress_v1_dense_component(dense,
        radius_p90_cosine_distance=np.asarray(geometry.radius_p90_cosine_distance.tolist(), dtype=np.float32),
        formal_phase2_eligible=True)
    out = Path(output_dir)
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True, exist_ok=False)
    npz_path = out / codec.NPZ_NAME
    np.savez_compressed(npz_path, **payload)
    # Current minimal-workflow metadata, deliberately not a forged historical
    # outer-signature manifest. Numeric payload is the exact v2 codec format.
    manifest = dict(schema=codec.SCHEMA, metadata_schema="cvs.matched.ground.v2",
        checkpoint_sha256=_sha(checkpoint_path), source_prototype_artifact_sha256=_sha(features_path),
        provenance_status="CURRENT_FINAL_SCRATCH_EXACT_SOURCE_L", formal_phase2_eligible=True,
        feature_dim=160, residual_rank=3, radius_histogram_bins=4096,
        member_allowlist=[codec.NPZ_NAME], npz_member_allowlist=sorted(codec.ALLOWED_NPZ_MEMBERS),
        resource_audit=audit, source_role="L_s", target_access=False,
        component_state="CURRENT_MATCHED_SOURCE_ONLY_V2", historical_outer_signature_claim=False)
    _write(out / "manifest.json", manifest)
    return dict(npz_path=str(npz_path), manifest_path=str(out / "manifest.json"), schema=codec.SCHEMA)


def load_ground(component_dir):
    from cvsrffi import phase1_center_lowrank_prototype_bundle as codec
    from cvsrffi.stage2_d89_v2_radius_cauchy_center import radius_reliability_ground_spectrum
    path = Path(component_dir)
    manifest = read(path / "manifest.json")
    if (manifest.get("schema") != codec.SCHEMA or manifest.get("metadata_schema") != "cvs.matched.ground.v2"
            or manifest.get("formal_phase2_eligible") is not True or
            manifest.get("provenance_status") != "CURRENT_FINAL_SCRATCH_EXACT_SOURCE_L"):
        raise ValueError("Only the current matched-run ground component is accepted")
    with np.load(path / codec.NPZ_NAME, allow_pickle=False) as data:
        payload = {name: data[name] for name in data.files}
    details = codec._validate_payload(payload)
    component = codec.CenterLowRankPrototypeComponent(
        **{name: payload[name] for name in ("core_q", "core_scale", "residual_basis_q", "residual_basis_scale",
            "residual_coeff_q", "residual_coeff_scale", "radius_q", "radius_scale")},
        domain_registry=details["domains"], residual_domain_registry=details["residual_domains"],
        class_registry=details["classes"], center_domain_handle=details["center"], manifest=manifest)
    centers = np.stack([component.reconstruct_domain(domain) for domain in component.domain_registry])
    radius = np.stack([component.radius_for_domain(domain) for domain in component.domain_registry])
    basis, weights, spectrum = radius_reliability_ground_spectrum(
        centers, radius, float(manifest["resource_audit"]["reconstruction_rmse"]))
    audit = dict(spectrum, **{f"d81_{key}": value for key, value in spectrum.items()})
    audit.update(ground_component_input_count=int(radius.size),
        ground_statistic_semantics="v2_cell_radius_reliability_ground_spectrum_for_d81_cauchy_center",
        ground_bundle_contains_sample_radius=False, ground_bundle_contains_aggregated_p90_radius=True,
        ground_bundle_contains_sample_count=False, dense_ground_bank_persisted=False)
    return dict(ground_basis=basis, ground_spectral_weights=weights, ground_audit=audit)


def predict_capsule(*, features_path, capsule, ground_dir, output_dir, seed):
    """One completed prediction stream compatible with the baseline scorer."""
    from cvsrffi.stage2_ablation_factory import resolve_stage2_config
    import os
    import platform
    import time
    capsule, out = Path(capsule), Path(output_dir)
    manifest = read(capsule / "manifest.json")
    if (manifest.get("protocol_schema") != "p2_min_v1"
            or manifest.get("phase2_data_status") != "VALIDATED_ONCE"):
        raise ValueError("Expected existing validated shared Phase2 capsule")
    if any((out / name).exists() for name in ("d92_startup.json", "predictions.jsonl", "predictions_complete.json")):
        raise FileExistsError("D92 output artifacts already exist")
    ground_manifest = read(Path(ground_dir) / "manifest.json")
    from cvsrffi.phase1_center_lowrank_prototype_bundle import NPZ_NAME
    with np.load(Path(ground_dir) / NPZ_NAME, allow_pickle=False) as data:
        old_classes = data["class_registry"].astype(str).tolist()
    with np.load(features_path, allow_pickle=False) as data:
        if set(data.files) != {"identity160", "logits", "ids"}:
            raise ValueError("Expected frozen identity160/logits/ids feature artifact")
        identity, logits, ids = data["identity160"], data["logits"], data["ids"].astype(str)
    with np.load(capsule / "received.npz", allow_pickle=False) as data:
        iq, capsule_ids = data["iq"], data["ids"].astype(str)
    if (not np.array_equal(ids, capsule_ids) or identity.shape != (len(ids), 160)
            or logits.shape != (len(ids), 6) or not np.isfinite(logits).all()):
        raise ValueError("Frozen feature/capsule alignment drift")
    ground = load_ground(ground_dir)
    features = registered_features(iq, identity)
    out.mkdir(parents=True, exist_ok=True)
    _write(out / "d92_startup.json", dict(command=sys.argv, seed=int(seed), pid=os.getpid(),
        python=sys.version, platform=platform.platform(), numpy=np.__version__, torch=torch.__version__,
        features=str(features_path), capsule=str(capsule), ground=str(ground_dir),
        checkpoint_sha256=ground_manifest["checkpoint_sha256"], resolved_config=resolve_stage2_config(ARM),
        new_count_zero_policy="same_D92_old_only_before_exact_d81", new_count_two_policy="same_covariance_added_registry_allowance",
        query_fit_access=False, truth_read=False))
    count, dg_seen = 0, set()
    paths = sorted((capsule / "splits").glob("*.json"))
    if not paths:
        raise ValueError("No capsule split files")
    with (out / "predictions.jsonl").open("x", encoding="utf-8") as stream:
        def save(split, mode, scores):
            nonlocal count
            record = dict(split_id=split["split_id"], capsule_id=manifest["capsule_id"], mode=mode,
                receiver=split["receiver"], scenario=split["scenario"], k=split["k"], support_seed=split["support_seed"],
                classes=split["registered_classes"], query_ids=ids[split["query_indices"]].tolist(),
                predicted_indices=scores.argmax(1).tolist(), scores=scores.tolist())
            stream.write(json.dumps(record) + "\n")
            stream.flush()
            count += 1
        for path in paths:
            split = read(path)
            classes = split["registered_classes"]
            s, q = split["support_indices"], split["query_indices"]
            if (split["capsule_id"] != manifest["capsule_id"] or classes[:6] != old_classes
                    or set(s) & set(q) or len(set(s)) != len(s) or len(set(q)) != len(q)
                    or min(s + q) < 0 or max(s + q) >= len(ids)):
                raise ValueError("Split identity, registry, or support/query boundary drift")
            started = time.monotonic()
            state = fit_d92(support_features=features[s], support_labels=split["support_labels"],
                classes=classes, ground=ground, seed=seed)
            scores = predict_scores(state, features[q])
            save(split, "d92_registration", scores)
            key = (split["receiver"], split["scenario"])
            if classes == old_classes and key not in dg_seen:
                save(split, "frozen_dg", logits[q])
                dg_seen.add(key)
            print(json.dumps(dict(split_id=split["split_id"], predictions=count,
                elapsed_seconds=time.monotonic() - started, feature_dim=state.active_feature_dim)), flush=True)
    result = dict(status="PREDICTIONS_COMPLETE", predictions=count, split_count=len(paths),
                  capsule_id=manifest["capsule_id"], truth_read=False)
    _write(out / "predictions_complete.json", result)
    return result


def runtime_smoke():
    """Synthetic runtime check with no pytest or real target records."""
    import tempfile
    torch.set_num_threads(1)
    rng = np.random.default_rng(9413)
    with tempfile.TemporaryDirectory(prefix="cvs_d92_synthetic_") as temporary:
        root = Path(temporary)
        labels = np.repeat(np.arange(6), 8)
        domains = np.tile(np.repeat(np.arange(4), 2), 6)
        ids = np.asarray([f"synthetic-L-{i}" for i in range(48)])
        source = root / "source_features.npz"
        np.savez(source, identity160=rng.normal(size=(48, 160)).astype(np.float32),
                 labels=labels, domains=domains, ids=ids)
        classes = [f"synthetic-class-{i}" for i in range(6)]
        _write(root / "source_contract.json", dict(role_ids=dict(L_s=ids.tolist()),
            native_role_comparison="EXACT_MATCH", classes=classes))
        _write(root / "initialization.json", dict(scratch_only=True, checkpoint_sources=[],
            target_contact=False, target_training_contact=False, source_roles="EXACT_MATCH"))
        _write(root / "completion.json", dict(status="TRAINING_COMPLETE", epochs=200,
            checkpoint="final_ssdg.pth", target_evaluated=False))
        # A synthetic marker is sufficient here: this check exercises the codec,
        # not native model reconstruction (checked independently before launch).
        (root / "final_ssdg.pth").write_bytes(b"SYNTHETIC_RUNTIME_SMOKE_NO_REAL_MODEL")
        ground_path = root / "ground"
        exported = build_ground(features_path=source, source_run=root, output_dir=ground_path)
        ground = load_ground(ground_path)
        manifest = read(exported["manifest_path"])
        if (manifest["feature_dim"] != 160 or manifest["residual_rank"] != 3
                or not ground["ground_audit"]["ground_bundle_contains_aggregated_p90_radius"]):
            raise AssertionError("Synthetic v2 ground dimensions/radius drift")
        checks = []
        for new_count, k in ((0, 1), (2, 5)):
            registry = classes + [f"synthetic-new-{i}" for i in range(new_count)]
            support_y = np.repeat(np.arange(len(registry)), k)
            support = registered_features(
                rng.normal(size=(len(support_y), 2, 256)).astype(np.float32),
                rng.normal(size=(len(support_y), 160)).astype(np.float32))
            state = fit_d92(support_features=support, support_labels=support_y,
                           classes=registry, ground=ground, seed=9413)
            query = registered_features(rng.normal(size=(4, 2, 256)).astype(np.float32),
                                        rng.normal(size=(4, 160)).astype(np.float32))
            batch = predict_scores(state, query)
            individual = np.concatenate([predict_scores(state, row[None]) for row in query])
            np.testing.assert_allclose(batch, individual, rtol=2e-5, atol=2e-4)
            np.testing.assert_array_equal(batch.argmax(1), individual.argmax(1))
            if state.active_feature_dim != 256 or state.resource["query_rows_used_for_fit"] != 0:
                raise AssertionError("Synthetic D92 dimension/query boundary drift")
            if new_count and state.audit["d92_registration_balanced_active"] is not True:
                raise AssertionError("Synthetic new2/K5 covariance was not active")
            checks.append(dict(new_classes=new_count, k=k, feature_dim=256,
                               batch_individual_argmax_equal=True))
    return dict(status="PASS", synthetic_only=True, real_query_access=False,
                ground_schema=manifest["schema"], ground_feature_dim=160,
                residual_rank=3, checks=checks, python=sys.version,
                numpy=np.__version__, torch=torch.__version__)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("smoke")
    ground = commands.add_parser("ground")
    for name in ("features", "source-run", "output"):
        ground.add_argument("--" + name, required=True, type=Path)
    predict = commands.add_parser("predict")
    for name in ("features", "capsule", "ground", "output"):
        predict.add_argument("--" + name, required=True, type=Path)
    predict.add_argument("--seed", required=True, type=int)
    args = parser.parse_args(argv)
    if args.command == "smoke":
        result = runtime_smoke()
    elif args.command == "ground":
        result = build_ground(features_path=args.features, source_run=args.source_run, output_dir=args.output)
    else:
        torch.set_num_threads(1)
        result = predict_capsule(features_path=args.features, capsule=args.capsule,
            ground_dir=args.ground, output_dir=args.output, seed=args.seed)
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
