from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import stat
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import pytest

from cvsrffi.stage2_d105_target25_launcher import (
    CONTEXT_MANIFEST_SCHEMA,
    D105Target25LauncherError,
    execute_d105_target25_with_evaluator,
    load_d105_target25_context_factory,
    seal_d105_target25_context_manifest,
)
from cvsrffi.stage2_d105_target25_runner import (
    ARMS,
    D105Target25GPUSchedule,
    D105Target25PredictionOutput,
    D105Target25RunnerError,
    D105Target25ScenarioPlan,
    D105Target25StatePlan,
    TARGET25_SEED,
    TARGET25_SLICES,
    canonical_sha256,
    freeze_d105_target25_plan,
    load_d105_target25_plan_manifest,
    prepare_d105_target25_run,
    verify_d105_target25_prediction_manifest,
    write_d105_target25_plan_manifest,
)
from cvsrffi.stage2_zid_student_t_qknn import Phase1ZIDStudentTLock
from cvsrffi.stage2_d105_phase1_bundle import (
    D105_CANDIDATE_RUNTIME_ENTRYPOINTS,
    D105_CANDIDATE_RUNTIME_FILES,
)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


_CANDIDATE_CHECKPOINT_SHA = hashlib.sha256(b"exact-test-checkpoint").hexdigest()
_CODE_ROOT = Path(__file__).resolve().parents[1] / "code"
_CANDIDATE_CORE_FILES = D105_CANDIDATE_RUNTIME_FILES
_CANDIDATE_RUNTIME_DOCUMENT = {
    "schema": "cvs.stage2.d105.candidate_runtime_manifest.v1",
    "candidate_id": "D105-CBRC+LPO-RC",
    "protocol_schema": "p2_min_v1",
    "checkpoint_sha256": _CANDIDATE_CHECKPOINT_SHA,
    "entrypoints": dict(D105_CANDIDATE_RUNTIME_ENTRYPOINTS),
    "core_file_sha256": {
        relative: hashlib.sha256((_CODE_ROOT / relative).read_bytes()).hexdigest()
        for relative in _CANDIDATE_CORE_FILES
    },
}
_CANDIDATE_RUNTIME_SHA = canonical_sha256(_CANDIDATE_RUNTIME_DOCUMENT)
_CANDIDATE_LOCK_DOCUMENT = {
    "schema": "cvs.stage2.d105.candidate_method_lock.v1",
    "candidate_id": "D105-CBRC+LPO-RC",
    "protocol_schema": "p2_min_v1",
    "checkpoint_sha256": _CANDIDATE_CHECKPOINT_SHA,
    "d105_candidate_runtime_manifest_sha256": _CANDIDATE_RUNTIME_SHA,
    "d105_cbrc": {
        "semantic_revision": "cbrc_mb4_task_balanced_huber_irls4_fp16_v1",
        "code_dim": 4,
        "domain_dim": 32,
        "allowed_k": [1, 5, 10],
        "irls_steps": 4,
        "old_new_task_mass": [0.5, 0.5],
        "k1_zero_coefficient": True,
        "ground_old_multiprototype_enabled": False,
        "deployment_coefficient_dtype": "float16",
        "query_transform": "relu_l2norm_pre_relu_plus_mb4",
        "query_state_updates": 0,
    },
    "student_t_qknn": {
        "student_nu": 3.0,
        "kernel_effective_dim": 12,
        "kernel_volume_gamma": 1.0,
        "shared_h0": 0.35,
        "scale_prior_strength": 2.0,
        "scale_min_ratio": 0.5,
        "scale_max_ratio": 2.0,
        "temperature": 0.85,
        "support_storage": "int8_fp16_scale",
    },
    "four_arm": {
        "arms": ["M0", "M_DA", "M_HEAD", "M_JOINT"],
        "same_da_state_for_da_and_joint": True,
        "same_head_code_config_for_head_and_joint": True,
        "query_truth_surface": False,
    },
    "source_held": {
        "receiver_held_k": [1, 5, 10],
        "class_loco_k": 1,
        "tx_probe_algorithm": "receiver_held_ridge_l2_0.01",
        "tx_probe_max_balanced_accuracy": 0.25,
        "int8_min_top1_agreement": 0.995,
        "large_margin_minimum": 0.10,
        "large_margin_flip_max": 0,
        "truth_open_after_prediction": True,
    },
    "target25": {
        "seed": 713102,
        "claim_scope": "DEVELOPMENT_SCREEN_ONLY_NON_PROMOTABLE",
        "formal_launch_authority": False,
        "slices": [[10, 5], [10, 10], [10, 20], [5, 20], [1, 20]],
        "leo_scenarios": [
            "leo_clear_weak",
            "leo_low_elev_weak",
            "leo_rain_weak",
        ],
        "outer_row_count": 25,
        "scenario_arm_pair_count": 300,
        "state_prediction_surface_count": 600,
    },
}
_CANDIDATE_LOCK_SHA = canonical_sha256(_CANDIDATE_LOCK_DOCUMENT)
_CANDIDATE_FILES = TemporaryDirectory()
_CANDIDATE_ROOT = Path(_CANDIDATE_FILES.name)
_CANDIDATE_RUNTIME_PATH = _CANDIDATE_ROOT / "candidate_runtime.json"
_CANDIDATE_LOCK_PATH = _CANDIDATE_ROOT / "candidate_lock.json"
for _path, _document in (
    (_CANDIDATE_RUNTIME_PATH, _CANDIDATE_RUNTIME_DOCUMENT),
    (_CANDIDATE_LOCK_PATH, _CANDIDATE_LOCK_DOCUMENT),
):
    _path.write_bytes(
        json.dumps(
            _document,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    _path.chmod(stat.S_IREAD)


def _state(
    receiver: str,
    k_shot: int,
    new_count: int,
    scenario: str,
    stage: str,
) -> D105Target25StatePlan:
    old = ("old-a", "old-b")
    new = tuple(f"new-{index:02d}" for index in range(new_count))
    classes = old if stage == "S_B" else old + new
    old_query = tuple(
        f"oldq-{receiver}-{scenario}-n{new_count}-{index}" for index in range(2)
    )
    query = old_query if stage == "S_B" else old_query + tuple(
        f"newq-{receiver}-{scenario}-n{new_count}-{index}" for index in range(2)
    )
    return D105Target25StatePlan(
        stage=stage,
        capsule_id=_sha(f"capsule:{stage}:{receiver}:{scenario}:n{new_count}"),
        split_id=_sha(f"split:{stage}:{receiver}:{scenario}:k{k_shot}:n{new_count}"),
        authority_receipt_sha256=_sha(
            f"authority:{stage}:{receiver}:{scenario}:k{k_shot}:n{new_count}"
        ),
        authority_envelope_sha256=_sha(f"authority-envelope:{receiver}"),
        data_feature_runtime_sha256=_sha("d92-feature-runtime"),
        data_materialization_lock_sha256=_sha("d92-materialization-lock"),
        d105_candidate_runtime_manifest_sha256=_CANDIDATE_RUNTIME_SHA,
        d105_candidate_method_lock_sha256=_CANDIDATE_LOCK_SHA,
        support_physical_ids=tuple(
            f"{stage}-support-{receiver}-{scenario}-{class_id}-{shot}"
            for class_id in classes
            for shot in range(k_shot)
        ),
        query_physical_ids=query,
        registered_classes=classes,
        old_classes=old,
        new_classes=() if stage == "S_B" else new,
        prediction_context_sha256=_sha(
            f"context:{stage}:{receiver}:{scenario}:k{k_shot}:n{new_count}"
        ),
    )


def _plan():
    receivers = ("rx-a", "rx-b", "rx-c", "rx-d", "rx-e")
    scenarios = ("leo_clear_weak", "leo_low_elev_weak", "leo_rain_weak")
    scenario_plans = {}
    for receiver in receivers:
        for k_shot, new_count in TARGET25_SLICES:
            scenario_plans[(receiver, k_shot, new_count)] = tuple(
                D105Target25ScenarioPlan(
                    scenario=scenario,
                    before=_state(receiver, k_shot, new_count, scenario, "S_B"),
                    after=_state(receiver, k_shot, new_count, scenario, "S_C"),
                )
                for scenario in scenarios
            )
    return freeze_d105_target25_plan(
        candidate_runtime_manifest_path=_CANDIDATE_RUNTIME_PATH,
        candidate_method_lock_path=_CANDIDATE_LOCK_PATH,
        receivers=receivers,
        scenario_plans=scenario_plans,
    )


@dataclass(frozen=True)
class _FakeState:
    stage: str
    registration_state: str
    query_physical_ids: tuple[str, ...]


@dataclass(frozen=True)
class _FakePair:
    scenario: str
    before: _FakeState
    after: _FakeState


@dataclass(frozen=True)
class _FakeContext:
    row: object
    gpu_id: int


class _FakeCompleteEvaluation:
    def __init__(self, row) -> None:
        self.row = row
        self.receiver = row.receiver
        self.seed = TARGET25_SEED
        self.k_shot = row.k_shot
        self.scenario_pairs = tuple(
            _FakePair(
                scenario.scenario,
                _FakeState(
                    "S_B",
                    "BEFORE_REGISTRATION",
                    scenario.before.query_physical_ids,
                ),
                _FakeState(
                    "S_C",
                    "AFTER_REGISTRATION",
                    scenario.after.query_physical_ids,
                ),
            )
            for scenario in row.scenarios
        )

    def target25_output_for(self, request):
        scenario = next(
            item for item in self.row.scenarios if item.scenario == request.scenario
        )
        state = scenario.before if request.stage == "S_B" else scenario.after
        assert request.registration_state == state.registration_state
        assert request.query_physical_ids == state.query_physical_ids
        assert request.prediction_context_sha256 == state.prediction_context_sha256
        base = tuple(
            request.registered_classes[index % len(request.registered_classes)]
            for index in range(len(request.query_physical_ids))
        )
        return D105Target25PredictionOutput(
            stage=request.stage,
            registration_state=request.registration_state,
            arm_predictions={arm: base for arm in ARMS},
            state_receipt_sha256=_sha(
                f"state:{request.row_id}:{request.scenario}:{request.stage}"
            ),
            predictor_receipt_sha256=_sha(
                f"predictor:{request.row_id}:{request.scenario}:{request.stage}"
            ),
            feature_receipt_sha256=_sha(
                f"feature:{request.row_id}:{request.scenario}:{request.stage}"
            ),
            resource_receipt_sha256=_sha(
                f"resource:{request.gpu_id}:{request.worker_slot}"
            ),
            logit_sha256_by_arm={
                arm: _sha(
                    f"logit:{request.row_id}:{request.scenario}:{request.stage}:{arm}"
                )
                for arm in ARMS
            },
            arm_prediction_sha256_by_arm={
                arm: _sha(
                    f"top1:{request.row_id}:{request.scenario}:{request.stage}:{arm}"
                )
                for arm in ARMS
            },
        )


def _context_document(plan, root: Path) -> tuple[dict, dict]:
    checkpoint = root / "checkpoint.pth"
    checkpoint.write_bytes(b"exact-test-checkpoint")
    checkpoint_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    phase1_dir = root / "formal_phase1"
    phase1_dir.mkdir()
    phase1 = {
        "bundle_dir": str(phase1_dir.resolve()),
        "manifest_sha256": _sha("phase1-manifest"),
        "bundle_wire_sha256": _sha("phase1-wire"),
        "validated_bundle_id_sha256": _sha("phase1-validated"),
        "validator_receipt_sha256": _sha("phase1-validator"),
        "expected_content_root_sha256": _sha("phase1-content"),
        "checkpoint_sha256": checkpoint_sha,
        "candidate_runtime_manifest_path": str(_CANDIDATE_RUNTIME_PATH),
        "candidate_method_lock_path": str(_CANDIDATE_LOCK_PATH),
        "d105_candidate_runtime_manifest_sha256": plan.d105_candidate_runtime_manifest_sha256,
        "d105_candidate_method_lock_sha256": plan.d105_candidate_method_lock_sha256,
    }
    references = {}
    for name in ("before_enrollment", "before_apply", "after_enrollment", "after_apply"):
        package_root = root / f"{name}_package"
        package_root.mkdir()
        seal = root / f"{name}.seal"
        seal.write_text("detached seal", encoding="utf-8")
        policy = root / f"{name}.policy.json"
        policy.write_text("{}", encoding="utf-8")
        authorization = root / f"{name}.authorization.json"
        authorization.write_text("{}", encoding="utf-8")
        envelope = root / f"{name}.authorization-envelope.json"
        envelope.write_text("{}", encoding="utf-8")
        references[name] = {
            "package_root": str(package_root.resolve()),
            "detached_seal_path": str(seal.resolve()),
            "expected_seal_sha256": _sha(f"seal:{name}"),
            "formal_policy_path": str(policy.resolve()),
            "formal_policy_authorization_path": str(authorization.resolve()),
            "signed_policy_authorization_envelope_path": str(envelope.resolve()),
            "expected_signed_policy_authorization_envelope_sha256": (
                hashlib.sha256(envelope.read_bytes()).hexdigest()
            ),
        }
    rows = []
    for row in plan.rows:
        authorities = []
        for scenario in row.scenarios:
            for state in (scenario.before, scenario.after):
                authorities.append(
                    {
                        "registration_state": state.registration_state,
                        "scenario": scenario.scenario,
                        "capsule_id": state.capsule_id,
                        "split_id": state.split_id,
                        "validator_receipt_sha256": state.authority_receipt_sha256,
                        "support_token_root_sha256": state.support_physical_root_sha256,
                        "query_token_root_sha256": state.query_physical_root_sha256,
                        "protocol_schema": "p2_min_v1",
                        "phase2_data_status": "VALIDATED_ONCE",
                    }
                )
        lock = Phase1ZIDStudentTLock(
            active_k=row.k_shot,
            student_nu=3.0,
            kernel_effective_dim=12,
            kernel_volume_gamma=1.0,
            shared_h0=0.35,
            scale_prior_strength=2.0,
            scale_min_ratio=0.5,
            scale_max_ratio=2.0,
            temperature=0.85,
            phase1_lodo_receipt_sha256=_sha(f"lodo:{row.row_id}"),
            quantization_margin_audit_sha256=_sha(f"quant:{row.row_id}"),
        )
        rows.append(
            {
                "row_id": row.row_id,
                "receiver": row.receiver,
                "k_shot": row.k_shot,
                "new_count": row.new_count,
                **references,
                "split_authorities": authorities,
                "phase1_bundle": phase1,
                "checkpoint_path": str(checkpoint.resolve()),
                "checkpoint_sha256": checkpoint_sha,
                "data_feature_runtime_sha256": plan.data_feature_runtime_sha256,
                "data_materialization_lock_sha256": plan.data_materialization_lock_sha256,
                "qknn_lock": asdict(lock),
                "feature_batch_size": 8,
                "score_chunk_size": None,
            }
        )
    return (
        {
            "schema": CONTEXT_MANIFEST_SCHEMA,
            "plan_receipt_sha256": plan.plan_receipt_sha256,
            "claim_scope": plan.claim_scope,
            "formal_launch_authority": plan.formal_launch_authority,
            "authority_envelope_root_sha256": plan.authority_envelope_root_sha256,
            "data_feature_runtime_sha256": plan.data_feature_runtime_sha256,
            "data_materialization_lock_sha256": plan.data_materialization_lock_sha256,
            "d105_candidate_runtime_manifest_sha256": plan.d105_candidate_runtime_manifest_sha256,
            "d105_candidate_method_lock_sha256": plan.d105_candidate_method_lock_sha256,
            "rows": rows,
        },
        phase1,
    )


def _fake_formal_asset(phase1: dict):
    return SimpleNamespace(
        manifest_sha256=phase1["manifest_sha256"],
        validated_bundle_id_sha256=phase1["validated_bundle_id_sha256"],
        validator_receipt_sha256=phase1["validator_receipt_sha256"],
        manifest={"bundle_wire_sha256": phase1["bundle_wire_sha256"]},
        bundle=SimpleNamespace(
            content_root_sha256=phase1["expected_content_root_sha256"],
            checkpoint_sha256=phase1["checkpoint_sha256"],
            runtime_sha256=phase1["d105_candidate_runtime_manifest_sha256"],
            method_lock_sha256=phase1["d105_candidate_method_lock_sha256"],
        ),
    )


def test_sealed_plan_roundtrip_and_real_evaluator_adapter_runs_once_per_row() -> None:
    plan = _plan()
    factory_calls = []
    evaluator_calls = []
    with TemporaryDirectory() as temp:
        root = Path(temp)
        plan_path = root / "target25_plan.json"
        write_d105_target25_plan_manifest(plan_path, plan)
        restored = load_d105_target25_plan_manifest(plan_path)
        assert restored.plan_receipt_sha256 == plan.plan_receipt_sha256
        run = prepare_d105_target25_run(
            restored,
            output_root=root,
            run_id="d105-target25-real-adapter-001",
            schedule=D105Target25GPUSchedule(gpu_ids=(2, 5), workers_per_gpu=1),
        )

        def context_factory(row, gpu_id):
            factory_calls.append((row.row_id, gpu_id))
            return _FakeContext(row=row, gpu_id=gpu_id)

        def evaluate_row(context):
            evaluator_calls.append((context.row.row_id, context.gpu_id))
            return _FakeCompleteEvaluation(context.row)

        summary = execute_d105_target25_with_evaluator(
            run,
            context_factory,
            evaluate_row=evaluate_row,
        )
        assert summary.status == "PREDICTIONS_COMPLETE"
        assert len(factory_calls) == len(restored.rows) == 25
        assert evaluator_calls == factory_calls
        assert {gpu_id for _, gpu_id in factory_calls} == {2, 5}
        manifest = verify_d105_target25_prediction_manifest(run)
        assert manifest["state_prediction_surface_count"] == 600


def test_plan_identity_sidecars_are_cross_directory_portable_and_reject_escape() -> None:
    plan = _plan()
    with TemporaryDirectory() as temp:
        root = Path(temp)
        source = root / "source-host-package"
        source.mkdir()
        plan_path = source / "target25_plan.json"
        write_d105_target25_plan_manifest(plan_path, plan)
        relocated = root / "simulated-remote-host" / "release"
        shutil.copytree(source, relocated)
        relocated_plan = relocated / "target25_plan.json"
        portable_document = json.loads(relocated_plan.read_text(encoding="utf-8"))
        assert all(
            not Path(value).is_absolute() and ".." not in Path(value).parts
            for value in portable_document["candidate_identity_sources"].values()
        )
        restored = load_d105_target25_plan_manifest(relocated_plan)
        assert restored.plan_receipt_sha256 == plan.plan_receipt_sha256
        assert relocated in restored.candidate_runtime_manifest_path.parents
        assert relocated in restored.candidate_method_lock_path.parents

        document = json.loads(relocated_plan.read_text(encoding="utf-8"))
        document["candidate_identity_sources"][
            "candidate_runtime_manifest_path"
        ] = "../escaped-candidate-runtime.json"
        document["plan_manifest_receipt_sha256"] = canonical_sha256(
            {
                key: value
                for key, value in document.items()
                if key != "plan_manifest_receipt_sha256"
            }
        )
        relocated_plan.chmod(relocated_plan.stat().st_mode | stat.S_IWUSR)
        relocated_plan.write_bytes(
            json.dumps(
                document,
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
            + b"\n"
        )
        relocated_plan.chmod(stat.S_IREAD)
        with pytest.raises(D105Target25RunnerError, match="portable-relative"):
            load_d105_target25_plan_manifest(relocated_plan)


def test_json_context_manifest_constructs_gpu_bound_real_context_and_rejects_schema_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _plan()
    with TemporaryDirectory() as temp:
        root = Path(temp)
        document, phase1 = _context_document(plan, root)
        context_path = root / "target25_context.json"
        seal_d105_target25_context_manifest(context_path, document)
        fake_asset = _fake_formal_asset(phase1)
        monkeypatch.setattr(
            "cvsrffi.stage2_d105_phase1_bundle.load_d105_phase1_asset",
            lambda *_args, **_kwargs: fake_asset,
        )
        factory = load_d105_target25_context_factory(context_path, plan)
        context = factory(plan.rows[0], 6)
        assert context.device == "cuda:6"
        assert context.qknn_lock.active_k == plan.rows[0].k_shot
        assert len(context.split_authorities) == 6
        assert context.phase1_bundle.manifest_sha256 == phase1["manifest_sha256"]

        broken = dict(document)
        broken["unexpected_truth_field"] = False
        broken_path = root / "target25_context_broken.json"
        seal_d105_target25_context_manifest(broken_path, broken)
        with pytest.raises(D105Target25LauncherError, match="top-level closure"):
            load_d105_target25_context_factory(broken_path, plan)


def test_launcher_script_exposes_context_factory_contract() -> None:
    script = Path(__file__).resolve().parents[1] / "code" / "scripts" / "run_d105_target25.py"
    predict_help = subprocess.run(
        [sys.executable, str(script), "predict", "--help"],
        text=True,
        capture_output=True,
        check=False,
    )
    score_help = subprocess.run(
        [sys.executable, str(script), "score", "--help"],
        text=True,
        capture_output=True,
        check=False,
    )
    assert predict_help.returncode == 0
    assert "--context-manifest" in predict_help.stdout
    assert "--plan-manifest" in predict_help.stdout
    assert "--prepare-receipt" in predict_help.stdout
    assert "--prepare-receipt-sha256" in predict_help.stdout
    assert "--prepare-authority-envelope" in predict_help.stdout
    assert "--prepare-authority-signature" in predict_help.stdout
    assert "--nonce-ledger-dir" in predict_help.stdout
    assert "--context-factory" not in predict_help.stdout
    assert score_help.returncode == 0
    assert "--truth-catalog" in score_help.stdout
    assert "--truth-catalog-sha256" in score_help.stdout
    assert "--score-root" in score_help.stdout
    assert "--context-manifest" in score_help.stdout
    assert "--prepare-receipt" in score_help.stdout
    assert "--truth-provider" not in score_help.stdout


def test_launcher_cli_returns_nonzero_for_partial_technical_summary(
    monkeypatch, tmp_path: Path
) -> None:
    script = Path(__file__).resolve().parents[1] / "code" / "scripts" / "run_d105_target25.py"
    spec = importlib.util.spec_from_file_location("run_d105_target25_cli_test", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    partial_path = tmp_path / "partial_prediction_manifest.json"
    partial_path.write_text('{"status":"TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT"}')
    fake_run = SimpleNamespace(run_root=tmp_path / "run")
    fake_summary = SimpleNamespace(
        status="TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT",
        manifest_path=partial_path,
        completed_outer_rows=2,
        scenario_arm_pair_count=0,
        state_prediction_surface_count=0,
        stop_dispatch=True,
        stop_fingerprint_sha256=hashlib.sha256(b"fault").hexdigest(),
    )
    monkeypatch.setattr(module, "load_d105_target25_plan_manifest", lambda _path: _plan())
    monkeypatch.setattr(module, "_verify_prepare_authority", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(
        module, "load_d105_target25_context_factory", lambda _path, _plan: object()
    )
    monkeypatch.setattr(module, "prepare_d105_target25_run", lambda *_args, **_kwargs: fake_run)
    monkeypatch.setattr(
        module,
        "execute_d105_target25_with_evaluator",
        lambda _run, _factory: fake_summary,
    )
    exit_code = module.main(
        [
            "predict",
            "--plan-manifest",
            str(tmp_path / "plan.json"),
            "--context-manifest",
            str(tmp_path / "context.json"),
            "--prepare-receipt",
            str(tmp_path / "prepare_receipt.json"),
            "--prepare-receipt-sha256",
            "a" * 64,
            "--d92-matrix-index",
            str(tmp_path / "matrix-index.json"),
            "--d92-matrix-index-sha256",
            "b" * 64,
            "--prepare-authority-envelope",
            str(tmp_path / "prepare-envelope.json"),
            "--prepare-authority-signature",
            str(tmp_path / "prepare-envelope.ed25519"),
            "--git-commit",
            "c" * 40,
            "--nonce-ledger-dir",
            str(tmp_path),
            "--output-root",
            str(tmp_path),
            "--run-id",
            "d105-cli-partial",
            "--gpu-ids",
            "0",
        ]
    )
    assert partial_path.exists()
    assert exit_code != 0
