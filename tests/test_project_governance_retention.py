from __future__ import annotations

import ast
import importlib
from dataclasses import replace
from pathlib import Path

import pytest

from tools.project_governance.models import (
    AccessStatus,
    ApprovalState,
    AssetKind,
    AssetRecord,
    ExecutionState,
    ExperimentState,
    GitOwnership,
    HashStatus,
    Location,
    RetentionClass,
)


def _retention_module():
    return importlib.import_module("tools.project_governance.classify_retention")


def _asset(
    *,
    asset_id: str = "asset:LOCAL:FIXTURE:cache/item.bin",
    evidence_role: str | None = None,
    size_bytes: int | None = 41,
    access_status: AccessStatus = AccessStatus.OK,
    hash_status: HashStatus = HashStatus.METADATA_ONLY,
    sha256: str | None = None,
    git_ownership: GitOwnership | None = GitOwnership.IGNORED_REGENERABLE,
) -> AssetRecord:
    return AssetRecord(
        asset_id=asset_id,
        scan_id="RETENTION_FIXTURE_SCAN",
        location=Location.LOCAL,
        root_id="FIXTURE",
        relative_path="cache/item.bin",
        display_name="item.bin",
        escaped_name="item.bin",
        asset_kind=AssetKind.FILE,
        size_bytes=size_bytes,
        mtime_utc="2026-08-17T00:00:00Z",
        access_status=access_status,
        hash_status=hash_status,
        sha256=sha256,
        evidence_role=evidence_role,
        git_ownership=git_ownership,
    )


def _delete_facts(retention, **changes):
    facts = retention.RetentionEvidence(
        evidence_asset_ids=("asset:LOCAL:FIXTURE:generator", "asset:LOCAL:FIXTURE:sources"),
        deletion_candidate_requested=True,
        provenance_known=True,
        purpose_known=True,
        active_process=False,
        current_publication_dependency=False,
        current_document_dependency=False,
        current_release_dependency=False,
        current_review_dependency=False,
        generator_recorded=True,
        source_dependencies_recorded=True,
        rebuild_command_recorded=True,
        git_worktree_dependency=False,
        report_dependency=False,
        manifest_dependency=False,
        receipt_dependency=False,
        recoverability="Rebuild from retained generator and source assets.",
        estimated_space_reclaim=41,
        absolute_path="E:/fixture/cache/item.bin",
    )
    return replace(facts, **changes)


@pytest.mark.parametrize(
    "evidence_role",
    (
        "dataset",
        "checkpoint",
        "formal_report",
        "log",
        "metrics",
        "prediction",
        "score",
        "receipt",
        "manifest",
        "run_output",
    ),
)
def test_protected_evidence_roles_are_immutable(evidence_role: str):
    retention = _retention_module()
    asset = _asset(evidence_role=evidence_role)

    decision = retention.classify_retention(asset, retention.RetentionEvidence())

    assert decision.retention_class is RetentionClass.KEEP_IMMUTABLE
    assert decision.rule_code == "PROTECTED_EVIDENCE"
    assert decision.evidence_asset_ids == (asset.asset_id,)


@pytest.mark.parametrize("state", (ExperimentState.ACTIVE_LIVE, ExperimentState.OPEN_INCOMPLETE))
def test_active_or_open_experiment_assets_are_kept_active(state: ExperimentState):
    retention = _retention_module()
    asset = _asset()
    facts = retention.RetentionEvidence(
        experiment_state=state,
        evidence_asset_ids=("asset:LOCAL:FIXTURE:run-status",),
    )

    decision = retention.classify_retention(asset, facts)

    assert decision.retention_class is RetentionClass.KEEP_ACTIVE
    assert decision.rule_code == "ACTIVE_OR_OPEN_EXPERIMENT"
    assert decision.evidence_asset_ids == (asset.asset_id, "asset:LOCAL:FIXTURE:run-status")


def test_current_publication_dependency_is_kept_until_published():
    retention = _retention_module()
    asset = _asset()
    facts = retention.RetentionEvidence(
        current_publication_dependency=True,
        evidence_asset_ids=("asset:LOCAL:FIXTURE:current-paper",),
    )

    decision = retention.classify_retention(asset, facts)

    assert decision.retention_class is RetentionClass.KEEP_UNTIL_PUBLISHED
    assert decision.rule_code == "CURRENT_PUBLICATION_DEPENDENCY"
    assert "asset:LOCAL:FIXTURE:current-paper" in (decision.evidence_asset_ids or ())


@pytest.mark.parametrize(
    ("terminal_evidence", "archive_marker", "expected_class", "expected_rule"),
    (
        (True, True, RetentionClass.HISTORICAL_ARCHIVE, "VERIFIED_HISTORICAL_ARCHIVE"),
        (False, True, RetentionClass.REVIEW_REQUIRED, "INSUFFICIENT_EVIDENCE"),
        (True, False, RetentionClass.REVIEW_REQUIRED, "INSUFFICIENT_EVIDENCE"),
    ),
)
def test_historical_archive_requires_explicit_terminal_and_archive_marker(
    terminal_evidence: bool,
    archive_marker: bool,
    expected_class: RetentionClass,
    expected_rule: str,
):
    retention = _retention_module()
    asset = _asset()
    facts = retention.RetentionEvidence(
        experiment_state=ExperimentState.HISTORICAL_ARCHIVE,
        terminal_evidence=terminal_evidence,
        archive_marker=archive_marker,
    )

    decision = retention.classify_retention(asset, facts)

    assert decision.retention_class is expected_class
    assert decision.rule_code == expected_rule


@pytest.mark.parametrize(
    ("generator_recorded", "source_dependencies_recorded", "rebuild_command_recorded", "expected_class"),
    (
        (True, True, True, RetentionClass.REGENERABLE_CACHE),
        (False, True, True, RetentionClass.REVIEW_REQUIRED),
        (True, False, True, RetentionClass.REVIEW_REQUIRED),
        (True, True, False, RetentionClass.REVIEW_REQUIRED),
    ),
)
def test_regenerable_cache_requires_all_rebuild_evidence(
    generator_recorded: bool,
    source_dependencies_recorded: bool,
    rebuild_command_recorded: bool,
    expected_class: RetentionClass,
):
    retention = _retention_module()
    asset = _asset(evidence_role="cache")
    facts = retention.RetentionEvidence(
        generator_recorded=generator_recorded,
        source_dependencies_recorded=source_dependencies_recorded,
        rebuild_command_recorded=rebuild_command_recorded,
    )

    decision = retention.classify_retention(asset, facts)

    assert decision.retention_class is expected_class
    assert decision.rule_code == (
        "PROVEN_REGENERABLE_CACHE" if expected_class is RetentionClass.REGENERABLE_CACHE else "INSUFFICIENT_EVIDENCE"
    )


@pytest.mark.parametrize(
    "asset,fact_changes",
    (
        (_asset(size_bytes=0), {}),
        (_asset(), {"old_mtime_signal": True}),
        (_asset(), {"unusual_name_signal": True}),
        (_asset(git_ownership=GitOwnership.UNTRACKED_IN_GIT_WORKTREE), {}),
        (_asset(git_ownership=GitOwnership.NON_GIT_EVIDENCE), {}),
    ),
)
def test_single_cleanup_heuristic_only_requires_review(asset: AssetRecord, fact_changes: dict[str, bool]):
    retention = _retention_module()
    facts = retention.RetentionEvidence(**fact_changes)

    decision = retention.classify_retention(asset, facts)

    assert decision.retention_class is RetentionClass.REVIEW_REQUIRED
    assert decision.rule_code == "INSUFFICIENT_EVIDENCE"


@pytest.mark.parametrize(
    "asset,fact_changes",
    (
        (_asset(access_status=AccessStatus.SCAN_ERROR), {}),
        (_asset(evidence_role="checkpoint"), {"read_error": True}),
        (_asset(evidence_role="checkpoint"), {"git_error": True}),
        (_asset(evidence_role="checkpoint"), {"evidence_conflict": True}),
    ),
)
def test_read_git_and_evidence_conflicts_override_every_other_retention_rule(
    asset: AssetRecord, fact_changes: dict[str, bool]
):
    retention = _retention_module()
    facts = retention.RetentionEvidence(**fact_changes)

    decision = retention.classify_retention(asset, facts)

    assert decision.retention_class is RetentionClass.REVIEW_REQUIRED
    assert decision.rule_code == "ERROR_OR_CONFLICT"


@pytest.mark.parametrize(
    "fact_changes",
    (
        {"provenance_known": False},
        {"purpose_known": False},
        {"generator_recorded": False},
        {"source_dependencies_recorded": False},
        {"rebuild_command_recorded": False},
        {"recoverability": None},
        {"estimated_space_reclaim": None},
        {"absolute_path": None},
        {"read_error": True},
    ),
)
def test_delete_candidate_requires_every_safety_predicate(fact_changes: dict[str, object]):
    retention = _retention_module()
    asset = _asset(evidence_role="cache")
    facts = _delete_facts(retention, **fact_changes)

    decision = retention.classify_retention(asset, facts)

    assert decision.retention_class is not RetentionClass.DELETE_CANDIDATE


def test_fully_proven_delete_candidate_is_only_an_approval_candidate():
    retention = _retention_module()
    asset = _asset(evidence_role="cache")
    facts = _delete_facts(retention)

    decision = retention.classify_retention(asset, facts)
    candidates = retention.build_deletion_candidates((asset,), {asset.asset_id: facts})

    assert decision.retention_class is RetentionClass.DELETE_CANDIDATE
    assert decision.rule_code == "FULLY_PROVEN_DELETE_CANDIDATE"
    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate.absolute_path == "E:/fixture/cache/item.bin"
    assert candidate.approval_state is ApprovalState.AWAITING_USER_APPROVAL
    assert candidate.execution_state is ExecutionState.NOT_AUTHORIZED
    assert candidate.approved_scope is None
    assert candidate.estimated_space_reclaim == 41
    assert candidate.dependencies == ("asset:LOCAL:FIXTURE:generator", "asset:LOCAL:FIXTURE:sources")


def test_delete_candidate_requires_explicit_negative_dependency_checks():
    retention = _retention_module()
    asset = _asset(evidence_role="cache")
    facts = _delete_facts(retention, git_worktree_dependency=None)

    decision = retention.classify_retention(asset, facts)

    assert decision.retention_class is RetentionClass.REVIEW_REQUIRED
    assert decision.rule_code == "INSUFFICIENT_EVIDENCE"


def test_explicit_reclaim_estimate_can_support_a_candidate_when_asset_size_is_unknown():
    retention = _retention_module()
    asset = _asset(evidence_role="cache", size_bytes=None)
    facts = _delete_facts(retention)

    candidates = retention.build_deletion_candidates((asset,), {asset.asset_id: facts})

    assert len(candidates) == 1
    assert candidates[0].size_bytes is None
    assert candidates[0].estimated_space_reclaim == 41


def test_boolean_value_is_not_a_recorded_reclaim_estimate():
    retention = _retention_module()
    asset = _asset(evidence_role="cache")
    facts = _delete_facts(retention, estimated_space_reclaim=True)

    decision = retention.classify_retention(asset, facts)

    assert decision.retention_class is RetentionClass.REVIEW_REQUIRED
    assert decision.rule_code == "INSUFFICIENT_EVIDENCE"


def test_retained_canonical_copy_with_matching_sha256_can_prove_delete_candidate():
    retention = _retention_module()
    digest = "a" * 64
    asset = _asset(evidence_role="cache", hash_status=HashStatus.SHA256, sha256=digest)
    facts = _delete_facts(
        retention,
        generator_recorded=False,
        source_dependencies_recorded=False,
        rebuild_command_recorded=False,
        canonical_copy_asset_id="asset:LOCAL:FIXTURE:canonical/item.bin",
        canonical_copy_sha256=digest,
    )

    decision = retention.classify_retention(asset, facts)

    assert decision.retention_class is RetentionClass.DELETE_CANDIDATE
    assert decision.rule_code == "FULLY_PROVEN_DELETE_CANDIDATE"


def test_mismatched_canonical_copy_sha256_requires_review():
    retention = _retention_module()
    asset = _asset(evidence_role="cache", hash_status=HashStatus.SHA256, sha256="a" * 64)
    facts = _delete_facts(
        retention,
        generator_recorded=False,
        source_dependencies_recorded=False,
        rebuild_command_recorded=False,
        canonical_copy_asset_id="asset:LOCAL:FIXTURE:canonical/item.bin",
        canonical_copy_sha256="b" * 64,
    )

    decision = retention.classify_retention(asset, facts)

    assert decision.retention_class is RetentionClass.REVIEW_REQUIRED
    assert decision.rule_code == "INSUFFICIENT_EVIDENCE"


def _dangerous_package_calls(package_root: Path) -> list[str]:
    module_calls = {
        "os": {
            "remove",
            "unlink",
            "rmdir",
            "removedirs",
            "rename",
            "renames",
            "replace",
            "chmod",
            "chown",
            "kill",
            "system",
            "popen",
        },
        "shutil": {"rmtree", "move"},
        "signal": {"kill", "pthread_kill"},
        "subprocess": {"Popen"},
    }
    path_mutation_methods = {"unlink", "rmdir", "rename", "chmod", "chown"}
    process_control_methods = {"kill", "terminate", "send_signal"}
    output_creation_methods = {"write_text", "write_bytes", "touch", "mkdir"}
    findings: list[str] = []
    for source_path in package_root.rglob("*.py"):
        tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
        module_aliases: dict[str, str] = {}
        dangerous_names: set[str] = set()
        pathlib_aliases: set[str] = set()
        path_constructor_names: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name in module_calls:
                        module_aliases[alias.asname or alias.name] = alias.name
                    if alias.name == "pathlib":
                        pathlib_aliases.add(alias.asname or alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module in module_calls:
                    for alias in node.names:
                        if alias.name in module_calls[node.module]:
                            dangerous_names.add(alias.asname or alias.name)
                if node.module == "pathlib":
                    for alias in node.names:
                        if alias.name == "Path":
                            path_constructor_names.add(alias.asname or alias.name)
        path_instance_names: set[str] = set()

        def is_path_expression(node: ast.AST) -> bool:
            if isinstance(node, ast.Name):
                return node.id in path_instance_names
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    return node.func.id in path_constructor_names
                if isinstance(node.func, ast.Attribute):
                    if (
                        isinstance(node.func.value, ast.Name)
                        and node.func.value.id in pathlib_aliases
                        and node.func.attr == "Path"
                    ):
                        return True
                    return is_path_expression(node.func.value)
            if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
                return is_path_expression(node.left) or is_path_expression(node.right)
            if isinstance(node, ast.Attribute) and node.attr == "parent":
                return is_path_expression(node.value)
            return False

        changed = True
        while changed:
            changed = False
            for node in ast.walk(tree):
                if isinstance(node, (ast.Assign, ast.AnnAssign)) and is_path_expression(node.value):
                    targets = node.targets if isinstance(node, ast.Assign) else (node.target,)
                    for target in targets:
                        if isinstance(target, ast.Name) and target.id not in path_instance_names:
                            path_instance_names.add(target.id)
                            changed = True
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if isinstance(node.func, ast.Name) and node.func.id in dangerous_names:
                findings.append(f"{source_path.name}:{node.lineno}:{node.func.id}")
                continue
            if not isinstance(node.func, ast.Attribute):
                continue
            receiver = node.func.value
            if isinstance(receiver, ast.Name):
                module = module_aliases.get(receiver.id)
                if module in module_calls and node.func.attr in module_calls[module]:
                    findings.append(f"{source_path.name}:{node.lineno}:{module}.{node.func.attr}")
                    continue
            if node.func.attr in output_creation_methods and source_path.name != "emit.py":
                findings.append(f"{source_path.name}:{node.lineno}:output.{node.func.attr}")
                continue
            if node.func.attr in process_control_methods:
                findings.append(f"{source_path.name}:{node.lineno}:process.{node.func.attr}")
                continue
            if node.func.attr in path_mutation_methods:
                findings.append(f"{source_path.name}:{node.lineno}:Path.{node.func.attr}")
                continue
            if node.func.attr == "replace" and is_path_expression(receiver):
                findings.append(f"{source_path.name}:{node.lineno}:Path.replace")
    return findings


def test_package_has_no_destructive_or_process_control_calls_and_keeps_pure_replace_legal():
    package_root = Path(__file__).resolve().parents[1] / "tools" / "project_governance"

    findings = _dangerous_package_calls(package_root)

    assert "safe/path".replace("/", "\\") == "safe\\path"
    assert findings == []


def test_ast_safety_scan_catches_real_path_and_process_mutation_without_false_replace_hits(tmp_path: Path):
    package_root = tmp_path / "project_governance"
    package_root.mkdir()
    (package_root / "safe.py").write_text(
        "from dataclasses import replace\ntext = 'safe/path'.replace('/', '\\\\')\nrecord = replace(record)\n",
        encoding="utf-8",
    )
    (package_root / "unsafe.py").write_text(
        "from pathlib import Path\nimport os\npath = Path('source')\npath.replace(Path('target'))\nos.kill(1, 9)\n",
        encoding="utf-8",
    )

    findings = _dangerous_package_calls(package_root)

    assert any(item.endswith(":Path.replace") for item in findings)
    assert any(item.endswith(":os.kill") for item in findings)
    assert not any(item.startswith("safe.py:") for item in findings)


def test_build_deletion_candidates_is_the_only_deletion_row_constructor():
    retention = _retention_module()
    source = Path(retention.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    constructors: list[tuple[str, int]] = []
    for function in (node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)):
        for call in (node for node in ast.walk(function) if isinstance(node, ast.Call)):
            if isinstance(call.func, ast.Name) and call.func.id == "DeletionCandidate":
                constructors.append((function.name, call.lineno))

    assert constructors
    assert {name for name, _ in constructors} == {"build_deletion_candidates"}
