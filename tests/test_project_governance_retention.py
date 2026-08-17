from __future__ import annotations

import ast
import importlib
from dataclasses import fields, replace
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


def _with_supported_evidence_fields(facts, **changes):
    supported_names = {field.name for field in fields(facts)}
    return replace(facts, **{name: value for name, value in changes.items() if name in supported_names})


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
    facts = _with_supported_evidence_fields(
        facts,
        generator_evidence_asset_ids=("asset:LOCAL:FIXTURE:generator",),
        source_dependency_evidence_asset_ids=("asset:LOCAL:FIXTURE:sources",),
        rebuild_command_evidence_asset_ids=("asset:LOCAL:FIXTURE:rebuild-command",),
    )
    return replace(facts, **changes)


def _canonical_delete_facts(retention, asset: AssetRecord, canonical_asset_id: str, digest: str, **changes):
    facts = _delete_facts(
        retention,
        generator_recorded=False,
        source_dependencies_recorded=False,
        rebuild_command_recorded=False,
        canonical_copy_asset_id=canonical_asset_id,
        canonical_copy_sha256=digest,
    )
    canonical_changes = {
        "canonical_copy_verified": True,
        "canonical_copy_retention_class": RetentionClass.KEEP_IMMUTABLE,
    }
    canonical_changes.update(changes)
    return _with_supported_evidence_fields(
        facts,
        **canonical_changes,
    )


def _regenerable_facts(retention, asset: AssetRecord, **changes):
    facts = retention.RetentionEvidence(
        generator_recorded=True,
        source_dependencies_recorded=True,
        rebuild_command_recorded=True,
    )
    proof_changes = {
        "generator_evidence_asset_ids": ("asset:LOCAL:FIXTURE:generator",),
        "source_dependency_evidence_asset_ids": ("asset:LOCAL:FIXTURE:sources",),
        "rebuild_command_evidence_asset_ids": ("asset:LOCAL:FIXTURE:rebuild-command",),
    }
    proof_changes.update(changes)
    return _with_supported_evidence_fields(
        facts,
        **proof_changes,
    )


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
    facts = _regenerable_facts(
        retention,
        asset,
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
    ("proof_field", "invalid_ids"),
    (
        ("generator_evidence_asset_ids", ()),
        ("source_dependency_evidence_asset_ids", ()),
        ("rebuild_command_evidence_asset_ids", ()),
        ("generator_evidence_asset_ids", ("asset:LOCAL:FIXTURE:cache/item.bin",)),
        ("source_dependency_evidence_asset_ids", ("asset:LOCAL:FIXTURE:cache/item.bin",)),
        ("rebuild_command_evidence_asset_ids", ("asset:LOCAL:FIXTURE:cache/item.bin",)),
        ("generator_evidence_asset_ids", (" asset:LOCAL:FIXTURE:cache/item.bin ",)),
        ("source_dependency_evidence_asset_ids", (" asset:LOCAL:FIXTURE:cache/item.bin ",)),
        ("rebuild_command_evidence_asset_ids", (" asset:LOCAL:FIXTURE:cache/item.bin ",)),
    ),
)
def test_regeneration_requires_nonself_traceable_evidence_for_every_proof_type(
    proof_field: str, invalid_ids: tuple[str, ...]
):
    retention = _retention_module()
    asset = _asset(evidence_role="cache")
    facts = _regenerable_facts(retention, asset, **{proof_field: invalid_ids})

    decision = retention.classify_retention(asset, facts)

    assert decision.retention_class is RetentionClass.REVIEW_REQUIRED
    assert decision.rule_code == "INSUFFICIENT_EVIDENCE"


@pytest.mark.parametrize(
    "proof_field",
    (
        "generator_evidence_asset_ids",
        "source_dependency_evidence_asset_ids",
        "rebuild_command_evidence_asset_ids",
    ),
)
def test_regeneration_requires_trimmed_canonical_proof_evidence_ids(proof_field: str):
    retention = _retention_module()
    asset = _asset(evidence_role="cache")
    facts = _regenerable_facts(
        retention,
        asset,
        **{proof_field: (" asset:LOCAL:FIXTURE:external-proof ",)},
    )

    decision = retention.classify_retention(asset, facts)

    assert decision.retention_class is RetentionClass.REVIEW_REQUIRED
    assert decision.rule_code == "INSUFFICIENT_EVIDENCE"


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
    assert candidate.dependencies == (
        "asset:LOCAL:FIXTURE:generator",
        "asset:LOCAL:FIXTURE:sources",
        "asset:LOCAL:FIXTURE:rebuild-command",
    )


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
    facts = _canonical_delete_facts(retention, asset, "asset:LOCAL:FIXTURE:canonical/item.bin", digest)

    decision = retention.classify_retention(asset, facts)

    assert decision.retention_class is RetentionClass.DELETE_CANDIDATE
    assert decision.rule_code == "FULLY_PROVEN_DELETE_CANDIDATE"


def test_mismatched_canonical_copy_sha256_requires_review():
    retention = _retention_module()
    asset = _asset(evidence_role="cache", hash_status=HashStatus.SHA256, sha256="a" * 64)
    facts = _canonical_delete_facts(retention, asset, "asset:LOCAL:FIXTURE:canonical/item.bin", "b" * 64)

    decision = retention.classify_retention(asset, facts)

    assert decision.retention_class is RetentionClass.REVIEW_REQUIRED
    assert decision.rule_code == "INSUFFICIENT_EVIDENCE"


def test_self_referential_canonical_copy_requires_review():
    retention = _retention_module()
    digest = "a" * 64
    asset = _asset(evidence_role="cache", hash_status=HashStatus.SHA256, sha256=digest)
    facts = _canonical_delete_facts(retention, asset, asset.asset_id, digest)

    decision = retention.classify_retention(asset, facts)

    assert decision.retention_class is RetentionClass.REVIEW_REQUIRED
    assert decision.rule_code == "INSUFFICIENT_EVIDENCE"


@pytest.mark.parametrize(
    "canonical_asset_id",
    (
        " asset:LOCAL:FIXTURE:cache/item.bin ",
        "\tasset:LOCAL:FIXTURE:cache/item.bin\t",
    ),
)
def test_whitespace_wrapped_self_referential_canonical_copy_requires_review(canonical_asset_id: str):
    retention = _retention_module()
    digest = "a" * 64
    asset = _asset(evidence_role="cache", hash_status=HashStatus.SHA256, sha256=digest)
    facts = _canonical_delete_facts(retention, asset, canonical_asset_id, digest)

    decision = retention.classify_retention(asset, facts)

    assert decision.retention_class is RetentionClass.REVIEW_REQUIRED
    assert decision.rule_code == "INSUFFICIENT_EVIDENCE"


def test_canonical_copy_requires_trimmed_identity():
    retention = _retention_module()
    digest = "a" * 64
    asset = _asset(evidence_role="cache", hash_status=HashStatus.SHA256, sha256=digest)
    facts = _canonical_delete_facts(retention, asset, " asset:LOCAL:FIXTURE:canonical/item.bin ", digest)

    decision = retention.classify_retention(asset, facts)

    assert decision.retention_class is RetentionClass.REVIEW_REQUIRED
    assert decision.rule_code == "INSUFFICIENT_EVIDENCE"


@pytest.mark.parametrize(
    "changes",
    (
        {"canonical_copy_verified": False},
        {"canonical_copy_retention_class": RetentionClass.REVIEW_REQUIRED},
        {"canonical_copy_retention_class": RetentionClass.DELETE_CANDIDATE},
    ),
)
def test_canonical_copy_requires_explicit_verified_retained_status(changes: dict[str, object]):
    retention = _retention_module()
    digest = "a" * 64
    asset = _asset(evidence_role="cache", hash_status=HashStatus.SHA256, sha256=digest)
    facts = _canonical_delete_facts(retention, asset, "asset:LOCAL:FIXTURE:canonical/item.bin", digest, **changes)

    decision = retention.classify_retention(asset, facts)

    assert decision.retention_class is RetentionClass.REVIEW_REQUIRED
    assert decision.rule_code == "INSUFFICIENT_EVIDENCE"


def test_batch_rejects_candidate_whose_canonical_copy_is_another_batch_candidate():
    retention = _retention_module()
    digest = "a" * 64
    dependent = _asset(
        asset_id="asset:LOCAL:FIXTURE:cache/dependent.bin",
        evidence_role="cache",
        hash_status=HashStatus.SHA256,
        sha256=digest,
    )
    canonical = _asset(asset_id="asset:LOCAL:FIXTURE:cache/canonical.bin", evidence_role="cache")
    dependent_facts = _canonical_delete_facts(retention, dependent, canonical.asset_id, digest)
    canonical_facts = _delete_facts(retention)

    candidates = retention.build_deletion_candidates(
        (dependent, canonical),
        {dependent.asset_id: dependent_facts, canonical.asset_id: canonical_facts},
    )

    assert [candidate.candidate_id for candidate in candidates] == [f"deletion-candidate:{canonical.asset_id}"]


def test_batch_rejects_mutually_referential_canonical_candidates():
    retention = _retention_module()
    digest = "a" * 64
    first = _asset(
        asset_id="asset:LOCAL:FIXTURE:cache/first.bin",
        evidence_role="cache",
        hash_status=HashStatus.SHA256,
        sha256=digest,
    )
    second = _asset(
        asset_id="asset:LOCAL:FIXTURE:cache/second.bin",
        evidence_role="cache",
        hash_status=HashStatus.SHA256,
        sha256=digest,
    )
    first_facts = _canonical_delete_facts(retention, first, second.asset_id, digest)
    second_facts = _canonical_delete_facts(retention, second, first.asset_id, digest)

    candidates = retention.build_deletion_candidates(
        (first, second),
        {first.asset_id: first_facts, second.asset_id: second_facts},
    )

    assert candidates == ()


def test_batch_rejects_whitespace_wrapped_mutually_referential_canonical_candidates():
    retention = _retention_module()
    digest = "a" * 64
    first = _asset(
        asset_id="asset:LOCAL:FIXTURE:cache/first.bin",
        evidence_role="cache",
        hash_status=HashStatus.SHA256,
        sha256=digest,
    )
    second = _asset(
        asset_id="asset:LOCAL:FIXTURE:cache/second.bin",
        evidence_role="cache",
        hash_status=HashStatus.SHA256,
        sha256=digest,
    )
    first_facts = _canonical_delete_facts(retention, first, f" {second.asset_id} ", digest)
    second_facts = _canonical_delete_facts(retention, second, f" {first.asset_id} ", digest)

    candidates = retention.build_deletion_candidates(
        (first, second),
        {first.asset_id: first_facts, second.asset_id: second_facts},
    )

    assert candidates == ()


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
        def is_path_annotation(node: ast.AST | None) -> bool:
            if isinstance(node, ast.Name):
                return node.id in path_constructor_names
            return (
                isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Name)
                and node.value.id in pathlib_aliases
                and node.attr == "Path"
            )

        module_scope = "module"
        scope_for_node: dict[int, str] = {}
        scope_parents: dict[str, str | None] = {module_scope: None}
        scope_function_nodes: dict[str, ast.FunctionDef | ast.AsyncFunctionDef] = {}
        functions_by_parent_scope: dict[str, dict[str, str]] = {}

        class ScopeCollector(ast.NodeVisitor):
            def __init__(self) -> None:
                self.current_scope = module_scope
                self.scope_index = 0

            def generic_visit(self, node: ast.AST) -> None:
                scope_for_node[id(node)] = self.current_scope
                super().generic_visit(node)

            def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
                self._visit_function(node)

            def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
                self._visit_function(node)

            def _visit_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
                parent_scope = self.current_scope
                scope_for_node[id(node)] = parent_scope
                for decorator in node.decorator_list:
                    self.visit(decorator)
                for default in (*node.args.defaults, *node.args.kw_defaults):
                    if default is not None:
                        self.visit(default)
                self.scope_index += 1
                function_scope = f"function:{self.scope_index}"
                scope_parents[function_scope] = parent_scope
                scope_function_nodes[function_scope] = node
                functions_by_parent_scope.setdefault(parent_scope, {})[node.name] = function_scope
                self.current_scope = function_scope
                self.visit(node.args)
                if node.returns is not None:
                    self.visit(node.returns)
                for statement in node.body:
                    self.visit(statement)
                self.current_scope = parent_scope

        ScopeCollector().visit(tree)
        scope_defined_names = {scope: set() for scope in scope_parents}
        path_names_by_scope = {scope: set() for scope in scope_parents}

        def add_target_names(target: ast.AST, names: set[str]) -> None:
            if isinstance(target, ast.Name):
                names.add(target.id)
            elif isinstance(target, (ast.List, ast.Tuple)):
                for element in target.elts:
                    add_target_names(element, names)

        for function_scope, function_node in scope_function_nodes.items():
            parent_scope = scope_parents[function_scope]
            if parent_scope is not None:
                scope_defined_names[parent_scope].add(function_node.name)
            arguments = (*function_node.args.posonlyargs, *function_node.args.args, *function_node.args.kwonlyargs)
            for argument in arguments:
                scope_defined_names[function_scope].add(argument.arg)
                if is_path_annotation(argument.annotation):
                    path_names_by_scope[function_scope].add(argument.arg)
            for argument in (function_node.args.vararg, function_node.args.kwarg):
                if argument is not None:
                    scope_defined_names[function_scope].add(argument.arg)
                    if is_path_annotation(argument.annotation):
                        path_names_by_scope[function_scope].add(argument.arg)

        for node in ast.walk(tree):
            scope = scope_for_node.get(id(node), module_scope)
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    add_target_names(target, scope_defined_names[scope])
            elif isinstance(node, ast.AnnAssign):
                add_target_names(node.target, scope_defined_names[scope])
                if is_path_annotation(node.annotation):
                    add_target_names(node.target, path_names_by_scope[scope])
            elif isinstance(node, (ast.For, ast.AsyncFor)):
                add_target_names(node.target, scope_defined_names[scope])
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    scope_defined_names[scope].add(alias.asname or alias.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    scope_defined_names[scope].add(alias.asname or alias.name)

        def is_path_name(name: str, scope: str) -> bool:
            current_scope: str | None = scope
            while current_scope is not None:
                if name in path_names_by_scope[current_scope]:
                    return True
                if name in scope_defined_names[current_scope]:
                    return False
                current_scope = scope_parents[current_scope]
            return False

        path_return_function_scopes: set[str] = set()

        def is_path_return_function(name: str, scope: str) -> bool:
            current_scope: str | None = scope
            while current_scope is not None:
                function_scope = functions_by_parent_scope.get(current_scope, {}).get(name)
                if function_scope is not None:
                    return function_scope in path_return_function_scopes
                if name in scope_defined_names[current_scope]:
                    return False
                current_scope = scope_parents[current_scope]
            return False

        def is_path_expression(node: ast.AST | None, scope: str) -> bool:
            if isinstance(node, ast.Name):
                return is_path_name(node.id, scope)
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    return node.func.id in path_constructor_names or is_path_return_function(node.func.id, scope)
                if isinstance(node.func, ast.Attribute):
                    if (
                        isinstance(node.func.value, ast.Name)
                        and node.func.value.id in pathlib_aliases
                        and node.func.attr == "Path"
                    ):
                        return True
                    return is_path_expression(node.func.value, scope)
            if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
                return is_path_expression(node.left, scope) or is_path_expression(node.right, scope)
            if isinstance(node, ast.Attribute) and node.attr == "parent":
                return is_path_expression(node.value, scope)
            return False

        changed = True
        while changed:
            changed = False
            for node in ast.walk(tree):
                if isinstance(node, (ast.Assign, ast.AnnAssign)):
                    scope = scope_for_node.get(id(node), module_scope)
                    if not is_path_expression(node.value, scope):
                        continue
                    targets = node.targets if isinstance(node, ast.Assign) else (node.target,)
                    for target in targets:
                        target_names: set[str] = set()
                        add_target_names(target, target_names)
                        for target_name in target_names:
                            if target_name not in path_names_by_scope[scope]:
                                path_names_by_scope[scope].add(target_name)
                                changed = True
            for function_scope, function_node in scope_function_nodes.items():
                if function_scope in path_return_function_scopes:
                    continue
                if is_path_annotation(function_node.returns) or any(
                    isinstance(return_node, ast.Return)
                    and return_node.value is not None
                    and scope_for_node.get(id(return_node), module_scope) == function_scope
                    and is_path_expression(return_node.value, function_scope)
                    for return_node in ast.walk(function_node)
                ):
                    path_return_function_scopes.add(function_scope)
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
            scope = scope_for_node.get(id(node), module_scope)
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
            if node.func.attr == "replace" and is_path_expression(receiver, scope):
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
    (package_root / "annotated.py").write_text(
        "from pathlib import Path as P\n"
        "def direct(src: P, dst: P):\n    src.replace(dst)\n"
        "def alias(src: P, dst: P):\n    copied = src\n    copied.replace(dst)\n"
        "def return_path(src: P):\n    return src\n"
        "def returned(src: P, dst: P):\n    rebound = return_path(src)\n    rebound.replace(dst)\n",
        encoding="utf-8",
    )
    (package_root / "scoped.py").write_text(
        "from pathlib import Path\n"
        "import dataclasses\n"
        "path = Path('module')\n"
        "def path_receiver(path: Path, target: Path):\n    path.replace(target)\n"
        "def string_receiver(path: str):\n    path.replace('old', 'new')\n"
        "def path_named_dataclasses(dataclasses: Path, target: Path):\n    dataclasses.replace(target)\n"
        "def pure_dataclasses(record):\n    return dataclasses.replace(record)\n"
        "def module_receiver():\n    path.replace(Path('target'))\n",
        encoding="utf-8",
    )
    (package_root / "emit.py").write_text(
        "from pathlib import Path\noutput = Path('fresh')\noutput.write_text('fresh')\noutput.replace(Path('old'))\n",
        encoding="utf-8",
    )

    findings = _dangerous_package_calls(package_root)

    assert any(item.endswith(":Path.replace") for item in findings)
    assert any(item.endswith(":os.kill") for item in findings)
    assert sum(item.startswith("annotated.py:") and item.endswith(":Path.replace") for item in findings) == 3
    assert sum(item.startswith("scoped.py:") and item.endswith(":Path.replace") for item in findings) == 3
    assert any(item.startswith("emit.py:") and item.endswith(":Path.replace") for item in findings)
    assert not any(item.startswith("emit.py:") and ":output.write_text" in item for item in findings)
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
