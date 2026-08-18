# Project Governance Agent Query Layer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a local, read-only SQLite query layer and five Agent-facing governance queries over the completed project asset inventory.

**Architecture:** Stream the immutable external CSV artifacts into a regenerable SQLite database, publish a small Git-tracked pointer to the latest validated index, and extend the existing governance CLI with one offline build command plus five bounded read-only queries. Original CSV/JSON/receipt artifacts remain authoritative; the query database never connects to N607 or mutates project assets.

**Tech Stack:** Python 3.10 standard library (`argparse`, `csv`, `dataclasses`, `json`, `pathlib`, `sqlite3`), pytest, existing `tools.project_governance` package.

**Spec:** `docs/superpowers/specs/2026-08-18-project-governance-agent-query-design.md`

## Global Constraints

- Use the existing isolated worktree `E:/type10-7/code/snapshots/project_governance_20260813_wt`; do not edit from the non-Git project root.
- Use Git for Windows Bash as the exact outer shell; never execute `pwsh` or `pwsh.exe`.
- Run all project tests serially through the `ssr-gpu` Conda environment.
- Build and query only local files; N607 SSH/SCP/network calls must remain zero.
- Do not move, overwrite, rename, sync or delete original local/N607 assets.
- SQLite is a regenerable external index; CSV/JSON and `scan_receipt.json` remain authoritative.
- `REVIEW_REQUIRED`, `ORPHAN_REVIEW`, `SCAN_ERROR`, zero-byte files and non-Git assets never become authorized deletion candidates.
- The query layer is navigation, not a new experiment/release/performance gate.
- Do not stage or modify existing DOCX changes, `.docx_qa*` directories or historical failed-scan evidence.
- Use TDD for every production behavior: write one failing test, observe the expected failure, implement minimally, then rerun.

---

### Task 1: Streamed SQLite index builder

**Files:**
- Create: `tools/project_governance/query_index.py`
- Create: `tests/test_project_governance_query.py`

**Interfaces:**
- Produces: `IndexBuildSummary(scan_id: str, database_path: Path, table_counts: Mapping[str, int])`
- Produces: `build_index(*, receipt_path: Path, external_root: Path, database_path: Path) -> IndexBuildSummary`
- Consumes: terminal `scan_receipt.json` and the six external CSV tables.

- [ ] **Step 1: Add a fixture writer and a failing successful-build test**

Create literal CSV fixtures with UTF-8 BOM and one local asset, one N607 asset, two experiments, two Git rows, two retention rows and zero deletion rows. Write a terminal receipt whose `counts` and `external_files` point at those files. The assertion must exercise the real SQLite output:

```python
def test_build_index_streams_validated_tables_into_a_new_database(tmp_path: Path):
    receipt, external = write_inventory_fixture(tmp_path)
    database = tmp_path / "governance.sqlite"

    summary = build_index(
        receipt_path=receipt,
        external_root=external,
        database_path=database,
    )

    assert summary.scan_id == "PGOV_TEST_001"
    assert summary.table_counts == {
        "assets": 2,
        "experiments": 2,
        "git_ownership": 2,
        "retention": 2,
        "deletion_candidates": 0,
    }
    with sqlite3.connect(database) as connection:
        assert connection.execute(
            "SELECT location, relative_path FROM assets ORDER BY location"
        ).fetchall() == [("LOCAL", "runs/local-a"), ("N607", "runs/remote-a")]
```

The production mutation caught by this test is a missing table import, wrong BOM handling, or failure to merge local/N607 assets.

- [ ] **Step 2: Run the focused test and verify RED**

Run serially:

```bash
conda run -n ssr-gpu python -c "import sys; print(sys.executable)"
conda run -n ssr-gpu python -m pytest tests/test_project_governance_query.py::test_build_index_streams_validated_tables_into_a_new_database -q
```

Expected: interpreter is inside `ssr-gpu`; pytest fails because `tools.project_governance.query_index` does not exist.

- [ ] **Step 3: Implement the minimal schema and streamed importer**

Implement exact table schemas for the source headers. Use `encoding="utf-8-sig"`, `newline=""`, `csv.DictReader`, parameterized `executemany` batches and a temporary database beside the target. Core shape:

```python
@dataclass(frozen=True)
class IndexBuildSummary:
    scan_id: str
    database_path: Path
    table_counts: Mapping[str, int]


def build_index(*, receipt_path: Path, external_root: Path, database_path: Path) -> IndexBuildSummary:
    receipt = _load_terminal_receipt(receipt_path)
    _require_new_database_target(database_path)
    sources = _resolve_csv_sources(receipt, external_root)
    temporary = database_path.with_name(f".{database_path.name}.building")
    _require_absent(temporary)
    connection = sqlite3.connect(temporary)
    try:
        _create_schema(connection)
        counts = _import_all(connection, sources, scan_id=receipt["scan_id"])
        _validate_counts(receipt, counts)
        connection.commit()
        connection.close()
        temporary.replace(database_path)
    except BaseException:
        connection.close()
        if temporary.exists():
            temporary.unlink()
        raise
    return IndexBuildSummary(receipt["scan_id"], database_path, counts)
```

The cleanup may remove only the temporary database created by this invocation; it must never unlink the requested final target or any source artifact.

- [ ] **Step 4: Run the focused test and verify GREEN**

```bash
conda run -n ssr-gpu python -m pytest tests/test_project_governance_query.py::test_build_index_streams_validated_tables_into_a_new_database -q
```

Expected: `1 passed`.

- [ ] **Step 5: Add failing validation and immutability tests**

Add parameterized tests proving the builder rejects, before publishing a final database:

```python
@pytest.mark.parametrize("mutation", (
    "receipt_not_complete",
    "scan_id_mismatch",
    "external_path_outside_root",
    "missing_csv",
    "wrong_header",
    "wrong_row_count",
    "existing_database",
))
def test_build_index_fails_closed_without_replacing_sources_or_target(tmp_path, mutation):
    case = write_inventory_fixture(tmp_path)
    source_bytes_before = {path: path.read_bytes() for path in case.source_paths}
    mutate_inventory_fixture(case, mutation)
    with pytest.raises((ValueError, FileExistsError, FileNotFoundError)):
        build_index(
            receipt_path=case.receipt,
            external_root=case.external_root,
            database_path=case.database,
        )
    assert {path: path.read_bytes() for path in case.source_paths} == source_bytes_before
    assert case.existing_target_bytes_are_unchanged()
```

Add a separate test with an empty deletion CSV proving no deletion row is invented.

- [ ] **Step 6: Run validation tests and verify RED**

```bash
conda run -n ssr-gpu python -m pytest tests/test_project_governance_query.py -k 'build_index and (fails_closed or deletion)' -q
```

Expected: failures identify each missing validation branch.

- [ ] **Step 7: Implement receipt, boundary, header and count validation**

Require:

```python
receipt["terminal_state"] == "COMPLETE"
receipt["source_asset_mutations"] == 0
receipt["moves"] == 0
receipt["overwrites"] == 0
receipt["deletions"] == 0
```

Resolve each required CSV from `external_files`, require that its resolved parent is exactly `external_root.resolve()`, verify its byte size against the receipt, verify exact headers, and compare imported counts against receipt counts. Validate every asset row's `scan_id` against the receipt. Do not recompute multi-GB hashes.

- [ ] **Step 8: Run all builder tests and commit Task 1**

```bash
conda run -n ssr-gpu python -m pytest tests/test_project_governance_query.py -k build_index -q
git diff --check
git add tools/project_governance/query_index.py tests/test_project_governance_query.py
git commit -m "feat: build read-only governance query index"
```

Expected: all builder tests pass; commit contains only the two Task 1 files.

---

### Task 2: Latest pointer loader and read-only query store

**Files:**
- Modify: `tools/project_governance/query_index.py`
- Modify: `tests/test_project_governance_query.py`

**Interfaces:**
- Produces: `LatestPointer(schema_version: int, scan_id: str, receipt_path: Path, external_root: Path, sqlite_path: Path, created_at_utc: str, implementation_git_head: str)`
- Produces: `load_latest(pointer_path: Path) -> LatestPointer`
- Produces: `QueryStore.open(pointer: LatestPointer) -> QueryStore`
- Produces query methods: `status()`, `find_assets(query, limit)`, `experiment(run_id)`, `repo(path)`, `review(filters, limit)` returning JSON-serializable mappings.
- Consumes: Task 1 SQLite schema and terminal receipt.

- [ ] **Step 1: Add failing pointer and read-only-open tests**

```python
def test_latest_pointer_opens_the_matching_database_read_only(tmp_path):
    pointer_path = write_built_pointer_fixture(tmp_path)
    pointer = load_latest(pointer_path)
    store = QueryStore.open(pointer)
    before = pointer.sqlite_path.stat().st_mtime_ns
    assert store.status()["scan_id"] == "PGOV_TEST_001"
    store.close()
    assert pointer.sqlite_path.stat().st_mtime_ns == before
    assert not Path(f"{pointer.sqlite_path}-wal").exists()
```

Mutation caught: opening SQLite in writable/default mode, creating side files, or accepting mismatched metadata.

- [ ] **Step 2: Run pointer test and verify RED**

```bash
conda run -n ssr-gpu python -m pytest tests/test_project_governance_query.py::test_latest_pointer_opens_the_matching_database_read_only -q
```

Expected: fails because `LatestPointer`, `load_latest` and `QueryStore` are absent.

- [ ] **Step 3: Implement strict pointer loading and read-only connection**

Use `Path.resolve(strict=True)`, exact schema keys and SQLite URI mode:

```python
uri = f"file:{quote(str(pointer.sqlite_path))}?mode=ro&immutable=1"
connection = sqlite3.connect(uri, uri=True)
```

Validate pointer scan ID against both receipt and SQLite `metadata`; reject nonexistent paths, path roots outside the approved governance roots, nonterminal receipts and schema mismatch.

- [ ] **Step 4: Run pointer test and verify GREEN**

```bash
conda run -n ssr-gpu python -m pytest tests/test_project_governance_query.py::test_latest_pointer_opens_the_matching_database_read_only -q
```

- [ ] **Step 5: Add failing real-behavior tests for all five queries**

Use literal expected results:

```python
def test_find_accepts_asset_id_and_normalized_absolute_path(built_store):
    assert built_store.find_assets("asset-local", limit=20)["items"][0]["relative_path"] == "runs/local-a"
    assert built_store.find_assets(r"E:\type10-7\runs\local-a", limit=20)["items"][0]["asset_id"] == "asset-local"

def test_experiment_keeps_one_run_evidence_together(built_store):
    result = built_store.experiment("RUN_A")
    assert result["run_id"] == "RUN_A"
    assert result["assets_by_location"] == {"LOCAL": 1, "N607": 1}

def test_repo_returns_ambiguous_instead_of_guessing(built_store):
    assert built_store.repo(r"E:\type10-7\code\same.py")["status"] == "AMBIGUOUS"

def test_review_never_promotes_review_rows_to_deletion_candidates(built_store):
    result = built_store.review({"retention_class": "REVIEW_REQUIRED"}, limit=20)
    assert result["authorized_deletion_count"] == 0
```

Also test: limit must be `1..100`; empty result is explicit; path-prefix queries are bounded; `SCAN_ERROR` is preserved.

- [ ] **Step 6: Run query tests and verify RED**

```bash
conda run -n ssr-gpu python -m pytest tests/test_project_governance_query.py -k 'find or experiment or repo or review or limit' -q
```

- [ ] **Step 7: Implement parameterized bounded queries**

All SQL values use parameter markers. `find_assets` tries exact asset ID, normalized absolute path and then bounded prefix. `experiment` joins only the exact normalized run ID. `repo` returns all distinct matching repository/worktree pairs and reports `AMBIGUOUS` when more than one remains. `review` accepts only controlled enum filters and reads deletion authorization exactly as stored.

- [ ] **Step 8: Run Task 2 tests and commit**

```bash
conda run -n ssr-gpu python -m pytest tests/test_project_governance_query.py -q
git diff --check
git add tools/project_governance/query_index.py tests/test_project_governance_query.py
git commit -m "feat: query project governance index"
```

---

### Task 3: CLI build and query commands

**Files:**
- Modify: `tools/project_governance/cli.py`
- Modify: `tests/test_project_governance_query.py`

**Interfaces:**
- Consumes: Task 1 `build_index` and Task 2 `load_latest`/`QueryStore`.
- Produces subcommands: `build-index`, `status`, `find`, `experiment`, `repo`, `review`.
- Produces: one JSON object on stdout when `--json` is passed; stable exit codes `0`, `2`, `3`, `4`.

- [ ] **Step 1: Add failing parser and no-side-effect tests**

```python
@pytest.mark.parametrize("command", ("build-index", "status", "find", "experiment", "repo", "review"))
def test_cli_exposes_only_the_approved_query_surface(command):
    parser = build_parser()
    assert parser.parse_args(valid_argv(command)).command == command


def test_query_commands_do_not_construct_collectors_or_create_outputs(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "LocalCollector", forbidden)
    monkeypatch.setattr(cli, "_make_n607_collector", forbidden)
    exit_code = cli.main(["status", "--latest", str(pointer), "--json"])
    assert exit_code in (0, 3)
    assert tree_bytes(tmp_path) == before
```

Mutation caught: routing a query through scan collection, accepting destructive flags, or writing output.

- [ ] **Step 2: Run CLI parser tests and verify RED**

```bash
conda run -n ssr-gpu python -m pytest tests/test_project_governance_query.py -k cli -q
```

- [ ] **Step 3: Extend `build_parser` and route commands before scan construction**

Add exact arguments:

```text
build-index --receipt PATH --external-root PATH --database PATH [--json]
status --latest PATH [--json]
find QUERY --latest PATH [--limit 20] [--json]
experiment RUN_ID --latest PATH [--json]
repo PATH --latest PATH [--json]
review --latest PATH [--location {LOCAL,N607}]
       [--retention-class RETENTION_CLASS]
       [--experiment-state EXPERIMENT_STATE]
       [--ownership OWNERSHIP] [--limit 20] [--json]
```

Dispatch query/build commands before any scan config, progress journal or collector is created.

- [ ] **Step 4: Implement deterministic rendering and exit mapping**

JSON uses `ensure_ascii=False`, sorted keys and one terminal line. Text output is concise and never claims live N607 state. Map validation/database inconsistency to`2`, conservative receipt warnings to`3`, and pre-query unsafe input to`4`.

- [ ] **Step 5: Run CLI tests and verify GREEN**

```bash
conda run -n ssr-gpu python -m pytest tests/test_project_governance_query.py -k cli -q
```

- [ ] **Step 6: Run existing CLI regressions and commit**

```bash
conda run -n ssr-gpu python -m pytest tests/test_project_governance_emit_cli.py tests/test_project_governance_query.py -q
git diff --check
git add tools/project_governance/cli.py tests/test_project_governance_query.py
git commit -m "feat: expose governance queries to project agents"
```

---

### Task 4: Agent policy, real baseline index and latest pointer

**Files:**
- Create: `docs/project_governance/agent-usage.md`
- Create: `docs/project_governance/latest.json`
- Modify: `AGENTS.md`
- Modify: `docs/superpowers/specs/2026-08-18-project-governance-agent-query-design.md`

**Interfaces:**
- Consumes: Task 3 CLI and completed scan `PGOV_20260818T062450Z`.
- Produces: a Git-tracked latest pointer and minimal Agent workflow documentation.
- Produces external file: `E:/type10-7/local_artifacts/project_governance/PGOV_20260818T062450Z/governance.sqlite`.

- [ ] **Step 1: Build the real SQLite index once**

Run the exact offline command with no N607 flag:

```bash
conda run -n ssr-gpu python tools/project_governance_inventory.py build-index \
  --receipt E:/type10-7/code/snapshots/project_governance_20260813_wt/docs/project_governance/PGOV_20260818T062450Z/scan_receipt.json \
  --external-root E:/type10-7/local_artifacts/project_governance/PGOV_20260818T062450Z \
  --database E:/type10-7/local_artifacts/project_governance/PGOV_20260818T062450Z/governance.sqlite \
  --json
```

Expected: one JSON summary, counts equal the receipt, no source mutation, no network.

- [ ] **Step 2: Independently query the real database before publishing the pointer**

Use a temporary pointer outside Git, run `status`, one exact known path query, one known run query and one repo query. Verify returned scan ID is`PGOV_20260818T062450Z`, SQLite opens read-only, and source CSV mtimes/bytes recorded before the build are unchanged.

- [ ] **Step 3: Write `latest.json` and Agent usage documentation**

`latest.json` uses schema version`1` and exact absolute paths. `agent-usage.md` documents:

```text
Start: status -> one task-specific query -> normal project work
Live checks: only when process/GPU/remote state may have changed
End: record new run/path for later delta registration
Never: full rescan per task, infer deletion, or treat governance as an extra gate
```

- [ ] **Step 4: Add the minimal tracked `AGENTS.md` rule**

Add a short section that points Agents to `docs/project_governance/latest.json` and `agent-usage.md`. It must explicitly state that queries are navigation-only, current experimental safety rules remain authoritative, and deletion still needs explicit user approval.

- [ ] **Step 5: Mark the approved spec as implemented only after real verification**

Change the spec status from “书面设计等待用户复核” to “设计已批准；第一阶段已实现并验证”, and record the final implementation commit only after the code commits exist. Do not add hashes, seals or a new approval chain.

- [ ] **Step 6: Validate docs and pointer, then commit**

```bash
conda run -n ssr-gpu python tools/project_governance_inventory.py status \
  --latest docs/project_governance/latest.json --json
git diff --check
git add AGENTS.md docs/project_governance/latest.json \
  docs/project_governance/agent-usage.md \
  docs/superpowers/specs/2026-08-18-project-governance-agent-query-design.md
git commit -m "docs: connect agents to governance queries"
```

Before committing, assert cached files are exactly the four listed paths.

---

### Task 5: Full verification and handoff

**Files:**
- Modify only if a verified defect requires a new RED test and minimal fix.

**Interfaces:**
- Consumes all prior tasks.
- Produces fresh evidence that the query layer and previous governance scanner coexist.

- [ ] **Step 1: Run the complete governance test suite serially**

```bash
conda run -n ssr-gpu python -m pytest tests/test_project_governance*.py tests/test_n607_training_inventory.py -q
```

Expected: zero failures.

- [ ] **Step 2: Compile and inspect the exact implementation surface**

```bash
conda run -n ssr-gpu python -m compileall -q tools/project_governance tools/project_governance_inventory.py
git diff --check
git status -sb
```

Expected: compile and diff checks exit`0`; status retains only pre-existing user DOCX/QA/historical scan state outside intended commits.

- [ ] **Step 3: Run real read-only command probes**

```bash
conda run -n ssr-gpu python tools/project_governance_inventory.py status --latest docs/project_governance/latest.json --json
conda run -n ssr-gpu python tools/project_governance_inventory.py find E:/type10-7/code/snapshots/project_governance_20260813_wt --latest docs/project_governance/latest.json --limit 5 --json
conda run -n ssr-gpu python tools/project_governance_inventory.py review --latest docs/project_governance/latest.json --retention-class REVIEW_REQUIRED --limit 5 --json
```

Accept exit`3` only when the JSON explicitly reports the known conservative baseline warnings; no query may create or modify files.

- [ ] **Step 4: Review commits and finish the branch**

```bash
git log --oneline --decorate -8
git show --stat --oneline HEAD
git status -sb
```

Then invoke `superpowers:finishing-a-development-branch`, report the external SQLite path and size, command/test evidence, known warning semantics, and confirm deletion/move/overwrite/N607 write counts remain zero.
