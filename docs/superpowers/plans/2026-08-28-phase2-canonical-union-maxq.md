# Phase2 Canonical Union Max-Query Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 构建跨ManySig、ManyTx、ManyRx和SingleDay去重的WiSig物理样本索引，为不同Phase1 source receiver profile生成`MAXQ_ALL_UNIQUE`与`BALANCED_4DAY_CORE`两类`p2_min_v1`数据split，并由现有LEO cache、predictor bundle和独立scorer安全消费。

**Architecture:** 新增SQLite-backed canonical inventory，把IQ保留在原始`.pkl`中，只保存稳定物理ID、来源引用、覆盖和冲突；split builder在LEO overlay前冻结唯一scene、support/query角色与rank。现有cache builder增加canonical manifest输入分支，旧`stage2_registered`路径保持不变；predictor bundle增加`manifest_all`查询模式，独立summary在truth-last评分后计算micro和多维macro指标。

**Tech Stack:** Python 3、标准库`sqlite3`、NumPy、PyTorch、pytest、JSON/CSV/NPZ、现有`dataset_wisig.py`、`leo_weak_cache.py`和Stage2 predictor/scorer；本地`ssr-gpu`环境；Windows-native Git与N607短连接SSH/SCP。

**Spec:** `docs/superpowers/specs/2026-08-28-phase2-canonical-union-maxq-design.md`

## Global Constraints

- `项目.md`和`docs/PROJECT_PROTOCOL.md`继续定义科学语义；本计划不得改写`p2_min_v1`。
- 每个`physical_sample_id`只能绑定一个scene、一个overlay seed和一份received IQ；三scene物理ID两两不交。
- 每个scene、receiver和已注册类的support数严格等于K；query物理ID不得进入support。
- `R_t∩R_s=∅`；`Y_old`、`Y_new`和`Y_unknown`按协议互斥。
- `MAXQ_ALL_UNIQUE`使用K-specific split；`BALANCED_4DAY_CORE`使用冻结Kmax query池；两者不得共享`split_id`。
- query predictor不得读取truth、真实old/new/unknown角色、真实batch类数、quota或跨query重排信息。
- 跨`.pkl`相同坐标且IQ摘要一致的记录合并；摘要冲突的坐标进入冲突表并从可用池排除。
- 只使用`equalized=1`作为canonical输入；不得把`equalized=0/1`计为两个shot。
- 不修改或覆盖现有`VALIDATED_ONCE`数据；新数据使用新`capsule_id/split_id`。
- 本地测试前在Windows-native会话激活`ssr-gpu`。计划命令使用：

```bat
cmd.exe /d /c "call F:\App\miniconda3\Library\bin\conda.bat activate ssr-gpu && python -c \"import os,sys; print(sys.executable); print(os.environ.get('CONDA_PREFIX'))\""
```

- 不执行`pwsh`。N607操作前完整阅读PowerShell failure catalog并运行`powershell.exe -NoProfile -NonInteractive -ExecutionPolicy Bypass -File tools\n607_ssh_preflight.ps1`。
- 每个任务只stage列出的文件，commit后自动push，并以`git ls-remote origin refs/heads/work/cvs-active`核对远端OID；不得stage现有`.pytest_tmp/`、`local_artifacts/`或其他无关文件。
- 第一次真实数据操作先做只读inventory；不启动训练。真实checkpoint no-query smoke只在Task 9进行。

---

## File Structure

**Create**

- `code/cvsrffi/wisig_canonical_inventory.py`：读取compact WiSig、生成跨asset canonical ID、SQLite索引、覆盖和冲突摘要。
- `code/cvsrffi/phase2_canonical_split.py`：profile校验、receiver/class资格、scene分配、嵌套support与MAXQ/BAL4D split。
- `code/cvsrffi/phase2_canonical_summary.py`：在truth-last评分输出上计算sample/class/receiver/day/scene聚合。
- `code/scripts/audit_wisig_canonical_union.py`：只读扫描四个数据资产并输出inventory摘要、coverage CSV和冲突CSV。
- `code/scripts/build_phase2_canonical_splits.py`：从inventory与profile配置生成class selection和split manifest。
- `code/scripts/summarize_phase2_canonical_union.py`：汇总多receiver、多scene正式评分行。
- `configs/phase2_canonical_union_profiles_v1.json`：`SRC5_MAXP2`、receiver tiers、旧类、22个新类候选、K与query policy。
- `tests/test_wisig_canonical_inventory.py`：跨asset去重、冲突、equalized过滤和SQLite确定性。
- `tests/test_phase2_canonical_split.py`：receiver资格、scene互斥、K-shot、MAXQ和BAL4D。
- `tests/test_phase2_canonical_union_cli.py`：audit/build CLI合成数据端到端。
- `tests/test_phase2_canonical_summary.py`：micro和多维macro。

**Modify**

- `code/cvsrffi/leo_weak_cache.py`：允许已验证canonical physical ID作为sample ID，同时保留旧asset-SHA ID回退。
- `code/scripts/build_cvs_leo_weak_iq_cache.py`：增加`stage2_canonical_registered`输入分支，消费split manifest而非硬编码前三天120条。
- `code/scripts/build_cvs_stage2_predictor_bundle.py`：增加`manifest_all`查询策略，允许每类/scene可变query数。
- `code/cvsrffi/stage2_metric_scorer.py`：在正式prediction行中保留day/receiver/scene维度，供独立summary使用。
- `tests/test_build_cvs_stage2_predictor_bundle.py`：canonical partition与variable-query回归。
- `tests/test_adv3b02_paper_full_ci_plan.py`：cache build spec v3兼容负测。
- `docs/PHASE2_DATA_VALIDATION_APPENDIX.md`：补充canonical pool是一次性builder输入，不新增每方法gate。

---

### Task 1: Canonical Record Identity and Asset Reader

**Files:**
- Create: `code/cvsrffi/wisig_canonical_inventory.py`
- Create: `tests/test_wisig_canonical_inventory.py`

**Interfaces:**
- Consumes: WiSig compact-pkl字段`data/tx_list/rx_list/capture_date_list/equalized_list`。
- Produces: `RawRecordRef`、`canonical_coordinate()`、`canonical_physical_id()`和`iter_wisig_records()`，供Task 2建立SQLite inventory。

- [ ] **Step 1: Write the failing identity and overlap tests**

```python
import pickle
from pathlib import Path

import numpy as np
import pytest

from cvsrffi.wisig_canonical_inventory import (
    canonical_coordinate,
    canonical_physical_id,
    iter_wisig_records,
)


def _write_one_cell(path: Path, values: list[float]) -> Path:
    eq0 = np.zeros((len(values), 8, 2), dtype=np.float32)
    eq1 = np.stack(
        [np.full((8, 2), value, dtype=np.float32) for value in values]
    )
    payload = {
        "data": [[[[eq0, eq1]]]],
        "tx_list": ["tx0"],
        "rx_list": ["rx0"],
        "capture_date_list": ["day0"],
        "equalized_list": [0, 1],
    }
    with path.open("wb") as handle:
        pickle.dump(payload, handle)
    return path


@pytest.fixture
def fake_wisig_pkl(tmp_path: Path) -> Path:
    return _write_one_cell(tmp_path / "ManyTx.pkl", [1.0, 2.0])


def test_canonical_identity_does_not_depend_on_asset_name(tmp_path: Path):
    left = canonical_physical_id("tx0", "rx0", "day0", "1", "7")
    right = canonical_physical_id("tx0", "rx0", "day0", "1", "7")
    assert left == right
    assert canonical_coordinate("tx0", "rx0", "day0", "1", "7") == (
        "tx0", "rx0", "day0", "1", "7"
    )


def test_reader_emits_only_equalized_one(fake_wisig_pkl: Path):
    rows = list(iter_wisig_records(fake_wisig_pkl, "ManyTx", equalized=1))
    assert rows
    assert {row.eq_id for row in rows} == {"1"}
    assert all(row.iq_sha256 for row in rows)
```

- [ ] **Step 2: Run the focused test and verify failure**

Run:

```bat
cmd.exe /d /c "call F:\App\miniconda3\Library\bin\conda.bat activate ssr-gpu && python -m pytest -p no:cacheprovider tests\test_wisig_canonical_inventory.py -q"
```

Expected: FAIL with `ModuleNotFoundError: cvsrffi.wisig_canonical_inventory`.

- [ ] **Step 3: Implement immutable record primitives**

```python
@dataclass(frozen=True)
class RawRecordRef:
    physical_sample_id: str
    asset_name: str
    dataset_path: str
    source_record_index: int
    tx_id: str
    rx_id: str
    day_id: str
    eq_id: str
    sig_id: str
    iq_sha256: str


def canonical_coordinate(
    tx_id: str, rx_id: str, day_id: str, eq_id: str, sig_id: str
) -> tuple[str, str, str, str, str]:
    return tuple(map(str, (tx_id, rx_id, day_id, eq_id, sig_id)))


def canonical_physical_id(
    tx_id: str, rx_id: str, day_id: str, eq_id: str, sig_id: str
) -> str:
    payload = canonical_coordinate(tx_id, rx_id, day_id, eq_id, sig_id)
    encoded = json.dumps(payload, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(encoded.encode("ascii")).hexdigest()
```

`iter_wisig_records()` must validate required pkl keys, resolve labels rather than raw indices, calculate source-record indices in deterministic nested traversal order, hash C-contiguous float bytes, and reject a requested equalization label absent from the file.

- [ ] **Step 4: Run the test and existing WiSig split regression**

Run:

```bat
cmd.exe /d /c "call F:\App\miniconda3\Library\bin\conda.bat activate ssr-gpu && python -m pytest -p no:cacheprovider tests\test_wisig_canonical_inventory.py tests\test_wisig_random_split.py -q"
```

Expected: PASS; existing random split tests remain unchanged.

- [ ] **Step 5: Commit and verify remote**

```bat
git.exe add -- code/cvsrffi/wisig_canonical_inventory.py tests/test_wisig_canonical_inventory.py
git.exe commit -m "feat: add canonical WiSig record identity"
git.exe rev-parse HEAD
git.exe ls-remote origin refs/heads/work/cvs-active
```

Expected: local and remote OIDs match.

---

### Task 2: SQLite Deduplication and Read-Only Audit CLI

**Files:**
- Modify: `code/cvsrffi/wisig_canonical_inventory.py`
- Modify: `tests/test_wisig_canonical_inventory.py`
- Create: `code/scripts/audit_wisig_canonical_union.py`
- Create: `tests/test_phase2_canonical_union_cli.py`

**Interfaces:**
- Consumes: `Iterator[RawRecordRef]` from Task 1.
- Produces: `build_inventory(asset_paths, sqlite_path) -> InventorySummary` and CLI outputs`summary.json/coverage.csv/conflicts.csv`.

- [ ] **Step 1: Write failing deduplication tests**

```python
@pytest.fixture
def overlapping_assets(tmp_path):
    return {
        "ManyTx": _write_one_cell(tmp_path / "ManyTx.pkl", [1.0, 2.0]),
        "SingleDay": _write_one_cell(
            tmp_path / "SingleDay.pkl", [1.0, 2.0, 3.0]
        ),
    }


@pytest.fixture
def conflicting_assets(tmp_path):
    return {
        "ManyTx": _write_one_cell(tmp_path / "ManyTx.pkl", [1.0]),
        "SingleDay": _write_one_cell(tmp_path / "SingleDay.pkl", [9.0]),
    }


def test_inventory_merges_same_coordinate_and_digest(tmp_path, overlapping_assets):
    db_path = tmp_path / "canonical.sqlite"
    summary = build_inventory(overlapping_assets, db_path, equalized=1)
    assert summary.source_record_count == 5
    assert summary.canonical_record_count == 3
    assert summary.merged_duplicate_count == 2
    assert summary.conflict_count == 0


def test_inventory_excludes_coordinate_digest_conflict(tmp_path, conflicting_assets):
    summary = build_inventory(
        conflicting_assets, tmp_path / "canonical.sqlite", equalized=1
    )
    assert summary.conflict_count == 1
    assert summary.eligible_record_count == summary.canonical_record_count - 1
```

- [ ] **Step 2: Run tests and verify expected failure**

Run the Task 1 pytest command. Expected: FAIL because`build_inventory`is not defined.

- [ ] **Step 3: Implement the SQLite schema and transactional merge**

Create tables:

```sql
CREATE TABLE canonical_records (
  physical_sample_id TEXT PRIMARY KEY,
  tx_id TEXT NOT NULL,
  rx_id TEXT NOT NULL,
  day_id TEXT NOT NULL,
  eq_id TEXT NOT NULL,
  sig_id TEXT NOT NULL,
  iq_sha256 TEXT NOT NULL,
  preferred_asset TEXT NOT NULL,
  preferred_source_record_index INTEGER NOT NULL,
  eligible INTEGER NOT NULL CHECK (eligible IN (0,1))
);
CREATE TABLE record_sources (
  physical_sample_id TEXT NOT NULL,
  asset_name TEXT NOT NULL,
  dataset_path TEXT NOT NULL,
  source_record_index INTEGER NOT NULL,
  iq_sha256 TEXT NOT NULL,
  PRIMARY KEY (physical_sample_id, asset_name, source_record_index)
);
CREATE TABLE identity_conflicts (
  physical_sample_id TEXT NOT NULL,
  first_iq_sha256 TEXT NOT NULL,
  conflicting_iq_sha256 TEXT NOT NULL,
  asset_name TEXT NOT NULL
);
```

Use asset preference`ManySig,SingleDay,ManyRx,ManyTx`only to select a materialization reference after equal digest proves identity; preference never changes counts.

Return:

```python
@dataclass(frozen=True)
class InventorySummary:
    source_record_count: int
    canonical_record_count: int
    eligible_record_count: int
    merged_duplicate_count: int
    conflict_count: int
```

- [ ] **Step 4: Implement the read-only audit CLI**

CLI:

```text
python code/scripts/audit_wisig_canonical_union.py
  --asset ManySig=/home/szu2070436088/2510044040/CV-SincNet/Dataset_WigSig/ManySig.pkl
  --asset ManyTx=/home/szu2070436088/2510044040/CV-SincNet/Dataset_WigSig/ManyTx.pkl
  --asset ManyRx=/home/szu2070436088/2510044040/CV-SincNet/Dataset_WigSig/ManyRx.pkl
  --asset SingleDay=/home/szu2070436088/2510044040/CV-SincNet/Dataset_WigSig/SingleDay.pkl
  --sqlite-out /home/szu2070436088/2510044040/CV-SincNet/runs/P2_CANONICAL_UNION_AUDIT_V1_20260828/inventory/canonical.sqlite
  --summary-json /home/szu2070436088/2510044040/CV-SincNet/runs/P2_CANONICAL_UNION_AUDIT_V1_20260828/inventory/summary.json
  --coverage-csv /home/szu2070436088/2510044040/CV-SincNet/runs/P2_CANONICAL_UNION_AUDIT_V1_20260828/inventory/coverage.csv
  --conflicts-csv /home/szu2070436088/2510044040/CV-SincNet/runs/P2_CANONICAL_UNION_AUDIT_V1_20260828/inventory/conflicts.csv
  --equalized 1
```

The CLI must use exclusive creation, reject duplicate asset names, emit no IQ bytes, and write coverage columns`tx_id,rx_id,day_id,record_count,asset_count`.

- [ ] **Step 5: Add and run CLI end-to-end test**

The test invokes`main(argv)`with this exact argument list:

```python
argv = [
    "--asset", f"ManySig={manysig}",
    "--asset", f"ManyTx={manytx}",
    "--asset", f"ManyRx={manyrx}",
    "--asset", f"SingleDay={single_day}",
    "--sqlite-out", str(tmp_path / "canonical.sqlite"),
    "--summary-json", str(tmp_path / "summary.json"),
    "--coverage-csv", str(tmp_path / "coverage.csv"),
    "--conflicts-csv", str(tmp_path / "conflicts.csv"),
    "--equalized", "1",
]
assert main(argv) == 0
assert summary["protocol_schema"] == "p2_min_v1"
assert summary["equalized"] == "1"
assert summary["source_record_count"] == 5
assert summary["canonical_record_count"] == 3
assert set(coverage_rows[0]) == {
    "tx_id", "rx_id", "day_id", "record_count", "asset_count"
}
```

Run:

```bat
cmd.exe /d /c "call F:\App\miniconda3\Library\bin\conda.bat activate ssr-gpu && python -m pytest -p no:cacheprovider tests\test_wisig_canonical_inventory.py tests\test_phase2_canonical_union_cli.py -q"
```

Expected: PASS.

- [ ] **Step 6: Commit and verify remote**

Stage only the four files in this task; commit`feat: audit canonical WiSig union`; verify remote OID.

---

### Task 3: Profile Schema and Deterministic Coverage Selection

**Files:**
- Create: `configs/phase2_canonical_union_profiles_v1.json`
- Create: `code/cvsrffi/phase2_canonical_split.py`
- Create: `tests/test_phase2_canonical_split.py`

**Interfaces:**
- Consumes: canonical SQLite coverage from Task 2.
- Produces: `CanonicalProfile`、`rank_new_classes(connection, profile, scenario_by_sample)`、`eligible_receivers()` and deterministic`Y_new5/10/20`.

- [ ] **Step 1: Add the exact profile config**

The JSON declares:

```json
{
  "schema": "cvs.phase2.canonical_union_profile.v1",
  "protocol_schema": "p2_min_v1",
  "source_profile_id": "SRC5_MAXP2",
  "source_receivers": ["1-19", "18-2", "19-2", "2-19", "3-19"],
  "receiver_tiers": {
    "dense": ["1-1", "14-7", "2-1", "20-1", "7-14", "7-7", "8-8"],
    "single_day": ["13-13", "2-20", "8-13"],
    "many_tx": ["1-20", "13-7", "18-19", "19-1", "20-19", "8-14", "8-7"]
  },
  "old_tx_ids": ["14-10", "14-7", "20-15", "20-19", "6-15", "8-20"],
  "new_tx_candidates": ["1-11", "10-11", "10-7", "11-1", "11-17", "11-4", "11-7", "13-3", "15-1", "16-16", "2-19", "20-12", "20-7", "3-13", "3-18", "4-11", "5-5", "6-1", "7-10", "7-11", "8-18", "8-3"],
  "new_class_sizes": [5, 10, 20],
  "k_values": [1, 5, 10, 20],
  "k_max": 20,
  "scenarios": ["leo_clear_weak", "leo_low_elev_weak", "leo_rain_weak"],
  "query_policies": ["MAXQ_ALL_UNIQUE", "BALANCED_4DAY_CORE"]
}
```

- [ ] **Step 2: Write failing profile and ranking tests**

```python
def test_profile_rejects_source_target_overlap(profile_payload):
    profile_payload["receiver_tiers"]["dense"].append("1-19")
    with pytest.raises(ValueError, match="R_t"):
        CanonicalProfile.from_mapping(profile_payload)


def test_new_class_ranking_is_nested_and_coverage_only(
    fake_inventory, profile, scenario_by_sample
):
    selected = rank_new_classes(
        fake_inventory, profile, scenario_by_sample
    )
    assert selected[5] == selected[10][:5]
    assert selected[10] == selected[20][:10]
    assert len(set(selected[20])) == 20
```

- [ ] **Step 3: Implement profile validation and ranking**

`CanonicalProfile.from_mapping()` enforces exact schema, disjoint receiver sets, six exact old TX, 22 unique candidates, ordered K`[1,5,10,20]` and three formal scenarios.

```python
@dataclass(frozen=True)
class CanonicalProfile:
    schema: str
    protocol_schema: str
    source_profile_id: str
    source_receivers: tuple[str, ...]
    receiver_tiers: Mapping[str, tuple[str, ...]]
    old_tx_ids: tuple[str, ...]
    new_tx_candidates: tuple[str, ...]
    new_class_sizes: tuple[int, ...]
    k_values: tuple[int, ...]
    k_max: int
    scenarios: tuple[str, ...]
    query_policies: tuple[str, ...]

    @classmethod
    def from_mapping(cls, payload: Mapping[str, object]) -> "CanonicalProfile":
        """Validate the v1 schema and return normalized immutable fields."""
```

`rank_new_classes()` sorts only by:

```python
key = (
    -int(all_receiver_kmax_feasible),
    -int(first_three_day_coverage),
    -int(unique_query_capacity),
    -int(min_receiver_day_scene_count),
    str(tx_id),
)
```

It must not accept predictions, metrics or truth inputs.

- [ ] **Step 4: Implement receiver eligibility**

```python
def eligible_receivers(
    connection: sqlite3.Connection,
    *,
    registered_tx_ids: Sequence[str],
    candidate_receivers: Sequence[str],
    scenario_by_sample: Mapping[str, str],
    k: int,
) -> tuple[str, ...]:
    """Return receivers with at least K rows for every class and scene."""
```

- [ ] **Step 5: Run tests**

```bat
cmd.exe /d /c "call F:\App\miniconda3\Library\bin\conda.bat activate ssr-gpu && python -m pytest -p no:cacheprovider tests\test_phase2_canonical_split.py tests\test_wisig_canonical_inventory.py -q"
```

Expected: PASS.

- [ ] **Step 6: Commit and verify remote**

Stage the config, split module and test; commit`feat: define canonical Phase2 profiles`; verify remote OID.

---

### Task 4: Scene Assignment, Nested Support and Dual Split Builder

**Files:**
- Modify: `code/cvsrffi/phase2_canonical_split.py`
- Modify: `tests/test_phase2_canonical_split.py`
- Create: `code/scripts/build_phase2_canonical_splits.py`
- Modify: `tests/test_phase2_canonical_union_cli.py`

**Interfaces:**
- Consumes: SQLite inventory and`CanonicalProfile`.
- Produces: `assign_scenes()`、`build_split_manifest()` and immutable split JSON.

- [ ] **Step 1: Write failing scenario and split tests**

```python
def test_scene_assignment_is_deterministic_and_disjoint(records):
    first = assign_scenes(records, seed=713101)
    second = assign_scenes(records, seed=713101)
    assert first == second
    ids = {scene: set() for scene in FORMAL_LEO_WEAK_SCENARIOS}
    for sample_id, scene in first.items():
        ids[scene].add(sample_id)
    assert all(ids[a].isdisjoint(ids[b]) for a, b in combinations(ids, 2))


@pytest.mark.parametrize("k", [1, 5, 10, 20])
def test_maxq_uses_every_non_support_record(records, profile, k):
    manifest = build_split_manifest(records, profile, k, "MAXQ_ALL_UNIQUE")
    assert set(manifest.support_ids).isdisjoint(manifest.query_ids)
    assert len(manifest.query_ids) == len(manifest.eligible_ids) - len(manifest.support_ids)
```

Add a BAL4D test proving all`TX×receiver×day×scene`query cells have the same frozen count and K=1/5/10/20 query IDs are identical.

- [ ] **Step 2: Run tests and verify failure**

Run Task 3 pytest command. Expected: FAIL because scene and split functions are absent.

- [ ] **Step 3: Implement scene assignment**

Group every`eligible=1`canonical record by`(tx_id,rx_id,day_id)`, sort by a SHA256 rank of`seed|physical_sample_id`, and assign scenes round-robin with a deterministic group rotation. Store`scene`and`scene_rank`in the manifest; do not duplicate records. Run class ranking and receiver eligibility only after this assignment exists.

- [ ] **Step 4: Implement nested support ranking**

For each`(receiver,scene,class)`, rank by SHA256 of`support_seed|physical_sample_id`. The firstK rows are support for MAXQ; ranks`0:1`、`0:5`、`0:10`、`0:20`guarantee nesting. For BAL4D, reserve the first20 rows and keep its query complement fixed for every K.

- [ ] **Step 5: Implement query policies**

`build_split_manifest()` returns:

```python
@dataclass(frozen=True)
class SplitManifest:
    protocol_schema: str
    profile_id: str
    query_policy: str
    k: int
    registered_tx_ids: tuple[str, ...]
    eligible_receivers: tuple[str, ...]
    rows: tuple[SplitRow, ...]
    capsule_id: str
    split_id: str
```

Each`SplitRow` contains`physical_sample_id,source_asset,source_record_index,tx_id,rx_id,day_id,scene,role,rank`. Compute`capsule_id`from physical IDs plus scene assignment and`split_id`from capsule ID plus receiver/TX/K/support-query identity.

```python
@dataclass(frozen=True)
class SplitRow:
    physical_sample_id: str
    source_asset: str
    source_record_index: int
    tx_id: str
    rx_id: str
    day_id: str
    scene: str
    role: str
    rank: int
```

- [ ] **Step 6: Implement CLI and exclusive outputs**

`build_phase2_canonical_splits.py` accepts`--inventory, --profile, --out-root, --seed` and writes:

```text
class_selection.json
MAXQ_ALL_UNIQUE/k1.json
MAXQ_ALL_UNIQUE/k5.json
MAXQ_ALL_UNIQUE/k10.json
MAXQ_ALL_UNIQUE/k20.json
BALANCED_4DAY_CORE/k1.json
BALANCED_4DAY_CORE/k5.json
BALANCED_4DAY_CORE/k10.json
BALANCED_4DAY_CORE/k20.json
```

Refuse an existing output root.

- [ ] **Step 7: Run split and CLI tests**

Run:

```bat
cmd.exe /d /c "call F:\App\miniconda3\Library\bin\conda.bat activate ssr-gpu && python -m pytest -p no:cacheprovider tests\test_phase2_canonical_split.py tests\test_phase2_canonical_union_cli.py -q"
```

Expected: PASS.

- [ ] **Step 8: Commit and verify remote**

Commit`feat: build max-query Phase2 splits`and verify remote OID.

---

### Task 5: Canonical IDs in LEO Cache Verification

**Files:**
- Modify: `code/cvsrffi/leo_weak_cache.py`
- Modify: `code/scripts/build_cvs_leo_weak_iq_cache.py`
- Create: `tests/test_phase2_canonical_cache.py`
- Modify: `tests/test_adv3b02_paper_full_ci_plan.py`

**Interfaces:**
- Consumes: Task 4 split manifest.
- Produces: verified LEO weak cache-set with canonical sample IDs and optional`split_roles/split_ranks`arrays.

- [ ] **Step 1: Write failing canonical cache tests**

Tests must prove:

```python
assert physical_sample_id(arrays_with_canonical_id, 0) == "canonical-id-0"
assert physical_sample_id(legacy_arrays, 0) == legacy_expected_id
```

Build-spec tests must accept`cache_scope=stage2_canonical_registered`with`canonical_inventory`and`split_manifest`, reject simultaneous legacy`role_specs`, and keep the existing v2 spec accepted unchanged.

- [ ] **Step 2: Run focused tests and verify failure**

Run:

```bat
cmd.exe /d /c "call F:\App\miniconda3\Library\bin\conda.bat activate ssr-gpu && python -m pytest -p no:cacheprovider tests\test_phase2_canonical_cache.py tests\test_adv3b02_paper_full_ci_plan.py -q"
```

Expected: canonical tests FAIL; legacy tests PASS.

- [ ] **Step 3: Extend cache identity without changing legacy IDs**

In`physical_sample_id()`:

```python
canonical = arrays.get("canonical_physical_sample_ids")
if canonical is not None:
    value = str(np.asarray(canonical)[index])
    if not value:
        raise ValueError("canonical physical sample ID must be nonempty")
    return value
return physical_sample_id_from_values(
    dataset_sha256=str(arrays["source_dataset_sha256"][index]),
    source_record_index=int(arrays["source_record_indices"][index]),
    role=str(arrays["dataset_role"][index]),
    tx_id=str(arrays["tx_ids"][index]),
    rx_id=str(arrays["rx_ids"][index]),
    day_id=str(arrays["day_ids"][index]),
    eq_id=str(arrays["eq_ids"][index]),
    sig_id=str(arrays["sig_ids"][index]),
)
```

Add optional-array length validation for`canonical_physical_sample_ids,split_roles,split_ranks`.

- [ ] **Step 4: Add build spec v3 canonical scope**

Add:

```python
CANONICAL_BUILD_SPEC_SCHEMA = "cvs_leo_weak_iq_cache_build_spec_v3"
SCOPE_ROLES["stage2_canonical_registered"] = {"target_old", "target_new"}
```

The canonical branch requires`canonical_inventory`and`split_manifest`, forbids`role_specs`, accepts all four declared days, and never applies the legacy`120 physical samples per TX`cap.

- [ ] **Step 5: Materialize selected rows by source reference**

Load each source pkl once, resolve`source_asset/source_record_index`from the inventory, verify the IQ digest matches inventory, and construct one dataset per scene from manifest rows. Preserve`canonical_physical_sample_ids`、`split_roles`and`split_ranks`through NPZ output. The existing overlay function remains the only LEO generator.

- [ ] **Step 6: Run canonical and legacy suites**

```bat
cmd.exe /d /c "call F:\App\miniconda3\Library\bin\conda.bat activate ssr-gpu && python -m pytest -p no:cacheprovider tests\test_phase2_canonical_cache.py tests\test_adv3b02_paper_full_ci_plan.py tests\test_somph_cache_build_matrix.py -q"
```

Expected: PASS.

- [ ] **Step 7: Commit and verify remote**

Commit`feat: materialize canonical Phase2 LEO cache`and verify remote OID.

---

### Task 6: Manifest-All Predictor Capsule

**Files:**
- Modify: `code/scripts/build_cvs_stage2_predictor_bundle.py`
- Modify: `tests/test_build_cvs_stage2_predictor_bundle.py`

**Interfaces:**
- Consumes: canonical cache arrays`split_roles/split_ranks`.
- Produces: support NPZ、query NPZ和truth sidecar，其中predictor artifact不含truth或role。

- [ ] **Step 1: Write failing variable-query tests**

Create three synthetic scenario caches with equal registered classes but different query counts. Assert:

```python
result = builder.build(args_with_manifest_all)
assert result["query_policy"] == "manifest_all"
assert result["query_count_by_scenario"] == {
    "leo_clear_weak": 17,
    "leo_low_elev_weak": 16,
    "leo_rain_weak": 15,
}
```

Also assert support count is exactly`K×registered_class_count`per scene, query tokens are disjoint across scenes, and query NPZ member names contain no truth/role/class-count fields.

- [ ] **Step 2: Run focused test and verify failure**

Run:

```bat
cmd.exe /d /c "call F:\App\miniconda3\Library\bin\conda.bat activate ssr-gpu && python -m pytest -p no:cacheprovider tests\test_build_cvs_stage2_predictor_bundle.py -q"
```

Expected: new`manifest_all`test FAIL.

- [ ] **Step 3: Add query policy CLI**

Add`--query-policy`with choices`fixed_per_tx/manifest_all`. Legacy default remains`fixed_per_tx`and still requires positive`--query-per-tx`. Canonical mode requires`query_per_tx=0`and cache`split_roles`exactly`support/query`.

- [ ] **Step 4: Select manifest rows without query truncation**

In`manifest_all`mode:

- support rows are`split_roles=="support"`ordered by`split_ranks`;
- query rows are every`split_roles=="query"`row;
- validate exact K for every`receiver×scene×registered class`;
- allow scenario-specific query counts;
- compare registered class order across scenes, not the full query structure;
- preserve truth only in the independent sidecar.

- [ ] **Step 5: Run focused and adjacent regressions**

```bat
cmd.exe /d /c "call F:\App\miniconda3\Library\bin\conda.bat activate ssr-gpu && python -m pytest -p no:cacheprovider tests\test_build_cvs_stage2_predictor_bundle.py tests\test_stage2_predictor_entry.py tests\test_stage2_sealed_pipeline_integration.py -q"
```

Expected: PASS.

- [ ] **Step 6: Commit and verify remote**

Commit`feat: support manifest-all Stage2 query capsules`and verify remote OID.

---

### Task 7: Truth-Last Multi-Dimensional Summary

**Files:**
- Create: `code/cvsrffi/phase2_canonical_summary.py`
- Create: `code/scripts/summarize_phase2_canonical_union.py`
- Create: `tests/test_phase2_canonical_summary.py`
- Modify: `code/cvsrffi/stage2_metric_scorer.py`

**Interfaces:**
- Consumes: immutable formal prediction rows after independent truth join.
- Produces: sample-micro、class-macro、receiver-macro、day-macro、scene-macro和cell-level counts。

- [ ] **Step 1: Write failing aggregation tests**

Use imbalanced synthetic rows where one receiver/day has eight correct rows and another has two incorrect rows. Assert:

```python
summary = summarize_scored_rows(rows)
assert summary["sample_micro_accuracy"] == 0.8
assert summary["receiver_macro_accuracy"] == 0.5
assert summary["day_macro_accuracy"] == 0.5
```

Add exact expected class/scene macros and reject duplicate`scenario+query_token`keys.

- [ ] **Step 2: Run test and verify failure**

Expected: module import failure.

- [ ] **Step 3: Preserve scorer dimensions**

Ensure each formal scored prediction row contains`true_class_index,predicted_class_index,receiver_label,day_label,scenario,query_token`. These fields exist only after truth-last join; predictor artifacts remain unchanged.

- [ ] **Step 4: Implement summary**

`summarize_scored_rows(rows)` groups correctness by sample, class, receiver, day and scene, computes unweighted mean over group accuracies, and emits group sizes so class/day imbalance remains visible.

- [ ] **Step 5: Implement exclusive-output CLI and run tests**

CLI accepts one or more formal prediction JSON/JSONL files and writes a new`summary.json`plus`cell_metrics.csv`using exclusive creation.

Run:

```bat
cmd.exe /d /c "call F:\App\miniconda3\Library\bin\conda.bat activate ssr-gpu && python -m pytest -p no:cacheprovider tests\test_phase2_canonical_summary.py tests\test_stage2_metric_scorer.py -q"
```

Expected: PASS.

- [ ] **Step 6: Commit and verify remote**

Commit`feat: summarize canonical Phase2 metrics`and verify remote OID.

---

### Task 8: Synthetic End-to-End and Protocol Negative Suite

**Files:**
- Modify: `tests/test_phase2_canonical_union_cli.py`
- Modify: `tests/test_phase2_canonical_cache.py`
- Modify: `docs/PHASE2_DATA_VALIDATION_APPENDIX.md`

**Interfaces:**
- Consumes: Tasks 1–7.
- Produces: one synthetic chain`assets→inventory→split→LEO cache→predictor capsule→truth-last summary`.

- [ ] **Step 1: Add an end-to-end synthetic test**

The test creates four overlapping pkl files, builds K=1 MAXQ and BAL4D splits, generates CPU LEO caches, builds a predictor capsule, creates deterministic fake predictions, joins truth in the scorer, and verifies exact counts.

- [ ] **Step 2: Add focused protocol negatives**

Tests must reject:

- same physical ID in two scenes;
- same physical ID in support and query;
- source/target receiver overlap;
- old/new TX overlap;
- K shortfall in one class/scene;
- query truth or role inside predictor NPZ;
- coordinate-equal but IQ-different duplicate treated as eligible;
- overwriting an existing output root.

- [ ] **Step 3: Update the validation appendix**

Add one short section stating canonical inventory and cross-asset digest checks are one-time builder responsibilities; runtime still consumes only`protocol_schema/capsule_id/split_id/phase2_data_status`. Explicitly state this creates no per-method hash, seal, receipt or revalidation gate.

- [ ] **Step 4: Run the complete focused suite serially**

```bat
cmd.exe /d /c "call F:\App\miniconda3\Library\bin\conda.bat activate ssr-gpu && python -m pytest -p no:cacheprovider tests\test_wisig_canonical_inventory.py tests\test_phase2_canonical_split.py tests\test_phase2_canonical_union_cli.py tests\test_phase2_canonical_cache.py tests\test_build_cvs_stage2_predictor_bundle.py tests\test_phase2_canonical_summary.py -q"
```

Then:

```bat
cmd.exe /d /c "call F:\App\miniconda3\Library\bin\conda.bat activate ssr-gpu && python -m py_compile code\cvsrffi\wisig_canonical_inventory.py code\cvsrffi\phase2_canonical_split.py code\cvsrffi\phase2_canonical_summary.py code\scripts\audit_wisig_canonical_union.py code\scripts\build_phase2_canonical_splits.py code\scripts\summarize_phase2_canonical_union.py"
git.exe diff --check
```

Expected: all tests and compile checks PASS.

- [ ] **Step 5: Perform the single P0/P1 review**

Review only defects that could directly produce duplicate physical IDs, query leakage, wrong receiver/TX/K/scene, output overwrite, invalid prediction, or a non-starting builder. P2 documentation/style findings are nonblocking. If a direct P0/P1 is fixed, perform one scoped re-review only.

- [ ] **Step 6: Commit and verify remote**

Commit`test: close canonical Phase2 data chain`and verify remote OID.

---

### Task 9: N607 Read-Only Inventory and Frozen Profiles

**Files:**
- Create from audit output: `docs/data/PHASE2_CANONICAL_UNION_INVENTORY_20260828.md`
- Modify from deterministic audit: `configs/phase2_canonical_union_profiles_v1.json`
- Create local formal record: `E:\type10-7\automation_reports\CV-SincNet\P2_CANONICAL_UNION_AUDIT_V1_20260828\report.md`
- Create Git mirror: `docs/reports/P2_CANONICAL_UNION_AUDIT_V1_20260828.md`

**Interfaces:**
- Consumes: verified local audit implementation and the four N607 WiSig files.
- Produces: exact deduplicated counts、frozen`Y_new5/10/20`、eligible receiver lists and exact MAXQ/BAL4D split counts。

- [ ] **Step 1: Record the audit preregistration**

Use run ID`P2_CANONICAL_UNION_AUDIT_V1_20260828`and these exact paths:

```text
local release:
E:\type10-7\release_archives\P2_CANONICAL_UNION_AUDIT_V1_20260828.zip

remote root:
/home/szu2070436088/2510044040/CV-SincNet/runs/P2_CANONICAL_UNION_AUDIT_V1_20260828

datasets:
/home/szu2070436088/2510044040/CV-SincNet/Dataset_WigSig/ManySig.pkl
/home/szu2070436088/2510044040/CV-SincNet/Dataset_WigSig/ManyTx.pkl
/home/szu2070436088/2510044040/CV-SincNet/Dataset_WigSig/ManyRx.pkl
/home/szu2070436088/2510044040/CV-SincNet/Dataset_WigSig/SingleDay.pkl
```

Record current commit, command,`ssr-gpu`environment, immutable output root, expected inventory files and stop rule. This is a data audit, not a performance experiment.

- [ ] **Step 2: Run direct N607 preflight**

```bat
powershell.exe -NoProfile -NonInteractive -ExecutionPolicy Bypass -File tools\n607_ssh_preflight.ps1
```

Expected: direct identity、server time、project root and GPU visibility are VERIFIED. If only direct TCP fails, use the documented lab bridge; identity ambiguity stops the task.

- [ ] **Step 3: Build and sync the audit release archive**

```bat
git.exe archive --format=zip --output=E:\type10-7\release_archives\P2_CANONICAL_UNION_AUDIT_V1_20260828.zip HEAD code/cvsrffi/wisig_canonical_inventory.py code/cvsrffi/phase2_canonical_split.py code/scripts/audit_wisig_canonical_union.py code/scripts/build_phase2_canonical_splits.py configs/phase2_canonical_union_profiles_v1.json
certutil.exe -hashfile E:\type10-7\release_archives\P2_CANONICAL_UNION_AUDIT_V1_20260828.zip SHA256
```

Confirm the remote root does not exist, create it, copy the archive with`scp.exe`, compare one remote`sha256sum`to the local SHA, extract under`/home/szu2070436088/2510044040/CV-SincNet/runs/P2_CANONICAL_UNION_AUDIT_V1_20260828/release`and run remote`py_compile`once.

- [ ] **Step 4: Run inventory and split enumeration**

Write under`/home/szu2070436088/2510044040/CV-SincNet/runs/P2_CANONICAL_UNION_AUDIT_V1_20260828/inventory`:

- source、canonical、duplicate and conflict counts;
-`TX×RX×day`coverage;
- 17 candidate receiver K=1/5/10/20 eligibility;
- deterministic ranking of 22 new-class candidates;
- exact`Y_new5/10/20`;
- exact MAXQ and BAL4D support/query counts.

If conflicts exist, mark those IDs ineligible; do not infer independence.

- [ ] **Step 5: Pull small outputs and freeze locally**

Copy only`summary.json,coverage.csv,conflicts.csv,class_selection.json`and eight split manifests to a new local audit directory. Update the profile JSON locally with audit-derived selected TX and eligible receiver lists, render both reports, run the focused profile/split tests and`git diff --check`.

- [ ] **Step 6: Commit the frozen inventory**

Stage only the profile、inventory report and mirrored report. Commit`docs: freeze canonical Phase2 inventory`, auto-push and verify remote OID. The audit does not produce`VALIDATED_ONCE`data.

---

### Task 10: Canonical Cache Validation and Real-Checkpoint No-Query Smoke

**Files:**
- Create: `configs/phase2_canonical_union_k20_smoke_v1.json`
- Create local formal record: `E:\type10-7\automation_reports\CV-SincNet\P2_CANONICAL_UNION_SMOKE_V1_20260828\report.md`
- Create Git mirror: `docs/reports/P2_CANONICAL_UNION_SMOKE_V1_20260828.md`

**Interfaces:**
- Consumes: Task 9 frozen profile/split and the verified ADV3B02 checkpoint.
- Produces: one K=20 canonical cache-set、`VALIDATED_ONCE`handles and one support-only no-query smoke artifact。

- [ ] **Step 1: Create and locally validate the smoke config**

Use:

```text
run ID:
P2_CANONICAL_UNION_SMOKE_V1_20260828

remote root:
/home/szu2070436088/2510044040/CV-SincNet/runs/P2_CANONICAL_UNION_SMOKE_V1_20260828

checkpoint:
/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3_mechanism32_queue_20260701/ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth
```

The config references the smallest audit-confirmed K=20 receiver/class row, its exact`capsule_id/split_id`, one GPU, support-only input and exclusive output paths. Run config parsing tests before commit.

- [ ] **Step 2: Commit config and prepare one smoke release**

Commit`config: add canonical Phase2 no-query smoke`and verify remote OID. Build one archive`E:\type10-7\release_archives\P2_CANONICAL_UNION_SMOKE_V1_20260828.zip`from that commit containing the canonical modules、cache/predictor builders、frozen profile、K=20 split and smoke config. Compare the archive SHA once locally and remotely.

- [ ] **Step 3: Run preflight, remote compile and cache validation**

Run the direct N607 preflight again, verify the smoke root does not exist, sync/extract the archive, compile once and materialize the exact K=20 cache-set. Advance the data slice to`VALIDATED_ONCE`only after the existing verifier confirms schema、capsule ID、split ID、single observation and all physical-ID disjointness checks.

- [ ] **Step 4: Run the real-checkpoint no-query smoke**

Invoke the recorded checkpoint with query input omitted. The smoke passes only if support-only adaptation/registration and expected artifact creation complete without accessing query members. No prediction or performance metric is produced.

- [ ] **Step 5: Verify ownership and artifacts**

Immediately check PID/CWD/cmdline/run-root、GPU mapping and log growth once. After exit, verify the cache manifest、`capsule_id/split_id`、no-query artifact and log parse. Do not leave SSH sessions open.

- [ ] **Step 6: Close and publish the smoke report**

Append final status、exact cache/support counts、anomalies and`NO_PERFORMANCE_RESULT`to the root report, mirror it to Git, run`git diff --check`, commit`docs: validate canonical Phase2 smoke`, push and verify remote OID.

Expected final state: code`LOCAL_VERIFIED`; the exact smoke data slice`VALIDATED_ONCE`; no`ARTIFACTS_COMPLETE`or`ANALYZED`performance claim.

---

## Final Verification Checklist

- [ ] All focused tests pass in`ssr-gpu`.
- [ ] Legacy cache v2 and fixed`query_per_tx`tests still pass.
- [ ] Inventory conflict rows are excluded, not double-counted.
- [ ] Scene/support/query physical ID sets are pairwise valid.
- [ ] K counts are exact per receiver/class/scene.
- [ ] MAXQ contains every eligible non-support physical ID.
- [ ] BAL4D query IDs are fixed across K and balanced by frozen cells.
- [ ] Predictor capsule contains no truth, role, quota or true batch class count.
- [ ] Independent summary reports sample/class/receiver/day/scene metrics.
- [ ] N607 audit and smoke use immutable output roots and short-lived SSH.
- [ ] Every commit is pushed and remote OID equals local HEAD.
