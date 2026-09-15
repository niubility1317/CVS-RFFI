"""Local experiment registration and evidence discovery. Never launches jobs or reads IQ/truth.

Canonical new records live beside the existing run report. Historical discovery records
are locators, not verified runs or performance results. Standard library only.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import contextmanager
import csv
import gzip
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import sys
import tempfile

SCHEMA = "experiment_record_v1"
KINDS = {"cvs", "comparison", "federated", "diagnostic", "data", "analysis"}
STATUSES = {"PLANNED", "LOCAL_VERIFIED", "LANDED", "QUEUED", "RUNNING",
            "SOURCE_TRAINING_COMPLETE_AWAITING_SOURCE_REVIEW", "TRAINING_COMPLETE",
            "PREDICTIONS_COMPLETE", "ARTIFACTS_COMPLETE", "ANALYZED", "FAILED",
            "STOPPED", "REPLACED", "PARTIAL", "UNKNOWN"}
SEED_ROLES = ("model", "split", "data", "augmentation", "support", "evaluation")
PRUNE = {".git", "__pycache__", ".pytest_cache", "node_modules", ".venv", "venv",
         "datasets", "dataset", "Dataset_WigSig", "Dataset_WiSig", "data", "wandb"}
SURFACES = ["automation_reports/CV-SincNet", "runs", "logs", "outputs",
            "remote_artifacts", "server_log_backups/N607", "local_artifacts", "analysis",
            "docs", "configs", "code/configs", "paper_reproduction", "baselines", "Fedbase",
            "automation_releases", "releases", "release_artifacts", "release_packages",
            "release_archives", "worklogs", "diagnostics"]
TEXT_SUFFIXES = {".json", ".yaml", ".yml", ".toml", ".md", ".txt", ".log", ".out", ".err", ".stdout", ".stderr", ".csv", ".jsonl"}
FACT_KEYS = {"seed", "seeds", "model_seed", "split_seed", "data_seed", "augmentation_seed",
             "support_seed", "eval_seed", "evaluation_seed", "train_seed", "random_seed",
             "dataset", "dataset_path", "data_path", "dataset_name", "train_ratio", "labeled_ratio",
             "source_receivers", "target_receivers", "source_days", "target_days", "receiver",
             "receivers", "rx", "tx", "tx_ids", "classes", "label_map", "split_id", "capsule_id",
             "phase2_data_status", "protocol_schema", "scenario", "scenarios", "sat_scenario",
             "k", "k_shot", "shot", "epochs", "fl_rounds", "fl_client_key", "batch_size",
             "labeled_batch", "unlabeled_batch", "steps_per_epoch", "lr", "learning_rate",
             "weight_decay", "optimizer", "initialization", "resume", "checkpoint", "checkpoint_path",
             "init_checkpoint", "selection_rule", "method", "phase1_method", "method_name",
             "run_id", "run_root", "output_root", "log_root", "code_root", "commit", "git_commit",
             "status", "verdict", "state", "family", "training_route", "model_key", "train_ratio"}
FAMILIES = ["DAOT", "FastTrust", "CORE90", "ECRS", "HCFDG", "NM FDU", "qknn", "D92",
            "CSIL", "MoPC", "RIEI", "DRIFT", "RadioNet", "SHOT", "MDD", "CSCNet",
            "FedFA", "FedRIEI", "FUCL", "RAFL", "FedCVS", "BEX", "CVS", "ADV3B02",
            "response", "Protonet", "receiver_agnostic", "cvcnn"]
LOG_CATALOGS = ["analysis/training_log_catalog_20260609_after_archive/log_catalog.csv",
                "analysis/20260604_federated_log_full_audit/catalog/log_catalog.csv"]


def read_metadata(path):
    data = Path(path).read_bytes()
    if data.startswith((b"\xff\xfe", b"\xfe\xff")):
        return data.decode("utf-16")
    return data.decode("utf-8-sig")


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def read_json(path):
    return json.loads(read_metadata(path))


def json_text(value):
    return json.dumps(value, ensure_ascii=False, indent=2) + "\n"


def write_text(path, text, exclusive=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if exclusive:
        with path.open("x", encoding="utf-8", newline="\n") as f:
            f.write(text)
    else:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="\n",
                                         dir=path.parent, delete=False, prefix=".registry-") as f:
            temp = Path(f.name)
            f.write(text)
        os.replace(temp, path)


def safe_id(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{2,159}", value):
        raise ValueError("run_id/row_id必须是3-160位ASCII字母、数字、下划线、点或短横线")
    if value.endswith(".") or value.split(".")[0].upper() in {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))}:
        raise ValueError("不允许Windows保留名称")
    return value


def record_dir(root, run_id):
    return Path(root) / "automation_reports/CV-SincNet" / safe_id(run_id)


@contextmanager
def lock(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    try:
        os.write(fd, str(os.getpid()).encode("ascii"))
        os.close(fd)
        yield
    finally:
        path.unlink()


def template():
    return {"schema": SCHEMA, "run_id": "20260916-phase1-method-dataset-s000001-r01",
            "group_id": "phase1-method-question", "display_name": "填写可读实验名称",
            "description": "填写研究问题、机制与对照差异", "kind": "cvs", "stage": "Phase1",
            "aliases": [], "tags": [], "comparison_group_id": None, "parent_run_ids": [],
            "replaces_run_id": None, "authorization": "填写本次用户授权范围或对话ID",
            "code": {"commit": None, "checkout": None, "environment": None, "cwd": None},
            "data": {"dataset": None, "version": None, "representation": None,
                     "contract_ref": None, "physical_ids_ref": None, "label_map_ref": None,
                     "source_receivers": [], "target_receivers": [], "source_days": [], "target_days": [],
                     "tx_sets_ref": None, "roles": {"L_s": None, "U_s": None, "V": None},
                     "train_ratio": None, "capsule_id": None, "split_id": None,
                     "validation_ref": None, "support_query_ref": None, "leo_config_ref": None},
            "permissions": {"regime": "source_only", "query_use": "test_only",
                            "external_method_exception": None, "claim_scope": None},
            "checkpoint": {"initialization": "scratch", "sources": [], "contract_check_ref": None,
                           "provenance_verdict": None, "selection_rule": None},
            "execution": {"host": None, "launch_owner": None, "gpu_policy": "max_two_training_runs_per_gpu",
                          "remote_run_root": None, "remote_log_root": None, "local_artifact_root": None,
                          "launch_command": None, "stop_rule": None},
            "expected_artifacts": [], "metrics_plan": {"metric_names": [], "dimensions": [],
                                                         "prediction_ref": None, "scorer_ref": None},
            "rows": [{"row_id": "method-s000001", "method": "填写方法和实际启用组件",
                      "purpose": "primary", "config_ref": None, "resolved_config_ref": None,
                      "data_overrides": {}, "seeds": {role: None for role in SEED_ROLES},
                      "seed_notes": "每个null解释为不适用或待补，不能默认为所有seed相同",
                      "k": None, "scenario": None, "optimizer": None, "lr": None,
                      "epochs": None, "fl_rounds": None, "budget_ref": None,
                      "output_root": None, "log_path": None, "command": None,
                      "expected_artifacts": []}], "notes": []}


def validate_spec(spec, ready=False):
    errors = []
    if spec.get("schema") != SCHEMA:
        errors.append(f"schema必须是{SCHEMA}")
    for key in ("run_id", "group_id", "display_name", "description", "stage", "authorization"):
        if not isinstance(spec.get(key), str) or not spec[key].strip():
            errors.append(f"缺少{key}")
    try:
        safe_id(spec.get("run_id"))
    except ValueError as e:
        errors.append(str(e))
    if spec.get("kind") not in KINDS:
        errors.append("kind无效")
    for key in ("code", "data", "permissions", "checkpoint", "execution", "metrics_plan"):
        if not isinstance(spec.get(key), dict):
            errors.append(f"缺少对象{key}")
    rows = spec.get("rows")
    if not isinstance(rows, list) or not rows:
        return errors + ["rows必须显式列出至少一行，不能从seed列表自动推断笛卡尔积"]
    ids, outputs = set(), set()
    for row in rows:
        if not isinstance(row, dict):
            errors.append("row必须是对象")
            continue
        rid = row.get("row_id")
        try:
            safe_id(rid)
        except ValueError as e:
            errors.append(str(e))
            continue
        if rid in ids:
            errors.append(f"重复row_id:{rid}")
        ids.add(rid)
        seeds = row.get("seeds", {})
        if not isinstance(seeds, dict):
            errors.append(f"{rid}:seeds必须是对象")
            continue
        for role in SEED_ROLES:
            if role not in seeds or (seeds[role] is not None and (type(seeds[role]) is not int or seeds[role] < 0)):
                errors.append(f"{rid}:seed {role}必须显式给出非负整数或null")
        if any(v is None for v in seeds.values()) and not row.get("seed_notes"):
            errors.append(f"{rid}:null seed需要seed_notes解释")
        output = row.get("output_root")
        if output:
            normalized = output.replace("\\", "/").rstrip("/")
            if normalized.casefold() in outputs:
                errors.append(f"重复output_root:{output}")
            outputs.add(normalized.casefold())
        if ready:
            for key in ("method", "config_ref", "output_root", "log_path", "command"):
                if not row.get(key):
                    errors.append(f"{rid}:启动登记缺少{key}")
    if ready:
        for section, keys in {"code": ("commit", "checkout", "environment", "cwd"),
                              "data": ("dataset", "contract_ref"),
                              "execution": ("host", "launch_owner", "launch_command", "stop_rule")}.items():
            for key in keys:
                if not spec.get(section, {}).get(key):
                    errors.append(f"启动登记缺少{section}.{key}")
        if not spec.get("expected_artifacts"):
            errors.append("启动登记缺少expected_artifacts")
    return errors


def report_text(spec):
    return f"""# {spec['display_name']}

- run_id：`{spec['run_id']}`
- group_id：`{spec['group_id']}`；类别：`{spec['kind']}`；阶段：`{spec['stage']}`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

{spec['description']}

## 数据、seed与模型来源

实际数据契约、权限例外、完整seed角色、checkpoint来源和选择规则见experiment.json。
逐行配置通过config_ref/resolved_config_ref定位；待补项必须在对应生命周期补齐。

## 执行与存储

命令、环境、CWD、commit、launch owner、输出和日志路径见experiment.json。
实际PID/GPU、读取时间、remote readback、失败或替代关系在此追加，并用record命令记录证据指针。

## 结果与覆盖

尚无结果。按预登记artifact逐项记录路径和缺项；保留每row与RX/day/TX/scene/K/seed的对应关系。
源域训练完成、预测完成、评分完成及协议有效性分别陈述。不得用总索引或旧状态证明当前运行。

## 交接

记录已完成、当前run/commit、证据路径、阻塞与下一步；恢复先查原run，不重复启动。
"""


def register(root, spec_path):
    spec = read_json(spec_path)
    errors = validate_spec(spec)
    if errors:
        raise ValueError("\n".join(errors))
    folder = record_dir(root, spec["run_id"])
    folder.parent.mkdir(parents=True, exist_ok=True)
    folder.mkdir(exist_ok=False)
    spec["registered_at"] = now()
    write_text(folder / "experiment.json", json_text(spec), exclusive=True)
    write_text(folder / "report.md", report_text(spec), exclusive=True)
    write_text(folder / "events.jsonl", json.dumps({"at": now(), "status": "PLANNED",
               "note": "已登记，未启动", "evidence": ["experiment.json"], "row_id": None}, ensure_ascii=False) + "\n", exclusive=True)
    return folder


def event(root, run_id, status, evidence, note, row_id=None):
    folder = record_dir(root, run_id)
    spec = read_json(folder / "experiment.json")
    if status not in STATUSES:
        raise ValueError("未知状态")
    if row_id and row_id not in {r["row_id"] for r in spec["rows"]}:
        raise ValueError("row_id不属于该run")
    if not evidence or not note:
        raise ValueError("状态更新需证据路径和说明；索引不验证远端事实")
    with lock(folder / ".registry.lock"):
        with (folder / "events.jsonl").open("a", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps({"at": now(), "status": status, "note": note,
                               "evidence": evidence, "row_id": row_id}, ensure_ascii=False) + "\n")


def category(path):
    n = path.name.lower()
    if path.suffix.lower() in {".pth", ".pt", ".ckpt", ".safetensors"}:
        return "checkpoint"
    if n.endswith((".tar", ".tar.gz", ".zip", ".7z", ".tgz", ".gz")):
        return "archive"
    if "predict" in n or "truth" in n:
        return "prediction_locator"
    if path.suffix.lower() in {".md", ".docx", ".pdf", ".html"}:
        return "report"
    if any(x in n for x in ("config", "manifest", "matrix", "contract", "plan", "split")) and path.suffix.lower() in TEXT_SUFFIXES:
        return "config"
    if any(x in n for x in ("status", "readback", "inventory", "acceptance", "verification", "delivery", "summary", "result", "state")) and path.suffix.lower() in TEXT_SUFFIXES:
        return "state_or_result"
    if path.suffix.lower() in {".log", ".out", ".err", ".stdout", ".stderr", ".jsonl", ".csv"}:
        return "log_or_metrics"
    if path.suffix.lower() in {".sh", ".ps1", ".bat", ".cmd"}:
        return "launcher"
    if path.suffix.lower() in {".json", ".yaml", ".yml", ".toml"}:
        return "config"
    return "other"


def facts(obj, scope="$", inherited_row=None):
    if isinstance(obj, dict):
        row = obj.get("row_id", obj.get("id", inherited_row))
        for key, value in obj.items():
            pointer = scope + "." + key
            if key.lower() in FACT_KEYS and value is not None:
                # Large physical-ID/class lists stay at the source; never copy sample-level truth.
                if isinstance(value, (str, int, float, bool)) or (isinstance(value, list) and len(value) <= 32 and all(isinstance(x, (str, int, float, bool)) for x in value)):
                    if len(json.dumps(value, ensure_ascii=False)) <= 1200:
                        yield {"field": key, "value": value, "scope": pointer, "row": row}
            if isinstance(value, (dict, list)) and key.lower() not in {"predictions", "truth", "y_true", "y_pred", "physical_ids", "role_ids", "samples", "confusion_matrix"}:
                yield from facts(value, pointer, row)
    elif isinstance(obj, list):
        for i, value in enumerate(obj):
            if isinstance(value, (dict, list)):
                yield from facts(value, f"{scope}[{i}]", inherited_row)


def scope_list(root):
    result = [(root / s, s, "workspace") for s in SURFACES]
    pub = root / "github_publish"
    if pub.exists():
        for repo in sorted(pub.iterdir()):
            if repo.is_dir() and (repo / ".git").exists():
                for s in SURFACES + ["artifacts", "ops"]:
                    result.append((repo / s, (repo / s).relative_to(root).as_posix(), "checkout"))
    return result


def walked(base, errors):
    def failed(exc):
        errors.append({"path": str(exc.filename), "error": str(exc)})
    for folder, dirs, names in os.walk(base, onerror=failed, followlinks=False):
        dirs[:] = sorted(d for d in dirs if d not in PRUNE and not d.startswith((".pytest", ".tmp")) and not (Path(folder) / d).is_symlink())
        for n in sorted(names):
            p = Path(folder) / n
            if not p.is_symlink():
                yield p


def group_path(base, path):
    parts = path.relative_to(base).parts
    if len(parts) == 1:
        return path
    # Log archives retain route/family folders; a backup date alone is too coarse.
    depth = 1
    if base.name in {"N607", "logs", "baselines", "paper_reproduction", "Fedbase"}:
        depth = min(len(parts) - 1, 3 if base.name == "N607" else 2)
    return base.joinpath(*parts[:depth])


def build(root, managed_only=False):
    root = Path(root).resolve()
    dest = root / "experiment_registry"
    dest.mkdir(parents=True, exist_ok=True)
    with lock(dest / ".build.lock"):
        return _build(root, dest, managed_only)


def _build(root, dest, managed_only):
    old = {}
    catalog = dest / "catalog.jsonl"
    if catalog.exists():
        old = {r["id"]: r for r in (json.loads(line) for line in catalog.read_text(encoding="utf-8").splitlines())}
    records, errors, coverage = {}, [], []
    artifact_lines, fact_lines = defaultdict(list), defaultdict(list)
    managed = root / "automation_reports/CV-SincNet"
    for path in sorted(managed.glob("*/experiment.json")):
        try:
            spec = read_json(path)
            if spec.get("schema") != SCHEMA:
                continue
            issues = validate_spec(spec)
            if issues:
                raise ValueError("; ".join(issues))
            event_path = path.parent / "events.jsonl"
            events = [json.loads(line) for line in event_path.read_text(encoding="utf-8").splitlines() if line.strip()] if event_path.exists() else []
            global_events = [e for e in events if not e.get("row_id")]
            row_events = {e["row_id"]: e for e in events if e.get("row_id")}
            status = global_events[-1]["status"] if global_events else "UNKNOWN"
            spec_rel = path.relative_to(root).as_posix()
            record = {"id": spec["run_id"], "record_type": "managed_run", "name": spec["display_name"],
                      "description": spec["description"], "kind": spec["kind"], "stage": spec["stage"],
                      "group_id": spec["group_id"], "aliases": spec.get("aliases", []), "tags": spec.get("tags", []),
                      "methods": sorted({r["method"] for r in spec["rows"]}), "status": status,
                      "seeds": sorted({s for row in spec["rows"] for s in row["seeds"].values() if type(s) is int}),
                      "data": spec["data"], "rows": spec["rows"], "spec": spec_rel,
                      "row_states": {r["row_id"]: row_events.get(r["row_id"], {"status": "UNKNOWN", "note": "尚无行级状态证据"}) for r in spec["rows"]},
                      "events": event_path.relative_to(root).as_posix(),
                      "report": (path.parent / "report.md").relative_to(root).as_posix(),
                      "path": path.parent.relative_to(root).as_posix(), "evidence_state": "REGISTERED_NOT_LIVE_VERIFIED",
                      "last_event": global_events[-1] if global_events else None, "indexed_at": now()}
            records[record["id"]] = record
        except (ValueError, OSError, KeyError, TypeError) as exc:
            errors.append({"path": str(path), "error": str(exc)})
    if managed_only:
        records.update({k: v for k, v in old.items() if v["record_type"] != "managed_run"})
        previous_coverage = read_json(dest / "coverage.json") if (dest / "coverage.json").exists() else {}
        coverage = previous_coverage.get("surfaces", [])
        errors.extend(previous_coverage.get("errors", []))
    else:
        managed_paths = {r["path"] for r in records.values() if r["record_type"] == "managed_run"}
        for base, label, origin in scope_list(root):
            count, selected, byte_count = 0, 0, 0
            if not base.exists():
                coverage.append({"path": label, "exists": False, "files": 0})
                continue
            for path in walked(base, errors):
                count += 1
                cat = category(path)
                if cat == "other":
                    continue
                selected += 1
                try:
                    st = path.stat()
                    byte_count += st.st_size
                except OSError as exc:
                    errors.append({"path": str(path), "error": str(exc)})
                    continue
                rel = path.relative_to(root).as_posix()
                group = group_path(base, path)
                group_rel = group.relative_to(root).as_posix()
                if group_rel in managed_paths:
                    continue
                key = "legacy:" + group_rel
                rec = records.setdefault(key, {"id": key, "record_type": "legacy_evidence_group",
                    "name": group.stem if group.is_file() else group.name, "description": "历史证据定位组；不是独立实验计数",
                    "kind": "unclassified", "stage": "unknown", "group_id": None, "aliases": [], "tags": [],
                    "methods": [], "status": "HISTORICAL_UNVERIFIED", "seeds": [], "path": group_rel,
                    "report": None, "origin": origin, "artifact_counts": {}, "evidence_state": "LOCAL_PATHS_OBSERVED_REMOTE_UNCHECKED",
                    "latest_file_mtime": 0, "indexed_at": now(), "fact_count": 0,
                    "locator_samples": {}, "reported_statuses": [], "title": None})
                rec["artifact_counts"][cat] = rec["artifact_counts"].get(cat, 0) + 1
                rec["latest_file_mtime"] = max(rec["latest_file_mtime"], st.st_mtime)
                samples = rec["locator_samples"].setdefault(cat, [])
                if len(samples) < 4:
                    samples.append(rel)
                if cat == "report" and (rec["report"] is None or
                    (path.name.lower() == "report.md", -len(path.parts)) >
                    (Path(rec["report"]).name.lower() == "report.md", -len((root / rec["report"]).parts))):
                    rec["report"] = rel
                artifact_lines[key].append(json.dumps({"record_id": key, "path": rel, "category": cat,
                    "bytes": st.st_size, "mtime": st.st_mtime}, ensure_ascii=False))
                content = None
                # Read small metadata/report files in full; logs, metric rows and model/data bytes are locator-only.
                readable = cat in {"config", "state_or_result", "report"} and path.suffix.lower() in {".json", ".md", ".yaml", ".yml", ".toml"}
                if readable and st.st_size <= 2_000_000:
                    try:
                        content = read_metadata(path)
                    except (OSError, UnicodeError) as exc:
                        errors.append({"path": rel, "error": str(exc), "action": "locator_only"})
                if content is not None:
                    extracted = []
                    if path.suffix.lower() == ".json":
                        try:
                            extracted = list(facts(json.loads(content)))
                        except (ValueError, RecursionError) as exc:
                            errors.append({"path": rel, "error": str(exc), "action": "locator_only"})
                    else:
                        for number, line in enumerate(content.splitlines(), 1):
                            if line.startswith("# ") and not rec["title"]:
                                rec["title"] = line[2:][:200]
                            # Only explicit key/value metadata. Prose mentions are never promoted to a runtime state.
                            m = re.match(r'^\s*[\-*>`\s]*[\"\']?([A-Za-z_][\w]*)[\"\']?\s*[:=：]\s*(.{1,1200})$', line)
                            if m and m[1].lower() in FACT_KEYS:
                                extracted.append({"field": m[1], "value": m[2].strip(), "scope": f"line:{number}", "row": None})
                    for fact in extracted:
                        fact.update({"record_id": key, "source": rel})
                        fact_lines[key].append(json.dumps(fact, ensure_ascii=False))
                        rec["fact_count"] += 1
                        field, value = fact["field"].lower(), fact["value"]
                        if "seed" in field:
                            values = value if isinstance(value, list) else [value]
                            for v in values:
                                if isinstance(v, str) and v.isdigit():
                                    v = int(v)
                                if type(v) is int and v not in rec["seeds"]:
                                    rec["seeds"].append(v)
                        if field in {"dataset", "dataset_name", "dataset_path", "data_path", "scenario", "split_id", "capsule_id"} and isinstance(value, (str, int)):
                            hints = rec.setdefault("metadata_mentions", {}).setdefault(field, [])
                            if value not in hints and len(hints) < 32:
                                hints.append(value)
                        if field in {"status", "state", "verdict"} and isinstance(value, str) and len(value) <= 100:
                            if value not in rec["reported_statuses"] and len(rec["reported_statuses"]) < 32:
                                rec["reported_statuses"].append(value)
                        if field in {"method", "phase1_method", "method_name", "family", "training_route"} and isinstance(value, str) and len(value) <= 100:
                            if value not in rec["methods"]:
                                rec["methods"].append(value)
                # Search tags are hints from the path, not authoritative dataset/method/state assignments.
                lower = group_rel.lower()
                rec["tags"] = sorted({f.lower() for f in FAMILIES if f.lower().replace(" ", "") in lower} |
                    {t for t in ("phase1", "phase2", "stage2", "phase3", "source", "failed", "stop", "repair", "dryrun", "smoke", "diagnostic", "comparison", "federated") if t in lower})
            coverage.append({"path": label, "exists": True, "files": count, "locator_files": selected, "locator_bytes": byte_count})
            print(f"indexed {label}: {selected}/{count} files", file=sys.stderr, flush=True)
        import_log_catalogs(root, records, fact_lines, errors, coverage)
        for key, previous in old.items():
            if key not in records and previous["record_type"] != "managed_run":
                previous["evidence_state"] = "NOT_SEEN_IN_LATEST_SCAN_USE_PREVIOUS_LOCATOR"
                records[key] = previous
        used_keys = {r.get("detail_key") for r in old.values() if r.get("detail_key")}
        next_key = max([int(x) for x in used_keys] + [0]) + 1
        for key in sorted(set(artifact_lines) | set(fact_lines)):
            detail_key = old.get(key, {}).get("detail_key")
            if not detail_key:
                detail_key = f"{next_key:06d}"
                next_key += 1
            rec = records[key]
            rec["detail_key"] = detail_key
            for section, items in (("artifacts", artifact_lines.get(key, [])), ("facts", fact_lines.get(key, []))):
                rel = f"details/{detail_key}.{section}.jsonl.gz"
                rec[section + "_file"] = rel
                target = dest / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                # Only derived locators/metadata, never compression or modification of original artifacts.
                raw = ("\n".join(items) + ("\n" if items else "")).encode("utf-8")
                target.write_bytes(gzip.compress(raw, compresslevel=6, mtime=0))
        print(f"wrote {len(artifact_lines)} artifact groups and {len(fact_lines)} metadata groups", file=sys.stderr, flush=True)
    snapshots = import_remote_snapshots(root, records, errors)
    for rec in records.values():
        report = rec.get("report")
        if report and report.endswith(".md"):
            try:
                with (root / report).open(encoding="utf-8-sig") as f:
                    for _, line in zip(range(12), f):
                        if line.startswith("# "):
                            rec["title"] = line[2:].strip()[:200]
                            break
            except (OSError, UnicodeError):
                pass
    ordered = sorted(records.values(), key=lambda r: (r["record_type"] != "managed_run", -r.get("latest_file_mtime", 0), r["id"]))
    write_text(catalog, "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in ordered))
    # CSV remains small enough to open without parsing per-artifact JSONL.
    import io
    csv_buffer = io.StringIO(newline="")
    columns = ["id", "record_type", "name", "title", "kind", "stage", "status", "methods", "tags", "seeds", "report", "path", "evidence_state", "fact_count"]
    writer = csv.DictWriter(csv_buffer, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    for rec in ordered:
        writer.writerow({k: json.dumps(rec[k], ensure_ascii=False) if isinstance(rec.get(k), (list, dict)) else rec.get(k, "") for k in columns})
    write_text(dest / "catalog.csv", csv_buffer.getvalue())
    coverage_doc = {"schema": "experiment_registry_coverage_v1", "indexed_at": now(), "workspace": str(root),
        "mode": "managed_only" if managed_only else "local_history", "record_counts": dict(Counter(r["record_type"] for r in ordered)),
        "surfaces": coverage, "remote_snapshots": snapshots, "errors": errors,
        "boundaries": ["本地扫描工作区和github_publish下已存在检出；远端来自单独只读目录快照，不代表实时进程或深层内容已核实。",
                       "历史组不是独立实验；同名镜像保留各自路径，不凭名称合并。",
                       "未解压归档；日志、指标行、预测和权重只登记路径；不读取IQ或truth。",
                       "仅完整读取不超过2MB的UTF-8/带BOM UTF-16元数据与Markdown；大文件与编码错误仍保留locator。",
                       "reported_statuses是原文声明，可能包含历史过程或嵌套row，不能作为当前状态。",
                       "目录不存在、旧索引记录未重见、结构化读取失败都在此披露，不宣称已完成全量性能审计。"]}
    write_text(dest / "coverage.json", json_text(coverage_doc))
    render_index(dest, ordered, coverage_doc)
    return coverage_doc


def import_remote_snapshots(root, records, errors):
    summary = []
    for path in sorted((root / "experiment_registry/snapshots").glob("*.json")):
        try:
            snapshot = read_json(path)
            if snapshot.get("schema") != "n607_experiment_directory_inventory_v1":
                continue
            rel = path.relative_to(root).as_posix()
            summary.append({"path": rel, "observed_at": snapshot["observed_at"],
                            "records": len(snapshot["records"]), "depth": snapshot["depth"],
                            "errors": snapshot["errors"]})
            for row in snapshot["records"]:
                key = "remote:N607:" + row["path"]
                if records.get(key, {}).get("observed_at", "") > snapshot["observed_at"]:
                    continue
                lower = row["path"].lower()
                tags = sorted({f.lower() for f in FAMILIES if f.lower().replace(" ", "") in lower})
                records[key] = {"id": key, "record_type": "remote_directory_locator", "name": row["name"],
                    "description": "N607只读目录定位快照；目录存在不等于已启动或完成，也不是独立实验计数",
                    "kind": "unclassified", "stage": "unknown", "status": "UNKNOWN", "tags": tags,
                    "seeds": [], "methods": [], "path": row["path"], "report": rel,
                    "observed_at": snapshot["observed_at"], "host": snapshot["host"],
                    "surface": row["surface"], "children": row["children"], "entry_type": row["entry_type"],
                    "evidence_state": "REMOTE_DIRECTORY_OBSERVED_STATUS_UNKNOWN", "indexed_at": now()}
        except (ValueError, OSError, KeyError, TypeError) as exc:
            errors.append({"path": str(path), "error": str(exc)})
    return summary


def import_log_catalogs(root, records, fact_lines, errors, coverage):
    """Import named existing log inventories without reinterpreting scores or dates."""
    for rel in LOG_CATALOGS:
        path = root / rel
        if not path.exists():
            continue
        try:
            import io
            reader = csv.DictReader(io.StringIO(read_metadata(path)))
            count = 0
            for count, row in enumerate(reader, 1):
                key = f"legacy-log:{rel}#row-{count}"
                name = row.get("model_key") or row.get("run_name") or row.get("source_path") or f"row-{count}"
                seed = row.get("seed", "")
                rec = {"id": key, "record_type": "legacy_log_record", "name": name,
                    "description": "既有日志目录中的一行，可能是同一实验的重复日志或备份；不是独立实验计数",
                    "kind": "unclassified", "stage": "unknown", "status": "HISTORICAL_UNVERIFIED",
                    "seeds": [int(seed)] if seed.isdigit() else [], "path": rel, "report": rel,
                    "source_path_reported": row.get("source_path"), "run_dir_reported": row.get("run_dir"),
                    "methods": [row[k] for k in ("family", "training_route") if row.get(k)],
                    "tags": [row[k].lower() for k in ("family", "training_route") if row.get(k)],
                    "reported_statuses": [row.get("status")] if row.get("status") else [],
                    "data": {k: row[k] for k in ("dataset", "train_ratio", "client_key", "domain") if row.get(k)},
                    "evidence_state": "IMPORTED_LOG_CATALOG_SOURCE_PATH_NOT_RELOCATED",
                    "indexed_at": now(), "source_row": count + 1, "fact_count": 0}
                for field in ("seed", "family", "training_route", "dataset", "train_ratio", "client_key",
                              "domain", "config", "cmd", "source_path", "run_dir", "status", "model_key"):
                    if row.get(field):
                        fact_lines[key].append(json.dumps({"record_id": key, "field": field,
                            "value": row[field], "source": rel, "scope": f"csv-row:{count + 1}", "row": name}, ensure_ascii=False))
                        rec["fact_count"] += 1
                records[key] = rec
            coverage.append({"path": rel, "exists": True, "files": 0, "imported_log_records": count})
        except (ValueError, OSError, UnicodeError, csv.Error) as exc:
            errors.append({"path": rel, "error": str(exc)})


def render_index(dest, records, coverage):
    tags = Counter(t for r in records for t in r.get("tags", []))
    text = ["# 实验总索引", "", f"更新：{coverage['indexed_at']}", "",
            "先按方法/问题检索，再用run ID读取精确配置与证据。历史组数量不等于独立实验数量。", "",
            "- [管理规范与常用命令](../docs/EXPERIMENT_MANAGEMENT.md)",
            "- [完整目录CSV](catalog.csv) · [结构化目录](catalog.jsonl) · [覆盖与缺项](coverage.json)",
            "- 通过show的`--section artifacts/facts`读取该条目的路径或配置出处；细目按记录压缩保存于details/，不扫描全库。", "",
            "## 登记规模", "", "|记录类型|数量|", "|---|---:|"]
    text.extend(f"|{k}|{v}|" for k, v in coverage["record_counts"].items())
    text += ["", "历史状态统一为HISTORICAL_UNVERIFIED；RUNNING等原文声明只供查证，不能证明此刻仍在运行。",
             "", "## 按方法与用途查找", "", "|路径标签（定位提示）|证据组数|", "|---|---:|"]
    for tag, count in tags.most_common():
        filename = re.sub(r"[^a-zA-Z0-9_-]", "_", tag) + ".md"
        text.append(f"|[{tag}](by_method/{filename})|{count}|")
        page = [f"# {tag}实验与历史证据", "", "[返回总索引](../README.md)", "",
                "路径标签/旧目录字段仅用于查找；历史记录和备份不等于独立实验，状态未实时核实。", "",
                "|名称|记录类型|证据入口|", "|---|---|---|"]
        for rec in records:
            if tag in rec.get("tags", []):
                name = (rec.get("title") or rec["name"]).replace("|", "/").replace("\n", " ")
                path = (rec.get("report") or rec["path"]).replace(" ", "%20")
                page.append(f"|{name}|{rec['record_type']}|[打开](../../{path})|")
        write_text(dest / "by_method" / filename, "\n".join(page) + "\n")
    text += ["", "## 最近记录入口", "", "|名称|类型|报告/原目录|", "|---|---|---|"]
    for rec in records[:120]:
        path = (rec.get("report") or rec["path"]).replace(" ", "%20")
        name = (rec.get("title") or rec["name"]).replace("|", "/").replace("\n", " ")
        text.append(f"|{name}|{rec['record_type']}|[打开](../{path})|")
    write_text(dest / "README.md", "\n".join(text) + "\n")


def catalog_rows(root):
    path = Path(root) / "experiment_registry/catalog.jsonl"
    if not path.exists():
        raise ValueError("尚无索引；先运行build")
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def search(root, query="", seed=None, tag=None, kind=None, status=None, record_type=None, limit=20):
    result = []
    terms = query.casefold().split()
    for rec in catalog_rows(root):
        if seed is not None and seed not in rec.get("seeds", []):
            continue
        if tag and tag.casefold() not in rec.get("tags", []):
            continue
        if kind and rec["kind"] != kind:
            continue
        if status and rec["status"] != status:
            continue
        if record_type and rec["record_type"] != record_type:
            continue
        hay = json.dumps(rec, ensure_ascii=False).casefold()
        if all(t in hay for t in terms):
            result.append(rec)
    return result[:limit], len(result)


def show(root, identifier, section="record", limit=40, field=None, row=None, category_filter=None):
    candidates = [r for r in catalog_rows(root) if r["id"] == identifier or r["name"] == identifier or identifier in r.get("aliases", [])]
    if len(candidates) != 1:
        raise ValueError(f"匹配{len(candidates)}项；请用search返回的完整id")
    rec = candidates[0]
    if rec["record_type"] == "remote_directory_locator":
        return rec
    if section == "record":
        if rec["record_type"] == "managed_run":
            return {"record": rec, "spec": read_json(Path(root) / rec["spec"])}
        return rec
    filename = rec.get(section + "_file", "facts.jsonl" if section == "facts" else "artifacts.jsonl")
    found = []
    has_more = False
    path = Path(root) / "experiment_registry" / filename
    if path.exists():
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rt", encoding="utf-8") as f:
            for line in f:
                value = json.loads(line)
                if value["record_id"] == rec["id"] and (field is None or value.get("field") == field) and (row is None or value.get("row") == row) and (category_filter is None or value.get("category") == category_filter):
                    found.append(value)
                    if len(found) > limit:
                        has_more = True
                        break
    return {"id": rec["id"], "total": None if has_more else len(found), "has_more": has_more,
            "detail_file": str(path), "shown": found[:limit]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("template"); p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("new"); p.add_argument("--spec", type=Path, required=True)
    p = sub.add_parser("validate"); p.add_argument("spec", type=Path); p.add_argument("--launch-ready", action="store_true")
    p = sub.add_parser("build"); p.add_argument("--managed-only", action="store_true")
    p = sub.add_parser("record"); p.add_argument("run_id"); p.add_argument("--status", choices=sorted(STATUSES), required=True)
    p.add_argument("--evidence", action="append", required=True); p.add_argument("--note", required=True); p.add_argument("--row-id")
    p = sub.add_parser("search"); p.add_argument("query", nargs="?", default=""); p.add_argument("--seed", type=int)
    p.add_argument("--tag"); p.add_argument("--kind", choices=sorted(KINDS)); p.add_argument("--status")
    p.add_argument("--record-type", choices=["managed_run", "legacy_evidence_group", "legacy_log_record", "remote_directory_locator"]); p.add_argument("--limit", type=int, default=20)
    p = sub.add_parser("show"); p.add_argument("id"); p.add_argument("--section", choices=["record", "facts", "artifacts"], default="record"); p.add_argument("--limit", type=int, default=40)
    p.add_argument("--field"); p.add_argument("--row"); p.add_argument("--category")
    args = parser.parse_args(argv)
    try:
        if args.command == "template":
            write_text(args.output, json_text(template()), exclusive=True)
            print(args.output)
        elif args.command == "new":
            print(register(args.root, args.spec))
            build(args.root, managed_only=True)
        elif args.command == "validate":
            issues = validate_spec(read_json(args.spec), args.launch_ready)
            print(json_text({"status": "INVALID" if issues else "VALID", "errors": issues,
                             "scope": "登记字段检查，不是实验许可或科学有效性证明"}), end="")
            return int(bool(issues))
        elif args.command == "build":
            result = build(args.root, args.managed_only)
            print(json_text({"counts": result["record_counts"], "errors": len(result["errors"])}), end="")
        elif args.command == "record":
            event(args.root, args.run_id, args.status, args.evidence, args.note, args.row_id)
            build(args.root, managed_only=True)
        elif args.command == "search":
            rows, total = search(args.root, args.query, args.seed, args.tag, args.kind, args.status, args.record_type, args.limit)
            print(f"matched={total} shown={len(rows)}")
            for r in rows:
                print(json.dumps({k: r.get(k) for k in ("id", "name", "title", "record_type", "status", "seeds", "report", "path")}, ensure_ascii=False))
        elif args.command == "show":
            print(json_text(show(args.root, args.id, args.section, args.limit, args.field, args.row, args.category)), end="")
    except (ValueError, OSError, KeyError, TypeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
