from __future__ import annotations

import ast
import io
import json
import os
import stat
from contextlib import redirect_stdout
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Callable

import pytest

from tools.project_governance.collect_n607 import (
    APPROVED_BRIDGE_HOST,
    APPROVED_N607_HOST,
    BRIDGE_ROUTE,
    DIRECT_ROUTE,
    CommandResult,
    ConnectionCheck,
    ConnectionEvidence,
    DEFAULT_SSH_CONFIG,
    N607Collector,
    RemoteOutcome,
    build_remote_payload,
)
from tools.project_governance.config import CarrierSurface, DiscoveryConfig, LocationConfig
from tools.project_governance.models import Location
from tools.project_governance.paths import normalize_relative_path, stable_asset_id


@dataclass
class FakeRunner:
    results: list[CommandResult]
    connection_results: list[ConnectionCheck | tuple[ConnectionEvidence, ...]]
    stream_output: bool = True
    calls: list[dict[str, object]] = field(default_factory=list)
    connection_calls: list[dict[str, object]] = field(default_factory=list)

    def run(
        self,
        command: tuple[str, ...],
        *,
        input_text: str | None,
        timeout_seconds: int,
        label: str,
        stdout_line_handler: Callable[[str | bytes], None] | None = None,
    ) -> CommandResult:
        self.calls.append(
            {
                "command": command,
                "input_text": input_text,
                "timeout_seconds": timeout_seconds,
                "label": label,
                "streamed": stdout_line_handler is not None,
            }
        )
        if not self.results:
            raise AssertionError("unexpected command execution")
        result = self.results.pop(0)
        assert result.command == command
        if stdout_line_handler is not None and self.stream_output:
            for line in result.stdout_lines:
                stdout_line_handler(line)
        return result

    def check_connections(
        self, *, attempt_pids: tuple[int, ...], endpoints: tuple[str, ...]
    ) -> ConnectionCheck | tuple[ConnectionEvidence, ...]:
        self.connection_calls.append(
            {"attempt_pids": attempt_pids, "endpoints": endpoints}
        )
        if not self.connection_results:
            raise AssertionError("unexpected disconnect check")
        return self.connection_results.pop(0)


def _config(root: str = "/srv/CV-SincNet") -> LocationConfig:
    return LocationConfig(
        location=Location.N607,
        root_id="N607_CVS_SINCNET",
        root=root,
        carrier_surfaces=(
            CarrierSurface("runs", "NOT_PRESENT"),
            CarrierSurface("logs", "NOT_PRESENT"),
        ),
    )


def _discovery() -> DiscoveryConfig:
    return DiscoveryConfig(
        control_evidence_max_depth=2,
        hash_max_bytes=1024,
        text_read_max_bytes=2048,
    )


def _preflight_command(script: Path = Path("C:/repo/tools/n607_ssh_preflight.ps1")) -> tuple[str, ...]:
    return (
        "powershell.exe",
        "-NoLogo",
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        str(script),
    )


def _direct_command(
    config: Path = Path("C:/repo/tools/n607_ssh_config"),
) -> tuple[str, ...]:
    return (
        "ssh",
        "-F",
        str(config),
        "-T",
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=10",
        "N607",
        "python3",
        "-",
    )


def _bridge_command() -> tuple[str, ...]:
    proxy = (
        "ProxyCommand=ssh -i C:/Users/lh594/.ssh/id_ed25519_lab_bridge_172_31_105_18 "
        "-o BatchMode=yes -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new "
        "-W %h:%p administrator@172.31.105.18"
    )
    return (
        "ssh",
        "-i",
        "C:/Users/lh594/.ssh/id_ed25519_n607",
        "-o",
        "BatchMode=yes",
        "-o",
        "IdentitiesOnly=yes",
        "-o",
        "ConnectTimeout=10",
        "-o",
        "StrictHostKeyChecking=accept-new",
        "-o",
        proxy,
        "szu2070436088@172.31.111.215",
        "python3",
        "-",
    )


def _result(
    command: tuple[str, ...],
    *,
    pid: int,
    stdout_lines: tuple[str, ...] = (),
    stderr_tail: str = "",
    returncode: int | None = 0,
    child_exited: bool = True,
    proxy_pids: tuple[int, ...] = (),
    proxy_children_exited: bool = True,
    timed_out: bool = False,
) -> CommandResult:
    return CommandResult(
        command=command,
        child_pid=pid,
        proxy_child_pids=proxy_pids,
        returncode=returncode,
        child_exited=child_exited,
        proxy_children_exited=proxy_children_exited,
        stdout_lines=stdout_lines,
        stderr_tail=stderr_tail,
        timed_out=timed_out,
    )


def _preflight_ok(command: tuple[str, ...] | None = None) -> CommandResult:
    command = command or _preflight_command()
    return _result(
        command,
        pid=101,
        stdout_lines=(
            "Config OK: N607 is direct. user szu2070436088; hostname 172.31.111.215",
            "SSH config: C:/repo/tools/n607_ssh_config",
            "Identity file OK: C:/Users/lh594/.ssh/id_ed25519_n607_codexsandboxoffline",
            "Identity file OK: C:/Users/lh594/.ssh/id_ed25519_n607",
            "user=szu2070436088",
            "project_root=/home/szu2070436088/2510044040/CV-SincNet",
            "Preflight OK: use ssh N607",
        ),
        proxy_pids=(102,),
    )


def _ndjson(*, scan_id: str = "SCAN-1", active: bool = False) -> tuple[str, ...]:
    asset_id = "asset:N607:N607_CVS_SINCNET:runs/demo/report.md"
    lines = [
        json.dumps(
            {
                "schema_version": 1,
                "scan_id": scan_id,
                "record_type": "SERVER_INFO",
                "hostname": "n607",
                "server_time_utc": "2026-08-17T00:00:00Z",
                "root": "/home/szu2070436088/2510044040/CV-SincNet",
            }
        ),
        json.dumps(
            {
                "schema_version": 1,
                "scan_id": scan_id,
                "record_type": "ASSET",
                "asset_id": asset_id,
                "location": "N607",
                "root_id": "N607_CVS_SINCNET",
                "relative_path": "runs/demo/report.md",
                "display_name": "report.md",
                "escaped_name": "report.md",
                "asset_kind": "file",
                "size_bytes": 12,
                "mtime_utc": "2026-08-17T00:00:00Z",
                "access_status": "OK",
                "hash_status": "SHA256",
                "sha256": "0" * 64,
                "evidence_role": "CONTROL_EVIDENCE",
            }
        ),
        json.dumps(
            {
                "schema_version": 1,
                "scan_id": scan_id,
                "record_type": "SCOPE",
                "location": "N607",
                "root_id": "N607_CVS_SINCNET",
                "relative_path": "",
                "status": "VERIFIED",
                "asset_ids": [asset_id],
            }
        ),
        json.dumps(
            {
                "schema_version": 1,
                "scan_id": scan_id,
                "record_type": "SCOPE",
                "location": "N607",
                "root_id": "N607_CVS_SINCNET",
                "relative_path": "runs",
                "status": "VERIFIED",
                "asset_ids": [],
            }
        ),
        json.dumps(
            {
                "schema_version": 1,
                "scan_id": scan_id,
                "record_type": "SCOPE",
                "location": "N607",
                "root_id": "N607_CVS_SINCNET",
                "relative_path": "logs",
                "status": "NOT_PRESENT",
                "asset_ids": [],
            }
        ),
    ]
    if active:
        lines.append(
            json.dumps(
                {
                    "schema_version": 1,
                    "scan_id": scan_id,
                    "record_type": "PROCESS",
                    "pid": 5151,
                    "ppid": 5000,
                    "cwd": "/home/szu2070436088/2510044040/CV-SincNet",
                    "cmdline": "python train.py",
                    "training_like": True,
                }
            )
        )
    lines.append(
        json.dumps(
            {
                "schema_version": 1,
                "scan_id": scan_id,
                "record_type": "COLLECTION_COMPLETE",
                "record_count": len(lines),
                "scan_error_count": 0,
            }
        )
    )
    return tuple(lines)


def _collector(runner: FakeRunner, *, scan_id: str = "SCAN-1") -> N607Collector:
    return N607Collector(
        _config("/home/szu2070436088/2510044040/CV-SincNet"),
        _discovery(),
        scan_id=scan_id,
        runner=runner,
        preflight_script=Path("C:/repo/tools/n607_ssh_preflight.ps1"),
    )


def test_remote_payload_is_ast_safe_and_imports_only_read_only_stdlib() -> None:
    payload = build_remote_payload(_config(), _discovery(), scan_id="SCAN-AST")
    tree = ast.parse(payload)
    imports: set[str] = set()
    forbidden_attributes = {
        "write_text",
        "write_bytes",
        "unlink",
        "remove",
        "rmdir",
        "rmtree",
        "mkdir",
        "makedirs",
        "rename",
        "replace",
        "chmod",
        "chown",
        "kill",
        "system",
        "Popen",
        "run",
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.add(node.module or "")
        elif isinstance(node, ast.Attribute):
            assert node.attr not in forbidden_attributes
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "open":
            mode = None
            if len(node.args) >= 2 and isinstance(node.args[1], ast.Constant):
                mode = node.args[1].value
            for keyword in node.keywords:
                if keyword.arg == "mode" and isinstance(keyword.value, ast.Constant):
                    mode = keyword.value.value
            assert mode in {"r", "rt", "rb"}
    assert imports == {
        "datetime",
        "hashlib",
        "json",
        "os",
        "socket",
        "stat",
        "unicodedata",
        "urllib.parse",
    }
    assert "subprocess" not in payload
    assert "os.lstat(ROOT)" in payload


def test_remote_payload_executes_locally_as_ndjson_without_mutating_source(tmp_path: Path) -> None:
    root = tmp_path / "remote-root"
    (root / "runs" / "demo").mkdir(parents=True)
    (root / "logs").mkdir()
    report = root / "runs" / "demo" / "report.md"
    report.write_text("run_id: demo\nterminal: true\n", encoding="utf-8")
    source = root / "runs" / "demo" / "worker.py"
    source.write_text("TOKEN = 'not evidence text'\n", encoding="utf-8")
    receipt = root / "runs" / "demo" / "scan_receipt.json"
    receipt.write_text('{"run_id":"utf16-demo"}\n', encoding="utf-16")
    before = {path.relative_to(root).as_posix() for path in root.rglob("*")}

    payload = build_remote_payload(_config(root.as_posix()), _discovery(), scan_id="SCAN-EXEC")
    output = io.StringIO()
    with redirect_stdout(output):
        exec(compile(payload, "<remote-payload>", "exec"), {"__name__": "__main__"})

    after = {path.relative_to(root).as_posix() for path in root.rglob("*")}
    records = [json.loads(line) for line in output.getvalue().splitlines()]
    assert after == before
    assert records[0]["record_type"] == "SERVER_INFO"
    assert records[-1]["record_type"] == "COLLECTION_COMPLETE"
    assert all(record["schema_version"] == 1 for record in records)
    assert all(record["scan_id"] == "SCAN-EXEC" for record in records)
    assert any(
        record["record_type"] == "SCAN_ERROR" and record["operation"] == "proc_scandir"
        for record in records
    )
    assets = [record for record in records if record["record_type"] == "ASSET"]
    assert any(record["relative_path"] == "runs/demo/report.md" for record in assets)
    assert all(not record["relative_path"].startswith("/") for record in assets)
    report_record = next(record for record in assets if record["relative_path"].endswith("report.md"))
    source_record = next(record for record in assets if record["relative_path"].endswith("worker.py"))
    receipt_record = next(
        record for record in assets if record["relative_path"].endswith("scan_receipt.json")
    )
    assert report_record["evidence_text"].startswith("run_id: demo")
    assert "evidence_text" not in source_record
    assert "utf16-demo" in receipt_record["evidence_text"]


def test_direct_success_requires_markers_valid_ndjson_and_clean_disconnect() -> None:
    preflight = _preflight_ok()
    direct = _result(_direct_command(), pid=202, stdout_lines=_ndjson())
    runner = FakeRunner([preflight, direct], [(), ()])

    result = _collector(runner).collect()

    assert result.receipt.outcome is RemoteOutcome.VERIFIED
    assert result.receipt.route == DIRECT_ROUTE
    assert result.receipt.disconnect_status == "VERIFIED"
    assert len(result.records) == len(_ndjson())
    assert [call["label"] for call in runner.calls] == ["PREFLIGHT", "DIRECT"]
    assert runner.calls[1]["command"] == _direct_command()
    assert runner.calls[1]["timeout_seconds"] == 45
    assert runner.calls[1]["streamed"] is True
    assert "SCAN-1" in str(runner.calls[1]["input_text"])
    assert runner.connection_calls[1]["attempt_pids"] == (202,)
    assert runner.connection_calls[1]["endpoints"] == (
        APPROVED_N607_HOST,
        APPROVED_BRIDGE_HOST,
    )


def test_known_direct_path_failure_with_valid_identity_uses_exact_bridge() -> None:
    preflight = _result(
        _preflight_command(),
        pid=301,
        returncode=1,
        stdout_lines=(
            "Config OK: N607 is direct. user szu2070436088; hostname 172.31.111.215",
            "SSH config: C:/repo/tools/n607_ssh_config",
            "Identity file OK: C:/Users/lh594/.ssh/id_ed25519_n607",
        ),
        stderr_tail="ssh: connect to host 172.31.111.215 port 22: Connection timed out",
        proxy_pids=(311,),
    )
    bridge = _result(
        _bridge_command(),
        pid=302,
        proxy_pids=(303,),
        stdout_lines=_ndjson(),
    )
    runner = FakeRunner([preflight, bridge], [(), ()])

    result = _collector(runner).collect()

    assert result.receipt.outcome is RemoteOutcome.VERIFIED
    assert result.receipt.route == BRIDGE_ROUTE
    assert runner.calls[1]["command"] == _bridge_command()
    assert runner.connection_calls[1]["attempt_pids"] == (302, 303)
    assert "N607-admin" not in " ".join(_bridge_command())
    assert "szu2310433034" not in " ".join(_bridge_command())


def test_identity_ambiguity_fails_without_any_collection_attempt() -> None:
    preflight = _result(
        _preflight_command(),
        pid=401,
        returncode=1,
        stderr_tail="No SSH identity file is configured for N607",
    )
    runner = FakeRunner([preflight], [()])

    result = _collector(runner).collect()

    assert result.receipt.outcome is RemoteOutcome.FAILED
    assert result.receipt.route is None
    assert len(runner.calls) == 1
    assert "identity" in (result.receipt.error or "").casefold()


@pytest.mark.parametrize(
    ("command_result", "connections", "expected"),
    [
        (
            _result(
                _direct_command(),
                pid=501,
                returncode=None,
                timed_out=True,
                child_exited=False,
            ),
            (ConnectionEvidence(501, APPROVED_N607_HOST, "SYN_SENT"),),
            RemoteOutcome.UNKNOWN,
        ),
        (
            _result(_direct_command(), pid=502, stdout_lines=("not-json",)),
            (),
            RemoteOutcome.FAILED,
        ),
        (
            _result(_direct_command(), pid=503, returncode=7, stderr_tail="remote error"),
            (),
            RemoteOutcome.FAILED,
        ),
        (
            _result(_direct_command(), pid=504, stdout_lines=_ndjson()),
            (ConnectionEvidence(504, APPROVED_N607_HOST, "ESTABLISHED"),),
            RemoteOutcome.UNKNOWN,
        ),
        (
            _result(_direct_command(), pid=505, stdout_lines=_ndjson(), child_exited=False),
            (),
            RemoteOutcome.UNKNOWN,
        ),
        (
            _result(_direct_command(), pid=506, stdout_lines=_ndjson()[:-1]),
            (),
            RemoteOutcome.FAILED,
        ),
    ],
)
def test_failure_and_unknown_semantics(
    command_result: CommandResult,
    connections: tuple[ConnectionEvidence, ...],
    expected: RemoteOutcome,
) -> None:
    runner = FakeRunner([_preflight_ok(), command_result], [(), connections])

    result = _collector(runner).collect()

    assert result.receipt.outcome is expected
    assert len(runner.calls) == 2
    assert result.receipt.outcome is not RemoteOutcome.VERIFIED


def test_active_training_is_observed_but_never_intervened_in() -> None:
    direct = _result(_direct_command(), pid=601, stdout_lines=_ndjson(active=True))
    runner = FakeRunner([_preflight_ok(), direct], [(), ()])

    result = _collector(runner).collect()

    assert result.receipt.outcome is RemoteOutcome.VERIFIED
    assert result.receipt.active_training_observed is True
    assert len(runner.calls) == 2
    combined_commands = " ".join(" ".join(call["command"]) for call in runner.calls)
    assert all(token not in combined_commands for token in ("taskkill", "pkill", "killall"))


def test_exit_zero_and_readable_output_are_not_preflight_success_evidence() -> None:
    preflight = _result(
        _preflight_command(),
        pid=701,
        stdout_lines=("some readable output",),
    )
    runner = FakeRunner([preflight], [()])

    result = _collector(runner).collect()

    assert result.receipt.outcome is RemoteOutcome.FAILED
    assert len(runner.calls) == 1


def test_preflight_requires_actual_remote_ordinary_user_evidence() -> None:
    preflight = _result(
        _preflight_command(),
        pid=801,
        proxy_pids=(802,),
        stdout_lines=(
            "Config OK: N607 is direct. user szu2070436088; hostname 172.31.111.215",
            "SSH config: C:/repo/tools/n607_ssh_config",
            "Identity file OK: C:/Users/lh594/.ssh/id_ed25519_n607",
            "user=someone_else",
            "project_root=/home/szu2070436088/2510044040/CV-SincNet",
            "Preflight OK: use ssh N607",
        ),
    )
    runner = FakeRunner([preflight], [()])

    result = _collector(runner).collect()

    assert result.receipt.outcome is RemoteOutcome.FAILED
    assert len(runner.calls) == 1
    assert "identity" in (result.receipt.error or "").casefold()


def test_missing_attempt_child_evidence_cannot_authorize_bridge_fallback() -> None:
    preflight = _result(
        _preflight_command(),
        pid=901,
        returncode=1,
        stdout_lines=(
            "Config OK: N607 is direct. user szu2070436088; hostname 172.31.111.215",
            "SSH config: C:/repo/tools/n607_ssh_config",
            "Identity file OK: C:/Users/lh594/.ssh/id_ed25519_n607",
        ),
        stderr_tail="ssh: connect to host 172.31.111.215 port 22: Connection timed out",
    )
    runner = FakeRunner([preflight], [()])

    result = _collector(runner).collect()

    assert result.receipt.outcome is RemoteOutcome.UNKNOWN
    assert len(runner.calls) == 1


def test_disconnect_probe_failure_is_unknown_and_stops() -> None:
    runner = FakeRunner(
        [_preflight_ok()],
        [ConnectionCheck(completed=False, error="netstat evidence unavailable")],
    )

    result = _collector(runner).collect()

    assert result.receipt.outcome is RemoteOutcome.UNKNOWN
    assert result.receipt.disconnect_status == "UNKNOWN"
    assert len(runner.calls) == 1


def test_preflight_path_fallback_requires_a_failed_preflight_exit() -> None:
    preflight = _result(
        _preflight_command(),
        pid=1001,
        proxy_pids=(1002,),
        returncode=0,
        stdout_lines=(
            "Config OK: N607 is direct. user szu2070436088; hostname 172.31.111.215",
            "SSH config: C:/repo/tools/n607_ssh_config",
            "Identity file OK: C:/Users/lh594/.ssh/id_ed25519_n607",
        ),
        stderr_tail="ssh: connect to host 172.31.111.215 port 22: Connection timed out",
    )
    runner = FakeRunner([preflight], [()])

    result = _collector(runner).collect()

    assert result.receipt.outcome is RemoteOutcome.FAILED
    assert len(runner.calls) == 1


def test_ndjson_requires_stable_asset_identity_and_one_final_completion() -> None:
    bad_identity = list(_ndjson())
    asset = json.loads(bad_identity[1])
    asset["asset_id"] = "asset:N607:N607_CVS_SINCNET:wrong"
    bad_identity[1] = json.dumps(asset)
    duplicate_complete = list(_ndjson())
    duplicate_complete.insert(-1, duplicate_complete[-1])
    final = json.loads(duplicate_complete[-1])
    final["record_count"] = len(duplicate_complete) - 1
    duplicate_complete[-1] = json.dumps(final)

    for pid, lines in ((1101, tuple(bad_identity)), (1102, tuple(duplicate_complete))):
        runner = FakeRunner(
            [_preflight_ok(), _result(_direct_command(), pid=pid, stdout_lines=lines)],
            [(), ()],
        )
        result = _collector(runner).collect()
        assert result.receipt.outcome is RemoteOutcome.FAILED


def test_n607_posix_backslash_is_literal_and_cannot_collide_with_nested_path() -> None:
    literal = r"runs/foo\bar.json"
    nested = "runs/foo/bar.json"

    assert normalize_relative_path(literal, location=Location.N607) == literal
    assert normalize_relative_path(r"runs/..\literal.json", location=Location.N607) == r"runs/..\literal.json"
    assert normalize_relative_path(r"\\leading.json", location=Location.N607) == r"\\leading.json"
    assert normalize_relative_path(r"C:\literal.json", location=Location.N607) == r"C:\literal.json"
    assert stable_asset_id(Location.N607, "N607_CVS_SINCNET", literal) != stable_asset_id(
        Location.N607, "N607_CVS_SINCNET", nested
    )


def test_verified_collection_requires_root_and_every_configured_carrier_scope() -> None:
    header = _ndjson()[0]
    complete = json.dumps(
        {
            "schema_version": 1,
            "scan_id": "SCAN-1",
            "record_type": "COLLECTION_COMPLETE",
            "record_count": 1,
            "scan_error_count": 0,
        }
    )
    missing_carrier = [line for line in _ndjson() if json.loads(line).get("relative_path") != "logs"]
    final = json.loads(missing_carrier[-1])
    final["record_count"] = len(missing_carrier) - 1
    missing_carrier[-1] = json.dumps(final)

    for pid, lines in ((1201, (header, complete)), (1202, tuple(missing_carrier))):
        runner = FakeRunner(
            [_preflight_ok(), _result(_direct_command(), pid=pid, stdout_lines=lines)],
            [(), ()],
        )
        result = _collector(runner).collect()
        assert result.receipt.outcome is RemoteOutcome.FAILED


def test_ndjson_rejects_not_present_root_scope() -> None:
    from tools.project_governance.collect_n607 import _parse_ndjson

    records = list(_ndjson())
    root_index = next(
        index
        for index, line in enumerate(records)
        if json.loads(line).get("record_type") == "SCOPE"
        and json.loads(line).get("relative_path") == ""
    )
    root_scope = json.loads(records[root_index])
    root_scope["status"] = "NOT_PRESENT"
    records[root_index] = json.dumps(root_scope)

    with pytest.raises(ValueError, match="root scope"):
        _parse_ndjson(
            tuple(records),
            scan_id="SCAN-1",
            expected_root="/home/szu2070436088/2510044040/CV-SincNet",
            expected_carriers=("runs", "logs"),
        )


def test_ndjson_rejects_scan_error_root_scope() -> None:
    from tools.project_governance.collect_n607 import _parse_ndjson

    records = list(_ndjson())
    root_index = next(
        index
        for index, line in enumerate(records)
        if json.loads(line).get("record_type") == "SCOPE"
        and json.loads(line).get("relative_path") == ""
    )
    root_scope = json.loads(records[root_index])
    root_scope["status"] = "SCAN_ERROR"
    records[root_index] = json.dumps(root_scope)

    with pytest.raises(ValueError, match="root scope"):
        _parse_ndjson(
            tuple(records),
            scan_id="SCAN-1",
            expected_root="/home/szu2070436088/2510044040/CV-SincNet",
            expected_carriers=("runs", "logs"),
        )


@pytest.mark.parametrize("duplicate_root", (False, True))
def test_ndjson_rejects_missing_or_duplicate_root_scope(duplicate_root: bool) -> None:
    from tools.project_governance.collect_n607 import _parse_ndjson

    records = list(_ndjson())
    root_index = next(
        index
        for index, line in enumerate(records)
        if json.loads(line).get("record_type") == "SCOPE"
        and json.loads(line).get("relative_path") == ""
    )
    if duplicate_root:
        records.insert(-1, records[root_index])
    else:
        del records[root_index]
    completion = json.loads(records[-1])
    completion["record_count"] = len(records) - 1
    records[-1] = json.dumps(completion)

    with pytest.raises(ValueError, match="root/carrier scope coverage"):
        _parse_ndjson(
            tuple(records),
            scan_id="SCAN-1",
            expected_root="/home/szu2070436088/2510044040/CV-SincNet",
            expected_carriers=("runs", "logs"),
        )


def test_ndjson_allows_not_present_optional_carrier_scope() -> None:
    from tools.project_governance.collect_n607 import _parse_ndjson

    records, _ = _parse_ndjson(
        _ndjson(),
        scan_id="SCAN-1",
        expected_root="/home/szu2070436088/2510044040/CV-SincNet",
        expected_carriers=("runs", "logs"),
    )

    assert any(
        record.get("record_type") == "SCOPE"
        and record.get("relative_path") == "logs"
        and record.get("status") == "NOT_PRESENT"
        for record in records
    )


def test_unknown_record_type_and_invalid_utf8_are_rejected() -> None:
    unknown = list(_ndjson())
    unknown.insert(
        -1,
        json.dumps(
            {
                "schema_version": 1,
                "scan_id": "SCAN-1",
                "record_type": "TRUST_ME",
            }
        ),
    )
    final = json.loads(unknown[-1])
    final["record_count"] = len(unknown) - 1
    unknown[-1] = json.dumps(final)
    cases: tuple[tuple[str | bytes, ...], ...] = (tuple(unknown), (b"\xff\xfe",))
    for offset, lines in enumerate(cases):
        runner = FakeRunner(
            [
                _preflight_ok(),
                _result(_direct_command(), pid=1301 + offset, stdout_lines=lines),
            ],
            [(), ()],
        )
        result = _collector(runner).collect()
        assert result.receipt.outcome is RemoteOutcome.FAILED


def test_payload_records_partial_proc_visibility_instead_of_silently_skipping() -> None:
    payload = build_remote_payload(_config(), _discovery(), scan_id="SCAN-PROC")
    namespace: dict[str, object] = {"__name__": "payload_test"}
    exec(compile(payload, "<remote-payload>", "exec"), namespace)

    class FakeScandir:
        def __enter__(self):
            return iter((SimpleNamespace(name="123"),))

        def __exit__(self, exc_type, exc, traceback):
            return False

    fake_os = SimpleNamespace(
        path=os.path,
        sep=os.sep,
        scandir=lambda path: FakeScandir(),
        readlink=lambda path: (_ for _ in ()).throw(PermissionError("denied")),
    )
    namespace["os"] = fake_os
    output = io.StringIO()
    with redirect_stdout(output):
        namespace["_processes"]()
    records = [json.loads(line) for line in output.getvalue().splitlines()]

    assert len(records) == 1
    assert records[0]["record_type"] == "SCAN_ERROR"
    assert records[0]["operation"] == "proc_partial_visibility"
    assert "123" in records[0]["error"]


def test_payload_deduplicates_revisited_assets_and_records_decode_errors(tmp_path: Path) -> None:
    payload = build_remote_payload(_config(), _discovery(), scan_id="SCAN-DEDUPE")
    namespace: dict[str, object] = {"__name__": "payload_test"}
    exec(compile(payload, "<remote-payload>", "exec"), namespace)
    link_metadata = SimpleNamespace(st_mode=stat.S_IFLNK | 0o777, st_size=4, st_mtime=0)

    output = io.StringIO()
    with redirect_stdout(output):
        namespace["_record_asset"]("ignored", "runs", "ROOT_DIRECT", link_metadata)
        namespace["_record_asset"]("ignored", "runs", "LINK", link_metadata)
    records = [json.loads(line) for line in output.getvalue().splitlines()]
    assert [record["record_type"] for record in records] == ["ASSET"]

    invalid_report = tmp_path / "report.md"
    invalid_report.write_bytes(b"\xff")
    output = io.StringIO()
    with redirect_stdout(output):
        namespace["_record_asset"](
            invalid_report.as_posix(),
            "runs/invalid/report.md",
            "CONTROL_EVIDENCE",
            os.lstat(invalid_report),
        )
    records = [json.loads(line) for line in output.getvalue().splitlines()]
    assert records[0]["record_type"] == "SCAN_ERROR"
    assert records[-1]["record_type"] == "ASSET"
    assert records[-1]["access_status"] == "SCAN_ERROR"


def test_payload_requires_ppid_for_a_project_bound_process() -> None:
    payload = build_remote_payload(_config(), _discovery(), scan_id="SCAN-PPID")
    namespace: dict[str, object] = {"__name__": "payload_test"}
    exec(compile(payload, "<remote-payload>", "exec"), namespace)

    class FakeScandir:
        def __enter__(self):
            return iter((SimpleNamespace(name="321"),))

        def __exit__(self, exc_type, exc, traceback):
            return False

    fake_os = SimpleNamespace(
        path=os.path,
        sep=os.sep,
        scandir=lambda path: FakeScandir(),
        readlink=lambda path: "/srv/CV-SincNet",
    )
    namespace["os"] = fake_os
    namespace["_read_process_text"] = lambda path, limit: (
        "python\x00train.py\x00" if path.endswith("/cmdline") else "Name:\tpython\n"
    )
    output = io.StringIO()
    with redirect_stdout(output):
        namespace["_processes"]()
    records = [json.loads(line) for line in output.getvalue().splitlines()]

    assert not any(record["record_type"] == "PROCESS" for record in records)
    assert any(
        record["record_type"] == "SCAN_ERROR"
        and record["operation"] == "proc_partial_visibility"
        for record in records
    )


def test_direct_command_reuses_the_preflight_config_path() -> None:
    preflight_script = DEFAULT_SSH_CONFIG.with_name("n607_ssh_preflight.ps1")
    direct = _result(
        _direct_command(DEFAULT_SSH_CONFIG),
        pid=1401,
        stdout_lines=_ndjson(),
    )
    preflight = _result(
        _preflight_command(preflight_script),
        pid=1400,
        proxy_pids=(1402,),
        stdout_lines=(
            "Config OK: N607 is direct. user szu2070436088; hostname 172.31.111.215",
            f"SSH config: {DEFAULT_SSH_CONFIG.as_posix()}",
            "Identity file OK: C:/Users/lh594/.ssh/id_ed25519_n607",
            "user=szu2070436088",
            "project_root=/home/szu2070436088/2510044040/CV-SincNet",
            "Preflight OK: use ssh N607",
        ),
    )
    runner = FakeRunner([preflight, direct], [(), ()])
    collector = N607Collector(
        _config("/home/szu2070436088/2510044040/CV-SincNet"),
        _discovery(),
        scan_id="SCAN-1",
        runner=runner,
        preflight_script=preflight_script,
    )

    result = collector.collect()

    assert result.receipt.outcome is RemoteOutcome.VERIFIED
    assert runner.calls[1]["command"][:3] == ("ssh", "-F", str(DEFAULT_SSH_CONFIG))


def test_process_ppid_and_scan_error_path_are_required_for_protocol_closure() -> None:
    bad_ppid = list(_ndjson(active=True))
    for index, line in enumerate(bad_ppid):
        record = json.loads(line)
        if record.get("record_type") == "PROCESS":
            record["ppid"] = None
            bad_ppid[index] = json.dumps(record)
            break

    missing_error_path = list(_ndjson())
    missing_error_path.insert(
        -1,
        json.dumps(
            {
                "schema_version": 1,
                "scan_id": "SCAN-1",
                "record_type": "SCAN_ERROR",
                "location": "N607",
                "root_id": "N607_CVS_SINCNET",
                "operation": "probe",
                "error_type": "OSError",
                "error": "missing path",
            }
        ),
    )
    completion = json.loads(missing_error_path[-1])
    completion["record_count"] = len(missing_error_path) - 1
    completion["scan_error_count"] = 1
    missing_error_path[-1] = json.dumps(completion)

    for pid, lines in ((1501, tuple(bad_ppid)), (1502, tuple(missing_error_path))):
        runner = FakeRunner(
            [_preflight_ok(), _result(_direct_command(), pid=pid, stdout_lines=lines)],
            [(), ()],
        )
        assert _collector(runner).collect().receipt.outcome is RemoteOutcome.FAILED


def test_runner_that_ignores_stream_handler_cannot_verify_readable_stdout() -> None:
    runner = FakeRunner(
        [_preflight_ok(), _result(_direct_command(), pid=1601, stdout_lines=_ndjson())],
        [(), ()],
        stream_output=False,
    )

    result = _collector(runner).collect()

    assert result.receipt.outcome is RemoteOutcome.FAILED
    assert "completion" in (result.receipt.error or "").casefold()


def test_linked_carrier_emits_one_asset_error_and_failed_scope() -> None:
    payload = build_remote_payload(_config(), _discovery(), scan_id="SCAN-LINK")
    namespace: dict[str, object] = {"__name__": "payload_test"}
    exec(compile(payload, "<remote-payload>", "exec"), namespace)
    directory_metadata = SimpleNamespace(st_mode=stat.S_IFDIR | 0o755, st_size=0, st_mtime=0)
    link_metadata = SimpleNamespace(st_mode=stat.S_IFLNK | 0o777, st_size=4, st_mtime=0)

    class FakeScandir:
        def __init__(self, entries=(), error: OSError | None = None):
            self.entries = entries
            self.error = error

        def __enter__(self):
            if self.error is not None:
                raise self.error
            return iter(self.entries)

        def __exit__(self, exc_type, exc, traceback):
            return False

    runs_entry = SimpleNamespace(name="runs", path="/srv/CV-SincNet/runs")

    def normalize(value: object) -> str:
        return str(value).replace("\\", "/")

    def fake_lstat(path: object):
        normalized = normalize(path)
        if normalized == "/srv/CV-SincNet":
            return directory_metadata
        if normalized == "/srv/CV-SincNet/runs":
            return link_metadata
        raise FileNotFoundError(normalized)

    def fake_scandir(path: object):
        normalized = normalize(path)
        if normalized == "/srv/CV-SincNet":
            return FakeScandir((runs_entry,))
        raise FileNotFoundError(normalized)

    namespace["os"] = SimpleNamespace(
        path=os.path,
        sep=os.sep,
        lstat=fake_lstat,
        scandir=fake_scandir,
        readlink=lambda path: "",
    )
    output = io.StringIO()
    with redirect_stdout(output):
        namespace["_main"]()
    records = [json.loads(line) for line in output.getvalue().splitlines()]

    assert sum(
        record["record_type"] == "ASSET" and record.get("relative_path") == "runs"
        for record in records
    ) == 1
    assert any(
        record["record_type"] == "SCAN_ERROR"
        and record["operation"] == "carrier_boundary"
        for record in records
    )
    assert any(
        record["record_type"] == "SCOPE"
        and record["relative_path"] == "runs"
        and record["status"] == "SCAN_ERROR"
        for record in records
    )
