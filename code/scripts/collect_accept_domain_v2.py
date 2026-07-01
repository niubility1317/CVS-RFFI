import argparse
import json
import re
from pathlib import Path
from typing import Any


SKIPPED_TEST_RE = re.compile(r"overall_tx\s*=\s*nan%\s*\(0/0\)", re.IGNORECASE)
AUX_GRAD_RE = re.compile(r"\[(?:GRAD|AUX|PAIC)[^\]]*\].*?\b(?:aux|grad)[^=\n]*=\s*nan\b", re.IGNORECASE)
LOSS_NAN_RE = re.compile(r"\b(?:loss|total_loss|train_loss)\s*[:=]\s*nan\b", re.IGNORECASE)
METRIC_NAN_RE = re.compile(r"\b(?:acc|accuracy|auc|far|metric)[A-Za-z0-9_\-]*\s*[:=]\s*nan\b", re.IGNORECASE)
FATAL_RE = re.compile(r"Traceback|RuntimeError|CUDA out of memory|out of memory|Killed|unrecognized arguments|NaN loss", re.IGNORECASE)


def classify_log_nan_lines(text: str) -> dict[str, int]:
    text = str(text or "")
    skipped = len(SKIPPED_TEST_RE.findall(text))
    aux = len(AUX_GRAD_RE.findall(text))
    real_loss = len(LOSS_NAN_RE.findall(text))
    real_metric = len(METRIC_NAN_RE.findall(text))
    fatal = len(FATAL_RE.findall(text)) + real_loss
    return {
        "skipped_test_placeholder": skipped,
        "aux_grad_telemetry": aux,
        "real_loss_nan": real_loss,
        "real_metric_nan": real_metric,
        "fatal_nan": fatal,
    }


def is_valid_effective_candidate(candidate_id: str) -> bool:
    cid = str(candidate_id or "").strip()
    m = re.match(r"^([TR])(\d{2})", cid)
    if not m:
        return True
    prefix, num_s = m.groups()
    num = int(num_s)
    if prefix == "T" and 16 <= num <= 31:
        return False
    return True


def collect_logs(paths: list[str]) -> dict[str, Any]:
    rows = []
    for p in paths:
        path = Path(p)
        text = path.read_text(encoding="utf-8", errors="replace")
        summary = classify_log_nan_lines(text)
        rows.append({"path": str(path), "nan_summary": summary, "fatal_count": summary["fatal_nan"]})
    return {
        "fatal_count": sum(r["fatal_count"] for r in rows),
        "rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect accept-domain v2 log diagnostics.")
    parser.add_argument("logs", nargs="*", help="Log files to parse.")
    parser.add_argument("--output", default="", help="Optional JSON output path.")
    args = parser.parse_args()
    data = collect_logs(args.logs)
    payload = json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True)
    if args.output:
        Path(args.output).write_text(payload, encoding="utf-8")
    else:
        print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

