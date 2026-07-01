#!/usr/bin/env python
import argparse
import json
import sys
from pathlib import Path


CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

import torch  # noqa: E402

from cvsrffi.gate_metrics import summarize_gate_decisions  # noqa: E402
from cvsrffi.hard_gate import GateThresholds, LocalComponentHardGate  # noqa: E402
from cvsrffi.negative_sampling import sample_interclass_slerp_negatives, sample_shell_negatives  # noqa: E402
from cvsrffi.prototype_bank import VacuumGaussianPrototypeBank  # noqa: E402


def _synthetic_logits(batch, num_classes: int) -> torch.Tensor:
    logits = torch.zeros((batch.z.size(0), int(num_classes)), dtype=torch.float32)
    for i, cls in enumerate(batch.source_class):
        if cls is not None and 0 <= int(cls) < num_classes:
            logits[i, int(cls)] = 4.0
    return logits


def run_benchmark(prototype_package: str | Path) -> dict:
    bank = VacuumGaussianPrototypeBank.from_phase2_package(prototype_package)
    gate = LocalComponentHardGate(bank, GateThresholds(use_energy_gate=False))
    rows = []
    shell = sample_shell_negatives(bank, n_per_component=4, seed=11)
    inter = sample_interclass_slerp_negatives(bank, n_per_pair=4, seed=13)
    for batch in (shell, inter):
        if batch.z.numel() == 0:
            continue
        logits = _synthetic_logits(batch, max(bank.classes.keys()) + 1)
        for kind, out in zip(batch.kind, gate.batch_decide(batch.z, logits)):
            row = dict(out)
            row["synthetic_kind"] = kind
            rows.append(row)
    summary = summarize_gate_decisions(rows)
    summary["shell_accept_rate"] = _accept_rate(rows, "shell")
    summary["inter_slerp_accept_rate"] = _accept_rate(rows, "inter_class")
    summary["source_note"] = "synthetic/proxy reject benchmark; not real Stage2 unknown evidence"
    return summary


def _accept_rate(rows, kind):
    subset = [r for r in rows if r.get("synthetic_kind") == kind]
    if not subset:
        return None
    return sum(1 for r in subset if str(r.get("decision", "")).startswith("ACCEPT")) / len(subset)


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate local component hard gate on synthetic open-space negatives.")
    parser.add_argument("--prototype_package", required=True)
    parser.add_argument("--output", default="")
    args = parser.parse_args()
    metrics = run_benchmark(args.prototype_package)
    payload = json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True)
    if args.output:
        Path(args.output).write_text(payload, encoding="utf-8")
    else:
        print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

