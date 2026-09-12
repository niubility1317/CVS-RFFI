"""Actual CORE90 entrypoint trajectory parity on disposable synthetic source IQ."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys
from unittest.mock import patch

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def compare_main_traces(traces, *, amp):
    import torch
    from cvsrffi.cross_response.replay_audit import freeze, first_divergence
    left, right = freeze(traces["U1"]), freeze(traces["head_only"])
    for a,b in zip(left,right):
        if a["stage"] == b["stage"] == "post_update":
            # V1 stores auxiliary optimizer groups in the same object. Their
            # existence is expected; compare only the actual main groups/moments.
            main = a["values"]["optimizer"]
            candidate = b["values"]["optimizer"]
            count = len(main["param_groups"])
            ids = {p for group in main["param_groups"] for p in group["params"]}
            candidate["param_groups"] = candidate["param_groups"][:count]
            candidate["state"] = {k:v for k,v in candidate["state"].items() if k in ids}
        for event in (a,b):
            if event["stage"] == "unscaled_gradients" and amp:
                event["values"]["gradients"] = {
                    name: ({"finite_values": torch.where(torch.isfinite(g), g, 0),
                            "nan_mask": torch.isnan(g), "positive_inf": torch.isposinf(g),
                            "negative_inf": torch.isneginf(g)} if g is not None else None)
                    for name,g in event["values"]["gradients"].items()}
    divergence = first_divergence(left,right)
    if divergence and divergence["path"].startswith("["):
        index = int(divergence["path"].split("]")[0][1:])
        divergence.update(step_id=left[index]["step"],stage=left[index]["stage"])
    return divergence


def run(root, *, amp=False, device="cuda:0", inject=None, version=2):
    import torch
    from scripts.verify_core90_cross_response import make_test_config, run_synthetic_variant
    from cvsrffi.cross_response.integration import CrossResponseRuntime
    from cvsrffi.cross_response.training import IndependentAuxiliaryTransaction
    from cvsrffi.cross_response.replay_audit import freeze, capture_rng_state, first_divergence
    root.mkdir(parents=True, exist_ok=True)
    config_path = root / "synthetic_cross_response_config.json"
    if not config_path.exists():
        make_test_config(config_path)
        config = json.loads(config_path.read_text(encoding="utf-8"))
        config["cross_response"]["implementation_version"] = version
        config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    traces = {}
    actual_init = CrossResponseRuntime.__init__
    actual_step = IndependentAuxiliaryTransaction.step
    def poisoned_step(self, loss):
        return actual_step(self, loss * float(inject))
    summaries = []
    for variant in ("U1", "head_only"):
        trace = []
        def observer(step, stage, values):
            trace.append({"step": step, "stage": stage, "values": freeze(values), "rng": capture_rng_state()})
        def initialize(self, *args, **kwargs):
            actual_init(self, *args, **kwargs)
            self.replay_observer = observer
        with patch.object(CrossResponseRuntime, "__init__", initialize):
            if variant == "head_only" and inject is not None:
                # Overflow checks intentionally cannot satisfy positive head activation.
                with patch.object(IndependentAuxiliaryTransaction, "step", poisoned_step), \
                     patch("scripts.verify_core90_cross_response.assert_activation", lambda _: None):
                    result = run_synthetic_variant(variant, root, device=device, epochs=2,
                        deterministic=True, amp=amp, records_per_cell=512)
            else:
                result = run_synthetic_variant(variant, root, device=device, epochs=2,
                    deterministic=True, amp=amp, records_per_cell=512)
        traces[variant] = trace
        summaries.append({k: result[k] for k in ("variant", "training_state_path", "elapsed_seconds", "train_exit_code")})
    # Matching production AMP overflow is a state transition, not a branch
    # difference. Preserve its exact location/sign and compare every finite value.
    # Do not relax nonfinite input/loss/model-state checks.
    divergence = compare_main_traces(traces, amp=amp)
    report = {"status": "EXACT_MAIN_TRAJECTORY_PARITY" if divergence is None else "FIRST_DIVERGENCE",
              "first_divergence": divergence, "events_per_branch": {k: len(v) for k,v in traces.items()},
              "steps": len({e["step"] for e in traces["U1"]}), "amp": amp, "device": device,
              "auxiliary_injection": inject, "implementation_version": version,
              "amp_overflow_steps": {k: [e["step"] for e in v if e["stage"] == "unscaled_gradients"
                                        and e["values"]["nonfinite"] is not None] for k,v in traces.items()},
              "atol": 0, "rtol": 0, "runs": summaries,
              "scope": "actual two-epoch CORE90 loop; synthetic source IQ; no target scoring; no formal weights"}
    torch.save(traces, root / "trajectory_traces.pt")
    (root / "trajectory_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    if divergence:
        raise AssertionError(divergence)
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--inject", choices=("nan", "inf"))
    p.add_argument("--version", type=int, choices=(1,2), default=2)
    p.add_argument("--analyze-existing", action="store_true")
    a = p.parse_args()
    if a.analyze_existing:
        import torch
        traces = torch.load(a.output_root / "trajectory_traces.pt", map_location="cpu", weights_only=False)
        report = json.loads((a.output_root / "trajectory_report.json").read_text(encoding="utf-8"))
        report["first_divergence"] = compare_main_traces(traces, amp=report["amp"])
        report["status"] = "FIRST_MAIN_DIVERGENCE" if report["first_divergence"] else "EXACT_MAIN_TRAJECTORY_PARITY"
        report["comparison_scope"] = "main optimizer groups only; auxiliary moments excluded; exact AMP nonfinite masks/signs retained"
        with (a.output_root / "trajectory_main_reanalysis.json").open("x", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
        print(json.dumps(report, indent=2))
    else:
        run(a.output_root, amp=a.amp, device=a.device, inject=a.inject, version=a.version)
