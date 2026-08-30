"""Real-checkpoint, source-only compatibility smoke for Phase1 BiCAD-XR.

This is a technical smoke, not a training or performance experiment.  It
strictly reconstructs the historical ADV3B02 deployment model, attaches fresh
BiCAD-XR training-only modules, performs one optimizer step on synthetic source
IQ, and exercises the clean plus three registered LEO_WEAK forward paths.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = REPO_ROOT / "code"
for _path in (CODE_ROOT, CODE_ROOT / "SSDG"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from cvsrffi.checkpoint_loading import build_exact_ssdg_model_from_checkpoint
from cvsrffi.phase1_bicad_xr.config import candidate_config
from cvsrffi.phase1_bicad_xr.trainer import BiCADXRTrainer


SCENARIOS = (
    "clean",
    "leo_clear_weak",
    "leo_low_elev_weak",
    "leo_rain_weak",
)


def _load_ssdg_module() -> Any:
    path = CODE_ROOT / "SSDG" / "train_ssdg.py"
    spec = importlib.util.spec_from_file_location("bicad_xr_checkpoint_smoke_ssdg", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load SSDG runtime: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_checkpoint(path: Path) -> Mapping[str, Any]:
    if not path.is_file() or path.stat().st_size <= 0:
        raise FileNotFoundError(f"checkpoint is missing or empty: {path}")
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, Mapping):
        raise TypeError("checkpoint payload must be a mapping")
    return payload


def _input_len(payload: Mapping[str, Any]) -> int:
    args = payload.get("args")
    if not isinstance(args, Mapping):
        raise ValueError("checkpoint is missing recorded args")
    for name in ("wisig_out_len", "input_len"):
        value = args.get(name)
        if value is not None and int(value) > 0:
            return int(value)
    raise ValueError("checkpoint does not record a positive input length")


def _tx_logits(output: Any) -> torch.Tensor:
    if torch.is_tensor(output):
        logits = output
    elif isinstance(output, Mapping):
        logits = output.get("tx_logits", output.get("logits"))
    else:
        logits = None
    if not torch.is_tensor(logits) or logits.ndim != 2:
        raise ValueError("model output does not contain rank-2 TX logits")
    if not bool(torch.isfinite(logits).all()):
        raise FloatingPointError("model output contains non-finite TX logits")
    return logits


def _write_result_once(output_dir: Path, result: Mapping[str, Any]) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "smoke_result.json"
    if path.exists():
        raise FileExistsError(f"refusing to overwrite smoke result: {path}")
    temporary = output_dir / f".{path.name}.{os.getpid()}.tmp"
    data = json.dumps(dict(result), ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(data)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def run_smoke(
    checkpoint_path: str | Path,
    *,
    output_dir: str | Path,
    device_name: str = "cpu",
    candidate_id: str = "ADV3B02-BiCAD-XDC-V1",
    seed: int = 392002,
    checkpoint_loader=_load_checkpoint,
    model_rebuilder=build_exact_ssdg_model_from_checkpoint,
    ssdg_module: Any | None = None,
) -> dict[str, Any]:
    """Run one bounded compatibility smoke and persist one explicit result."""

    checkpoint_path = Path(checkpoint_path).resolve()
    output_dir = Path(output_dir).resolve()
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    torch.manual_seed(int(seed))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(seed))

    payload = checkpoint_loader(checkpoint_path)
    ssdg = _load_ssdg_module() if ssdg_module is None else ssdg_module
    input_len = _input_len(payload)
    model, reconstruction = model_rebuilder(
        payload,
        input_len=input_len,
        device=device,
        ssdg_module=ssdg,
    )
    if any(int(reconstruction.get(name, 0)) != 0 for name in (
        "missing_keys", "unexpected_keys", "skipped_mismatch"
    )):
        raise RuntimeError(f"strict checkpoint reconstruction failed: {reconstruction}")

    concat_model = ssdg._BiCADXRConcatForward(model)
    trainer = BiCADXRTrainer(concat_model, candidate_config(candidate_id), num_receivers=4).to(device)
    optimizer = torch.optim.AdamW(trainer.parameters(), lr=2e-4, weight_decay=1e-4)
    sat_args = SimpleNamespace(
        seed=int(seed),
        sat_view_seed=int(seed),
        sat_fs_hz=25e6,
        sat_fc_hz=2.462e9,
    )
    augmenter = ssdg._build_bicad_xr_concat_augmenter(sat_args)

    generator = torch.Generator(device=device).manual_seed(int(seed))
    x = torch.randn((8, 2, input_len), generator=generator, device=device) * 0.05
    tx = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3], device=device, dtype=torch.long)
    receiver = torch.tensor([0, 1, 2, 3, 1, 2, 3, 0], device=device, dtype=torch.long)
    day = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1], device=device, dtype=torch.long)

    trainer.train()
    optimizer.zero_grad(set_to_none=True)
    step, view = ssdg._bicad_xr_labeled_step(
        trainer,
        augmenter,
        x,
        tx,
        receiver,
        day,
        args=sat_args,
        epoch=80,
        batch_idx=1,
        update=501,
        total_updates=5000,
    )
    if not bool(torch.isfinite(step.total)):
        raise FloatingPointError("BiCAD-XR smoke loss is non-finite")
    backward_audit = trainer.apply_backward_controls(step)
    finite_gradients = [
        bool(torch.isfinite(parameter.grad).all())
        for parameter in trainer.parameters()
        if parameter.grad is not None
    ]
    if not finite_gradients or not all(finite_gradients):
        raise FloatingPointError("BiCAD-XR smoke gradients are missing or non-finite")
    optimizer.step()

    evaluations: dict[str, Any] = {}
    model.eval()
    with torch.no_grad():
        for index, scenario in enumerate(SCENARIOS):
            observed = x
            if scenario != "clean":
                scenario_generator = torch.Generator(device=device).manual_seed(
                    int(seed) + (index * 1_000_003)
                )
                received = ssdg.apply_sat_channel_for_scenario(
                    x,
                    scenario,
                    sat_args,
                    gen=scenario_generator,
                    return_meta=False,
                )
                observed = received[0] if isinstance(received, tuple) else received
            logits = _tx_logits(model(observed))
            evaluations[scenario] = {
                "finite": True,
                "logits_shape": list(logits.shape),
            }

    result = {
        "status": "PASS",
        "scope": "TECHNICAL_COMPATIBILITY_SMOKE",
        "performance_claim": False,
        "checkpoint": str(checkpoint_path),
        "checkpoint_bytes": int(checkpoint_path.stat().st_size),
        "candidate_id": str(candidate_id),
        "seed": int(seed),
        "device": str(device),
        "input_len": input_len,
        "strict_reconstruction": True,
        "checkpoint_load_strict": True,
        "missing_keys": [],
        "unexpected_keys": [],
        "shape_mismatches": [],
        "reconstruction_audit": dict(reconstruction),
        "fresh_bicad_runtime": True,
        "historical_checkpoint_has_bicad_runtime": bool(payload.get("bicad_xr_runtime")),
        "optimizer_step_complete": True,
        "backward_controls_complete": True,
        "backward_controls": backward_audit,
        "training_loss_finite": True,
        "concat_sat_ce_only": True,
        "concat_forward_batch_size": int(view["total_batch_size"]),
        "satellite_training_scenario": str(view["scenario"]),
        "evaluations": evaluations,
        "four_scenarios_complete": set(evaluations) == set(SCENARIOS),
        "source_only": True,
        "target_access": False,
        "phase2_access": False,
        "support_access": False,
        "query_access": False,
        "truth_access": False,
    }
    _write_result_once(output_dir, result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--candidate", default="ADV3B02-BiCAD-XDC-V1")
    parser.add_argument("--seed", type=int, default=392002)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = run_smoke(
            args.checkpoint,
            output_dir=args.output_dir,
            device_name=args.device,
            candidate_id=args.candidate,
            seed=args.seed,
        )
    except Exception as exc:
        failure = {
            "status": "FAILED",
            "scope": "TECHNICAL_COMPATIBILITY_SMOKE",
            "performance_claim": False,
            "exception_type": type(exc).__name__,
            "message": str(exc),
        }
        try:
            _write_result_once(Path(args.output_dir), failure)
        except FileExistsError:
            pass
        print(json.dumps(failure, ensure_ascii=False, sort_keys=True))
        return 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
