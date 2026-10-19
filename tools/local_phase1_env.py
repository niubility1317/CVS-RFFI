"""Native Windows Phase1 environment, launcher and synthetic GPU verification.

Machine paths and generated evidence belong in an ignored local profile.
This tool never launches an existing registered experiment by itself.
"""
from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time


def write_new(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def read_profile(path: str) -> dict:
    profile = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    for key in ("python", "checkout"):
        if not Path(profile[key]).exists():
            raise FileNotFoundError(profile[key])
    return profile


def child_env(profile: dict) -> dict:
    env = dict(os.environ)
    threads = str(profile["cpu_threads"])
    env.update(PYTHONUTF8="1", PYTHONIOENCODING="utf-8",
               OMP_NUM_THREADS=threads, MKL_NUM_THREADS=threads,
               OPENBLAS_NUM_THREADS=threads, NUMEXPR_NUM_THREADS=threads,
               CUDA_VISIBLE_DEVICES=str(profile["physical_gpu"]))
    root = Path(profile["checkout"])
    env["PYTHONPATH"] = os.pathsep.join(str(p) for p in (root, root / "code"))
    # Put the selected environment's DLL locations first, without changing Windows.
    base = Path(profile["base_python"]).parent
    env["PATH"] = os.pathsep.join(str(p) for p in
                                (Path(profile["python"]).parent, base,
                                 base / "Library" / "bin", base / "Scripts")) + os.pathsep + env.get("PATH", "")
    return env


def create(args) -> None:
    if os.name != "nt":
        raise RuntimeError("This entrypoint configures native Windows only")
    prefix = Path(args.prefix).resolve()
    profile_path = Path(args.profile).resolve()
    checkout = Path(args.checkout).resolve()
    if prefix.exists() or profile_path.exists():
        raise FileExistsError("Use the existing profile, or choose a new prefix/profile")
    if not (checkout / "experiments" / "cvs_f_supcon50" / "model.py").is_file():
        raise FileNotFoundError("Checkout must contain the F + cross-RX implementation")
    # Package files are shared read-only. No install/upgrade touches the parent.
    subprocess.run([sys.executable, "-m", "venv", "--system-site-packages", str(prefix)], check=True)
    python = prefix / "Scripts" / "python.exe"
    profile = dict(schema="local_phase1_env_v1", python=str(python),
                   base_python=sys.executable, checkout=str(checkout),
                   physical_gpu=0, max_gpu_workers=1, cpu_threads=2,
                   dataloader_workers=0, batch_size=128, precision="FP32",
                   compile=False, inherits_parent_packages=True,
                   source_data_status="NOT_CONFIGURED",
                   note="Resource profile does not change any registered scientific recipe")
    subprocess.run([str(python), "-X", "utf8", "-c", "import torch; assert torch.cuda.is_available(); print(torch.__version__)"],
                   env=child_env(profile), cwd=checkout, check=True)
    write_new(profile_path, profile)
    print(json.dumps(dict(status="CREATED", profile=str(profile_path)), ensure_ascii=False))


def verify(profile: dict, steps: int) -> dict:
    import numpy as np
    import torch
    for name in ("scipy", "h5py", "pandas", "sklearn", "matplotlib", "yaml", "tqdm", "joblib", "pytest", "torchvision"):
        importlib.import_module(name)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    torch.set_num_threads(profile["cpu_threads"])
    torch.set_num_interop_threads(1)
    # Preserve the registered F full-FP32 policy, including disabled TF32.
    from experiments.cvs_equivariant_identity.precision import numerical_context
    from experiments.cvs_f_supcon50 import design, model, checks
    from experiments.cvs_rdrop_sinc_units.optimizer import build, clip_physical

    torch.manual_seed(20261019)
    bridge = np.arange(12, dtype=np.float32).reshape(3, 4)
    if not np.array_equal(torch.from_numpy(bridge).cuda().cpu().numpy(), bridge):
        raise RuntimeError("NumPy/CUDA conversion mismatch")
    config = design.config(next(row for row in design.training_rows() if row["arm"] == "CROSS_RX"))
    loss_values, timings = [], []
    with numerical_context(design.FULL_FP32_POLICY):
        network = model.build(config, "cuda:0").train()
        optimizer = build(network, network.parameters(), "CENTER_UNITS",
                          lr=0.0002, eps=1e-8, weight_decay=0.0001, foreach=False)
        head = dict(network.named_parameters())[checks.W]
        head_before = head.detach().clone()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        for index in range(steps + 2):
            started = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            loss, metrics = checks.fixture(network, 15, "cuda:0", seed=20261019 + index)
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite synthetic loss")
            loss.backward()
            for parameter in network.parameters():
                if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                    raise RuntimeError("Nonfinite synthetic gradient")
            clip_physical(optimizer, network.parameters(), 5.0)
            optimizer.step()
            torch.cuda.synchronize()
            if index >= 2:
                timings.append(time.perf_counter() - started)
                loss_values.append(float(loss.detach()))
        if torch.equal(head_before, head):
            raise RuntimeError("Default head did not participate in joint updates")
        peak = torch.cuda.max_memory_allocated() / 1024**2
        if network._supcon_handles or network._supcon_pending is not None:
            raise RuntimeError("Temporary feature hooks were not cleared")
        if network._supcon_applied != steps + 2:
            raise RuntimeError("Cross-RX objective was not applied on every synthetic step")
    packages = {name: importlib.metadata.version(name) for name in
                ("torch", "torchvision", "numpy", "scipy", "h5py", "pandas", "scikit-learn", "matplotlib", "PyYAML", "tqdm", "joblib", "pytest")}
    return dict(status="VERIFIED", purpose="synthetic_environment_compatibility_only",
                python=sys.version, executable=sys.executable, base_prefix=sys.base_prefix,
                packages=packages, cuda_runtime=torch.version.cuda,
                gpu=torch.cuda.get_device_name(0), compute_capability=list(torch.cuda.get_device_capability(0)),
                compiled_archs=torch.cuda.get_arch_list(), cpu_threads=torch.get_num_threads(),
                model="actual F + CROSS_RX shallow/middle SupCon + paired RDrop",
                parameters=sum(p.numel() for p in network.parameters()),
                batch_size=128, input_shape=[128, 2, 256], precision="FP32", tf32=False,
                warmup_steps=2, measured_steps=steps, median_step_seconds=statistics.median(timings),
                peak_allocated_MiB=peak, default_head_jointly_updated=True,
                cross_rx_applied_steps=network._supcon_applied, numpy_cuda_roundtrip=True,
                losses=loss_values, dataset_reads=0, checkpoint_reads=0, target_reads=0,
                limitation="No data loading, source training, U/EMA stage or target scoring benchmark; not an epoch-speed estimate")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True)
    commands = parser.add_subparsers(dest="command", required=True)
    setup = commands.add_parser("create")
    setup.add_argument("--prefix", required=True)
    setup.add_argument("--checkout", required=True)
    check = commands.add_parser("check")
    check.add_argument("--output", required=True)
    check.add_argument("--steps", type=int, default=5, choices=range(1, 21))
    run = commands.add_parser("run")
    run.add_argument("python_args", nargs=argparse.REMAINDER)
    commands.add_parser("_verify").add_argument("--steps", type=int, required=True)
    args = parser.parse_args()
    if args.command == "create":
        create(args)
        return
    profile = read_profile(args.profile)
    if args.command == "_verify":
        print(json.dumps(verify(profile, args.steps), ensure_ascii=False, allow_nan=False))
        return
    if args.command == "run":
        argv = args.python_args
        if argv[:1] == ["--"]:
            argv = argv[1:]
        if not argv:
            parser.error("run requires Python arguments after --")
        raise SystemExit(subprocess.call([profile["python"], "-X", "utf8", *argv],
                                        cwd=profile["checkout"], env=child_env(profile)))
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(output)
    result = subprocess.run([profile["python"], "-X", "utf8", str(Path(__file__).resolve()),
                             "--profile", str(Path(args.profile).resolve()), "_verify", "--steps", str(args.steps)],
                            cwd=profile["checkout"], env=child_env(profile), check=True,
                            capture_output=True, text=True, encoding="utf-8")
    if result.stderr:
        print(result.stderr, file=sys.stderr, end="")
    report = json.loads(result.stdout)
    write_new(output, report)
    print(json.dumps(report, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
