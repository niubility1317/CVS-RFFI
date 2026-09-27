import os
from typing import Any, Dict, Optional


CPU_THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
)
DEFAULT_CPU_THREADS = 4
DEFAULT_CPU_INTEROP_THREADS = 1


def _positive_int(value: Any, default: Optional[int] = None) -> Optional[int]:
    if value is None:
        return default
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def _resolve_cpu_threads(cpu_threads: Any = None) -> int:
    return (
        _positive_int(cpu_threads)
        or _positive_int(os.environ.get("CVSRFFI_CPU_THREADS"))
        or _positive_int(os.environ.get("OMP_NUM_THREADS"))
        or DEFAULT_CPU_THREADS
    )


def _resolve_interop_threads(cpu_interop_threads: Any = None) -> int:
    return (
        _positive_int(cpu_interop_threads)
        or _positive_int(os.environ.get("CVSRFFI_CPU_INTEROP_THREADS"))
        or DEFAULT_CPU_INTEROP_THREADS
    )


def configure_cpu_thread_env(
    cpu_threads: Any = None,
    cpu_interop_threads: Any = None,
    *,
    force: bool = False,
) -> Dict[str, Any]:
    """Set conservative CPU thread defaults before torch/numpy heavy work starts."""
    resolved_threads = _resolve_cpu_threads(cpu_threads)
    resolved_interop = _resolve_interop_threads(cpu_interop_threads)

    for key in CPU_THREAD_ENV_VARS:
        if force or key not in os.environ:
            os.environ[key] = str(resolved_threads)
    if force or "CVSRFFI_CPU_THREADS" not in os.environ:
        os.environ["CVSRFFI_CPU_THREADS"] = str(resolved_threads)
    if force or "CVSRFFI_CPU_INTEROP_THREADS" not in os.environ:
        os.environ["CVSRFFI_CPU_INTEROP_THREADS"] = str(resolved_interop)

    return snapshot_thread_runtime(cpu_threads=resolved_threads, cpu_interop_threads=resolved_interop)


def configure_torch_thread_runtime(
    cpu_threads: Any = None,
    cpu_interop_threads: Any = None,
    *,
    force: bool = False,
) -> Dict[str, Any]:
    info = configure_cpu_thread_env(cpu_threads, cpu_interop_threads, force=force)
    import torch

    torch.set_num_threads(int(info["cpu_threads"]))
    try:
        torch.set_num_interop_threads(int(info["cpu_interop_threads"]))
    except RuntimeError as exc:
        info["torch_interop_warning"] = str(exc)
    info["torch_num_threads"] = int(torch.get_num_threads())
    try:
        info["torch_num_interop_threads"] = int(torch.get_num_interop_threads())
    except RuntimeError:
        info["torch_num_interop_threads"] = None
    return info


def snapshot_thread_runtime(
    *,
    cpu_threads: Any = None,
    cpu_interop_threads: Any = None,
) -> Dict[str, Any]:
    try:
        import torch
    except Exception:
        torch = None
    return {
        "cpu_threads": _resolve_cpu_threads(cpu_threads),
        "cpu_interop_threads": _resolve_interop_threads(cpu_interop_threads),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS", ""),
        "mkl_num_threads": os.environ.get("MKL_NUM_THREADS", ""),
        "openblas_num_threads": os.environ.get("OPENBLAS_NUM_THREADS", ""),
        "numexpr_num_threads": os.environ.get("NUMEXPR_NUM_THREADS", ""),
        "vec_maximum_threads": os.environ.get("VECLIB_MAXIMUM_THREADS", ""),
        "blis_num_threads": os.environ.get("BLIS_NUM_THREADS", ""),
        "torch_num_threads": int(torch.get_num_threads()) if torch is not None else None,
        "torch_num_interop_threads": int(torch.get_num_interop_threads()) if torch is not None else None,
    }
