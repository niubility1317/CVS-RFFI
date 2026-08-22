#!/usr/bin/env python
"""Audit Phase1 source-only spectral identifiability on the frozen L_s split."""

from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402


CODE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = CODE_ROOT.parent
for candidate in (str(REPO_ROOT), str(CODE_ROOT)):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

from SSDG.train_ssdg import _build_ssdg_wisig_data, build_arg_parser  # noqa: E402
from cvsrffi.eval import apply_sat_channel_for_scenario  # noqa: E402
from cvsrffi.spectral_identifiability import (  # noqa: E402
    SpectralIdentifiabilityAccumulator,
    build_center_mask,
    extract_band_descriptors,
    select_hsid_role_masks,
    select_sid_mask,
)
from cvsrffi.tensors import make_torch_generator, set_seed, unpack_batch  # noqa: E402


LEO_WEAK_SCENARIOS = (
    "leo_clear_weak",
    "leo_low_elev_weak",
    "leo_rain_weak",
)
EXPECTED_RATIOS = (0.07, 0.63, 0.15, 0.15)


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_json_safe(payload), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_csv(path: Path, stats: Mapping[str, np.ndarray], mask: np.ndarray) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("band", "j_score", "tx_scatter", "domain_scatter", "noise_scatter", "selected"))
        for index in range(mask.size):
            writer.writerow(
                (
                    index,
                    float(stats["j_score"][index]),
                    float(stats["tx_scatter"][index]),
                    float(stats["domain_scatter"][index]),
                    float(stats["noise_scatter"][index]),
                    int(mask[index]),
                )
            )
    os.replace(temporary, path)


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)


def _torch_mask_to_numpy(mask: torch.Tensor) -> np.ndarray:
    values = mask.detach().to(device="cpu", dtype=torch.uint8).tolist()
    return np.asarray(values, dtype=np.uint8)


def _atomic_plot(path: Path, stats: Mapping[str, np.ndarray], mask: np.ndarray) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    x = np.arange(mask.size)
    figure, axis = plt.subplots(figsize=(10, 4.5), constrained_layout=True)
    axis.plot(x, stats["j_score"], label="J score", color="#1f5a94", linewidth=1.5)
    axis.scatter(x[mask], stats["j_score"][mask], label="selected", color="#d1495b", s=18)
    axis.set_xlabel("FFT band")
    axis.set_ylabel("TX scatter / nuisance scatter")
    axis.set_title("Phase1 L_s spectral identifiability")
    axis.grid(alpha=0.2)
    axis.legend()
    figure.savefig(temporary, format="png", dpi=160)
    plt.close(figure)
    os.replace(temporary, path)


def _validate_protocol(args) -> None:
    if str(args.dataset).lower() != "wisig":
        raise ValueError("P0 spectral audit supports only the Phase1 WiSig protocol")
    if str(args.phase1_source_role_protocol) != "l_s_u_s_v_cal_v_select":
        raise ValueError("P0 requires phase1_source_role_protocol=l_s_u_s_v_cal_v_select")
    if str(args.split_mode) != "tx_rx_day_1_7_2":
        raise ValueError("P0 requires split_mode=tx_rx_day_1_7_2")
    actual = (
        float(args.labeled_ratio),
        float(args.unlabeled_ratio),
        float(args.source_cal_ratio),
        float(args.source_select_ratio),
    )
    if any(abs(got - expected) > 1e-9 for got, expected in zip(actual, EXPECTED_RATIOS)):
        raise ValueError(f"P0 source-role ratios must be {EXPECTED_RATIOS}, got {actual}")
    if int(args.fft_bins) != int(args.wisig_out_len):
        raise ValueError("fft_bins must equal wisig_out_len so the selected mask fits model input")


def _meta_tensor(meta: Mapping[str, Any], key: str, batch_size: int) -> np.ndarray:
    if key not in meta:
        raise ValueError(f"L_s batch metadata is missing {key}")
    value = meta[key]
    if torch.is_tensor(value):
        result = value.detach().cpu().numpy().reshape(-1)
    else:
        result = np.asarray(value).reshape(-1)
    if result.size != batch_size:
        raise ValueError(f"L_s metadata {key} count mismatch")
    return result.astype(np.int64, copy=False)


def _expand_band_mask(band_mask: np.ndarray, fft_bins: int) -> np.ndarray:
    edges = np.linspace(0, fft_bins, band_mask.size + 1, dtype=np.int64)
    mask = np.zeros(fft_bins, dtype=bool)
    for band, selected in enumerate(band_mask):
        if selected:
            mask[edges[band] : edges[band + 1]] = True
    return mask


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    parser.description = "Audit Phase1 L_s spectral identifiability and freeze an SID mask."
    parser.add_argument("--fft_bins", type=int, default=256)
    parser.add_argument("--num_bands", type=int, default=64)
    parser.add_argument("--keep_fraction", type=float, default=0.50)
    parser.add_argument("--dc_notch", type=int, default=1)
    parser.add_argument("--max_batches", type=int, default=0)
    parser.add_argument("--bootstrap_repeats", type=int, default=64)
    parser.add_argument("--bootstrap_keep_fraction", type=float, default=0.30)
    args = parser.parse_args(argv)
    _validate_protocol(args)
    if int(args.num_bands) <= 0 or int(args.num_bands) > int(args.fft_bins):
        raise ValueError("num_bands must be in [1, fft_bins]")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    expected_outputs = (
        output_dir / "spectral_identifiability.json",
        output_dir / "spectral_identifiability.csv",
        output_dir / "spectral_identifiability.png",
        output_dir / "sid_mask.npz",
        output_dir / "sid_mask_hierarchical.npz",
    )
    if any(path.exists() for path in expected_outputs):
        raise FileExistsError(f"refusing to overwrite P0 output in {output_dir}")

    set_seed(int(args.seed))
    device = torch.device(str(args.device))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but unavailable: {device}")
    data_context = _build_ssdg_wisig_data(args, device)
    split_info = data_context["split_info"]
    if str(split_info.get("source_role_protocol")) != "l_s_u_s_v_cal_v_select":
        raise ValueError("data builder returned a non-current Phase1 source protocol")

    accumulator = SpectralIdentifiabilityAccumulator(num_bands=int(args.num_bands), feature_dim=5)
    generators = {
        scenario: make_torch_generator(device, int(args.seed) + 1009 * (index + 1))
        for index, scenario in enumerate(LEO_WEAK_SCENARIOS)
    }
    sample_views = 0
    batches = 0
    physical_offset = 0
    for batch_index, batch in enumerate(data_context["probe_train_loader"], start=1):
        if int(args.max_batches) > 0 and batch_index > int(args.max_batches):
            break
        x, y, extra = unpack_batch(batch)
        x = x.to(device, non_blocking=True)
        y_np = y.detach().cpu().numpy().reshape(-1).astype(np.int64, copy=False)
        if len(extra) < 2 or not isinstance(extra[1], Mapping):
            raise ValueError("L_s probe batch must expose source metadata")
        meta = extra[1]
        rx = _meta_tensor(meta, "rx_i", y_np.size)
        day = _meta_tensor(meta, "day_i", y_np.size)
        views = (("clean", x),) + tuple(
            (
                scenario,
                apply_sat_channel_for_scenario(
                    x,
                    scenario,
                    args,
                    gen=generators[scenario],
                    return_meta=False,
                )[0],
            )
            for scenario in LEO_WEAK_SCENARIOS
        )
        for view_index, (_, view_iq) in enumerate(views):
            descriptor = extract_band_descriptors(view_iq, int(args.num_bands)).detach().cpu().numpy()
            for row in range(y_np.size):
                accumulator.update(
                    descriptor[row],
                    tx=int(y_np[row]),
                    rx=int(rx[row]),
                    day=int(day[row]),
                    view=view_index,
                    cluster=physical_offset + row,
                )
            sample_views += int(y_np.size)
        physical_offset += int(y_np.size)
        batches += 1

    stats = accumulator.finalize(
        bootstrap_repeats=int(args.bootstrap_repeats),
        bootstrap_keep_fraction=float(args.bootstrap_keep_fraction),
        bootstrap_seed=int(args.seed) + 4049,
    )
    band_mask = select_sid_mask(stats, float(args.keep_fraction), int(args.dc_notch))
    role_band_masks = select_hsid_role_masks(stats, dc_notch=int(args.dc_notch))
    fft_mask = _expand_band_mask(band_mask, int(args.fft_bins))
    role_fft_masks = {
        name: _expand_band_mask(mask, int(args.fft_bins))
        for name, mask in role_band_masks.items()
    }
    if not fft_mask.any():
        raise RuntimeError("P0 selected an empty FFT mask")
    if not role_fft_masks["common_mask"].any():
        raise RuntimeError("P0 selected an empty HSID common-spectrum mask")

    payload = {
        "schema": "phase1_spectral_identifiability_v1",
        "status": "VERIFIED",
        "source_role": "L_s",
        "source_only": True,
        "target_or_query_access": False,
        "seed": int(args.seed),
        "fft_bins": int(args.fft_bins),
        "num_bands": int(args.num_bands),
        "keep_fraction": float(args.keep_fraction),
        "dc_notch": int(args.dc_notch),
        "batches": batches,
        "sample_views": sample_views,
        "physical_samples": physical_offset,
        "bootstrap_repeats": int(args.bootstrap_repeats),
        "bootstrap_keep_fraction": float(args.bootstrap_keep_fraction),
        "views": ["clean", *LEO_WEAK_SCENARIOS],
        "source_split_receipt": split_info.get("source_split_receipt", {}),
        "source_role_counts": {
            "L_s": int(split_info["labeled_size"]),
            "U_s": int(split_info["unlabeled_size"]),
            "V_cal": int(split_info["source_calibration_size"]),
            "V_select": int(split_info["source_selection_size"]),
        },
        "statistics": stats,
        "selected_band_indices": np.flatnonzero(band_mask),
        "selected_fft_indices": np.flatnonzero(fft_mask),
        "hsid_role_band_indices": {
            name: np.flatnonzero(mask) for name, mask in role_band_masks.items()
        },
        "hsid_role_fft_indices": {
            name: np.flatnonzero(mask) for name, mask in role_fft_masks.items()
        },
    }
    _atomic_json(expected_outputs[0], payload)
    _atomic_csv(expected_outputs[1], stats, band_mask)
    center_mask = _torch_mask_to_numpy(
        build_center_mask(
            int(args.fft_bins),
            half_width=max(2, int(args.fft_bins) // 4),
            dc_notch=int(args.dc_notch),
        )
    )
    phase_mask = _torch_mask_to_numpy(
        build_center_mask(
            int(args.fft_bins),
            half_width=max(2, int(args.fft_bins) // 2),
            dc_notch=int(args.dc_notch),
        )
    )
    _atomic_npz(
        expected_outputs[3],
        mask=fft_mask.astype(np.uint8),
        center_mask=center_mask.astype(np.uint8),
        phase_mask=phase_mask.astype(np.uint8),
        band_mask=band_mask.astype(np.uint8),
        j_score=stats["j_score"].astype(np.float32),
        fft_bins=np.asarray([int(args.fft_bins)], dtype=np.int64),
    )
    _atomic_npz(
        expected_outputs[4],
        mask=role_fft_masks["common_mask"].astype(np.uint8),
        common_mask=role_fft_masks["common_mask"].astype(np.uint8),
        nonlinear_mask=role_fft_masks["nonlinear_mask"].astype(np.uint8),
        domain_mask=role_fft_masks["domain_mask"].astype(np.uint8),
        common_band_mask=role_band_masks["common_mask"].astype(np.uint8),
        nonlinear_band_mask=role_band_masks["nonlinear_mask"].astype(np.uint8),
        domain_band_mask=role_band_masks["domain_mask"].astype(np.uint8),
        j_score=stats["j_score"].astype(np.float32),
        nonlinear_score=stats["nonlinear_score"].astype(np.float32),
        domain_score=stats["domain_score"].astype(np.float32),
        bootstrap_selection_probability=stats["bootstrap_selection_probability"].astype(np.float32),
        fft_bins=np.asarray([int(args.fft_bins)], dtype=np.int64),
    )
    _atomic_plot(expected_outputs[2], stats, band_mask)
    print(json.dumps({"status": "VERIFIED", "output_dir": str(output_dir), "sample_views": sample_views}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
