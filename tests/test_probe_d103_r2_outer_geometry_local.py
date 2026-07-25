from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import numpy as np


def test_outer_geometry_probe_module_imports_component_dependencies() -> None:
    script_root = Path(__file__).resolve().parents[1] / "code" / "scripts"
    sys.path.insert(0, str(script_root))
    try:
        import probe_d103_r2_outer_geometry_local as probe

        assert callable(probe.frozen_qknn)
        assert callable(probe._int8_component_audit)
        rows = np.zeros((2, 160), dtype=np.float32)
        rows[0, :3] = np.asarray([1.0, 0.2, -0.1], dtype=np.float32)
        rows[1, :4] = np.asarray(
            [0.7, -0.4, 0.2, 0.1],
            dtype=np.float32,
        )
        decoded, factors, cosines = probe._angular_grid_decode(rows)
        assert decoded.shape == rows.shape
        assert factors.shape == (2,)
        assert cosines.shape == (2,)
        assert np.allclose(np.linalg.norm(decoded, axis=1), 1.0)
        assert np.all((factors >= 0.75) & (factors <= 1.25))
        assert np.all(cosines > 0.999)
        _, _, legacy = probe._quantize_rows(
            probe.normalize_zid_rows(rows)
        )
        normalized = probe.normalize_zid_rows(rows).astype(np.float64)
        legacy_cosines = np.sum(
            normalized * legacy.astype(np.float64),
            axis=1,
        )
        assert np.all(cosines >= legacy_cosines - 1.0e-12)
        normalized_rows = probe.normalize_zid_rows(rows)
        legacy_codes, legacy_scales, legacy_decoded = probe._quantize_rows(
            normalized_rows
        )
        for index, normalized_row in enumerate(normalized_rows):
            scale16, codes, c1_decoded = probe._quantize_row_at_factor(
                normalized_row,
                1.0,
            )
            assert scale16.tobytes() == legacy_scales[index].tobytes()
            assert np.array_equal(codes, legacy_codes[index])
            assert np.array_equal(c1_decoded, legacy_decoded[index])
            assert not np.any(codes == np.int8(-128))
        basis = np.zeros((1, 160), dtype=np.float32)
        basis[0, 0] = 1.0
        _, basis_factor, _ = probe._angular_grid_decode(basis)
        assert basis_factor.tolist() == [0.75]
        for bad_row, bad_factor in (
            (np.zeros(160, dtype=np.float32), 1.0),
            (np.full(160, np.nan, dtype=np.float32), 1.0),
            (normalized_rows[0], 0.0),
        ):
            try:
                probe._quantize_row_at_factor(bad_row, bad_factor)
            except ValueError:
                pass
            else:
                raise AssertionError("invalid ANGQ input did not fail closed")
        tie_audit = probe._top1_and_margin_flips(
            np.asarray([[2.0, 2.0, 1.0]], dtype=np.float64),
            np.asarray([[3.0, 2.0, 1.0]], dtype=np.float64),
        )
        assert tie_audit == {
            "top1_agreement": 1.0,
            "teacher_winner_margin_flip_count": 0,
        }
    finally:
        sys.path.remove(str(script_root))


def test_outer_geometry_probe_cli_is_truth_free() -> None:
    script = (
        Path(__file__).resolve().parents[1]
        / "code"
        / "scripts"
        / "probe_d103_r2_outer_geometry_local.py"
    )
    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "--tap-archive" in result.stdout
    assert "--dual-archive" in result.stdout
    assert "--held-receiver" in result.stdout
    assert "--k-values" in result.stdout
    assert "truth" not in result.stdout.lower()
