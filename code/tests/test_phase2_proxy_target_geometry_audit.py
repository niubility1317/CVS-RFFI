import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
for path in (str(ROOT), str(SCRIPTS)):
    if path not in sys.path:
        sys.path.insert(0, path)


def _rows(center, n):
    center = np.asarray(center, dtype=np.float32)
    return np.repeat(center[None, :], n, axis=0)


class ProxyTargetGeometryAuditTest(unittest.TestCase):
    def test_audit_reports_proxy_and_target_unknown_separately(self):
        from phase2_proxy_target_geometry_audit import parse_args, run_audit

        features = []
        roles = []
        tx_ids = []
        rx_ids = []
        for role, tx, rx, center, n in [
            ("target_old", "old-a", "rx-a", [1.0, 0.0], 24),
            ("target_new", "new-a", "rx-a", [0.0, 1.0], 24),
            ("target_unknown", "unk-a", "rx-a", [1.0, 0.0], 12),
            ("proxy_unknown", "proxy-a", "src-a", [-1.0, 0.0], 10),
            ("source", "old-a", "src-a", [1.0, 0.0], 10),
        ]:
            arr = _rows(center, n)
            features.append(arr)
            roles.extend([role] * n)
            tx_ids.extend([tx] * n)
            rx_ids.extend([rx] * n)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "features.npz"
            out_json = Path(tmp) / "audit.json"
            np.savez_compressed(
                path,
                features=np.vstack(features).astype(np.float32),
                dataset_role=np.asarray(roles),
                tx_ids=np.asarray(tx_ids),
                rx_ids=np.asarray(rx_ids),
                day_ids=np.asarray(["d0"] * len(roles)),
                sig_ids=np.asarray([str(i % 24) for i in range(len(roles))]),
                manifest_json=np.asarray("{}"),
            )
            args = parse_args(
                [
                    "--feature_npz",
                    str(path),
                    "--output_json",
                    str(out_json),
                    "--k_shot",
                    "4",
                    "--query_per_class",
                    "4",
                    "--seed",
                    "1",
                ]
            )
            result = run_audit(args)

        self.assertFalse(result["support_threshold_uses_target_unknown"])
        self.assertIn("proxy_unknown", result["groups"])
        self.assertIn("target_unknown_query", result["groups"])
        self.assertGreater(result["groups"]["target_unknown_query"]["accept_rate_at_support_threshold"], 0.0)
        self.assertEqual(result["groups"]["proxy_unknown"]["accept_rate_at_support_threshold"], 0.0)


if __name__ == "__main__":
    unittest.main()
