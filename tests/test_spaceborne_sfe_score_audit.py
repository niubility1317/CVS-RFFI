import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))


def write_score_table(path: Path) -> None:
    fieldnames = [
        "row",
        "true_label",
        "true_group",
        "query_tx_id",
        "predicted_label",
        "accepted",
        "score",
        "unknown_score_kind",
        "unknown_score",
        "cosine_unknown_score",
        "margin",
        "mahalanobis",
        "openmax_distance",
        "seen_new_evidence",
        "seen_new_support_affinity",
        "seen_new_support_residual",
        "seen_new_anchor_similarity",
        "seen_new_anchor_delta",
        "gate_reason",
    ]
    rows = [
        (0, 0, "old", "old0", 0, True, "accepted", 0.10, 0.20, 0.30, 0.20, -0.70),
        (1, 1, "old", "old1", -1, False, "low_cosine", 0.20, 0.30, 0.40, 0.30, -0.60),
        (2, 4, "new", "new0", 0, True, "accepted", 0.70, 0.80, 0.10, 0.95, 0.05),
        (3, 4, "new", "new0", -1, False, "high_mahalanobis", 0.60, 0.75, 0.20, 0.90, 0.00),
        (4, 4, "new", "new1", 4, True, "accepted", 0.80, 0.90, 0.10, 0.85, -0.05),
        (5, -1, "unknown", "unk0", -1, False, "unknown_rejected", 0.10, 0.20, 0.70, 0.40, -0.50),
        (6, -1, "unknown", "unk1", 0, True, "accepted", 0.30, 0.40, 0.50, 0.55, -0.35),
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row, true_label, group, tx_id, pred, accepted, reason, evidence, affinity, residual, anchor, anchor_delta in rows:
            writer.writerow(
                {
                    "row": row,
                    "true_label": true_label,
                    "true_group": group,
                    "query_tx_id": tx_id,
                    "predicted_label": pred,
                    "accepted": str(accepted),
                    "score": "0.8",
                    "unknown_score_kind": "mahalanobis_unknown_score",
                    "unknown_score": "0.2",
                    "cosine_unknown_score": "0.1",
                    "margin": "0.5",
                    "mahalanobis": "1.0",
                    "openmax_distance": "0.0",
                    "seen_new_evidence": evidence,
                    "seen_new_support_affinity": affinity,
                    "seen_new_support_residual": residual,
                    "seen_new_anchor_similarity": anchor,
                    "seen_new_anchor_delta": anchor_delta,
                    "gate_reason": reason,
                }
            )


class SpaceborneSfeScoreAuditTest(unittest.TestCase):
    def test_summarizes_group_confusion_gate_and_tx_failures(self):
        from spaceborne_sfe_score_audit import summarize_score_tables

        with tempfile.TemporaryDirectory() as td:
            score_table = Path(td) / "score_table_mahal.csv"
            write_score_table(score_table)

            summary = summarize_score_tables([score_table])

        self.assertEqual(summary["overall"]["rows"], 7)
        self.assertEqual(summary["overall"]["known_rows"], 5)
        self.assertEqual(summary["overall"]["unknown_rows"], 2)
        self.assertAlmostEqual(summary["overall"]["old_accuracy"], 0.5)
        self.assertAlmostEqual(summary["overall"]["new_accuracy"], 1 / 3)
        self.assertAlmostEqual(summary["overall"]["unknown_rejection_rate"], 0.5)
        self.assertEqual(summary["overall"]["new_to_old_count"], 1)
        self.assertEqual(summary["overall"]["new_rejected_count"], 1)
        self.assertEqual(summary["overall"]["unknown_false_accept_count"], 1)

        confusion = {
            (row["true_group"], row["outcome"]): row["count"]
            for row in summary["confusion_summary"]
        }
        self.assertEqual(confusion[("new", "new_to_old")], 1)
        self.assertEqual(confusion[("new", "known_rejected")], 1)
        self.assertEqual(confusion[("unknown", "unknown_false_accept")], 1)

        gate_reasons = {
            (row["true_group"], row["gate_reason"]): row["count"]
            for row in summary["gate_reason_summary"]
        }
        self.assertEqual(gate_reasons[("new", "high_mahalanobis")], 1)

        per_tx = {(row["true_group"], row["query_tx_id"]): row for row in summary["per_tx_summary"]}
        self.assertEqual(per_tx[("new", "new0")]["count"], 2)
        self.assertEqual(per_tx[("new", "new0")]["correct"], 0)
        self.assertEqual(per_tx[("new", "new0")]["rejected"], 1)

        evidence = {row["true_group"]: row for row in summary["seen_new_evidence_summary"]}
        self.assertAlmostEqual(evidence["new"]["seen_new_evidence_mean"], 0.70)
        self.assertAlmostEqual(evidence["unknown"]["seen_new_evidence_mean"], 0.20)
        self.assertAlmostEqual(evidence["new"]["seen_new_anchor_similarity_mean"], 0.90)
        self.assertAlmostEqual(evidence["unknown"]["seen_new_anchor_delta_mean"], -0.425)

    def test_cli_writes_json_and_group_csv(self):
        from spaceborne_sfe_score_audit import main

        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            score_table = tmp / "score_table_combined.csv"
            out_json = tmp / "summary.json"
            out_csv = tmp / "group_summary.csv"
            write_score_table(score_table)

            self.assertEqual(
                main(
                    [
                        "--score_table",
                        str(score_table),
                        "--out_json",
                        str(out_json),
                        "--out_group_csv",
                        str(out_csv),
                    ]
                ),
                0,
            )

            payload = json.loads(out_json.read_text(encoding="utf-8"))
            self.assertEqual(payload["overall"]["rows"], 7)

            with out_csv.open("r", encoding="utf-8", newline="") as f:
                group_rows = list(csv.DictReader(f))
            self.assertEqual({row["true_group"] for row in group_rows}, {"old", "new", "unknown"})


if __name__ == "__main__":
    unittest.main()
