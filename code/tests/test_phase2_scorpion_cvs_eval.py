import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
for path in (str(ROOT), str(SCRIPTS)):
    if path not in sys.path:
        sys.path.insert(0, path)


def _row(event_id, role, true_label, pred, rx, risk, score=0.9, margin=0.4, pvalue=1.0):
    return {
        "event_id": event_id,
        "role": role,
        "true_label": true_label,
        "predicted_label": pred,
        "receiver_id": rx,
        "mahalanobis_risk": risk,
        "evt_risk": risk,
        "margin_risk": risk,
        "oldness_risk": risk,
        "score_risk": risk,
        "known_score": score,
        "known_margin": margin,
        "label_score_gap": margin,
        "effective_score_threshold": 0.1,
        "class_evidence_top1_conformal_pvalue": pvalue,
        "reliability": 1.0,
        "receiver_class_reliability": 1.0,
        "receiver_deployment_prior": 1.0,
        "support_density": 1.0,
        "bytes": 40.0,
        "latency_ms": 0.2,
    }


class Phase2ScorpionCvsEvalTest(unittest.TestCase):
    def test_rejects_unknown_and_reports_all_receiver_counts(self):
        from phase2_scorpion_cvs_eval import evaluate_scorpion, _parse_weighted_components

        rows = [
            _row("old-1", "old", "old-a", "old-a", "rx-a", 0.1),
            _row("old-1", "old", "old-a", "old-a", "rx-b", 0.1),
            _row("new-1", "seen_new", "new-a", "new-a", "rx-a", 0.1),
            _row("new-1", "seen_new", "new-a", "new-a", "rx-b", 0.1),
            _row("unk-1", "unknown", "__unknown__", "old-a", "rx-a", 0.95),
            _row("unk-1", "unknown", "__unknown__", "new-a", "rx-b", 0.95),
        ]

        result = evaluate_scorpion(
            rows,
            risk_components=_parse_weighted_components("mahalanobis_risk:1"),
            unknown_gate=0.5,
            old_shield_gate=0.7,
        )

        self.assertEqual(set(result["counts"]), {"1", "2"})
        self.assertEqual(result["counts"]["2"]["old_acc"], 1.0)
        self.assertEqual(result["counts"]["2"]["seen_new_acc"], 1.0)
        self.assertEqual(result["counts"]["2"]["unknown_reject_rate"], 1.0)
        self.assertEqual(result["counts"]["2"]["bytes_per_event_mean"], 80.0)

    def test_old_retention_shield_accepts_old_without_using_unknown_fit(self):
        from phase2_scorpion_cvs_eval import evaluate_scorpion, _parse_weighted_components

        rows = [
            _row("old-1", "old", "old-a", "old-a", "rx-a", 0.60, score=0.95, margin=0.5),
            _row("old-1", "old", "old-a", "old-a", "rx-b", 0.60, score=0.95, margin=0.5),
            _row("unk-1", "unknown", "__unknown__", "old-a", "rx-a", 0.95),
            _row("unk-1", "unknown", "__unknown__", "old-a", "rx-b", 0.95),
        ]

        result = evaluate_scorpion(
            rows,
            risk_components=_parse_weighted_components("mahalanobis_risk:1"),
            unknown_gate=0.5,
            old_shield_gate=0.7,
        )

        old_row = [row for row in result["rows"] if row["event_id"] == "old-1" and row["receiver_count"] == 2][0]
        self.assertTrue(old_row["old_shield"])
        self.assertFalse(old_row["reject"])
        self.assertFalse(result["unknown_query_used_for_threshold_fit"])

    def test_cli_writes_json_and_rows_csv(self):
        from phase2_scorpion_cvs_eval import main

        rows = [
            _row("old-1", "old", "old-a", "old-a", "rx-a", 0.1),
            _row("unk-1", "unknown", "__unknown__", "old-a", "rx-a", 0.95),
        ]
        with tempfile.TemporaryDirectory() as td:
            evidence = Path(td) / "evidence.csv"
            output_json = Path(td) / "out.json"
            output_csv = Path(td) / "rows.csv"
            with evidence.open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)

            self.assertEqual(
                main([
                    "--evidence_csv",
                    str(evidence),
                    "--output_json",
                    str(output_json),
                    "--output_rows_csv",
                    str(output_csv),
                ]),
                0,
            )

            payload = json.loads(output_json.read_text(encoding="utf-8"))
            self.assertEqual(payload["receiver_count"], 1)
            self.assertTrue(output_csv.exists())


if __name__ == "__main__":
    unittest.main()
