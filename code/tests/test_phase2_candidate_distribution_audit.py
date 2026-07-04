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


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


class Phase2CandidateDistributionAuditTest(unittest.TestCase):
    def test_audit_summarizes_role_and_top_label_set_distribution(self):
        from phase2_candidate_distribution_audit import audit_evidence_rows

        rows = [
            {
                "event_id": "seen|new-a|s|rank00000",
                "role": "seen_new",
                "true_label": "new-a",
                "receiver_id": "rx-a",
                "top_label": "old-a",
                "top_label_set": "old",
                "known_score": "0.80",
                "unknown_score": "0.10",
                "margin": "0.03",
                "old_anchor_score": "0.91",
            },
            {
                "event_id": "seen|new-a|s|rank00000",
                "role": "seen_new",
                "true_label": "new-a",
                "receiver_id": "rx-b",
                "top_label": "new-a",
                "top_label_set": "seen_new",
                "known_score": "0.70",
                "unknown_score": "0.20",
                "margin": "0.08",
                "old_anchor_score": "0.05",
            },
            {
                "event_id": "unk|unk-a|s|rank00000",
                "role": "unknown",
                "true_label": "unk-a",
                "receiver_id": "rx-a",
                "top_label": "old-a",
                "top_label_set": "old",
                "known_score": "0.72",
                "unknown_score": "0.22",
                "margin": "0.01",
                "old_anchor_score": "0.86",
            },
        ]

        result = audit_evidence_rows(rows, algorithm="unit")

        seen = result["role_summary"]["seen_new"]
        self.assertEqual(seen["row_count"], 2)
        self.assertEqual(seen["top_label_set_counts"]["old"], 1)
        self.assertEqual(seen["top_label_set_counts"]["seen_new"], 1)
        self.assertEqual(seen["top_label_match_rate"], 0.5)
        self.assertAlmostEqual(seen["known_score_mean"], 0.75)

        unknown = result["role_summary"]["unknown"]
        self.assertEqual(unknown["top_label_set_counts"]["old"], 1)
        self.assertEqual(unknown["unknown_score_mean"], 0.22)

    def test_main_writes_json_and_csv_outputs(self):
        from phase2_candidate_distribution_audit import main

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            evidence = root / "evidence.csv"
            output_json = root / "audit.json"
            output_csv = root / "audit_by_role_label.csv"
            _write_csv(
                evidence,
                [
                    {
                        "event_id": "old|old-a|s|rank00000",
                        "role": "old",
                        "true_label": "old-a",
                        "receiver_id": "rx-a",
                        "top_label": "old-a",
                        "top_label_set": "old",
                        "known_score": "0.90",
                        "unknown_score": "0.10",
                        "margin": "0.20",
                        "old_anchor_score": "0.95",
                    },
                    {
                        "event_id": "seen|new-a|s|rank00000",
                        "role": "seen_new",
                        "true_label": "new-a",
                        "receiver_id": "rx-a",
                        "top_label": "old-a",
                        "top_label_set": "old",
                        "known_score": "0.70",
                        "unknown_score": "0.15",
                        "margin": "0.02",
                        "old_anchor_score": "0.80",
                    },
                ],
            )

            rc = main(
                [
                    "--evidence_csv",
                    str(evidence),
                    "--output_json",
                    str(output_json),
                    "--output_by_role_label_csv",
                    str(output_csv),
                    "--algorithm",
                    "unit",
                ]
            )

            data = json.loads(output_json.read_text(encoding="utf-8"))
            with output_csv.open("r", encoding="utf-8") as handle:
                by_role_rows = list(csv.DictReader(handle))

        self.assertEqual(rc, 0)
        self.assertEqual(data["algorithm"], "unit")
        self.assertIn("seen_new", data["role_summary"])
        self.assertEqual(len(by_role_rows), 2)
        self.assertIn("top_label_set_counts_json", by_role_rows[0])


if __name__ == "__main__":
    unittest.main()
