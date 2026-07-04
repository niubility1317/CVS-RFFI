import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))


class Phase2OrbitC3RFailureAuditTest(unittest.TestCase):
    def test_extracts_floor_failures_and_confusion(self):
        from phase2_orbit_c3r_failure_audit import main

        payload = {
            "algorithm": "ORBIT-C3R Guard",
            "feature_npz": "features.npz",
            "target_gates": {"old_acc": 0.99},
            "profile_results": {
                "old_preserving": {
                    "counts": {
                        "2": {
                            "per_old_class_acc": {"old-a": 1.0, "old-b": 0.0},
                            "per_old_class_total": {"old-a": 3, "old-b": 2, "old-empty": 0},
                            "per_old_class_decision_counts": {
                                "old-a": {"accept": 3},
                                "old-b": {"unknown_reject": 2},
                            },
                            "per_old_class_output_counts": {"old-a": {"old-a": 3}},
                            "per_seen_new_class_acc": {"new-a": 0.0},
                            "per_seen_new_class_total": {"new-a": 2},
                            "per_seen_new_class_decision_counts": {"new-a": {"defer": 2}},
                            "per_seen_new_class_output_counts": {},
                            "open_set_confusion": {"old->old": 3, "old->reject": 2},
                        }
                    }
                }
            },
        }
        with tempfile.TemporaryDirectory() as td:
            input_json = Path(td) / "in.json"
            output_json = Path(td) / "out.json"
            class_csv = Path(td) / "classes.csv"
            confusion_csv = Path(td) / "confusion.csv"
            input_json.write_text(json.dumps(payload), encoding="utf-8")
            rc = main(
                [
                    "--input_json",
                    str(input_json),
                    "--output_json",
                    str(output_json),
                    "--output_class_csv",
                    str(class_csv),
                    "--output_confusion_csv",
                    str(confusion_csv),
                ]
            )
            result = json.loads(output_json.read_text(encoding="utf-8"))
            class_csv_exists = class_csv.exists()
            confusion_csv_exists = confusion_csv.exists()

        self.assertEqual(rc, 0)
        self.assertEqual(result["floor_failure_count"], 2)
        self.assertEqual(result["no_event_count"], 1)
        self.assertTrue(any(row["label"] == "old-b" for row in result["floor_failures"]))
        self.assertFalse(any(row["label"] == "old-empty" for row in result["floor_failures"]))
        self.assertTrue(class_csv_exists)
        self.assertTrue(confusion_csv_exists)

    def test_schema_error_returns_nonzero_without_allow_empty(self):
        from phase2_orbit_c3r_failure_audit import main

        with tempfile.TemporaryDirectory() as td:
            input_json = Path(td) / "in.json"
            output_json = Path(td) / "out.json"
            input_json.write_text(json.dumps({"algorithm": "bad"}), encoding="utf-8")
            rc = main(["--input_json", str(input_json), "--output_json", str(output_json)])
            result = json.loads(output_json.read_text(encoding="utf-8"))

        self.assertEqual(rc, 2)
        self.assertIn("missing_or_empty_profile_results", result["schema_errors"])

    def test_parses_decimal_count_and_records_invalid_acc(self):
        from phase2_orbit_c3r_failure_audit import main

        payload = {
            "profile_results": {
                "p": {
                    "counts": {
                        "1.0": {
                            "per_old_class_acc": {"a": "nan"},
                            "per_old_class_total": {"a": "1.0"},
                            "per_old_class_decision_counts": {"a": {"accept": "1.0"}},
                        }
                    }
                }
            }
        }
        with tempfile.TemporaryDirectory() as td:
            input_json = Path(td) / "in.json"
            output_json = Path(td) / "out.json"
            input_json.write_text(json.dumps(payload), encoding="utf-8")
            rc = main(["--input_json", str(input_json), "--output_json", str(output_json)])
            result = json.loads(output_json.read_text(encoding="utf-8"))

        self.assertEqual(rc, 0)
        self.assertEqual(result["class_rows"][0]["collab_count"], 1)
        self.assertEqual(result["class_rows"][0]["class_total"], 1)
        self.assertEqual(result["class_rows"][0]["decision_accept"], 1)
        self.assertEqual(result["class_rows"][0]["acc_status"], "invalid")
        self.assertEqual(result["floor_failure_count"], 0)


if __name__ == "__main__":
    unittest.main()
