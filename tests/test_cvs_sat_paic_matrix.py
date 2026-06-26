import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "code"
TOOLS = ROOT / "tools"
for path in (CODE, TOOLS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


class CvsSatPaicMatrixTest(unittest.TestCase):
    def test_matrix_contains_central_federated_and_stage2_paic_rows(self):
        from cvsrffi.paic_star_ground import PAIC_CURRICULUM_SCHEDULE, build_paic_matrix

        payload = build_paic_matrix()
        rows = payload["candidates"]
        by_id = {row["candidate_id"]: row for row in rows}

        self.assertEqual(payload["schema"], "cvs_sat_paic_matrix_v1")
        self.assertIn("C2_PAIC_CURRICULUM_CE_ONLY", by_id)
        self.assertIn("C3_PAIC_LATE_WEAK_ALIGN", by_id)
        self.assertIn("F2_FL_PAIC_CURRICULUM", by_id)
        self.assertIn("F3_FL_PAIC_LATE_ALIGN", by_id)
        self.assertIn("S2C_PAIC_PROTOCOL_CHECK", by_id)
        self.assertEqual(by_id["C2_PAIC_CURRICULUM_CE_ONLY"]["sat_view_schedule"], PAIC_CURRICULUM_SCHEDULE)
        self.assertTrue(by_id["C2_PAIC_CURRICULUM_CE_ONLY"]["concat_sat_ce_only"])
        self.assertEqual(by_id["C3_PAIC_LATE_WEAK_ALIGN"]["sat_cons_start_epoch"], 60)
        self.assertEqual(by_id["F2_FL_PAIC_CURRICULUM"]["fl_client_key"], "receiver")
        self.assertEqual(by_id["F2_FL_PAIC_CURRICULUM"]["wisig_train_ratio"], 0.1)
        self.assertEqual(by_id["S2C_PAIC_PROTOCOL_CHECK"]["target_channel_view"], "satellite/LEO")
        self.assertTrue(by_id["S2C_PAIC_PROTOCOL_CHECK"]["unknown_query_eval_only"])
        self.assertEqual(by_id["S2C_PAIC_PROTOCOL_CHECK"]["clean_view_role"], "control_only")

    def test_artifact_writer_emits_json_and_markdown_boundary_report(self):
        from cvs_sat_paic_matrix import write_paic_artifacts
        from optimizer_validate_matrix import expected_count_for_matrix

        with tempfile.TemporaryDirectory() as tmp:
            output = write_paic_artifacts(Path(tmp))
            matrix = json.loads(output["json_path"].read_text(encoding="utf-8"))
            report = output["report_path"].read_text(encoding="utf-8")
            validation = json.loads(output["validation_path"].read_text(encoding="utf-8"))

        self.assertEqual(matrix["schema"], "cvs_sat_paic_matrix_v1")
        self.assertEqual(matrix["expected_count"], len(matrix["candidates"]))
        self.assertEqual(expected_count_for_matrix(matrix, None), len(matrix["candidates"]))
        self.assertEqual(validation["verdict"], "PASS")
        self.assertEqual(validation["expected_count"], len(matrix["candidates"]))
        self.assertIn("CVS-SAT-PAIC", report)
        self.assertIn("--expected-count 14", report)
        self.assertIn("不是真实在轨 IQ 验证", report)
        self.assertIn("C3_PAIC_LATE_WEAK_ALIGN", report)
        self.assertIn("F2_FL_PAIC_CURRICULUM", report)
        self.assertIn("S2C_PAIC_PROTOCOL_CHECK", report)

    def test_optimizer_validator_rejects_paic_rows_missing_required_gates(self):
        from cvsrffi.paic_star_ground import build_paic_matrix
        from optimizer_validate_matrix import validate

        rows = build_paic_matrix()["candidates"]
        bad_fl = dict(next(row for row in rows if row["candidate_id"] == "F2_FL_PAIC_CURRICULUM"))
        bad_fl["fl_baseline_view_ce_only"] = False
        result = validate([bad_fl], expected_count=1)
        issues = {issue["issue"] for issue in result["issues"]}
        self.assertIn("paic_fl_requires_ce_only_baseline_view", issues)

        bad_stage2 = dict(next(row for row in rows if row["candidate_id"] == "S2C_PAIC_PROTOCOL_CHECK"))
        bad_stage2["clean_view_role"] = "metric_primary"
        result = validate([bad_stage2], expected_count=1)
        issues = {issue["issue"] for issue in result["issues"]}
        self.assertIn("paic_clean_view_role_must_be_control_only", issues)

    def test_fl82_launcher_has_explicit_paic_plan(self):
        from cvsrffi.paic_star_ground import PAIC_CURRICULUM_SCHEDULE

        script = (ROOT / "code" / "scripts" / "run_fed_fl82_validation_4gpu.sh").read_text(encoding="utf-8")

        self.assertIn("PAIC", script)
        self.assertIn("F2_FL_PAIC_CURRICULUM", script)
        self.assertIn("F3_FL_PAIC_LATE_ALIGN", script)
        self.assertIn(PAIC_CURRICULUM_SCHEDULE, script)

    def test_paic_federated_rows_do_not_rely_on_comment_sidecar_for_validation(self):
        from cvsrffi.paic_star_ground import build_paic_matrix
        from optimizer_validate_matrix import validate

        rows = build_paic_matrix()["candidates"]
        fed_rows = [row for row in rows if row["paic_matrix_group"] == "federated"]

        self.assertTrue(all("#" not in row["exact_command"] for row in fed_rows))
        result = validate(fed_rows, expected_count=len(fed_rows))
        self.assertEqual(result["verdict"], "PASS")

    def test_paic_rows_are_shell_safe_and_not_n607_launchable_by_status(self):
        from cvsrffi.paic_star_ground import PAIC_CURRICULUM_SCHEDULE, build_paic_matrix
        from optimizer_validate_matrix import is_launchable_status

        rows = build_paic_matrix()["candidates"]
        central_schedule_rows = [
            row for row in rows
            if row["paic_matrix_group"] == "central" and row.get("sat_view_schedule")
        ]
        self.assertTrue(central_schedule_rows)
        for row in central_schedule_rows:
            self.assertIn(f"--sat_view_schedule '{PAIC_CURRICULUM_SCHEDULE}'", row["exact_command"])

        self.assertTrue(all(row.get("n607_launch_allowed") is False for row in rows))
        self.assertTrue(all(not is_launchable_status(row["launchability_status"]) for row in rows))

    def test_stage2_a_b_target_new_query_is_rejection_eval_only(self):
        from cvsrffi.paic_star_ground import build_paic_matrix

        rows = build_paic_matrix()["candidates"]
        stage_rows = {row["candidate_id"]: row for row in rows if row["paic_matrix_group"] == "stage2"}

        self.assertEqual(stage_rows["S2A_PAIC_PROTOCOL_CHECK"]["target_new_query_role"], "reject_eval_only_not_seen_new_identity")
        self.assertEqual(stage_rows["S2B_PAIC_PROTOCOL_CHECK"]["target_new_query_role"], "reject_eval_only_not_seen_new_identity")
        self.assertEqual(stage_rows["S2C_PAIC_PROTOCOL_CHECK"]["target_new_query_role"], "seen_new_identity_eval")


if __name__ == "__main__":
    unittest.main()
