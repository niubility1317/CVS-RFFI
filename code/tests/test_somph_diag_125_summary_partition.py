import unittest


class SomphDiag125SummaryPartitionTest(unittest.TestCase):
    def setUp(self):
        from scripts import summarize_cvs_somph_diag_125_stability as summary

        self.summary = summary
        self.scenarios = tuple(summary.FORMAL_LEO_WEAK_SCENARIOS)

    def test_accepts_disjoint_scenario_union(self):
        by_scenario = {
            self.scenarios[0]: {"a", "b"},
            self.scenarios[1]: {"c"},
            self.scenarios[2]: {"d", "e"},
        }

        self.summary._require_exact_scenario_partition(
            by_scenario, {"a", "b", "c", "d", "e"}, context="row:after"
        )

    def test_rejects_scenario_overlap(self):
        by_scenario = {
            self.scenarios[0]: {"a", "b"},
            self.scenarios[1]: {"b", "c"},
            self.scenarios[2]: {"d"},
        }

        with self.assertRaises(self.summary.StabilitySummaryError):
            self.summary._require_exact_scenario_partition(
                by_scenario, {"a", "b", "c", "d"}, context="row:after"
            )

    def test_rejects_incomplete_union(self):
        by_scenario = {
            self.scenarios[0]: {"a"},
            self.scenarios[1]: {"b"},
            self.scenarios[2]: {"c"},
        }

        with self.assertRaises(self.summary.StabilitySummaryError):
            self.summary._require_exact_scenario_partition(
                by_scenario, {"a", "b", "c", "d"}, context="row:after"
            )

    def test_accepts_explicit_diagnostic_preopen_state(self):
        self.assertTrue(
            self.summary._preopen_status_is_coherent(
                {
                    "status": "UNVERIFIED_UNDER_CURRENT_PROTOCOL_DIAGNOSTIC_ONLY",
                    "diagnostic_only": True,
                    "formal_launch_authority": False,
                    "formal_metric_claim_allowed": False,
                }
            )
        )

    def test_rejects_incoherent_diagnostic_preopen_state(self):
        self.assertFalse(
            self.summary._preopen_status_is_coherent(
                {
                    "status": "UNVERIFIED_UNDER_CURRENT_PROTOCOL_DIAGNOSTIC_ONLY",
                    "diagnostic_only": True,
                    "formal_launch_authority": True,
                    "formal_metric_claim_allowed": False,
                }
            )
        )


if __name__ == "__main__":
    unittest.main()
