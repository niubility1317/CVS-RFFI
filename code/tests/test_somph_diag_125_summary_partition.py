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


if __name__ == "__main__":
    unittest.main()
