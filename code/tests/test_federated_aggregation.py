import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from federated.fed_aggregate import aggregate_state_dicts, domain_balanced_aggregation_weights, resolve_exclude_keys


class FederatedAggregationTest(unittest.TestCase):
    def test_aggregation_skips_explicit_excluded_keys_and_averages_float_tensors(self):
        client_states = {
            "a": {
                "encoder.weight": torch.full((1, 1, 1), 1.0),
                "local_buffer": torch.tensor([10.0]),
                "counter": torch.tensor(3, dtype=torch.long),
            },
            "b": {
                "encoder.weight": torch.full((1, 1, 1), 3.0),
                "local_buffer": torch.tensor([100.0]),
                "counter": torch.tensor(8, dtype=torch.long),
            },
        }

        out = aggregate_state_dicts(
            client_states,
            {"a": 1, "b": 1},
            exclude_keys={"local_buffer"},
            agg_weight="num_samples",
        )

        self.assertTrue(torch.equal(out["encoder.weight"], torch.full((1, 1, 1), 2.0)))
        self.assertNotIn("local_buffer", out)
        self.assertTrue(torch.equal(out["counter"], torch.tensor(3, dtype=torch.long)))

    def test_num_sample_weighting_uses_client_sizes(self):
        client_states = {
            "small": {"w": torch.tensor([0.0])},
            "large": {"w": torch.tensor([10.0])},
        }

        out = aggregate_state_dicts(client_states, {"small": 1, "large": 3}, exclude_keys=set())

        self.assertTrue(torch.allclose(out["w"], torch.tensor([7.5])))

    def test_explicit_client_weights_override_sample_counts_for_server_regularizers(self):
        client_states = {
            "small": {"w": torch.tensor([0.0])},
            "large": {"w": torch.tensor([10.0])},
        }

        out = aggregate_state_dicts(
            client_states,
            {"small": 1, "large": 99},
            exclude_keys=set(),
            client_weights={"small": 0.8, "large": 0.2},
        )

        self.assertTrue(torch.allclose(out["w"], torch.tensor([2.0])))

    def test_resolve_exclude_keys_supports_exact_keys_and_prefixes(self):
        state = {
            "encoder.weight": torch.tensor([1.0]),
            "local_adapter.weight": torch.tensor([2.0]),
            "dom_head.bias": torch.tensor([3.0]),
        }

        excluded = resolve_exclude_keys(
            state,
            exact_keys={"dom_head.bias"},
            prefixes=("local_adapter.",),
        )

        self.assertEqual(excluded, {"local_adapter.weight", "dom_head.bias"})

    def test_domain_balanced_aggregation_weights_equalize_domains(self):
        weights = domain_balanced_aggregation_weights(
            ["rx0_a", "rx0_b", "rx1"],
            {"rx0_a": "rx0", "rx0_b": "rx0", "rx1": "rx1"},
        )

        self.assertAlmostEqual(weights["rx0_a"], 0.25)
        self.assertAlmostEqual(weights["rx0_b"], 0.25)
        self.assertAlmostEqual(weights["rx1"], 0.5)
        self.assertAlmostEqual(sum(weights.values()), 1.0)


if __name__ == "__main__":
    unittest.main()
