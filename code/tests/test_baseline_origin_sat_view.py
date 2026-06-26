import unittest
from types import SimpleNamespace

import torch


class BaselineOriginSatViewAugmentTest(unittest.TestCase):
    def test_schedule_parser_supports_probabilities_and_repeated_scenarios(self):
        from baseline_origin_sat_view import parse_sat_view_schedule

        stages = parse_sat_view_schedule(
            "1@1.0:mixed_orbit;"
            "61@0.75:mixed_orbit*2,low_elev_leo,rain_leo;"
            "121:mixed_orbit,rain_leo,storm_mp",
            default_prob=0.5,
        )

        self.assertEqual([stage.start_epoch for stage in stages], [1, 61, 121])
        self.assertEqual([stage.view_prob for stage in stages], [1.0, 0.75, 0.5])
        self.assertEqual(stages[0].scenarios, ("mixed_orbit",))
        self.assertEqual(stages[1].scenarios, ("mixed_orbit", "mixed_orbit", "low_elev_leo", "rain_leo"))
        self.assertEqual(stages[2].scenarios, ("mixed_orbit", "rain_leo", "storm_mp"))

    def test_schedule_parser_rejects_ambiguous_schedule_specs(self):
        from baseline_origin_sat_view import parse_sat_view_schedule

        invalid_specs = [
            "20:mixed_orbit",
            "1:mixed_orbit;1:rain_leo",
            "1:mixed_orbit*0",
            "1@1.5:mixed_orbit",
            "1@-0.1:mixed_orbit",
            "1@nan:mixed_orbit",
            "1@inf:mixed_orbit",
        ]

        for spec in invalid_specs:
            with self.subTest(spec=spec):
                with self.assertRaises(ValueError):
                    parse_sat_view_schedule(spec)

    def test_augment_rejects_invalid_default_probability(self):
        from baseline_origin_sat_view import BaselineOriginSatViewAugment

        def fake_apply(x, scenario, args, gen=None, return_meta=False):
            return x, {"scenario": scenario}

        for prob in [float("nan"), float("inf"), -0.1, 1.1]:
            with self.subTest(prob=prob):
                with self.assertRaises(ValueError):
                    BaselineOriginSatViewAugment(
                        scenarios=["mixed_orbit"],
                        p=prob,
                        seed=123,
                        apply_fn=fake_apply,
                    )

    def test_expand_doubles_batch_and_uses_epoch_schedule(self):
        from baseline_origin_sat_view import BaselineOriginSatViewAugment

        calls = []

        def fake_apply(x, scenario, args, gen=None, return_meta=False):
            calls.append((scenario, gen is not None))
            return x + 10.0, {"scenario": scenario}

        aug = BaselineOriginSatViewAugment(
            schedule="1:mixed_orbit;3:storm_mp",
            p=1.0,
            seed=123,
            apply_fn=fake_apply,
        )
        x = torch.arange(12, dtype=torch.float32).view(2, 2, 3)
        y = torch.tensor([1, 2])
        d_raw = torch.tensor([7, 8])

        early = aug.expand(x, y, d_raw, args=SimpleNamespace(), epoch=1, batch_idx=0)
        late = aug.expand(x, y, d_raw, args=SimpleNamespace(), epoch=3, batch_idx=0)

        self.assertEqual(early.scenario, "mixed_orbit")
        self.assertEqual(early.stage_start_epoch, 1)
        self.assertEqual(late.scenario, "storm_mp")
        self.assertEqual(late.stage_start_epoch, 3)
        self.assertTrue(late.applied)
        self.assertTrue(torch.equal(late.x[:2], x))
        self.assertTrue(torch.equal(late.x[2:], x + 10.0))
        self.assertTrue(torch.equal(late.y, torch.tensor([1, 2, 1, 2])))
        self.assertTrue(torch.equal(late.d_raw, torch.tensor([7, 8, 7, 8])))
        self.assertEqual(late.clean_batch_size, 2)
        self.assertEqual(late.total_batch_size, 4)
        self.assertEqual(calls, [("mixed_orbit", True), ("storm_mp", True)])

    def test_transform_returns_clean_duplicate_when_probability_skips(self):
        from baseline_origin_sat_view import BaselineOriginSatViewAugment

        def fake_apply(x, scenario, args, gen=None, return_meta=False):
            raise AssertionError("apply_fn should not be called when p=0")

        aug = BaselineOriginSatViewAugment(
            scenarios=["mixed_orbit"],
            p=0.0,
            seed=123,
            apply_fn=fake_apply,
        )
        x = torch.arange(8, dtype=torch.float32).view(2, 2, 2)

        out = aug.transform(x, args=SimpleNamespace(), epoch=1, batch_idx=0)

        self.assertFalse(out.applied)
        self.assertEqual(out.scenario, "clean_duplicate")
        self.assertEqual(out.clean_batch_size, 2)
        self.assertTrue(torch.equal(out.x, x))
        self.assertNotEqual(out.x.data_ptr(), x.data_ptr())

    def test_transform_calls_real_sat_channel_adapter(self):
        from baseline_origin_sat_view import BaselineOriginSatViewAugment
        from cvsrffi.eval import apply_sat_channel_for_scenario

        aug = BaselineOriginSatViewAugment(
            schedule="1:clear_leo",
            p=1.0,
            seed=123,
            apply_fn=apply_sat_channel_for_scenario,
        )
        x = torch.ones(2, 2, 32, dtype=torch.float32)

        out = aug.transform(
            x,
            args=SimpleNamespace(sat_fs_hz=25e6, sat_fc_hz=2.462e9),
            epoch=1,
            batch_idx=0,
        )

        self.assertTrue(out.applied)
        self.assertEqual(out.scenario, "clear_leo")
        self.assertEqual(out.x.shape, x.shape)
        self.assertEqual(out.x.dtype, x.dtype)
        self.assertTrue(torch.isfinite(out.x).all())

    def test_concat_wrapper_accepts_bosv_schedule(self):
        from concat_sat_channel_aug import ConcatSatChannelAugment

        def fake_apply(x, scenario, args, gen=None, return_meta=False):
            return x + 1.0, {"scenario": scenario}

        aug = ConcatSatChannelAugment(
            scenarios=["mixed_orbit"],
            schedule="1:mixed_orbit;2:rain_leo",
            p=1.0,
            seed=123,
            apply_fn=fake_apply,
        )
        x = torch.zeros(1, 2, 3)
        y = torch.tensor([0])

        out = aug.expand(x, y, None, args=SimpleNamespace(), epoch=2, batch_idx=0)

        self.assertEqual(out.scenario, "rain_leo")
        self.assertEqual(out.stage_start_epoch, 2)
        self.assertTrue(torch.equal(out.x, torch.cat([x, x + 1.0], dim=0)))
        self.assertEqual(aug.scenarios, ["mixed_orbit"])

    def test_train_py_exposes_schedule_to_central_and_federated_paths(self):
        from pathlib import Path

        train_text = (Path(__file__).resolve().parents[1] / "train.py").read_text(encoding="utf-8")

        self.assertIn("BaselineOriginSatViewAugment", train_text)
        self.assertIn("--sat_view_schedule", train_text)
        self.assertIn("schedule=str(getattr(args, \"sat_view_schedule\", \"\") or \"\")", train_text)
        self.assertIn("fed_baseline_sat_view_aug", train_text)


if __name__ == "__main__":
    unittest.main()
