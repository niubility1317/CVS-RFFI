import unittest
from types import SimpleNamespace

import torch


class ConcatSatChannelAugmentTest(unittest.TestCase):
    def test_concat_sat_channel_aug_doubles_iq_labels_and_domain(self):
        from concat_sat_channel_aug import ConcatSatChannelAugment

        calls = []

        def fake_apply(x, scenario, args, gen=None, return_meta=False):
            calls.append(scenario)
            return x + 10.0, {"scenario": scenario}

        aug = ConcatSatChannelAugment(
            scenarios=["clear_leo"],
            p=1.0,
            seed=123,
            apply_fn=fake_apply,
        )
        x = torch.arange(12, dtype=torch.float32).view(2, 2, 3)
        y = torch.tensor([1, 2])
        d_raw = torch.tensor([7, 8])

        out = aug.expand(
            x,
            y,
            d_raw,
            args=SimpleNamespace(),
            epoch=1,
            batch_idx=0,
        )

        self.assertEqual(out.x.shape[0], 4)
        self.assertTrue(torch.equal(out.x[:2], x))
        self.assertTrue(torch.equal(out.x[2:], x + 10.0))
        self.assertTrue(torch.equal(out.y, torch.tensor([1, 2, 1, 2])))
        self.assertTrue(torch.equal(out.d_raw, torch.tensor([7, 8, 7, 8])))
        self.assertEqual(out.clean_batch_size, 2)
        self.assertEqual(out.total_batch_size, 4)
        self.assertEqual(out.scenario, "clear_leo")
        self.assertEqual(calls, ["clear_leo"])

    def test_concat_sat_channel_aug_keeps_clean_and_duplicate_when_probability_skips(self):
        from concat_sat_channel_aug import ConcatSatChannelAugment

        def fake_apply(x, scenario, args, gen=None, return_meta=False):
            raise AssertionError("apply_fn should not be called when p=0")

        aug = ConcatSatChannelAugment(
            scenarios=["clear_leo"],
            p=0.0,
            seed=123,
            apply_fn=fake_apply,
        )
        x = torch.arange(8, dtype=torch.float32).view(2, 2, 2)
        y = torch.tensor([0, 1])

        out = aug.expand(x, y, None, args=SimpleNamespace(), epoch=1, batch_idx=0)

        self.assertTrue(torch.equal(out.x, torch.cat([x, x], dim=0)))
        self.assertTrue(torch.equal(out.y, torch.tensor([0, 1, 0, 1])))
        self.assertIsNone(out.d_raw)
        self.assertEqual(out.scenario, "clean_duplicate")

    def test_ce_only_training_helper_keeps_clean_batch_separate_from_sat_view(self):
        from train import prepare_concat_sat_batch_for_training

        calls = []

        class FakeAug:
            def transform(self, x, *, args, epoch, batch_idx):
                calls.append(("transform", epoch, batch_idx))
                return SimpleNamespace(
                    x=x + 100.0,
                    scenario="mixed_orbit",
                    applied=True,
                    clean_batch_size=int(x.size(0)),
                )

            def expand(self, *args, **kwargs):
                raise AssertionError("CE-only path must not concatenate satellite samples into the main CVS batch")

        args = SimpleNamespace(concat_sat_start_epoch=1, concat_sat_ce_only=True)
        x = torch.arange(8, dtype=torch.float32).view(2, 2, 2)
        y = torch.tensor([0, 1])
        d_raw = torch.tensor([4, 5])

        clean_x, clean_y, clean_d, sat_view = prepare_concat_sat_batch_for_training(
            FakeAug(),
            x,
            y,
            d_raw,
            args=args,
            epoch=1,
            batch_idx=3,
        )

        self.assertTrue(torch.equal(clean_x, x))
        self.assertTrue(torch.equal(clean_y, y))
        self.assertTrue(torch.equal(clean_d, d_raw))
        self.assertIsNotNone(sat_view)
        self.assertTrue(torch.equal(sat_view.x, x + 100.0))
        self.assertEqual(calls, [("transform", 1, 3)])

    def test_training_helper_defaults_to_full_dg_concat_batch(self):
        from train import prepare_concat_sat_batch_for_training

        calls = []

        class FakeAug:
            def transform(self, *args, **kwargs):
                raise AssertionError("full-DG path must expand the main CVS batch")

            def expand(self, x, y, d_raw, *, args, epoch, batch_idx):
                calls.append(("expand", epoch, batch_idx))
                return SimpleNamespace(
                    x=torch.cat([x, x + 0.25], dim=0),
                    y=torch.cat([y, y], dim=0),
                    d_raw=torch.cat([d_raw, d_raw], dim=0),
                )

        args = SimpleNamespace(concat_sat_start_epoch=1, concat_sat_ce_only=False)
        x = torch.arange(8, dtype=torch.float32).view(2, 2, 2)
        y = torch.tensor([0, 1])
        d_raw = torch.tensor([4, 5])

        train_x, train_y, train_d, sat_view = prepare_concat_sat_batch_for_training(
            FakeAug(),
            x,
            y,
            d_raw,
            args=args,
            epoch=1,
            batch_idx=3,
        )

        self.assertIsNone(sat_view)
        self.assertTrue(torch.equal(train_x, torch.cat([x, x + 0.25], dim=0)))
        self.assertTrue(torch.equal(train_y, torch.tensor([0, 1, 0, 1])))
        self.assertTrue(torch.equal(train_d, torch.tensor([4, 5, 4, 5])))
        self.assertEqual(calls, [("expand", 1, 3)])

    def test_ce_only_satellite_auxiliary_losses_can_apply_late_consistency(self):
        from train import satellite_auxiliary_losses

        args = SimpleNamespace(lambda_sat_cons=0.03, sat_cons_start_epoch=60)
        ce_tx = torch.nn.CrossEntropyLoss()
        out_sat = {
            "tx_logits": torch.tensor([[4.0, 0.1], [0.2, 3.5]], dtype=torch.float32),
            "z_id": torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=torch.float32),
        }
        clean_z = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
        y = torch.tensor([0, 1])

        early = satellite_auxiliary_losses(
            out_sat,
            y,
            clean_z,
            ce_tx,
            args=args,
            epoch=59,
            cls_weight=1.0,
        )
        late = satellite_auxiliary_losses(
            out_sat,
            y,
            clean_z,
            ce_tx,
            args=args,
            epoch=60,
            cls_weight=1.0,
        )

        self.assertGreater(late["loss_sat_cls"].item(), 0.0)
        self.assertEqual(early["loss_sat_cons"].item(), 0.0)
        self.assertGreater(late["loss_sat_cons"].item(), 0.0)
        self.assertTrue(late["diag_sat_cons_active"])


if __name__ == "__main__":
    unittest.main()
