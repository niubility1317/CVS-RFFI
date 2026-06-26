import unittest

from cvsrffi.logging import AverageMeter, NanMeter, format_epoch_block


def _meters():
    names = [
        "loss", "cls", "dom", "adv", "orth", "cons", "group_ce", "txacc",
        "cls_pa", "cls_dac", "pa_joint_inv", "pa_kl", "dac_reg", "pa_reg",
        "gap_dac", "gap_pa", "cos_joint_pa", "cos_imp_pa",
        "sat_cls", "sat_cons", "sat_cos",
        "proto", "proto_pull_cos", "supcon", "fishr",
        "w_cls", "w_dom", "w_adv", "w_orth", "w_cons", "w_group_ce",
        "w_cls_pa", "w_cls_dac", "w_pa_joint_inv", "w_pa_kl", "w_dac_reg", "w_pa_reg",
        "w_sat_cls", "w_sat_cons", "w_proto", "w_supcon", "w_fishr",
        "grad_total", "grad_backbone", "grad_aux", "grad_domain",
    ]
    meters = {name: AverageMeter() for name in names}
    for meter in meters.values():
        meter.update(1.0)
    return meters


class EpochTimingLoggingTest(unittest.TestCase):
    def test_epoch_block_reports_train_and_eval_timing_breakdown(self):
        domacc = NanMeter()
        domacc.update(50.0)

        text = format_epoch_block(
            2,
            3,
            1e-3,
            12.5,
            _meters(),
            domacc,
            0.25,
            {"tx_acc": 80.0, "dom_acc": 40.0},
            {"tx_acc": 70.0, "tx_correct": 7, "tx_total": 10},
            {"test_unseen_day_unseen_rx": {"tx_acc": 70.0, "tx_correct": 7, "tx_total": 10}},
            {"test_unseen_day_unseen_rx": {}},
            80.0,
            70.0,
            2,
            "latest.pth",
            "best.pth",
            True,
            None,
            1.0,
            time_stats={
                "train_time_s": 8.0,
                "val_time_s": 1.0,
                "test_time_s": 2.0,
                "sat_test_time_s": 1.5,
                "eval_time_s": 4.5,
            },
        )

        self.assertIn("[EPOCH-BEGIN] E002/003 | time=12.5s", text)
        self.assertIn("[TIME] train=8.0s val=1.0s test=2.0s sat_test=1.5s eval=4.5s", text)


if __name__ == "__main__":
    unittest.main()
