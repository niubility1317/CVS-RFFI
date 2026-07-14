import argparse

import torch


def test_drift_paper_default_uses_minibatch_receiver_centers():
    from baselines.drift import train_cvs

    parser = argparse.ArgumentParser()
    train_cvs.add_drift_method_args(parser)

    assert parser.parse_args([]).center_mode == "batch"


def test_drift_eq25_sums_domain_conditional_batch_center_losses():
    from baselines.drift.losses import receiver_style_transfer_center_loss

    features = torch.tensor([[0.0, 0.0], [2.0, 0.0], [0.0, 0.0], [4.0, 0.0]])
    receivers = torch.tensor([0, 0, 1, 1], dtype=torch.long)

    # Domain 0 contributes 1 and domain 1 contributes 4; Eq. (25) sums domains.
    assert torch.isclose(receiver_style_transfer_center_loss(features, receivers), torch.tensor(5.0))
