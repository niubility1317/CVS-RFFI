import torch


def test_riei_optimizer_supports_paper_gradient_descent_and_adam_control():
    from baselines.riei_fd.train_cvs import build_riei_optimizer

    layer = torch.nn.Linear(2, 2)
    sgd = build_riei_optimizer("sgd", layer.parameters(), lr=1e-4, momentum=0.0)
    adam = build_riei_optimizer("adam", layer.parameters(), lr=1e-4)
    assert isinstance(sgd, torch.optim.SGD)
    assert sgd.defaults["momentum"] == 0.0
    assert isinstance(adam, torch.optim.Adam)
