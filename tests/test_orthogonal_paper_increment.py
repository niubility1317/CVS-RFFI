import torch
from experiments.orthogonal_paper_repro.increment import register_session


def test_registration_keeps_encoder_and_old_weights_and_uses_only_new_support():
    torch.manual_seed(72)
    encoder = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(8, 12))
    old_encoder = {key: value.detach().clone() for key, value in encoder.state_dict().items()}
    old_weights = torch.randn(5, 12)
    old_copy = old_weights.clone()
    x = torch.randn(15, 2, 4)
    y = torch.arange(3).repeat_interleave(5)
    result, history = register_session(encoder, old_weights, x, y, epochs=3, patience=4)
    assert result.shape == (8, 12)
    torch.testing.assert_close(result[:5], old_copy)
    torch.testing.assert_close(old_weights, old_copy)
    assert len(history) == 3 and all(torch.isfinite(torch.tensor(v)) for v in history)
    for key, value in encoder.state_dict().items():
        torch.testing.assert_close(value, old_encoder[key])


def test_registration_rejects_malformed_support():
    model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(8, 12))
    try:
        register_session(model, torch.randn(5, 12), torch.randn(3, 2, 4), torch.tensor([0, 0, 1]))
    except ValueError:
        return
    raise AssertionError('expected five examples for each new class')
