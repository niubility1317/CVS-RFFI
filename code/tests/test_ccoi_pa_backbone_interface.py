import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from model import build_model  # noqa: E402


def test_pa_token_map_is_exposed_without_adding_checkpoint_parameters():
    model = build_model(
        num_classes=3,
        model_size="S",
        dataset="wisig",
        input_len=64,
        sample_rate_hz=25e6,
        model_variant="lite_h",
    )
    state_keys_before = tuple(model.state_dict())

    out = model(torch.randn(2, 2, 64), return_aux=True)

    assert "pa_token_map" in out
    assert out["pa_token_map"].ndim == 3
    assert out["pa_token_map"].shape[0] == 2
    assert out["pa_token_map"].shape[1] > 0
    assert out["pa_token_map"].shape[2] > 0
    assert tuple(model.state_dict()) == state_keys_before
