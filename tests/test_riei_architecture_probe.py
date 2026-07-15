import pytest
import torch

from baselines.riei_fd.architecture import RIEIResNet1D18FED


def test_riei_fed_variants_preserve_expected_embedding_shape() -> None:
    x = torch.randn(2, 2, 256)
    for variant in ("imagenet1d", "short_stem1d"):
        model = RIEIResNet1D18FED(variant=variant, use_projection=False)
        assert model(x).shape == (2, 512)


def test_riei_fed_short_stem_is_distinct_and_fail_closed() -> None:
    standard = RIEIResNet1D18FED(variant="imagenet1d")
    short = RIEIResNet1D18FED(variant="short_stem1d")
    assert standard.stem[0].kernel_size == (7,)
    assert standard.stem[0].stride == (2,)
    assert len(standard.stem) == 4
    assert short.stem[0].kernel_size == (3,)
    assert short.stem[0].stride == (1,)
    assert len(short.stem) == 3
    with pytest.raises(ValueError, match="Unsupported RIEI FED variant"):
        RIEIResNet1D18FED(variant="unknown")
