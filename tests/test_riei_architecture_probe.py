from pathlib import Path
import re

import pytest
import torch

from baselines.riei_fd.architecture import RIEIResNet1D18FED


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "code" / "scripts" / "launch_riei_table3_architecture_probe_20260715.sh"


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


def test_architecture_probe_is_pre_registered_and_single_variable() -> None:
    text = SCRIPT.read_text(encoding="utf-8")
    rows = re.findall(r'^\s+"(3|6|10|12)\|', text, re.MULTILINE)
    assert rows == ["3", "6", "10", "12"]
    assert "VARIANTS=(imagenet1d short_stem1d)" in text
    assert "jobs=8" in text
    for setting in (
        '"BASELINE_EPOCHS=200"',
        '"RIEI_PAPER_EVAL_LAST_N=10"',
        '"RIEI_OPTIMIZER=sgd"',
        '"RIEI_SGD_MOMENTUM=0"',
        '"RIEI_CE_REDUCTION=mean"',
        '"RIEI_MI_REDUCTION=mean"',
        '"RIEI_IE_REDUCTION=mean"',
        '"RIEI_WISIG_RMS_NORMALIZE=0"',
        '"RIEI_LAMBDA_FEATURE_NORM=0"',
    ):
        assert setting in text
    assert '"RIEI_FED_VARIANT=${variant}"' in text
    assert "planned_peak=1" in text
    assert "total <= MAX_TRAIN_PER_GPU" in text
