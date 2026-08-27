import json
from pathlib import Path

import pytest

from paper_reproduction.gaskin_tweak_2023.method_config import MethodConfigError as TweakConfigError
from paper_reproduction.gaskin_tweak_2023.method_config import load_method_config as load_tweak_config
from paper_reproduction.hu_feature_separation_2024.method_config import MethodConfigError as HuConfigError
from paper_reproduction.hu_feature_separation_2024.method_config import load_method_config as load_hu_config


@pytest.mark.parametrize(
    ("loader", "expected_method", "required_default"),
    [
        (load_tweak_config, "gaskin_tweak_2023", "triplet_mining"),
        (load_hu_config, "hu_feature_separation_2024", "loss_weights"),
    ],
)
def test_method_metadata_exposes_documented_unpublished_defaults(loader, expected_method, required_default):
    metadata = loader().method_metadata()
    assert metadata["method"] == expected_method
    assert metadata["parity_status"] == "PAPER_METHOD_PARITY_WITH_UNPUBLISHED_DEFAULTS"
    assert metadata["unpublished_defaults"][required_default]["status"] == "UNPUBLISHED_DEFAULT"
    assert metadata["unpublished_defaults"][required_default]["rationale"]


@pytest.mark.parametrize(
    ("loader", "error_type"),
    [(load_tweak_config, TweakConfigError), (load_hu_config, HuConfigError)],
)
def test_method_config_rejects_default_without_status_or_rationale(loader, error_type, tmp_path: Path):
    config = loader().raw
    config["unpublished_defaults"][next(iter(config["unpublished_defaults"]))].pop("rationale")
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(error_type, match="rationale"):
        loader(path)
