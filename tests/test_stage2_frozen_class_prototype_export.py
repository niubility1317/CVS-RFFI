from __future__ import annotations

import importlib

import numpy as np
import pytest
import torch


subject = importlib.import_module("cvsrffi.stage2_frozen_class_prototype_export")


def test_exports_only_precomputed_2d_class_prototypes_and_ids(tmp_path) -> None:
    source = {
        "feature_key": "z_id",
        "prototypes": torch.tensor(
            [[3.0, 4.0, 0.0], [0.0, 0.0, 5.0]], dtype=torch.float32
        ),
        "tx_domain_prototypes": torch.ones(2, 3, 3),
        "tx_domain_counts": torch.ones(2, 3),
    }
    output = tmp_path / "frozen_class_prototypes.npz"

    audit = subject.export_minimal_frozen_class_prototypes(
        source,
        class_ids=[0, 1],
        output_path=output,
    )

    with np.load(output, allow_pickle=False) as artifact:
        assert set(artifact.files) == {"prototypes", "class_ids"}
        assert artifact["prototypes"].shape == (2, 3)
        assert artifact["prototypes"].dtype == np.float32
        assert artifact["class_ids"].tolist() == [0, 1]
        assert np.allclose(np.linalg.norm(artifact["prototypes"], axis=1), 1.0)
    assert audit["source_member"] == "prototypes"
    assert audit["output_members"] == ["class_ids", "prototypes"]


@pytest.mark.parametrize(
    "source",
    [
        {"feature_key": "z_id", "tx_domain_prototypes": torch.ones(2, 3, 4)},
        {"feature_key": "wrong", "prototypes": torch.ones(2, 4)},
        {"feature_key": "z_id", "prototypes": torch.ones(2, 3, 4)},
    ],
)
def test_rejects_runtime_reconstruction_or_non_zid_inputs(tmp_path, source) -> None:
    with pytest.raises(ValueError):
        subject.export_minimal_frozen_class_prototypes(
            source,
            class_ids=[0, 1],
            output_path=tmp_path / "bad.npz",
        )
