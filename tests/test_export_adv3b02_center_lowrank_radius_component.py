import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from cvsrffi.phase1_center_lowrank_prototype_bundle import (
    ALLOWED_NPZ_MEMBERS,
    MANIFEST_NAME,
    MANIFEST_SHA_NAME,
    NPZ_NAME,
    PENDING_OUTER_JOINT_SEAL,
    V1_ALLOWED_MEMBERS,
)
from scripts.export_adv3b02_center_lowrank_radius_component import (
    PHASE1_LABELED_RATIO,
    PHASE1_SOURCE_VAL_RATIO,
    PHASE1_UNLABELED_RATIO,
    Phase1ExportError,
    _build_checkpoint_data_context,
    build_arg_parser,
    build_v1_aggregate_payload,
    class_handle_binding_sha256,
    class_handles_from_binding_source,
    export_from_loader,
    save_aggregate_component,
    verify_file_sha256,
)


class RecordingIdentityModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.forward_calls = 0
        self.grad_enabled = []

    def forward(
        self,
        x,
        y_tx=None,
        grl_lambda=1.0,
        return_aux=False,
        domain_labels=None,
    ):
        self.forward_calls += 1
        self.grad_enabled.append(torch.is_grad_enabled())
        assert return_aux is True
        assert float(grl_lambda) == 0.0
        assert y_tx is not None and domain_labels is not None
        return {
            "z_id": x * 3.0,
            "tx_logits": torch.zeros((x.shape[0], 2), device=x.device),
        }


def _synthetic_loader(batch_size: int = 5):
    rows = []
    classes = []
    domains = []
    for domain in range(4):
        for class_id in range(2):
            for sample in range(3):
                row = torch.zeros(160, dtype=torch.float32)
                row[class_id] = 1.0
                row[10 + domain] = 0.08 * float(domain + 1)
                row[30 + domain] = 0.01 * float(sample - 1)
                rows.append(row)
                classes.append(class_id)
                domains.append(domain)
    x = torch.stack(rows)
    y = torch.tensor(classes, dtype=torch.int64)
    d = torch.tensor(domains, dtype=torch.int64)
    meta_placeholder = torch.arange(len(rows), dtype=torch.int64)
    return DataLoader(
        TensorDataset(x, y, d, meta_placeholder),
        batch_size=batch_size,
        shuffle=False,
    )


def _geometry_and_model():
    loader = _synthetic_loader()
    model = RecordingIdentityModel()
    geometry = export_from_loader(
        model,
        loader,
        device="cpu",
        num_domains=4,
        num_classes=2,
        radius_histogram_bins=256,
    )
    return geometry, model, len(loader)


def test_export_from_loader_is_two_pass_no_grad_and_aggregate_only() -> None:
    geometry, model, batches_per_pass = _geometry_and_model()

    assert model.forward_calls == 2 * batches_per_pass
    assert model.grad_enabled == [False] * model.forward_calls
    assert geometry.domain_class_centroids.shape == (4, 2, 160)
    assert torch.equal(geometry.domain_class_counts, torch.full((4, 2), 3))
    assert not hasattr(geometry, "sample_features")


def test_in_memory_v1_and_saved_bundle_obey_strict_allowlists(tmp_path: Path) -> None:
    geometry, _model, _batches = _geometry_and_model()
    domains = tuple(f"rx_day:{index}" for index in range(4))
    classes = ("tx-alpha", "tx-beta")
    v1 = build_v1_aggregate_payload(
        geometry, domain_registry=domains, class_registry=classes
    )

    assert set(v1) == V1_ALLOWED_MEMBERS
    assert "domain_class_counts" not in v1
    assert "sample_features" not in v1
    binding = class_handle_binding_sha256(classes)
    output = tmp_path / "component"
    saved = save_aggregate_component(
        output,
        geometry,
        domain_registry=domains,
        class_registry=classes,
        checkpoint_sha256="1" * 64,
        class_handle_binding_sha256_value=binding,
        generation_code_sha256="2" * 64,
        generation_config_sha256="3" * 64,
    )

    assert set(saved["members"]) == {NPZ_NAME, MANIFEST_NAME, MANIFEST_SHA_NAME}
    assert {item.name for item in output.iterdir()} == {
        NPZ_NAME,
        MANIFEST_NAME,
        MANIFEST_SHA_NAME,
    }
    with np.load(output / NPZ_NAME, allow_pickle=False) as arrays:
        assert set(arrays.files) == ALLOWED_NPZ_MEMBERS
        assert "domain_class_counts" not in arrays.files
        assert "sample_features" not in arrays.files
    manifest_text = (output / MANIFEST_NAME).read_text(encoding="utf-8")
    manifest = json.loads(manifest_text)
    assert manifest["checkpoint_sha256"] == "1" * 64
    assert manifest["class_handle_binding_sha256"] == binding
    assert manifest["component_state"] == PENDING_OUTER_JOINT_SEAL
    assert manifest["formal_phase2_eligible"] is False
    assert manifest["outer_bundle_signature_required"] is True
    assert "detached_signature_sha256" not in manifest
    assert len(manifest["radius_generation_proof_sha256"]) == 64
    assert len(saved["pre_sign_content_root_sha256"]) == 64
    assert "domain_class_counts" not in manifest_text
    assert "sample_features" not in manifest_text
    assert str(tmp_path) not in manifest_text


def test_hash_and_class_binding_checks_are_fail_closed(tmp_path: Path) -> None:
    artifact = tmp_path / "checkpoint.pt"
    artifact.write_bytes(b"sealed-checkpoint")
    actual = hashlib.sha256(b"sealed-checkpoint").hexdigest()

    assert verify_file_sha256(artifact, actual, field="checkpoint") == actual
    with pytest.raises(Phase1ExportError, match="SHA256 mismatch"):
        verify_file_sha256(artifact, "0" * 64, field="checkpoint")
    assert class_handle_binding_sha256(("a", "b")) != class_handle_binding_sha256(
        ("b", "a")
    )


def test_existing_registered_handles_are_reused_only_after_tx_order_check(
    tmp_path: Path,
) -> None:
    source = tmp_path / "binding.json"
    payload = {
        "schema": "cvs.phase2.d19_adv3b02_class_binding.v1",
        "checkpoint_sha256": "0" * 64,
        "entries": [
            {
                "class_index": 0,
                "phase1_tx": "tx-a",
                "registered_class_handle": "cls-a",
            },
            {
                "class_index": 1,
                "phase1_tx": "tx-b",
                "registered_class_handle": "cls-b",
            },
        ],
        "evidence": {},
    }
    source.write_text(json.dumps(payload), encoding="utf-8")

    assert class_handles_from_binding_source(
        source, expected_phase1_txs=("tx-a", "tx-b")
    ) == ("cls-a", "cls-b")
    payload["entries"][0]["phase1_tx"] = "tx-b"
    source.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(Phase1ExportError, match="TX/order"):
        class_handles_from_binding_source(
            source, expected_phase1_txs=("tx-a", "tx-b")
        )


def test_output_must_be_empty_before_codec_write(tmp_path: Path) -> None:
    geometry, _model, _batches = _geometry_and_model()
    output = tmp_path / "component"
    output.mkdir()
    (output / "unowned.txt").write_text("preserve", encoding="utf-8")

    with pytest.raises(Phase1ExportError, match="absent or empty"):
        save_aggregate_component(
            output,
            geometry,
            domain_registry=tuple(f"d:{index}" for index in range(4)),
            class_registry=("a", "b"),
            checkpoint_sha256="1" * 64,
            class_handle_binding_sha256_value=class_handle_binding_sha256(("a", "b")),
            generation_code_sha256="2" * 64,
            generation_config_sha256="3" * 64,
        )
    assert (output / "unowned.txt").read_text(encoding="utf-8") == "preserve"


def test_cli_requires_all_provenance_bindings() -> None:
    args = build_arg_parser().parse_args(
        [
            "--checkpoint",
            "checkpoint.pt",
            "--wisig-pkl",
            "ManySig.pkl",
            "--output",
            "bundle",
            "--device",
            "cpu",
            "--expected-checkpoint-sha256",
            "1" * 64,
            "--expected-wisig-sha256",
            "2" * 64,
            "--expected-class-handle-binding-sha256",
            "3" * 64,
            "--generation-config",
            "generation.json",
            "--expected-generation-config-sha256",
            "4" * 64,
        ]
    )
    assert args.checkpoint == "checkpoint.pt"
    assert args.wisig_pkl == "ManySig.pkl"
    assert args.expected_class_handle_binding_sha256 == "3" * 64
    with pytest.raises(SystemExit):
        build_arg_parser().parse_args(
            [
                "--checkpoint", "checkpoint.pt",
                "--wisig-pkl", "ManySig.pkl",
                "--output", "bundle",
                "--expected-checkpoint-sha256", "1" * 64,
                "--expected-wisig-sha256", "2" * 64,
                "--expected-class-handle-binding-sha256", "3" * 64,
                "--generation-config", "generation.json",
                "--expected-generation-config-sha256", "4" * 64,
                "--detached-signature", "fake.sig",
            ]
        )


def test_phase1_component_generation_uses_current_protocol_split(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    observed = {}

    class FakeParser:
        def parse_args(self, _argv):
            return SimpleNamespace()

    class FakeSSDG:
        @staticmethod
        def build_arg_parser():
            return FakeParser()

        @staticmethod
        def _build_ssdg_wisig_data(args, _device):
            observed.update(
                labeled_ratio=args.labeled_ratio,
                unlabeled_ratio=args.unlabeled_ratio,
                source_val_ratio=args.source_val_ratio,
            )
            return {"split_info": {"rho_label": 0.1, "labeled_size": 6}}

    monkeypatch.setitem(__import__("sys").modules, "SSDG", SimpleNamespace(train_ssdg=FakeSSDG))
    checkpoint = {
        "args": {
            "labeled_ratio": 0.10,
            "unlabeled_ratio": 0.70,
            "source_val_ratio": 0.20,
        }
    }
    _ssdg, data_args, _ctx = _build_checkpoint_data_context(
        checkpoint,
        wisig_pkl=tmp_path / "ManySig.pkl",
        device=torch.device("cpu"),
        batch_size=8,
        num_workers=0,
    )

    assert observed == {
        "labeled_ratio": PHASE1_LABELED_RATIO,
        "unlabeled_ratio": PHASE1_UNLABELED_RATIO,
        "source_val_ratio": PHASE1_SOURCE_VAL_RATIO,
    }
    assert data_args.labeled_ratio / (
        data_args.labeled_ratio + data_args.unlabeled_ratio
    ) == pytest.approx(0.1)


def test_same_sum_but_different_ordered_stream_is_rejected() -> None:
    class TwoPassDifferentOrder:
        def __init__(self) -> None:
            self.batches = list(_synthetic_loader(batch_size=8))
            self.pass_count = 0

        def __iter__(self):
            self.pass_count += 1
            batches = list(self.batches)
            if self.pass_count == 2:
                first = batches[0]
                order = torch.arange(first[0].shape[0] - 1, -1, -1)
                batches[0] = tuple(value[order] for value in first)
            return iter(batches)

    with pytest.raises(Phase1ExportError, match="ordered normalized"):
        export_from_loader(
            RecordingIdentityModel(),
            TwoPassDifferentOrder(),
            device="cpu",
            num_domains=4,
            num_classes=2,
            radius_histogram_bins=256,
        )
