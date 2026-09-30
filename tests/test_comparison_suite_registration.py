"""Mechanism and data-boundary tests; no real target IQ or truth is loaded."""
import copy

import pytest
import torch
from torch import nn
import torch.nn.functional as F

from comparison_suite import registration as r


class NativeToy(nn.Module):
    def __init__(self, method, classes=4):
        super().__init__()
        self.method = method
        self.encoder = nn.Sequential(nn.Linear(6, 8), nn.BatchNorm1d(8), nn.Tanh())
        if method == "csil":
            self.fc_bf_fp = nn.Linear(8, classes)
            self.fingerprints = nn.Parameter(torch.randn(classes, classes))
        elif method == "mopc_hr":
            self.fc = nn.Linear(8, classes)
        elif method == "orthogonal":
            self.register_buffer("pseudo_targets", F.normalize(torch.randn(8, 8), dim=1))
            self.register_buffer("classifier_weight", self.pseudo_targets[:classes].clone())
        else:
            self.register_buffer("source_prototypes", torch.randn(classes, 8))

    def forward(self, x):
        z = self.encoder(x)
        if self.method == "csil":
            z = self.fc_bf_fp(z)
            logits = 5 * F.normalize(z, dim=1) @ F.normalize(self.fingerprints, dim=1).T + 5
        elif self.method == "mopc_hr":
            logits = self.fc(z)
        elif self.method == "orthogonal":
            logits = F.normalize(z, dim=1) @ F.normalize(self.classifier_weight, dim=1).T
        else:
            logits = -torch.cdist(z, self.source_prototypes).square()
        return {"features": z, "tx_logits": logits}


@pytest.fixture(autouse=True)
def fixed_rng_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(21)
    yield
    torch.set_num_threads(previous)


def inputs():
    source_x = torch.randn(16, 6)
    source_y = torch.arange(4).repeat_interleave(4)
    old_x, old_y = torch.randn(4, 6), torch.tensor([0, 0, 1, 1])
    new_x, new_y = torch.randn(4, 6), torch.tensor([2, 2, 3, 3])
    return source_x, source_y, old_x, old_y, new_x, new_y


def tiny_config(**kwargs):
    return r.RegistrationConfig(old_adaptation_epochs=2, csil_epochs=3,
        mopc_epochs=2, orthogonal_epochs=3, mopc_stage_size=1, **kwargs)


def assert_same_state(before, after):
    assert set(before) == set(after)
    for key in before:
        assert torch.equal(before[key], after[key]), key


@pytest.mark.parametrize("method", r.METHODS)
def test_paired_support_only_allclass_inference_is_pure(method):
    sx, sy, ox, oy, nx, ny = inputs()
    model = NativeToy(method).eval()
    original = copy.deepcopy(model.state_dict())
    cfg = tiny_config()
    base = r.build_registration_base_metadata(method, model, sx, sy, config=cfg)
    events = []
    b = r.prepare_registration_state(method, model, ox, oy, [3, 1],
        config=cfg, base_metadata=base, logger=events.append)
    assert_same_state(original, model.state_dict())
    b_before = copy.deepcopy(b.model.state_dict())
    b_prototypes = None if b.prototypes is None else b.prototypes.clone()
    c = r.register_new_classes(b, nx, ny, [4, 5], config=cfg, logger=events.append)
    assert_same_state(b_before, b.model.state_dict())
    assert c.classes == (3, 1, 4, 5)
    assert c.metadata["C_inherits_B_state"] is True
    before = copy.deepcopy(c.model.state_dict())
    query = torch.randn(7, 6)
    batch = c.score(query)
    single = torch.cat([c.score(row.unsqueeze(0)) for row in query])
    assert batch.shape == (7, 4)
    assert torch.allclose(batch, single, atol=1e-6)
    assert torch.equal(c.predict(query), batch.argmax(1))
    assert_same_state(before, c.model.state_dict())
    assert c.resource["resident_state_bytes"] > 0
    assert c.resource["training_seconds"] >= 0
    assert c.resource["peak_cuda_allocated_bytes"] is None
    if method == "protonet":
        assert torch.equal(c.prototypes[:2], b_prototypes)
        assert c.resource["optimizer_steps"] == 0
        expected = -torch.cdist(r._embedding(c.model, method, query), c.prototypes).square()
        assert torch.equal(batch, expected)
    else:
        assert events and any(row["phase"] == "C" for row in events)
        assert all(row["source_validation"] is None for row in events)


def test_csil_expansion_masks_source_fisher_and_distillation_executed():
    sx, sy, ox, oy, nx, ny = inputs()
    model = NativeToy("csil").eval()
    base = r.build_registration_base_metadata("csil", model, sx, sy)
    assert base["source_role"] == "L_s"
    assert base["computed_before_target_adaptation"]
    assert set(base["fisher"]) == set(dict(model.named_parameters()))
    cfg = tiny_config()
    b = r.prepare_registration_state("csil", model, ox, oy, [3, 1], base_metadata=base, config=cfg)
    c = r.register_new_classes(b, nx, ny, [4, 5], config=cfg)
    assert c.model.fc_bf_fp.out_features == 6
    assert c.model.fingerprints.shape == (4, 6)
    assert_same_state(b.model.encoder.state_dict(), c.model.encoder.state_dict())
    assert torch.equal(c.model.fc_bf_fp.weight[:4], b.model.fc_bf_fp.weight)
    assert not torch.equal(c.model.fingerprints[:2, :4], b.model.fingerprints)
    assert torch.count_nonzero(c.model.fingerprints[:2, 4:]) == 0
    assert torch.count_nonzero(c.model.fingerprints[2:, :4]) == 0
    rows = [row for row in c.loss_trace if row["phase"] == "C"]
    assert all("ewc" in row and "knowledge_distillation" in row for row in rows)
    assert max(row["gradient_norm"] for row in rows) > 0
    assert max(row["knowledge_distillation"] for row in rows) > 0
    assert max(row["ewc"] for row in rows) > 0
    assert c.protocol_metadata["official_fingerprint_mask_corefix"]


def test_mopc_correction_used_by_next_stage_and_native_losses_run():
    sx, sy, ox, oy, nx, ny = inputs()
    model = NativeToy("mopc_hr").eval()
    cfg = tiny_config()
    b = r.prepare_registration_state("mopc_hr", model, ox, oy, [3, 1],
        source_x=sx, source_y=sy, config=cfg)
    c = r.register_new_classes(b, nx, ny, [4, 5], config=cfg)
    assert c.protocol_metadata["prototype_correction_consumed_by_later_stages"]
    assert c.prototypes.shape == (4, 8)
    assert c.protocol_metadata["mopc_version"] == "paper_equations_7_to_22"
    rows = [row for row in c.loss_trace if row["phase"] == "C"]
    assert max(row["prototype_augmentation"] for row in rows) > 0
    assert max(row["hierarchical_regularization"] for row in rows) > 0
    assert not torch.equal(b.model.fc.weight, c.model.fc.weight[:2])


def test_orthogonal_only_new_weights_change_and_gradient_is_logged():
    _, _, ox, oy, nx, ny = inputs()
    cfg = tiny_config()
    b = r.prepare_registration_state("orthogonal", NativeToy("orthogonal").eval(),
        ox, oy, [3, 1], config=cfg)
    c = r.register_new_classes(b, nx, ny, [4, 5], config=cfg)
    assert_same_state(b.model.encoder.state_dict(), c.model.encoder.state_dict())
    assert torch.equal(b.model.classifier_weight, c.model.classifier_weight[:2])
    rows = [row for row in c.loss_trace if row["phase"] == "C"]
    assert max(row["gradient_norm"] for row in rows) > 0
    assert all("margin" in row and "align" in row for row in rows)


def test_no_fisher_fallback_or_missing_support_or_overlapping_new_class():
    _, _, ox, oy, nx, ny = inputs()
    with pytest.raises(ValueError, match="source base metadata"):
        r.prepare_registration_state("csil", NativeToy("csil"), ox, oy, [0, 1])
    with pytest.raises(ValueError, match="source Fisher"):
        r.prepare_registration_state("csil", NativeToy("csil"), ox, oy, [0, 1],
            base_metadata={"source_role": "L_s", "computed_before_target_adaptation": True})
    with pytest.raises(ValueError, match="cover exactly"):
        r.prepare_registration_state("protonet", NativeToy("protonet"), ox, oy, [0, 1, 2])
    b = r.prepare_registration_state("protonet", NativeToy("protonet"), ox, oy, [0, 1])
    with pytest.raises(ValueError, match="disjoint"):
        r.register_new_classes(b, nx, ny, [1, 4])


def test_source_empirical_fisher_is_exact_per_example_mean_and_declared():
    sx, sy, *_ = inputs()
    model = NativeToy("csil").double().eval()
    result = r.build_registration_base_metadata("csil", model, sx, sy)
    expected = {name: torch.zeros_like(value) for name, value in model.named_parameters()}
    for index in range(len(sx)):
        model.zero_grad(set_to_none=True)
        F.cross_entropy(model(sx[index:index + 1].double())["tx_logits"], sy[index:index + 1]).backward()
        for name, value in model.named_parameters():
            if value.grad is not None:
                expected[name] += value.grad.detach().square() / len(sx)
    for name, value in expected.items():
        assert torch.allclose(result["fisher"][name], value, rtol=1e-12, atol=1e-12)
        assert torch.isfinite(value).all()
    assert result["fisher_objective"] == "empirical_source_label_CE_grad_squared"
    assert result["fisher_precision"] == "float64"


@pytest.mark.parametrize("method", r.METHODS)
def test_k1_remains_valid_and_native_factory_interoperates(method):
    from comparison_suite.models import factory
    model = factory(method, 4, 2).eval()
    if method == "csil":
        # This is an untrained synthetic model, not a completed source checkpoint.
        # Avoid singular near-zero fingerprints in author's exp-gradient Fisher.
        with torch.no_grad():
            model.fingerprints.copy_(F.normalize(model.fingerprints, dim=1))
    ox, oy = torch.randn(2, 2, 256), torch.tensor([0, 1])
    nx, ny = torch.randn(2, 2, 256), torch.tensor([2, 3])
    source_x = torch.randn(8, 2, 256)
    source_y = torch.arange(4).repeat_interleave(2)
    cfg = r.RegistrationConfig(old_adaptation_epochs=1, csil_epochs=1, mopc_epochs=1,
                               orthogonal_epochs=1, mopc_stage_size=2)
    b = r.prepare_registration_state(method, model, ox, oy, [3, 1], config=cfg,
                                     source_x=source_x, source_y=source_y)
    c = r.register_new_classes(b, nx, ny, [4, 5], config=cfg)
    assert c.score(torch.randn(3, 2, 256)).shape == (3, 4)
