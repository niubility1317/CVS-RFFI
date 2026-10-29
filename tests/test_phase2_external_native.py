"""Support-only integration checks; no real target/source artifacts are opened."""
import copy
import json

import pytest
import torch
from torch import nn

from experiments.pamf_phase2_practical import external_native as bridge


@pytest.fixture(autouse=True)
def deterministic_cpu():
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    with torch.random.fork_rng():
        torch.manual_seed(19)
        yield
    torch.set_num_threads(threads)


def bundle(profile, classes=6):
    p = bridge.profiles()[profile]
    method = p["method"]
    model = bridge.nets.factory(method, classes, 4)
    model.eval()
    summaries = {}
    if method == "csil":
        summaries = dict(source_role="L_s", computed_before_target_adaptation=True,
            fisher={k: torch.ones_like(v, dtype=torch.float64) for k, v in model.named_parameters()},
            fisher_objective="empirical_source_label_CE_grad_squared")
    elif method == "mopc_hr":
        summaries = dict(source_role="L_s", computed_before_target_adaptation=True,
            old_prototypes=torch.randn(classes, model.fc.in_features), source_class_ids=list(range(classes)))
    return dict(profile=profile, method=method, model=model, classes=tuple(range(classes)),
                base_metadata=summaries, provenance=dict(status="VERIFIED", synthetic_test=True),
                source_path="synthetic_source_only_fixture")


def small_config(profile):
    return dict(native_registration_overrides={profile: dict(
        old_adaptation_epochs=1, csil_epochs=1, mopc_epochs=1,
        orthogonal_epochs=1, mopc_stage_size=2)},
        native_adaptation_overrides={profile: dict(epochs=1, iterations=1,
            dann_epochs=1, lmmd_epochs=1, finetune_epochs=1, estimate_steps=1)})


def iq(classes, k=1):
    return torch.randn(classes * k, 2, 256), torch.arange(classes).repeat_interleave(k)


def snapshot(model):
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


def unchanged(before, model):
    assert before.keys() == model.state_dict().keys()
    assert all(torch.equal(v, model.state_dict()[k]) for k, v in before.items())


def test_profiles_are_versioned_and_complete_budgets():
    ps = bridge.profiles()
    assert {"radionet_ada_ft_knn", "radionet_ada_knn", "protonet_cda", "csil_corefix",
            "mopc_paper", "mopc_public_code", "orthogonal", "mrior_sda", "dadda_sda",
            "twostage_sda", "feature_separation_legacy", "csil_author"} <= set(ps)
    assert ps["csil_author"]["available"] is False
    assert "NOT_Hu2024" in ps["feature_separation_legacy"]["claim"]
    assert bridge._adaptation_config({}, "radionet_ada_ft_knn", 1, "cpu", 1, None)["iterations"] == 10000
    assert bridge._adaptation_config({}, "radionet_ada_ft_knn", 1, "cpu", 1, None)["finetune_epochs"] == 30
    assert bridge._adaptation_config({}, "mrior_sda", 1, "cpu", 1, None)["estimate_steps"] == 7
    assert bridge._adaptation_config({}, "twostage_sda", 1, "cpu", 1, None)["finetune_epochs"] == 100
    legacy = bridge._adaptation_config({}, "feature_separation_legacy", 1, "cpu", 1, None)
    assert (legacy["epochs"], legacy["steps_per_epoch"], legacy["lr"]) == (100, 1, .001)
    assert "B100epochs" in ps["feature_separation_legacy"]["B_budget"]
    rcfg = bridge._registration_config({}, "mopc_paper", 1)
    assert (rcfg["old_adaptation_epochs"], rcfg["mopc_epochs"], rcfg["mopc_stage_size"], rcfg["csil_epochs"], rcfg["orthogonal_epochs"]) == (10, 20, 5, 3, 50)
    with pytest.raises(NotImplementedError, match="MATLAB"):
        bridge.load_source({}, "csil_author", 2026092701, "cpu")


@pytest.mark.parametrize('profile',['mopc_paper','mopc_public_code'])
def test_mopc_C_trains_each_session_on_support_and_freezes_query_BN(profile):
    b=bundle(profile);x,y=iq(6,2)
    B=bridge.fit_B({},b,x,y,41);before=snapshot(B['model']);train_forwards=[]
    def observe(module,args):
        if torch.is_grad_enabled():
            assert module.training, 'MoPC incremental optimization must enable training mode'
            train_forwards.append(len(args[0]))
    B['model'].register_forward_pre_hook(observe)
    nx,ny=iq(10,2)
    C=bridge.native_C({},b,B,nx,ny+6,tuple(range(6,16)),41)
    assert len(train_forwards)==40 and sum(train_forwards)==400
    assert C['resources']['optimizer_steps']==40
    unchanged(before,B['model'])
    assert not C['model'].training
    for name,v in C['model'].state_dict().items():
        if name.endswith('num_batches_tracked'):
            assert int(v-before[name])==40
    after=snapshot(C['model'])
    with torch.no_grad():
        q=nx[:3];together=C['score_fn'](C['model'],q)
        apart=torch.cat([C['score_fn'](C['model'],row[None]) for row in q])
    torch.testing.assert_close(together,apart,rtol=2e-4,atol=2e-4)
    unchanged(after,C['model'])


@pytest.mark.parametrize("profile", ["protonet_cda", "csil_corefix", "mopc_paper", "mopc_public_code", "orthogonal"])
def test_native_registration_inherits_B_and_independent_C_forks(profile):
    b = bundle(profile)
    initial = snapshot(b["model"])
    x, y = iq(6)
    B = bridge.fit_B(small_config(profile), b, x, y, 31)
    unchanged(initial, b["model"])
    prior = snapshot(B["model"])
    summaries = copy.deepcopy(B["native_state"].base_metadata)
    nx, ny = iq(4)
    C = bridge.native_C(small_config(profile), b, B, nx, ny + 6, (6, 7, 8, 9), 31)
    C2 = bridge.native_C(small_config(profile), b, B, nx[:2], ny[:2] + 6, (6, 7), 31)
    assert B["classes"] == tuple(range(6))
    assert C["classes"] == tuple(range(10)) and C2["classes"] == tuple(range(8))
    unchanged(prior, B["model"])
    assert C["resources"]["C_inherits_B"] and C["resources"]["native_mechanism"]
    q = torch.randn(3, 2, 256)
    state_before = snapshot(C["model"])
    with torch.no_grad():
        together = C["score_fn"](C["model"], q)
        apart = torch.cat([C["score_fn"](C["model"], row[None]) for row in q])
    assert together.shape == (3, 10) and torch.isfinite(together).all()
    torch.testing.assert_close(together, apart, rtol=2e-4, atol=2e-4)
    unchanged(state_before, C["model"])
    if profile == "csil_corefix":
        assert C["native_state"].base_metadata["fisher_objective"] == "empirical_source_label_CE_grad_squared"
        assert all(torch.equal(v, C["native_state"].base_metadata["fisher"][k]) for k, v in summaries["fisher"].items())
        assert C["native_state"].protocol_metadata["source_fisher_preserved"]
    if profile.startswith("mopc"):
        assert len(C["native_state"].protocol_metadata["incremental_stages"]) == 2
        assert C["native_state"].prototypes.shape[0] == 10
        assert C["native_state"].protocol_metadata["prototype_correction_consumed_by_later_stages"]
    if profile == "mopc_public_code":
        assert C["metadata"]["prototype_CE_temperature"] == 2
        assert C["metadata"]["hierarchical_regularization"] == "unsquared_layer_L2"
        assert C["metadata"]["prototype_correction"] == "raw_dot_softmax"
    if profile == "orthogonal":
        assert torch.equal(C["model"].classifier_weight[:6], B["model"].classifier_weight)
        assert torch.equal(C["model"].pseudo_targets, B["model"].pseudo_targets)
    with pytest.raises(ValueError, match="preceding C"):
        bridge.native_C(small_config(profile), b, C2, nx[:2], ny[:2] + 8, (8, 9), 31)


@pytest.mark.parametrize("profile", ["mrior_sda", "dadda_sda", "twostage_sda"])
def test_closed_set_B_uses_only_explicit_L_replay_and_native_head_expansion(profile, monkeypatch):
    b = bundle(profile)
    sx, sy = iq(6, 2)
    calls = []
    def labeled(rt, passed):
        calls.append(passed)
        return sx, sy
    monkeypatch.setattr(bridge, "_source_L", labeled)
    x, y = iq(6)
    source_before = snapshot(b["model"])
    B = bridge.fit_B(small_config(profile), b, x, y, 41)
    assert calls == [b]
    unchanged(source_before, b["model"])
    assert B["metadata"]["source_replay"] is True
    Bbefore = snapshot(B["model"])
    fullx, fully = iq(8)
    expanded = B["expand_head"](B, fullx, fully, 8, 51)
    model = expanded["model"]
    oldmodel = B["model"].base
    assert isinstance(model, bridge.da._ActiveClasses)
    assert expanded["declared_class_mapping_keys"] == ["columns"]
    assert torch.equal(model.columns, torch.arange(8))
    assert set(model.state_dict()) == set(B["model"].state_dict())
    owner, key = bridge._head_location(model.base, b["method"])
    oldowner, oldkey = bridge._head_location(oldmodel, b["method"])
    head, oldhead = owner._modules[key], oldowner._modules[oldkey]
    assert head.out_features == 8
    assert torch.equal(head.weight[:6], oldhead.weight)
    assert torch.equal(head.bias[:6], oldhead.bias) and torch.equal(head.bias[6:], torch.zeros(2))
    assert {id(p) for p in model.parameters() if p.requires_grad} == {id(p) for p in head.parameters()}
    # Frozen hidden head layers, source estimators/discriminators and BN buffers
    # are inherited, with only the final affine state changing shape.
    for name, value in oldmodel.state_dict().items():
        if value.shape == model.base.state_dict()[name].shape:
            assert torch.equal(value, model.base.state_dict()[name])
    captured = []
    h = head.register_forward_pre_hook(lambda module, args: captured.append(args[0].detach()))
    with torch.no_grad():
        expanded["logits_fn"](model, fullx)
    h.remove()
    prehead = captured[0]
    scale = oldhead.weight.norm(dim=1).median()
    for c in (6, 7):
        torch.testing.assert_close(head.weight[c], torch.nn.functional.normalize(prehead[fully == c].mean(0), dim=0) * scale)
    optimizer = torch.optim.Adam(expanded["trainable_parameters"], lr=.001)
    before_update = snapshot(model)
    model.train()  # callback still protects all inherited buffers
    loss = torch.nn.functional.cross_entropy(expanded["logits_fn"](model, fullx), fully)
    loss.backward(); optimizer.step()
    assert not torch.equal(head.weight, before_update[next(n for n, p in model.named_parameters() if p is head.weight)])
    for name, value in before_update.items():
        if name not in {n for n, p in model.named_parameters() if p.requires_grad}:
            assert torch.equal(value, model.state_dict()[name])
    unchanged(Bbefore, B["model"])
    with pytest.raises(NotImplementedError, match="C=N/A"):
        bridge.native_C({}, b, B, fullx[6:], fully[6:], (6, 7), 41)


@pytest.mark.parametrize("profile", ["radionet_ada_knn", "radionet_ada_ft_knn"])
def test_radionet_real_alternating_ADA_and_frozen_knn_append(profile, monkeypatch):
    b = bundle(profile)
    sx, sy = iq(6, 2)
    monkeypatch.setattr(bridge, "_source_L", lambda rt, bundle: (sx, sy))
    x, y = iq(6)
    B = bridge.fit_B(small_config(profile), b, x, y, 13)
    assert B["metadata"]["ada_update"] == "author_alternating_flipped_discriminator_labels"
    assert any(h["stage"] == "ADA" for h in B["native_state"].history)
    assert any(h["stage"] == "support_ft" for h in B["native_state"].history) == profile.endswith("ft_knn")
    nx, ny = iq(2)
    C = bridge.native_C({}, b, B, nx, ny + 6, (6, 7), 13)
    assert C["score_fn"].decision == "knn" and C["native_state"].knn_k == 1
    assert len(B["native_state"].support_features) == 6 and len(C["native_state"].support_features) == 8
    assert C["resources"]["optimizer_steps"] == 0
    with torch.no_grad():
        assert C["score_fn"](C["model"], nx).shape == (2, 8)


def test_callbacks_use_the_model_argument_after_state_copy():
    class Tiny(nn.Module):
        def __init__(self):
            super().__init__(); self.fc = nn.Linear(3, 2)
        def forward(self, x):
            return {"features": x, "tx_logits": self.fc(x)}
    first, second = Tiny(), Tiny()
    with torch.no_grad():
        second.fc.weight.zero_(); second.fc.bias.fill_(7)
    scores = bridge._Scores("mrior", count=2)
    state = copy.deepcopy(dict(model=first, score_fn=scores))
    torch.testing.assert_close(state["score_fn"](second, torch.ones(2, 3)), torch.full((2, 2), 7.))


def test_legacy_feature_separation_is_auxiliary_and_never_replays_source(monkeypatch):
    b = bundle("feature_separation_legacy")
    def forbidden(*args, **kwargs):
        raise AssertionError("Legacy support SFT does not read source IQ")
    monkeypatch.setattr(bridge, "_source_L", forbidden)
    x, y = iq(6)
    B = bridge.fit_B({}, b, x, y, 27)
    assert B["metadata"]["source_replay"] is False
    assert B["resources"]["optimizer_steps"] == 100
    assert B["metadata"]["resolved_config"]["epochs"] == 100
    assert B["profile"] == "feature_separation_legacy" and callable(B["expand_head"])
    assert bridge._profile("legacy_feature_separation")["id"] == B["profile"]
    with torch.no_grad():
        assert B["score_fn"](B["model"], x[:2]).shape == (2, 6)


@pytest.mark.parametrize("profile,expected_B_steps,expected_outer", [
    ("mrior_sda", 800, 100), ("dadda_sda", 100, 100), ("twostage_sda", 300, 300)])
def test_actual_B_to_public_native_control_retains_wrapper_and_full_budget(
        profile, expected_B_steps, expected_outer, monkeypatch):
    from experiments.pamf_phase2_practical.external_controls import build_control
    b = bundle(profile)
    sx, sy = iq(6, 2)
    monkeypatch.setattr(bridge, "_source_L", lambda rt, passed: (sx, sy))
    x, y = iq(6)
    # Run the actual complete_master_r05 B defaults, including MRIOR's seven
    # estimator steps and all three TwoStage stages. No shortened overrides.
    B = bridge.fit_B({}, b, x, y, 67)
    assert B["resources"]["optimizer_steps"] == expected_B_steps
    assert B["resources"]["outer_logged_updates"] == expected_outer
    before = snapshot(B["model"])
    nx, ny = iq(2)
    fullx, fully = torch.cat((x, nx)), torch.cat((y, ny + 6))
    trace = []
    C = build_control("native_output_expand_ft100", B, fullx, fully, tuple(range(8)), 67,
                      logger=lambda row: trace.append(row))
    assert isinstance(C["model"], bridge.da._ActiveClasses)
    assert set(C["model"].state_dict()) == set(before)
    assert torch.equal(C["model"].columns[:6], B["model"].columns)
    assert torch.equal(C["model"].columns, torch.arange(8))
    assert len(trace) == 100 and trace[0]["step"] == 1 and trace[-1]["step"] == 100
    receipt = C["resources"]
    assert receipt["optimizer_steps"] == 100
    assert receipt["optimizer"] == "Adam" and receipt["learning_rate"] == .001
    assert receipt["weight_decay"] == 0 and receipt["support_training_exposures"] == 800
    assert receipt["source_rows"] == 0 and receipt["query_rows_fitted"] == 0
    assert receipt["inherited_B"]["optimizer_steps"] == expected_B_steps
    trainable = {name for name, p in C["model"].named_parameters() if p.requires_grad}
    assert len(trainable) == 2 and all(name.startswith("base.") for name in trainable)
    for name, value in before.items():
        if name != "columns" and name not in trainable:
            assert torch.equal(value, C["model"].state_dict()[name])
    unchanged(before, B["model"])
    q = torch.randn(2, 2, 256)
    assert C["score_fn"](C["model"], q).shape == (2, 8)


def test_generator_compatible_init_is_deterministic_and_does_not_use_init_generator(monkeypatch):
    def old_init(*args, **kwargs):
        if "generator" in kwargs:
            raise TypeError("Torch2.1 init has no generator keyword")
    monkeypatch.setattr(nn.init, "kaiming_uniform_", old_init)
    a, b = torch.empty(9, 7), torch.empty(9, 7)
    bridge.cil._seeded_kaiming_uniform(a, a=5 ** .5, generator=torch.Generator().manual_seed(77))
    bridge.cil._seeded_kaiming_uniform(b, a=5 ** .5, generator=torch.Generator().manual_seed(77))
    assert torch.equal(a, b) and a.abs().max() <= 7 ** -.5


def contract():
    return dict(role_ids=dict(L_s=["l0"], U_s=["u0"], V=["v0"]), source_rxs=[1], source_days=[1],
                ratios=[.1, .8, .1], split_seed=99, classes=["old0", "old1"], num_classes=2,
                equalized=1, out_len=256, normalize=True)


def write_source(root):
    c = contract()
    initial = dict(status="SCRATCH", scratch_only=True, checkpoint=None, checkpoint_sources=[],
                   ancestors=[], target_contact=False, target_access=False, model_seed=2026092701)
    root.mkdir()
    for name, value in (("source_contract", c), ("initialization", initial),
                        ("completion", dict(status="SOURCE_TRAINED", epoch=200))):
        (root / (name + ".json")).write_text(json.dumps(value), encoding="utf-8")
    m = bridge.nets.factory("orthogonal", 2, 1)
    torch.save(dict(epoch=200, method="orthogonal", classes=c["classes"], source_contract=c,
                    initialization=initial, selection="fixed_last_epoch", num_classes=2,
                    num_receivers=1, model_kwargs={}, model=m.state_dict()), root / "last.pt")
    torch.save({}, root / "base_state.pt")
    return c


def test_load_source_checks_exact_roles_scratch_and_frozen_state(tmp_path):
    root = tmp_path / "orthogonal-s2026092701"
    c = write_source(root)
    rt = dict(source_contract=c, native_source_root=str(tmp_path))
    b = bridge.load_source(rt, "orthogonal", 2026092701, "cpu")
    assert b["source_only"] and b["provenance"]["status"] == "VERIFIED"
    assert b["source_path"] == str(root) and not any(p.requires_grad for p in b["model"].parameters())
    bad = copy.deepcopy(c); bad["role_ids"]["U_s"] = ["not_the_same_physical_role"]
    with pytest.raises(ValueError, match="DATA_CONTRACT_MISMATCH: role_ids"):
        bridge.load_source(dict(rt, source_contract=bad), "orthogonal", 2026092701, "cpu")
    init = json.loads((root / "initialization.json").read_text())
    init["target_access"] = True
    (root / "initialization.json").write_text(json.dumps(init), encoding="utf-8")
    with pytest.raises(ValueError, match="TARGET_CONTAMINATED"):
        bridge.load_source(rt, "orthogonal", 2026092701, "cpu")


def test_source_replay_reader_only_requests_L(monkeypatch):
    import experiments.cric_dura_comparison.f_data as source
    calls = []
    class Dataset:
        def __init__(self, root, role, c):
            calls.append((root, role, c))
            self.x, self.metadata = torch.zeros(2, 2, 256), dict(y=torch.tensor([0, 1]))
    monkeypatch.setattr(source, "SourceDataset", Dataset)
    c = contract()
    x, y = bridge._source_L(dict(source_cache="legal_source_cache"), dict(source_contract=c))
    assert calls == [("legal_source_cache", "L_s", c)] and y.tolist() == [0, 1]


def test_reject_future_class_labels_and_missing_source_fisher():
    b = bundle("csil_corefix")
    x, y = iq(6)
    b["base_metadata"] = {}
    with pytest.raises(ValueError, match="fixed source base metadata"):
        bridge.fit_B(small_config("csil_corefix"), b, x, y, 3)
    with pytest.raises(ValueError, match="covering every"):
        bridge.fit_B({}, b, x, y + 1, 3)
