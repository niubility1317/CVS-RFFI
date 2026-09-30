import json
from types import SimpleNamespace
import pytest
import torch

from comparison_suite.models import factory, embedding, classifier_logits, METHODS
from comparison_suite.source import source_args, source_episode, source_loss, train


@pytest.mark.parametrize("method", METHODS)
def test_native_model_state_roundtrip_and_real_backward(method, tmp_path):
    torch.set_num_threads(2)
    torch.manual_seed(3)
    model = factory(method, 3, 12)
    x, y = torch.randn(6, 2, 256), torch.tensor([0, 0, 1, 1, 2, 2])
    if method == "protonet":
        from paper_reproduction.protonet_cda.model import prototypical_nll
        z = embedding(model, method, x)
        loss, _ = prototypical_nll(z[::2], y[::2], z[1::2], y[1::2])
    else:
        loss, terms = source_loss(model, method, x, y, torch.arange(6) % 12, {})
        assert terms
    assert torch.isfinite(loss)
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
    model.eval()
    before = classifier_logits(model, method, x).detach()
    path = tmp_path / "initial.pt"
    torch.save({"model": model.state_dict(), "method": method, "scratch_only": True}, path)
    restored = factory(method, 3, 12).eval()
    restored.load_state_dict(torch.load(path, weights_only=False)["model"])
    assert torch.equal(before, classifier_logits(restored, method, x))
    assert embedding(restored, method, x).ndim == 2


def test_source_args_forbid_inheritance_and_fixed_roles():
    cfg = {"dataset": "ManySig.pkl", "source_contract": "contract.json", "output_root": "out"}
    args = source_args(cfg)
    assert args.wisig_split_seed == 392005
    assert (args.wisig_labeled_ratio, args.wisig_unlabeled_ratio, args.wisig_source_val_ratio) == (.07, .63, .30)
    with pytest.raises(ValueError, match="scratch"):
        source_args(dict(cfg, resume="old.pt"))


class Samples:
    def __len__(self):
        return 12
    def __getitem__(self, i):
        return {"iq": torch.full((2, 256), float(i + 1)), "label": i // 4,
            "receiver": 1, "domain": 1, "meta": {"sample_id": str(i), "session_id": "rx1/day1"}}


def test_episode_physical_ids_disjoint():
    import random
    s, q = source_episode(Samples(), {c: list(range(c * 4, c * 4 + 4)) for c in range(3)},
        rng=random.Random(4), n_way=3, k_shot=2, query_per_class=2)
    assert not {m["sample_id"] for m in s["meta"]} & {m["sample_id"] for m in q["meta"]}


def test_source_training_artifacts_no_target_and_fisher(monkeypatch, tmp_path):
    import baselines.common.practical_source as practical
    torch.set_num_threads(2)
    samples = Samples()
    def split(args):
        from pathlib import Path
        (Path(args.output_dir) / "source_contract.json").write_text(json.dumps({"classes": ["a", "b", "c"]}), encoding="utf-8")
        return SimpleNamespace(train=samples, val=samples, num_classes=3, num_receivers=12,
            split_info={"counts": {"L_s": 12, "U_s": 108, "V": 40}})
    class Augment:
        def __init__(self, **kwargs): pass
        def set_epoch(self, epoch): pass
        def __call__(self, x, metadata): return x * .97
    monkeypatch.setattr(practical, "build_contract_split", split)
    monkeypatch.setattr(practical, "PracticalResidualAugment", Augment)
    config = dict(method="csil", model_seed=9, dataset="not-read.pkl", source_contract="mock-contract",
        output_root=str(tmp_path), device="cpu", epochs=1, batch_size=6, print_steps=False, grad_clip=None)
    checkpoint = train(config)
    payload = torch.load(checkpoint, weights_only=False)
    assert payload["epoch"] == 1 and payload["selection"] == "fixed_last_epoch"
    assert payload["initialization"]["scratch_only"]
    state = torch.load(tmp_path / "base_state.pt", weights_only=False)
    assert state["fisher"] and state["source_role"] == "L_s"
    assert state["fisher_objective"] == "empirical_source_label_CE_grad_squared"
    assert all(v.dtype == torch.float64 and torch.isfinite(v).all() for v in state["fisher"].values())
    assert not state["query_access"]
    done = json.loads((tmp_path / "completion.json").read_text())
    assert done["status"] == "SOURCE_TRAINED" and not done["target_evaluated"]
    assert (tmp_path / "epoch_metrics.csv").exists()
    assert "source_val_accuracy" in (tmp_path / "training.log").read_text()
    with pytest.raises(FileExistsError):
        train(config)
