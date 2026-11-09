"""Focused structural tests using the locally preserved Phase1 releases."""
import json
import sys
from pathlib import Path

import torch
import pytest

PROJECT = Path("E:/type10-7")
WORKTREE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from experiments.cric_dura_comparison.bootstrap import activate

roots = PROJECT / "local_artifacts/cric_dura_comparison_20261021/frozen_sources"
activate(dict(frozen_sources={k: str(roots / k) for k in ("F", "AF", "BASE", "FOLD", "G")}))
import experiments
# Keep the historical frozen releases first, matching the real launcher.
experiments.__path__.append(str(WORKTREE / "experiments"))
from experiments.pamf_branch_factorial.graph import SPECS, install
from experiments.pamf_branch_factorial.method import optimizer_scope, rows, configuration, validate_config, RUN
from experiments.cric_dura_comparison.f_model import install as supervised
from experiments.cric_dura_comparison.f_train import arguments, class_dimension, original_config
from experiments.cvs_phase1_stack.runtime import installed

context = json.loads((PROJECT / "local_artifacts/pamf_no_signature_ablation_20261026/context.json").read_text(encoding="utf-8"))
config = context["baselines"]["2026092701"]["complete"]["config"]
torch.set_num_threads(2)


def make(arm):
    torch.manual_seed(2026092701)
    with class_dimension(config), installed(original_config(2026092701)) as native:
        args = arguments(config, "cpu")
        args.input_len = 256
        net = supervised(native.build_baseline_model(args, "cpu"), config)
        return install(net, arm, 2026092701)


def test_all_arms_finite_and_removed_modules_are_disabled():
    torch.manual_seed(42)
    x = torch.randn(2, 2, 256)
    for arm, spec in SPECS.items():
        net = make(arm)
        net.eval()
        body = net.id_backbone.encoder.core.id_backbone
        time, frequency, pa, signature, rho = spec
        assert (body.t2 is not None) == bool(time), arm
        assert (body.f1 is not None) == bool(frequency), arm
        assert bool(body.use_pa_path) == bool(pa), arm
        assert (hasattr(net.id_backbone.encoder.sequence_pool, "views")) == bool(signature), arm
        y = net(x)
        assert y.shape == (2, 6) and torch.isfinite(y).all(), arm
        torch.nn.functional.cross_entropy(y, torch.tensor([0, 1])).backward()
        assert all(torch.isfinite(p.grad).all() for p in net.parameters() if p.grad is not None), arm
        twin = make(arm).eval()
        twin.load_state_dict(net.state_dict(), strict=True)
        with torch.no_grad():
            assert torch.equal(twin(x), y.detach()), arm
        del net


def test_frequency_removal_preserves_pa_statistics_and_declared_rho():
    torch.manual_seed(123)
    x = torch.randn(2, 2, 256)
    for arm in ("strict_no_frequency", "no_frequency_with_rho", "pa_only", "no_path_no_frequency"):
        net = make(arm).eval()
        body = net.id_backbone.encoder.core.id_backbone
        with torch.no_grad():
            _, expected_rho, _, expected_pa = body._mirror_compressed_features(x, sinc_iq=None)
        capture = {}
        hs = [body.pa_stats_proj.register_forward_pre_hook(lambda m, a: capture.update(pa=a[0].detach().clone())),
              body.fuse.register_forward_pre_hook(lambda m, a: capture.update(fuse=a[0].detach().clone()))]
        with torch.no_grad():
            net(x)
        for h in hs:
            h.remove()
        assert torch.equal(capture["pa"], expected_pa), arm
        assert torch.count_nonzero(capture["fuse"][:, 160:320]) == 0, arm
        actual_rho = capture["fuse"][:, 320:321]
        if SPECS[arm][4]:
            assert torch.equal(actual_rho, expected_rho), arm
        else:
            assert torch.count_nonzero(actual_rho) == 0, arm


def test_absent_time_uses_original_adamw_for_remaining_parameters_only():
    from types import SimpleNamespace
    from experiments.cric_dura_comparison import f_optimizer
    net = make("no_time")
    assert not any(name.startswith("id_backbone.encoder.core.id_backbone.sinc.")
                   for name, _ in net.named_parameters())
    original_adamw = torch.optim.AdamW
    original_scope = f_optimizer.native_scope
    owner = SimpleNamespace(model=net, optimizer=None)
    with optimizer_scope(True):
        with f_optimizer.native_scope(owner, "CENTER_UNITS"):
            opt = torch.optim.AdamW(net.parameters(), lr=0.0002, weight_decay=0.0001)
            net.train()
            output = net(torch.randn(2, 2, 256))
            torch.nn.functional.cross_entropy(output, torch.tensor([0, 1])).backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            opt.step()
            assert owner.optimizer is opt
    assert torch.optim.AdamW is original_adamw
    assert f_optimizer.native_scope is original_scope


def test_fixed_matrix_and_truth_permissions():
    from experiments.cric_dura_comparison.evaluation import reject_truth
    plan = rows()
    assert len(plan) == 36 and sum(r["reuse"] for r in plan) == 9
    row = next(r for r in plan if r["arm"] == "no_time" and r["model_seed"] == 2026092701)
    c = configuration(context, row, "/source-only", "commit")
    assert c["run_id"] == RUN and c["U_use"] is False
    for key, value in (("truth", "target.json"), ("U_use", True), ("source_days", [0, 1, 2]),
                       ("satellite_CE_weight", 0.), ("shared_prefix", "unverified")):
        bad = dict(c, **{key: value})
        with pytest.raises(ValueError):
            validate_config(bad)
    with pytest.raises(ValueError):
        reject_truth({"tests": {"cross_rx": {"truth": "secret"}}})


def test_population_pooling_recalculates_macro_f1():
    from experiments.pamf_branch_factorial.analyze import pooled
    a = {"count": 126000, "confusion": [[0] * 6 for _ in range(6)]}
    b = {"count": 42000, "confusion": [[0] * 6 for _ in range(6)]}
    for i in range(6):
        a["confusion"][i][i] = 21000
        b["confusion"][i][(i + 1) % 6] = 7000
    acc, f1 = pooled(a, b)
    assert acc == .75 and f1 == .75
    with pytest.raises(ValueError):
        pooled(dict(a, count=1), b)


def test_native_supervised_loop_with_removed_time(tmp_path, monkeypatch):
    """Exercise actual trainer and optimizer on synthetic source IQ only."""
    import shutil
    import numpy as np
    from experiments.cric_dura_comparison import f_train
    from experiments.pamf_repl_completion.objectives import installed as objective_scope
    from experiments.practical_full_2x2 import augmentation
    from experiments.pamf_branch_factorial.method import architecture_scope

    src = PROJECT / "local_artifacts/cric_dura_comparison_20261021/native_loop_e9cdde7e/cache"
    dst = tmp_path / "source"
    shutil.copytree(src, dst)
    contract = json.loads((dst / "source_contract.json").read_text(encoding="utf-8"))
    contract["classes"] = [str(n) for n in range(6)]
    (dst / "source_contract.json").write_text(json.dumps(contract), encoding="utf-8")
    for role in ("L_s", "V"):
        path = dst / (role + "_metadata.npz")
        with np.load(path, allow_pickle=False) as z:
            metadata = {k: z[k] for k in z.files}
        metadata["y"] = metadata["y"] % 6
        np.savez(path, **metadata)

    class SyntheticConcat:
        def __init__(self, cfg): pass
        def prepare(self, raw, meta, epoch, step): self.raw = raw
        def __call__(self, ordinary, progress, batch_index, stage=1):
            return self.raw, dict(applied=False, applied_count=0)

    monkeypatch.setattr(augmentation, "PAMFConcat", SyntheticConcat)
    row = next(r for r in rows() if r["arm"] == "no_time" and r["model_seed"] == 2026092701)
    c = configuration(context, row, "/source-only", "test")
    c.update(output_root=str(tmp_path / "result"), source_cache=str(dst),
             source_contract=str(dst / "source_contract.json"), source_rxs=[1, 3])
    with architecture_scope(), optimizer_scope(True), objective_scope(c, {}, False):
        f_train.train(c, device="cpu", smoke_steps=2)
    done = json.loads((tmp_path / "result/complete.json").read_text(encoding="utf-8"))
    assert done["status"] == "LOCAL_SYNTHETIC_SMOKE_ONLY"
    assert done["updates"] == 4 and done["target_access"] is False and done["U_truth_read"] is False


def test_handoff_process_identity_and_gpu_limits():
    from experiments.pamf_branch_factorial.continuation import limits, can_start, command, matches_worker, matches_dispatcher
    rt = {'python': '/python', 'source_release': '/old-release'}
    job = {'kind': 'train', 'config': '/run/configs/train__no_time.json'}
    args = command(rt, job)
    assert matches_worker(rt, job, '/old-release', args)
    assert not matches_worker(rt, job, '/other', args)
    assert not matches_worker(rt, job, '/old-release', args + ['different'])
    dispatch = ['/python', '-u', '-m', 'experiments.pamf_branch_factorial.runner',
                'dispatch', '--config', '/old-release/runtime.json']
    assert matches_dispatcher(rt, '/old-release', dispatch)
    assert not matches_dispatcher(rt, '/old-release', command(rt, job))
    assert [limits(g) for g in range(8)] == [3, 3, 3, 3, 3, 3, 3, 2]
    assert can_start(0, 2, 12000) and not can_start(0, 3, 24000)
    assert not can_start(7, 2, 24000) and not can_start(0, 1, 11999)

