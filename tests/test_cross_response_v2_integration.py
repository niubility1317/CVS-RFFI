import copy
import sys
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code" / "tests"))
from test_cross_response_runtime_contract import SourceDataset
from cvsrffi.cross_response.integration import CrossResponseRuntime
from cvsrffi.cross_response.config import validate_runtime_config
from scripts.core90_cross_response_matrix import resolve_variant
from scripts.register_cross_response_v2 import configuration


class Head(nn.Linear):
    def forward(self, x, labels=None):
        return super().forward(x)


class Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.time_fuse = nn.Linear(2, 2)
        self.cls_head = nn.Module()
        self.cls_head.proj = nn.Linear(2, 3)
        self.cls_head.head = Head(3, 4)

    def features(self, x):
        return self.cls_head.proj(self.time_fuse(x.mean(-1)))


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.id_backbone, self.dom_backbone = Backbone(), Backbone()

    def forward(self, x):
        zi, zd = self.id_backbone.features(x), self.dom_backbone.features(x)
        return dict(z_id=zi, z_dom=zd, tx_logits=self.id_backbone.cls_head.head(zi), aux_id={"feat_joint":zi})


def mechanism_controls():
    """Explicit synthetic controls, never production source-frozen parameters."""
    return dict(source_audit_enabled=True, source_baseline_fit_steps=2, source_baseline_fit_lr=.001,
        mechanism_audit=dict(interval_steps=1, pairs_per_observation=2, query_records_per_cell=1,
                             seed=392005, identity_scope='cls_head'))


def runtime(tmp_path, variant="U1", **overrides):
    torch.manual_seed(392005)
    model = Model()
    artifact = tmp_path / "source.bin"
    artifact.write_bytes(b"synthetic-legal-source")
    args = SimpleNamespace(cross_response_variant=variant, wisig_train_rxs="0,1,2,3", wisig_test_rxs="4",
        wisig_train_days="0", wisig_test_days="1", output_dir=str(tmp_path), batch_size=32,
        eval_batch_size=16, seed=392005, wisig_pkl=str(artifact), lr=.01, weight_decay=.01,
        amp=False, max_grad_norm=1.)
    c = resolve_variant(configuration(), variant)
    if overrides.get('mechanism_gate') is not None:
        c.update(mechanism_controls())
    c.update(target_family="iq", target_dim=5, source_fit_max_records=32, source_eval_max_blocks=1,
             gate_min_blocks=1, **overrides)
    validate_runtime_config(c)
    ctx = dict(train_loader=DataLoader(SourceDataset(),batch_size=32),
               val_loader=DataLoader(SourceDataset(validation=True),batch_size=32),input_len=32)
    rt = CrossResponseRuntime(model, ctx, c, args, torch.device("cpu"))
    return model, rt, ctx


def update(model, rt, ctx, optimizer):
    x,y,d,meta = next(iter(ctx["train_loader"]))
    plan = rt.begin_batch((d,meta))
    out = model(x)
    terms = rt.losses(model,out,y,plan)
    base = torch.nn.functional.cross_entropy(out["tx_logits"],y)
    optimizer.zero_grad(set_to_none=True)
    scaler = torch.amp.GradScaler("cuda",enabled=False)
    rt.backward(model,base,base,terms,scaler)
    optimizer.step()
    rt.commit(True)


def test_v2_u1_no_statistics_and_diagnostics_do_not_skip_updates(tmp_path):
    model,rt,ctx = runtime(tmp_path)
    assert rt.statistics is None and rt.normalizer is None
    optimizer = torch.optim.AdamW(model.parameters(),lr=.01)
    for _ in range(3):
        before = copy.deepcopy(model.state_dict())
        update(model,rt,ctx,optimizer)
        assert any(not torch.equal(v,before[k]) for k,v in model.state_dict().items())
    assert rt.counts["successful_steps"] == 3
    assert rt.counts["audited_steps"] == 1
    assert rt.sampler.scheduler.last_finished_step == 2
    assert rt.sampler.scheduler.flush_count == 0


def test_head_only_own_optimizer_and_resume(tmp_path):
    model,rt,ctx = runtime(tmp_path,"head_only")
    assert rt.main_optimizer_parameters() == []
    optimizer = torch.optim.AdamW(model.parameters(),lr=.01)
    update(model,rt,ctx,optimizer)
    state = copy.deepcopy(rt.state_dict())
    assert state["auxiliary_transaction"]["optimizer"]["state"]
    rt.load_state_dict(state)
    assert rt.sampler.scheduler.last_finished_step == 0
    assert rt.auxiliary_transaction.state_dict()["optimizer"]["param_groups"] == state["auxiliary_transaction"]["optimizer"]["param_groups"]


def test_legacy_gate_cannot_open_v2_identity(tmp_path):
    _,rt,_ = runtime(tmp_path,"U2")
    rt.gate.opened = True
    assert not rt.joint_open
    assert rt.activation_report()["mechanism_gate"]["status"] == "SOURCE_PARAMETERS_UNFROZEN"


@pytest.mark.parametrize("variant",["U3_delta","U3_delta_pairs","U4_decomposed"])
def test_unfrozen_source_candidates_rejected(variant):
    with pytest.raises(ValueError,match="UNFROZEN"):
        validate_runtime_config(resolve_variant(configuration(),variant))


def test_v2_u3_vectorized_real_runtime_and_parameter_audit(tmp_path):
    model,rt,ctx = runtime(tmp_path,"U3")
    update(model,rt,ctx,torch.optim.AdamW(model.parameters(),lr=.01))
    assert rt.counts["successful_steps"] == 1
    assert any(v["kind"] == "weighted_decision_parameter_gradients" for v in rt.detailed_audits)
    assert (tmp_path / "cross_response_diagnostics.jsonl").exists()


def test_extension_scope_cannot_be_reopened_by_head_only_evidence(tmp_path):
    from test_cross_response_v2_geometry_gate import make_gate, evidence
    config = make_gate(1).config
    config["terminal_identity_module"] = "time_fuse"
    model,rt,_ = runtime(tmp_path,"U2",mechanism_gate=config)
    def local_evidence(index, scope='cls_head'):
        row = evidence(index, scope)
        row['update_value'].update(training_physical_ids=list(rt.source_contract['train'])[:2],
                                  query_physical_ids=list(rt.source_contract['validation'])[:4])
        return row
    assert rt.observe_source_mechanism(model,local_evidence(0,"time_fuse"),extension="time_fuse")["authorized"]
    assert rt.joint_open
    assert any(n.startswith("id_backbone.time_fuse.") for n,_ in rt.roles["identity_tail"])
    assert rt.observe_source_mechanism(model,local_evidence(1))["authorized"]
    assert rt.joint_open and rt.extended_identity_module is None
    assert not any(n.startswith("id_backbone.time_fuse.") for n,_ in rt.roles["identity_tail"])
    # Even inconsistent external state cannot turn head-scope evidence into an extension.
    rt.extended_identity_module = "time_fuse"
    assert not rt.joint_open
