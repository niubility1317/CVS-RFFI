"""Bounded CPU tests of configured mechanisms reaching the real runtime."""
import copy
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cvsrffi.cross_response.config import validate_runtime_config
from cvsrffi.cross_response.integration import CrossResponseRuntime
from cvsrffi.cross_response.training import _grad, response_backward
from scripts.core90_cross_response_matrix import load_config, resolve_variant, DEFAULT_CONFIG


class SourceDataset:
    def __init__(self, validation=False, events=False):
        self.split_source = "ssdg_source_v_select" if validation else "ssdg_labeled_tx_visible"
        self.index = [SimpleNamespace(tx_i=t,rx_i=r,day_i=0,eq_i=0,
            sig_i=s+(20 if validation else 0),
            event_id=f"packet-{validation}-{t}-{r}-{s}" if events else None)
            for t in range(4) for r in range(4) for s in range(2)]

    def __len__(self):
        return len(self.index)

    def raw_iq(self,index):
        row=self.index[index]
        time=torch.arange(32.)
        return torch.stack((torch.cos(time*(row.tx_i+1)/8),
                            torch.sin(time*(row.rx_i+1)/8)))*(row.rx_i+1)*(1+row.sig_i*.02)

    def __getitem__(self,index):
        row=self.index[index]
        x=self.raw_iq(index)
        return x/x.square().mean().sqrt(),row.tx_i,row.rx_i,{}


class Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.tail=nn.Linear(2,3)
        self.cls_head=nn.Module()
        self.cls_head.head=nn.Linear(3,4)


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.id_backbone=Backbone()
        self.dom_backbone=Backbone()

    def forward(self,x):
        zi=self.id_backbone.tail(x.mean(-1))
        zd=self.dom_backbone.tail(x.mean(-1))
        return dict(z_id=zi,z_dom=zd,tx_logits=self.id_backbone.cls_head.head(zi))


def config(family="iq"):
    c=resolve_variant(load_config(DEFAULT_CONFIG),"U2")
    c.update(target_family=family,target_dim={"iq":5,"autocorr":6,"event":6}[family],
             gradient_tail_prefixes=["id_backbone.tail"],source_fit_max_records=32,
             source_eval_max_blocks=1,gate_min_blocks=1,event_windows=[[0,8],[8,16]])
    return validate_runtime_config(c)


def make_runtime(tmp_path,family="iq",events=False):
    artifact=tmp_path/"synthetic-source.bin"
    artifact.write_bytes(b"bounded-synthetic-runtime-fixture")
    model=TinyModel()
    ctx=dict(train_loader=DataLoader(SourceDataset(events=events),batch_size=32),
             val_loader=DataLoader(SourceDataset(validation=True,events=events),batch_size=32),input_len=32)
    args=SimpleNamespace(cross_response_variant="U2",wisig_train_rxs="0,1,2,3",wisig_test_rxs="4",
                         wisig_train_days="0",wisig_test_days="1",output_dir=str(tmp_path),
                         batch_size=32,eval_batch_size=16,seed=11,wisig_pkl=str(artifact))
    runtime=CrossResponseRuntime(model,ctx,config(family),args,torch.device("cpu"))
    return model,ctx,runtime


def prepare(model,ctx,runtime):
    x,y,d,meta=next(iter(ctx["train_loader"]))
    plan=runtime.begin_batch((d,meta))
    assert plan.blocks
    out=model(x)
    terms=runtime.losses(model,out,y,plan)
    return out,terms,plan


@pytest.mark.parametrize("family,dim",[("autocorr",6),("iq",5),("event",6)])
def test_optional_targets_reach_runtime_prediction_and_gradients(tmp_path,family,dim):
    model,ctx,runtime=make_runtime(tmp_path,family,events=family=="event")
    out,terms,plan=prepare(model,ctx,runtime)
    assert runtime.normalizer.location.numel()==dim
    assert runtime.auxiliary["predictor"].bias.shape==(dim,)
    raw=torch.stack([runtime.source_dataset.raw_iq(r.index) for r in plan.blocks[0].records])
    actual=runtime.statistics(raw,event_metadata=runtime._event(plan.blocks[0].records))
    assert actual.shape==(32,dim) and torch.isfinite(actual).all()
    assert not actual.requires_grad
    assert all(torch.isfinite(term) for term in terms)
    terms[0].backward()
    assert any(p.grad is not None and p.grad.norm()>0 for p in runtime.parameters())
    assert any(p.grad is not None and p.grad.norm()>0 for p in model.dom_backbone.parameters())
    assert all(p.grad is None for p in model.id_backbone.parameters())


def test_event_target_requires_real_metadata_during_runtime_initialization(tmp_path):
    with pytest.raises(ValueError,match="event"):
        make_runtime(tmp_path,"event",events=False)


@pytest.mark.parametrize("failure",["optimizer_skipped","nonfinite_gradient"])
def test_failed_step_cannot_commit_sampler_feedback_or_success_activation(tmp_path,failure):
    model,ctx,runtime=make_runtime(tmp_path)
    out,terms,plan=prepare(model,ctx,runtime)
    before=copy.deepcopy(runtime.sampler.scheduler.state_dict())
    runtime.logs.update(response_grad_auxiliary=1.,response_grad_domain=1.,response_grad_identity_tail=1.)
    if failure=="nonfinite_gradient":
        parameter=runtime.parameters()[0]
        parameter.grad=torch.full_like(parameter,float("nan"))
        assert runtime.finite_gradients() is not None
    runtime.commit(False)
    assert runtime.sampler.scheduler.state_dict()==before
    assert runtime.sampler.scheduler.coverage.exposed_physical
    report=runtime.activation_report()
    assert all(count==0 for name,count in report["counts"].items() if name!="batches")
    assert report["status"]=="ACTIVATION_INCOMPLETE"
    assert "NO_EFFECTIVE_CROSS_BLOCKS" in report["missing"]


def test_nonfinite_feedback_cannot_claim_successful_activation(tmp_path):
    model,ctx,runtime=make_runtime(tmp_path)
    prepare(model,ctx,runtime)
    block,response,decision,reliability,noise,valid=runtime.pending_blocks[0]
    runtime.pending_blocks[0]=(block,float("nan"),decision,reliability,noise,valid)
    runtime.commit(True)
    assert not runtime.sampler.scheduler.coverage.joint
    assert not runtime.sampler.scheduler.history and not runtime.sampler.scheduler.pending
    # An already-applied optimizer step remains real; poisoned block telemetry
    # must never become evidence that the auxiliary mechanism activated.
    assert runtime.counts["successful_steps"]==1
    assert runtime.counts["effective_blocks"]==0
    assert runtime.counts["response_batches"]==0


@pytest.mark.parametrize("flag,weight",[("response_enabled","lambda_resp"),
    ("decision_enabled","lambda_dec"),("identity_interaction_enabled","lambda_cross")])
def test_enabled_zero_weight_is_rejected(flag,weight):
    c=config()
    c.update({flag:True,weight:0.})
    with pytest.raises(ValueError,match="cannot activate"):
        validate_runtime_config(c)


def test_unknown_runtime_control_is_not_silently_ignored():
    c=config()
    c["invented_mechanism_switch"]=True
    with pytest.raises(ValueError,match="unused"):
        validate_runtime_config(c)


class FixedScaler:
    def __init__(self,value): self.value=float(value)
    def get_scale(self): return self.value
    def scale(self,loss): return loss*self.value


def test_scaled_autograd_survives_fp16_intermediate_underflow():
    parameter=nn.Parameter(torch.tensor([.5],dtype=torch.float32))
    response=parameter.to(torch.float16).float().sum()*1e-9
    unscaled=_grad(response,[parameter],scale=1.)[0]
    scaled=_grad(response,[parameter],scale=65536.)[0]
    assert unscaled.item()==0.  # Scaling after this traversal cannot recover it.
    assert scaled.item()==pytest.approx(1e-9,rel=.001)
    roles={key:[] for key in ("shared","identity_tail","identity_front","domain","other","auxiliary")}
    roles["domain"]=[("domain",parameter)]
    zero=parameter.sum()*0.
    logs=response_backward(baseline_loss=zero,identity_loss=zero,response_loss=response,
        decision_loss=zero,cross_loss=zero,roles=roles,scaler=FixedScaler(65536.),
        lambda_resp=1.,lambda_dec=0.,lambda_cross=0.,joint_open=False,gradient_cap=.2)
    assert logs["response_grad_domain"]==pytest.approx(1e-9,rel=.001)
    assert (parameter.grad/65536.).item()==pytest.approx(1e-9,rel=.001)


def test_amp_scale_does_not_change_fp32_routed_norm_or_cap():
    outputs=[]
    for scale in (1.,65536.):
        parameter=nn.Parameter(torch.tensor([.3],dtype=torch.float32))
        roles={key:[] for key in ("shared","identity_tail","identity_front","domain","other","auxiliary")}
        roles["identity_tail"]=[("tail",parameter)]
        loss=parameter.square().sum()
        zero=parameter.sum()*0.
        logs=response_backward(baseline_loss=loss,identity_loss=loss,response_loss=loss,
            decision_loss=zero,cross_loss=zero,roles=roles,scaler=FixedScaler(scale),
            lambda_resp=.4,lambda_dec=0.,lambda_cross=0.,joint_open=True,gradient_cap=.2)
        outputs.append((logs,parameter.grad/scale))
    assert outputs[0][0]==pytest.approx(outputs[1][0])
    torch.testing.assert_close(outputs[0][1],outputs[1][1],rtol=0,atol=0)
    assert outputs[0][0]["response_identity_cap_scale"]==pytest.approx(.5)


@pytest.mark.parametrize("name",["proxy_unknown_energy_loss","source_episode_three_sigma_loss"])
def test_legacy_geometry_wrapper_cpu_autocast_exact_loss_and_gradient(name):
    from cvsrffi.cross_response import baseline_compat as compat
    generator=torch.Generator().manual_seed(73)
    initial=torch.randn(24,12,generator=generator)
    labels=torch.arange(4).repeat_interleave(6)
    domains=torch.arange(3).repeat(8)
    outcomes=[]
    for amp in (False,True):
        z=initial.clone().requires_grad_()
        torch.manual_seed(43)
        with torch.autocast(device_type="cpu",dtype=torch.bfloat16,enabled=amp):
            if name=="proxy_unknown_energy_loss":
                loss,_=getattr(compat,name)(z,labels,virtual_count=12,virtual_mode="hard",
                    virtual_detach=False,vaccept_weight=1.,core_accept_weight=.45,
                    component_gate_weight=.65,core_quantile=.9,accept_quantile=.85)
            else:
                loss,_=getattr(compat,name)(z,labels,domains,mixup_weight=.75)
        loss.backward()
        assert loss.dtype==torch.float32 and torch.isfinite(loss)
        assert torch.isfinite(z.grad).all()
        outcomes.append((loss.detach(),z.grad.clone()))
    for actual,reference in zip(outcomes[1],outcomes[0]):
        torch.testing.assert_close(actual,reference,rtol=0,atol=0)


@pytest.mark.parametrize("change", [{"gradient_cap": 0.}, {"gate_min_blocks": 17, "source_eval_max_blocks": 16}])
def test_impossible_joint_gate_configuration_is_rejected(change):
    c=config()
    c.update(response_enabled=True, head_only=False, permanent_detach=False, **change)
    with pytest.raises(ValueError,match="gradient_cap|cannot open"):
        validate_runtime_config(c)
