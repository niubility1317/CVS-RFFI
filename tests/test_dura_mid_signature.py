import json
from types import SimpleNamespace
from pathlib import Path
import numpy as np
import pytest
import torch
from experiments.pamf_reception_restore.models import Core
from experiments.pamf_dura_efficiency.mid_signature import attach,capture,auxiliary_loss,PARAMETERS
from experiments.pamf_dura_direct_train import mid_signature as m


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__();self.frozen=torch.nn.Module();self.frozen.coarse=torch.nn.Module()
        self.frozen.coarse.temporal=Core(incoming=2,widths=(12,16,28,40));self.initialization_report=dict(checkpoint_sources=[],ancestors=[])
    def forward(self,x):return self.frozen.coarse.temporal(x)


def test_projection_preserves_initial_G_and_rng_and_is_not_an_inference_branch():
    torch.set_num_threads(2);torch.manual_seed(62);model=Model();x=torch.randn(2,2,256)
    before={k:v.clone() for k,v in model.state_dict().items()};expected=model(x);rng=torch.get_rng_state().clone();attach(model,2026092701)
    assert torch.equal(rng,torch.get_rng_state()) and sum(v.numel() for v in model.mid_signature_aux.parameters())==PARAMETERS
    assert all(torch.equal(v,model.state_dict()[k]) for k,v in before.items())
    calls=[];hook=model.mid_signature_aux.register_forward_hook(lambda *args:calls.append(1))
    actual=model(x);hook.remove();torch.testing.assert_close(actual,expected,atol=0,rtol=0);assert not calls


def test_mid_loss_reaches_shared_shallow_middle_and_projection_but_detaches_teacher():
    torch.manual_seed(63);model=attach(Model(),2026092701);x=torch.randn(2,2,256);target=torch.randn(2,160,requires_grad=True)
    restored,h=capture(model,lambda:model(x));assert h.shape==(2,16,128)
    error=auxiliary_loss(model,h,target);error.mean().backward()
    for p in [model.frozen.coarse.temporal.encoder.blocks[0][0].net[0].weight,
              model.frozen.coarse.temporal.encoder.blocks[1][0].net[0].weight,model.mid_signature_aux.projection.weight]:
        assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0
    assert target.grad is None and not model.frozen.coarse.temporal.encoder.blocks[1]._forward_hooks


def test_failure_cleans_capture_and_wrong_feature_shape_is_rejected():
    model=attach(Model(),2026092701)
    with pytest.raises(RuntimeError):capture(model,lambda:(_ for _ in ()).throw(RuntimeError('native failure')))
    assert not model.frozen.coarse.temporal.encoder.blocks[1]._forward_hooks
    with pytest.raises(ValueError):model.mid_signature_aux(torch.zeros(2,16,64))
    with pytest.raises(ValueError):auxiliary_loss(model,torch.zeros(2,16,128),torch.zeros(1,160))


def test_new_middle_loss_is_weighted_once_and_original_SIG_objective_retained(monkeypatch):
    model=attach(Model(),2026092701);received=torch.randn(2,2,256);target=torch.randn(2,160)
    def original(student,*args):
        restored=student(received)
        return torch.full((2,),3.),{'old_SIG_term':torch.full((2,),2.)},restored
    monkeypatch.setattr(m,'_SIG_OBJECTIVE',original)
    loss,terms,_=m.objective(model,None,received,received,None,None,None,target)
    torch.testing.assert_close(loss,torch.full((2,),3.)+terms[m.GUIDE])
    torch.testing.assert_close(terms[m.GUIDE],.01*terms[m.METRIC]);assert 'old_SIG_term' in terms


def test_reused_SIG_cache_rejects_changed_source_permissions_before_array_read(tmp_path,monkeypatch):
    monkeypatch.setattr(m.f_guided,'source_ids',lambda *args:np.asarray(['source1','source2']))
    output=tmp_path/'runs/new';cache=output.parent/m.parent.RUN/'source_signature_cache';cache.mkdir(parents=True)
    metadata=dict(status='SOURCE_ACTUAL_SIG_COMPLETE',count=63000,shape=[63000,160],numeric_bytes=40320000,source_contract='original7/63/30',
        F_teacher=m.f_guided.teacher_reference(),actual_weighted_TripleSIG=True,full_F_inputs=0,F_prefix_inputs=63000,F_input_VJP_inputs=0,
        F_updates=0,model_optimizer_updates=0,TX_labels_read=0,V_reads=0,target_reads=0)
    for key,value in [('V_reads',1),('F_teacher',{}),('source_contract','different')]:
        (cache/'complete.json').write_text(json.dumps(dict(metadata,**{key:value})),encoding='utf-8')
        with pytest.raises(ValueError,match='Same-source'):m.reuse_targets(dict(output=output),None,None,None)


def test_actual_E30_configuration_parameters_cache_and_score_routes():
    m.configure();cfg=m.base.configuration()
    assert (cfg['parameters'],cfg['mid_signature_aux_parameters'],cfg['epochs'],cfg['updates'])==(287591,20640,30,7410)
    assert cfg['mid_signature_train_only'] and cfg['mid_signature_inference_calls']==0 and cfg['mid_signature_weight']==.01
    assert cfg['new_F_clean_cache_inputs']==0 and cfg['mid_aux_feature_inputs']==1890000 and cfg['F_prefix_training_inputs']==1890000
    assert cfg['matched_control']==m.parent.RUN and cfg['matched_control_commit']=='47591cdfe2e6e6dd7534cdea846badd942321b8f'
    args=SimpleNamespace(root=Path('/project'),commit='commit')
    for mode,role in [('worker',None),('score','source'),('score','target')]:
        command=m.base.command(args,mode,'scratch_joint',role);assert m.MODULE in command and mode in command
    assert cfg['all_versions_automatic_formal'] and cfg['target_count']==168000 and len(cfg['views'])==7
