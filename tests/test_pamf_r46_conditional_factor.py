import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from experiments.pamf_r44_pruned_refine.models import CompactSuccessor as ParentModel
from experiments.pamf_r46_conditional_factor import design as d,models as m,runtime as r,evaluation as e,runner
from experiments.pamf_uncertainty_restore import design as ancestor

torch.set_num_threads(2)


def lineage():
    return dict(coarse_run=ancestor.COARSE_RUN,coarse_commit=ancestor.COARSE_COMMIT,
        action_run=ancestor.ACTION_RUN,action_commit=ancestor.ACTION_COMMIT,
        data_contract='original7/63/30',training_TX_F_reads=0,target_training=False)


def parent_model():
    model=ParentModel('prune_refine').eval().requires_grad_(False)
    with torch.no_grad():
        for core in model.prior:core.out.weight.normal_(std=.001)
        model.frozen.coarse.temporal.out.weight.normal_(std=.001)
        model.frozen.estimator.core.out.weight.normal_(std=.001)
        model.frozen.estimator.global_head.weight.normal_(std=.001)
    return model


@pytest.mark.parametrize('arm',d.ARMS)
def test_existing_reduced_models_joint_gradients_detached_teacher_and_standalone_E20(tmp_path,arm):
    torch.manual_seed(43);parent=parent_model();before=copy.deepcopy(parent.state_dict())
    model=m.compact_from_parent(parent,arm);model.initialization_report['seconds']=0.
    r.validate_aliases(model)
    assert sum(p.numel() for p in model.parameters())==d.PARAMETERS[arm]<=350521
    assert sum(p.numel() for p in model.frozen.parameters())==d.CONTEXT_PARAMETERS[arm]
    clean=(torch.randn(2,2,256)*.1).requires_grad_(True);received=clean.detach()+torch.randn_like(clean)*.005
    outputs,desired,conditional=m.paired_outputs(model,parent,received)
    desired=[v.detach().requires_grad_(True) for v in desired]
    loss,terms=m.objective(outputs,desired,clean,1,arm,conditional)
    assert torch.allclose(loss,terms['iq_nmse']+.05*terms['spectral_error']+terms['teacher_stage_nmse']+d.CONDITIONAL_WEIGHT[arm]*terms['conditional_prior_nmse'])
    loss.mean().backward()
    for group in (model.frozen.coarse,model.frozen.estimator,*model.prior):
        grads=[p.grad for p in group.parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads) and any(g.abs().sum()>0 for g in grads)
    assert clean.grad is None and all(v.grad is None for v in desired)
    assert all(p.grad is None for p in parent.parameters()) and all(torch.equal(v,before[k]) for k,v in parent.state_dict().items())
    optimizer=torch.optim.AdamW(model.parameters(),lr=2e-4);optimizer.step()
    model.eval();path=tmp_path/'final.pt';r.save_checkpoint(path,model,None,20,19700,lineage())
    fresh,value=r.load_checkpoint(path,'cpu')
    assert value['optimizer'] is None and value['cumulative_component_epochs']==100 and not value['training_F_access']
    with torch.no_grad():assert torch.equal(model(received),fresh(received))
    with pytest.raises(FileExistsError):r.save_checkpoint(path,model,None,20,19700,lineage())


@pytest.mark.parametrize('field,bad',[('parent_R46_row','retain_refine'),('parent_R46_commit','unknown'),
    ('training_F_access',True),('F_outputs_read',1),('target_training',True),('V_training',True),('updates',19701),('epoch',21)])
def test_checkpoint_lineage_permissions_and_budget_reject(field,bad):
    model=m.compact_from_parent(parent_model(),d.ARMS[0]);model.initialization_report['seconds']=0.
    value=r.metadata(d.ARMS[0],20,19700,lineage(),model.initialization_report);r.validate_payload(value);value[field]=bad
    with pytest.raises(ValueError):r.validate_payload(value)


def test_alias_conflict_and_invalid_arm_reject_before_parent_IO(monkeypatch):
    model=m.CompactSuccessor(d.ARMS[1]);state=copy.deepcopy(model.state_dict())
    key=next(k for k in state if k.startswith('prior.1.') and k.endswith('net.0.left.weight'));state[key]=state[key]+1
    with pytest.raises(ValueError,match='Conflicting'):r.validate_aliases(model,state)
    monkeypatch.setattr(r,'validate_parent_root',lambda *a:pytest.fail('No parent IO'))
    with pytest.raises(ValueError):r.initialize('keep_iq','/runs','cpu')


@pytest.mark.parametrize('bad',['root','role','arm','kind','plan'])
def test_invalid_target_reference_stops_before_arrays(monkeypatch,bad):
    args=SimpleNamespace(output=Path('/runs')/d.RUN/d.ARMS[0],kind=d.KIND);role='target'
    if bad=='root':args.output=Path('/runs/other')/d.ARMS[0]
    elif bad=='role':role='query'
    elif bad=='arm':args.output=args.output.parent/'keep_iq'
    elif bad=='kind':args.kind='other'
    plan=copy.deepcopy(d.configuration());plan['updates']=2470
    monkeypatch.setattr(e,'read_json',lambda *a:plan)
    monkeypatch.setattr(e.np,'load',lambda *a,**k:pytest.fail('Arrays stay closed'))
    with pytest.raises(ValueError):e.reference(args,role)


@pytest.mark.parametrize('arm',d.ARMS)
def test_each_fixed_width_always_gets_formal_seven_predictions_after_source(tmp_path,monkeypatch,arm):
    args=SimpleNamespace(output=tmp_path,arm=arm);(tmp_path/arm).mkdir();seen=[]
    monkeypatch.setattr(runner,'train',lambda *a:seen.append('train'))
    monkeypatch.setattr(runner,'predict',lambda a,role:seen.append('predict_'+role))
    monkeypatch.setattr(runner,'profile',lambda *a:seen.append('profile'))
    monkeypatch.setattr(runner,'score_subprocess',lambda a,role:seen.append('score_'+role))
    monkeypatch.setattr(runner.torch.cuda,'empty_cache',lambda:None);runner.worker(args)
    assert seen==['train','predict_source','profile','score_source','predict_target','score_target']
    assert (tmp_path/arm/'complete.json').exists()


def test_truth_side_waits_for_all_prediction_artifacts(monkeypatch):
    args=SimpleNamespace(output=Path('/runs')/d.RUN,arm=d.ARMS[0]);monkeypatch.setenv('CUDA_VISIBLE_DEVICES','')
    monkeypatch.setattr(e,'reference',lambda *a:(Path('/formal'),e.np.arange(2),None,None))
    class Missing(Exception):pass
    def incomplete(*a):raise Missing()
    monkeypatch.setattr(e.base,'prediction_arrays',incomplete)
    monkeypatch.setattr(e.base,'target_baseline',lambda *a:pytest.fail('No truth before predictions'))
    with pytest.raises(Missing):e.score(args,'target')


def test_new_E20_checkpoint_reaches_reader_only_after_budget_validation(monkeypatch):
    args=SimpleNamespace(output=Path('/runs')/d.RUN,arm=d.ARMS[0],reader_folder=Path('/reader'),source_cache=Path('/cache'))
    class Model:
        def requires_grad_(self,value):return self
    class ReachedReader(Exception):pass
    monkeypatch.setattr(e,'reference',lambda *a:(Path('/received'),e.np.arange(2),None,None))
    monkeypatch.setattr(e,'load_checkpoint',lambda *a:(Model(),dict(arm=d.ARMS[0],epoch=20,updates=19700,kind=d.KIND)))
    def reader(*a,**k):raise ReachedReader()
    monkeypatch.setattr(e.base,'load_reader',reader)
    with pytest.raises(ReachedReader):e.predict(args,'source')


def test_queue_preserves_live_predecessor_and_uses_actual_owned_handle(tmp_path,monkeypatch):
    arm=d.ARMS[0];previous=tmp_path/d.PREDECESSOR;previous.mkdir();(previous/'old').mkdir()
    (previous/'workers.json').write_text(json.dumps(dict(workers=[dict(arm='old',pid=123,GPU=d.GPUS[arm])])),encoding='utf-8')
    args=SimpleNamespace(output=tmp_path/d.RUN);assert not runner.predecessor_ready(args,arm)
    (previous/'old/complete.json').write_text('{}',encoding='utf-8')
    class Proc:
        state='S';owned=True
        def __truediv__(self,key):self.key=key;return self
        def read_text(self):return '123 (python) '+self.state+' unused'
        def read_bytes(self):return chr(0).join([str(previous),d.PREDECESSOR_MODULE[d.PREDECESSOR]] if self.owned else ['unrelated']).encode()
    proc=Proc();monkeypatch.setattr(runner,'Path',lambda value:proc if value=='/proc' else Path(value))
    assert not runner.predecessor_ready(args,arm)
    proc.state='Z';assert runner.predecessor_ready(args,arm)
    proc.state='S';proc.owned=False;assert runner.predecessor_ready(args,arm)


def test_finite_source_failure_preserves_and_does_not_retry_or_touch_target(tmp_path,monkeypatch):
    args=SimpleNamespace(output=tmp_path,arm=d.ARMS[0])
    def fail(*a):raise FloatingPointError('nonfinite')
    monkeypatch.setattr(runner,'train',fail);monkeypatch.setattr(runner,'predict',lambda *a:pytest.fail('No prediction'))
    with pytest.raises(FloatingPointError):runner.worker(args)
    assert (tmp_path/args.arm/'failure.json').exists()


def test_fixed_source_budget_and_conditional_only_matched_difference():
    cfg=d.configuration();assert json.loads(json.dumps(cfg))==cfg
    assert set(d.PARAMETERS.values())=={278180}
    assert d.EPOCHS==20 and d.UPDATES==19700 and cfg['cumulative_component_epochs']==100
    assert cfg['F_weights_read']==cfg['F_outputs_read']==cfg['TX_labels_read']==0
    assert d.teacher_weight(1)==1 and d.teacher_weight(20)==pytest.approx(.1)
    assert d.learning_rate(0)==2e-4/985 and d.learning_rate(19699)==pytest.approx(2e-5)
    assert d.CONDITIONAL_WEIGHT==dict(factor625_stage=0.,factor625_cond=.05)
    with pytest.raises(ValueError):d.teacher_weight(21)


def test_factors_preserve_channels_GN_and_matched_initialization():
    from experiments.pamf_r17_compact.models import FactorConv
    parent=parent_model();before=copy.deepcopy(parent.state_dict())
    a=m.compact_from_parent(parent,d.ARMS[0]);b=m.compact_from_parent(parent,d.ARMS[1])
    assert all(torch.equal(v,b.state_dict()[k]) for k,v in a.state_dict().items())
    assert sum(p.numel() for p in a.parameters())==278180<350521
    assert all(torch.equal(v,before[k]) for k,v in parent.state_dict().items())
    original=dict(parent.named_modules());new=dict(a.named_modules())
    for name,module in original.items():
        if isinstance(module,torch.nn.GroupNorm):
            assert new[name].num_groups==module.num_groups and new[name].num_channels==module.num_channels
            assert torch.equal(new[name].weight,module.weight) and torch.equal(new[name].bias,module.bias)
        if isinstance(new.get(name),FactorConv):
            assert new[name].left.in_channels==module.in_channels and new[name].right.out_channels==module.out_channels
            assert torch.equal(new[name].right.bias,module.bias)


def test_spectral_term_detects_amplitude_distortion_keeps_clean_gradient_closed_and_zero_IQ_finite():
    torch.manual_seed(46);clean=(torch.randn(2,2,256)*.1).requires_grad_(True)
    assert torch.equal(m.spectral_error(clean.detach(),clean),torch.zeros(2))
    restored=(clean.detach()*1.5).requires_grad_(True);loss=m.spectral_error(restored,clean)
    assert (loss>0).all();loss.mean().backward()
    assert clean.grad is None and torch.isfinite(restored.grad).all() and restored.grad.abs().sum()>0
    zeros=torch.zeros(2,2,256,requires_grad=True);zero=m.spectral_error(zeros,zeros.detach())
    zero.sum().backward();assert torch.isfinite(zero).all() and torch.isfinite(zeros.grad).all()


@pytest.mark.parametrize('arm',d.ARMS)
def test_shared_factors_keep_step_specific_bias_norm(arm):
    model=m.compact_from_parent(parent_model(),arm)
    a=model.prior[0].encoder.blocks[0][0].net;b=model.prior[1].encoder.blocks[0][0].net
    assert a[0].left.weight is b[0].left.weight and a[0].right.weight is b[0].right.weight
    assert a[0].right.bias is not b[0].right.bias and a[1].weight is not b[1].weight


def test_conditional_teacher_uses_detached_actual_student_inputs_and_has_no_persistent_hook():
    parent=parent_model();model=m.compact_from_parent(parent,'factor625_cond');seen=[];handles=[]
    def record(module,inputs,output):seen.append(inputs[0].requires_grad)
    handles=[p.register_forward_hook(record) for p in parent.prior]
    outputs,desired,conditional=m.paired_outputs(model,parent,torch.randn(2,2,256)*.1)
    for handle in handles:handle.remove()
    assert len(seen)==6 and not any(seen) and conditional.requires_grad
    assert conditional.shape==(2,) and torch.isfinite(conditional).all()
    assert all(not p._forward_hooks for p in model.prior)
    with pytest.raises(ValueError):m.objective(outputs,desired,torch.zeros(2,2,256),1,'factor625_cond',None)
    with pytest.raises(ValueError):m.objective(outputs,desired,torch.zeros(2,2,256),1,'factor625_stage',conditional)


@pytest.mark.parametrize('loss,accepted',[(1344,True),(1345,False)])
def test_exact_per_scene_08pp_boundary(loss,accepted):
    def summary(correct):return dict(role='target',count=168000,views={v:dict(count=168000,correct=correct,accuracy=100*correct/168000) for v in d.VIEWS})
    result=e.compact_acceptance(summary(140000-loss),summary(140000))
    assert result['accepted']==accepted and result['maximum_correct_loss']==1344 and result['allowance_pp']==.8
