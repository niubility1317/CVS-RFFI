import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from experiments.pamf_iq_joint_compact.models import JointCompactRestorer
from experiments.pamf_r38_compression_refine import design as d,models as m,runtime as r,evaluation as e,runner
from experiments.pamf_uncertainty_restore import design as ancestor

torch.set_num_threads(2)


def lineage():
    return dict(coarse_run=ancestor.COARSE_RUN,coarse_commit=ancestor.COARSE_COMMIT,
        action_run=ancestor.ACTION_RUN,action_commit=ancestor.ACTION_COMMIT,
        data_contract='original7/63/30',training_TX_F_reads=0,target_training=False)


def parent_model():
    model=JointCompactRestorer('dense_half').eval().requires_grad_(False)
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
    assert sum(p.numel() for p in model.parameters())==d.PARAMETERS[arm]<794529
    assert sum(p.numel() for p in model.frozen.parameters())==d.CONTEXT_PARAMETERS[arm]
    clean=(torch.randn(2,2,256)*.1).requires_grad_(True);received=clean.detach()+torch.randn_like(clean)*.005
    with torch.no_grad():desired=parent.stage_outputs(received)[0]
    desired=[v.detach().requires_grad_(True) for v in desired]
    outputs=model.stage_outputs(received)[0]
    loss,terms=m.objective(outputs,desired,clean,1)
    assert torch.allclose(loss,terms['iq_nmse']+10*terms['teacher_stage_nmse'])
    loss.mean().backward()
    for group in (model.frozen.coarse,model.frozen.estimator,*model.prior):
        grads=[p.grad for p in group.parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads) and any(g.abs().sum()>0 for g in grads)
    assert clean.grad is None and all(v.grad is None for v in desired)
    assert all(p.grad is None for p in parent.parameters()) and all(torch.equal(v,before[k]) for k,v in parent.state_dict().items())
    optimizer=torch.optim.AdamW(model.parameters(),lr=2e-4);optimizer.step()
    model.eval();path=tmp_path/'final.pt';r.save_checkpoint(path,model,None,20,4940,lineage())
    fresh,value=r.load_checkpoint(path,'cpu')
    assert value['optimizer'] is None and value['cumulative_component_epochs']==60 and not value['training_F_access']
    with torch.no_grad():assert torch.equal(model(received),fresh(received))
    with pytest.raises(FileExistsError):r.save_checkpoint(path,model,None,20,4940,lineage())


@pytest.mark.parametrize('field,bad',[('parent_R38_row','dense_three_quarter'),('parent_R38_commit','unknown'),
    ('training_F_access',True),('F_outputs_read',1),('target_training',True),('V_training',True),('updates',4941),('epoch',21)])
def test_checkpoint_lineage_permissions_and_budget_reject(field,bad):
    model=m.compact_from_parent(parent_model(),d.ARMS[0]);model.initialization_report['seconds']=0.
    value=r.metadata(d.ARMS[0],20,4940,lineage(),model.initialization_report);r.validate_payload(value);value[field]=bad
    with pytest.raises(ValueError):r.validate_payload(value)


def test_alias_conflict_and_invalid_arm_reject_before_parent_IO(monkeypatch):
    model=m.CompactSuccessor(d.ARMS[1]);state=copy.deepcopy(model.state_dict())
    key=next(k for k in state if k.startswith('prior.1.') and k.endswith('net.0.weight'));state[key]=state[key]+1
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
    monkeypatch.setattr(e,'load_checkpoint',lambda *a:(Model(),dict(arm=d.ARMS[0],epoch=20,updates=4940,kind=d.KIND)))
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


def test_frozen_two_width_plan_20_epochs_and_teacher_curriculum():
    cfg=d.configuration();assert json.loads(json.dumps(cfg))==cfg
    assert d.PARAMETERS==dict(three_quarter_iq=442805,half_iq=204561)
    assert d.EPOCHS==20 and d.UPDATES==4940 and cfg['cumulative_component_epochs']==60
    assert cfg['F_weights_read']==cfg['F_outputs_read']==cfg['TX_labels_read']==0
    assert d.teacher_weight(1)==10 and d.teacher_weight(20)==1
    assert d.learning_rate(0)==2e-4/247 and d.learning_rate(4939)==pytest.approx(2e-5)
    with pytest.raises(ValueError):d.teacher_weight(21)
