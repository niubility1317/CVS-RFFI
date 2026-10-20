import copy,json
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
from experiments.pamf_r44_pruned_refine.models import CompactSuccessor as Parent
from experiments.pamf_r46_prior_preserve import design as d,models as m,runtime as r,evaluation as e,runner
from experiments.pamf_uncertainty_restore import design as ancestor
torch.set_num_threads(2)

def lineage():return dict(coarse_run=ancestor.COARSE_RUN,coarse_commit=ancestor.COARSE_COMMIT,action_run=ancestor.ACTION_RUN,action_commit=ancestor.ACTION_COMMIT,data_contract='original7/63/30',training_TX_F_reads=0,target_training=False)

def parent_model():
    torch.manual_seed(19);value=Parent('prune_refine').eval().requires_grad_(False)
    with torch.no_grad():
        for core in value.prior:core.out.weight.normal_(std=.001)
        value.frozen.coarse.temporal.out.weight.normal_(std=.001);value.frozen.estimator.core.out.weight.normal_(std=.001);value.frozen.estimator.global_head.weight.normal_(std=.001)
    return value

@pytest.mark.parametrize('arm',d.ARMS)
def test_only_prior_updates_exact_context_and_standalone_source_final(tmp_path,arm):
    parent=parent_model();before=copy.deepcopy(parent.state_dict());model=m.compact_from_parent(parent,arm);model.initialization_report['seconds']=0.;r.validate_aliases(model)
    assert sum(p.numel() for p in model.parameters())==d.PARAMETERS[arm]<350521
    assert sum(p.numel() for p in model.parameters() if p.requires_grad)==d.PRIOR_PARAMETERS[arm]
    assert sum(p.numel() for p in model.frozen.parameters())==237927
    assert all(torch.equal(v,parent.frozen.state_dict()[k]) for k,v in model.frozen.state_dict().items())
    model.train();assert not model.frozen.training and model.prior.training
    clean=(torch.randn(2,2,256)*.1).requires_grad_(True);received=clean.detach()+torch.randn_like(clean)*.01
    student,teacher,conditional=m.paired_outputs(model,parent,received)
    with torch.no_grad():assert all(torch.equal(a,b) for a,b in zip(student,model.stage_outputs(received)[0]))
    teacher=[v.detach().requires_grad_(True) for v in teacher]
    loss,terms=m.objective(student,teacher,clean,20,arm,conditional)
    assert torch.allclose(loss,terms['iq_nmse']+.05*terms['spectral_error']+terms['teacher_stage_nmse']+.05*terms['conditional_prior_nmse'])
    loss.mean().backward()
    assert all(p.grad is None and not p.requires_grad for p in model.frozen.parameters())
    assert clean.grad is None and all(p.grad is None for p in teacher) and all(p.grad is None for p in parent.parameters())
    for core in model.prior:
        grads=[p.grad for p in core.parameters() if p.grad is not None];assert grads and all(torch.isfinite(g).all() for g in grads) and any(g.abs().sum()>0 for g in grads)
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=2e-4);optimizer.step()
    assert all(torch.equal(v,parent.frozen.state_dict()[k]) for k,v in model.frozen.state_dict().items())
    assert all(torch.equal(v,before[k]) for k,v in parent.state_dict().items())
    model.eval();path=tmp_path/'final.pt';r.save_checkpoint(path,model,None,20,4940,lineage());fresh,value=r.load_checkpoint(path,'cpu')
    assert value['optimizer'] is None and value['cumulative_coarse_epochs']==80 and value['cumulative_prior_epochs']==100 and value['cumulative_component_epoch_sum']==260 and not value['training_F_access']
    with torch.no_grad():assert torch.equal(model(received),fresh(received))
    with pytest.raises(FileExistsError):r.save_checkpoint(path,model,None,20,4940,lineage())

@pytest.mark.parametrize('field,bad',[('parent_R46_row','retain_refine'),('parent_R46_commit','unknown'),('training_F_access',True),('F_outputs_read',1),('target_training',True),('V_training',True),('updates',4941),('epoch',21),('cumulative_coarse_epochs',100)])
def test_source_provenance_budget_and_permissions_reject(field,bad):
    model=m.compact_from_parent(parent_model(),d.ARMS[0]);model.initialization_report['seconds']=0.;value=r.metadata(d.ARMS[0],20,4940,lineage(),model.initialization_report);r.validate_payload(value);value[field]=bad
    with pytest.raises(ValueError):r.validate_payload(value)

def test_shared_factor_conflict_and_invalid_arm_before_parent_IO(monkeypatch):
    model=m.CompactSuccessor(d.ARMS[0]);state=copy.deepcopy(model.state_dict());key=next(k for k in state if k.startswith('prior.1.') and k.endswith('left.weight'));state[key]=state[key]+1
    with pytest.raises(ValueError,match='Conflicting'):r.validate_aliases(model,state)
    monkeypatch.setattr(r,'validate_parent_root',lambda *a:pytest.fail('ParentIO forbidden'))
    with pytest.raises(ValueError):r.initialize('factor625_stage','/runs','cpu')

@pytest.mark.parametrize('bad',['root','role','arm','kind','plan'])
def test_target_input_rejected_before_arrays(monkeypatch,bad):
    args=SimpleNamespace(output=Path('/runs')/d.RUN/d.ARMS[0],kind=d.KIND);role='target'
    if bad=='root':args.output=Path('/runs/other')/d.ARMS[0]
    elif bad=='role':role='query'
    elif bad=='arm':args.output=args.output.parent/'factor625_cond'
    elif bad=='kind':args.kind='other'
    plan=copy.deepcopy(d.configuration());plan['updates']=19700;monkeypatch.setattr(e,'read_json',lambda *a:plan);monkeypatch.setattr(e.np,'load',lambda *a,**k:pytest.fail('Target arrays stayclosed'))
    with pytest.raises(ValueError):e.reference(args,role)

@pytest.mark.parametrize('arm',d.ARMS)
def test_every_version_gets_all_formal_after_source_without_source_score_gate(tmp_path,monkeypatch,arm):
    args=SimpleNamespace(output=tmp_path,arm=arm);(tmp_path/arm).mkdir();seen=[]
    monkeypatch.setattr(runner,'train',lambda *a:seen.append('train'));monkeypatch.setattr(runner,'predict',lambda a,role:seen.append('predict_'+role));monkeypatch.setattr(runner,'profile',lambda *a:seen.append('profile'));monkeypatch.setattr(runner,'score_subprocess',lambda a,role:seen.append('score_'+role));monkeypatch.setattr(runner.torch.cuda,'empty_cache',lambda:None)
    runner.worker(args);assert seen==['train','predict_source','profile','score_source','predict_target','score_target'] and (tmp_path/arm/'complete.json').exists()

def test_independent_truth_never_precedes_all_prediction_files(monkeypatch):
    args=SimpleNamespace(output=Path('/runs')/d.RUN,arm=d.ARMS[0]);monkeypatch.setenv('CUDA_VISIBLE_DEVICES','');monkeypatch.setattr(e,'reference',lambda *a:(Path('/formal'),e.np.arange(2),None,None))
    class Missing(Exception):pass
    def incomplete(*a):raise Missing()
    monkeypatch.setattr(e.base,'prediction_arrays',incomplete);monkeypatch.setattr(e.base,'target_baseline',lambda *a:pytest.fail('Truth staysclosed'))
    with pytest.raises(Missing):e.score(args,'target')

def test_shared_context_single_forward_conditional_teacher_detached_and_no_hooks():
    parent=parent_model();model=m.compact_from_parent(parent,d.ARMS[0]);calls=[];inputs=[];handles=[]
    handles.append(parent.frozen.coarse.register_forward_hook(lambda *a:calls.append('parent_context')))
    handles.append(model.frozen.coarse.register_forward_hook(lambda *a:calls.append('student_context')))
    for core in parent.prior:handles.append(core.register_forward_hook(lambda module,args,output:inputs.append(args[0].requires_grad)))
    output,desired,conditional=m.paired_outputs(model,parent,torch.randn(2,2,256)*.1)
    for handle in handles:handle.remove()
    assert calls==['parent_context'] and inputs==[False]*6 and conditional.requires_grad and torch.isfinite(conditional).all()
    assert all(not core._forward_hooks for core in model.prior)
    with pytest.raises(ValueError):m.objective(output,desired,torch.zeros(2,2,256),1,d.ARMS[0],None)

def test_live_actual_predecessor_blocks_next_GPU_worker(tmp_path,monkeypatch):
    arm=d.ARMS[0];previous=tmp_path/d.PREDECESSOR;previous.mkdir();(previous/'old').mkdir();(previous/'workers.json').write_text(json.dumps(dict(workers=[dict(arm='old',pid=123,GPU=d.GPUS[arm])])),encoding='utf-8');args=SimpleNamespace(output=tmp_path/d.RUN)
    assert not runner.predecessor_ready(args,arm);(previous/'old/complete.json').write_text('{}',encoding='utf-8')
    class Proc:
        state='S';owned=True
        def __truediv__(self,key):self.key=key;return self
        def read_text(self):return '123 (python) '+self.state+' unused'
        def read_bytes(self):return chr(0).join([str(previous),d.PREDECESSOR_MODULE[d.PREDECESSOR]] if self.owned else ['unrelated']).encode()
    proc=Proc();monkeypatch.setattr(runner,'Path',lambda v:proc if v=='/proc' else Path(v));assert not runner.predecessor_ready(args,arm);proc.state='Z';assert runner.predecessor_ready(args,arm);proc.state='S';proc.owned=False;assert runner.predecessor_ready(args,arm)

@pytest.mark.parametrize('loss,accepted',[(1344,True),(1345,False)])
def test_exact08pp_scene_boundary(loss,accepted):
    def summary(n):return dict(role='target',count=168000,views={v:dict(count=168000,correct=n,accuracy=100*n/168000) for v in d.VIEWS})
    a=e.compact_acceptance(summary(140000-loss),summary(140000));assert a['accepted']==accepted and a['maximum_correct_loss']==1344

def test_fixed_budget_loss_and_source_exposure_configuration():
    cfg=d.configuration();assert json.loads(json.dumps(cfg))==cfg
    assert d.EPOCHS==20 and d.UPDATES==4940 and d.BATCH==d.MICRO==256
    assert d.PARAMETERS==dict(prior625_cond=329404,prior750_cond=343974)
    assert cfg['cumulative_coarse_epochs']==cfg['cumulative_action_epochs']==80 and cfg['cumulative_prior_epochs']==100 and cfg['cumulative_component_epoch_sum']==260
    assert d.teacher_weight(1)==d.teacher_weight(20)==1.
    assert d.learning_rate(0)==2e-4/247 and d.learning_rate(4939)==pytest.approx(2e-5)
    assert cfg['TX_labels_read']==cfg['F_weights_read']==cfg['F_outputs_read']==cfg['student_context_inputs']==0
