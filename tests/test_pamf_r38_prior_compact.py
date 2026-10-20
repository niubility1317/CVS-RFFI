import copy
import json
import inspect
from types import SimpleNamespace
import pytest
import torch
from experiments.pamf_r38_prior_compact import design as d,training,evaluation
from experiments.pamf_r38_prior_compact.models import CompactSuccessor,compact_from_parent,paired_stages,objective
from experiments.pamf_iq_joint_compact.models import JointCompactRestorer
from experiments.pamf_r38_prior_compact.runtime import metadata,validate_payload,save_checkpoint,load_checkpoint,validate_parent_root
from experiments.pamf_uncertainty_restore import design as ancestor

torch.set_num_threads(2)

def parent_metadata():
 return dict(coarse_run=ancestor.COARSE_RUN,coarse_commit=ancestor.COARSE_COMMIT,action_run=ancestor.ACTION_RUN,
  action_commit=ancestor.ACTION_COMMIT,data_contract='original7/63/30',training_TX_F_reads=0,target_training=False)

@pytest.mark.parametrize('arm',d.ARMS)
def test_complete_context_exact_frozen_and_only_shared_prior_trainable(arm):
 torch.manual_seed(7);parent=JointCompactRestorer('dense_half').eval().requires_grad_(False)
 model=compact_from_parent(parent,arm)
 assert sum(p.numel() for p in model.parameters())==d.PARAMETERS[arm]
 assert sum(p.numel() for p in model.parameters() if p.requires_grad)==d.PRIOR_PARAMETERS[arm]
 assert all(torch.equal(v,parent.frozen.state_dict()[k]) for k,v in model.frozen.state_dict().items())
 model.train();assert not model.frozen.training
 x=torch.randn(2,2,256)*.1;clean=x+.01
 context_calls=[];hook=parent.frozen.coarse.register_forward_hook(lambda *args:context_calls.append(1))
 outputs,desired=paired_stages(model,parent,x);hook.remove()
 assert len(context_calls)==1
 with torch.no_grad():expected=model.stage_outputs(x)[0]
 assert all(torch.equal(a,b) for a,b in zip(outputs,expected))
 loss,terms=objective(outputs,desired,clean);assert torch.isfinite(loss).all()
 loss.mean().backward()
 assert all(p.grad is None for p in model.frozen.parameters())
 assert all(p.grad is None for p in parent.parameters())
 assert all(sum(float(p.grad.norm()) for p in prior.parameters() if p.grad is not None)>0 for prior in model.prior)
 if arm=='prior_keep':assert all(torch.equal(a,b) for a,b in zip(outputs,desired))
 assert torch.allclose(loss,terms['iq_nmse']+terms['teacher_stage_nmse'])

@pytest.mark.parametrize('arm',d.ARMS)
def test_standalone_strict_reload_preserves_context_and_function(tmp_path,arm):
 parent=JointCompactRestorer('dense_half').eval().requires_grad_(False);model=compact_from_parent(parent,arm).eval()
 model.initialization_report['seconds']=0.
 path=tmp_path/'final.pt';save_checkpoint(path,model,None,10,2470,parent_metadata());fresh,payload=load_checkpoint(path,'cpu')
 x=torch.randn(1,2,256)*.1
 with torch.no_grad():assert torch.equal(model(x),fresh(x))
 assert not any(p.requires_grad for p in fresh.frozen.parameters())
 assert payload['cumulative_context_epochs']==40 and payload['cumulative_prior_epochs']==50
 assert payload['optimizer'] is None and not payload['training_F_access']

@pytest.mark.parametrize('bad',['F','target','contextbudget','wrongparent','mode'])
def test_permission_and_parent_mismatch_rejected(bad):
 parent=JointCompactRestorer('dense_half').eval().requires_grad_(False);model=compact_from_parent(parent,'prior_half')
 model.initialization_report['seconds']=0.
 value=metadata('prior_half',10,2470,parent_metadata(),model.initialization_report);validate_payload(value)
 if bad=='F':value['training_F_access']=True
 elif bad=='target':value['target_training']=True
 elif bad=='contextbudget':value['cumulative_context_epochs']=50
 elif bad=='wrongparent':value['parent_R38_row']='R13_keep_iq'
 else:value['initialization']['mode']='joint_context_narrowing'
 with pytest.raises(ValueError):validate_payload(value)

def test_training_has_no_reader_load_or_task_loss_and_budget_keeps_all_rows():
 source=inspect.getsource(training)
 assert 'load_reader' not in source and 'frozen_task_fidelity' not in source
 assert 'paired_stages(model,teacher,r)' in source
 assert 'context_updates=0,coarse_updates=0,action_updates=0' in source
 assert d.configuration()['student_context_inputs']==0
 assert not any(d.TASK.values()) and d.EPOCHS==10 and d.UPDATES==2470
 assert d.configuration()['cumulative_component_epoch_sum']==130

def test_actual_prediction_entry_explicitly_passes_ten_epoch_budget(tmp_path,monkeypatch):
 calls=[]
 monkeypatch.setattr(evaluation.base,'predict',lambda *args,**kwargs:calls.append(kwargs))
 args=SimpleNamespace(arm='prior_half',output=tmp_path/d.RUN)
 evaluation.predict(args,'source');evaluation.predict(args,'target')
 assert len(calls)==2 and all(c['checkpoint_budget']==(10,2470) for c in calls)
 with pytest.raises(ValueError):evaluation.predict(SimpleNamespace(arm='unregistered',output=tmp_path/d.RUN),'target')

def test_objective_rejects_mismatched_stage_before_any_F_access():
 x=torch.zeros(2,2,256)
 with pytest.raises(ValueError):objective([x],[x,x,x],x)

def test_actual_parent_root_interface_uses_original_R38_configuration(tmp_path):
 root=tmp_path/d.COMPACT_PARENT_RUN;(root/d.COMPACT_PARENT_ROW).mkdir(parents=True)
 (root/'launch.json').write_text(json.dumps(dict(commit=d.COMPACT_PARENT_COMMIT)),encoding='utf-8')
 (root/'comparison_plan.json').write_text(json.dumps(d.parent.configuration()),encoding='utf-8')
 trained=dict(status='SOURCE_TRAINED',arm=d.COMPACT_PARENT_ROW,epochs=20,updates=4940,L_exposures=126000,U_exposures=1134000,
  parameters=794529,TX_labels_read=0,F_weights_read=0,F_outputs_read=0,training_F_access=False,V_reads=0,target_reads=0,
  cumulative_prior_epochs=40,cumulative_coarse_epochs=40,cumulative_action_epochs=40)
 path=root/d.COMPACT_PARENT_ROW/'training_complete.json';path.write_text(json.dumps(trained),encoding='utf-8')
 validate_parent_root(root)
 trained['F_weights_read']=1;path.write_text(json.dumps(trained),encoding='utf-8')
 with pytest.raises(ValueError):validate_parent_root(root)
