import copy,json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from torch import nn
from experiments.pamf_r17_task import design as d,models as m,runtime,evaluation,runner,calibration as c
from experiments.pamf_action_restore.models import ActionRestorer
from experiments.pamf_reception_restore.models import ComplexTFUNet
from experiments.pamf_unrolled_restore.models import UnrolledRestorer
from experiments.pamf_uncertainty_restore import design as ancestor
from tools.r17_task_fidelity import frozen_task_fidelity

torch.set_num_threads(2)


def parent():
    return dict(coarse_run=ancestor.COARSE_RUN,coarse_commit=ancestor.COARSE_COMMIT,action_run=ancestor.ACTION_RUN,
        action_commit=ancestor.ACTION_COMMIT,data_contract='original7/63/30',training_TX_F_reads=0,target_training=False)


def info(arm,smoke=True):
    if arm=='step_norm_task':return dict(c.none_info(),observed_examples=0,eligible_layers=0,sampled_patches_min=0,sampled_patches_max=0,factor_reports=[],seconds=1.)
    value=c.smoke_info() if smoke else dict(c.smoke_info(),mode='source_train512',examples=512,L_s=55,U_s=457,received_epoch=1,selection_seed=2026102002)
    return dict(value,observed_examples=value['examples'],eligible_layers=84,sampled_patches_min=value['examples']*8,
        sampled_patches_max=value['examples']*8,factor_reports=[],seconds=1.)


@pytest.mark.parametrize('arm',d.ARMS)
def test_exact_checkpoint_inference_has_no_reader_or_teacher_dependency(tmp_path,arm):
    model=m.TaskRestorer(arm,ActionRestorer(ComplexTFUNet())).eval();model.calibration_report=info(arm)
    assert sum(p.numel() for p in model.parameters())==d.PARAMETERS[arm]
    assert all(key.startswith('inner.') for key in model.state_dict())
    path=tmp_path/'smoke.pt';runtime.save_checkpoint(path,model,None,0,0,parent())
    fresh,value=runtime.load_checkpoint(path,'cpu')
    assert value['training_F_access'] is True and value['training_F_teacher']['head_fit_updates']==0
    with torch.no_grad():
        x=torch.randn(2,2,256)*.1
        assert torch.equal(model(x),fresh(x))
    assert fresh.calibration_report==model.calibration_report
    with pytest.raises(FileExistsError):runtime.save_checkpoint(path,model,None,0,0,parent())


class Reader(nn.Module):
    def __init__(self):
        super().__init__();self.encoder=nn.Linear(512,160);self.head=nn.Linear(160,6)
    def features(self,x):return self.encoder(x.flatten(1)).tanh()
    def classify_features(self,x):return self.head(x)


def test_combined_loss_updates_only_shared_step_priors_and_keeps_both_teachers_frozen():
    torch.manual_seed(19)
    teacher=UnrolledRestorer('unroll_unshared',ActionRestorer(ComplexTFUNet())).eval().requires_grad_(False)
    with torch.no_grad():
        for core in teacher.prior:core.out.weight.normal_(std=.001)
    teacher_before=copy.deepcopy(teacher.state_dict());reader=Reader().eval().requires_grad_(False);reader_before=copy.deepcopy(reader.state_dict())
    model,_=m.compact_from_teacher(teacher,'step_norm_task',None);model.train()
    assert not model.frozen.training
    x=torch.randn(2,2,256)*.1
    with torch.no_grad():context=model.frozen.estimate(x);desired=m.stages(teacher.prior,x,context)[0]
    output=model.stage_outputs(x,context)[0];wave,terms=m.distillation_loss(output,desired,x*.9)
    task,task_terms=frozen_task_fidelity(output[-1],desired[-1],reader)
    assert torch.equal(wave,10*terms['teacher_stage_nmse'])
    assert torch.equal(task,task_terms['task_KL']+task_terms['task_feature_distance'])
    (wave+task).mean().backward()
    assert all(any(p.grad is not None and p.grad.abs().sum()>0 for p in core.parameters()) for core in model.prior)
    assert all(p.grad is None for p in model.frozen.parameters()) and all(p.grad is None for p in teacher.parameters())
    assert all(torch.equal(v,teacher_before[k]) for k,v in teacher.state_dict().items())
    assert all(p.grad is None for p in reader.parameters()) and all(torch.equal(v,reader_before[k]) for k,v in reader.state_dict().items())


@pytest.mark.parametrize('field,bad',[('training_F_access',False),('training_TX_labels_read',1),('target_training',True),('parent_R17_epoch',19)])
def test_knowledge_permissions_are_explicit_and_source_teacher_budget_cannot_change(field,bad):
    value=runtime.metadata(d.ARMS[0],20,4940,parent(),info(d.ARMS[0],False));runtime.validate_payload(value)
    with pytest.raises(ValueError):runtime.validate_payload(dict(value,**{field:bad}))
    invalid=copy.deepcopy(value);invalid['training_F_teacher']['seed']=2026092702
    with pytest.raises(ValueError):runtime.validate_payload(invalid)


def test_zero_and_source_only_calibration_are_separate_contracts():
    c.validate_calibration(info('step_norm_task',False),'step_norm_task',20)
    c.validate_calibration(info('activation_task',False),'activation_task',20)
    bad=info('step_norm_task',False);bad['examples']=1
    with pytest.raises(ValueError):c.validate_calibration(bad,'step_norm_task',20)
    bad=info('activation_task',False);bad['F_reads']=1
    with pytest.raises(ValueError):c.validate_calibration(bad,'activation_task',20)


def summary(accuracy=90.,role='source',knowledge=False):
    n=1800 if role=='source' else 168000
    return dict(status='ANALYZED',role=role,count=n,F_seed=d.F_SEED,original_default_head=True,all_predictions_before_truth=True,
        training_F_access=knowledge,training_TX_labels_read=0,head_fit_updates=0,
        records=[dict(view=v,readout='restored',dimension='overall',value='all',count=n,accuracy=accuracy) for v in d.VIEWS],
        views={v:dict(restored=dict(waveform=dict(iq_nmse=.001 if v=='clean' else .03))) for v in d.VIEWS})


def test_source_health_allows_small_fluctuation_but_does_not_claim_nonregression(tmp_path,monkeypatch):
    rows={arm:summary(89.,knowledge=True) for arm in d.ARMS}
    chosen=evaluation.select_source(rows,summary())
    assert chosen['selected']=='step_norm_task' and chosen['source_screening_allowance_pp']==1.
    assert chosen['checks']['step_norm_task']['qualified']
    assert not any(chosen['checks']['step_norm_task']['per_view_accuracy_nonregression'].values())
    rows['step_norm_task']['records'][-1]['accuracy']-=100/1800
    assert evaluation.select_source(rows,summary())['selected']=='activation_task'
    rows['activation_task']['records'][-1]['accuracy']-=100/1800
    assert not evaluation.select_source(rows,summary())['target_rows']
    (tmp_path/'source_decision.json').write_text(json.dumps(chosen))
    monkeypatch.setattr(evaluation,'original_predict',lambda *a,**k:pytest.fail('Unselected target must not open data'))
    with pytest.raises(ValueError):evaluation.predict(SimpleNamespace(output=tmp_path,arm='activation_task'),'target')


def test_target_one_error_regression_fails_despite_wider_source_screen(tmp_path,monkeypatch):
    output=tmp_path/'output';teacher=tmp_path/'teacher';arm=d.ARMS[0]
    folder=output/arm/'target_confirmation'/d.KIND;folder.mkdir(parents=True)
    parent_folder=teacher/'unroll_unshared/target_confirmation'/d.KIND;parent_folder.mkdir(parents=True)
    current=summary(80.,role='target',knowledge=True);current['records'][0]['accuracy']-=100/168000
    (folder/'summary.json').write_text(json.dumps(current));(parent_folder/'summary.json').write_text(json.dumps(summary(80.,role='target')))
    monkeypatch.setattr(evaluation,'score_original_head',lambda *a:None);monkeypatch.setattr(evaluation,'validate_teacher_root',lambda *a:None)
    evaluation.score(SimpleNamespace(output=output,teacher_run=teacher,arm=arm),'target')
    result=json.loads((output/arm/'compression_assessment.json').read_text())
    assert not result['target_accuracy_nonregression'] and result['per_view']['clean'] is False
    assert all(result['per_view'][v] for v in d.VIEWS[1:])


def test_truth_last_scorer_writes_actual_F_knowledge_without_fitting_head(tmp_path,monkeypatch):
    base=evaluation.base_eval;monkeypatch.setenv('CUDA_VISIBLE_DEVICES','');n=6
    row=tmp_path/'output'/d.KIND;row.mkdir(parents=True);cache=tmp_path/'cache';(cache/'V').mkdir(parents=True)
    source=tmp_path/'source';source.mkdir();np.save(cache/'V/labels.npy',np.arange(n));np.save(source/'clean.npy',np.ones((n,2,256),np.float32))
    logits=np.eye(n,dtype=np.float32)*5;features=np.eye(n,160,dtype=np.float32)
    arrays=dict(iq=np.ones((n,2,256),np.float32),logits=logits,features=features)
    refs={v:dict(identity=arrays) for v in d.VIEWS};preds={v:dict(restored=arrays) for v in d.VIEWS};seen=[]
    monkeypatch.setattr(base,'reference',lambda *a:(source,np.arange(n),SimpleNamespace(rx=np.ones(n),day=np.ones(n)),np.arange(n)))
    def predictions(path,*a):seen.append(path);return refs if 'reference_controls' in str(path) else preds
    monkeypatch.setattr(base,'prediction_arrays',predictions)
    args=SimpleNamespace(output=tmp_path/'output',kind=d.KIND,source_cache=cache)
    evaluation.score_original_head(args,'source');result=json.loads((row/'summary.json').read_text())
    assert len(seen)==2 and result['training_F_access'] is True and result['head_fit_updates']==0
    assert result['training_TX_labels_read']==0 and result['all_predictions_before_truth']
    assert all(r['accuracy']==100 for r in result['records'])


def test_predecessor_actual_R03_handles_and_new_worker_module(tmp_path):
    previous=tmp_path/'previous';previous.mkdir();(previous/'workers.json').write_text(json.dumps(dict(workers=[dict(arm='a',pid=11)])))
    proc=tmp_path/'proc';(proc/'11').mkdir(parents=True);(proc/'11/stat').write_text('11 (python) S rest')
    (proc/'11/cmdline').write_bytes(('experiments.pamf_r17_calibrated.runner worker_source '+str(previous)).encode())
    assert runner.previous_GPU_rows_live(previous,proc)==[11]
    (proc/'11/stat').write_text('11 (python) Z rest');assert runner.previous_GPU_rows_live(previous,proc)==[]
    args=SimpleNamespace(expected_commit='fixed',**{k:Path('/root')/k for k in runner.PATH_ARGS})
    assert 'experiments.pamf_r17_task.runner' in runner.command(args,'worker_source',d.ARMS[0])
