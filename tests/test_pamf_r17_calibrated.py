import copy,json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from torch import nn
from experiments.pamf_r17_calibrated import design as d,models as m,runtime,calibration as c,evaluation,runner
from experiments.pamf_action_restore.models import ActionRestorer
from experiments.pamf_reception_restore.models import ComplexTFUNet
from experiments.pamf_unrolled_restore.models import UnrolledRestorer
from experiments.pamf_uncertainty_restore import design as ancestor

torch.set_num_threads(2)


def parent():
    return dict(coarse_run=ancestor.COARSE_RUN,coarse_commit=ancestor.COARSE_COMMIT,
        action_run=ancestor.ACTION_RUN,action_commit=ancestor.ACTION_COMMIT,
        data_contract='original7/63/30',training_TX_F_reads=0,target_training=False)


def info(arm,smoke=True):
    value=c.smoke_info() if smoke else dict(c.smoke_info(),mode='source_train512',examples=512,L_s=51,U_s=461,
        received_epoch=1,selection_seed=d.CALIBRATION_SEED)
    return dict(value,observed_examples=value['examples'],eligible_layers=84 if arm=='activation_half' else 81,
        sampled_patches_min=value['examples']*8,sampled_patches_max=value['examples']*8,factor_reports=[],seconds=1.)


class ToyTeacher(nn.Module):
    def __init__(self):
        super().__init__();self.candidate='unroll_unshared'
        self.prior=nn.ModuleList([nn.Sequential(nn.Conv1d(2,4,3,padding=1),nn.SiLU(),nn.Conv1d(4,2,1)) for _ in range(3)])
    def forward(self,x):
        for core in self.prior:x=x+.1*core(x)
        return x


def test_collects_teacher_actual_layer_inputs_and_cleans_hooks_on_success_and_failure():
    teacher=ToyTeacher().eval().requires_grad_(False);before=copy.deepcopy(teacher.state_dict())
    x=torch.randn(3,2,256)
    moments,report=c.collect_moments(teacher,[x[:1],x[1:]],.5)
    assert set(moments)=={(i,'0') for i in range(3)} and report['observed_examples']==3
    assert report['sampled_patches_min']==report['sampled_patches_max']==24
    assert all(torch.isfinite(v).all() and v.shape==(6,6) for v in moments.values())
    assert all(torch.equal(v,before[k]) and v.grad is None for k,v in teacher.named_parameters())
    assert all(not module._forward_pre_hooks for module in teacher.modules())
    with pytest.raises(ValueError):c.collect_moments(teacher,[torch.zeros(1,2,255)],.5)
    assert all(not module._forward_pre_hooks for module in teacher.modules())
    with pytest.raises(ValueError):c.collect_moments(teacher.train(),[x],.5)


def test_calibration_opens_only_registered_train_received_and_roles(tmp_path,monkeypatch):
    root=tmp_path/d.ACTION_RUN;roles=np.full(63000,'U_s',dtype='<U3');roles[:6300]='L_s';opened=[]
    class Received:
        shape=(63000,2,256)
        def __getitem__(self,selected):return np.broadcast_to(np.asarray(selected,dtype=np.float32)[:,None,None]/63000,(len(selected),2,256)).copy()
    def load(path,**kwargs):
        opened.append(Path(path).relative_to(root).as_posix())
        if opened[-1]=='cache/train/roles.npy':return roles
        if opened[-1]=='cache/epochs/01/received.npy':return Received()
        pytest.fail('Calibration opened an undeclared input')
    monkeypatch.setattr(c.np,'load',load)
    monkeypatch.setattr(c,'read_json',lambda path:dict(status='ACTION_SOURCE_CACHE_COMPLETE',train_count=63000,source_V_count=1800,epochs=20,TX_labels_read=0,target_reads=0))
    batches,metadata=c.source_batches(root,'cpu');received=torch.cat(list(batches))
    selected=np.random.default_rng(d.CALIBRATION_SEED).permutation(63000)[:512]
    assert received.shape==(512,2,256) and metadata['L_s']==int((roles[selected]=='L_s').sum())
    assert metadata['L_s']+metadata['U_s']==512
    assert torch.allclose(received[:,0,0],torch.tensor(selected,dtype=torch.float32)/63000)
    assert opened==['cache/train/roles.npy','cache/epochs/01/received.npy']
    with pytest.raises(ValueError):c.source_batches(tmp_path/'target','cpu')
    assert len(opened)==2


@pytest.mark.parametrize('arm',d.ARMS)
def test_full_checkpoint_parameter_count_and_exact_inference_without_calibration_inputs(tmp_path,arm):
    model=m.CalibratedRestorer(arm,ActionRestorer(ComplexTFUNet())).eval();model.calibration_report=info(arm)
    assert sum(p.numel() for p in model.parameters())==d.PARAMETERS[arm]
    path=tmp_path/'smoke.pt';runtime.save_checkpoint(path,model,None,0,0,parent())
    fresh,value=runtime.load_checkpoint(path,'cpu')
    assert fresh.calibration_report==model.calibration_report and value['optimizer'] is None
    with torch.no_grad():
        x=torch.randn(2,2,256)*.1
        assert torch.equal(model(x),fresh(x))
    with pytest.raises(FileExistsError):runtime.save_checkpoint(path,model,None,0,0,parent())


def test_actual_full_R17_calibration_initializes_trainable_priors_and_preserves_teacher():
    torch.manual_seed(18)
    original=UnrolledRestorer('unroll_unshared',ActionRestorer(ComplexTFUNet())).eval().requires_grad_(False)
    with torch.no_grad():
        for core in original.prior:core.out.weight.normal_(std=.001)
    before=copy.deepcopy(original.state_dict());x=torch.randn(2,2,256)*.1
    moments,observed=c.collect_moments(original,[x],.5)
    student,reports=m.compact_from_teacher(original,'activation_half',moments)
    assert observed['eligible_layers']==len(reports)==84
    assert all(v['factor_kernel_parameters']<v['original_kernel_parameters'] and np.isfinite(v['regularized_output_relative_squared_error']) for v in reports)
    assert sum(p.numel() for p in student.parameters())==3775105
    with torch.no_grad():context=student.frozen.estimate(x);desired=m.stages(original.prior,x,context)[0]
    output=student.stage_outputs(x,context)[0];loss,terms=m.distillation_loss(output,desired,x*.9)
    assert torch.equal(loss,10*terms['teacher_stage_nmse']);loss.mean().backward()
    assert all(any(p.grad is not None and p.grad.abs().sum()>0 for p in core.parameters()) for core in student.prior)
    assert all(p.grad is None for p in student.frozen.parameters())
    assert all(torch.equal(value,before[key]) for key,value in original.state_dict().items())
    assert all(p.grad is None for p in original.parameters())


@pytest.mark.parametrize('field,bad',[('target_reads',1),('F_reads',1),('examples',511),('received_epoch',2),('selection_seed',1)])
def test_calibration_permissions_and_extra_source_budget_are_not_optional(field,bad):
    value=runtime.metadata(d.ARMS[0],20,4940,parent(),info(d.ARMS[0],False));runtime.validate_payload(value)
    invalid=copy.deepcopy(value);invalid['calibration'][field]=bad
    with pytest.raises(ValueError):runtime.validate_payload(invalid)
    invalid=copy.deepcopy(value);invalid['calibration']['L_s']+=1
    with pytest.raises(ValueError):runtime.validate_payload(invalid)


def test_teacher_contract_and_invalid_arm_rejected_before_checkpoint_IO(tmp_path,monkeypatch):
    monkeypatch.setattr(runtime,'load_original',lambda *a:pytest.fail('No checkpoint IO'))
    with pytest.raises(ValueError):runtime.initialize('unknown',tmp_path,'cpu',calibration_batches=[],calibration_info={})
    value=runtime.metadata(d.ARMS[0],20,4940,parent(),info(d.ARMS[0],False))
    for field,bad in [('training_F_access',True),('target_training',True),('training_TX_labels_read',1),('parent_R17_epoch',19)]:
        with pytest.raises(ValueError):runtime.validate_payload(dict(value,**{field:bad}))


def summary(accuracy=90.):
    return dict(status='ANALYZED',role='source',count=1800,F_seed=2026092701,original_default_head=True,
        all_predictions_before_truth=True,training_F_access=False,training_TX_labels_read=0,
        records=[dict(view=v,readout='restored',dimension='overall',value='all',count=1800,accuracy=accuracy) for v in d.VIEWS],
        views={v:dict(restored=dict(waveform=dict(iq_nmse=.001 if v=='clean' else .03))) for v in d.VIEWS})


def test_source_zero_regression_and_unselected_target_rejection(tmp_path,monkeypatch):
    rows={a:summary() for a in d.ARMS};choice=evaluation.select_source(rows,summary())
    assert choice['selected']=='activation_half'
    rows['activation_half']['records'][-1]['accuracy']-=.001
    assert evaluation.select_source(rows,summary())['selected']=='activation_two_thirds'
    rows['activation_two_thirds']['records'][0]['accuracy']-=.001
    assert not evaluation.select_source(rows,summary())['target_rows']
    (tmp_path/'source_decision.json').write_text(json.dumps(choice))
    monkeypatch.setattr(evaluation,'original_predict',lambda *a,**k:pytest.fail('No target IO'))
    with pytest.raises(ValueError):evaluation.predict(SimpleNamespace(output=tmp_path,arm='activation_two_thirds'),'target')


def test_predecessor_actual_R02_handles_and_new_worker_module(tmp_path):
    previous=tmp_path/'previous';previous.mkdir();(previous/'workers.json').write_text(json.dumps(dict(workers=[dict(arm='a',pid=11)])))
    proc=tmp_path/'proc';(proc/'11').mkdir(parents=True);(proc/'11/stat').write_text('11 (python) S rest')
    (proc/'11/cmdline').write_bytes(('experiments.pamf_r17_fidelity.runner worker_source '+str(previous)).encode())
    assert runner.previous_GPU_rows_live(previous,proc)==[11]
    (proc/'11/stat').write_text('11 (python) Z rest');assert runner.previous_GPU_rows_live(previous,proc)==[]
    args=SimpleNamespace(expected_commit='fixed',**{k:Path('/root')/k for k in runner.PATH_ARGS})
    command=runner.command(args,'worker_source',d.ARMS[0]);assert 'experiments.pamf_r17_calibrated.runner' in command
    assert all('--'+k.replace('_','-') in command for k in runner.PATH_ARGS)
