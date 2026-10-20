import copy,json
from types import SimpleNamespace
from pathlib import Path
import numpy as np
import pytest
import torch
from torch import nn

from experiments.pamf_r17_fidelity import design as d,models as m,runtime,evaluation,runner
from experiments.pamf_action_restore.models import ActionRestorer
from experiments.pamf_reception_restore.models import ComplexTFUNet
from experiments.pamf_unrolled_restore.models import UnrolledRestorer
from experiments.pamf_uncertainty_restore import design as ancestor

torch.set_num_threads(2)


def teacher():
    torch.manual_seed(17)
    value=UnrolledRestorer('unroll_unshared',ActionRestorer(ComplexTFUNet())).eval().requires_grad_(False)
    with torch.no_grad():
        for index,core in enumerate(value.prior):
            core.out.weight.normal_(std=.001*(index+1))
            core.encoder.blocks[0][0].net[0].bias.add_(index*.01)
            core.encoder.blocks[0][0].net[1].weight.add_(index*.02)
    return value


def parent():
    return dict(coarse_run=ancestor.COARSE_RUN,coarse_commit=ancestor.COARSE_COMMIT,
        action_run=ancestor.ACTION_RUN,action_commit=ancestor.ACTION_COMMIT,
        data_contract='original7/63/30',training_TX_F_reads=0,target_training=False)


def summary(accuracy=90.):
    return dict(status='ANALYZED',role='source',count=1800,F_seed=2026092701,original_default_head=True,
        all_predictions_before_truth=True,training_F_access=False,training_TX_labels_read=0,
        records=[dict(view=v,readout='restored',dimension='overall',value='all',count=1800,accuracy=accuracy) for v in d.VIEWS],
        views={v:dict(restored=dict(waveform=dict(iq_nmse=.001 if v=='clean' else .03))) for v in d.VIEWS})


def test_step_specific_norm_bias_and_short_kernels_survive_shared_long_kernels():
    original=teacher();student=m.compact_from_teacher(original,'step_norm_fidelity')
    maps=[dict(core.named_modules()) for core in student.prior];references=[dict(core.named_modules()) for core in original.prior]
    for name,layer in maps[0].items():
        if isinstance(layer,nn.Conv1d):
            if layer.kernel_size[0]>1:
                assert all(layer.weight is values[name].weight for values in maps)
                assert torch.equal(layer.weight,torch.stack([v[name].weight for v in references]).mean(0))
            else:
                assert all(maps[i][name].weight is not maps[j][name].weight for i,j in ((0,1),(1,2)))
                assert all(torch.equal(values[name].weight,references[i][name].weight) for i,values in enumerate(maps))
            assert all(torch.equal(values[name].bias,references[i][name].bias) for i,values in enumerate(maps))
        if isinstance(layer,nn.GroupNorm):
            assert all(maps[i][name].weight is not maps[j][name].weight for i,j in ((0,1),(1,2)))
            assert all(torch.equal(values[name].weight,references[i][name].weight) for i,values in enumerate(maps))
    assert sum(p.numel() for p in student.parameters())==3118001


def test_matching_teacher_has_zero_loss_even_when_paired_clean_differs():
    value=[torch.randn(2,2,256,requires_grad=True) for _ in range(3)]
    desired=[v.detach().clone().requires_grad_() for v in value]
    clean=torch.full((2,2,256),3.,requires_grad=True)
    loss,terms=m.distillation_loss(value,desired,clean)
    assert loss.eq(0).all() and terms['iq_nmse'].gt(0).all()
    loss.mean().backward()
    assert all(v.grad.eq(0).all() for v in value) and all(v.grad is None for v in desired)
    assert clean.grad is None


@pytest.mark.parametrize('arm',d.ARMS)
def test_final_loss_reaches_student_priors_without_teacher_or_context_updates(arm):
    original=teacher();before=copy.deepcopy(original.state_dict());student=m.compact_from_teacher(original,arm).train()
    x=torch.randn(2,2,256)*.2
    with torch.no_grad():context=student.frozen.estimate(x);desired=m.stages(original.prior,x,context)[0]
    result=student.stage_outputs(x,context)[0];loss,terms=m.distillation_loss(result,desired,.9*x)
    assert torch.equal(loss,10*terms['teacher_stage_nmse'])
    loss.mean().backward()
    assert all(any(p.grad is not None and p.grad.abs().sum()>0 for p in core.parameters()) for core in student.prior)
    assert all(p.grad is None for p in student.frozen.parameters())
    assert all(torch.equal(v,before[k]) and v.grad is None for k,v in original.named_parameters())


@pytest.mark.parametrize('arm',d.ARMS)
def test_new_checkpoint_preserves_real_parameter_count_output_and_weight_aliases(tmp_path,arm):
    model=m.FidelityRestorer(arm,ActionRestorer(ComplexTFUNet())).eval()
    assert sum(p.numel() for p in model.parameters())==d.PARAMETERS[arm]
    path=tmp_path/'final.pt';runtime.save_checkpoint(path,model,None,20,4940,parent())
    fresh,value=runtime.load_checkpoint(path,'cpu');assert value['arm']==arm
    with torch.no_grad():
        x=torch.randn(2,2,256)*.2
        assert torch.equal(model(x),fresh(x))
    if arm=='step_norm_fidelity':
        maps=[dict(core.named_modules()) for core in fresh.prior]
        for name,layer in maps[0].items():
            if isinstance(layer,nn.Conv1d) and layer.kernel_size[0]>1:
                assert all(layer.weight is values[name].weight for values in maps)
    with pytest.raises(FileExistsError):runtime.save_checkpoint(path,model,None,20,4940,parent())


def test_training_contract_rejects_wrong_teacher_loss_or_permission():
    value=runtime.metadata(d.ARMS[0],20,4940,parent());runtime.validate_payload(value)
    for field,bad in [('training_F_access',True),('target_training',True),('training_TX_labels_read',1),('parent_R17_epoch',19)]:
        invalid=dict(value,**{field:bad})
        with pytest.raises(ValueError):runtime.validate_payload(invalid)
    invalid=copy.deepcopy(value);invalid['config']['loss']='pairedcleanIQNMSE'
    with pytest.raises(ValueError):runtime.validate_payload(invalid)


def test_source_zero_regression_selection_and_target_reject_unselected_before_IO(tmp_path,monkeypatch):
    original=summary();rows={a:summary() for a in d.ARMS}
    choice=evaluation.select_source(rows,original);assert choice['selected']=='step_norm_fidelity'
    rows['step_norm_fidelity']['records'][-1]['accuracy']-=.001
    assert evaluation.select_source(rows,original)['selected']=='factor_half_fidelity'
    (tmp_path/'source_decision.json').write_text(json.dumps(choice))
    monkeypatch.setattr(evaluation,'original_predict',lambda *a,**k:pytest.fail('No target IO'))
    with pytest.raises(ValueError):evaluation.predict(SimpleNamespace(output=tmp_path,arm='factor_half_fidelity'),'target')


def test_predecessor_checks_previous_R17_actual_owned_handles(tmp_path):
    previous=tmp_path/'previous';previous.mkdir();(previous/'workers.json').write_text(json.dumps(dict(workers=[dict(arm='a',pid=11)])))
    proc=tmp_path/'proc';(proc/'11').mkdir(parents=True);(proc/'11/stat').write_text('11 (python) S rest')
    (proc/'11/cmdline').write_bytes(('experiments.pamf_r17_compact.runner worker_source '+str(previous)).encode())
    assert runner.previous_GPU_rows_live(previous,proc)==[11]
    (proc/'11/stat').write_text('11 (python) Z rest')
    assert runner.previous_GPU_rows_live(previous,proc)==[]


def test_worker_commands_pin_new_module_and_all_data_paths():
    args=SimpleNamespace(expected_commit='fixed',**{k:Path('/root')/k for k in runner.PATH_ARGS})
    command=runner.command(args,'worker_source',d.ARMS[0])
    assert 'experiments.pamf_r17_fidelity.runner' in command and 'experiments.pamf_r17_compact.runner' not in command
    assert all('--'+k.replace('_','-') in command for k in runner.PATH_ARGS)
