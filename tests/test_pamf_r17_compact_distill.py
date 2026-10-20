import copy
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from experiments.pamf_r17_compact import design as d
from experiments.pamf_r17_compact.models import FactorConv, CompactRestorer, compact_from_teacher, stages, distillation_loss
from experiments.pamf_r17_compact import runtime, evaluation
from experiments.pamf_unrolled_restore.models import UnrolledRestorer
from experiments.pamf_action_restore.operators import Action
from experiments.pamf_action_restore.models import ActionRestorer
from experiments.pamf_reception_restore.models import ComplexTFUNet
from experiments.pamf_uncertainty_restore import design as ancestor

torch.set_num_threads(2)


class Frozen(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.tensor(.9))

    def estimate(self, x):
        value = torch.complex(x[:, 0], x[:, 1])
        kernel = torch.tensor([.9, .05+.02j], device=x.device, dtype=value.dtype)[None].expand(len(x), -1)
        return self.anchor*x, Action(kernel, torch.ones_like(value), .03*torch.ones_like(value)), x.new_full((len(x),), .01)


def teacher():
    torch.manual_seed(17)
    model = UnrolledRestorer('unroll_unshared', Frozen()).eval().requires_grad_(False)
    with torch.no_grad():
        for index, core in enumerate(model.prior):
            core.out.weight.normal_(std=.001*(index+1))
            core.encoder.blocks[0][0].net[0].weight.add_(index*.01)
    return model


def test_full_rank_factor_preserves_convolution_and_input_gradient():
    torch.manual_seed(8)
    conv = nn.Conv1d(4, 7, 5, padding=4, dilation=2, stride=2).double()
    compact = FactorConv.from_conv(conv, fraction=1.)
    x = torch.randn(2, 4, 31, dtype=torch.float64, requires_grad=True)
    expected, actual = conv(x), compact(x)
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
    left = torch.autograd.grad(expected.square().sum(), x)[0]
    right = torch.autograd.grad(actual.square().sum(), x)[0]
    torch.testing.assert_close(left, right, atol=1e-11, rtol=1e-11)


def test_half_rank_is_actual_SVD_reduction_not_zeroed_dense_weights():
    conv = nn.Conv1d(32, 64, 5, padding=2)
    compact = FactorConv.from_conv(conv)
    rank = 32
    reconstructed = compact.right.weight[:, :, 0]@compact.left.weight.flatten(1)
    singular = torch.linalg.svdvals(conv.weight.flatten(1))
    error = (reconstructed-conv.weight.flatten(1)).square().sum()
    torch.testing.assert_close(error, singular[rank:].square().sum(), atol=2e-5, rtol=2e-5)
    assert sum(p.numel() for p in compact.parameters()) < sum(p.numel() for p in conv.parameters())
    assert compact.left.out_channels == rank and torch.equal(compact.right.bias, conv.bias)


@pytest.mark.parametrize('fraction', [0., -1., 1.1, float('nan')])
def test_invalid_factor_rank_rejected(fraction):
    with pytest.raises(ValueError):
        FactorConv.from_conv(nn.Conv1d(4, 8, 5), fraction=fraction)


def test_original_teacher_stage_function_is_exact_and_sharing_uses_all_three_weights():
    original = teacher()
    x = torch.randn(2, 2, 256)*.2
    with torch.no_grad():
        context = original.frozen.estimate(x)
        expected = original.stage_outputs(x)[0]
        actual = stages(original.prior, x, context)[0]
    assert all(torch.equal(a, b) for a, b in zip(expected, actual))
    student = compact_from_teacher(original, 'shared_mean')
    assert len(student.prior) == 1
    for key, value in student.prior[0].state_dict().items():
        assert torch.equal(value, torch.stack([core.state_dict()[key] for core in original.prior]).mean(0))
    assert all(torch.equal(v, student.frozen.state_dict()[k]) for k, v in original.frozen.state_dict().items())


@pytest.mark.parametrize('arm', d.ARMS)
def test_all_student_priors_receive_gradients_and_teacher_context_stays_frozen(arm):
    original = teacher()
    before = copy.deepcopy(original.state_dict())
    student = compact_from_teacher(original, arm).train()
    x = torch.randn(2, 2, 256)*.2
    with torch.no_grad():
        context = student.frozen.estimate(x)
        desired = stages(original.prior, x, context)[0]
    output = student.stage_outputs(x, context)[0]
    loss, terms = distillation_loss(output, desired, .95*x)
    loss.mean().backward()
    assert set(terms) == {'iq_nmse', 'teacher_stage_nmse'}
    assert all(any(p.grad is not None and p.grad.abs().sum()>0 for p in core.parameters()) for core in student.prior)
    assert not student.frozen.training and all(p.grad is None for p in student.frozen.parameters())
    assert all(v.grad is None and torch.equal(v, before[k]) for k, v in original.named_parameters())
    assert sum(p.numel() for p in student.parameters()) < 4665073-2157547+1  # toy frozen context has one parameter
    if arm == 'factor_half':
        assert len(student.prior) == 3 and all(any(isinstance(v, FactorConv) for v in core.modules()) for core in student.prior)


def test_distillation_detaches_teacher_even_if_caller_supplies_grad_tensors():
    value = [torch.randn(2, 2, 256, requires_grad=True) for _ in range(3)]
    target = [torch.randn_like(v, requires_grad=True) for v in value]
    loss, _ = distillation_loss(value, target, torch.randn(2, 2, 256))
    loss.mean().backward()
    assert all(v.grad is not None for v in value) and all(v.grad is None for v in target)
    with pytest.raises(ValueError):
        distillation_loss(value[:2], target, torch.randn(2, 2, 256))


def source_summary(accuracy=90., role='source'):
    return dict(status='ANALYZED', role=role, count=1800, F_seed=2026092701, original_default_head=True,
                all_predictions_before_truth=True, training_F_access=False, training_TX_labels_read=0,
                records=[dict(view=v, readout='restored', dimension='overall', value='all', count=1800, accuracy=accuracy) for v in d.VIEWS],
                views={v:dict(restored=dict(waveform=dict(iq_nmse=.001 if v=='clean' else .03))) for v in d.VIEWS})


def test_source_qualification_requires_each_view_and_source_role():
    original, shared, factor = source_summary(), source_summary(), source_summary()
    result = evaluation.select_source({'shared_mean': shared, 'factor_half': factor}, original)
    assert result['selected'] == min(d.ARMS, key=lambda a:d.PARAMETERS[a])
    shared['records'][-1]['accuracy'] -= .001
    result = evaluation.select_source({'shared_mean': shared, 'factor_half': factor}, original)
    assert result['selected'] == 'factor_half'
    factor['records'][0]['accuracy'] -= .001
    assert not evaluation.select_source({'shared_mean': shared, 'factor_half': factor}, original)['qualified']
    with pytest.raises(ValueError):
        evaluation.select_source({'shared_mean': source_summary(role='target'), 'factor_half': factor}, original)


def test_target_rejects_unselected_candidate_before_shared_predict(tmp_path, monkeypatch):
    import json
    args = SimpleNamespace(output=tmp_path, arm='factor_half')
    decision = dict(status='SOURCE_DECISION_FROZEN', qualified=True, selected='shared_mean', target_rows=['shared_mean'],
                    target_feedback=False, target_reads_before_decision=0)
    (tmp_path/'source_decision.json').write_text(json.dumps(decision))
    monkeypatch.setattr(evaluation, 'original_predict', lambda *a, **k:pytest.fail('Target IO must stay closed'))
    with pytest.raises(ValueError):
        evaluation.predict(args, 'target')


def test_checkpoint_rejects_nonfinal_R17_and_training_permission_changes():
    parent = dict(coarse_run=ancestor.COARSE_RUN, coarse_commit=ancestor.COARSE_COMMIT,
                  action_run=ancestor.ACTION_RUN, action_commit=ancestor.ACTION_COMMIT,
                  data_contract='original7/63/30', training_TX_F_reads=0, target_training=False)
    value = runtime.metadata('shared_mean', 20, 4940, parent)
    runtime.validate_payload(value)
    for key, bad in [('training_F_access', True), ('target_training', True), ('training_TX_labels_read', 1),
                     ('parent_R17_epoch', 19), ('parent_R17_run', 'other'), ('inherited_prior_epochs', 0)]:
        invalid = dict(value, **{key: bad})
        with pytest.raises(ValueError):
            runtime.validate_payload(invalid)


@pytest.mark.parametrize('arm', d.ARMS)
def test_actual_complete_parameter_count_and_strict_checkpoint_roundtrip(tmp_path, arm):
    model=CompactRestorer(arm,ActionRestorer(ComplexTFUNet())).eval()
    assert sum(p.numel() for p in model.parameters())==d.PARAMETERS[arm]
    parent=dict(coarse_run=ancestor.COARSE_RUN,coarse_commit=ancestor.COARSE_COMMIT,
                action_run=ancestor.ACTION_RUN,action_commit=ancestor.ACTION_COMMIT,
                data_contract='original7/63/30',training_TX_F_reads=0,target_training=False)
    path=tmp_path/'final.pt'
    runtime.save_checkpoint(path,model,None,20,4940,parent)
    fresh,value=runtime.load_checkpoint(path,'cpu')
    assert value['inherited_prior_epochs']==20 and value['epoch']==20
    assert all(torch.equal(v,fresh.state_dict()[k]) for k,v in model.state_dict().items())
    with pytest.raises(FileExistsError):runtime.save_checkpoint(path,model,None,20,4940,parent)


def test_GPU_predecessor_uses_actual_owned_handles_and_preserves_unrelated_processes(tmp_path):
    import json
    from experiments.pamf_r17_compact.runner import previous_GPU_rows_live
    previous=tmp_path/'previous';previous.mkdir();(previous/'arm').mkdir()
    (previous/'workers.json').write_text(json.dumps(dict(workers=[dict(arm='arm',pid=11)])))
    (previous/'arm/target_launch.json').write_text(json.dumps(dict(pid=12)))
    proc=tmp_path/'proc';proc.mkdir()
    for pid,module,state in [(11,'experiments.pamf_scratch_guidance.runner','S'),(12,'other_owner','S')]:
        folder=proc/str(pid);folder.mkdir();(folder/'stat').write_text(f'{pid} (python) {state} rest')
        (folder/'cmdline').write_bytes((module+' '+str(previous)).encode())
    assert previous_GPU_rows_live(previous,proc)==[11]
    (proc/'11/stat').write_text('11 (python) Z rest')
    assert previous_GPU_rows_live(previous,proc)==[]


def test_wrong_action_cache_is_rejected_before_opening_data(tmp_path):
    from experiments.pamf_r17_compact.training import train
    with pytest.raises(ValueError):train(SimpleNamespace(arm='shared_mean',parent_action=tmp_path/'wrong'))


@pytest.mark.parametrize('arm', d.ARMS)
def test_actual_new_schema_is_used_for_all_seven_source_predictions(tmp_path,monkeypatch,arm):
    import numpy as np
    import experiments.pamf_uncertainty_restore.evaluation as shared
    source=tmp_path/'source';source.mkdir()
    for view in d.VIEWS:np.save(source/(view+'.npy'),np.ones((16,2,256),np.float32))
    output=tmp_path/'output';(output/arm/d.KIND).mkdir(parents=True)
    args=SimpleNamespace(arm=arm,output=output,reader_folder=tmp_path/'unused',source_cache=tmp_path/'unused')
    class Toy(nn.Module):
        def training_outputs(self,x):return x,torch.ones((len(x),256)),None,x
        def forward(self,x):return x
    class Reader(nn.Module):
        def features(self,x):return x.flatten(1)[:,:160]
        def classify_features(self,x):return x[:,:6]
    monkeypatch.setattr(evaluation,'load_checkpoint',lambda *a:(Toy(),dict(arm=arm,kind=d.KIND,epoch=20,updates=4940)))
    monkeypatch.setattr(shared,'load_checkpoint',lambda *a:pytest.fail('Old schema loader forbidden'))
    monkeypatch.setattr(shared,'reference',lambda *a:(source,np.arange(16),None,None))
    monkeypatch.setattr(shared,'load_reader',lambda *a:Reader())
    monkeypatch.setattr(shared,'array_tensor',lambda x,device:torch.tensor(x.tolist(),dtype=torch.float32))
    monkeypatch.setattr(shared,'tensor_array',lambda x:np.asarray(x.tolist(),np.float32))
    for name in ('synchronize','reset_peak_memory_stats'):monkeypatch.setattr(torch.cuda,name,lambda:None)
    monkeypatch.setattr(torch.cuda,'max_memory_allocated',lambda:0)
    evaluation.predict(args,'source')
    assert tuple(shared.prediction_arrays(output/arm/d.KIND/'prediction',np.arange(16),d.KIND,'source'))==d.VIEWS


def test_controller_updates_status_repeatedly_then_dispatches_only_selected_target(tmp_path,monkeypatch):
    import json
    from experiments.pamf_r17_compact import runner
    teacher_run=tmp_path/'teacher'
    control=teacher_run/'unroll_unshared'/d.KIND;control.mkdir(parents=True)
    (control/'summary.json').write_text(json.dumps(source_summary()))
    output=tmp_path/d.RUN
    args=SimpleNamespace(output=output,teacher_run=teacher_run,expected_commit='test')
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','')
    for name in ('validate_teacher_root','smoke','wait_GPU'):monkeypatch.setattr(runner,name,lambda *a:None)
    monkeypatch.setattr(runner.time,'sleep',lambda *a:None)
    launched=[]
    class Process:
        def __init__(self,pid):self.pid=pid;self.calls=0;self.returncode=0
        def poll(self):self.calls+=1;return None if self.calls==1 else 0
        def wait(self):return 0
    def launch(args,arm,role):
        launched.append((arm,role));row=output/arm
        if role=='source':
            (row/d.KIND).mkdir(parents=True);(row/d.KIND/'summary.json').write_text(json.dumps(source_summary()))
        (row/('complete.json' if role=='source' else 'target_complete.json')).write_text('{}')
        return Process(100+len(launched)),dict(arm=arm,role=role,pid=100+len(launched))
    monkeypatch.setattr(runner,'launch',launch)
    runner.run(args)
    assert launched==[('shared_mean','source'),('factor_half','source'),('shared_mean','target')]
    assert len(json.loads((output/'workers.json').read_text())['workers'])==3
    assert json.loads((output/'complete.json').read_text())['target_rows']==['shared_mean']
    assert not (output/'failure.json').exists()
