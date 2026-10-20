import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from experiments.pamf_r44_pruned_refine.models import CompactSuccessor as ParentModel
from experiments.pamf_r46_source_prune import design as d,models as m,runtime as r,evaluation as e,runner
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


def calibration():
    from experiments.pamf_r46_source_prune.calibration import make_maps
    widths={'coarse_spectral':(8,12,20,32),**{n:(12,20,32,44) for n in ('coarse_temporal','action','prior0','prior1','prior2')}}
    scores={method:{name:[torch.arange(w,dtype=torch.float64)+1 if method=='activation_energy' else torch.arange(w,0,-1,dtype=torch.float64) for w in levels] for name,levels in widths.items()} for method in d.ARMS}
    return dict(schema='R46_source_channel_scores_v1',parent_run=d.COMPACT_PARENT_RUN,parent_commit=d.COMPACT_PARENT_COMMIT,
        source_unique_IQ=128,frozen_parent_IQ_inputs=256,L_s=64,U_s=64,received_views=['01','20'],sampling_seed=d.CAL_SEED,projection_seed=d.CAL_PROJECTION_SEED,
        projection_draws=2,TX_labels_read=0,F_reads=0,V_reads=0,target_reads=0,optimizer_updates=0,parent_unchanged=True,maps=make_maps(scores))


@pytest.mark.parametrize('arm',d.ARMS)
def test_existing_reduced_models_joint_gradients_detached_teacher_and_standalone_E20(tmp_path,arm):
    torch.manual_seed(43);parent=parent_model();before=copy.deepcopy(parent.state_dict())
    model=m.compact_from_parent(parent,arm,calibration());model.initialization_report['seconds']=0.
    r.validate_aliases(model)
    assert sum(p.numel() for p in model.parameters())==d.PARAMETERS[arm]<=350521
    assert sum(p.numel() for p in model.frozen.parameters())==d.CONTEXT_PARAMETERS[arm]
    clean=(torch.randn(2,2,256)*.1).requires_grad_(True);received=clean.detach()+torch.randn_like(clean)*.005
    with torch.no_grad():desired=parent.stage_outputs(received)[0]
    desired=[v.detach().requires_grad_(True) for v in desired]
    outputs=model.stage_outputs(received)[0]
    loss,terms=m.objective(outputs,desired,clean,1,arm)
    assert torch.allclose(loss,terms['iq_nmse']+.05*terms['spectral_error']+terms['teacher_stage_nmse'])
    loss.mean().backward()
    for group in (model.frozen.coarse,model.frozen.estimator,*model.prior):
        grads=[p.grad for p in group.parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads) and any(g.abs().sum()>0 for g in grads)
    assert clean.grad is None and all(v.grad is None for v in desired)
    assert all(p.grad is None for p in parent.parameters()) and all(torch.equal(v,before[k]) for k,v in parent.state_dict().items())
    optimizer=torch.optim.AdamW(model.parameters(),lr=2e-4);optimizer.step()
    model.eval();path=tmp_path/'final.pt';r.save_checkpoint(path,model,None,20,4940,lineage())
    fresh,value=r.load_checkpoint(path,'cpu')
    assert value['optimizer'] is None and value['cumulative_component_epochs']==100 and not value['training_F_access']
    with torch.no_grad():assert torch.equal(model(received),fresh(received))
    with pytest.raises(FileExistsError):r.save_checkpoint(path,model,None,20,4940,lineage())


@pytest.mark.parametrize('field,bad',[('parent_R46_row','retain_refine'),('parent_R46_commit','unknown'),
    ('training_F_access',True),('F_outputs_read',1),('target_training',True),('V_training',True),('updates',4941),('epoch',21)])
def test_checkpoint_lineage_permissions_and_budget_reject(field,bad):
    model=m.compact_from_parent(parent_model(),d.ARMS[0],calibration());model.initialization_report['seconds']=0.
    value=r.metadata(d.ARMS[0],20,4940,lineage(),model.initialization_report);r.validate_payload(value);value[field]=bad
    with pytest.raises(ValueError):r.validate_payload(value)


def test_alias_conflict_and_invalid_arm_reject_before_parent_IO(monkeypatch):
    model=m.CompactSuccessor(d.ARMS[1]);state=copy.deepcopy(model.state_dict())
    key=next(k for k in state if k.startswith('prior.1.') and k.endswith('net.0.weight'));state[key]=state[key]+1
    with pytest.raises(ValueError,match='Conflicting'):r.validate_aliases(model,state)
    monkeypatch.setattr(r,'validate_parent_root',lambda *a:pytest.fail('No parent IO'))
    with pytest.raises(ValueError):r.initialize('keep_iq','/runs','cpu','/calibration.json')


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


def test_finite_source_failure_preserves_and_does_not_retry_or_touch_target(tmp_path,monkeypatch):
    args=SimpleNamespace(output=tmp_path,arm=d.ARMS[0])
    def fail(*a):raise FloatingPointError('nonfinite')
    monkeypatch.setattr(runner,'train',fail);monkeypatch.setattr(runner,'predict',lambda *a:pytest.fail('No prediction'))
    with pytest.raises(FloatingPointError):runner.worker(args)
    assert (tmp_path/args.arm/'failure.json').exists()


def test_fixed_two_source_selection_methods_and_original_objective():
    cfg=d.configuration();assert json.loads(json.dumps(cfg))==cfg
    assert list(d.PARAMETERS.values())==[273765,273765]
    assert cfg['calibration']['frozen_parent_IQ_inputs']==256 and cfg['calibration']['unique_source_IQ']==128
    assert d.EPOCHS==20 and d.UPDATES==4940 and cfg['cumulative_component_epochs']==100
    assert all(d.teacher_weight(20,a)==pytest.approx(.1) for a in d.ARMS)
    assert d.REUSED_CONTROL['row']=='prune_decay' and d.REUSED_CONTROL['run']==d.PREDECESSOR_RUN


def test_all_width_maps_are_identity_and_channel_quota_rejects_bad_selection():
    from experiments.pamf_reception_restore.models import Core
    source=Core(incoming=4,outgoing=4,widths=(8,12,16,20));target=copy.deepcopy(source)
    with torch.no_grad():source.out.weight.normal_(std=.01)
    maps=[torch.arange(w) for w in (8,12,16,20)]
    m.prune_with_maps(source,target,maps);x=torch.randn(2,4,256)
    with torch.no_grad():assert torch.equal(source(x),target(x))
    bad=copy.deepcopy(maps);bad[-1][0]=1
    with pytest.raises(ValueError):m.prune_with_maps(source,target,bad)
    assert m.selected_channels(torch.ones(16),8).tolist()==[0,1,4,5,8,9,12,13]
    with pytest.raises(ValueError):m.selected_channels(torch.ones(16),7)


def test_real_complete_parent_source_scores_are_deterministic_nonnegative_and_do_not_fit_parent():
    from experiments.pamf_r46_source_prune.calibration import collect_scores,make_maps
    torch.manual_seed(54);parent=parent_model();before=copy.deepcopy(parent.state_dict())
    clean=(torch.randn(3,2,256)*.1).requires_grad_(True);received=clean.detach()+torch.randn_like(clean)*.005
    scores,count=collect_scores(parent,clean,[received,received*.98],batch=2,draws=2,seed=54)
    repeated,count2=collect_scores(parent,clean,[received,received*.98],batch=2,draws=2,seed=54)
    assert count==count2==6 and clean.grad is None
    for method,groups in scores.items():
        for name,levels in groups.items():
            assert len(levels)==4
            for i,value in enumerate(levels):
                assert torch.equal(value,repeated[method][name][i]) and torch.isfinite(value).all() and (value>=0).all()
    maps=make_maps(scores)
    for method,groups in maps.items():assert groups['prior0']==groups['prior1']==groups['prior2']
    assert all(p.grad is None for p in parent.parameters()) and all(torch.equal(v,before[k]) for k,v in parent.state_dict().items())


def test_random_projection_channel_gain_derivative_matches_small_exact_linear_jacobian():
    activation=torch.tensor([[[1.,2.],[3.,4.]]],requires_grad=True)
    matrix=torch.tensor([[1.,2.,-1.,.5],[2.,-1.,.25,3.]])
    output=activation.flatten(1)@matrix.T
    actual=[]
    for signs in (torch.tensor([[1.,1.]]),torch.tensor([[1.,-1.]]),torch.tensor([[-1.,1.]]),torch.tensor([[-1.,-1.]])):
        gradient=torch.autograd.grad((output*signs).sum(),activation,retain_graph=True)[0]
        actual.append((activation*gradient).sum(-1).square())
    desired=(matrix.reshape(2,2,2)*activation.detach()[0]).sum(-1).square().sum(0)
    assert torch.allclose(torch.stack(actual).mean(0)[0],desired)


@pytest.mark.parametrize('field,bad',[('TX_labels_read',1),('F_reads',1),('V_reads',1),('target_reads',1),('source_unique_IQ',127),('parent_commit','unknown')])
def test_calibration_protocol_negative(field,bad):
    from experiments.pamf_r46_source_prune.calibration import validate_calibration
    value=calibration();value[field]=bad
    with pytest.raises(ValueError):validate_calibration(value)


def test_shared_prior_channel_maps_cannot_differ():
    value=calibration();value['maps'][d.ARMS[0]]['prior1'][1][0]=19
    with pytest.raises(ValueError):m.compact_from_parent(parent_model(),d.ARMS[0],value)


def test_spectral_term_detects_amplitude_distortion_keeps_clean_gradient_closed_and_zero_IQ_finite():
    torch.manual_seed(46);clean=(torch.randn(2,2,256)*.1).requires_grad_(True)
    assert torch.equal(m.spectral_error(clean.detach(),clean),torch.zeros(2))
    restored=(clean.detach()*1.5).requires_grad_(True);loss=m.spectral_error(restored,clean)
    assert (loss>0).all();loss.mean().backward()
    assert clean.grad is None and torch.isfinite(restored.grad).all() and restored.grad.abs().sum()>0
    zeros=torch.zeros(2,2,256,requires_grad=True);zero=m.spectral_error(zeros,zeros.detach())
    zero.sum().backward();assert torch.isfinite(zero).all() and torch.isfinite(zeros.grad).all()


@pytest.mark.parametrize('arm',d.ARMS)
def test_pruned_shared_weights_keep_step_specific_norm_and_bias(arm):
    model=m.compact_from_parent(parent_model(),arm,calibration())
    a=model.prior[0].encoder.blocks[0][0].net;b=model.prior[1].encoder.blocks[0][0].net
    assert a[0].weight is b[0].weight and a[3].weight is b[3].weight
    assert a[0].bias is not b[0].bias and a[1].weight is not b[1].weight


def test_buffer_exact_noncontiguous_readonly_and_invalid_dtype():
    import numpy as np
    from experiments.pamf_r46_source_prune.tensor_io import array_tensor
    a=np.arange(4*2*256,dtype=np.float32).reshape(4,2,256);before=a.copy()
    for value in (a,a[::2],a[:,:,::-1]):
        got=array_tensor(value,'cpu');assert got.tolist()==value.tolist()
    a.setflags(write=False);got=array_tensor(a,'cpu');got.add_(1)
    assert np.array_equal(a,before)
    with pytest.raises(ValueError):array_tensor(a.astype(np.float64),'cpu')


def test_predecessor_final_marker_and_actual_handle_are_both_required(tmp_path):
    from experiments.pamf_r46_source_prune.queueing import predecessor_closed
    gpu=3;arm='activation_energy';root=tmp_path/d.PREDECESSOR_RUN;row=root/'retain_anchor';row.mkdir(parents=True)
    args=SimpleNamespace(output=tmp_path/d.RUN,arm=arm);proc=tmp_path/'proc';proc.mkdir()
    assert not predecessor_closed(args,proc_root=proc)
    (row/'complete.json').write_text(json.dumps(dict(status='ANALYZED')))
    (root/'launch.json').write_text(json.dumps(dict(commit=d.PREDECESSOR_COMMIT)))
    (root/'workers.json').write_text(json.dumps(dict(workers=[dict(GPU=3,pid=123)])))
    handle=proc/'123';handle.mkdir();(handle/'stat').write_text('123 (python) S 1')
    (handle/'cmdline').write_bytes((d.PREDECESSOR_MODULE+'\0'+str(root)+'\0').encode())
    assert not predecessor_closed(args,proc_root=proc)
    (handle/'stat').write_text('123 (python) Z 1');assert predecessor_closed(args,proc_root=proc)
    args.arm='projection_sensitivity'
    assert not predecessor_closed(args,proc_root=proc),'GPU2 must wait queued prune_anchor,not merely first prune_decay'
