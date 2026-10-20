import copy
import importlib
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch

torch.set_num_threads(2)

def api():
    try:
        return importlib.import_module('experiments.pamf_r46_response_distill.probes')
    except ModuleNotFoundError:
        pytest.fail('The source finite-response training objective is not implemented')


class Affine(torch.nn.Module):
    def __init__(self, scale, bias=0.):
        super().__init__()
        self.scale=torch.nn.Parameter(torch.tensor(float(scale)))
        self.bias=torch.nn.Parameter(torch.tensor(float(bias)))
        self.calls=[]

    def forward(self,x):
        self.calls.append(x.detach().clone())
        return self.scale*x+self.bias


def test_response_loss_detects_wrong_slope_and_preserves_teacher_grad_boundary():
    p=api();x=torch.randn(3,2,256)*.1;delta=x*.01
    student=Affine(1.);teacher=Affine(2.)
    value=p.pair_loss(student(x+delta),student(x-delta),teacher(x+delta),teacher(x-delta),delta,'odd_response')
    torch.testing.assert_close(value,torch.full((3,),.25),rtol=1e-5,atol=1e-6)
    value.mean().backward()
    assert student.scale.grad.item()==pytest.approx(-.5,rel=1e-4)
    assert teacher.scale.grad is None and teacher.bias.grad is None


def test_output_control_and_odd_response_measure_different_errors():
    p=api();x=torch.randn(3,2,256)*.1;delta=x*.01
    a,b=2*(x+delta)+3,2*(x-delta)+3
    c,e=2*(x+delta)+2,2*(x-delta)+2
    response=p.pair_loss(a,b,c,e,delta,'odd_response')
    control=p.pair_loss(a,b,c,e,delta,'pair_aug_control')
    assert response.max()<1e-8 and control.min()>0


@pytest.mark.parametrize('arm',['pair_aug_control','odd_response'])
def test_probe_arms_share_inputs_and_equal_actual_forward_exposure(arm):
    p=api();torch.manual_seed(83);x=(torch.randn(24,2,256)*.1).requires_grad_()
    s=Affine(.9);t=Affine(1.);before=copy.deepcopy(t.state_dict())
    loss,info=p.paired_probe(s,t,x,2,arm)
    assert loss.shape==(16,) and info['count']==16
    assert info['probe']=='random' and info['epsilon']==.01
    assert len(s.calls)==len(t.calls)==2
    assert all(z.shape==(16,2,256) for z in s.calls+t.calls)
    assert all(torch.equal(a,b) for a,b in zip(s.calls,t.calls))
    other,_=p.paired_probe(Affine(.9),Affine(1.),x,2,'odd_response' if arm=='pair_aug_control' else 'pair_aug_control')
    loss.mean().backward()
    assert torch.isfinite(s.scale.grad) and s.scale.grad.abs()>0
    assert x.grad is None and all(v.grad is None for v in t.parameters())
    assert all(torch.equal(v,before[k]) for k,v in t.state_dict().items())


@pytest.mark.parametrize('update,expected',[(0,('random',.005)),(1,('phase',.005)),(2,('random',.01)),(3,('phase',.01)),(19700,('random',.005))])
def test_predeclared_probe_cycle_and_equal_total_cells(update,expected):
    p=api();assert p.probe_spec(update)==expected
    assert {spec:sum(p.probe_spec(i)==spec for i in range(19700)) for spec in (('random',.005),('phase',.005),('random',.01),('phase',.01))}=={
        ('random',.005):4925,('phase',.005):4925,('random',.01):4925,('phase',.01):4925}


@pytest.mark.parametrize('update',[0,1,2,3])
def test_deterministic_equal_RMS_perturbation_without_global_rng_mutation(update):
    p=api();x=torch.randn(16,2,256)*.1;state=torch.get_rng_state().clone()
    delta,info=p.make_delta(x,update)
    assert torch.equal(torch.get_rng_state(),state)
    assert torch.equal(delta,p.make_delta(x,update)[0])
    torch.testing.assert_close(delta.square().sum(1).mean(1),x.square().sum(1).mean(1)*info['epsilon']**2)
    if info['probe']=='phase':torch.testing.assert_close(delta,torch.stack((-x[:,1],x[:,0]),1)*info['epsilon'])


@pytest.mark.parametrize('arm',['pair_aug_control','odd_response'])
def test_per_example_loss_independent_of_other_batch_examples_and_teacher_detached(arm):
    p=api();x=torch.randn(3,2,256)*.1;delta=x*.005
    sp=(x+delta).requires_grad_();sm=(x-delta).requires_grad_()
    tp=(1.2*(x+delta)).requires_grad_();tm=(1.2*(x-delta)).requires_grad_()
    full=p.pair_loss(sp,sm,tp,tm,delta,arm)
    single=p.pair_loss(sp[:1],sm[:1],tp[:1],tm[:1],delta[:1],arm)
    torch.testing.assert_close(full[:1],single)
    full.sum().backward();assert tp.grad is None and tm.grad is None
    assert sp.grad is not None and sm.grad is not None


@pytest.mark.parametrize('arm',['pair_aug_control','odd_response'])
def test_zero_energy_is_finite_and_plus_minus_joint_swap_is_invariant(arm):
    p=api();z=torch.zeros(2,2,256,requires_grad=True)
    value=p.pair_loss(z,z,z,z,z,arm);value.sum().backward()
    assert torch.equal(value,torch.zeros(2)) and torch.isfinite(z.grad).all()
    x=torch.randn(2,2,256)*.1;delta=x*.01
    torch.testing.assert_close(p.pair_loss(x+delta,x-delta,2*(x+delta),2*(x-delta),delta,arm),
        p.pair_loss(x-delta,x+delta,2*(x-delta),2*(x+delta),-delta,arm))


@pytest.mark.parametrize('bad',[torch.zeros(2,3,256),torch.zeros(0,2,256),torch.ones(2,2,256,dtype=torch.int64),torch.full((2,2,256),float('nan'))])
def test_invalid_native_IQ_rejected_before_model_forward(bad):
    p=api();s=Affine(1.);t=Affine(1.)
    with pytest.raises(ValueError):p.paired_probe(s,t,bad,0,'odd_response')
    assert not s.calls and not t.calls


@pytest.mark.parametrize('bad',[-1,True,1.5])
def test_invalid_update_or_unknown_arm_rejected_before_forward(bad):
    p=api();s=Affine(1.);t=Affine(1.);x=torch.ones(2,2,256)
    with pytest.raises(ValueError):p.paired_probe(s,t,x,bad,'odd_response')
    with pytest.raises(ValueError):p.paired_probe(s,t,x,0,'other')
    assert not s.calls and not t.calls


def test_final_configuration_has_equal_probe_exposure_and_fixed_f_free_budget():
    api();d=importlib.import_module('experiments.pamf_r46_response_distill.design');cfg=d.configuration()
    assert cfg==json.loads(json.dumps(cfg))
    assert d.EPOCHS==20 and d.UPDATES==19700 and d.BATCH==d.MICRO==64
    assert cfg['extra_probe_inputs_per_model']==630400
    assert cfg['selected_probe_examples_per_model']==315200
    assert cfg['teacher_IQ_inputs']==1890400 and cfg['student_total_IQ_inputs']==3150400
    assert cfg['F_weights_read']==cfg['F_outputs_read']==cfg['TX_labels_read']==0
    assert all(d.PARAMETERS[a]==278180 and d.CLEAN_WEIGHT[a]==.25 for a in d.ARMS)
    assert d.PREDECESSOR.endswith('-r23') and len(d.VIEWS)==7


@pytest.mark.parametrize('arm',['pair_aug_control','odd_response'])
def test_full_compact_model_gradient_and_strict_checkpoint_roundtrip(tmp_path,arm):
    api();package='experiments.pamf_r46_response_distill'
    m=importlib.import_module(package+'.models');r=importlib.import_module(package+'.runtime')
    base=importlib.import_module('tests.test_pamf_r46_complex_fidelity')
    parent=base.parent_model();before=copy.deepcopy(parent.state_dict());model=m.compact_from_parent(parent,arm)
    model.initialization_report['seconds']=0.;x=torch.randn(2,2,256)*.1;received=x+.005*torch.randn_like(x)
    outputs,desired,identity=m.paired_outputs(model,parent,received,x)
    loss,terms=m.objective(outputs,desired,x,1,arm,identity)
    probe,info=api().paired_probe(model,parent,received,1,arm)
    (loss.mean()+.01*probe.mean()).backward()
    for group in (model.frozen.coarse,model.frozen.estimator,*model.prior):
        grads=[p.grad for p in group.parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads) and any(g.abs().sum()>0 for g in grads)
    assert all(p.grad is None for p in parent.parameters())
    assert all(torch.equal(v,before[k]) for k,v in parent.state_dict().items())
    path=tmp_path/'final.pt';model.eval();r.save_checkpoint(path,model,None,20,19700,base.lineage())
    fresh,value=r.load_checkpoint(path,'cpu')
    with torch.no_grad():assert torch.equal(model(received),fresh(received))
    assert sum(p.numel() for p in fresh.parameters())==278180
    value['config']['probe_weight']=.02
    with pytest.raises(ValueError):r.validate_payload(value)


@pytest.mark.parametrize('arm',['pair_aug_control','odd_response'])
def test_full_formal_chain_executes_for_both_rows_without_source_gain_gate(tmp_path,monkeypatch,arm):
    api();r=importlib.import_module('experiments.pamf_r46_response_distill.runner')
    root=tmp_path/r.d.RUN;root.mkdir();(root/'launch.json').write_text(json.dumps(dict(commit='method')))
    (root/'comparison_plan.json').write_text(json.dumps(r.d.configuration()))
    args=SimpleNamespace(output=root,arm=arm,expected_commit='method');seen=[]
    def train(a):(a.output/a.arm).mkdir();seen.append('train')
    monkeypatch.setattr(r,'train',train);monkeypatch.setattr(r,'predict',lambda a,role:seen.append('predict_'+role))
    monkeypatch.setattr(r,'profile',lambda *a:seen.append('profile'));monkeypatch.setattr(r,'score_subprocess',lambda a,role:seen.append('score_'+role))
    monkeypatch.setattr(r.torch.cuda,'empty_cache',lambda:None);r.worker(args)
    assert seen==['train','predict_source','profile','score_source','predict_target','score_target']
    assert (root/arm/'complete.json').is_file()


def test_permission_contract_rejects_F_target_and_wrong_parent_commit():
    api();r=importlib.import_module('experiments.pamf_r46_response_distill.runtime')
    m=importlib.import_module('experiments.pamf_r46_response_distill.models')
    base=importlib.import_module('tests.test_pamf_r46_complex_fidelity')
    model=m.compact_from_parent(base.parent_model(),'odd_response');model.initialization_report['seconds']=0.
    original=r.metadata('odd_response',20,19700,base.lineage(),model.initialization_report)
    for field,value in [('training_F_access',True),('F_outputs_read',1),('target_training',True),('parent_R46_commit','other')]:
        changed=copy.deepcopy(original);changed[field]=value
        with pytest.raises(ValueError):r.validate_payload(changed)


def test_identical_initial_state_for_both_probe_objectives():
    api();m=importlib.import_module('experiments.pamf_r46_response_distill.models')
    base=importlib.import_module('tests.test_pamf_r46_complex_fidelity');parent=base.parent_model()
    a=m.compact_from_parent(parent,'pair_aug_control');b=m.compact_from_parent(parent,'odd_response')
    assert all(torch.equal(value,b.state_dict()[key]) for key,value in a.state_dict().items())


@pytest.mark.parametrize('bad',['root','role','arm'])
def test_target_inputs_stay_closed_for_invalid_reference(monkeypatch,bad):
    api();e=importlib.import_module('experiments.pamf_r46_response_distill.evaluation')
    args=SimpleNamespace(output=Path('/runs')/e.d.RUN/e.d.ARMS[0],kind=e.d.KIND);role='target'
    if bad=='root':args.output=Path('/runs/other')/e.d.ARMS[0]
    elif bad=='role':role='query'
    else:args.output=args.output.parent/'other'
    monkeypatch.setattr(e.np,'load',lambda *a,**k:pytest.fail('Target arrays must remain closed'))
    with pytest.raises(ValueError):e.reference(args,role)


def test_existing_row_is_preserved_without_failure_marker(tmp_path,monkeypatch):
    api();r=importlib.import_module('experiments.pamf_r46_response_distill.runner')
    root=tmp_path/r.d.RUN;root.mkdir();row=root/r.d.ARMS[0];row.mkdir()
    marker=row/'complete.json';marker.write_text('{}')
    monkeypatch.setattr(r,'train',lambda *a:pytest.fail('Existing row cannot train'))
    with pytest.raises(ValueError):r.worker(SimpleNamespace(output=root,arm=r.d.ARMS[0]))
    assert list(row.iterdir())==[marker]


def test_truth_is_not_read_before_all_predictions(monkeypatch):
    api();e=importlib.import_module('experiments.pamf_r46_response_distill.evaluation')
    args=SimpleNamespace(output=Path('/runs')/e.d.RUN,arm=e.d.ARMS[0]);monkeypatch.setenv('CUDA_VISIBLE_DEVICES','')
    monkeypatch.setattr(e,'reference',lambda *a:(Path('/formal'),e.np.arange(2),None,None))
    class Incomplete(Exception):pass
    def stop(*a):raise Incomplete()
    monkeypatch.setattr(e.base,'prediction_arrays',stop)
    monkeypatch.setattr(e.base,'target_baseline',lambda *a:pytest.fail('Truth must remain closed'))
    with pytest.raises(Incomplete):e.score(args,'target')


def test_predecessor_requires_terminal_row_and_actual_owned_process_exit(tmp_path):
    api();r=importlib.import_module('experiments.pamf_r46_response_distill.runner')
    root=tmp_path/r.d.RUN;previous=tmp_path/r.d.PREDECESSOR;previous.mkdir()
    arm=r.d.ARMS[0];args=SimpleNamespace(output=root)
    entry=dict(GPU=r.d.GPUS[arm],arm='preceding_row',pid=999999999)
    (previous/'workers.json').write_text(json.dumps(dict(workers=[entry])))
    assert not r.predecessor_ready(args,arm)
    (previous/entry['arm']).mkdir();(previous/entry['arm']/'complete.json').write_text('{}')
    assert r.predecessor_ready(args,arm)
