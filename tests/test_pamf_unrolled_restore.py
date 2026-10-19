import copy,json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from torch import nn
from experiments.pamf_unrolled_restore import design as d,models as m,runtime,evaluation as e,training as t
from experiments.pamf_action_restore.operators import Action
from experiments.pamf_uncertainty_restore.models import weighted_map
from experiments.pamf_uncertainty_restore import design as ancestor
from experiments.pamf_action_restore.models import ActionRestorer
from experiments.pamf_reception_restore.models import ComplexTFUNet
from experiments.pamf_fullresolution_restore import runtime as old_runtime

torch.set_num_threads(2)

def complex_random(*shape):return torch.complex(torch.randn(*shape,dtype=torch.float64),torch.randn(*shape,dtype=torch.float64))

def test_warm_start_MAP_matches_dense_real_system_and_keeps_gradients():
    torch.manual_seed(9);n=10;h=.1*complex_random(1,3);h[:,0]+=.8
    a=1+.1*complex_random(1,n);b=.05*complex_random(1,n);action=Action(h,a,b)
    y=complex_random(1,n);prior=complex_random(1,n).requires_grad_();initial=complex_random(1,n).requires_grad_();var=torch.linspace(.1,.8,n)[None].double()
    eye=torch.eye(2*n,dtype=torch.float64);basis=torch.complex(eye[:,:n],eye[:,n:])
    expanded=Action(h.expand(2*n,-1),a.expand(2*n,-1),b.expand(2*n,-1));values=expanded(basis)
    matrix=torch.cat((values.real,values.imag),1).T;w=torch.cat((1/var[0],1/var[0]));yp=torch.cat((y[0].real,y[0].imag));pp=torch.cat((prior[0].real,prior[0].imag))
    expected=torch.linalg.solve(matrix.T@(w[:,None]*matrix)+eye/.3,matrix.T@(w*yp)+pp/.3)
    result=weighted_map(action,y,prior,var,.3,60,initial=initial)
    torch.testing.assert_close(torch.cat((result[0].real,result[0].imag)),expected,atol=1e-9,rtol=1e-9)
    result.abs().square().mean().backward();assert torch.isfinite(prior.grad).all() and torch.isfinite(initial.grad).all()
    for invalid in (torch.zeros_like(initial.real),initial[:,:-1],initial.detach()*float('nan')):
        with pytest.raises(ValueError):weighted_map(action,y,prior,var,initial=invalid)

def test_two_step_warm_solve_decreases_its_soft_objective():
    torch.manual_seed(17);action=Action(.1*complex_random(2,3),1+.1*complex_random(2,14),.05*complex_random(2,14))
    y,prior,initial=complex_random(2,14),complex_random(2,14),complex_random(2,14);var=torch.ones(2,14)*.1
    objective=lambda z:((action(z)-y).abs().square()/var+(z-prior).abs().square()/.2).sum(-1)
    result=weighted_map(action,y,prior,var,.2,2,initial=initial);assert (objective(result)<objective(initial)).all()

class Frozen(nn.Module):
    def __init__(self):super().__init__();self.anchor=nn.Parameter(torch.tensor(1.))
    def estimate(self,x):
        n=x.shape[-1];z=torch.arange(n,device=x.device,dtype=x.dtype)
        main=((.8+.2*(z/13).sin())*torch.exp(.2j*z/n))[None].expand(len(x),-1)
        image=torch.ones_like(main)*(.05+.02j);kernel=torch.tensor([.8,.15+.05j,0.,.03],dtype=main.dtype,device=x.device)[None].expand(len(x),-1)
        return .9*x,Action(kernel,main,image),torch.full((len(x),),.01,device=x.device)

def test_three_steps_clone_initial_weights_equal_output_and_all_gradients(monkeypatch):
    inputs=torch.randn(2,2,256)*.4;outputs=[];counts=[];real=m.weighted_map
    def solve(*args,**kwargs):counts.append((kwargs['steps'],kwargs['initial'].requires_grad));return real(*args,**kwargs)
    monkeypatch.setattr(m,'weighted_map',solve)
    for candidate in d.ROWS:
        torch.manual_seed(3);model=m.UnrolledRestorer(candidate,Frozen()).train()
        assert not model.frozen.training and not any(p.requires_grad for p in model.frozen.parameters())
        stages,*_=model.stage_outputs(inputs);assert len(stages)==3 and all(v.shape==inputs.shape for v in stages)
        outputs.append(stages[-1].detach());(stages[-1]-inputs-.01).square().mean().backward()
        assert all(any(p.grad is not None and p.grad.abs().sum()>0 for p in core.parameters()) for core in model.prior)
        assert all(p.grad is None for p in model.frozen.parameters())
        for core in model.prior:assert all(torch.equal(v,core.state_dict()[k]) for k,v in model.prior[0].state_dict().items())
    assert torch.equal(*outputs) and len(counts)==6 and all(steps==2 for steps,_ in counts)
    assert counts[1][1] and counts[2][1]

def test_unrolled_is_per_packet_with_nonzero_priors_and_threefold_cost():
    torch.manual_seed(5);model=m.UnrolledRestorer('unroll_unshared',Frozen()).eval()
    for core in model.prior:nn.init.normal_(core.out.weight,std=.001)
    value=torch.randn(3,2,256)*.3
    torch.testing.assert_close(model(value),torch.cat([model(x[None]) for x in value]),atol=2e-6,rtol=2e-5)
    shared=m.UnrolledRestorer('unroll_shared',Frozen())
    assert sum(p.numel() for p in model.prior.parameters())==3*sum(p.numel() for p in shared.prior.parameters())

def parent():return dict(coarse_run=ancestor.COARSE_RUN,coarse_commit=ancestor.COARSE_COMMIT,action_run=ancestor.ACTION_RUN,action_commit=ancestor.ACTION_COMMIT,data_contract='original7/63/30',training_TX_F_reads=0,target_training=False)

def test_unrolled_checkpoint_full_state_and_permissions(tmp_path):
    model=m.UnrolledRestorer('unroll_unshared',ActionRestorer(ComplexTFUNet()));path=tmp_path/'final.pt'
    runtime.save_checkpoint(path,model,None,20,4940,parent());fresh,value=runtime.load_checkpoint(path,'cpu')
    assert len(fresh.prior)==3 and all(torch.equal(v,fresh.state_dict()[k]) for k,v in model.state_dict().items())
    with pytest.raises(ValueError):old_runtime.validate_payload(value)
    for key,bad in [('training_F_access',True),('target_training',True),('inherited_prior_epochs',20),('candidate','unet_control')]:
        invalid=copy.deepcopy(value);invalid[key]=bad
        with pytest.raises(ValueError):runtime.validate_payload(invalid)
    invalid=copy.deepcopy(value);invalid['config']['steps']=5
    with pytest.raises(ValueError):runtime.validate_payload(invalid)

def test_prior_seed_reset_is_after_original_parent_loading(monkeypatch):
    def load(*a):torch.randn(load.count);return Frozen(),parent()
    load.count=3;monkeypatch.setattr(runtime,'frozen_context',load);first,_=runtime.initialize(d.ROWS[0],None,'cpu');load.count=99
    second,_=runtime.initialize(d.ROWS[0],None,'cpu');assert all(torch.equal(v,second.prior.state_dict()[k]) for k,v in first.prior.state_dict().items())

def summary(low=82.,mid=90.,nmse=.03):
    value=dict(status='ANALYZED',kind=d.KIND,role='source',count=1800,F_run='20261016-phase1-F-supcon-e50-clean-manysig-m9-r02',F_seed=d.MODEL_SEED,original_default_head=True,all_predictions_before_truth=True,training_TX_labels_read=0,training_F_access=False,head_fit_updates=0,target_access=False)
    value['views']={v:dict(restored=dict(waveform=dict(iq_nmse=.001 if v=='clean' else nmse))) for v in e.VIEWS};records=[]
    for view in e.VIEWS:
        acc=98.5 if view=='clean' else low if view=='practical_low_urban' else mid if view=='practical_mid_urban' else 97.
        records.append(dict(view=view,dimension='overall',value='all',readout='restored',accuracy=acc,baseline_accuracy=98.5))
        for dimension,groups in [('TX',range(6)),('RX',(1,3,4,6,8))]:records.extend(dict(view=view,dimension=dimension,value=str(i),readout='restored',accuracy=acc,baseline_accuracy=98.5) for i in groups)
    value['records']=records;return value

def test_three_reference_guards_and_only_qualified_fixed_source_ranking():
    single,native,fft=summary(),summary(low=81.,mid=89.),summary(low=81.5,mid=89.5)
    candidates={d.ROWS[0]:summary(low=83.,mid=91.,nmse=.028),d.ROWS[1]:summary(low=84.,mid=92.,nmse=.028)}
    result=e.select_source(candidates,single,native,fft);assert result['selected']==d.ROWS[1] and result['target_rows']==[d.ROWS[1],d.CONTROL]
    next(r for r in fft['records'] if r['view']=='practical_low_urban' and r['dimension']=='TX')['accuracy']=88.
    result=e.select_source(candidates,single,native,fft);assert not result['qualified'] and result['target_rows']==[]
    assert all(v['vs_single_native']['qualified'] for v in result['checks'].values())

def test_fixed_rank_uses_worst_RX_before_name_and_rejects_role():
    a=summary(low=83.,mid=91.,nmse=.028);b=copy.deepcopy(a)
    next(r for r in a['records'] if r['view']=='practical_low_urban' and r['dimension']=='RX')['accuracy']=70.
    assert e.selection_rank(d.ROWS[1],b)<e.selection_rank(d.ROWS[0],a)
    a['role']='target'
    with pytest.raises(ValueError):e.select_source({d.ROWS[0]:a,d.ROWS[1]:b},summary(),summary(),summary())

def decision():return dict(status='SOURCE_DECISION_FROZEN',qualified=True,selected=d.ROWS[0],target_rows=[d.ROWS[0],d.CONTROL])

def test_unselected_or_unqualified_target_stops_before_IO(tmp_path,monkeypatch):
    monkeypatch.setattr(e,'original_predict',lambda *a,**k:pytest.fail('Forbidden target input IO'))
    for row,value in [(d.ROWS[1],decision()),(d.ROWS[0],dict(decision(),qualified=False)),(d.ROWS[0],dict(decision(),selected='unet_control'))]:
        (tmp_path/'source_decision.json').write_text(json.dumps(value))
        with pytest.raises(ValueError):e.predict(SimpleNamespace(row=row,output=tmp_path),'target')

@pytest.mark.parametrize('row,role',[(d.ROWS[0],'source'),(d.CONTROL,'target')])
def test_unrolled_and_single_checkpoint_routes_make_all_seven_predictions(tmp_path,monkeypatch,row,role):
    import experiments.pamf_uncertainty_restore.evaluation as shared
    source=tmp_path/'input';source.mkdir()
    for view in e.VIEWS:np.save(source/(view+'.npy'),np.ones((16,2,256),np.float32))
    output=tmp_path/'output';(output/row/d.KIND).mkdir(parents=True);(output/'source_decision.json').write_text(json.dumps(decision()))
    args=SimpleNamespace(row=row,output=output,reader_folder=tmp_path/'unused',source_cache=tmp_path/'unused',single_run=tmp_path/'single',fft_run=tmp_path/'fft')
    class Toy(nn.Module):
        def forward(self,x):return x
        def training_outputs(self,x):return x,torch.ones((len(x),256)),None,x
    class Reader(nn.Module):
        def __init__(self):super().__init__();self.anchor=nn.Parameter(torch.ones(1))
        def features(self,x):assert self.anchor._version>=0;return x.flatten(1)[:,:160]
        def classify_features(self,x):return x[:,:6]
    calls=[]
    def loader(path,device):calls.append(str(path));return Toy(),dict(candidate='unet_control' if row==d.CONTROL else row,kind=d.KIND,epoch=20,updates=4940)
    monkeypatch.setattr(e,'load_checkpoint',loader);monkeypatch.setattr(e,'load_single',loader);monkeypatch.setattr(e,'validate_references',lambda *a:None)
    monkeypatch.setattr(shared,'load_checkpoint',lambda *a:pytest.fail('Default oldschema loader forbidden'))
    monkeypatch.setattr(shared,'reference',lambda *a:(source,np.arange(16),None,None));monkeypatch.setattr(shared,'load_reader',lambda *a:Reader())
    monkeypatch.setattr(shared,'array_tensor',lambda x,device:torch.tensor(x.tolist(),dtype=torch.float32));monkeypatch.setattr(shared,'tensor_array',lambda x:np.asarray(x.tolist(),np.float32))
    for key in ('synchronize','reset_peak_memory_stats'):monkeypatch.setattr(torch.cuda,key,lambda:None)
    monkeypatch.setattr(torch.cuda,'max_memory_allocated',lambda:0)
    e.predict(args,role);folder=output/row
    if role=='target':folder=folder/'target_confirmation'
    assert tuple(shared.prediction_arrays(folder/d.KIND/'prediction',np.arange(16),d.KIND,role))==e.VIEWS
    assert len(calls)==1 and (str(args.single_run) in calls[0])==(row==d.CONTROL)

def test_new_training_rows_validate_before_GPU_and_fixed_budget():
    with pytest.raises(ValueError):t.train(SimpleNamespace(row='unet_control',parent_action=Path(d.ACTION)))
    cfg=d.configuration();assert cfg['steps']==3 and cfg['CG_per_step']==2 and cfg['CG_total']==6 and cfg['epochs']==20 and cfg['updates']==4940
    assert cfg['L_exposures']==126000 and cfg['U_exposures']==1134000 and cfg['calibration_count']==0 and not cfg['training_F_access']
