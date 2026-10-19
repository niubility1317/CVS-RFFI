from types import SimpleNamespace
import importlib
import importlib.util
from pathlib import Path
import pytest
import torch

torch.set_num_threads(2)


def implementation():
    assert importlib.util.find_spec('experiments.pamf_analytic_unrolled_restore') is not None,'R22 analytic module is not implemented'
    return importlib.import_module('experiments.pamf_analytic_unrolled_restore.descriptors')


def packets(n=5):
    return torch.randn((n,2,256),generator=torch.Generator().manual_seed(4421))*.2


def test_public_reference_descriptor_matches_original_without_trained_F():
    d=implementation()
    from experiments.cvs_reference_identity.model import ExcitationResponse
    x=packets();reference=ExcitationResponse('reference_response')
    assert not list(reference.parameters())
    expected=reference(x);expected=torch.cat((expected[:,:27],expected[:,28:]),1)
    assert torch.allclose(d.AnalyticViews()(x)['reference'],expected,atol=2e-6,rtol=2e-5)


def test_mirror_PA_and_circularity_match_native_math():
    d=implementation()
    from experiments.cvs_identity_ce.model import build_model
    native=build_model(num_classes=6,model_size='M',dataset='wisig',input_len=256,sample_rate_hz=25e6,model_variant='lite_d',branch_ablation='no_dac',mixstyle_on=False,use_crra=False,physical_gate_variant='none')
    x=packets();feat,rho,_,pa=native._mirror_compressed_features(x)
    out=d.AnalyticViews()(x)
    assert torch.allclose(out['mirror'],feat.flatten(1),atol=1e-6,rtol=1e-5)
    assert torch.allclose(out['spectral'],torch.cat((pa,rho),1),atol=1e-6,rtol=1e-5)


@pytest.mark.parametrize('kind',['mirror','spectral','reference'])
def test_each_fixed_group_has_finite_nonzero_output_gradient(kind):
    d=implementation();x=packets().requires_grad_();target=packets().roll(3,-1)
    a=d.AnalyticViews()(x)[kind];b=d.AnalyticViews()(target)[kind].detach()
    gradient=torch.autograd.grad((a-b).square().mean(),x)[0]
    assert torch.isfinite(gradient).all() and gradient.norm()>0


@pytest.mark.parametrize('amplitude',[0.,1e-10,1.])
def test_silent_and_near_silent_features_and_gradients_are_finite(amplitude):
    d=implementation();x=(packets()*amplitude).requires_grad_()
    out=d.AnalyticViews()(x)
    assert {key:value.shape for key,value in out.items()}=={'mirror':(5,128),'spectral':(5,4),'reference':(5,28)}
    loss=sum(value.square().mean() for value in out.values());grad=torch.autograd.grad(loss,x)[0]
    assert all(torch.isfinite(value).all() for value in out.values()) and torch.isfinite(grad).all()


def test_features_are_packet_independent_and_have_no_learned_parameters():
    d=implementation();model=d.AnalyticViews();x=packets()
    assert not list(model.parameters())
    whole=model(x);single=[model(v[None]) for v in x]
    for key in whole:assert torch.allclose(whole[key],torch.cat([v[key] for v in single]),atol=1e-6,rtol=1e-5)


@pytest.mark.parametrize('bad',[torch.ones(2,256),torch.ones(2,2,128),torch.full((2,2,256),float('nan')),torch.ones(2,2,256,dtype=torch.int64)])
def test_invalid_current_IQ_is_rejected(bad):
    with pytest.raises(ValueError):implementation().AnalyticViews()(bad)


def test_exact_pair_loss_decomposition_and_only_IQ_skips_analytic_forward():
    d=implementation();scales={k:torch.ones(v) for k,v in d.DIMENSIONS.items()}
    x=packets();y=(x+.01*packets().roll(5,-1)).requires_grad_()
    loss,metrics=d.paired_loss(y,x,scales,analytic=True)
    assert torch.allclose(loss,metrics['iq_nmse']+.01*metrics['analytic_loss'])
    assert torch.allclose(metrics['analytic_loss'],sum(metrics[k+'_loss'] for k in d.DIMENSIONS)/3)
    class Forbidden:
        def __call__(self,*args):raise AssertionError('IQ control must not execute analytic branch')
    control,values=d.paired_loss(y,x,None,analytic=False,extractor=Forbidden())
    assert set(values)=={'iq_nmse'} and torch.equal(control,metrics['iq_nmse'])


def test_scales_are_population_std_from_clean_training_only_with_fixed_floor():
    d=implementation();x=packets(8);views=d.AnalyticViews()(x)
    scales=d.fit_scales([x[:3],x[3:]])
    for key,value in views.items():assert torch.allclose(scales[key],value.double().std(0,unbiased=False).clamp_min(.05),atol=1e-8,rtol=1e-6)
    assert not any(v.requires_grad for v in scales.values())


@pytest.mark.parametrize('mutation',['missing','shape','nan','small'])
def test_invalid_scale_cannot_silently_change_loss(mutation):
    d=implementation();scales={k:torch.ones(v) for k,v in d.DIMENSIONS.items()}
    if mutation=='missing':scales.pop('reference')
    elif mutation=='shape':scales['mirror']=torch.ones(127)
    elif mutation=='nan':scales['mirror'][0]=float('nan')
    else:scales['mirror'][0]=.049
    with pytest.raises(ValueError):d.paired_loss(packets(),packets(),scales,analytic=True)


def test_clean_target_has_no_gradient():
    d=implementation();clean=packets().requires_grad_();estimate=(packets()+.01).requires_grad_()
    loss,_=d.paired_loss(estimate,clean,{k:torch.ones(v) for k,v in d.DIMENSIONS.items()},analytic=True)
    loss.mean().backward();assert clean.grad is None and torch.isfinite(estimate.grad).all() and estimate.grad.norm()>0


def full_module(name):
    assert importlib.util.find_spec('experiments.pamf_analytic_unrolled_restore.'+name) is not None,'R22 '+name+' missing'
    return importlib.import_module('experiments.pamf_analytic_unrolled_restore.'+name)


def test_matched_R17_initialization_and_gradient_through_all_three_stages():
    import copy
    from torch import nn
    from experiments.pamf_action_restore.operators import Action
    from experiments.pamf_unrolled_restore.models import UnrolledRestorer
    m=full_module('models');d=full_module('design');desc=implementation()
    class Frozen(nn.Module):
        def __init__(self):super().__init__();self.anchor=nn.Parameter(torch.tensor(1.))
        def estimate(self,x):
            main=torch.ones((len(x),256),dtype=torch.complex64,device=x.device)*(.9+.1j)
            return .9*x,Action(torch.ones((len(x),1),dtype=main.dtype,device=x.device),main,.02*main),torch.full((len(x),),.01,device=x.device)
    torch.manual_seed(3);base=UnrolledRestorer('unroll_unshared',Frozen()).eval();x=packets(2);states=[];outputs=[]
    for row in d.ROWS:
        torch.manual_seed(3);model=m.AnalyticRestorer(row,Frozen()).train();states.append(copy.deepcopy(model.state_dict()))
        assert all(torch.equal(v,states[-1][k]) for k,v in base.state_dict().items())
        stages,*_=model.stage_outputs(x)
        for stage in stages:stage.retain_grad()
        loss,_=desc.paired_loss(stages[-1],x+.01,{k:torch.ones(v) for k,v in desc.DIMENSIONS.items()},analytic=row=='analytic_views')
        loss.mean().backward();outputs.append(stages[-1].detach())
        assert len(stages)==3 and all(s.grad is not None and torch.isfinite(s.grad).all() and s.grad.norm()>0 for s in stages)
        assert all(any(p.grad is not None and p.grad.norm()>0 for p in core.parameters()) for core in model.prior)
        assert not model.frozen.training and all(p.grad is None and not p.requires_grad for p in model.frozen.parameters())
    assert torch.equal(*outputs) and all(torch.equal(v,states[1][k]) for k,v in states[0].items())


def scaling_metadata():
    d=implementation()
    return dict(schema='training_clean_analytic_scale_v1',status='SOURCE_TRAINING_SCALES_FIXED',role='train',count=8192,L_count=819,U_count=7373,
        TX_labels_read=0,F_outputs_read=0,F_weights_read=0,V_reads=0,target_reads=0,optimizer_updates=0,
        data_contract='original7/63/30',order='original_first8192',floor=.05,scales={k:[1.]*v for k,v in d.DIMENSIONS.items()})


@pytest.mark.parametrize('key,bad',[('role','V'),('count',8191),('L_count',0),('TX_labels_read',1),('F_outputs_read',1),('V_reads',1),('target_reads',1),('optimizer_updates',1)])
def test_scaling_metadata_does_not_allow_validation_or_truth_calibration(key,bad):
    cal=full_module('calibration');value=scaling_metadata();value[key]=bad
    with pytest.raises(ValueError):cal.validate_metadata(value)


def test_full_checkpoint_uses_new_schema_and_contains_legal_training_scales(tmp_path):
    import copy
    from experiments.pamf_action_restore.models import ActionRestorer
    from experiments.pamf_reception_restore.models import ComplexTFUNet
    from experiments.pamf_uncertainty_restore import design as ancestor
    from experiments.pamf_unrolled_restore.runtime import validate_payload as old_validate
    r=full_module('runtime');m=full_module('models')
    parent=dict(coarse_run=ancestor.COARSE_RUN,coarse_commit=ancestor.COARSE_COMMIT,action_run=ancestor.ACTION_RUN,action_commit=ancestor.ACTION_COMMIT,data_contract='original7/63/30',training_TX_F_reads=0,target_training=False)
    model=m.AnalyticRestorer('analytic_views',ActionRestorer(ComplexTFUNet()));path=tmp_path/'final.pt'
    r.save_checkpoint(path,model,None,20,4940,parent,calibration=scaling_metadata());fresh,payload=r.load_checkpoint(path,'cpu')
    assert all(torch.equal(v,fresh.state_dict()[k]) for k,v in model.state_dict().items())
    assert payload['source_scaling']['count']==8192
    with pytest.raises(ValueError):old_validate(payload)
    for key,bad in [('training_F_access',True),('target_training',True),('candidate','unroll_unshared'),('schema','unrolled_prior_checkpoint_v1')]:
        invalid=copy.deepcopy(payload);invalid[key]=bad
        with pytest.raises(ValueError):r.validate_payload(invalid)
    invalid=copy.deepcopy(payload);invalid['source_scaling']=None
    with pytest.raises(ValueError):r.validate_payload(invalid)


def test_invalid_training_row_rejected_before_GPU_or_calibration_IO():
    t=full_module('training');d=full_module('design')
    with pytest.raises(ValueError):t.train(SimpleNamespace(row='analytic_views_r12',parent_action=Path(d.ACTION)))


def test_five_fixed_references_and_selected_only_target_access(tmp_path,monkeypatch):
    import json,copy
    from test_pamf_unrolled_restore import summary
    e=full_module('evaluation');d=full_module('design')
    refs=[summary(low=80+i,mid=88+i) for i in range(5)]
    candidates={name:summary(low=86+i,mid=94+i,nmse=.028) for i,name in enumerate(d.ROWS)}
    value=e.select_source(candidates,*refs);assert value['selected']=='analytic_views' and value['target_rows']==['analytic_views']
    # A sole fifth-reference TX failure must invalidate both candidates.
    fifth=copy.deepcopy(refs[-1]);next(r for r in fifth['records'] if r['view']=='practical_low_urban' and r['dimension']=='TX')['accuracy']=98.
    assert not e.select_source(candidates,*refs[:-1],fifth)['qualified']
    monkeypatch.setattr(e,'original_predict',lambda *a,**k:pytest.fail('Forbidden target access'))
    for row,decision in [('iq_control',value),('analytic_views',dict(value,qualified=False))]:
        (tmp_path/'source_decision.json').write_text(json.dumps(decision))
        with pytest.raises(ValueError):e.predict(SimpleNamespace(row=row,output=tmp_path),'target')


def test_R21_resource_predecessor_must_be_terminal_and_original_PID_absent(tmp_path):
    import json
    c=full_module('controller');d=full_module('design');root=tmp_path/d.PRECEDING;root.mkdir();proc=tmp_path/'proc';proc.mkdir()
    for name,value in [('launch.json',dict(commit=d.PRECEDING_COMMIT,pid=1)),('workers.json',dict(rows=[dict(pid=2)])),('complete.json',dict(status='ANALYZED'))]:
        (root/name).write_text(json.dumps(value))
    c.require_preceding_finished(SimpleNamespace(preceding_run=root),proc)
    p=proc/'2';p.mkdir();(p/'status').write_text('State:\tR (running)\n');(p/'cmdline').write_bytes((str(root)+' experiments.pamf_endpoint_restore.runner train').encode())
    with pytest.raises(ValueError):c.require_preceding_finished(SimpleNamespace(preceding_run=root),proc)


def test_actual_loss_gradient_report_uses_weighted_analytic_and_does_not_accumulate():
    desc=implementation();training=full_module('training')
    estimate=packets().requires_grad_();clean=packets().roll(1,-1)
    loss,metrics=desc.paired_loss(estimate,clean,{k:torch.ones(v) for k,v in desc.DIMENSIONS.items()},analytic=True)
    assert hasattr(training,'objective_gradients'),'Actual first-batch gradient measurement missing'
    expected=float(torch.autograd.grad(.01*metrics['analytic_loss'].mean(),estimate,retain_graph=True)[0].norm())
    result=training.objective_gradients(estimate,metrics,True)
    assert result['weighted_analytic']==pytest.approx(expected) and result['IQ']>0
    assert estimate.grad is None and all(result[key]>0 for key in ('mirror','spectral','reference'))
    loss.mean().backward();assert torch.isfinite(estimate.grad).all()


@pytest.mark.parametrize('role',['source','target'])
def test_registered_checkpoint_path_produces_all_seven_views(tmp_path,monkeypatch,role):
    import json,numpy as np
    from torch import nn
    import experiments.pamf_uncertainty_restore.evaluation as shared
    e=full_module('evaluation');d=full_module('design');source=tmp_path/'input';source.mkdir()
    for view in e.VIEWS:np.save(source/(view+'.npy'),np.ones((16,2,256),np.float32))
    row=d.ROWS[0];output=tmp_path/'output';(output/row/d.KIND).mkdir(parents=True)
    (output/'source_decision.json').write_text(json.dumps(dict(status='SOURCE_DECISION_FROZEN',qualified=True,selected=row,target_rows=[row])))
    class Toy(nn.Module):
        def forward(self,x):return x
        def training_outputs(self,x):return x,torch.ones((len(x),256)),None,x
    class Reader(nn.Module):
        def __init__(self):super().__init__();self.anchor=nn.Parameter(torch.ones(1))
        def features(self,x):assert self.anchor._version>=0;return x.flatten(1)[:,:160]
        def classify_features(self,x):return x[:,:6]
    calls=[]
    def loader(path,device):calls.append(path);return Toy(),dict(candidate=row,kind=d.KIND,epoch=20,updates=4940)
    monkeypatch.setattr(e,'load_checkpoint',loader)
    monkeypatch.setattr(shared,'load_checkpoint',lambda *a:pytest.fail('Oldschema loader forbidden'))
    monkeypatch.setattr(shared,'reference',lambda *a:(source,np.arange(16),None,None))
    monkeypatch.setattr(shared,'load_reader',lambda *a:Reader())
    monkeypatch.setattr(shared,'array_tensor',lambda x,device:torch.tensor(x.tolist(),dtype=torch.float32))
    monkeypatch.setattr(shared,'tensor_array',lambda x:np.asarray(x.tolist(),np.float32))
    for key in ('synchronize','reset_peak_memory_stats'):monkeypatch.setattr(torch.cuda,key,lambda:None)
    monkeypatch.setattr(torch.cuda,'max_memory_allocated',lambda:0)
    e.predict(SimpleNamespace(row=row,output=output,reader_folder=tmp_path/'unused',source_cache=tmp_path/'unused'),role)
    folder=output/row/('target_confirmation' if role=='target' else '')/d.KIND/'prediction'
    assert tuple(shared.prediction_arrays(folder,np.arange(16),d.KIND,role))==e.VIEWS and len(calls)==1
