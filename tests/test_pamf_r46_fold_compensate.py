import copy,json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from torch import nn
from experiments.pamf_r44_pruned_refine.models import CompactSuccessor as ParentModel
from experiments.pamf_r46_fold_compensate import design as d,models as m,runtime as r,evaluation as e,runner
from experiments.pamf_r46_fold_compensate.calibration import ChannelMoment,collect,validate_calibration
from experiments.pamf_r46_fold_compensate.compensation import Link,reconstruction,folded_weight,make_links,grouped_links
from experiments.pamf_uncertainty_restore import design as ancestor

torch.set_num_threads(2)

def lineage():
    return dict(coarse_run=ancestor.COARSE_RUN,coarse_commit=ancestor.COARSE_COMMIT,action_run=ancestor.ACTION_RUN,
        action_commit=ancestor.ACTION_COMMIT,data_contract='original7/63/30',training_TX_F_reads=0,target_training=False)

@pytest.fixture(scope='module')
def calibrated():
    torch.manual_seed(55);parent=ParentModel('prune_refine').eval().requires_grad_(False)
    with torch.no_grad():
        for core in (*parent.prior,parent.frozen.coarse.spectral,parent.frozen.coarse.temporal,parent.frozen.estimator.core):core.out.weight.normal_(std=.001)
        parent.frozen.estimator.global_head.weight.normal_(std=.001)
    iq=torch.randn(128,2,256)*.1
    observed=collect(parent,[iq,iq*.98],batch=8)
    value=dict(schema='R46_source_channel_reconstruction_v1',parent_run=d.COMPACT_PARENT_RUN,parent_commit=d.COMPACT_PARENT_COMMIT,
        source_unique_IQ=128,L_s=64,U_s=64,received_views=['01','20'],sampling_seed=d.CAL_SEED,positions_per_conv_input=d.CAL_POSITIONS,
        ridge_fraction=d.RIDGE_FRACTION,TX_labels_read=0,F_reads=0,V_reads=0,target_reads=0,clean_IQ_reads=0,optimizer_updates=0,
        parent_unchanged=True,no_intercept=True,weight_only_maps=True,calibration_shared_once=True,**{k:v for k,v in observed.items() if k!='moments'})
    validate_calibration(value)
    return parent,value,observed['moments']

def test_channel_second_moment_matches_independent_matrix_product_and_actual_position_counts():
    values=np.arange(2*3*32,dtype=np.float64).reshape(2,3,32)/17
    stat=ChannelMoment(3,positions=2);stat.update(torch.tensor(values));stat.update(torch.tensor(values[:1]))
    columns=np.concatenate([values[:,:,[0,31]].transpose(1,0,2).reshape(3,-1),values[:1,:,[0,31]].transpose(1,0,2).reshape(3,-1)],axis=1)
    assert stat.count==6
    np.testing.assert_allclose(np.array(stat.value().tolist()),columns@columns.T/6,rtol=1e-13)
    pooled=ChannelMoment(3);pooled.update(torch.tensor(values.mean(-1)));assert pooled.count==2

def test_ridge_preserves_retained_rows_and_reduces_deleted_channel_error_even_with_collinearity():
    a=torch.tensor([[1.,2.,3.],[2.,4.,6.],[.5,1.,1.5],[1.,-1.,0.]],dtype=torch.float64)
    gram=a@a.T/3;keep=torch.tensor([0,3]);projection,info=reconstruction(gram,keep,1e-3)
    assert torch.equal(projection[keep],torch.eye(2,dtype=torch.float64))
    assert info['reconstructed_channel_error']<=info['uncorrected_channel_error']
    zeros,_=reconstruction(torch.zeros(4,4,dtype=torch.float64),keep,1e-3)
    assert torch.isfinite(zeros).all()
    identity,_=reconstruction(gram,torch.arange(4),1e-3);assert torch.equal(identity,torch.eye(4,dtype=torch.float64))

@pytest.mark.parametrize('bad',['nonfinite','asymmetric','duplicate','outofrange','ridge'])
def test_bad_channel_fit_inputs_reject(bad):
    gram=torch.eye(4,dtype=torch.float64);keep=torch.tensor([0,2]);ridge=1e-3
    if bad=='nonfinite':gram[0,0]=float('nan')
    elif bad=='asymmetric':gram[0,1]=1
    elif bad=='duplicate':keep=torch.tensor([0,0])
    elif bad=='outofrange':keep=torch.tensor([0,4])
    else:ridge=0.
    with pytest.raises(ValueError):reconstruction(gram,keep,ridge)

def test_fold_exact_linear_channel_relation_including_zero_padded_boundary_and_bias():
    torch.manual_seed(56);source=nn.Conv1d(3,4,5,padding=2).double();target=nn.Conv1d(2,4,5,padding=2).double()
    keep=torch.tensor([0,2]);projection=torch.tensor([[1.,0.],[.7,-.2],[0.,1.]],dtype=torch.float64)
    link=Link('toy',source,target,keep,torch.arange(4));x=torch.randn(2,2,19,dtype=torch.float64)
    with torch.no_grad():target.weight.copy_(folded_weight(link,projection,1.));target.bias.copy_(source.bias)
    full=torch.einsum('ij,bjt->bit',projection,x)
    assert torch.allclose(source(full),target(x),rtol=1e-12,atol=1e-12)
    with torch.no_grad():target.weight.copy_(folded_weight(link,projection,.5))
    desired=.5*source.weight[:,keep]+.5*torch.einsum('oik,ij->ojk',source.weight,projection)
    assert torch.allclose(target.weight,desired,rtol=1e-12,atol=1e-12)

def test_linear_pooled_action_head_fold_uses_same_channel_rule():
    source=nn.Linear(3,5).double();target=nn.Linear(2,5).double();keep=torch.tensor([0,2])
    p=torch.tensor([[1.,0.],[.5,.25],[0.,1.]],dtype=torch.float64);link=Link('head',source,target,keep,torch.arange(5))
    with torch.no_grad():target.weight.copy_(folded_weight(link,p,1.));target.bias.copy_(source.bias)
    x=torch.randn(4,2,dtype=torch.float64);assert torch.allclose(source(x@p.T),target(x),atol=1e-12)

def test_complete_calibration_ties_pool_moments_once_and_keeps_parent_frozen(calibrated):
    parent,value,moments=calibrated
    assert value['frozen_parent_IQ_inputs']==256 and value['folded_module_uses']>value['folded_unique_weights']>0
    assert all(p.grad is None and not p.requires_grad for p in parent.parameters())
    for name,item in value['projections'].items():
        if name.startswith('prior') and len(item['members'])>1:assert len(item['members'])==3 and item['sampled_channel_vectors']==256*16*3
        assert item['reconstructed_channel_error']<=item['uncorrected_channel_error']+1e-8
        assert moments[name]['sampled_channel_vectors']==item['sampled_channel_vectors']

def test_decoder_concat_offset_and_shared_map_contract(calibrated):
    parent,value,_=calibrated;model,maps=m.weight_initialization(parent);links=make_links(parent,model,maps)
    decoder=next(v for v in links if v.name=='action.decode.0.0.net.0')
    assert torch.equal(decoder.inputs[:40],maps['action'][3])
    assert torch.equal(decoder.inputs[40:],44+maps['action'][2])
    tied=next(g for g in grouped_links(links) if len(g)==3);tied[1].inputs=tied[1].inputs.flip(0)
    with pytest.raises(ValueError):grouped_links(links)

@pytest.mark.parametrize('arm',d.ARMS)
def test_same_inference_graph_joint_gradients_freeze_teacher_and_strict_standalone_checkpoint(calibrated,tmp_path,arm):
    parent,cal,_=calibrated;before=copy.deepcopy(parent.state_dict());model=m.compact_from_parent(parent,arm,cal);model.initialization_report['seconds']=0.
    r.validate_aliases(model)
    assert sum(p.numel() for p in model.parameters())==273765
    assert sum(p.numel() for p in model.frozen.parameters())==185615
    clean=(torch.randn(2,2,256)*.1).requires_grad_(True);received=clean.detach()+torch.randn_like(clean)*.005
    with torch.no_grad():teacher=parent.stage_outputs(received)[0]
    teacher=[v.requires_grad_(True) for v in teacher];student=model.stage_outputs(received)[0]
    loss,terms=m.objective(student,teacher,clean,1,arm);assert torch.allclose(loss,terms['iq_nmse']+.05*terms['spectral_error']+terms['teacher_stage_nmse'])
    loss.mean().backward()
    for component in (model.frozen.coarse,model.frozen.estimator,*model.prior):
        gradients=[p.grad for p in component.parameters() if p.grad is not None]
        assert gradients and all(torch.isfinite(g).all() for g in gradients) and any(g.abs().sum()>0 for g in gradients)
    assert clean.grad is None and all(v.grad is None for v in teacher)
    assert all(p.grad is None for p in parent.parameters()) and all(torch.equal(v,before[k]) for k,v in parent.state_dict().items())
    a=model.prior[0].encoder.blocks[1][0].net;b=model.prior[1].encoder.blocks[1][0].net
    assert a[0].weight is b[0].weight and a[0].bias is not b[0].bias and a[1].weight is not b[1].weight
    path=tmp_path/'final.pt';model.eval();r.save_checkpoint(path,model,None,20,4940,lineage());fresh,payload=r.load_checkpoint(path,'cpu')
    with torch.no_grad():assert torch.equal(model(received),fresh(received))
    assert payload['optimizer'] is None and not payload['training_F_access']
    with pytest.raises(FileExistsError):r.save_checkpoint(path,model,None,20,4940,lineage())

@pytest.mark.parametrize('field,bad',[('TX_labels_read',1),('F_reads',1),('V_reads',1),('target_reads',1),('source_unique_IQ',127),('clean_IQ_reads',1),('parent_commit','unknown')])
def test_calibration_permissions_and_input_budget_reject(calibrated,field,bad):
    value=copy.deepcopy(calibrated[1]);value[field]=bad
    with pytest.raises(ValueError):validate_calibration(value)

@pytest.mark.parametrize('field,bad',[('training_F_access',True),('F_outputs_read',1),('target_training',True),('V_training',True),('updates',4941),('parent_R46_row','retain_refine')])
def test_checkpoint_source_permissions_and_fixed_endpoint_reject(calibrated,field,bad):
    model=m.compact_from_parent(calibrated[0],d.ARMS[0],calibrated[1]);model.initialization_report['seconds']=0.
    value=r.metadata(d.ARMS[0],20,4940,lineage(),model.initialization_report);r.validate_payload(value);value[field]=bad
    with pytest.raises(ValueError):r.validate_payload(value)

def test_corrupted_projection_or_shared_alias_rejects(calibrated):
    parent,cal,_=calibrated;value=copy.deepcopy(cal);name=next(iter(value['projections']));value['projections'][name]['projection'][value['projections'][name]['inputs'][0]][0]=.9
    with pytest.raises(ValueError):m.compact_from_parent(parent,d.ARMS[0],value)
    model=m.CompactSuccessor(d.ARMS[0]);state=copy.deepcopy(model.state_dict());key=next(k for k in state if k.startswith('prior.1.') and k.endswith('net.0.weight'));state[key]=state[key]+1
    with pytest.raises(ValueError):r.validate_aliases(model,state)

@pytest.mark.parametrize('arm',d.ARMS)
def test_each_fixed_row_automatically_tests_after_source_without_performance_gate(tmp_path,monkeypatch,arm):
    args=SimpleNamespace(output=tmp_path,arm=arm);(tmp_path/arm).mkdir();seen=[]
    monkeypatch.setattr(runner,'train',lambda *a:seen.append('train'));monkeypatch.setattr(runner,'predict',lambda a,role:seen.append('predict_'+role))
    monkeypatch.setattr(runner,'profile',lambda *a:seen.append('profile'));monkeypatch.setattr(runner,'score_subprocess',lambda a,role:seen.append('score_'+role))
    monkeypatch.setattr(runner.torch.cuda,'empty_cache',lambda:None);runner.worker(args)
    assert seen==['train','predict_source','profile','score_source','predict_target','score_target']

def test_target_truth_waits_for_full_prediction_artifacts(monkeypatch):
    args=SimpleNamespace(output=Path('/runs')/d.RUN,arm=d.ARMS[0]);monkeypatch.setenv('CUDA_VISIBLE_DEVICES','')
    monkeypatch.setattr(e,'reference',lambda *a:(Path('/formal'),e.np.arange(2),None,None))
    class Missing(Exception):pass
    def absent(*a):raise Missing()
    monkeypatch.setattr(e.base,'prediction_arrays',absent);monkeypatch.setattr(e.base,'target_baseline',lambda *a:pytest.fail('Truth must remain closed'))
    with pytest.raises(Missing):e.score(args,'target')

def test_fixed_source_only_compensation_configuration():
    cfg=d.configuration();assert json.loads(json.dumps(cfg))==cfg
    assert d.FOLD_STRENGTH=={'fold_full':1.,'fold_half':.5} and d.RIDGE_FRACTION==1e-3
    assert d.REUSED_CONTROL['row']=='prune_decay' and d.REUSED_CONTROL['run']!=d.PREDECESSOR_RUN
    assert all(d.teacher_weight(20,a)==pytest.approx(.1) for a in d.ARMS)
    assert (d.EPOCHS,d.UPDATES,d.BATCH,d.MICRO)==(20,4940,256,256)
