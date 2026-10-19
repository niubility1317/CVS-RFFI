import json
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from torch import nn
from experiments.pamf_parameter_balance import design as d
from experiments.pamf_parameter_balance.gradients import merge_gradients,parameter_gradients
from experiments.pamf_parameter_balance.models import initialize,save_checkpoint,load_checkpoint,JointResidual
from experiments.pamf_reception_restore.models import ComplexTFUNet

torch.set_num_threads(2)


def test_parameter_group_caps_differ_from_whole_vector_cap():
    w=torch.tensor([1.,0.,100.,0.]);g=torch.tensor([20.,0.,.1,0.])
    a,global_stats=merge_gradients(w,g,2,d.ARMS[0]);b,parts_stats=merge_gradients(w,g,2,d.ARMS[1])
    assert global_stats['all']['scale']==1 and parts_stats['coarse']['scale']==.05 and parts_stats['correction']['scale']==1
    assert torch.equal(a,w+g) and torch.allclose(b,torch.tensor([2.,0.,100.1,0.]))


@pytest.mark.parametrize('arm',d.ARMS)
@pytest.mark.parametrize('seed',[2,5,11])
def test_norm_cap_only_reduces_identity_and_preserves_wave_direction(arm,seed):
    torch.manual_seed(seed);w=torch.randn(19);g=10*torch.randn_like(w)
    merged,stats=merge_gradients(w,g,7,arm)
    ranges=[('all',0,19)] if arm==d.ARMS[0] else [('coarse',0,7),('correction',7,19)]
    for key,start,end in ranges:
        adjusted=(merged-w)[start:end];wave=w[start:end];row=stats[key]
        assert 0<=row['scale']<=1 and torch.allclose(adjusted,row['scale']*g[start:end],atol=2e-6)
        assert adjusted.norm()<=wave.norm()+1e-5
        assert torch.dot(merged[start:end],wave)>=-1e-5
        assert row['identity_after_norm2']<=row['wave_norm2']+1e-4


@pytest.mark.parametrize('arm',d.ARMS)
def test_zero_task_vectors_have_defined_finite_behavior(arm):
    z=torch.zeros(6);one=torch.ones(6)
    merged,stats=merge_gradients(z,one,3,arm);assert torch.equal(merged,z) and all(s['scale']==0 for s in stats.values())
    merged,stats=merge_gradients(one,z,3,arm);assert torch.equal(merged,one) and all(s['scale']==1 for s in stats.values())
    merged,stats=merge_gradients(z,z,3,arm);assert torch.equal(merged,z) and all(s['scale']==1 for s in stats.values())


class TinyJoint(nn.Module):
    def __init__(self):super().__init__();self.coarse=nn.Linear(2,3);self.correction=nn.Linear(3,1)
    def forward(self,x):return self.correction(self.coarse(x)).flatten()


@pytest.mark.parametrize('arm',d.ARMS)
@pytest.mark.parametrize('count',[1,5])
def test_inactive_cap_matches_task_mean_gradient_and_adamw_step(arm,count):
    torch.manual_seed(2);a=TinyJoint();b=TinyJoint();b.load_state_dict(a.state_dict());x=torch.randn(count,2)
    oa=torch.optim.AdamW(a.parameters(),lr=3e-5);ob=torch.optim.AdamW(b.parameters(),lr=3e-5)
    pa=a(x);pb=b(x);wa=(pa+3).square();wb=(pb+3).square();ia=1e-5*pa.square();ib=1e-5*pb.square()
    stats=parameter_gradients(a,wa,ia,arm);(wb.mean()+ib.mean()).backward()
    assert all(s['scale']==1 for s in stats.values())
    assert all(torch.allclose(p.grad,q.grad,atol=1e-7,rtol=1e-6) for p,q in zip(a.parameters(),b.parameters()))
    oa.step();ob.step();assert all(torch.allclose(p,q,atol=1e-8) for p,q in zip(a.parameters(),b.parameters()))


@pytest.mark.parametrize('arm',d.ARMS)
def test_actual_parameter_grads_and_component_statistics_accumulate(arm):
    from experiments.pamf_parameter_balance.runner import empty_gradient_state,accumulate_gradients
    model=TinyJoint();x=torch.randn(4,2);p=model(x)
    stats=parameter_gradients(model,(p-.7).square(),100*(p+2).square(),arm)
    assert all(v.grad is not None and torch.isfinite(v.grad).all() for v in model.parameters())
    state=empty_gradient_state(arm);accumulate_gradients(state,stats,2)
    assert state['calls']==1 and state['IQ_examples']==4 and set(state['groups'])==set(stats)
    assert all(s['scaled_steps']==1 and 0<s['scale_sum']<1 for s in state['groups'].values())


@pytest.mark.parametrize('bad',['shape','dtype','nan','cut0','cutall','arm'])
def test_invalid_parameter_gradient_contract(bad):
    w=torch.ones(5);g=w.clone();cut=2;arm=d.ARMS[0]
    if bad=='shape':g=g[:-1]
    elif bad=='dtype':g=g.double()
    elif bad=='nan':g[0]=float('nan')
    elif bad=='cut0':cut=0
    elif bad=='cutall':cut=5
    else:arm='unknown'
    with pytest.raises(ValueError):merge_gradients(w,g,cut,arm)


def test_parameter_assignment_rejects_unregistered_network_grouping():
    model=nn.Linear(2,1);p=model(torch.ones(1,2)).flatten()
    with pytest.raises(ValueError):parameter_gradients(model,p.square(),p.square(),d.ARMS[0])


@pytest.mark.parametrize('arm',d.ARMS)
def test_real_restorer_both_branches_receive_finite_parameter_gradients(arm):
    model=JointResidual(ComplexTFUNet(),.1).train();x=torch.randn(2,2,256);p=model(x)
    wave=(p-x-.1).square().mean((1,2));identity=3*(p+.3).square().mean((1,2))
    stats=parameter_gradients(model,wave,identity,arm)
    for part in (model.coarse,model.correction):
        grads=[p.grad for p in part.parameters()]
        assert all(g is not None and torch.isfinite(g).all() for g in grads) and sum(float(g.abs().sum()) for g in grads)>0
    assert set(stats)==({'all'} if arm==d.ARMS[0] else {'coarse','correction'})


def test_parent_state_batch_independence_checkpoint_and_external_control(monkeypatch,tmp_path):
    from experiments.pamf_parameter_balance import models
    from experiments.pamf_residual_restore.models import ResidualDeterministic
    parent=ResidualDeterministic(ComplexTFUNet(),.1).eval();nn.init.normal_(parent.correction.out.weight,0,.02);opened=[]
    monkeypatch.setattr(models,'load_restorer',lambda path:(opened.append(path),parent)[1]);x=torch.randn(2,2,256);values=[]
    for arm in d.ARMS:
        model=initialize('original_R03_only','cpu',arm);values.append(model);assert sum(p.numel() for p in model.parameters())==2140840
        before={k:v.clone() for k,v in model.state_dict().items()}
        with torch.no_grad():assert torch.equal(model(x),parent(x));assert torch.allclose(model(x),torch.cat([model(v[None]) for v in x]),atol=1e-6,rtol=1e-5)
        assert all(torch.equal(v,before[k]) for k,v in model.state_dict().items())
        path=tmp_path/(arm+'.pt');save_checkpoint(path,model,None,arm,10,2470);loaded,_=load_checkpoint(path,arm,'cpu')
        with torch.no_grad():assert torch.equal(model(x),loaded(x))
    assert opened==['original_R03_only']*2 and all(torch.equal(v,values[1].state_dict()[k]) for k,v in values[0].state_dict().items())
    assert d.CONTROL_ARM=='per_frame_cap' and d.target_matrix()['external_control']['new_training'] is False
    assert all(d.configuration(arm)['control_run']==d.CONTROL_RUN for arm in d.ARMS)


@pytest.mark.parametrize('field,value',[('TX_labels_read',1),('target_training',True),('F_optimizer_updates',1),('head_fit_updates',1),
 ('parent_corrector_epochs',0),('parent_coarse_epochs',0),('coarse_updates',1),('updates',2471),('gradient_updates',1),('gradient_epochs',1)])
def test_checkpoint_permissions_budget_and_parent(tmp_path,field,value):
    path=tmp_path/'m.pt';save_checkpoint(path,JointResidual(ComplexTFUNet(),.1),None,d.ARM,0,0)
    item=torch.load(path,weights_only=False);item[field]=value;torch.save(item,path)
    with pytest.raises(ValueError):load_checkpoint(path,d.ARM,'cpu')


def fixed_source(root,arm):
    from experiments.pamf_parameter_balance import design as d
    root=root/d.RUN;row=root/arm;(row/'prediction').mkdir(parents=True)
    values={
      root/'launch.json':dict(commit='x',fixed_target_arms=list(d.ARMS)),root/'matrix.json':d.target_matrix(),
      row/'source_frozen.json':dict(status='SOURCE_FROZEN',arm=arm,epochs=10,updates=2470,source_only=True,target_reads=0,
         checkpoint_selection='fixed_final_E10',target_policy='FIXED_COMPARISON'),
      row/'training_complete.json':dict(epochs=10,updates=2470,coarse_updates=2470,trainable_parameters=2140840,gradient_updates=2470,gradient_epochs=10,
         F_optimizer_updates=0,head_fit_updates=0,TX_labels_read=0,target_reads=0),
      row/'prediction/complete.json':dict(status='SOURCE_PREDICTIONS_COMPLETE',count=1800,views=d.target_matrix()['views'],
         role='source_V',TX_label_reads=0,target_reads=0,state_updates=0),
      row/'summary.json':dict(status='ANALYZED',source_only=True,truth_last=True,arm=arm,count=1800,target_reads=0,
         diagnostic_accuracy=1.,diagnostic_NMSE=100.)}
    for path,value in values.items():path.write_text(json.dumps(value),encoding='utf-8')
    return SimpleNamespace(source_run=root,arm=arm,expected_commit='x'),values


def test_both_fixed_rows_may_test_without_source_score_selection_or_other_row_completion(tmp_path):
    from experiments.pamf_parameter_balance.target import source_check
    for arm in d.ARMS:
        args,_=fixed_source(tmp_path/arm,arm);source_check(args)
        assert not (args.source_run/'source_decision.json').exists()


@pytest.mark.parametrize('file,field,value',[
 ('launch','fixed_target_arms',[d.ARMS[0]]),('launch','commit','other'),('source_frozen','checkpoint_selection','best_V'),
 ('source_frozen','target_reads',1),('source_frozen','target_policy','SELECTED_CANDIDATE'),('training_complete','updates',2469),('training_complete','gradient_updates',0),
 ('training_complete','head_fit_updates',1),('training_complete','TX_labels_read',1),('training_complete','target_reads',1),
 ('prediction/complete','TX_label_reads',1),('prediction/complete','views',['clean']),('summary','truth_last',False)])
def test_incomplete_unregistered_or_contaminated_source_row_cannot_test(tmp_path,file,field,value):
    from experiments.pamf_parameter_balance.target import source_check
    args,_=fixed_source(tmp_path,d.ARMS[0]);path=(args.source_run if file=='launch' else args.source_run/args.arm)/(file+'.json')
    item=json.loads(path.read_text());item[field]=value;path.write_text(json.dumps(item))
    with pytest.raises(ValueError):source_check(args)


def packets(tmp_path):
    from experiments.pamf_parameter_balance.target import VIEWS,SEEDS,READOUTS
    saved=tmp_path/'prediction';saved.mkdir();ids=np.array(['a','b'])
    done=dict(status='PREDICTIONS_COMPLETE',views=list(VIEWS),count=2,F_seeds=list(SEEDS),readouts=list(READOUTS),
        TX_truth_reads=0,paired_clean_in_practical_inference=False,state_updates=0)
    (saved/'complete.json').write_text(json.dumps(done));values=dict(ids=ids,seeds=np.asarray(SEEDS),**{k:np.zeros((2,3),np.uint8) for k in READOUTS})
    for view in VIEWS:
        np.savez(saved/(view+'.npz'),**values);np.save(saved/(view+'_refined_iq.npy'),np.zeros((2,2,256),np.float32))
    return saved,ids,done,values


def test_full_seven_view_three_F_packet_before_truth(tmp_path):
    from experiments.pamf_parameter_balance.target import VIEWS,completed_predictions
    _,ids,_,_=packets(tmp_path);assert set(completed_predictions(tmp_path,ids))==set(VIEWS)


@pytest.mark.parametrize('bad',['missingview','earlytruth','pairedclean','stateupdate','badseed','badclass','ids','missingIQ','nanIQ'])
def test_invalid_target_packets_fail_before_truth(tmp_path,bad):
    from experiments.pamf_parameter_balance.target import VIEWS,SEEDS,completed_predictions
    saved,ids,done,values=packets(tmp_path)
    if bad=='missingview':done['views']=list(VIEWS[:-1])
    elif bad=='earlytruth':done['TX_truth_reads']=1
    elif bad=='pairedclean':done['paired_clean_in_practical_inference']=True
    elif bad=='stateupdate':done['state_updates']=1
    elif bad=='badseed':values['seeds']=np.asarray(SEEDS[:1])
    elif bad=='badclass':values['refined'][0,0]=6
    elif bad=='ids':values['ids']=ids[::-1]
    elif bad=='missingIQ':np.save(saved/(VIEWS[-1]+'_refined_iq.npy'),np.zeros((1,2,256),np.float32))
    else:np.save(saved/(VIEWS[-1]+'_refined_iq.npy'),np.full((2,2,256),np.nan,np.float32))
    (saved/'complete.json').write_text(json.dumps(done));np.savez(saved/(VIEWS[-1]+'.npz'),**values)
    with pytest.raises(ValueError):completed_predictions(tmp_path,ids)


def test_controller_progress_can_update_through_source_target_completion(tmp_path):
    from experiments.pamf_parameter_balance.runner import write_stage_status
    for stage in ('source','target','complete'):
        state=dict(active={} if stage=='complete' else {d.ARMS[0]:dict(stage=stage,pid=123)},completed=stage=='complete')
        write_stage_status(tmp_path,state)
        assert json.loads((tmp_path/'stages.json').read_text())==state
        assert not (tmp_path/'stages.json.tmp').exists()
