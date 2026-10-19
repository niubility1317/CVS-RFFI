from dataclasses import replace
import json
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from experiments.pamf_style_generalization.mixing import ARMS,make_plan,mix_statistics


def source():
    return np.array([1,1,3,3,4,4,6,6]),np.array([0,0,0,0,1,1,1,1]),np.ones(8,dtype=np.bool_)


def test_identity_protection_and_matched_effective_exposure_across_RX_controls():
    rx,pseudo,accepted=source();plans=[make_plan(rx,pseudo,accepted,np.random.default_rng(7),a,p=1.) for a in ARMS]
    assert np.array_equal(plans[0].physical_active,plans[1].physical_active)
    assert torch.equal(plans[0].coefficient,plans[1].coefficient)
    for arm,plan in zip(ARMS,plans):
        active=np.flatnonzero(plan.physical_active);donor=np.array(plan.donor[:8].tolist())
        assert len(active)>0 and np.all(pseudo[active]==pseudo[donor[active]])
        assert np.all((rx[active]==rx[donor[active]]) if arm==ARMS[0] else (rx[active]!=rx[donor[active]]))
        assert np.all(donor[active]!=active)
        assert torch.equal(plan.donor[:8]+8,plan.donor[8:])
        assert plan.donor[:8].max()<8 and plan.donor[8:].min()>=8
        assert torch.equal(plan.coefficient[:8],plan.coefficient[8:]) and torch.equal(plan.active[:8],plan.active[8:])
        assert plan.counts()['paired_IQ_applications_per_layer']==2*len(active)


def test_no_fallback_when_either_RX_regime_has_no_identity_partner():
    rx=np.array([1,3]);pseudo=np.array([0,0]);accepted=np.ones(2,dtype=np.bool_)
    for arm in ARMS:
        plan=make_plan(rx,pseudo,accepted,np.random.default_rng(1),arm,p=1.)
        assert not plan.active.any() and not plan.common_eligible.any()
        x=torch.randn(4,3,16);assert torch.equal(x,mix_statistics(x,plan))


def test_rejected_pseudo_records_are_never_donors_or_recipients_and_labels_permute():
    rx,pseudo,accepted=source();accepted[0]=False
    renamed=np.array([5,2])[pseudo]
    for arm in ARMS:
        a=make_plan(rx,pseudo,accepted,np.random.default_rng(2),arm,p=1.)
        b=make_plan(rx,renamed,accepted,np.random.default_rng(2),arm,p=1.)
        assert not a.physical_active[0] and 0 not in a.donor[a.active].tolist()
        assert torch.equal(a.donor,b.donor) and torch.equal(a.coefficient,b.coefficient)


def test_donor_statistics_detach_and_clean_half_cannot_feed_observed_half():
    rx,pseudo,accepted=source();plan=make_plan(rx,pseudo,accepted,np.random.default_rng(3),ARMS[1],p=1.)
    i=int(np.flatnonzero(plan.physical_active)[0]);donor=int(plan.donor[i])
    x=torch.randn(16,3,16,requires_grad=True);y=mix_statistics(x,plan);y[i].square().mean().backward()
    assert x.grad[i].abs().sum()>0 and torch.count_nonzero(x.grad[donor])==0
    assert torch.count_nonzero(x.grad[8:])==0
    x2=x.detach().clone();x2[8:]+=100.
    assert torch.equal(mix_statistics(x.detach(),plan)[:8],mix_statistics(x2,plan)[:8])


def test_mixed_moments_and_zero_variance_are_finite():
    rx,pseudo,accepted=source();plan=make_plan(rx,pseudo,accepted,np.random.default_rng(3),ARMS[1],p=1.)
    plan=replace(plan,coefficient=torch.full((16,1,1),.5))
    x=torch.arange(16.).reshape(1,1,16).repeat(16,3,1)
    for i in range(16):x[i]=x[i]*(i+1)+i
    y=mix_statistics(x,plan)
    for i in np.flatnonzero(plan.physical_active):
        j=int(plan.donor[i]);expected_mean=(x[i].mean(-1)+x[j].mean(-1))/2
        expected_std=(x[i].std(-1,unbiased=False)+x[j].std(-1,unbiased=False))/2
        assert torch.allclose(y[i].mean(-1),expected_mean,atol=2e-5,rtol=1e-5)
        assert torch.allclose(y[i].std(-1,unbiased=False),expected_std,atol=2e-5,rtol=1e-5)
    assert torch.isfinite(mix_statistics(torch.ones_like(x),plan)).all()


@pytest.mark.parametrize('bad',['RX','pseudo','mask','shape','arm','p','alpha'])
def test_invalid_source_mixing_inputs_rejected(bad):
    rx,pseudo,accepted=source();arm=ARMS[1];p=.5;alpha=.1
    if bad=='RX':rx[0]=2
    elif bad=='pseudo':pseudo[0]=6
    elif bad=='mask':accepted=accepted.astype(np.int64)
    elif bad=='shape':pseudo=pseudo[:1]
    elif bad=='arm':arm='unknown'
    elif bad=='p':p=2.
    else:alpha=0.
    with pytest.raises(ValueError):make_plan(rx,pseudo,accepted,np.random.default_rng(1),arm,p=p,alpha=alpha)


def test_actual_shallow_middle_mixing_and_inference_preserves_original_core():
    from experiments.pamf_style_generalization.models import StyleCore
    from experiments.pamf_reception_restore.models import Core
    torch.set_num_threads(2);torch.manual_seed(11)
    parent=Core(incoming=4);torch.nn.init.normal_(parent.out.weight,0,.02)
    model=StyleCore();model.load_state_dict(parent.state_dict(),strict=True)
    rx,pseudo,accepted=source();plan=make_plan(rx,pseudo,accepted,np.random.default_rng(3),ARMS[1],p=1.)
    x=torch.randn(16,4,256,requires_grad=True)
    with pytest.raises(ValueError):model(x)
    model.eval();assert torch.equal(model(x),parent(x))
    before={k:v.clone() for k,v in model.state_dict().items()}
    model.train();mixed=model(x,plan)
    assert not torch.allclose(mixed,parent(x))
    mixed.square().mean().backward()
    assert x.grad.abs().sum()>0 and torch.isfinite(x.grad).all()
    assert model.encoder.blocks[0][0].parameters().__next__().grad.abs().sum()>0
    applications=dict(model.style_applications)
    assert applications=={str(i):2*int(plan.physical_active.sum()) for i in (0,2)}
    assert model.style_calls=={'0':1,'2':1}
    model.eval();assert torch.equal(model(x),parent(x))
    assert applications==model.style_applications and all(torch.equal(v,before[k]) for k,v in model.state_dict().items())


def test_checkpoint_roundtrip_IQ_only_batch_independent_and_matched_parameters(tmp_path):
    from experiments.pamf_style_generalization.models import JointStyle,save_checkpoint,load_checkpoint
    from experiments.pamf_reception_restore.models import ComplexTFUNet
    from experiments.pamf_style_generalization import design as d
    model=JointStyle(ComplexTFUNet(),.1).eval();torch.nn.init.normal_(model.correction.out.weight,0,.02)
    assert sum(p.numel() for p in model.parameters() if p.requires_grad)==d.TRAINABLE_PARAMETERS==2140840
    x=torch.randn(2,2,256)
    with torch.no_grad():
        y=model(x);alone=torch.cat([model(v[None]) for v in x])
    assert torch.allclose(y,alone,atol=1e-6,rtol=1e-5)
    for arm in ARMS:
        path=tmp_path/(arm+'.pt');save_checkpoint(path,model,None,arm,d.EPOCHS,d.UPDATES)
        loaded,_=load_checkpoint(path,arm,'cpu')
        assert not loaded.training and all(not p.requires_grad for p in loaded.parameters())
        with torch.no_grad():assert torch.equal(loaded(x),y)
    assert model.correction.style_calls=={'0':0,'2':0}


@pytest.mark.parametrize('field,value',[('TX_labels_read',1),('target_training',True),('F_optimizer_updates',1),
 ('head_fit_updates',1),('parent_corrector_epochs',0),('parent_coarse_epochs',0),('coarse_updates',1),('updates',2471)])
def test_checkpoint_permissions_budget_and_parent(tmp_path,field,value):
    from experiments.pamf_style_generalization.models import JointStyle,save_checkpoint,load_checkpoint
    from experiments.pamf_reception_restore.models import ComplexTFUNet
    path=tmp_path/'m.pt';save_checkpoint(path,JointStyle(ComplexTFUNet(),.1),None,ARMS[0],0,0)
    item=torch.load(path,weights_only=False);item[field]=value;torch.save(item,path)
    with pytest.raises(ValueError):load_checkpoint(path,ARMS[0],'cpu')


def fixed_source(root,arm):
    from experiments.pamf_style_generalization import design as d
    root=root/d.RUN;row=root/arm;(row/'prediction').mkdir(parents=True)
    values={
      root/'launch.json':dict(commit='x',fixed_target_arms=list(ARMS)),root/'matrix.json':d.target_matrix(),
      row/'source_frozen.json':dict(status='SOURCE_FROZEN',arm=arm,epochs=10,updates=2470,source_only=True,target_reads=0,
         checkpoint_selection='fixed_final_E10',target_policy='FIXED_COMPARISON'),
      row/'training_complete.json':dict(epochs=10,updates=2470,coarse_updates=2470,trainable_parameters=2140840,
         F_optimizer_updates=0,head_fit_updates=0,TX_labels_read=0,target_reads=0),
      row/'prediction/complete.json':dict(status='SOURCE_PREDICTIONS_COMPLETE',count=1800,views=d.target_matrix()['views'],
         role='source_V',TX_label_reads=0,target_reads=0,state_updates=0),
      row/'summary.json':dict(status='ANALYZED',source_only=True,truth_last=True,arm=arm,count=1800,target_reads=0,
         diagnostic_accuracy=1.,diagnostic_NMSE=100.)}
    for path,value in values.items():path.write_text(json.dumps(value),encoding='utf-8')
    return SimpleNamespace(source_run=root,arm=arm,expected_commit='x'),values


def test_both_fixed_rows_may_test_without_source_score_selection_or_other_row_completion(tmp_path):
    from experiments.pamf_style_generalization.target import source_check
    for arm in ARMS:
        args,_=fixed_source(tmp_path/arm,arm);source_check(args)
        assert not (args.source_run/'source_decision.json').exists()


@pytest.mark.parametrize('file,field,value',[
 ('launch','fixed_target_arms',[ARMS[0]]),('launch','commit','other'),('source_frozen','checkpoint_selection','best_V'),
 ('source_frozen','target_reads',1),('source_frozen','target_policy','SELECTED_CANDIDATE'),('training_complete','updates',2469),
 ('training_complete','head_fit_updates',1),('training_complete','TX_labels_read',1),('training_complete','target_reads',1),
 ('prediction/complete','TX_label_reads',1),('prediction/complete','views',['clean']),('summary','truth_last',False)])
def test_incomplete_unregistered_or_contaminated_source_row_cannot_test(tmp_path,file,field,value):
    from experiments.pamf_style_generalization.target import source_check
    args,_=fixed_source(tmp_path,ARMS[0]);path=(args.source_run if file=='launch' else args.source_run/args.arm)/(file+'.json')
    item=json.loads(path.read_text());item[field]=value;path.write_text(json.dumps(item))
    with pytest.raises(ValueError):source_check(args)


def packets(tmp_path):
    from experiments.pamf_style_generalization.target import VIEWS,SEEDS,READOUTS
    saved=tmp_path/'prediction';saved.mkdir();ids=np.array(['a','b'])
    done=dict(status='PREDICTIONS_COMPLETE',views=list(VIEWS),count=2,F_seeds=list(SEEDS),readouts=list(READOUTS),
        TX_truth_reads=0,paired_clean_in_practical_inference=False,state_updates=0)
    (saved/'complete.json').write_text(json.dumps(done));values=dict(ids=ids,seeds=np.asarray(SEEDS),**{k:np.zeros((2,3),np.uint8) for k in READOUTS})
    for view in VIEWS:
        np.savez(saved/(view+'.npz'),**values);np.save(saved/(view+'_refined_iq.npy'),np.zeros((2,2,256),np.float32))
    return saved,ids,done,values


def test_full_seven_view_three_F_packet_before_truth(tmp_path):
    from experiments.pamf_style_generalization.target import VIEWS,completed_predictions
    _,ids,_,_=packets(tmp_path);assert set(completed_predictions(tmp_path,ids))==set(VIEWS)


@pytest.mark.parametrize('bad',['missingview','earlytruth','pairedclean','stateupdate','badseed','badclass','ids','missingIQ','nanIQ'])
def test_invalid_target_packets_fail_before_truth(tmp_path,bad):
    from experiments.pamf_style_generalization.target import VIEWS,SEEDS,completed_predictions
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
    from experiments.pamf_style_generalization.runner import write_stage_status
    for stage in ('source','target','complete'):
        state=dict(active={} if stage=='complete' else {ARMS[0]:dict(stage=stage,pid=123)},completed=stage=='complete')
        write_stage_status(tmp_path,state)
        assert json.loads((tmp_path/'stages.json').read_text())==state
        assert not (tmp_path/'stages.json.tmp').exists()
