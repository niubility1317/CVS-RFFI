import json
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from experiments.pamf_consensus_identity import design as d
from experiments.pamf_consensus_identity.identity import BankBuilder,identity_terms,rx_indices
from experiments.pamf_consensus_identity.models import JointResidual,save_checkpoint,load_checkpoint
from experiments.pamf_consensus_identity.target import completed_predictions,source_check,SEEDS,READOUTS,VIEWS
from experiments.pamf_reception_restore.models import ComplexTFUNet


def confident_logits(n,label=0):
    logits=torch.zeros(n,3,6);logits[:,:,label]=10.
    return logits


def test_native_consensus_rejects_disagreement_and_low_confidence_without_truth():
    builder=BankBuilder();features=torch.randn(3,3,160,requires_grad=True)
    logits=confident_logits(3,2);logits[1,2]=0.;logits[1,2,3]=10.;logits[2]=0.
    pseudo,mask=builder.update(features,torch.tensor([1,3,4]),logits)
    assert pseudo.tolist()==[2,2,0] and mask.tolist()==[True,False,False]
    bank=builder.finalize();assert bank.counts.sum()==1 and not bank.prototypes.requires_grad
    assert bank.state()['TX_labels_read']==0 and not bank.state()['inference_uses_bank']


def test_otherRX_targets_exclude_own_group_balance_RX_and_preserve_teacher_coordinates():
    features=torch.zeros(4,3,160)
    for teacher in range(3):
        features[:2,teacher,teacher*10]=1.
        features[2,teacher,teacher*10+1]=1.;features[3,teacher,teacher*10+2]=1.
    builder=BankBuilder();pseudo,accepted=builder.update(features,torch.tensor([1,1,3,4]),confident_logits(4))
    bank=builder.finalize();prototype,available=bank.lookup(torch.tensor([1,3]),pseudo[:2],accepted[:2],'cpu')
    assert available.all()
    for teacher in range(3):
        assert prototype[0,teacher,teacher*10]==0
        assert prototype[0,teacher,teacher*10+1].item()==pytest.approx(2**-.5)
        assert prototype[0,teacher,teacher*10+2].item()==pytest.approx(2**-.5)
        assert prototype[1,teacher,teacher*10].item()==pytest.approx(2**-.5)
        assert prototype[1,teacher,teacher*10+2].item()==pytest.approx(2**-.5)
    assert bank.counts[0,0]==2 and bank.counts[1,0]==1


def test_missing_otherRX_class_does_not_fall_back_to_own_RX():
    builder=BankBuilder();pseudo,accepted=builder.update(torch.randn(2,3,160),torch.tensor([1,1]),confident_logits(2))
    bank=builder.finalize();prototype,mask=bank.lookup(torch.tensor([1,1]),pseudo,accepted,'cpu')
    assert not mask.any() and torch.count_nonzero(prototype)==0


@pytest.mark.parametrize('kind',['RX','shape','nonfinite','dtype'])
def test_illegal_identity_bank_inputs_fail(kind):
    features=torch.randn(2,3,160);logits=confident_logits(2);rx=torch.tensor([1,3])
    if kind=='RX':rx[0]=2
    elif kind=='shape':features=features[:,:1]
    elif kind=='nonfinite':logits[0,0,0]=float('nan')
    else:rx=rx.float()
    with pytest.raises(ValueError):BankBuilder().update(features,rx,logits)


def test_identity_additions_only_differentiate_accepted_observed_records_and_detach_targets():
    embedding=torch.randn(4,160,requires_grad=True);logits=torch.randn(4,6,requires_grad=True)
    prototype=torch.randn(2,160,requires_grad=True);pseudo=torch.tensor([1,2]);mask=torch.tensor([True,False])
    native,alignment=identity_terms(embedding[:2],logits[:2],pseudo,mask,prototype,mask,d.ARMS[1])
    (native.mean()+alignment.mean()).backward()
    assert embedding.grad[0].abs().sum()>0 and logits.grad[0].abs().sum()>0
    assert torch.count_nonzero(embedding.grad[1:])==torch.count_nonzero(logits.grad[1:])==0
    assert prototype.grad is None
    embed2=embedding.detach().requires_grad_();scores2=logits.detach().requires_grad_()
    ce,cosine=identity_terms(embed2[:2],scores2[:2],pseudo,mask,prototype,mask,d.ARMS[0])
    assert cosine.tolist()==[0.,0.]
    (ce.mean()+cosine.mean()).backward();assert embed2.grad is None


def test_matched_training_configuration_and_train_only_identity_bank():
    left,right=(d.configuration(a) for a in d.ARMS)
    assert left['epochs']==right['epochs']==10 and left['updates']==right['updates']==2470
    assert left['native_weight']==right['native_weight']==.3 and left['consensus_confidence']==.9
    assert left['prototype_weight']==0 and right['prototype_weight']==.3
    assert left['teacher_seeds']==[2026092701,2026092702,2026092703]
    with pytest.raises(ValueError):d.configuration('unknown')


@pytest.mark.parametrize('field,value',[('TX_labels_read',1),('target_training',True),('F_optimizer_updates',1),
 ('head_fit_updates',1),('parent_corrector_epochs',0),('coarse_updates',1),('updates',2471),('parent_coarse_epochs',50)])
def test_checkpoint_permissions_and_budget(tmp_path,field,value):
    path=tmp_path/'m.pt';save_checkpoint(path,JointResidual(ComplexTFUNet(),.1),None,d.ARM,0,0)
    item=torch.load(path,weights_only=False);item[field]=value;torch.save(item,path)
    with pytest.raises(ValueError):load_checkpoint(path,d.ARM,'cpu')


def test_checkpoint_does_not_need_bank_or_RX_at_inference(tmp_path):
    path=tmp_path/'m.pt';save_checkpoint(path,JointResidual(ComplexTFUNet(),.1),None,d.ARMS[1],0,0,dict(inference_uses_bank=False))
    model,item=load_checkpoint(path,d.ARMS[1],'cpu')
    assert item['identity_training_state']==dict(inference_uses_bank=False)
    assert not model.training and all(not p.requires_grad for p in model.parameters())
    assert all('prototype' not in k and 'pseudo' not in k for k in model.state_dict())
    item['config']['native_weight']=1.;torch.save(item,path)
    with pytest.raises(ValueError):load_checkpoint(path,d.ARMS[1],'cpu')


def summary(gain=0,nmse=.028,drop=0):
    records=[]
    for view in VIEWS:
        for name in ('identity','refined'):
            accuracy=98.5 if view=='clean' else 80.+(gain if name=='refined' else 0)
            if view=='clean' and name=='refined':accuracy-=drop
            records.append(dict(view=view,readout=name,baseline='identity',dimension='overall',accuracy=accuracy))
    return dict(source_only=True,count=1800,target_reads=0,records=records,waveform=[dict(view=v,NMSE=dict(refined=nmse)) for v in VIEWS])


def test_prospective_low_first_rule_preserves_mid_clean_waveform_and_other_views():
    result=d.source_rule(dict(zip(d.ARMS,(summary(1),summary(2)))),summary())
    assert result['selected_arm']==d.ARMS[1]
    result=d.source_rule(dict(zip(d.ARMS,(summary(.2),summary(1,.034)))),summary())
    assert not result['qualified']
    for bad in ('mid','other','clean','truth'):
        item=summary(1)
        for row in item['records']:
            if row['readout']=='refined' and ((bad=='mid' and row['view']=='practical_mid_urban') or (bad=='other' and row['view']=='practical_high')):row['accuracy']=79.
            if bad=='clean' and row['readout']=='refined' and row['view']=='clean':row['accuracy']=98.
        if bad=='truth':item['target_reads']=1
        if bad=='truth':
            with pytest.raises(ValueError):d.source_rule({a:item for a in d.ARMS},summary())
        else:assert not d.source_rule({a:item for a in d.ARMS},summary())['qualified']
    good=summary(1)
    for row in good['records']:
        if row['readout']=='refined' and row['view']=='practical_mid_urban':row['accuracy']=80.
    assert d.source_rule({a:good for a in d.ARMS},summary())['qualified']
    with pytest.raises(ValueError):d.source_rule({d.ARM:good},summary())


def packets(tmp_path):
    saved=tmp_path/'prediction';saved.mkdir();ids=np.array(['a','b'])
    done=dict(status='PREDICTIONS_COMPLETE',views=list(VIEWS),count=2,F_seeds=list(SEEDS),readouts=list(READOUTS),
        TX_truth_reads=0,paired_clean_in_practical_inference=False,state_updates=0)
    (saved/'complete.json').write_text(json.dumps(done));values=dict(ids=ids,seeds=np.asarray(SEEDS),**{k:np.zeros((2,3),np.uint8) for k in READOUTS})
    for view in VIEWS:
        np.savez(saved/(view+'.npz'),**values);np.save(saved/(view+'_refined_iq.npy'),np.zeros((2,2,256),np.float32))
    return saved,ids,done,values


def test_all_seven_views_before_truth(tmp_path):
    _,ids,_,_=packets(tmp_path);assert set(completed_predictions(tmp_path,ids))==set(VIEWS)


@pytest.mark.parametrize('bad',['missingview','earlytruth','pairedclean','stateupdate','badseed','badclass','ids','missingIQ'])
def test_invalid_target_packets_fail_before_truth(tmp_path,bad):
    saved,ids,done,values=packets(tmp_path)
    if bad=='missingview':done['views']=list(VIEWS[:-1])
    elif bad=='earlytruth':done['TX_truth_reads']=1
    elif bad=='pairedclean':done['paired_clean_in_practical_inference']=True
    elif bad=='stateupdate':done['state_updates']=1
    elif bad=='badseed':values['seeds']=np.asarray(SEEDS[:1])
    elif bad=='badclass':values['refined'][0,0]=6
    elif bad=='ids':values['ids']=ids[::-1]
    else:np.save(saved/(VIEWS[-1]+'_refined_iq.npy'),np.zeros((1,2,256),np.float32))
    (saved/'complete.json').write_text(json.dumps(done));np.savez(saved/(VIEWS[-1]+'.npz'),**values)
    with pytest.raises(ValueError):completed_predictions(tmp_path,ids)


def test_unselected_second_arm_cannot_access_target(tmp_path):
    root=tmp_path/d.RUN;root.mkdir();(root/'launch.json').write_text(json.dumps(dict(commit='x')))
    (root/'source_complete.json').write_text(json.dumps(dict(status='ANALYZED',source_only=True,arms=list(d.ARMS),target_reads=0)))
    (root/'source_decision.json').write_text(json.dumps(dict(qualified=True,selected_arm=d.ARMS[0],target_reads=0)))
    row=root/d.ARMS[1];row.mkdir();(row/'training_complete.json').write_text('{}')
    with pytest.raises(ValueError):source_check(SimpleNamespace(source_run=root,expected_commit='x',arm=d.ARMS[1]))
