import copy
import numpy as np
import pytest
import torch
from experiments.pamf_dura_efficiency import r17_teacher_paths as trial


def valid_metadata():
    value=dict(schema='unrolled_prior_checkpoint_v1',candidate='unroll_unshared',epoch=20,updates=4940,
        prior_initialization='scratch',inherited_prior_epochs=0,training_TX_labels_read=0,training_F_access=False,target_training=False,
        parent=dict(coarse_run='20261019-phase1-reception-restoration-selfsup-manysig-m6-r02',
        coarse_commit='7efdba92932a86bf4b0492a565d35377564a26da',action_run=trial.base.ACTION,
        action_commit='c8ccdfdbd03ab5ad8828b1eb9210b99a664ebe7b',data_contract='original7/63/30',training_TX_F_reads=0,target_training=False))
    done=dict(status='SOURCE_TRAINED',candidate='unroll_unshared',epochs=20,updates=4940,L_exposures=126000,U_exposures=1134000,
        TX_labels_read=0,F_outputs_read=0,F_weights_read=0,target_reads=0,frozen_context_updates=0,parameters=4665073,
        prior_initialization='scratch',inherited_prior_epochs=0)
    decision=dict(status='SOURCE_DECISION_FROZEN',selected='unroll_unshared',target_reads_before_decision=0,training_TX_F_access=False)
    return value,done,dict(commit=trial.TEACHER_COMMIT),decision


def test_only_original_source_teacher_and_before_target_choice_are_accepted():
    metadata=valid_metadata();trial.validate_teacher(*metadata)
    for field,newvalue in [('training_F_access',True),('training_TX_labels_read',1),('target_training',True),('updates',4939)]:
        changed=copy.deepcopy(metadata);changed[0][field]=newvalue
        with pytest.raises(ValueError):trial.validate_teacher(*changed)
    changed=copy.deepcopy(metadata);changed[0]['parent']['data_contract']='other'
    with pytest.raises(ValueError):trial.validate_teacher(*changed)
    changed=copy.deepcopy(metadata);changed[3]['target_reads_before_decision']=1
    with pytest.raises(ValueError):trial.validate_teacher(*changed)


def test_IQ_and_F_teacher_choices_are_separate_metrics():
    x=torch.ones(2,2,256);iq=x*2;features=torch.eye(2);logits=torch.tensor([[3.,1.],[0.,4.]])
    values=trial.measures(iq,x,features,logits,features,logits.flip(1))
    torch.testing.assert_close(values['iq_NMSE'],torch.ones(2))
    torch.testing.assert_close(values['feature_cosine_error'],torch.zeros(2))
    assert values['teacher_choice_agreement'].sum()==0


def test_aggregate_preserves_paired_probe_count_and_teacher_agreement():
    values=dict(iq_NMSE=np.array([.1,.2,.3]),teacher_choice_agreement=np.array([1.,0.,1.]))
    value=trial.aggregate(values,np.array([True,False,True]))
    assert value['count']==2 and value['teacher_correct']==2 and value['metrics']['iq_NMSE']['mean']==pytest.approx(.2)
