import pytest
import torch

from experiments.pamf_dura_direct_train.runner import load_model,configuration


def test_single_stage_loader_rejects_inherited_or_wrong_source_state_before_build(tmp_path):
    value=dict(schema='dura_direct_joint_v1',arm='scratch_joint',epoch=20,updates=4940,
        code_commit='a'*40,config=configuration(),source_contract='original7/63/30',
        parent_run=None,parent_arm=None,initialization='scratch_no_checkpoint',
        training_TX_labels_read=0,training_F_access=False,V_training=False,target_training=False,
        model={})
    for key,wrong in [('parent_run','old-trained-parent'),('training_F_access',True),
                      ('source_contract','differentphysicalsplit'),('target_training',True)]:
        payload=dict(value);payload[key]=wrong
        path=tmp_path/(key+'.pt');torch.save(payload,path)
        with pytest.raises(ValueError,match='fixed-source checkpoint'):
            load_model(path,'cpu',arm='scratch_joint',commit='a'*40)
