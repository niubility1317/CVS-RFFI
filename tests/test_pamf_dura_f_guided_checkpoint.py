from types import SimpleNamespace
import pytest
import torch
from experiments.pamf_dura_direct_train import f_guided


def test_guided_loader_rejects_wrong_teacher_permissions_and_source_before_build(tmp_path,monkeypatch):
    config={'training_F_access':True,'F_guided_training':True}
    monkeypatch.setattr(f_guided.base,'configuration',lambda:config)
    payload=dict(schema='dura_source_F_guided_v1',arm='scratch_joint',epoch=20,updates=4940,code_commit='a'*40,
        config=config,source_contract='original7/63/30',parent_run=None,parent_arm=None,initialization='scratch_no_checkpoint',
        training_TX_labels_read=0,training_F_access=True,V_training=False,target_training=False,
        F_teacher=f_guided.teacher_reference(),F_updates=0,model={})
    for key,wrong in (('training_F_access',False),('training_TX_labels_read',1),('target_training',True),
                      ('source_contract','other'),('F_teacher',{}),('F_updates',1),('schema','dura_direct_joint_v1')):
        value=dict(payload);value[key]=wrong
        path=tmp_path/(key+'.pt');torch.save(value,path)
        with pytest.raises(ValueError,match='F-guided checkpoint metadata'):
            f_guided.load_model(path,'cpu',arm='scratch_joint',commit='a'*40)


def test_guided_permission_and_loader_routing_reaches_worker_and_scorer():
    base=f_guided.base
    keys=('RUN','configuration','command','learning_rate','direct_objective','save_model','load_model','smoke','train')
    original={key:getattr(base,key) for key in keys}
    try:
        f_guided.configure()
        args=SimpleNamespace(root='root',commit='a'*40)
        assert all(base.command(args,mode,'scratch_joint','target')[3]==f_guided.MODULE for mode in ('worker','score'))
        config=base.configuration()
        assert config['training_F_access'] and config['F_guided_training'] and config['F_feature_weight']==.1
        assert config['teacher_model']==f_guided.teacher_reference()
        assert config['training_TX_labels']==0 and not config['target_training']
        assert base.load_model is f_guided.load_model and base.train is f_guided.train
    finally:
        for key,value in original.items():setattr(base,key,value)
