from types import SimpleNamespace
import pytest
import torch
from experiments.pamf_dura_direct_train import f_guided,head_guided


def test_source_calibrated_branch_routes_all_children_and_rejects_cosine_checkpoint(tmp_path):
    base=f_guided.base
    keys=('RUN','configuration','command','learning_rate','direct_objective','save_model','load_model','smoke','train')
    original={key:getattr(base,key) for key in keys};objective=f_guided.objective;metric=f_guided.GUIDANCE_METRIC
    try:
        head_guided.configure();config=base.configuration()
        assert config['run']==head_guided.RUN and config['F_feature_weight']==0.006607891036090233
        assert config['training_F_access'] and config['training_TX_labels']==0 and not config['target_training']
        assert config['F_head_basis_rank']==5 and config['training_stages']==1 and config['epochs']==20
        assert base.train is f_guided.train and f_guided.objective is head_guided.objective
        assert f_guided.GUIDANCE_METRIC==config['F_guidance_metric']=='F_head_projected_feature_error'
        args=SimpleNamespace(root='root',commit='a'*40)
        assert all(base.command(args,mode,'scratch_joint','target')[3]==head_guided.MODULE for mode in ('worker','score'))
        old=dict(config,recipe='source_F_guided_cos010',F_feature_weight=.1)
        payload=dict(schema='dura_source_F_guided_v1',arm='scratch_joint',epoch=20,updates=4940,code_commit='a'*40,
            config=old,source_contract='original7/63/30',parent_run=None,parent_arm=None,initialization='scratch_no_checkpoint',
            training_TX_labels_read=0,training_F_access=True,V_training=False,target_training=False,
            F_teacher=f_guided.teacher_reference(),F_updates=0,model={})
        path=tmp_path/'old_cosine.pt';torch.save(payload,path)
        with pytest.raises(ValueError,match='F-guided checkpoint metadata'):
            base.load_model(path,'cpu',arm='scratch_joint',commit='a'*40)
    finally:
        for key,value in original.items():setattr(base,key,value)
        f_guided.objective=objective;f_guided.GUIDANCE_METRIC=metric
