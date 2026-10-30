import json
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from experiments.pamf_dura_direct_train import fresh_views as trial


def write_view(folder,view=21,count=4):
    for name,shape in trial.SHAPES.items():np.save(folder/(name+'.npy'),np.zeros((count,*shape),np.float32))
    np.savez(folder/'groups.npz',scene=trial.source_scenes(view,count),state=np.full(count,'state'),lock=np.full(count,'lock'))
    record=dict(count=count,namespace=f'learned-action/train/epoch{view:02d}',TX_labels_read=0,F_outputs_read=0,
        target_reads=0,native_engine_modified=False,max_native_replay_error=0.)
    (folder/'complete.json').write_text(json.dumps(record),encoding='utf-8')
    return record


def test_new_scene_draws_keep_fixed_distribution_and_do_not_repeat_old_namespace():
    a=trial.source_scenes(21);b=trial.source_scenes(22)
    assert np.array_equal(a,trial.source_scenes(21)) and not np.array_equal(a,b)
    assert set(a)==set(trial.SCENES)|{'clean'} and .13<(a=='clean').mean()<.17
    with pytest.raises(ValueError):trial.source_scenes(20)
    with pytest.raises(ValueError):trial.source_scenes(31)


def test_new_cache_boundary_rejects_hidden_truth_and_nonfinite_iq(tmp_path):
    record=write_view(tmp_path);result=trial.validate_fresh_view(tmp_path,21,count=4)
    assert result['count']==4 and result['numeric_bytes']==4*6660
    record['TX_labels_read']=1;(tmp_path/'complete.json').write_text(json.dumps(record),encoding='utf-8')
    with pytest.raises(ValueError):trial.validate_fresh_view(tmp_path,21,count=4)
    write_view(tmp_path);values=np.zeros((4,2,256),np.float32);values[0,0,0]=np.nan;np.save(tmp_path/'received.npy',values)
    with pytest.raises(ValueError):trial.validate_fresh_view(tmp_path,21,count=4)


def test_new_cache_rejects_wrong_view_dtype_and_scene_order(tmp_path):
    write_view(tmp_path)
    with pytest.raises(ValueError):trial.validate_fresh_view(tmp_path,22,count=4)
    np.save(tmp_path/'variance.npy',np.zeros(4,np.float64))
    with pytest.raises(ValueError):trial.validate_fresh_view(tmp_path,21,count=4)
    write_view(tmp_path);np.savez(tmp_path/'groups.npz',scene=np.full(4,'unknown'),state=np.full(4,'state'),lock=np.full(4,'lock'))
    with pytest.raises(ValueError):trial.validate_fresh_view(tmp_path,21,count=4)


def test_fixed_joint_budget_checkpoint_and_all_child_routes(monkeypatch,tmp_path):
    for key,value in vars(trial.base).copy().items():monkeypatch.setattr(trial.base,key,value)
    trial.configure();base=trial.base;c=base.configuration();args=SimpleNamespace(root=tmp_path,commit='a'*40)
    assert (c['epochs'],c['updates'],c['corruption_epochs'],c['parameters'],c['training_stages'])==(30,7410,30,266951,1)
    assert c['teacher_model'] is None and not c['training_F_access'] and c['allowed_GPUs']==[6]
    assert c['original_source_draws_reused']==list(range(1,21)) and c['new_source_draws']==list(range(21,31))
    assert base.paths(tmp_path)['cache']==tmp_path/'runs'/trial.RUN/'source_cache'
    assert all(base.command(args,m,'scratch_joint','target')[3]==trial.MODULE for m in ('worker','score'))
    assert trial.learning_rate(7409)==pytest.approx(.00002)
    model=base.scratch_model(2026092701).eval();x=torch.ones(1,2,256)
    with torch.no_grad():expected=model(x)
    checkpoint=tmp_path/'model.pt';base.save_model(checkpoint,model,'scratch_joint',0,0,args.commit)
    loaded=base.load_model(checkpoint,'cpu',arm='scratch_joint',commit=args.commit,epoch=0)
    with torch.no_grad():torch.testing.assert_close(loaded(x),expected,atol=0,rtol=0)
