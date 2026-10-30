from types import SimpleNamespace
import json
import pytest

from experiments.pamf_dura_direct_train import runner as base


def test_E40_never_accepts_E20_checkpoint_as_final(monkeypatch):
    config=dict(base.configuration(),epochs=40,updates=9880)
    monkeypatch.setattr(base,'configuration',lambda:config)
    payload=dict(schema='dura_direct_joint_v1',arm='scratch_joint',epoch=20,updates=4940,
        code_commit='a'*40,config=config,source_contract='original7/63/30',parent_run=None,parent_arm=None,
        initialization='scratch_no_checkpoint',training_TX_labels_read=0,training_F_access=False,V_training=False,target_training=False)
    monkeypatch.setattr(base.torch,'load',lambda *a,**k:payload)
    with pytest.raises(ValueError,match='fixed-source checkpoint'):
        base.load_model('unused','cpu',arm='scratch_joint',commit='a'*40)


def test_prediction_waits_for_complete_registered_budget(tmp_path,monkeypatch):
    from experiments.pamf_dura_direct_train import evaluation
    config=dict(base.configuration(),epochs=40,updates=9880)
    monkeypatch.setattr(evaluation,'configuration',lambda:config)
    row=tmp_path/'scratch_joint';row.mkdir()
    done=dict(status='SOURCE_TRAINED',epoch=20,updates=4940,training_F_access=False,target_reads=0)
    file=row/'training_complete.json';file.write_text(json.dumps(done))
    touched=[]
    def reference(*a):touched.append(True);raise RuntimeError('budget passed')
    monkeypatch.setattr(evaluation,'reference',reference)
    with pytest.raises(ValueError,match='Complete frozen legal source training'):
        evaluation.predict(SimpleNamespace(arm='scratch_joint'),{'output':tmp_path},'target')
    assert not touched
    file.write_text(json.dumps(dict(done,epoch=40,updates=9880)))
    with pytest.raises(RuntimeError,match='budget passed'):
        evaluation.predict(SimpleNamespace(arm='scratch_joint'),{'output':tmp_path},'target')


def test_long_budget_is_one_scratch_stage_and_every_child_keeps_recipe():
    from experiments.pamf_dura_direct_train import long_budget
    original=(base.RUN,base.configuration,base.command,base.learning_rate,base.direct_objective)
    try:
        long_budget.configure();config=base.configuration()
        assert (config['epochs'],config['updates'],config['training_stages'])==(40,9880,1)
        assert config['L_exposures']==252000 and config['U_exposures']==2268000
        assert not config['ancestor_checkpoints'] and not config['training_F_access']
        assert (config['allowed_GPUs'],config['max_total_visible_GPU_processes'])==([4,6,7],3)
        args=SimpleNamespace(root='root',commit='a'*40)
        for mode in ('worker','score'):
            assert base.command(args,mode,'scratch_joint','target')[3]==long_budget.MODULE
        assert base.learning_rate(246)==pytest.approx(.001)
        assert base.learning_rate(9879)==pytest.approx(.00002)
        with pytest.raises(ValueError):base.learning_rate(9880)
    finally:
        base.RUN,base.configuration,base.command,base.learning_rate,base.direct_objective=original
