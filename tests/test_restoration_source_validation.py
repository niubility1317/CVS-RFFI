import argparse
import copy
import numpy as np
import pytest
import torch
from experiments.pamf_phase3_restoration import source_validation as val


def source_roles():
    rx=np.repeat([1,3,4,6,8],360);day=np.tile(np.repeat([1,2,3],120),5)
    return np.asarray(['v'+str(i) for i in range(1800)]),np.full(1800,'V'),rx,day,np.asarray(['train'])


def test_source_V_is_fixed_disjoint_original_RX_day_and_never_training_role():
    args=source_roles();val.validation_roles(*args)
    for index,replacement in [(1,np.full(1800,'U_s')),(2,np.zeros(1800)),(4,np.asarray([args[0][0]]))]:
        bad=list(args);bad[index]=replacement
        with pytest.raises(ValueError,match='source-V'):val.validation_roles(*bad)


def test_loader_rejects_other_config_commit_epoch_and_freezes_valid_state(monkeypatch,tmp_path):
    config={'arm':'toy','source_only':True,'initialization':'scratch'}
    factory=lambda seed:torch.nn.Linear(2,2)
    monkeypatch.setitem(val.SPECS,'toy',('parent','fixed-code',lambda arm:config,factory))
    state=factory(0).state_dict();payload=dict(config=config,code_commit='fixed-code',epoch=30,updates=7410,state=state)
    path=tmp_path/'fixed.pt';torch.save(payload,path)
    G=val.load_frozen(path,'toy','cpu')
    assert not G.training and all(not p.requires_grad for p in G.parameters())
    for key,value in [('config',dict(config,source_only=False)),('code_commit','wrong'),('epoch',29),('updates',7163)]:
        bad=copy.deepcopy(payload);bad[key]=value;torch.save(bad,path)
        with pytest.raises(ValueError,match='fixedfinal30'):val.load_frozen(path,'toy','cpu')


def test_all_seven_validation_views_are_required():
    with pytest.raises(ValueError,match='x7views'):val.summarize({'scene':np.asarray(val.SCENES)})
    records={'scene':np.repeat(val.SCENES,1800)}
    for method in val.METHODS:
        for metric in val.METRICS:records[method+'__'+metric]=np.ones(12600)
    result=val.summarize(records)
    assert len(result)==7 and all(q['count']==1800 for q in result.values())
    records['scene'][0]=val.SCENES[1]
    with pytest.raises(ValueError,match='Every sourceV'):val.summarize(records)


def test_queue_does_not_admit_GPU_before_full_parent_Main_completion(monkeypatch,tmp_path):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','');(tmp_path/'runs').mkdir()
    def forbidden():raise AssertionError('GPU requested before parent completed')
    def stop_wait(seconds):raise RuntimeError('test-stop-wait')
    monkeypatch.setattr(val.varf,'recovery_gpu_choice',forbidden);monkeypatch.setattr(val.time,'sleep',stop_wait)
    with pytest.raises(RuntimeError,match='test-stop-wait'):val.controller(argparse.Namespace(root=tmp_path))
    out=tmp_path/'runs'/val.RUN
    assert (out/'waiting.json').is_file() and not (out/'submitted.json').exists()
