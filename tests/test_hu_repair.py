import json

import pytest
import torch

from experiments.pamf_phase2_practical import hu_repair
from experiments.pamf_phase2_practical import group1_source


def test_hu_weight_is_in_source_recipe_and_only_hu():
    hu = group1_source.recipe('feature_separation_hu2024')
    assert hu['hu_similarity_weight'] == 1 / 8192
    assert 'hu_similarity_weight' not in group1_source.recipe('mrior_sda')


def test_phase1_jobs_separate_truth_and_only_depend_on_own_source(tmp_path):
    contract = tmp_path/'contract.json'
    contract.write_text('{}', encoding='utf8')
    context = tmp_path/'context.json'
    context.write_text(json.dumps({'handles': {'F0': {'classes':['a'], 'source_contract':str(contract),
        'tests': {k:{'views_root':str(tmp_path/k), 'truth':'secret'} for k in ('cross_rx','cross_both')}}}}), encoding='utf8')
    phase1 = tmp_path/'phase1.json'
    phase1.write_text(json.dumps({'context':str(context)}), encoding='utf8')
    rt = {'hu_phase1_runtime':str(phase1), 'old_classes':['a'], 'source_cache':'source',
          'source_contract':str(contract), 'run_root':str(tmp_path/'run'), 'run_id':'r',
          'method_profiles':['feature_separation_hu2024'], 'model_seeds':[1,2,3,4]}
    jobs=hu_repair.phase1_jobs(rt)
    assert len(jobs)==8
    for j in jobs:
        if j['kind']=='phase1_predict':
            assert 'secret' not in json.dumps(j['config'])
            assert j['dependencies']==['source-feature_separation_hu2024-s'+str(j['seed'])]
        else:
            assert j['dependencies']==['phase1-predict-feature_separation_hu2024-s'+str(j['seed'])]
    rt['old_classes']=['different']
    with pytest.raises(ValueError):
        hu_repair.phase1_jobs(rt)


def test_previous_owner_requires_terminal_state_and_dead_workers(tmp_path, monkeypatch):
    state=tmp_path/'previous.json'
    state.write_text(json.dumps({'status':'RUNNING','controller_pid':9,'active':{'x':{'pid':10}}}))
    (tmp_path/'USER_STOP.json').write_text('{}')
    monkeypatch.setattr(hu_repair, 'alive', lambda pid: True)
    with pytest.raises(RuntimeError):
        hu_repair.wait_for_previous_owner({'wait_for_phase2_state':str(state)},tmp_path)
    assert json.loads((tmp_path/'pipeline_state.json').read_text())['status']=='WAITING_FOR_PHASE2_OWNER'
    state.write_text(json.dumps({'status':'PARTIAL_TECHNICAL_FAILURE','controller_pid':9,'active':{}}))
    monkeypatch.setattr(hu_repair, 'alive', lambda pid: False)
    hu_repair.wait_for_previous_owner({'wait_for_phase2_state':str(state)},tmp_path)
