import copy,json
from pathlib import Path
import pytest
from experiments.cvs_volterra_identity.source import validate_config
from experiments.cvs_volterra_identity.dispatch import select_source_candidate,validate_spec,CANDIDATES,SEEDS

ROOT=Path(__file__).resolve().parents[1]


def config():
    return json.loads((ROOT/'experiments/cvs_volterra_identity/configs/volterra_lag1-s2026092701.json').read_text(encoding='utf-8'))


def test_actual_preregistered_source_configs_and_exclusive_eight_rows():
    spec=json.loads((ROOT/'experiments/cvs_volterra_identity/configs/launch_spec.json').read_text(encoding='utf-8'))
    # Same remote source references resolved locally solely for no-data config checking.
    for row in spec['rows']:
        row['source_config']=str(ROOT/'experiments/cvs_volterra_identity/configs'/(row['row_id']+'.json'))
    validate_spec(spec)
    bad=copy.deepcopy(spec);bad['rows'][1]=copy.deepcopy(bad['rows'][0])
    with pytest.raises(ValueError):validate_spec(bad)
    bad=copy.deepcopy(spec);bad['target_truth']='forbidden'
    with pytest.raises(ValueError):validate_spec(bad)
    bad=copy.deepcopy(spec);bad['numerical_policy']['cudnn_allow_tf32']=True
    with pytest.raises(ValueError):validate_spec(bad)


@pytest.mark.parametrize('field,value',[('augmentation',True),('domain_backbone',True),('extra_losses',['contrastive']),
    ('checkpoint','legacy.pt'),('resume','legacy.pt'),('teacher','legacy.pt'),('target_inputs','query.npy'),('target_truth','truth.json'),
    ('epochs',50),('batch_size',64),('selection','best_target'),('split_seed',1)])
def test_source_refuses_inheritance_target_augmentation_and_budget_changes(field,value):
    c=config();c[field]=value
    with pytest.raises(ValueError):validate_config(c)


def test_source_refuses_missing_precision_or_changed_physical_contract():
    c=config();validate_config(c)
    for change in ('missing_precision','changed_precision','changed_window'):
        bad=copy.deepcopy(c)
        if change=='missing_precision':del bad['numerical_policy']
        elif change=='changed_precision':bad['numerical_policy']['cudnn_allow_tf32']=True
        else:bad['volterra']['estimator_window']=[0,80]
        with pytest.raises(ValueError):validate_config(bad)


def records():
    return [dict(variant=v,seed=s,accuracy=.95+(v==CANDIDATES[1])*.0001,worst_rx=.90,
                 parameters=9999999 if v==CANDIDATES[1] else 10,macs=9999999 if v==CANDIDATES[1] else 10)
            for v in CANDIDATES for s in sorted(SEEDS)]


def test_source_performance_first_requires_all_twelve_and_never_has_cost_window():
    r=records();chosen=select_source_candidate(r)
    assert chosen['selected_variant']==CANDIDATES[1] and chosen['new_candidate_selected']
    assert chosen['target_access'] is False and chosen['target_score_used'] is False
    assert len(chosen['candidate_universe'])==3
    with pytest.raises(ValueError):select_source_candidate(r[:-1])
    bad=copy.deepcopy(r);bad[-1]['seed']=bad[-2]['seed']
    with pytest.raises(ValueError):select_source_candidate(bad)


def test_exact_performance_ties_use_cost_then_fixed_candidate_order():
    r=records()
    for a in r:a['accuracy']=.95;a['parameters']=100;a['macs']=100
    assert select_source_candidate(r)['selected_variant']=='coupled_lag4'
    for a in r:
        if a['variant']==CANDIDATES[2]:a['macs']=99
    assert select_source_candidate(r)['selected_variant']==CANDIDATES[2]

def test_collector_uses_verified_python_and_immutable_release(monkeypatch,tmp_path):
    from experiments.cvs_volterra_identity.collect import collect
    from experiments.cvs_volterra_identity import publish,prepare
    import ast
    captured=[]
    def fake_ssh(script):
        captured.append(script);return b'{"ready":false,"status":"SOURCE_TRAINING"}'
    monkeypatch.setattr(publish,'ssh',fake_ssh)
    assert collect(tmp_path,tmp_path/'output') is False
    script=captured[0];ast.parse(script)
    assert '/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python' in script
    assert prepare.PROJECT+'/releases/'+prepare.RELEASE in script
    assert 'PYTHONPATH=' in script and 'subprocess.run(' in script
