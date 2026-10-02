"""Bounded independent weighted-gate checks; no registered inputs or results."""
import ast
from copy import deepcopy
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'tools'),str(ROOT/'code'),str(ROOT/'tests')]
import numpy as np
import pytest
import analyze_d92_group_balanced_support_metric_joint_probe as analysis
from test_d92_group_balanced_support_metric_joint_probe import production_parent,RESOURCES,produce


def verify_production(value):
    archives=analysis.Archives(value['root'],'literal-run','literal-row')
    stages=iter(value['stages']);predictions=iter(value['fixed'])
    result=analysis.verify_parent(value['parent'],value['identity'],value['physical'],row_id='literal-row',run_id='literal-run',
        archives=archives,stage_stream=stages,prediction_stream=predictions,resources=RESOURCES,ground_binding=value['ground'])
    archives.close();assert next(stages,None) is None and next(predictions,None) is None
    return result


def test_real_updated_producer_full_numeric_closure_before_metrics(production_parent):
    checked=verify_production(production_parent)
    assert checked['counters']['trial_count']>0 and checked['counters']['optimizer_steps']>0
    assert checked['counters']['sequence_paths']==6
    assert checked['counters']['gate_forward_weight_constructions_completed']>0
    assert checked['counters']['gate_forward_weighted_entropy_record_evaluations']>0
    for name in analysis.PATHS:
        scored=analysis.score_fixed([p for p in checked['paths'] if p['path']['scope']=='support_oof'],name,
            production_parent['parent']['old_classes'])
        reported=production_parent['parent']['oof']['paths'][name]['metrics']
        for key in set(scored)&set(reported):
            if reported[key] is None:assert scored[key] is None
            else:assert scored[key]==pytest.approx(reported[key])


@pytest.mark.parametrize('fault',['weights','class_counts','labels','qeff','objective','scope','actual_B'])
def test_weight_or_physical_identity_corruption_rejected_before_scoring(production_parent,monkeypatch,fault):
    value=deepcopy(production_parent)
    if fault=='objective':
        stage=value['parent']['folds'][0]['candidate_stages'][1]
        next(op['audit'] for op in stage['operation_audits'] if op['operation']=='gate_forward')['objective']+=1.
    elif fault=='scope':value['parent']['folds'][0]['scope']='query'
    elif fault=='actual_B':value['parent']['folds'][0]['candidate_stages'][1]['final_prior_ref']={}
    else:
        original=analysis.Archives.load
        def corrupted(self,*args,**kwargs):
            result=original(self,*args,**kwargs)
            if 'gate_weights' in result:
                result={k:v.copy() for k,v in result.items()}
                if fault=='weights':result['gate_weights']*=1.01
                elif fault=='class_counts':result['gate_class_counts'][0]+=1
                elif fault=='labels':result['gate_labels'][0]=result['gate_labels'][-1]
                else:result['gate_qeff'][0]+=.125
            return result
        monkeypatch.setattr(analysis.Archives,'load',corrupted)
    calls=[]
    monkeypatch.setattr(analysis,'score_fixed',lambda *a,**k:calls.append('truth'))
    with pytest.raises((ValueError,KeyError)):verify_production(value)
    assert calls==[]


def literal_spec():
    import itertools
    old=['class-'+str(i) for i in range(6)];cohorts={}
    for co in ('rx3','rx1'):
        pairs=[[co+'-receiver','scene-a'],[co+'-receiver','scene-b']]
        splits=[dict(split_id=f'{co}-{scene}-{k}-{q}',receiver=rx,scenario=scene,k=k,new_count=q,support_seed=7,
            registered_classes=old+['new-'+str(j) for j in range(q)])
            for (rx,scene),k,q in itertools.product(pairs,[1,5,10,20],[0,2,5,10,20])]
        cohorts[co]=dict(capsule_id='synthetic-'+co,matrix={},selection=dict(receiver_scenes=pairs,ks=[1,5,10,20],
            new_counts=[0,2,5,10,20],support_seed=7,splits=splits))
    rows=[dict(row_id=f'{co}-{seed}',cohort=co,seeds=dict(model=seed),expected_checkpoint_sha256=('a' if seed==2026092701 else 'c')*64,
        ground_packet=None,ground_summary='/synthetic/ground',ground_summary_already_deployed=True,support_features='/synthetic/support')
        for co,seed in itertools.product(cohorts,[2026092701,2026092702])]
    return dict(schema=analysis.SCHEMA,group_id='d92-group-balanced-support-metric-joint-support',run_id='literal-only',
        rows=rows,code=dict(commit='b'*40),execution=dict(remote_run_root='/synthetic/unopened'),
        probe=dict(algorithm=deepcopy(analysis.ALGORITHM),support_metric_resources=RESOURCES.copy(),cohorts=cohorts))


@pytest.mark.parametrize('fault',['missing_row','missing_parent','old_method','old_weights'])
def test_full_four_row_160_parent_declaration_not_partial_or_old_method(fault):
    spec=literal_spec();analysis.validate_spec(spec)
    if fault=='missing_row':spec['rows'].pop()
    elif fault=='missing_parent':spec['probe']['cohorts']['rx1']['selection']['splits'].pop()
    elif fault=='old_method':spec['probe']['algorithm']['method']='D92-ProtoFrameSupportMetric-GGN1-LocalRidge'
    else:spec['probe']['algorithm'].pop('gate_weight_rule')
    with pytest.raises(ValueError):analysis.validate_spec(spec)


def test_analyzer_has_only_stdlib_numpy_imports_and_native_scalar_tolerance_boundary():
    source=(ROOT/'tools/analyze_d92_group_balanced_support_metric_joint_probe.py').read_text(encoding='utf-8')
    tree=ast.parse(source)
    imported={node.module.split('.')[0] for node in ast.walk(tree) if isinstance(node,ast.ImportFrom)}
    imported|={alias.name.split('.')[0] for node in ast.walk(tree) if isinstance(node,ast.Import) for alias in node.names}
    assert imported<=set(('argparse','collections','csv','itertools','json','math','pathlib','time','numpy'))
    assert "float(trial['comparison_tolerance']/tolerance)" in source
    assert 'no independent kernel/JVP replay or complete-head interval certificate' in source


@pytest.mark.parametrize('new',[0,2])
def test_K1_numeric_complete_head_has_no_scored_held_observation(tmp_path,new):
    value=produce(tmp_path,k=1,new=new);result=verify_production(value)
    assert result['counters']['optimizer_steps']==0 and result['counters']['sequence_paths']==1
    assert value['parent']['oof'] is value['parent']['oneshot_proxy'] is None
    assert value['fixed']==[]
