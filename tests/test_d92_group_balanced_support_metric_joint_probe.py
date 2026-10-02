"""Focused new-method integration, all inputs literal synthetic; root runs tests."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'tools'),str(ROOT/'code'),str(ROOT/'tests')]
import numpy as np
import pytest
import torch
import evaluate_d92_group_balanced_support_metric_joint_probe as producer
from cvsrffi import d92_proto_frame_primitives as primitive
from cvsrffi.d92_ground_classifier_a import GroundClassifierA,GroundFeatureContract,GroundHeadMetadata

RESOURCES=dict(max_newton_iterations=100,max_line_search_trials=64,max_factor_buffer_bytes=167772160,
    max_integer_bits=16384,max_fraction_operations=1000000,max_secular_iterations=128)
PATH='R_GROUP_BALANCED_SUPPORT_METRIC_seq'


def synthetic(k=3,new=2):
    # Fixed before execution, same literal geometry distribution as the original
    # SupportMetric unit composition; no seed selection from outcomes.
    rng=np.random.default_rng(7301);classes=tuple('class-'+str(i) for i in range(6+new))
    labels=np.repeat(np.arange(len(classes)),k);raw={}
    for name,width in zip(producer.BRANCHES,(160,96,160,160,160)):
        center=rng.normal(size=(len(classes),width))
        raw[name]=center[labels]+.7*rng.normal(size=(len(labels),width))
    prototypes=np.zeros((6,160));prototypes[[0,2,4],0]=[1,-1,1];prototypes[[1,3,5],1]=[1,-1,1]
    frame=primitive.build_proto_frame_dictionary(prototypes=prototypes,classes=classes[:6])
    return dict(raw,support_labels=labels,support_ids=tuple(f'physical-{i}-{j}' for i in range(len(classes)) for j in range(k)),
        classes=classes,old_classes=classes[:6],prototype_frame=frame)


def native_ground(classes):
    contract=GroundFeatureContract('a'*64,'z_id','feat_joint','raw','float32',160)
    metadata=GroundHeadMetadata('a'*64,'id_backbone.cls_head.head.weight',tuple(reversed(classes)),16.,1e-4,
        contract,'MATCHED_SOURCE_ONLY_SCRATCH',False,(),'none')
    return GroundClassifierA(weight=torch.zeros((6,160),dtype=torch.float32),metadata=metadata)


def produce(root,k=3,new=2):
    data=synthetic(k,new);archive=producer.StateArchive(root);stages=[];fixed=[]
    def log(value):
        stages.append(producer.json_native(dict(producer.compact_event(value),schema=producer.SCHEMA,
            method=producer.METHOD,split_id='literal')))
    result=producer.probe_group_balanced_support_metric_joint(**data,**RESOURCES,
        ground_head=native_ground(data['old_classes']),context=dict(run_id='literal-run',row_id='literal-row',split_id='literal'),
        state_callback=archive,log_callback=log,prediction_callback=fixed.append)
    manifest=archive.finalize('COMPLETE')
    identity=dict(split_id='literal',receiver='receiver',scenario='scene',k=k,support_seed=7,new_count=new,
        registered_classes=list(data['classes']))
    parent=producer.json_native(dict(result,**{key:identity[key] for key in ('split_id','receiver','scenario','k','support_seed','new_count')},
        scope=producer.SCOPE,query_rows_used=0,source_rows_used=0))
    return dict(root=root,parent=parent,identity=identity,physical=list(data['support_ids']),stages=stages,fixed=fixed,manifest=manifest,
        ground=dict(status='MATCHED_SOURCE_ONLY_PACKET',ordered_classes=list(reversed(data['old_classes']))))


_PRODUCTION_PARENT=None


@pytest.fixture(scope='session')
def production_parent(tmp_path_factory):
    global _PRODUCTION_PARENT
    # Both owned test modules import this single producer fixture. Keep one
    # actual generation; corruption tests copy metadata or intercept readback.
    if _PRODUCTION_PARENT is None:
        _PRODUCTION_PARENT=produce(tmp_path_factory.mktemp('group-balanced-production'))
    return _PRODUCTION_PARENT


def test_real_nonzero_stage_actual_B_inheritance_all_paths_and_weight_work(production_parent):
    result=production_parent['parent'];paths=result['folds']+result['oneshot_proxy']['trials']
    assert result['schema']==producer.SCHEMA and result['method']=='D92-GroupBalancedSupportMetric-GGN1-LocalRidge'
    assert result['sequence_paths']==6 and result['candidate_stage_count']==12
    assert result['row_basis_construction_count']==result['actual_work']['basis_calls']==1
    assert result['trial_count']>0 and result['optimizer_steps']>0
    assert any(s['actual_updated_coordinate_count']>0 for p in paths for s in p['candidate_stages'])
    for path in paths:
        b,c=path['candidate_stages'];bp,cp=path['preparations']
        assert c['final_prior_ref']==cp['full_prior_ref']==b['final_state_ref']
        assert cp['inherited_adapter_from']=='B_GROUP_BALANCED_SUPPORT_METRIC'
        assert set(path['b_training_ids'])<=set(path['c_training_ids'])
        assert set(path['b_ids'])<=set(path['c_ids'])
        assert b['actual_work']['gate_forward_calls']==0
        work=c['actual_work'];assert work['gate_forward_weight_construction_attempts']==work['gate_forward_calls']
        assert work['gate_forward_weight_constructions_completed']==work['gate_forward_calls']
        assert work['gate_jvp_weight_construction_attempts']==0
        assert work['gate_forward_weighted_logistic_record_evaluations']>0
        assert c['structural_predict_workload'] is None
    assert result['preparation_actual_work']['basis_calls']==result['stage_actual_work']['basis_calls']==0
    assert production_parent['manifest']['schema']=='d92_group_balanced_support_metric_joint_state_archive_v1'


@pytest.mark.parametrize('new',[0,2])
def test_K1_complete_heads_no_held_and_new0_exact_reuse(tmp_path,new):
    value=produce(tmp_path,new=new,k=1);p=value['parent'];full=p['full_support']
    assert p['oof'] is p['oneshot_proxy'] is None and p['optimizer_steps']==0 and value['fixed']==[]
    assert len(full['candidate_stages'])==(1 if new==0 else 2)
    assert full['c_reuses_b_candidates']==(new==0)
    assert p['actual_work']['gate_forward_calls']==(0 if new==0 else 1)
    assert full['held_labels']=={} and full['b_ids']==full['c_ids']==[]


def test_new_identity_and_full_actual_SUM_MAX_literals_match_analyzer():
    import analyze_d92_group_balanced_support_metric_joint_probe as analysis
    assert analysis.ALGORITHM==producer.FROZEN_CONFIG
    assert set(analysis.WORK_SUM)==set(producer.WORK_SUM_KEYS) and len(analysis.WORK_SUM)==len(producer.WORK_SUM_KEYS)
    assert set(analysis.WORK_MAX)==set(producer.WORK_MAX_KEYS) and len(analysis.WORK_MAX)==len(producer.WORK_MAX_KEYS)
    assert set(analysis.COUNTERS)==set(producer.COUNTERS)
    assert analysis.PEAKS==producer.PEAK_COUNTERS
    assert analysis.SCHEMA==producer.SCHEMA and analysis.PATHS==producer.PATHS
    gate=dict(objective=3.,logistic_loss_sum=2.,ridge_penalty=.5,barrier_term=.5,old_supervision_mass=4.,
        new_supervision_mass=4.,weight_constructions_completed=1,weight_construction_seconds=.01,
        canonical_residual=1e-13,intercept_residual=1e-13,wall_seconds=.02)
    record=dict(event='FINAL',audit=dict(operation_audits=[dict(operation='gate_forward',audit=gate)]))
    assert producer.compact_event(record)['audit']['operation_audits'][0]['audit']==gate
    assert analysis.compact_event(record)==producer.compact_event(record)
    text=producer.training_text(record)
    for part in ('weighted_gate_objective=3.0','weighted_gate_logistic=2.0','gate_barrier=0.5',
            'gate_old_mass=4.0','gate_new_mass=4.0','gate_weight_constructions=1','source_validation=N/A'):
        assert part in text
    measured=producer._small_measurements(dict(theta=np.zeros(2),new_alpha=np.zeros((4,2)),
        new_intercept=np.zeros(2),gate_alpha=np.zeros(16),gate_b=np.asarray(0.),actual_B_alpha=np.zeros((12,6))))
    assert measured['current_stage_fitted_head_parameters']==8+2+16+1
