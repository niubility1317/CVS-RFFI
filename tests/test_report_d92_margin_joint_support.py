"""Synthetic verified-summary fixtures; no existing result files are read."""
from copy import deepcopy
from pathlib import Path
import sys

import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import report_d92_margin_joint_support as report


def fixture(per_cell=3):
    stats=dict(overall=[],by_k_new_count=[],by_receiver_scene=[],by_model_cohort=[])
    def metric_rows(count,k=5,new=2,**fields):
        for metric in report.METRICS:
            missing=k==1 or metric in ('A_old_accuracy','adaptation_gain_B_minus_A') or new==0 and metric in ('C_new_accuracy','C_h','C_abs_new_old_gap')
            # Deliberately do not equal H(mean accuracies) or abs(mean new-old).
            value={'B_old_accuracy':.72,'C_old_accuracy':.7,'C_new_accuracy':.6,'C_h':.61,
                'C_abs_new_old_gap':.18,'total_old_accuracy_drop':.02}.get(metric,.03)
            if fields['path']=='R_MARGIN_seq' and metric=='C_h':value+=.012
            yield dict(fields,metric=metric,mean=None if missing else value,parent_count=count,
                measured_parent_count=0 if missing else count)
    for path in report.PATHS:
        stats['overall']+=list(metric_rows(12*per_cell,diagnostic='oof',population='new_present',path=path))
    for diagnostic in ('oof','proxy'):
        for path in report.PATHS:
            for k in (1,5,10,20):
                for new in (0,2,5,10,20):
                    rows=list(metric_rows(per_cell,k,new,diagnostic=diagnostic,path=path))
                    for row in rows:row.update(k=k,new_count=new)
                    stats['by_k_new_count']+=rows
                    stats['by_receiver_scene'] += [dict(r,cohort='synthetic',receiver='rx',scenario='practical_high') for r in rows]
                    stats['by_model_cohort'] += [dict(r,model_seed=7,cohort='synthetic') for r in rows]
    summary=dict(status='COMPLETE_MARGIN_JOINT_PROBE_VERIFIED',summary_schema='d92_margin_joint_support_summary_v1',
        schema='d92_margin_joint_local_ridge_v1',algorithm=dict(schema='d92_margin_joint_local_ridge_v1',proximal_coefficient=0.,coordinate_ball_radius=.5),
        method='D92-MarginJointLocalRidge-v1',scope='SUPPORT_ONLY_MARGIN_JOINT_OOF_AND_PROXY_NOT_QUERY_EVALUATION',
        run_id='synthetic',release_commit='a'*40,actual_A=None,adaptation_gain_B_minus_A=None,
        query_rows_used=0,source_rows_used=0,old_class_count=6,
        qp_resources=dict(max_transitions=32,max_factor_buffer_bytes=1000000),
        coverage=dict(episodes=20*per_cell,k1_episodes=5*per_cell,oof_episodes=15*per_cell,sequence_paths=225*per_cell,
            margin_qp_peak_factor_buffer_bytes=10000,margin_qp_peak_explicit_solve_temporary_bytes=500),
        statistics=stats,resources=dict(wall_seconds=12.3,numeric_work_proxy=500,deployment_package_bytes=None))
    return summary,dict(status='VERIFIED',runtime_commit='a'*40,commit='b'*40)


@pytest.mark.parametrize('per_cell',[1,3,8])
def test_dynamic_complete_parent_tables_and_parent_first_metrics(per_cell):
    s,e=fixture(per_cell);text,value=report.assemble(s,e,source='synthetic-summary')
    assert len(value['matrix'])==80 and all(len(rows)==80 for rows in value['strata'].values())
    assert value['overall']['R0']['C_h']==.61 and value['overall']['R0']['C_abs_new_old_gap']==.18
    assert value['delta_vs_R0']['C_h']==pytest.approx(.012)
    assert value['automatic_promotion'] is False and value['actual_A'] is None
    assert all(all(v is None for v in row['means'].values()) for row in value['matrix'] if row['k']==1)
    assert 'N/A' in text and 'soft' in text and value['resources']['deployment_package_bytes'] is None
    assert 'g_Z' in text and '+Z' in text and 'n²' in text
    assert 'KKT' in text and 'MAX' in text and '自由截距' in text
    assert value['qp_resources']==s['qp_resources']


@pytest.mark.parametrize('corruption',['method','binding','query','A','K1','missing_cell','duplicate','missing_stratum','counts'])
def test_reject_malformed_or_misrepresented_results(corruption):
    s,e=fixture()
    if corruption=='method':s['schema']='wrong'
    elif corruption=='binding':e['runtime_commit']='c'*40
    elif corruption=='query':s['query_rows_used']=1
    elif corruption=='A':s['actual_A']=.9
    elif corruption=='K1':
        row=next(r for r in s['statistics']['by_k_new_count'] if r['k']==1 and r['metric']=='B_old_accuracy')
        row.update(mean=.9,measured_parent_count=3)
    elif corruption=='missing_cell':s['statistics']['by_k_new_count'].pop()
    elif corruption=='duplicate':s['statistics']['overall'].append(deepcopy(s['statistics']['overall'][0]))
    elif corruption=='missing_stratum':s['statistics']['by_receiver_scene'].pop()
    elif corruption=='counts':s['coverage']['sequence_paths']-=1
    with pytest.raises(ValueError):report.assemble(s,e,source='synthetic')


@pytest.mark.parametrize('corruption',['missing','extra','bool','zero','peak_exceeds','missing_peak','bool_peak'])
def test_resource_contract_and_actual_peak_scope(corruption):
    s,e=fixture()
    if corruption=='missing':s['qp_resources'].pop('max_transitions')
    elif corruption=='extra':s['qp_resources']['accuracy_budget']=1
    elif corruption=='bool':s['qp_resources']['max_transitions']=True
    elif corruption=='zero':s['qp_resources']['max_factor_buffer_bytes']=0
    elif corruption=='peak_exceeds':s['coverage']['margin_qp_peak_factor_buffer_bytes']=1000001
    elif corruption=='missing_peak':s['coverage'].pop('margin_qp_peak_explicit_solve_temporary_bytes')
    else:s['coverage']['margin_qp_peak_factor_buffer_bytes']=True
    with pytest.raises(ValueError):report.assemble(s,e,source='synthetic')
