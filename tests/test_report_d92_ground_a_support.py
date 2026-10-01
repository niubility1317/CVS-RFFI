"""Literal completed-pairing fixtures; no experiment artifacts are read."""
from copy import deepcopy
from pathlib import Path
import sys

import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import report_d92_ground_a_support as report

def fixture():
    spec=dict(run_id='synthetic-supplement',rows=[]);pairings={}
    for method,path in report.METHODS.items():
        for index in range(4):
            row_id=method+str(index)
            row=dict(row_id=row_id,source_run_id='source-'+method,source_row_id='row-'+str(index),
                expected_checkpoint_sha256=str(index)*64,expected_capsule_id='capsule',expected_model_seed=100+index,
                expected_method=method)
            spec['rows'].append(row)
            binding=dict(run_id=row['source_run_id'],row_id=row['source_row_id'],
                checkpoint_sha256=row['expected_checkpoint_sha256'],capsule_id='capsule',model_seed=100+index)
            stats=dict(overall=[],by_k_new_count=[],by_receiver_scene=[],by_model_row=[])
            def values(count,*,k=5,new=2,**identity):
                for metric in report.METRICS:
                    missing=k==1 or new==0 and metric in ('C_new_accuracy','C_h','C_abs_new_old_gap')
                    # H/gap deliberately differ from values computed after averaging.
                    value={'A_old_accuracy':.5+index*.01,'B_old_accuracy':.7,'C_old_accuracy':.65,
                        'C_new_accuracy':.6,'C_h':.61,'adaptation_gain_B_minus_A':.2-index*.01,
                        'total_old_accuracy_drop':.05,'C_abs_new_old_gap':.12}[metric]
                    yield dict(identity,path=path,metric=metric,parent_count=count,
                        measured_parent_count=0 if missing else count,mean=None if missing else value)
            for diagnostic in ('oof','proxy'):
                stats['overall']+=list(values(24,diagnostic=diagnostic,population='new_present'))
                stats['overall']+=list(values(6,new=0,diagnostic=diagnostic,population='old_only'))
                for k in report.KS:
                    for new in report.NEWS:
                        rows=list(values(2,k=k,new=new,diagnostic=diagnostic))
                        for r in rows:r.update(k=k,new_count=new)
                        stats['by_k_new_count']+=rows
                        stats['by_receiver_scene'] += [dict(r,receiver='rx'+str(index),scenario='scene') for r in rows]
                        stats['by_model_row'] += [dict(r,model_seed=100+index,row_id=row['source_row_id']) for r in rows]
            pairings[row_id]=dict(status='GROUND_A_SUPPORT_PAIRING_COMPLETE',scope=report.SCOPE,
                schema='d92_ground_a_support_pairing_v1',binding=binding,method=method,adapted_path=path,
                query_rows_used=0,source_rows_used=0,original_summary_modified=False,original_trace_modified=False,
                training_performed=False,calibration_performed=False,model_called_during_truth_join=False,
                parent_count=40,k1_parent_count=10,ordered_ground_classes=list('abcdef'),statistics=stats,
                resources=dict(wall_seconds=1.3,incremental_transmission_bytes=None))
    done=dict(schema='d92_ground_a_support_supervisor_v1',status='GROUND_A_SUPPORT_SUPERVISOR_COMPLETE',
        run_id=spec['run_id'],rows=8,completed_rows=8,parent_count=320,commit='a'*40,
        query_rows_used=0,source_rows_used=0,training_performed=False,original_inputs_modified=False,
        row_outputs={r['row_id']:'/synthetic/'+r['row_id'] for r in spec['rows']},
        wall_seconds=15.4,peak_process_rss_bytes=5000000,peak_gpu_memory_bytes=None,
        incremental_network_transfer_bytes=None,energy=None)
    return spec,done,pairings

def test_complete_three_stage_matrix_preserves_parent_first_metrics():
    spec,done,rows=fixture();text,value=report.assemble(spec,done,rows,source='synthetic')
    assert len(value['tables']['by_k_new_count'])==80
    assert len(value['tables']['by_model_row'])==320
    assert len(value['overall'])==2
    for row in value['overall']:
        assert row['means']['A_old_accuracy']==pytest.approx(.515)
        assert row['means']['adaptation_gain_B_minus_A']==pytest.approx(.185)
        assert row['means']['C_h']==.61 and row['means']['C_abs_new_old_gap']==.12
        assert row['measured_parents']==96
    assert '旧类数' in text and '总注册类数' in text and 'N/A' in text and 'query' in text
    assert value['automatic_promotion'] is False and value['goal_complete'] is False
    assert value['supervisor_resources']['incremental_network_transfer_bytes'] is None
    for row in value['tables']['by_k_new_count']:
        if row['k']==1:assert all(v is None for v in row['means'].values())

@pytest.mark.parametrize('case',['binding','method','query','training','partial','missing_row','duplicate_metric',
    'missing_metric','K1','new0','wrong_count','extra_matrix','head_order','modified_trace','wrong_supervisor'])
def test_reject_misrepresented_or_incomplete_pairing(case):
    spec,done,rows=fixture();p=next(iter(rows.values()))
    if case=='binding':p['binding']['model_seed']+=1
    elif case=='method':p['adapted_path']='R0'
    elif case=='query':p['query_rows_used']=1
    elif case=='training':p['training_performed']=True
    elif case=='partial':done['completed_rows']=7
    elif case=='missing_row':rows.pop(next(iter(rows)))
    elif case=='duplicate_metric':p['statistics']['overall'].append(deepcopy(p['statistics']['overall'][0]))
    elif case=='missing_metric':p['statistics']['by_k_new_count'].pop()
    elif case=='K1':
        r=next(r for r in p['statistics']['by_k_new_count'] if r['k']==1)
        r.update(mean=.8,measured_parent_count=2)
    elif case=='new0':
        r=next(r for r in p['statistics']['by_k_new_count'] if r['k']==5 and r['new_count']==0 and r['metric']=='C_new_accuracy')
        r.update(mean=.8,measured_parent_count=2)
    elif case=='wrong_count':p['statistics']['by_k_new_count'][0]['parent_count']=3
    elif case=='extra_matrix':p['statistics']['by_k_new_count'][0]['k']=2
    elif case=='head_order':p['ordered_ground_classes'][0]='b'
    elif case=='modified_trace':p['original_trace_modified']=True
    else:done['status']='RUNNING'
    with pytest.raises(ValueError):report.assemble(spec,done,rows,source='synthetic')
