"""Read-only closure and truth-last scoring after the known scalar verifier failure.

The original supervisor remains FAILED. This entry never modifies its files,
launches a predictor, imports a fitting module, or treats partial outputs as fixed.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import re
import sys
import time

import score_d92_support_metric_joint_benchmark_scalar_r02 as fixed_score

SCHEMA='d92_support_metric_fixed_outputs_scalar_failure_closure_v1'
STATUS='INDEPENDENT_FIXED_OUTPUT_CLOSURE_COMPLETE'
SCOPE='READ_ONLY_INDEPENDENT_STRUCTURAL_INTERFACE_REPAIR_NOT_A_NEW_METHOD'
KNOWN_ERROR='Trial float tolerance differs'
GLOBAL_FAILED='SUPPORT_METRIC_QUERY_BENCHMARK_PREDICTIONS_FAILED'


def close_fixed_outputs(*,spec,run_root=None):
    """Validate and independently reread all four rows before any truth access."""
    check=fixed_score.require
    check(spec['schema']=='d92_support_metric_joint_query_benchmark_v1','Wrong original benchmark schema')
    fixed_score.validate_config(spec['benchmark']['config'])
    rows,cohorts=fixed_score.validate_declared_matrix(spec)
    root=Path(spec['execution']['remote_run_root']) if run_root is None else Path(run_root)
    startup=fixed_score.read(root/'startup.json');terminal=fixed_score.read(root/'complete.json')
    states=fixed_score.read(root/'state.json')
    check(startup['schema']==terminal['schema']==spec['schema']
        and startup['status']=='SUPPORT_METRIC_QUERY_BENCHMARK_STARTED'
        and startup['resolved_spec']==terminal['resolved_spec']==spec
        and startup['run_id']==terminal['run_id']==spec['run_id']
        and startup['group_id']==terminal['group_id']==spec['group_id'],'Original terminal run/spec identity differs')
    check(terminal['status']==GLOBAL_FAILED and terminal['all_predictions_fixed'] is False
        and terminal['row_count']==4 and terminal['declared_episode_count']==2400
        and terminal['truth_read'] is False and terminal['scorer_invoked'] is False
        and terminal['automatic_retry'] is False,'Expected preserved terminal verifier failure')
    row_ids={r['row_id'] for r in rows}
    check(set(startup['rows'])==set(terminal['rows'])==set(states)==row_ids and states==terminal['rows'],
        'Original terminal row/state closure differs')
    runtime=terminal['runtime_commit']
    check(runtime==startup['runtime_commit'] and re.fullmatch('[0-9a-f]{40}',runtime) is not None
        and terminal['code_commit']==startup['code_commit']==spec['code']['commit'],'Original actual runtime binding differs')
    completed=[r for r in rows if states[r['row_id']]['status']=='COMPLETE']
    failed=[r for r in rows if states[r['row_id']]['status']=='FAILED']
    check(bool(failed) and len(completed)+len(failed)==4,'Original supervisor has unfinished or unknown rows')
    check(terminal['completed_row_count']==len(completed) and terminal['completed_episode_count']==sum(
        cohorts[r['cohort']]['expected_split_count'] for r in completed),'Original failed terminal counts were changed')
    fixed=[];cohort_physical={};evidence=[]
    for row in rows:
        co=cohorts[row['cohort']];lane=states[row['row_id']]
        check(type(lane.get('process_returncode')) is int and lane['process_returncode']==0
            and type(lane.get('pid')) is int and lane['pid']>0,'Every producer must have observed exit zero and PID')
        if lane['status']=='FAILED':
            check(lane.get('phase')=='PREDICTION' and lane.get('error_type')=='ValueError'
                and lane.get('error')==KNOWN_ERROR,'Different failure cannot use scalar interface closure')
            check('marker' not in lane,'Failed verifier lane unexpectedly claims a completed marker')
        path=fixed_score._mapped(str(Path(row['output_root'])/'predictions'),spec,run_root)
        # This is the same implementation used by the public zero-truth
        # validate_row_output API, returning the fixed streams as well as marker.
        value=fixed_score._validated_row_fixed(path,spec,row,runtime,lane['preflight'])
        marker=value['complete']
        check(marker['pid']==lane['pid'],'Producer marker PID differs from observed process')
        if lane['status']=='COMPLETE':check(lane['marker']==marker,'Original COMPLETE lane marker changed')
        else:check(Path(lane['preserved_prediction_output'])==Path(row['output_root'])/'predictions',
            'Failed lane preserved output binding differs')
        check(lane['release_commit']==runtime and lane['expected_model_seed']==row['expected_model_seed']
            and lane['expected_checkpoint_sha256']==row['expected_checkpoint_sha256']
            and lane['expected_capsule_id']==co['expected_capsule_id'] and lane['output_root']==row['output_root']
            and lane['prediction_output']==str(Path(row['output_root'])/'predictions'), 'Original lane source/output binding differs')
        check(lane['source_paths']==dict(row_root=row['row_root'],branch_features=row['branch_features'],
            ground_packet=row['ground_packet'],ground_summary=row['ground_summary'],capsule=co['capsule']),
            'Original lane source paths changed')
        check(cohort_physical.setdefault(row['cohort'],value['splits'])==value['splits'],
            'Same cohort model rows changed physical support/query metadata')
        fixed.append(value)
        evidence.append(dict(row_id=row['row_id'],original_status=lane['status'],
            original_phase=lane.get('phase'),original_error_type=lane.get('error_type'),original_error=lane.get('error'),
            producer_pid=lane['pid'],producer_returncode=lane['process_returncode'],prediction_status=marker['status'],
            predictions_complete_path=str(path/'predictions_complete.json'),split_count=len(value['splits']),
            binding=value['binding'],row_basis_rank=marker['row_basis_rank'],state_manifest=marker['state_manifest'],
            row_basis_construction_count=marker['row_basis_construction_count']))
    check(len(fixed)==4 and sum(len(v['splits']) for v in fixed)==2400,'All four rows and 2400 parents required before truth')
    for row,old in zip(rows,fixed):
        co=cohorts[row['cohort']]
        reread=fixed_score.load_fixed_predictions(predictions=old['predictions_root'],capsule=co['capsule'],
            config=spec['benchmark']['config'],expected_binding=old['binding'])
        check(reread==old,'Fixed predictions changed during independent second readback')
    check(fixed_score.read(root/'startup.json')==startup and fixed_score.read(root/'complete.json')==terminal
        and fixed_score.read(root/'state.json')==states,'Original supervisor evidence changed before truth')
    closure=dict(schema=SCHEMA,status=STATUS,scope=SCOPE,run_id=spec['run_id'],group_id=spec['group_id'],
        method=fixed_score.METHOD,release_commit=runtime,code_commit=spec['code']['commit'],row_count=4,parent_count=2400,
        original_supervisor_status=terminal['status'],original_completed_row_count=terminal['completed_row_count'],
        original_completed_episode_count=terminal['completed_episode_count'],original_run_root=str(root),
        original_supervisor_files=['startup.json','state.json','complete.json'],rows=evidence,
        all_declared_fixed_outputs_independently_verified=True,second_readback_complete=True,
        original_status_unchanged=True,truth_read=False,training_performed=False,model_called=False,
        automatic_retry=False,new_data_validation=False,authority_or_approval_artifact=False)
    return closure,fixed,startup,terminal


def score_fixed_outputs(*,spec,output,run_root=None):
    """Create a fresh derived directory; never overwrite the original run."""
    out=Path(output);root=Path(spec['execution']['remote_run_root']) if run_root is None else Path(run_root)
    fixed_score.require(not out.exists(),'Exclusive independent closure output required')
    for original in (root,Path(spec['execution']['remote_run_root'])):
        fixed_score.require(not out.resolve().is_relative_to(original.resolve()),'Derived output must be outside original run')
    out.mkdir(parents=True,exist_ok=False)
    try:
        closure,fixed,startup,terminal=close_fixed_outputs(spec=spec,run_root=run_root)
        fixed_score.write(out/'closure.json',closure)
        fixed_score.require(fixed_score.read(out/'closure.json')==closure,'Independent closure artifact readback failed')
        rows=spec['rows'];cohorts=spec['benchmark']['cohorts'];runtime=closure['release_commit']
        resources=fixed_score.aggregate_training_resources([dict(binding=v['binding'],resources=v['complete']['resources']) for v in fixed])
        parents=[];truths={};old_sets={};started=time.perf_counter()
        for row,value in zip(rows,fixed):
            truth_path=str(cohorts[row['cohort']]['truth'])
            if truth_path not in truths:truths[truth_path]=fixed_score.read(truth_path)
            for sid in value['splits']:
                parent=fixed_score._truth_parent(value,sid,truths[truth_path]);parent['cohort']=row['cohort']
                key=(row['row_id'],parent['receiver'],parent['scenario'],parent['k'],parent['support_seed'])
                paired=(sorted(parent['old_query_ids']),value['splits'][sid]['old_support_ids'])
                fixed_score.require(old_sets.setdefault(key,paired)==paired,'Old physical queries/support changed across new counts')
                parents.append(parent)
        resources['scoring_truth_join_seconds']=time.perf_counter()-started
        result=dict(schema=fixed_score.SCHEMA,method=fixed_score.METHOD,status=fixed_score.STATUS,scope=fixed_score.SCOPE,
            repair_scope=SCOPE,run_id=spec['run_id'],group_id=spec['group_id'],release_commit=runtime,
            code_commit=spec['code']['commit'],algorithm=spec['benchmark']['config']['algorithm'],
            support_metric_resources=spec['benchmark']['config']['support_metric_resources'],row_count=4,parent_count=len(parents),
            parents=parents,statistics=fixed_score._statistics(parents),resources=resources,
            independent_closure=closure,original_supervisor_status=terminal['status'],original_status_unchanged=True,
            current_metadata=dict(spec=deepcopy(spec),supervisor_startup=startup,supervisor_complete=terminal,
                rows=[dict(binding=v['binding'],source_identity=v['complete']['source_identity'],ground_packet_identity=v['complete']['ground_packet_identity'],
                    ground_geometry_identity=v['complete']['ground_geometry_identity'],ground_geometry_binding=v['complete']['ground_geometry_binding'],
                    row_basis_rank=v['complete']['row_basis_rank'],row_basis_ref=v['complete']['row_basis_ref'],
                    row_basis_actual_work=v['complete']['row_basis_actual_work'],parameter_evidence=v['parameter_evidence'],
                    streams=v['complete']['streams'],split_count=len(v['splits']),resources=v['complete']['resources'],
                    predictor_device=v['startup'].get('device')) for v in fixed]),
            prediction_validation_complete_before_truth=True,prediction_streams_independently_reread=True,
            query_fit_access=False,source_fit_access=False,training_performed=False,model_called=False,
            selection_feedback_forbidden=True,automatic_promotion=False,
            data_reuse_statement='Repeated frozen benchmark data; not a fresh independent confirmation',
            score_units='FRACTIONS; SUBTRACTIONS_ARE_FRACTION_DIFFERENCES',
            metric_definitions=dict(total_old_accuracy_drop='B_old_accuracy-C_old_accuracy',
                adaptation_gain_B_minus_A='B_old_accuracy-A_old_accuracy',
                C_h='HARMONIC_MEAN_COMPUTED_WITHIN_EACH_PHYSICAL_PARENT_BEFORE_AGGREGATION',
                R0='INDEPENDENT_FULL_SUPPORT_R0_B_AND_R0_C_ON_IDENTICAL_PHYSICAL_QUERY_IDS; NOT_NATIVE_GROUND_A',
                C_abs_new_old_gap='ABS_C_OLD_MINUS_C_NEW_WITHIN_PARENT',
                K1='QUERY_BENCHMARK_IS_VALID; NOT_SUPPORT_HELD_DIAGNOSTICS',
                new0='C_EXACT_ACTUAL_B; C_NEW_ACCURACY/H/GAP/NEW_MACRO_F1_NULL'))
        fixed_score.write(out/'summary.json',result)
        fixed_score.require(fixed_score.read(out/'summary.json')==result,'Independent score artifact readback failed')
        return result
    except Exception as exc:
        try:
            fixed_score.write(out/'failure.json',dict(schema=SCHEMA,status='TECHNICAL_FAILURE',scope=SCOPE,
                error_type=type(exc).__name__,error=str(exc),original_status_unchanged=True,automatic_retry=False))
        except Exception as secondary:
            print('Failure evidence write failed: '+repr(secondary)+'; primary: '+repr(exc),file=sys.stderr)
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True,help='Fresh directory outside the original run')
    parser.add_argument('--run-root',type=Path)
    args=parser.parse_args()
    result=score_fixed_outputs(spec=fixed_score.read(args.spec),output=args.output,run_root=args.run_root)
    print(json.dumps(dict(status=result['status'],row_count=result['row_count'],parent_count=result['parent_count'])))


if __name__=='__main__':main()
