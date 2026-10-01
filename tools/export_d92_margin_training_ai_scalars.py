"""Export all completed Margin diagnostic records as bounded primitive scalars.

No model, numeric archive, source sample, query, or original run is opened.
Only the supplied diagnostic summary and three JSONL streams are read.
"""
import argparse
import csv
import itertools
import json
import math
from pathlib import Path
import re
import time

INPUT_STATUS='COMPLETE_MARGIN_JOINT_TRAINING_DIAGNOSTICS_DERIVED'
STATUS='COMPLETE_MARGIN_TRAINING_AI_SCALARS_EXPORTED'
SCHEMA='d92_margin_training_ai_scalars_v1'
METHOD_SCHEMA='d92_margin_joint_local_ridge_v1'
METHOD='D92-MarginJointLocalRidge-v1'
SCOPE='COMPLETE_TRAINING_ONLY_NO_OUTER_QUERY_TRUTH'
STREAMS=('stages','curves','preparations')
COORDS=('run_id','row_id','split_id','scope','fold','outer_trial','parent_k','train_k','state')
STATES={'B_MARGIN','C_MARGIN_seq'}
PREP_STATES={'B_prepare':'B_MARGIN','C_prepare':'C_MARGIN_seq'}
EVENTS={'MARGIN_JOINT_'+s for s in ('PREPARED','INITIAL','GRADIENT','TRIAL','STEP','FINAL')}
PEAKS=('margin_qp_peak_factor_buffer_bytes','margin_qp_peak_explicit_solve_temporary_bytes')
QP_WORK=('transitions','full_constraint_scans','spectral_checks','compact_snapshot_rebuilds',
    'compact_snapshot_dense_work_units','spectral_cubic_dimension_units','independence_checks',
    'factorization_attempts','factorizations_completed','condition_estimation_calls','triangular_calls',
    'triangular_rhs_columns','triangular_rhs_elements','triangular_dense_work_units')
MAX_CATEGORY_LENGTH=160
STRING_KEYS=set(COORDS)|{'event','mode','status','schema','method','stop_reason','no_information_reason',
    'hardware_dtype','device_model','source_validation_reason','counter_ownership','objective_scope','CE_gradient_rule'}
PAYLOAD_KEYS={'audit','text','message','log','logs','records','record','arrays','array','scores','labels',
    'gradients','trials','steps','inner_folds','prior_folds','final_fit','preparation','config','actual_parameters',
    'limitations','interpretation','source_summary','path','namespace'}


def require(condition,message):
    if not condition:raise ValueError(message)


def _constant(value):raise ValueError('Nonfinite JSON constant: '+value)


def read_json(path):
    with Path(path).open(encoding='utf-8') as stream:return json.load(stream,parse_constant=_constant)


def records(path):
    with Path(path).open(encoding='utf-8') as stream:
        for number,line in enumerate(stream,1):
            require(bool(line.strip()),f'Empty diagnostic record: {path.name}:{number}')
            value=json.loads(line,parse_constant=_constant)
            require(type(value) is dict,'Diagnostic record must be an object')
            yield value


def validate_summary(value):
    require(value.get('status')==INPUT_STATUS and value.get('schema')==METHOD_SCHEMA
        and value.get('method')==METHOD and value.get('scope')==SCOPE,'Completed Margin diagnostics required')
    require(isinstance(value.get('run_id'),str) and 0<len(value['run_id'])<=MAX_CATEGORY_LENGTH,'Explicit run binding required')
    require(isinstance(value.get('release_commit'),str) and re.fullmatch('[0-9a-f]{40}',value['release_commit']),
        'Actual runtime commit required')
    require(isinstance(value.get('source_summary'),str) and bool(value['source_summary']),'Source summary binding required')
    require(value.get('objective')=='RMSCE_only' and value.get('proximal_coefficient')==0
        and value.get('coordinate_ball_radius')==.5,'Frozen CE-only diagnostic identity mismatch')
    limits=value.get('qp_resources')
    require(type(limits) is dict and set(limits)=={'max_transitions','max_factor_buffer_bytes'}
        and all(type(v) is int and v>0 for v in limits.values()),'Explicit positive integer QP resources required')
    resources=value.get('resources',{})
    require(resources.get('peak_aggregation')=='MAX' and resources.get('work_aggregation')=='SUM'
        and isinstance(resources.get('scope'),str) and resources['scope'],'Actual work aggregation scope required')
    counters=resources.get('actual_counters')
    expected=set(PEAKS)|{'margin_qp_'+op+'_'+k for op in ('forward','adjoint') for k in QP_WORK}
    require(type(counters) is dict and expected<=set(counters) and all(
        isinstance(k,str) and re.fullmatch('[A-Za-z_][A-Za-z0-9_]*',k)
        and (v is None or type(v) is int and v>=0) for k,v in counters.items()),'Invalid actual work resource schema')
    for key in ('query_rows_used','source_rows_used'):
        if key in value:require(type(value[key]) is int and value[key]==0,'Forbidden diagnostic access')
    totals=value.get('totals',{})
    for key in ('actual_stages','curve_records','preparation_records'):
        require(type(totals.get(key)) is int and totals[key]>0,'Missing complete training stream counts: '+key)


def _primitive(key,value):
    if not isinstance(key,str) or len(key)>100 or not re.fullmatch('[A-Za-z_][A-Za-z0-9_]*',key):return False
    if key in PAYLOAD_KEYS or key.endswith(('_ref','_refs','_records','_text','_json','_path')):return False
    if value is None or type(value) in (bool,int):return True
    if type(value) is float:
        require(math.isfinite(value),'Nonfinite measured scalar: '+key);return True
    if type(value) is str:
        return (key in STRING_KEYS or key.endswith(('_reason','_rule'))) and len(value)<=MAX_CATEGORY_LENGTH \
            and not any(c in value for c in '\r\n') and not value.lstrip().startswith(('{','['))
    return False


def _accepted_increase(metrics,kind):
    accepted=metrics.get('accepted')
    if type(accepted) is not bool:return None,'ACCEPTANCE_NOT_RECORDED'
    if not accepted:return False,'TRIAL_NOT_ACCEPTED'
    before,after=(metrics.get('objective_before'),metrics.get('objective_after')) if kind=='RMSCE' else (
        metrics.get('mean_CE_before'),metrics.get('mean_CE_after'))
    if not all(type(v) in (int,float) and math.isfinite(v) for v in (before,after)):
        return None,'EXACT_MEASURED_BEFORE_AFTER_UNAVAILABLE'
    if kind=='RMSCE' and metrics.get('recorded_objective_minus_RMSCE')!=0:
        return None,'RMSCE_OBJECTIVE_EQUALITY_NOT_RECORDED'
    return bool(after>before),'FROM_RECORDED_BEFORE_AFTER'


def compact(record,kind):
    result={key:value for key,value in record.items() if '__' not in key and _primitive(key,value)}
    containers=('audit','initial_objective','final_objective') if kind=='stages' else (
        ('audit',) if kind=='preparations' else ('measured_scalars','objective','metrics'))
    for name in containers:
        values=record.get(name)
        require(values is None or type(values) is dict,'Invalid diagnostic scalar container: '+name)
        if values is None:continue
        require(values.get('source_validation') is None,'Phase2 source validation must remain unavailable')
        for key,value in values.items():
            if _primitive(key,value):result[name+'__'+key]=value
    require(record.get('source_validation') is None,'Phase2 source validation must remain unavailable')
    result.update(source_validation=None,source_validation_reason='PHASE2_SOURCE_ACCESS_FORBIDDEN')
    if kind=='curves':
        for target in ('RMSCE','mean_CE'):
            value,reason=_accepted_increase(record.get('metrics') or {},target) if record.get('event')=='MARGIN_JOINT_TRIAL' else (None,'NOT_A_TRIAL')
            result['accepted_'+target+'_increase']=value
            result['accepted_'+target+'_increase_reason']=reason
    return result


def identity(record,run_id,*,preparation=False):
    require(all(k in record for k in COORDS),'Missing training stage identity')
    require(record['run_id']==run_id,'Training stream belongs to another run')
    for key in ('row_id','split_id'):
        require(type(record[key]) is str and 0<len(record[key])<=MAX_CATEGORY_LENGTH,'Invalid physical training identity')
    require(record['scope'] in ('support_full_k1','support_oof','support_oneshot_proxy'),'Non-training scope rejected')
    require(all(type(record[k]) is int and record[k]>0 for k in ('parent_k','train_k')),'Invalid physical K identity')
    for key in ('fold','outer_trial'):require(record[key] is None or type(record[key]) is int and record[key]>=0,'Invalid fold/trial identity')
    state=record['state']
    require(state in (PREP_STATES if preparation else STATES),'Invalid training state')
    return tuple(record[k] if k!='state' else PREP_STATES[state] if preparation else state for k in COORDS)


def inspect_streams(root,summary):
    counts={};headers={};mapping={};stages={};preps=set();phases={};accepted={key:dict(true=0,false=0,unknown=0) for key in ('RMSCE','mean_CE')}
    for kind in STREAMS:
        count=0;keys=set()
        for record in records(root/(kind+'.jsonl')):
            small=compact(record,kind);keys.update(small);count+=1
            if kind=='stages':
                key=identity(record,summary['run_id']);require(key not in stages,'Duplicate final stage identity')
                require(record.get('audit',{}).get('status')=='COMPLETED','Incomplete final training stage')
                expected={event:record.get('actual_'+name+'_records') for event,name in
                    (('GRADIENT','gradient'),('TRIAL','trial'),('STEP','step'))}
                require(all(type(v) is int and v>=0 for v in expected.values()),'Missing actual training phase counts')
                stages[key]=dict(expected,INITIAL=1,FINAL=1)
            elif kind=='preparations':
                key=identity(record,summary['run_id'],preparation=True)
                require(key not in preps,'Duplicate preparation identity');preps.add(key)
            else:
                event=record.get('event');require(event in EVENTS,'Unknown training phase')
                key=identity(record,summary['run_id'],preparation=event=='MARGIN_JOINT_PREPARED')
                phase=event.removeprefix('MARGIN_JOINT_')
                row=phases.setdefault(key,dict.fromkeys(('PREPARED','INITIAL','GRADIENT','TRIAL','STEP','FINAL'),0))
                row[phase]+=1
                if phase=='TRIAL':
                    for name in accepted:
                        value=small['accepted_'+name+'_increase'];accepted[name]['unknown' if value is None else 'true' if value else 'false']+=1
        counts[kind]=count;headers[kind]=sorted(keys)
        mapping[kind]={key:('CONTRACT:PHASE2_SOURCE_ACCESS_FORBIDDEN' if key in ('source_validation','source_validation_reason')
            else 'DERIVED:$.metrics exact before/after and acceptance; see accepted_increase_scope' if key.startswith('accepted_')
            else '$.'+key.replace('__','.')) for key in headers[kind]}
    expected_counts=dict(stages=summary['totals']['actual_stages'],curves=summary['totals']['curve_records'],
        preparations=summary['totals']['preparation_records'])
    require(counts==expected_counts,'Complete diagnostic stream counts mismatch')
    require(set(stages)==preps==set(phases),'Missing training stage/preparation/phase identity')
    for key,expected in stages.items():
        require(phases[key]==dict(expected,PREPARED=1),'Missing or duplicated actual training phase')
    return dict(counts=counts,headers=headers,field_mapping=mapping,accepted_increase_observations=accepted)


def _csv_scalar(value):
    return value if type(value) is str else json.dumps(value,allow_nan=False)


def verify_outputs(out,inspection):
    for kind in STREAMS:
        header=inspection['headers'][kind];count=0
        with (out/(kind+'_compact.csv')).open(encoding='utf-8',newline='') as stream:
            reader=csv.DictReader(stream);require(reader.fieldnames==header,'Compact CSV header mismatch')
            for value,row in itertools.zip_longest(records(out/(kind+'_compact.jsonl')),reader):
                require(value is not None and row is not None and set(value)==set(row)==set(header),'Compact keys/count mismatch')
                require(all(v is None or type(v) in (str,bool,int,float) for v in value.values()),'Nonprimitive exported field')
                require(row=={k:_csv_scalar(value[k]) for k in header},'Compact CSV/JSONL value mismatch');count+=1
        require(count==inspection['counts'][kind],'Compact readback count mismatch')


def write_json(path,value):
    with path.open('x',encoding='utf-8') as stream:json.dump(value,stream,ensure_ascii=False,allow_nan=False,indent=2);stream.write('\n')


def export_scalars(*,diagnostics_root,output):
    root=Path(diagnostics_root).resolve();out=Path(output).resolve()
    if out.exists():raise FileExistsError(out)
    require(not out.is_relative_to(root),'Output must be outside original diagnostics')
    started=time.perf_counter();summary=read_json(root/'summary.json');validate_summary(summary)
    require(not (root/'failed.json').exists(),'Failed diagnostic evidence cannot be relabeled complete')
    inputs=[root/'summary.json']+[root/(name+'.jsonl') for name in STREAMS]
    stamps={path:(path.stat().st_size,path.stat().st_mtime_ns) for path in inputs}
    inspection=inspect_streams(root,summary)
    out.mkdir(parents=True,exist_ok=False)
    try:
        for kind in STREAMS:
            header=inspection['headers'][kind]
            with (out/(kind+'_compact.jsonl')).open('x',encoding='utf-8') as js,(out/(kind+'_compact.csv')).open('x',encoding='utf-8',newline='') as cs:
                writer=csv.DictWriter(cs,fieldnames=header);writer.writeheader()
                for record in records(root/(kind+'.jsonl')):
                    small=compact(record,kind);require(set(small)<=set(header),'Input scalar keys changed during conversion')
                    # An absent measured column remains null, never an inferred zero.
                    value={key:small.get(key) for key in header}
                    js.write(json.dumps(value,ensure_ascii=False,allow_nan=False)+'\n')
                    writer.writerow({key:_csv_scalar(value[key]) for key in header})
        require(all((path.stat().st_size,path.stat().st_mtime_ns)==stamp for path,stamp in stamps.items()),'Original diagnostics changed during conversion')
        verify_outputs(out,inspection)
        result=dict(status=STATUS,schema=SCHEMA,method=METHOD,run_id=summary['run_id'],release_commit=summary['release_commit'],
            scope='SCALAR_CONVERSION_OF_COMPLETE_TRAINING_DIAGNOSTICS_ONLY',diagnostics_root=str(root),
            source_summary=summary['source_summary'],source_diagnostic_status=summary['status'],qp_resources=summary['qp_resources'],
            **inspection,source_actual_counters=summary['resources']['actual_counters'],
            source_counter_scope=summary['resources']['scope'],source_peak_aggregation='MAX',source_work_aggregation='SUM',
            conversion_wall_seconds=time.perf_counter()-started,
            input_file_bytes={path.name:stamp[0] for path,stamp in stamps.items()},
            output_file_bytes={kind+'_compact.'+suffix:(out/(kind+'_compact.'+suffix)).stat().st_size for kind in STREAMS for suffix in ('jsonl','csv')},
            byte_scope='Physical summary/input JSONL and six compact files; excludes output summary itself, process memory and transmission',
            source_validation=None,source_validation_reason='PHASE2_SOURCE_ACCESS_FORBIDDEN',
            conversion_training_operations=0,conversion_model_updates=0,source_or_query_files_read=0,npz_files_read=0,
            original_inputs_modified=False,complete_streams_read=True,readback_verified=True,
            unknown_numeric_policy='NULL_NO_ZERO_IMPUTATION',max_categorical_string_length=MAX_CATEGORY_LENGTH,
            accepted_increase_scope='Trial observations only, from existing exact measured before/after; RMSCE is not arithmetic mean CE')
        write_json(out/'summary.json',result)
        require(read_json(out/'summary.json')==result,'Export summary readback mismatch')
        return result
    except Exception as exc:
        write_json(out/'failed.json',dict(status='MARGIN_TRAINING_AI_SCALARS_EXPORT_FAILED',run_id=summary['run_id'],
            error_type=type(exc).__name__,error=str(exc),original_inputs_preserved=True,automatic_retry=False))
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--diagnostics-root',required=True);parser.add_argument('--output',required=True)
    result=export_scalars(**vars(parser.parse_args()))
    print(json.dumps(dict(status=result['status'],counts=result['counts']),allow_nan=False))


if __name__=='__main__':main()
