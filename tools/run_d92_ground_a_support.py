"""Sequential read-only Ground A support supplement for eight declared source rows."""
import argparse
import csv
import json
import os
from pathlib import Path, PurePosixPath
import platform
import re
import sys
import time

from score_d92_ground_a_support import score_support, METHODS, STATUS as ROW_STATUS, SCHEMA as ROW_SCHEMA, SCOPE

ROOT=Path(__file__).resolve().parents[1]
STATUS='GROUND_A_SUPPORT_SUPERVISOR_COMPLETE'
SCHEMA='d92_ground_a_support_supervisor_v1'
SUMMARY_STATUSES={
    'd92_affine_joint_local_ridge_v1':'COMPLETE_AFFINE_JOINT_PROBE_VERIFIED',
    'd92_conditional_joint_local_ridge_v1':'COMPLETE_CONDITIONAL_JOINT_PROBE_VERIFIED'}
SUMMARY_FIELDS={'status','scope','run_id','release_commit','coverage','old_class_count','query_rows_used','source_rows_used','schema','method'}
LANE_FIELDS={'run_id','row_id','checkpoint_sha256','capsule_id','model_seed','schema','method','scope',
    'query_rows_used','source_rows_used','truth_read','support_features','episodes','sequence_paths','status','config','selection'}
LANE_FIELDS={key:({'selection':None} if key=='config' else None) for key in LANE_FIELDS}
ROW_FIELDS={'row_id','source_run_id','source_row_id','packet','support_features','fit_trace','source_summary',
    'expected_summary_status','expected_summary_schema','expected_method','expected_runtime_commit',
    'expected_checkpoint_sha256','expected_capsule_id','expected_model_seed','output_root'}


def require(condition,message):
    if not condition:raise ValueError(message)


def selected_json(text,fields):
    """Decode only permitted top-level metadata; lexically skip results arrays."""
    decoder=json.JSONDecoder();position=0;result={};seen=set()
    def space(index):
        while index<len(text) and text[index].isspace():index+=1
        return index
    position=space(position);require(text[position:position+1]=='{','Metadata must be a JSON object');position+=1
    while True:
        position=space(position)
        if text[position:position+1]=='}':position+=1;break
        key,position=decoder.raw_decode(text,position)
        require(isinstance(key,str) and key not in seen,'Duplicate/invalid metadata key');seen.add(key)
        position=space(position);require(text[position:position+1]==':','Malformed metadata member');position=space(position+1)
        nested=fields.get(key) if isinstance(fields,dict) else None
        if key in fields and nested is None:result[key],position=decoder.raw_decode(text,position)
        else:
            stack=[];quoted=False;escaped=False;start=position
            while position<len(text):
                char=text[position]
                if quoted:
                    if escaped:escaped=False
                    elif char=='\\':escaped=True
                    elif char=='"':quoted=False
                elif char=='"':quoted=True
                elif char in '[{':stack.append(char)
                elif char in ']}':
                    if not stack:break
                    require(stack.pop()==('[' if char==']' else '{'),'Malformed skipped metadata structure')
                elif char==',' and not stack:break
                position+=1
            require(position>start and not stack and not quoted,'Truncated skipped metadata')
            if key in fields:result[key]=selected_json(text[start:position],nested)
        position=space(position);delimiter=text[position:position+1]
        if delimiter=='}':position+=1;break
        require(delimiter==',','Malformed metadata separator');position+=1
        require(text[space(position):space(position)+1]!='}','Trailing metadata comma')
    require(space(position)==len(text),'Trailing metadata content')
    return result


def read_selected(path,fields):return selected_json(Path(path).read_text(encoding='utf-8'),fields)


def absolute(value):
    return isinstance(value,str) and bool(value) and '..' not in PurePosixPath(value).parts and '\\' not in value and (
        PurePosixPath(value).is_absolute() or Path(value).is_absolute())


def binding(row):
    return dict(run_id=row['source_run_id'],row_id=row['source_row_id'],checkpoint_sha256=row['expected_checkpoint_sha256'],
        capsule_id=row['expected_capsule_id'],model_seed=row['expected_model_seed'])


def validate_spec(spec):
    require(isinstance(spec.get('run_id'),str) and spec['run_id'],'Explicit supplement run ID required')
    relative=PurePosixPath(spec['spec_path'])
    require(not relative.is_absolute() and not Path(spec['spec_path']).is_absolute() and ':' not in spec['spec_path']
        and '..' not in relative.parts and '\\' not in spec['spec_path'] and relative.suffix=='.json','Relative spec path required')
    code=spec['code'];execution=spec['execution'];rows=spec['rows']
    require(absolute(code['cwd']) and absolute(code['environment']) and absolute(execution['remote_run_root']),'Explicit absolute runtime paths required')
    require(execution['launch_owner']=='root' and execution['cpu_lanes']==1 and execution['blas_threads_per_lane']==2,
        'One root-owned CPU lane / two BLAS threads required')
    root=PurePosixPath(execution['remote_run_root']);release=PurePosixPath(code['cwd'])
    require(root!=release and root not in release.parents and release not in root.parents,'Release/run paths overlap')
    require(isinstance(rows,list) and len(rows)==8,'Exactly eight explicitly declared source rows required')
    seen=set();source_seen=set();groups={}
    for row in rows:
        require(ROW_FIELDS<=set(row),'Incomplete explicit support supplement row')
        name=row['row_id'];require(isinstance(name,str) and name not in ('','.','..') and '/' not in name and '\\' not in name and name not in seen,'Invalid/duplicate analysis row ID');seen.add(name)
        require(PurePosixPath(row['output_root'])==root/name,'Supplement row output escaped new run')
        for key in ('packet','support_features','fit_trace','source_summary'):
            require(absolute(row[key]),'Explicit source path required: '+key)
            source=PurePosixPath(row[key]);source=source.parent if key in ('fit_trace','source_summary') else source
            require(root!=source and root not in source.parents and source not in root.parents,'Supplement run overlaps an original input')
        require(PurePosixPath(row['fit_trace']).name=='fit_trace.jsonl','Explicit original fit trace required')
        for key in ('source_run_id','source_row_id','expected_capsule_id'):
            require(isinstance(row[key],str) and row[key],'Explicit source binding required: '+key)
        require(type(row['expected_model_seed']) is int and row['expected_model_seed']>=0,'Explicit model seed required')
        require(isinstance(row['expected_checkpoint_sha256'],str) and re.fullmatch('[0-9a-f]{64}',row['expected_checkpoint_sha256']),'Explicit checkpoint SHA256 required')
        require(isinstance(row['expected_runtime_commit'],str) and re.fullmatch('[0-9a-f]{40}',row['expected_runtime_commit']),'Explicit original runtime commit required')
        schema=row['expected_summary_schema'];require(schema in METHODS,'Unsupported source method schema')
        require(row['expected_summary_status']==SUMMARY_STATUSES[schema] and row['expected_method']==METHODS[schema][0],'Expected source summary contract mismatch')
        identity=(row['source_run_id'],row['source_row_id']);require(identity not in source_seen,'Repeated source row');source_seen.add(identity)
        groups.setdefault(row['source_run_id'],[]).append(row)
    require(len(groups)==2 and {group[0]['expected_summary_schema'] for group in groups.values()}==set(METHODS)
        and all(len(group)==4 for group in groups.values()),'Four rows from each independent Affine/Conditional source run required')
    keys=('source_summary','expected_summary_status','expected_summary_schema','expected_method','expected_runtime_commit')
    for group in groups.values():
        require(all(all(row[key]==group[0][key] for key in keys) for row in group),'Inconsistent shared source summary declaration')
    return spec


def verify_sources(spec):
    """Read summary/lane metadata only, once per source summary, before any A score."""
    summaries={};lanes={};counts={}
    for row in spec['rows']:
        source_run=row['source_run_id'];schema=row['expected_summary_schema'];method,complete_status,scope,*_=METHODS[schema]
        if source_run not in summaries:
            value=read_selected(row['source_summary'],SUMMARY_FIELDS)
            expected=dict(status=row['expected_summary_status'],schema=schema,method=method,scope=scope,run_id=source_run,
                release_commit=row['expected_runtime_commit'],old_class_count=6,query_rows_used=0,source_rows_used=0)
            require(all(value.get(key)==item for key,item in expected.items()),'Independent source summary identity/access mismatch')
            require(value.get('coverage',{}).get('episodes')==160 and value['coverage'].get('sequence_paths')==1800,
                'Independent source summary is not complete 160-parent/1800-path evidence')
            summaries[source_run]=value;counts[source_run]=dict(episodes=0,sequence_paths=0)
        directory=Path(row['fit_trace']).parent
        startup=read_selected(directory/'startup.json',LANE_FIELDS)
        complete=read_selected(directory/'probe_complete.json',LANE_FIELDS)
        expected=dict(binding(row),schema=schema,method=method,scope=scope,query_rows_used=0,source_rows_used=0,truth_read=False)
        require(all(all(value.get(key)==item for key,item in expected.items()) for value in (startup,complete)),
            'Current source lane binding/access mismatch')
        require(Path(startup['support_features']).resolve()==Path(row['support_features']).resolve(),'Source lane feature cache mismatch')
        require(complete.get('status')==complete_status and type(complete.get('episodes')) is int
            and complete['episodes']>0 and complete['episodes']==startup.get('episodes')
            and type(complete.get('sequence_paths')) is int and complete['sequence_paths']>0,'Incomplete source lane')
        selection=startup.get('config',{}).get('selection');splits=selection.get('splits') if isinstance(selection,dict) else None
        require(isinstance(splits,list) and len(splits)==complete['episodes'] and complete.get('selection')==selection
            and len({item['split_id'] for item in splits})==len(splits),'Source lane selection binding mismatch')
        require(all(type(item.get('k')) is int and item['k'] in (1,5,10,20) for item in splits)
            and sum(1 if item['k']==1 else min(item['k'],3)+item['k'] for item in splits)==complete['sequence_paths'],
            'Source lane selected physical path count mismatch')
        lanes[row['row_id']]=dict(binding=binding(row),schema=schema,method=method,episodes=complete['episodes'],
            sequence_paths=complete['sequence_paths'],fit_trace=row['fit_trace'],source_summary=row['source_summary'],
            summary_release_commit=row['expected_runtime_commit'],runtime_commit_scope='BOUND_BY_INDEPENDENT_SOURCE_SUMMARY_NOT_ABSENT_LANE_FIELD')
        for key in counts[source_run]:counts[source_run][key]+=complete[key]
    for run_id,value in counts.items():
        require(value==dict(episodes=160,sequence_paths=1800),'Declared four source lanes do not cover verified source totals')
    return dict(source_summaries=summaries,lanes=lanes,statistics_deserialized=False,source_or_query_data_read=False)


def score_arguments(row):
    return dict(packet=row['packet'],support_features=row['support_features'],fit_trace=row['fit_trace'],output=row['output_root'],
        run_id=row['source_run_id'],row_id=row['source_row_id'],expected_checkpoint_sha256=row['expected_checkpoint_sha256'],
        expected_capsule_id=row['expected_capsule_id'],expected_model_seed=row['expected_model_seed'])


def score_row(row,source,score_fn=score_support):
    """The frozen scorer owns prediction persistence before support truth join."""
    result=score_fn(**score_arguments(row))
    marker=read_selected(Path(row['output_root'])/'complete.json',
        {'schema','status','binding','parent_count','query_rows_used','original_summary_modified','prediction_status'})
    expected=dict(schema=ROW_SCHEMA,status=ROW_STATUS,binding=binding(row),parent_count=source['episodes'],
        query_rows_used=0,original_summary_modified=False,prediction_status='GROUND_A_SUPPORT_PREDICTIONS_FIXED')
    require(marker==expected,'Independent pairing completion readback mismatch')
    require(all(result.get(key)==value for key,value in dict(schema=ROW_SCHEMA,status=ROW_STATUS,scope=SCOPE,
        binding=binding(row),parent_count=source['episodes'],query_rows_used=0,source_rows_used=0,
        training_performed=False,calibration_performed=False,original_summary_modified=False,original_trace_modified=False).items()),
        'Returned pairing result binding/access mismatch')
    return dict(status=ROW_STATUS,parent_count=result['parent_count'],output_root=row['output_root'],
        resources=result['resources'],source_binding=source)


def write(path,value,mode='x'):
    with Path(path).open(mode,encoding='utf-8') as stream:json.dump(value,stream,ensure_ascii=False,allow_nan=False,indent=2);stream.write('\n')


def peak_rss():
    if sys.platform.startswith('linux'):
        import resource
        return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
    return None


def run(spec,commit,*,score_fn=score_support):
    validate_spec(spec);require(isinstance(commit,str) and re.fullmatch('[0-9a-f]{40}',commit),'Actual supplement runtime commit required')
    root=Path(spec['execution']['remote_run_root']);root.mkdir(parents=True,exist_ok=False)
    import torch
    for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[name]='2'
    torch.set_num_threads(2)
    started=time.perf_counter();states={row['row_id']:dict(status='PENDING') for row in spec['rows']}
    startup=dict(schema=SCHEMA,run_id=spec['run_id'],scope=SCOPE,spec=spec,commit=commit,pid=os.getpid(),argv=sys.argv,
        python=sys.executable,started=time.time(),cpu_lanes=1,blas_threads_per_lane=2,torch_threads=torch.get_num_threads(),
        hardware=dict(platform=platform.platform(),processor=platform.processor(),cpu_count=os.cpu_count(),gpu_use=False),
        query_rows_used=0,source_rows_used=0,training_performed=False,original_inputs_modified=False)
    write(root/'startup.json',startup);write(root/'state.json',states)
    fields=('event','row_id','source_run_id','source_row_id','status','parent_count','elapsed_seconds','peak_process_rss_bytes','error_type','error')
    active=None;phase='source_summary_and_lane_bindings'
    with (root/'events.jsonl').open('x',encoding='utf-8') as full,(root/'compact.jsonl').open('x',encoding='utf-8') as compact, \
        (root/'compact.csv').open('x',encoding='utf-8',newline='') as csvfile,(root/'scoring.log').open('x',encoding='utf-8') as log:
        writer=csv.DictWriter(csvfile,fieldnames=fields);writer.writeheader()
        def emit(event,**values):
            value=dict(event=event,run_id=spec['run_id'],**values);text=json.dumps(value,ensure_ascii=False,allow_nan=False)
            full.write(text+'\n');full.flush();log.write(text+'\n');log.flush();print(text,flush=True)
            small={key:value.get(key) for key in fields};compact.write(json.dumps(small,ensure_ascii=False,allow_nan=False)+'\n');compact.flush()
            writer.writerow(small);csvfile.flush()
        emit('STARTUP',**{key:value for key,value in startup.items() if key!='run_id'})
        try:
            verified=verify_sources(spec);write(root/'source_bindings.json',verified)
            for row in spec['rows']:
                active=row['row_id'];phase='fixed_A_then_support_truth_pairing';tick=time.perf_counter()
                states[active]=dict(status='SCORING');write(root/'state.json',states,'w')
                emit('ROW_START',row_id=active,source_run_id=row['source_run_id'],source_row_id=row['source_row_id'],
                    status='SCORING',actual_call=dict(function='score_d92_ground_a_support.score_support',kwargs=score_arguments(row)))
                result=score_row(row,verified['lanes'][active],score_fn)
                states[active]=dict(result,elapsed_seconds=time.perf_counter()-tick,peak_process_rss_bytes=peak_rss())
                write(root/'state.json',states,'w')
                emit('ROW_COMPLETE',row_id=active,source_run_id=row['source_run_id'],source_row_id=row['source_row_id'],**states[active])
            complete=dict(schema=SCHEMA,status=STATUS,run_id=spec['run_id'],commit=commit,rows=len(spec['rows']),completed_rows=len(spec['rows']),
                parent_count=sum(value['parent_count'] for value in states.values()),row_outputs={key:value['output_root'] for key,value in states.items()},
                wall_seconds=time.perf_counter()-started,peak_process_rss_bytes=peak_rss(),peak_gpu_memory_bytes=None,
                incremental_network_transfer_bytes=None,energy=None,query_rows_used=0,source_rows_used=0,training_performed=False,
                resources_scope='CPU wall/RSS and per-row scorer measurements; numeric/file bytes are not deployment or transmission costs',
                original_inputs_modified=False,automatic_retry=False)
            write(root/'complete.json',complete);emit('COMPLETE',**{key:value for key,value in complete.items() if key!='run_id'});return complete
        except Exception as exc:
            if active is not None:states[active]=dict(status='FAILED',phase=phase,error_type=type(exc).__name__,error=str(exc))
            write(root/'state.json',states,'w')
            failure=dict(schema=SCHEMA,status='GROUND_A_SUPPORT_SUPERVISOR_FAILED',run_id=spec['run_id'],commit=commit,
                row_id=active,phase=phase,error_type=type(exc).__name__,error=str(exc),
                completed_rows=sum(value['status']==ROW_STATUS for value in states.values()),automatic_retry=False)
            write(root/'failed.json',failure);emit('FAILED',**{key:value for key,value in failure.items() if key!='run_id'});raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--spec',type=Path,required=True);parser.add_argument('--commit',required=True)
    args=parser.parse_args();run(json.loads(args.spec.read_text(encoding='utf-8')),args.commit)
