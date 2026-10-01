"""One-owner sequential source-only Ground A packet export; no sample inference."""
import argparse
import csv
import json
import os
from pathlib import Path,PurePosixPath
import platform
import re
import sys
import time

from export_d92_ground_classifier_a_packet import export_packet,load_packet,bound_metadata,STATUS as PACKET_STATUS

ROOT=Path(__file__).resolve().parents[1]
STATUS='GROUND_A_PACKET_EXPORT_COMPLETE'
FAILED='GROUND_A_PACKET_EXPORT_FAILED'
ROLE_FIELDS=('role_ids','source_rxs','source_days','ratios','split_seed','num_classes')
SOURCE_FIELDS=set(ROLE_FIELDS)|{'native_role_comparison','classes','target_access_before_freeze','checkpoint_init'}
INITIAL_FIELDS={'scratch_only','checkpoint_sources','target_contact','target_training_contact','source_roles','ema_origin'}
SELECTED_FIELDS={'schema','selected_checkpoint','selected_checkpoint_sha256','selection_source'}
ROW_FIELDS={'row_id','checkpoint','source_contract_ref','expected_source_contract_ref','initialization_ref',
    'selected_checkpoint_ref','training_startup_ref','binding','provenance','output_root'}


def require(condition,message):
    if not condition:raise ValueError(message)


def selected_json(text,fields):
    """Parse permitted top-level fields; skip all other values lexically.

    In particular terminal checkpoint score sections are not deserialized.
    """
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
        if key in fields:result[key],position=decoder.raw_decode(text,position)
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


def validate_spec(spec):
    require(isinstance(spec.get('run_id'),str) and spec['run_id'],'Explicit run ID required')
    relative=PurePosixPath(spec['spec_path'])
    require(not relative.is_absolute() and not Path(spec['spec_path']).is_absolute() and ':' not in spec['spec_path']
        and '..' not in relative.parts and '\\' not in spec['spec_path'] and relative.suffix=='.json','Relative spec path required')
    code=spec['code'];execution=spec['execution'];rows=spec['rows']
    require(absolute(code['cwd']) and absolute(code['environment']) and absolute(execution['remote_run_root']),'Explicit absolute runtime paths required')
    require(execution['launch_owner']=='root' and execution['cpu_lanes']==1 and execution['blas_threads_per_lane']==2,'One root-owned CPU lane / two BLAS threads required')
    release=PurePosixPath(code['cwd']);root=PurePosixPath(execution['remote_run_root'])
    require(release!=root and release not in root.parents and root not in release.parents,'Release/run paths overlap')
    require(isinstance(rows,list) and rows and len({r['row_id'] for r in rows})==len(rows),'Explicit unique rows required')
    for row in rows:
        require(ROW_FIELDS<=set(row),'Incomplete explicit packet row')
        name=row['row_id'];require(isinstance(name,str) and name not in ('','.','..') and '/' not in name and '\\' not in name,'Invalid row ID')
        require(PurePosixPath(row['output_root'])==root/name,'Row packet output escaped run root')
        for key in ('checkpoint','source_contract_ref','expected_source_contract_ref','initialization_ref','selected_checkpoint_ref','training_startup_ref'):
            require(absolute(row[key]),'Explicit source path required: '+key)
            source=PurePosixPath(row[key]);require(source!=root and root not in source.parents,'Source input lies inside packet output')
        bound_metadata(row['binding'],row['provenance'])
        require(type(row['provenance'].get('checkpoint_epoch')) is int and row['provenance']['checkpoint_epoch']==200,
            'Explicit final-epoch-200 provenance required')
        require(PurePosixPath(row['training_startup_ref'])==PurePosixPath(row['checkpoint']).parent/'startup.json',
            'Training startup must belong to the original checkpoint source directory')
    return spec


def verify_row_sources(row):
    """Existing contract consistency only; no samples, IQ, scores or feature reads."""
    require(type(row['provenance'].get('checkpoint_epoch')) is int and row['provenance']['checkpoint_epoch']==200,
        'Explicit final-epoch-200 provenance required')
    require(PurePosixPath(row['training_startup_ref'])==PurePosixPath(row['checkpoint']).parent/'startup.json',
        'Training startup must belong to the original checkpoint source directory')
    actual=read_selected(row['source_contract_ref'],SOURCE_FIELDS)
    expected=read_selected(row['expected_source_contract_ref'],set(ROLE_FIELDS))
    initial=read_selected(row['initialization_ref'],INITIAL_FIELDS)
    selected=read_selected(row['selected_checkpoint_ref'],SELECTED_FIELDS)
    startup=read_selected(row['training_startup_ref'],{'selection'})
    require(startup.get('selection')=='final_epoch_200','Original source startup does not bind final epoch 200')
    require(actual.get('native_role_comparison')=='EXACT_MATCH' and actual.get('checkpoint_init')=='scratch_only'
        and actual.get('target_access_before_freeze') is False,'Source contract verdict or origin mismatch')
    for key in ROLE_FIELDS:
        require(key in actual and key in expected and actual[key]==expected[key],'Source role contract mismatch: '+key)
    require(isinstance(actual['role_ids'],dict) and actual['role_ids'] and actual['num_classes']==6,'Explicit original source role IDs/classes required')
    require(initial.get('scratch_only') is True and initial.get('checkpoint_sources')==[] and initial.get('target_contact') is False
        and initial.get('target_training_contact') is False and initial.get('source_roles')=='EXACT_MATCH'
        and initial.get('ema_origin')=='this_run_student','Checkpoint initialization/inheritance mismatch')
    metadata,_=bound_metadata(row['binding'],row['provenance'])
    require(actual.get('classes')==list(metadata.ordered_classes),'Original ctx classifier class row order mismatch')
    require(selected.get('schema')=='phase1_terminal_status_v2' and selected.get('selection_source')=='training_final_only'
        and selected.get('selected_checkpoint')==row['checkpoint']
        and selected.get('selected_checkpoint_sha256')==metadata.checkpoint_sha256,'Selected final source checkpoint identity mismatch')
    require(row['provenance'].get('source_role_comparison')=='EXACT_MATCH','Existing source-role provenance differs')
    return dict(status='EXISTING_SOURCE_INPUT_BINDINGS_MATCHED',source_contract_ref=row['source_contract_ref'],
        expected_source_contract_ref=row['expected_source_contract_ref'],initialization_ref=row['initialization_ref'],
        selected_checkpoint_ref=row['selected_checkpoint_ref'],training_startup_ref=row['training_startup_ref'],
        checkpoint_sha256=metadata.checkpoint_sha256,checkpoint_epoch=200,training_selection=startup['selection'],
        native_role_comparison='EXACT_MATCH',checkpoint_inheritance=[],target_access_before_freeze=False,
        ordered_classes=list(metadata.ordered_classes),role_counts={k:len(v) for k,v in actual['role_ids'].items()},
        selection_source=selected['selection_source'],scores_deserialized=False,data_revalidated=False)


def write(path,value,mode='x'):
    with Path(path).open(mode,encoding='utf-8') as stream:json.dump(value,stream,ensure_ascii=False,allow_nan=False,indent=2);stream.write('\n')


def peak_rss():
    if sys.platform.startswith('linux'):
        import resource
        return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
    return None


def run(spec,commit,*,export_fn=export_packet):
    validate_spec(spec);require(isinstance(commit,str) and re.fullmatch('[0-9a-f]{40}',commit),'Actual runtime commit required')
    root=Path(spec['execution']['remote_run_root']);root.mkdir(parents=True,exist_ok=False)
    import torch
    for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[name]='2'
    torch.set_num_threads(2)
    start=time.perf_counter();states={row['row_id']:dict(status='PENDING') for row in spec['rows']}
    startup=dict(run_id=spec['run_id'],schema='d92_ground_a_packet_export_run_v1',scope='SOURCE_CLASSIFIER_PACKET_ONLY',
        spec=spec,commit=commit,pid=os.getpid(),argv=sys.argv,python=sys.executable,started=time.time(),
        cpu_lanes=1,blas_threads_per_lane=2,torch_threads=torch.get_num_threads(),
        hardware=dict(platform=platform.platform(),processor=platform.processor(),cpu_count=os.cpu_count(),gpu_use=False),
        source_sample_rows_read=0,source_feature_rows_read=0,support_rows_read=0,query_rows_read=0,
        encoder_constructed=False,encoder_executed=False,actual_A_evaluated=False)
    write(root/'startup.json',startup);write(root/'state.json',states)
    fields=('event','row_id','status','checkpoint_sha256','model_seed','scale','norm_eps','head_rows','feature_dim','packet_total_file_bytes',
        'weight_numeric_bytes','elapsed_seconds','peak_process_rss_bytes','error_type','error')
    active=None;phase='startup'
    with (root/'events.jsonl').open('x',encoding='utf-8') as full,(root/'compact.jsonl').open('x',encoding='utf-8') as compact, \
        (root/'compact.csv').open('x',encoding='utf-8',newline='') as csvfile,(root/'training.log').open('x',encoding='utf-8') as log:
        writer=csv.DictWriter(csvfile,fieldnames=fields);writer.writeheader()
        def emit(event,**values):
            value=dict(event=event,run_id=spec['run_id'],**values);text=json.dumps(value,ensure_ascii=False,allow_nan=False)
            full.write(text+'\n');full.flush();log.write(text+'\n');log.flush();print(text,flush=True)
            small={key:value.get(key) for key in fields};compact.write(json.dumps(small,ensure_ascii=False,allow_nan=False)+'\n');compact.flush()
            writer.writerow(small);csvfile.flush()
        emit('STARTUP',**{k:v for k,v in startup.items() if k!='run_id'})
        try:
            for row in spec['rows']:
                active=row['row_id'];phase='source_input_binding';tick=time.perf_counter()
                states[active]=dict(status='VERIFYING_SOURCE_INPUTS');write(root/'state.json',states,'w')
                source=verify_row_sources(row);write(root/(active+'_source_binding.json'),source)
                meta,_=bound_metadata(row['binding'],row['provenance'])
                emit('PACKET_EXPORT_START',row_id=active,status='EXPORTING',checkpoint_sha256=meta.checkpoint_sha256,
                    model_seed=row['provenance'].get('model_seed'),
                    ordered_classes=list(meta.ordered_classes),scale=meta.scale,norm_eps=meta.norm_eps,head_rows=6,feature_dim=160,
                    source_binding=source,encoder_constructed=False,encoder_executed=False)
                phase='packet_export';marker=export_fn(checkpoint=row['checkpoint'],binding=row['binding'],provenance=row['provenance'],output=row['output_root'])
                phase='packet_readback';head=load_packet(row['output_root'])
                require(marker['status']==PACKET_STATUS and marker['checkpoint_sha256']==meta.checkpoint_sha256 and head.metadata==meta,'Exported packet binding mismatch')
                require(json.loads((Path(row['output_root'])/'complete.json').read_text(encoding='utf-8'))==marker,'Packet completion readback mismatch')
                states[active]=dict(status=PACKET_STATUS,packet=str(row['output_root']),packet_total_file_bytes=marker['packet_total_file_bytes'],
                    weight_numeric_bytes=marker['weight_numeric_bytes'],elapsed_seconds=time.perf_counter()-tick)
                write(root/'state.json',states,'w')
                emit('PACKET_EXPORT_COMPLETE',row_id=active,checkpoint_sha256=meta.checkpoint_sha256,scale=meta.scale,norm_eps=meta.norm_eps,
                    model_seed=row['provenance'].get('model_seed'),
                    head_rows=6,feature_dim=160,peak_process_rss_bytes=peak_rss(),**states[active])
            complete=dict(status=STATUS,run_id=spec['run_id'],commit=commit,rows=len(spec['rows']),completed_rows=len(spec['rows']),
                packet_total_file_bytes=sum(v['packet_total_file_bytes'] for v in states.values()),
                weight_numeric_bytes=sum(v['weight_numeric_bytes'] for v in states.values()),
                wall_seconds=time.perf_counter()-start,peak_process_rss_bytes=peak_rss(),peak_gpu_memory_bytes=None,
                incremental_network_transfer_bytes=None,encoder_executed=False,actual_A_evaluated=False,
                byte_scope='SUM_OF_ACTUAL_PACKET_FILES_EXCLUDES_SUPERVISOR_LOGS_AND_NETWORK_TRANSFER')
            write(root/'complete.json',complete);emit('COMPLETE',**{k:v for k,v in complete.items() if k!='run_id'});return complete
        except Exception as exc:
            if active is not None:states[active]=dict(status=FAILED,phase=phase,error_type=type(exc).__name__,error=str(exc))
            write(root/'state.json',states,'w')
            failure=dict(status=FAILED,run_id=spec['run_id'],commit=commit,row_id=active,phase=phase,
                error_type=type(exc).__name__,error=str(exc),completed_rows=sum(v['status']==PACKET_STATUS for v in states.values()),
                wall_seconds=time.perf_counter()-start,automatic_retry=False)
            write(root/'failed.json',failure);emit('FAILED',**{k:v for k,v in failure.items() if k!='run_id'});raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--spec',type=Path,required=True);p.add_argument('--commit',required=True)
    a=p.parse_args();run(json.loads(a.spec.read_text(encoding='utf-8')),a.commit)
