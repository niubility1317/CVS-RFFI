"""Complete read-only Conditional training diagnostics, excluding outer/query results.

Only generic JSON, numeric archive, CSV and data-only transport primitives are
reused from the Affine collector. No Affine objective or gradient formula runs.
"""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import shlex
import subprocess

import numpy as np
import collect_d92_affine_joint_training_diagnostics as generic

require=generic.require
statistics=generic.statistics
selected_json=generic.selected_json
jsonlines=generic.jsonlines
referenced_arrays=generic.referenced_arrays
write_json=generic.write_json
write_csv=generic.write_csv
SCHEMA='d92_conditional_joint_local_ridge_v1'
METHOD='D92-ConditionalJointLocalRidge-v1'
SCOPE='SUPPORT_ONLY_CONDITIONAL_JOINT_OOF_AND_PROXY_NOT_QUERY_EVALUATION'
INPUT_STATUS='COMPLETE_CONDITIONAL_JOINT_PROBE_VERIFIED'
SNAPSHOT_STATUS='COMPLETE_CONDITIONAL_JOINT_TRAINING_READONLY_SNAPSHOT'
STATUS='COMPLETE_CONDITIONAL_JOINT_TRAINING_DIAGNOSTICS_DERIVED'
STATES={'B_prepare','C_prepare','B_CONDITIONAL','C_CONDITIONAL_seq'}
EVENTS={'CONDITIONAL_JOINT_'+key for key in ('PREPARED','INITIAL','GRADIENT','TRIAL','STEP','FINAL')}
COORDS=('run_id','row_id','split_id','scope','fold','outer_trial','parent_k','train_k','state')
META_FIELDS={'status','summary_schema','schema','method','scope','run_id','release_commit','coverage','algorithm',
    'training_stage_count','raw_training_sources','state_archives','query_rows_used','source_rows_used'}
PREFIXES=('projection','residual','projection_adjoint','residual_adjoint')
SUFFIXES=('factorization_count','triangular_solve_count','triangular_rhs_count','triangular_rhs_element_count','triangular_dense_work_unit_count')


def validate_metadata(meta):
    require(meta['status']==INPUT_STATUS and meta['summary_schema']=='d92_conditional_joint_support_summary_v1'
        and meta['schema']==meta['algorithm']['schema']==SCHEMA and meta['method']==METHOD and meta['scope']==SCOPE,
        'Complete independently verified Conditional metadata required')
    require(meta['query_rows_used']==meta['source_rows_used']==0,'Forbidden input use')
    require(meta['algorithm']['proximal_coefficient']==0. and meta['algorithm']['coordinate_ball_radius']==.5
        and meta['algorithm']['objective']=='RMS_across_classes_of_cross_fold_mean_CE_only','CE-only/hard-ball contract mismatch')
    count=meta['training_stage_count'];sources=meta['raw_training_sources']
    require(type(count) is int and count>0 and count==meta['coverage']['candidate_stage_count'],'Training stage coverage mismatch')
    require(sources and len({s['row_id'] for s in sources})==len(sources),'Missing/duplicate training lanes')


def event_key(name):
    return name=='outer_trial' or not name.startswith(('outer_','query_','score_','held_')) and name not in ('scores','labels','held_scoring')


def identity(row):return {key:row.get(key) for key in COORDS}


def training_reference(ref,run_id,row_id):
    path=Path(ref['path']);namespace=json.loads(ref['namespace'])
    require(not path.is_absolute() and path.parts and path.parts[0]=='state_arrays' and '..' not in path.parts
        and path.suffix=='.npz' and '\\' not in ref['path'],'Archive escaped training directory')
    require(namespace.get('run_id')==run_id and namespace.get('row_id')==row_id and namespace.get('state') in STATES
        and namespace.get('scope') in ('support_full_k1','support_oof','support_oneshot_proxy'),
        'Archive is not current-run training state')


def checked_events(lane,run_id,row_id):
    events=list(jsonlines(lane/'training_events.jsonl',event_key))
    compact=list(jsonlines(lane/'training_events_compact.jsonl',event_key))
    require(len(events)==len(compact),'Full/compact training stream coverage mismatch')
    for event,small in zip(events,compact):
        require(event['event'] in EVENTS and event['event']==small['event'] and identity(event)==identity(small),
            'Training event identity/order mismatch')
        require(event['schema']==SCHEMA and event['method']==METHOD and event['run_id']==run_id
            and event['row_id']==row_id and event['state'] in STATES,'Training event method/lineage mismatch')
    return events


def snapshot(summary_root,run_root=None):
    root=Path(summary_root);meta=selected_json((root/'summary.json').read_text(encoding='utf-8'),META_FIELDS);validate_metadata(meta)
    lanes=[];stage_count=0
    for source in meta['raw_training_sources']:
        row=source['row_id'];lane=Path(run_root)/row/'probe' if run_root else Path(source['compact_training_events']).parent
        require(not (lane/'probe_failed.json').exists(),'Original failure is preserved; cannot derive complete diagnostics')
        events=checked_events(lane,meta['run_id'],row);refs={}
        for event in events:
            for ref in referenced_arrays(event):
                training_reference(ref,meta['run_id'],row)
                require(ref['path'] not in refs or refs[ref['path']]==ref,'Conflicting training array reference')
                refs[ref['path']]=ref
        finals=[identity(e) for e in events if e['event']=='CONDITIONAL_JOINT_FINAL']
        require(len({json.dumps(v,sort_keys=True) for v in finals})==len(finals),'Duplicate final training stage')
        stage_count+=len(finals)
        lanes.append(dict(row_id=row,lane=str(lane),training_refs=list(refs.values()),
            event_index=[dict(identity(e),event=e['event'],iteration=e.get('iteration'),trial=e.get('trial')) for e in events]))
    require(stage_count==meta['training_stage_count'],'Missing full training stages')
    return dict(status=SNAPSHOT_STATUS,metadata=meta,lanes=lanes,
        source_summary=str(root/'summary.json'),boundary='COMPLETE_TRAINING_REFERENCES_NO_OUTER_QUERY_TRUTH_OR_RESULT_TABLES')


class Arrays(generic.Arrays):
    def __call__(self,ref):
        data=super().__call__(ref)
        require(set(data)==set(ref['arrays']),'Training archive key mismatch')
        for name,array in data.items():
            metadata=ref['arrays'][name]
            require(list(array.shape)==metadata['shape'] and str(array.dtype)==metadata['dtype']
                and array.nbytes==metadata['nbytes'],'Training array metadata mismatch')
        return data


def norm(value):return float(np.linalg.norm(value))


def gradient_metrics(event,arrays):
    data=arrays(event['state_ref']);g=data['g_Z'];direction=data['d_Z'];length=norm(g)
    return dict(CE_gradient_rule='g_Z_DIRECT_NO_PLUS_Z',proximal_gradient_norm=0.,CE_gradient_norm=length,
        coordinate_norm=norm(data['Z']),coordinate_parameter_count=int(g.size),
        normalized_direction_deviation_norm=norm(direction+g/length) if length else norm(direction),
        hard_ball_excess=max(0.,norm(data['Z'])-.5),
        CE_vs_coordinate_dot=float(np.sum(g*data['Z'])))


def trial_metrics(event,arrays):
    current=arrays(event['state_ref']);gradient=arrays(event['gradient_state_ref'])
    delta=current['Z']-gradient['Z'];g=gradient['g_Z']
    slope=float(np.sum(g*delta));before=float(event['loss_before']);after=float(event['loss_after'])
    threshold=before+1e-4*slope
    tolerance=128*np.finfo(np.float64).eps*max(1.,abs(before),abs(after),abs(threshold))
    expected=bool(after<=before+tolerance and after<=threshold+tolerance)
    require(event['accepted']==expected,'Archived actual-delta Armijo decision mismatch')
    require(norm(delta-current['delta_Z'])<=128*np.finfo(np.float64).eps*max(1.,norm(delta)),'Archived projected delta mismatch')
    require(norm(current['Z'])<=.5+128*np.finfo(np.float64).eps,'Hard coordinate ball exceeded')
    return dict(actual_projected_delta_norm=norm(delta),actual_delta_gradient_dot=slope,
        armijo_threshold_recomputed=threshold,comparison_tolerance=tolerance,accepted=expected,
        hard_ball_norm=norm(current['Z']),raw_step_direction_norm=float(event['step_size'])*norm(current['d_Z']))


def work_counters(audit):
    return {k:v for k,v in audit.items() if type(v) is int and k.endswith(('_count','_calls','_solves'))}


def validate_stage_work(audit):
    require(audit['status']=='COMPLETED','Incomplete training stage cannot be relabeled complete')
    for prefix in PREFIXES:
        values=[audit[prefix+'_'+suffix] for suffix in SUFFIXES]
        require(all(type(v) is int and v>=0 for v in values),'Invalid measured kernel work')
        factor,calls,rhs,elements,dense=values
        require(calls%2==0 and calls<=rhs and rhs<=elements<=dense and elements*elements<=rhs*dense,'Invalid RHS dimensional ledger')
        require(factor==0 if prefix.endswith('adjoint') else calls==2*factor,'Factor reuse ledger mismatch')
    require(audit['spectral_diagnostic_count']==3*audit['projection_factorization_count'],'Three-spectrum ledger mismatch')
    require(audit['inner_factorization_count']+audit['final_factorization_count']==
        audit['projection_factorization_count']+audit['residual_factorization_count'],'Projection/residual factor total mismatch')


def array_diagnostic(ref,data):
    # Every captured training array is visited; no outer held namespace is allowed.
    result=dict(namespace=json.loads(ref['namespace']),path=ref['path'],key=ref['key'],
        numeric_bytes=sum(a.nbytes for a in data.values()),arrays={})
    for name,array in data.items():
        result['arrays'][name]=dict(shape=list(array.shape),dtype=str(array.dtype),nbytes=int(array.nbytes),
            norm=norm(array.reshape(-1)),maximum_absolute=float(np.max(np.abs(array))) if array.size else None)
    if {'A','B','alpha','beta','v'}<=set(data):
        residual=data['B']@data['alpha']-data['A']@data['beta']+data['v']
        result['old_anchor_residual_norm']=norm(residual)
        result['constant_v_norm']=norm(data['v'])
    if {'g_Z','Z'}<=set(data):result['gradient_is_CE_only']=True
    return result


def extract(value,run_root=None):
    require(value['status']==SNAPSHOT_STATUS,'Captured training snapshot required')
    meta=value['metadata'];validate_metadata(meta);stages=[];curves=[];preparations=[];archives=[];totals=Counter()
    for source in value['lanes']:
        row=source['row_id'];lane=Path(run_root)/row/'probe' if run_root else Path(source['lane'])
        require(not (lane/'probe_failed.json').exists(),'Original failure is preserved; no retry or mutation')
        events=checked_events(lane,meta['run_id'],row)
        index=[dict(identity(e),event=e['event'],iteration=e.get('iteration'),trial=e.get('trial')) for e in events]
        require(index==source['event_index'],'Training event stream changed since snapshot')
        for ref in source['training_refs']:training_reference(ref,meta['run_id'],row)
        arrays=Arrays(lane,source['training_refs']);seen={};allowed={r['path']:r for r in source['training_refs']}
        stage_events={}
        for event in events:stage_events.setdefault(tuple(event.get(k) for k in COORDS),[]).append(event)
        for event in events:
            for ref in referenced_arrays(event):
                require(allowed.get(ref['path'])==ref,'Uncaptured training reference')
                seen[ref['path']]=ref
            item=dict(identity(event),event=event['event'],iteration=event.get('iteration'),trial=event.get('trial'))
            if event['event']=='CONDITIONAL_JOINT_GRADIENT':item['metrics']=gradient_metrics(event,arrays)
            elif event['event']=='CONDITIONAL_JOINT_TRIAL':item['metrics']=trial_metrics(event,arrays)
            else:item['metrics']={}
            # Full objective curves are training observations; no result-table join.
            item['objective']=event.get('objective')
            item['measured_scalars']={k:v for k,v in event.items() if v is None or type(v) in (str,bool,int,float)}
            curves.append(item)
            if event['event']=='CONDITIONAL_JOINT_PREPARED':preparations.append(dict(identity(event),audit=event));totals.update(work_counters(event))
            if event['event']=='CONDITIONAL_JOINT_FINAL':
                audit=event['audit'];validate_stage_work(audit);totals.update(work_counters(audit))
                same=stage_events[tuple(event.get(k) for k in COORDS)]
                for suffix,field in (('GRADIENT','gradients'),('TRIAL','trials'),('STEP','steps')):
                    observed=[e['state_ref'] for e in same if e['event']=='CONDITIONAL_JOINT_'+suffix]
                    require(observed==[e['state_ref'] for e in audit[field]],'Complete training curve differs from final audit: '+field)
                stages.append(dict(identity(event),mode=event['mode'],audit=audit,
                    CE_only=True,initial_objective=audit.get('initial_objective'),final_objective=audit.get('final_objective'),
                    actual_gradient_records=len(audit['gradients']),actual_trial_records=len(audit['trials']),actual_step_records=len(audit['steps'])))
        require(set(seen)=={r['path'] for r in source['training_refs']},'Captured training reference omitted')
        for ref in source['training_refs']:archives.append(dict(row_id=row,**array_diagnostic(ref,arrays(ref))))
        require(arrays.read_paths==set(seen),'Not all training archives were read')
    require(len(stages)==meta['training_stage_count'],'Complete stage coverage mismatch')
    require(len(preparations)==meta['coverage']['candidate_preparation_count'],'Complete preparation coverage mismatch')
    for key,count in totals.items():
        if key in meta['coverage']:require(count==meta['coverage'][key],'Training work differs from verified summary: '+key)
    groups={}
    for stage in stages:
        key=(stage['mode'],stage['train_k']);groups.setdefault(key,[]).append(stage)
    strata=[]
    for (mode,k),items in sorted(groups.items()):
        strata.append(dict(mode=mode,train_k=k,stage_count=len(items),
            optimizer_steps=sum(v['audit']['optimizer_steps'] for v in items),
            initial_RMSCE=statistics([(v['initial_objective'] or {}).get('RMSCE') for v in items]),
            final_RMSCE=statistics([(v['final_objective'] or {}).get('RMSCE') for v in items])))
    return dict(status=STATUS,schema=SCHEMA,method=METHOD,run_id=meta['run_id'],scope='COMPLETE_TRAINING_ONLY_NO_OUTER_QUERY_TRUTH',
        release_commit=meta['release_commit'],source_summary=value['source_summary'],stages=stages,curves=curves,
        preparations=preparations,archives=archives,strata=strata,totals=dict(totals,actual_stages=len(stages),
        curve_records=len(curves),preparation_records=len(preparations),training_archive_files=len(archives)),
        objective='RMSCE_only',proximal_coefficient=0.,coordinate_ball_radius=.5,
        resource_scope='MEASURED_TRAINING_AUDITS_AND_NUMERIC_ARRAY_PROXIES_NOT_FLOPS_OR_DEPLOYMENT_PACKAGE',
        limitations=['Inner supervised training curves are not independent performance evidence.',
            'Numeric bytes exclude Python overhead and unmeasured full deployment/transfer bytes.',
            'Old-anchor residual zero does not guarantee query margins; the constant v remains in deployment.',
            'Original files and failures are read-only; no automatic retry.'])


def collect(summary_root,run_root=None):return extract(snapshot(summary_root,run_root),run_root)


def write_outputs(output,value):
    out=Path(output);out.mkdir(parents=True,exist_ok=False)
    streams=('stages','curves','preparations','archives','strata')
    write_json(out/'summary.json',{k:v for k,v in value.items() if k not in streams})
    for name in streams:
        with (out/(name+'.jsonl')).open('x',encoding='utf-8') as stream:
            for row in value[name]:stream.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
        write_csv(out/(name+'.csv'),value[name])
    text=['# ConditionalJoint 完整训练诊断','',value['status'],
        '全部训练事件与捕获的训练数组均已遍历，无 outer/query/truth 读取。',
        'CE 梯度为 g_Z 原值，proximal 梯度为 0；0.5 硬球与投影后的实际 delta Armijo 单独核对。',
        'projection/residual 两个因子、两组伴随 RHS 和三项谱诊断分别保留。',
        'JSONL/CSV 保留全部曲线，不抽样。真实阶段计时与 numeric 工作代理不可互换。','']+value['limitations']
    with (out/'report.md').open('x',encoding='utf-8') as stream:stream.write('\n'.join(text)+'\n')


def transport_source(request=None):
    # Keep the proven bounded parser transport: the snapshot is JSON data, never a Python literal.
    import base64
    import gzip
    primitive=Path(generic.__file__).read_text(encoding='utf-8')
    source="import sys,types\n_m=types.ModuleType('collect_d92_affine_joint_training_diagnostics')\nsys.modules[_m.__name__]=_m\nexec("+repr(primitive)+",_m.__dict__)\n"
    if request is not None:
        encoded=base64.b64encode(gzip.compress(json.dumps(request,ensure_ascii=False,allow_nan=False).encode('utf-8'),mtime=0)).decode('ascii')
        source+='import base64,gzip,json\nSNAPSHOT_REQUEST=json.loads(gzip.decompress(base64.b64decode('+repr(encoded)+')).decode("utf-8"))\n'
    return source+Path(__file__).read_text(encoding='utf-8')


def remote_operation(args,operation,request=None):
    require(bool(args.ssh_config) and bool(args.remote_python),'Explicit SSH config/Python required')
    command=[args.remote_python,'-',operation]
    if args.summary_root:command+=['--summary-root',args.summary_root]
    if args.run_root:command+=['--run-root',args.run_root]
    result=subprocess.run(['ssh','-F',args.ssh_config,args.ssh_host,shlex.join(command)],
        input=transport_source(request),text=True,encoding='utf-8',capture_output=True,timeout=1800)
    require(result.returncode==0,'Read-only remote operation failed: '+result.stderr[-2000:])
    return json.loads(result.stdout)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('summary-root','run-root','output','snapshot-output','snapshot','ssh-host','ssh-config','remote-python'):p.add_argument('--'+name)
    p.add_argument('--snapshot-stdout',action='store_true');p.add_argument('--extract-stdout',action='store_true');args=p.parse_args()
    if args.snapshot_stdout or args.extract_stdout:
        require(not args.ssh_host and not args.output and not args.snapshot_output,'Read-only stdout operation required')
        value=snapshot(args.summary_root,args.run_root) if args.snapshot_stdout else extract(globals()['SNAPSHOT_REQUEST'],args.run_root)
        print(json.dumps(value,ensure_ascii=False,allow_nan=False));return
    if args.snapshot_output:
        require(args.summary_root and not args.snapshot and not args.output and not Path(args.snapshot_output).exists(),'Exclusive snapshot phase required')
        value=remote_operation(args,'--snapshot-stdout') if args.ssh_host else snapshot(args.summary_root,args.run_root)
        require(value['status']==SNAPSHOT_STATUS,'Snapshot status mismatch');write_json(args.snapshot_output,value);return
    require(args.output and not Path(args.output).exists(),'Exclusive diagnostics output required')
    if args.snapshot:
        request=json.loads(Path(args.snapshot).read_text(encoding='utf-8'))
        value=remote_operation(args,'--extract-stdout',request) if args.ssh_host else extract(request,args.run_root)
    else:
        require(args.summary_root and not args.ssh_host,'Remote collection requires snapshot then extract')
        value=collect(args.summary_root,args.run_root)
    require(value['status']==STATUS,'Diagnostics status mismatch');write_outputs(args.output,value)


if __name__=='__main__':main()
