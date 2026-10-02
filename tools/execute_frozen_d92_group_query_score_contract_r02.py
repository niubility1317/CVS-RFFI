"""One root-invoked contract-r02 score over the immutable Group query run.

Only metadata is read here. The frozen r02 scorer owns fixed-stream closure and
truth-last scoring. Exit zero never certifies a score artifact in this wrapper.
"""
import argparse
import getpass
import json
import os
from pathlib import Path, PurePosixPath
import platform
import re
import subprocess
import sys
import time

SCHEMA='d92_frozen_group_query_score_execution_v1'
RUN_ID='20261001-phase2-d92-group-barrier-joint-repeat-m2-r01'
RUNTIME='73aa1b31fc68aeddfb3d0ef7882acafb5be5a155'
REMOTE_USER='szu2070436088'
BASE='/home/szu2070436088/2510044040/CV-SincNet'
RELEASE=BASE+'/releases/d92_group_barrier_joint_repeat_20261001_r01'
SPEC=RELEASE+'/configs/d92_group_barrier_joint_repeat_20261001.json'
ORIGINAL_RUN=BASE+'/runs/'+RUN_ID
SCORER_RELEASE=BASE+'/releases/d92_group_query_score_contract_20261002_r02'
OUTPUT_DIRECTORY=BASE+'/runs/'+RUN_ID+'-score-contract-r02'
OUTPUT=OUTPUT_DIRECTORY+'/summary.json'
PYTHON='/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python'
SCORER=SCORER_RELEASE+'/tools/score_d92_group_barrier_joint_benchmark_contract_r02.py'
SCORE_STATUS='GROUP_BARRIER_JOINT_QUERY_SCORE_COMPLETE'
PREDICTION_SCHEMA='d92_group_barrier_joint_query_benchmark_v1'
CPU_ENV=dict(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',
    NUMEXPR_NUM_THREADS='2',CUDA_VISIBLE_DEVICES='',PYTHONUNBUFFERED='1')


def require(condition,message):
    if not condition:raise ValueError(message)


def oid(value):
    return isinstance(value,str) and re.fullmatch('[0-9a-f]{40}',value) is not None


def _object(pairs):
    value={}
    for key,item in pairs:
        require(key not in value,'Duplicate JSON metadata key: '+key);value[key]=item
    return value


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'),object_pairs_hook=_object)


def write(path,value):
    with Path(path).open('x',encoding='utf-8') as stream:
        json.dump(value,stream,ensure_ascii=False,allow_nan=False,indent=2);stream.write('\n')


def validate_plan(plan,publication_source_commit):
    require(oid(publication_source_commit),'Actual publication source commit required')
    expected=dict(schema=SCHEMA,run_id=RUN_ID,attempt_id='r02',fitted_runtime_commit=RUNTIME,
        launch_owner='root',automatic_retry=False,training=False,fit=False,parameter_selection=False,
        original_release=RELEASE,original_run=ORIGINAL_RUN,scorer_release=SCORER_RELEASE,spec_path=SPEC,
        output_directory=OUTPUT_DIRECTORY,output=OUTPUT,expected_score_status=SCORE_STATUS,
        expected_rows=4,expected_parents=2400,environment=CPU_ENV,cpu_lanes=2,blas_threads=2)
    require(isinstance(plan,dict) and all(plan.get(k)==v and
        (type(plan.get(k)) is type(v) if type(v) in (int,bool) else True)
        for k,v in expected.items()),'Fixed score plan/path/permission identity differs')
    command=[PYTHON,'-u',SCORER,'--spec',SPEC,'--run-root',ORIGINAL_RUN,'--output',OUTPUT]
    require(plan.get('command')==command,'Only the frozen contract-r02 scorer argv is allowed')
    if 'source_preparation_commit' in plan:
        require(plan['source_preparation_commit']=='c7c46558d5eb8f34bb57fb09c12adc7ac2c5109f',
            'Publication preparation identity differs')
    out=PurePosixPath(plan['output_directory'])
    for original in (PurePosixPath(RELEASE),PurePosixPath(ORIGINAL_RUN),PurePosixPath(SCORER_RELEASE)):
        require(out!=original and original not in out.parents and out not in original.parents,
            'Scoring must preserve original release/run')
    return plan


def validate_original_metadata(spec,startup,complete):
    """Metadata only; do not repeat capsule, prediction or truth validation."""
    require(isinstance(spec,dict) and spec.get('schema')==PREDICTION_SCHEMA
        and spec.get('run_id')==RUN_ID and spec.get('code',{}).get('cwd')==RELEASE
        and spec['code'].get('environment')==PYTHON
        and spec.get('execution',{}).get('remote_run_root')==ORIGINAL_RUN,
        'Immutable original spec binding differs')
    rows=spec.get('rows');cohorts=spec.get('benchmark',{}).get('cohorts')
    require(isinstance(rows,list) and len(rows)==4 and isinstance(cohorts,dict)
        and set(cohorts)=={'rx3','rx1'},'Original row/cohort declarations incomplete')
    ids={r.get('row_id') for r in rows}
    require(len(ids)==4 and all(isinstance(k,str) and k for k in ids),'Original row identities ambiguous')
    pairs={(r.get('cohort'),r.get('expected_model_seed')) for r in rows}
    require(pairs=={(co,seed) for co in ('rx3','rx1') for seed in (2026092701,2026092702)},
        'Original four model/cohort rows differ')
    require(cohorts['rx3'].get('expected_split_count')==900
        and cohorts['rx1'].get('expected_split_count')==300,'Original 900/900/300/300 counts differ')
    require(startup.get('schema')==complete.get('schema')==PREDICTION_SCHEMA
        and startup.get('status')=='GROUP_BARRIER_QUERY_SUPERVISOR_STARTED'
        and complete.get('status')=='GROUP_BARRIER_QUERY_BENCHMARK_PREDICTIONS_COMPLETE'
        and startup.get('resolved_spec')==complete.get('resolved_spec')==spec
        and startup.get('run_id')==complete.get('run_id')==RUN_ID
        and startup.get('group_id')==complete.get('group_id')==spec.get('group_id'),
        'Original startup/completion spec identity differs')
    require(startup.get('runtime_commit')==complete.get('runtime_commit')==RUNTIME
        and startup.get('code_commit')==complete.get('code_commit')==spec['code'].get('commit')
        and oid(spec['code'].get('commit')),'Original runtime/preparation binding differs')
    require(complete.get('row_count')==complete.get('completed_row_count')==4
        and complete.get('all_predictions_fixed') is True and
        all(item.get(k) is False for item in (startup,complete)
            for k in ('truth_read','scorer_invoked','automatic_retry')),'All original predictions must be fixed before scoring')
    require(set(startup.get('rows',{}))==set(complete.get('rows',{}))==ids,
        'Original row inventory incomplete')
    total=0
    for row in rows:
        lane=complete['rows'][row['row_id']];count=cohorts[row['cohort']]['expected_split_count']
        marker=lane.get('marker',{})
        require(lane.get('status')=='COMPLETE' and lane.get('release_commit')==RUNTIME
            and lane.get('expected_model_seed')==row['expected_model_seed']
            and lane.get('expected_checkpoint_sha256')==row.get('expected_checkpoint_sha256')
            and lane.get('output_root')==row.get('output_root')
            and marker.get('status')=='COMPLETE' and marker.get('split_count')==count
            and marker.get('completed_split_count')==count
            and marker.get('release_commit')==RUNTIME,'Original row prediction closure incomplete')
        total+=count
    require(total==2400,'Original 2400-parent closure incomplete')
    return dict(rows=4,parents=2400,runtime_commit=RUNTIME,all_predictions_fixed=True,
        truth_read=False,scorer_invoked=False,automatic_retry=False,
        validation_scope='ORIGINAL_SPEC_STARTUP_COMPLETION_METADATA_ONLY; FULL_FIXED_STREAM_CLOSURE_OWNED_BY_ORIGINAL_SCORER')


def wait_for_child(child,*,platform_name=None,wait4=None):
    """Linux wait4 returns this child's observed ru_maxrss in KiB."""
    platform_name=sys.platform if platform_name is None else platform_name
    wait4=getattr(os,'wait4',None) if wait4 is None else wait4
    if platform_name.startswith('linux') and callable(wait4):
        pid,status,usage=wait4(child.pid,0)
        require(pid==child.pid,'Unexpected child wait identity')
        code=os.waitstatus_to_exitcode(status);child.returncode=code
        return dict(returncode=code,child_peak_rss_bytes=int(usage.ru_maxrss)*1024,
            child_peak_rss_scope='LINUX_WAIT4_CHILD_RU_MAXRSS_KIB; NOT_WRAPPER_OR_TOTAL_PROCESS_TREE_RSS')
    return dict(returncode=child.wait(),child_peak_rss_bytes=None,
        child_peak_rss_scope='N/A; PER_CHILD_LINUX_WAIT4_NOT_AVAILABLE')


def execute(plan,publication_source_commit,*,read_metadata=read,popen=subprocess.Popen,
            waiter=wait_for_child,user_fn=getpass.getuser,path_factory=Path):
    """Injectable stdlib boundaries exist for literal offline fixtures only."""
    validate_plan(plan,publication_source_commit)
    require(user_fn()==REMOTE_USER,'Ordinary szu2070436088 required; administrator forbidden')
    spec=read_metadata(SPEC);startup=read_metadata(ORIGINAL_RUN+'/startup.json')
    complete=read_metadata(ORIGINAL_RUN+'/complete.json')
    closure=validate_original_metadata(spec,startup,complete)
    out=path_factory(OUTPUT_DIRECTORY)
    require(not out.exists(),'Existing scoring attempt; preserve and reconcile, no repeated execution')
    out.mkdir(parents=True,exist_ok=False)
    hardware=dict(hostname=platform.node(),platform=platform.platform(),machine=platform.machine(),
        processor=platform.processor() or None,cpu_count=os.cpu_count(),wrapper_python=sys.executable,
        wrapper_python_version=platform.python_version(),scorer_python=PYTHON,
        gpu_measurement=None,gpu_measurement_reason='CUDA_DISABLED; NO_GPU_MEASUREMENT')
    environment=dict(os.environ,**CPU_ENV);started=time.time();tick=time.perf_counter()
    initial=dict(schema=SCHEMA,status='STARTING',run_id=RUN_ID,attempt_id='r02',
        wrapper_pid=os.getpid(),pid=None,argv=list(plan['command']),cwd=SCORER_RELEASE,
        publication_source_commit=publication_source_commit,fitted_runtime_commit=RUNTIME,
        original_release=RELEASE,original_run=ORIGINAL_RUN,scorer_release=SCORER_RELEASE,
        output=OUTPUT,output_directory=OUTPUT_DIRECTORY,
        environment=dict(CPU_ENV),cpu_lanes=2,blas_threads=2,hardware=hardware,started=started,
        original_metadata_closure=closure,launch_owner='root',training=False,fit=False,parameter_selection=False,
        automatic_retry=False,summary_read=False,analysis_verified=False)
    child=None;initial_written=False
    try:
        with (out/'score.log').open('x',encoding='utf-8') as log:
            child=popen(plan['command'],cwd=SCORER_RELEASE,env=environment,stdin=subprocess.DEVNULL,
                stdout=log,stderr=subprocess.STDOUT)
            require(type(child.pid) is int and child.pid>0,'Actual scoring child PID missing')
            initial.update(status='RUNNING',pid=child.pid)
            write(out/'score_startup.json',initial);initial_written=True
            measured=waiter(child)
        code=measured['returncode']
        require(type(code) is int,'Actual child return code missing')
        result=dict(initial,status='PROCESS_EXITED_AWAITING_ARTIFACT_READBACK' if code==0 else 'FAILED',
            returncode=code,wall_seconds=time.perf_counter()-tick,finished=time.time(),
            child_peak_rss_bytes=measured.get('child_peak_rss_bytes'),
            child_peak_rss_scope=measured.get('child_peak_rss_scope'),summary_read=False,
            expected_score_status=SCORE_STATUS,score_artifact_verified=False,
            execution_scope='CHILD_PROCESS_ONLY; EXIT_STATUS_IS_NOT_SCORE_ARTIFACT_VERIFICATION')
        write(out/'score_execution.json',result)
        return result
    except Exception as exc:
        if not initial_written:
            write(out/'score_startup.json',dict(initial,status='FAILED_TO_RECORD_CHILD_START',
                pid=None if child is None else getattr(child,'pid',None)))
        write(out/'score_execution.json',dict(initial,status='FAILED',returncode=None,
            wall_seconds=time.perf_counter()-tick,finished=time.time(),error_type=type(exc).__name__,error=str(exc),
            child_peak_rss_bytes=None,child_peak_rss_scope='N/A; CHILD_WAIT_NOT_COMPLETED',
            score_artifact_verified=False,summary_read=False,automatic_retry=False,
            next_action='Root performs read-only artifact/process reconciliation; never automatically re-execute'))
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--publication-source-commit',required=True)
    args=parser.parse_args();result=execute(read(args.plan),args.publication_source_commit)
    print(json.dumps(dict(status=result['status'],pid=result['pid'],returncode=result['returncode'],
        fitted_runtime_commit=RUNTIME,publication_source_commit=args.publication_source_commit,
        summary_read=False,score_artifact_verified=False),allow_nan=False))
    if result['returncode']!=0:raise SystemExit(result['returncode'] if result['returncode']>0 else 1)


if __name__=='__main__':main()
