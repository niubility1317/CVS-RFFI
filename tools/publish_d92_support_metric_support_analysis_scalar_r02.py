"""Root-invoked scalar-r02 publication of independent SupportMetric support analysis.

This module imports only the standard library. It never changes the experiment
record or original fit output, and cannot promote an analysis to VERIFIED.
"""
import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import tarfile
import time

RUN_ID='20261002-phase2-d92-support-metric-joint-support-m2-r01'
PREPARATION_COMMIT='f291990c45f14cf322dfb1c6d2fc5a97e23f5a91'
FITTED_COMMIT='3c3eb6c513aaf20fe84514dcbeab3584bad1f238'
PATHS=('tools/analyze_d92_support_metric_joint_probe_scalar_r02.py',
       'configs/d92_support_metric_joint_support_20261002.json')
COMPLETE_STATUS='SUPPORT_METRIC_JOINT_PROBE_COMPLETE'
SUMMARY_STATUS='COMPLETE_SUPPORT_METRIC_JOINT_SUPPORT_ANALYSIS_VERIFIED'
REMOTE_USER='szu2070436088'
ENVIRONMENT=dict(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',
    MKL_NUM_THREADS='2',NUMEXPR_NUM_THREADS='2',PYTHONUNBUFFERED='1')
DEFAULT_SSH_CONFIG='E:/type10-7/tools/n607_ssh_config'
PLAN_KEY='support_analysis_scalar_r02'
ORIGINAL_RUN='/home/szu2070436088/2510044040/CV-SincNet/runs/'+RUN_ID
RELEASE='/home/szu2070436088/2510044040/CV-SincNet/releases/d92_support_metric_support_analysis_scalar_20261002_r02'
OUTPUT=ORIGINAL_RUN+'-analysis-scalar-r02'


def require(condition,message):
    if not condition:raise ValueError(message)


def oid(value):
    return isinstance(value,str) and len(value)==40 and all(c in '0123456789abcdef' for c in value)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def exclusive_json(path,value):
    with Path(path).open('x',encoding='utf-8') as stream:
        json.dump(value,stream,ensure_ascii=False,allow_nan=False,indent=2);stream.write('\n')


def validate_remote_paths(binding):
    """Lexical POSIX separation; remote callers also check resolved separation."""
    names=('release','output','log','original_run','remote_archive')
    for name in names:
        value=binding.get(name);p=PurePosixPath(value) if isinstance(value,str) else None
        require(p is not None and p.is_absolute() and len(p.parts)>1 and '..' not in p.parts
            and '\\' not in value and ':' not in value,'Unsafe remote path: '+name)
    release,output,original,archive=(PurePosixPath(binding[k]) for k in
        ('release','output','original_run','remote_archive'))
    require(PurePosixPath(binding['log'])==release/'analysis.log','Analysis log must be owned by the fresh release')
    require(archive==release.parent/(release.name+'.tar'),'Archive must belong to the fresh release')
    for left,right in ((release,original),(output,original),(archive,original),(release,output),(archive,output)):
        require(left!=right and left not in right.parents and right not in left.parents,
            'Analysis paths overlap original/output/release')


def validate_record(record):
    """Validate only the supplied preregistration; never read a spec or dataset."""
    require(isinstance(record,dict) and record.get('run_id')==RUN_ID,'Wrong original run identity')
    require(record.get('status')=='TRAINING_COMPLETE','Original training must be complete before dispatch')
    execution=record.get('execution');require(isinstance(execution,dict),'Execution metadata missing')
    plan=execution.get(PLAN_KEY);require(isinstance(plan,dict),'Analysis plan missing')
    require(plan.get('status')=='PREREGISTERED_NOT_LAUNCHED' and plan.get('attempt_id')=='r02'
        and plan.get('source_commit') is None,'Only the registered, undispatched attempt is supported')
    require(plan.get('source_preparation_commit')==PREPARATION_COMMIT
        and plan.get('fitted_runtime_commit')==FITTED_COMMIT,'Immutable source/runtime binding differs')
    require(plan.get('paths')==list(PATHS),'Analysis archive whitelist differs')
    require(execution.get('remote_run_root')==ORIGINAL_RUN and plan.get('release')==RELEASE
        and plan.get('output')==OUTPUT,'Fixed scalar-r02 release/output/original run differs')
    require(plan.get('launch_owner')=='root' and plan.get('automatic_retry') is False
        and all(plan.get(k) is False for k in ('query_read','source_sample_read','fit','parameter_selection')),
        'Root-only no-fit/no-query analysis permissions differ')
    require(plan.get('expected_complete_status')==COMPLETE_STATUS
        and plan.get('expected_summary_status')==SUMMARY_STATUS
        and plan.get('coverage')==dict(rows=4,parents=160,sequence_paths=1800),
        'Declared complete analysis coverage/status differs')
    binding=dict(plan,run_id=record['run_id'],original_run=execution.get('remote_run_root'),
        remote_archive=str(PurePosixPath(plan['release']).parent/(PurePosixPath(plan['release']).name+'.tar')),
        expected_remote_user=REMOTE_USER,environment=dict(ENVIRONMENT))
    validate_remote_paths(binding)
    command=plan.get('command')
    require(isinstance(command,list) and len(command)==11 and all(isinstance(v,str) and v for v in command),
        'Exact registered analysis argv required')
    require(PurePosixPath(command[0]).is_absolute() and '\\' not in command[0]
        and '..' not in PurePosixPath(command[0]).parts,'Explicit remote Python required')
    expected=[command[0],'-u',str(PurePosixPath(plan['release'])/PATHS[0]),
        '--spec',str(PurePosixPath(plan['release'])/PATHS[1]),'--run-root',binding['original_run'],
        '--output',plan['output'],'--expected-runtime-commit',FITTED_COMMIT]
    require(command==expected,'Analysis argv differs from the frozen preregistration')
    return binding


def validate_original_complete(complete,binding):
    """Structural metadata closure only; numerical closure belongs to analyzer."""
    require(isinstance(complete,dict) and complete.get('schema')=='d92_support_metric_joint_support_probe_v1'
        and complete.get('status')==binding['expected_complete_status']
        and complete.get('run_id')==binding['run_id']
        and complete.get('runtime_commit')==binding['fitted_runtime_commit'],
        'Original fit completion/runtime identity differs')
    require(complete.get('workload_complete') is True and complete.get('model_rows')==4
        and complete.get('completed_rows')==4 and complete.get('episodes')==160
        and complete.get('sequence_paths')==1800,'Original 4/160/1800 structural closure incomplete')
    require(all(complete.get(k) is False for k in ('truth_read','query_access','source_sample_access'))
        and complete.get('automatic_retry') is False,'Original access/automatic-retry flags differ')
    rows=complete.get('rows')
    require(isinstance(rows,dict) and len(rows)==4 and all(isinstance(k,str) and k for k in rows),
        'Original row inventory incomplete')
    for row in rows.values():
        require(isinstance(row,dict) and row.get('status')==binding['expected_complete_status']
            and row.get('release_commit')==binding['fitted_runtime_commit']
            and row.get('episodes')==40 and row.get('sequence_paths')==450,
            'Original row coverage/runtime is incomplete')
    return dict(run_id=complete['run_id'],runtime_commit=complete['runtime_commit'],
        status=complete['status'],model_rows=4,completed_rows=4,episodes=160,sequence_paths=1800,
        truth_read=False,query_access=False,source_sample_access=False)


def validate_archive_members(members,release,paths):
    """Only two ordinary source files and their ancestor directories are legal."""
    root=PurePosixPath(release).name
    require(paths==list(PATHS),'Analysis-only archive whitelist differs')
    expected={root+'/'+p for p in paths}
    directories={root}
    for name in expected:
        directories.update(str(parent) for parent in PurePosixPath(name).parents if str(parent)!='.')
    seen=set();files=set()
    for member in members:
        name=member.name;p=PurePosixPath(name)
        require(isinstance(name,str) and '\\' not in name and ':' not in name and
            not p.is_absolute() and '..' not in p.parts and p.parts and p.parts[0]==root
            and str(p)==name.rstrip('/'),'Unsafe archive member path')
        require(str(p) not in seen,'Duplicate archive member');seen.add(str(p))
        require(member.isfile() or member.isdir(),'Links/devices are forbidden in the source archive')
        if member.isfile():
            require(name in expected and name not in files,'Unlisted/duplicate source file');files.add(name)
        else:require(str(p) in directories,'Unlisted archive directory')
    require(files==expected,'Analysis source archive is incomplete')
    return sorted(files)


def git_binding(repository,paths,*,check_output=subprocess.check_output):
    """Read current source and independently compare its remote branch OID."""
    require(paths==list(PATHS),'Analysis-only source whitelist required')
    def git(*args):
        return check_output(['git',*args],cwd=repository,text=True,encoding='utf-8').strip()
    head=git('rev-parse','HEAD');branch=git('branch','--show-current')
    require(oid(head) and branch,'Committed branch identity required')
    ref='refs/heads/'+branch;remote=git('ls-remote','origin',ref).split()
    require(remote==[head,ref],'Source HEAD is not independently present at the remote branch')
    require(not git('status','--porcelain','--',*paths),'Analysis source/spec whitelist is not clean')
    return dict(source_commit=head,record_commit=head,branch=branch,remote_oid=remote[0])


def _common_program(binding):
    # The exact same metadata/archive validators run locally in tests and remotely.
    helpers='\n\n'.join(inspect.getsource(fn) for fn in
        (require,validate_remote_paths,validate_original_complete,validate_archive_members))
    return ('import getpass,hashlib,json,os,subprocess,tarfile,time\n'
        'from pathlib import Path,PurePosixPath\nPATHS='+repr(PATHS)+'\n'+helpers+'\n'
        'c='+repr(binding)+'\n'
        "if getpass.getuser()!=c['expected_remote_user']:raise PermissionError('Ordinary N607 user required; administrator forbidden')\n"
        'validate_remote_paths(c)\n'
        "original=Path(c['original_run']);release=Path(c['release']);output=Path(c['output']);archive=Path(c['remote_archive'])\n"
        "for candidate in (release,output,archive):\n"
        " if candidate.resolve()==original.resolve() or candidate.resolve().is_relative_to(original.resolve()) or original.resolve().is_relative_to(candidate.resolve()):raise ValueError('Resolved analysis/original path overlap')\n"
        "complete=validate_original_complete(json.loads((original/'complete.json').read_text(encoding='utf-8')),c)\n")


def preflight_program(binding):
    program=_common_program(binding)+StringTemplates.PREFLIGHT
    compile(program,'support-metric-analysis-preflight','exec')
    return program


WRAPPER=r'''
import getpass,json,os,subprocess,sys,time
from pathlib import Path
c=json.loads(sys.argv[1]);release=Path(c['release'])
if getpass.getuser()!=c['expected_remote_user']:raise PermissionError('Ordinary N607 user required; administrator wrapper forbidden')
def write(name,value):
 with (release/name).open('x',encoding='utf-8') as stream:json.dump(value,stream,allow_nan=False);stream.write('\n')
started=time.time()
child=subprocess.Popen(c['command'],cwd=release,env=dict(os.environ,**c['environment']),stdin=subprocess.DEVNULL)
write('analysis_startup.json',dict(schema='d92_support_analysis_execution_v1',status='RUNNING',
 pid=child.pid,wrapper_pid=os.getpid(),argv=c['command'],cwd=str(release),source_commit=c['source_commit'],
 fitted_runtime_commit=c['fitted_runtime_commit'],output=c['output'],original_run=c['original_run'],
 environment=c['environment'],started=started,launch_owner='root',fit=False,query_read=False,
 source_sample_read=False,parameter_selection=False,automatic_retry=False))
code=child.wait()
write('analysis_execution.json',dict(schema='d92_support_analysis_execution_v1',
 status='PROCESS_EXITED_AWAITING_ARTIFACT_READBACK' if code==0 else 'FAILED',
 pid=child.pid,wrapper_pid=os.getpid(),returncode=code,elapsed_seconds=time.time()-started,
 finished=time.time(),source_commit=c['source_commit'],fitted_runtime_commit=c['fitted_runtime_commit'],
 output=c['output'],original_run=c['original_run'],fit=False,query_read=False,
 source_sample_read=False,parameter_selection=False,automatic_retry=False))
'''


class StringTemplates:
    PREFLIGHT=r'''
if any(p.exists() for p in (release,output,archive)):raise FileExistsError('Existing analysis release/output/archive; reconcile, no repeat dispatch')
print(json.dumps(dict(status='PREFLIGHT_COMPLETE_NOT_DISPATCHED',remote_user=getpass.getuser(),
 run_id=c['run_id'],source_commit=c['source_commit'],fitted_runtime_commit=c['fitted_runtime_commit'],
 release=str(release),output=str(output),original_run=str(original),original_complete=complete,
 fit=False,query_read=False,source_sample_read=False,parameter_selection=False,automatic_retry=False)))
'''
    DISPATCH=r'''
if release.exists() or output.exists():raise FileExistsError('Existing analysis release/output; reconcile, no repeat dispatch')
if not archive.is_file() or hashlib.sha256(archive.read_bytes()).hexdigest()!=c['archive_sha256']:raise ValueError('Transferred source archive differs')
with tarfile.open(archive) as bundle:
 members=bundle.getmembers();validate_archive_members(members,c['release'],c['paths'])
 release.mkdir(parents=False,exist_ok=False);bundle.extractall(release.parent)
if not all((release/path).is_file() and not (release/path).is_symlink() for path in c['paths']):raise ValueError('Extracted analysis sources missing')
with Path(c['log']).open('x',encoding='utf-8') as log:
 process=subprocess.Popen([c['command'][0],'-u','-c',c['wrapper'],json.dumps(c)],cwd=release,
  env=dict(os.environ,**c['environment']),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
launch=dict(status='DISPATCHED_AWAITING_READBACK',pid=process.pid,cwd=str(release),
 source_commit=c['source_commit'],fitted_runtime_commit=c['fitted_runtime_commit'],run_id=c['run_id'],
 output=str(output),original_run=str(original),log=c['log'],argv=c['command'],launch_owner='root',
 remote_user=getpass.getuser(),environment=c['environment'],query_read=False,source_sample_read=False,
 fit=False,parameter_selection=False,automatic_retry=False)
with (release/'analysis_launch.json').open('x',encoding='utf-8') as stream:json.dump(launch,stream,allow_nan=False);stream.write('\n')
print(json.dumps(launch))
'''


def dispatch_program(binding):
    program=_common_program(dict(binding,wrapper=WRAPPER))+StringTemplates.DISPATCH
    compile(program,'support-metric-analysis-dispatch','exec')
    return program


def readback_program(binding):
    """Metadata only: no summary, numerical arrays, truth, or model imports."""
    program=_common_program(binding)+r'''
value=dict(status='DISPATCH_METADATA_READBACK',run_id=c['run_id'],metadata={},metadata_errors={},
 original_complete=complete,original_run=str(original),fit=False,query_read=False,source_sample_read=False)
for key,name in (('launch','analysis_launch.json'),('startup','analysis_startup.json'),('execution','analysis_execution.json')):
 path=release/name
 if path.is_file():
  try:value['metadata'][key]=json.loads(path.read_text(encoding='utf-8'))
  except (OSError,json.JSONDecodeError) as error:value['metadata_errors'][key]=str(error)
print(json.dumps(value))
'''
    compile(program,'support-metric-analysis-readback','exec')
    return program


def validate_launch(launch,binding):
    require(isinstance(launch,dict) and launch.get('status')=='DISPATCHED_AWAITING_READBACK'
        and type(launch.get('pid')) is int and launch['pid']>0,'Dispatch PID/status missing')
    for name in ('source_commit','fitted_runtime_commit','run_id','output','original_run','log','environment'):
        require(launch.get(name)==binding[name],'Dispatch binding differs: '+name)
    require(launch.get('cwd')==binding['release'] and launch.get('argv')==binding['command']
        and launch.get('remote_user')==REMOTE_USER and launch.get('launch_owner')=='root'
        and all(launch.get(k) is False for k in
            ('query_read','source_sample_read','fit','parameter_selection','automatic_retry')),
        'Dispatch user/permissions/argv differ')
    return launch


def _save_transport(folder,phase,result):
    for name in ('stdout','stderr'):
        value=getattr(result,name,None)
        if value is not None:
            payload=value if isinstance(value,bytes) else str(value).encode('utf-8')
            (folder/(phase+'.'+name)).write_bytes(payload)


def publish(record_path,*,repository=None,artifact_root,ssh_config=DEFAULT_SSH_CONFIG,
            host='N607',call=subprocess.run,source_check=git_binding):
    """One explicit invocation; timeout is UNKNOWN and artifacts block resubmission."""
    record=read(record_path);binding=validate_record(record)
    repository=Path(repository or Path.cwd());folder=Path(artifact_root)
    require(isinstance(host,str) and host and not host.startswith('-'),'Explicit SSH host required')
    require(not folder.exists(),'Existing publication folder; read-only reconcile, no repeat dispatch')
    folder.mkdir(parents=True,exist_ok=False)
    phase='LOCAL_SOURCE_BINDING';published=False
    flags=['-F',str(ssh_config),'-o','BatchMode=yes','-o','ConnectTimeout=10']
    try:
        source=source_check(repository,list(PATHS));require(oid(source.get('source_commit')),'Actual source commit missing')
        require(source.get('remote_oid')==source['source_commit'],'Source remote OID differs')
        binding.update(source)
        binding.update(archive=str(folder/(PurePosixPath(binding['release']).name+'.tar')))
        exclusive_json(folder/'binding.json',binding)
        phase='REMOTE_METADATA_PREFLIGHT'
        preflight=call(['ssh',*flags,'-T',host,'python3 -'],input=preflight_program(binding),
            text=True,encoding='utf-8',capture_output=True,timeout=30)
        _save_transport(folder,phase,preflight);require(preflight.returncode==0,'Remote metadata preflight failed')
        evidence=json.loads(preflight.stdout)
        require(evidence.get('status')=='PREFLIGHT_COMPLETE_NOT_DISPATCHED'
            and evidence.get('remote_user')==REMOTE_USER and evidence.get('source_commit')==binding['source_commit']
            and evidence.get('fitted_runtime_commit')==binding['fitted_runtime_commit']
            and evidence.get('run_id')==binding['run_id'] and evidence.get('release')==binding['release']
            and evidence.get('output')==binding['output'] and evidence.get('original_run')==binding['original_run'],
            'Metadata preflight response binding differs')
        closed=dict(run_id=binding['run_id'],runtime_commit=binding['fitted_runtime_commit'],
            status=binding['expected_complete_status'],model_rows=4,completed_rows=4,episodes=160,
            sequence_paths=1800,truth_read=False,query_access=False,source_sample_access=False)
        require(evidence.get('original_complete')==closed and all(evidence.get(k) is False for k in
            ('fit','query_read','source_sample_read','parameter_selection','automatic_retry')),
            'Metadata preflight response is not the closed no-fit/no-query run')
        exclusive_json(folder/'preflight.json',evidence)
        phase='LOCAL_SOURCE_ARCHIVE'
        call(['git','archive','--format=tar','--prefix='+PurePosixPath(binding['release']).name+'/',
            '--output='+binding['archive'],binding['source_commit'],*PATHS],cwd=repository,check=True,timeout=60)
        archive=Path(binding['archive'])
        with tarfile.open(archive) as bundle:validate_archive_members(bundle.getmembers(),binding['release'],list(PATHS))
        binding.update(archive_bytes=archive.stat().st_size,archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest())
        exclusive_json(folder/'archive_binding.json',binding)
        phase='TRANSFER'
        transfer=call(['scp',*flags,str(archive),host+':'+binding['remote_archive']],
            text=True,encoding='utf-8',capture_output=True,timeout=60)
        _save_transport(folder,phase,transfer);require(transfer.returncode==0,'Source archive transfer failed; reconcile')
        phase='REMOTE_EXTRACT_DISPATCH'
        landing=call(['ssh',*flags,'-T',host,'python3 -'],input=dispatch_program(binding),
            text=True,encoding='utf-8',capture_output=True,timeout=60)
        _save_transport(folder,phase,landing);require(landing.returncode==0,'Remote dispatch returned failure; reconcile')
        launch=validate_launch(json.loads(landing.stdout),binding);published=True
        exclusive_json(folder/'dispatch.json',launch)
        phase='INDEPENDENT_DISPATCH_METADATA_READBACK'
        observed=call(['ssh',*flags,'-T',host,'python3 -'],input=readback_program(binding),
            text=True,encoding='utf-8',capture_output=True,timeout=30)
        _save_transport(folder,phase,observed);require(observed.returncode==0,'Dispatch metadata readback unavailable')
        value=json.loads(observed.stdout);require(value.get('status')=='DISPATCH_METADATA_READBACK'
            and value.get('run_id')==binding['run_id'] and value.get('metadata_errors')=={},
            'Independent dispatch metadata readback differs')
        require(value.get('original_complete')==closed and value.get('original_run')==binding['original_run']
            and all(value.get(k) is False for k in ('fit','query_read','source_sample_read')),
            'Original closed metadata/access binding differs at readback')
        require(validate_launch(value.get('metadata',{}).get('launch'),binding)==launch,'Independent launch readback differs')
        exclusive_json(folder/'readback.json',value)
        result=dict(status='DISPATCHED_AWAITING_INDEPENDENT_READBACK',phase=phase,
            scope='DISPATCH_METADATA_ONLY_NOT_ANALYSIS_COMPLETION',binding=binding,launch=launch,
            automatic_retry=False,record_modified=False,original_output_modified=False,
            summary_status=None,analysis_verified=False,publication=str(folder/'publication.json'))
        exclusive_json(folder/'publication.json',result)
        return result
    except Exception as exc:
        if isinstance(exc,subprocess.TimeoutExpired):_save_transport(folder,phase,exc)
        status='UNKNOWN' if (isinstance(exc,subprocess.TimeoutExpired) or phase in
            ('TRANSFER','REMOTE_EXTRACT_DISPATCH','INDEPENDENT_DISPATCH_METADATA_READBACK')) else 'FAILED'
        result=dict(status=status,phase=phase,error_type=type(exc).__name__,error=str(exc),binding=binding,
            dispatch_response_observed=published,automatic_retry=False,record_modified=False,
            analysis_verified=False,summary_status=None,
            next_action='Preserve artifacts; root performs read-only reconciliation before any separately registered attempt')
        exclusive_json(folder/'publication.json',result)
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--record',type=Path,required=True)
    parser.add_argument('--repository',type=Path,default=Path.cwd())
    parser.add_argument('--artifact-root',type=Path,required=True)
    parser.add_argument('--ssh-config',default=DEFAULT_SSH_CONFIG)
    args=vars(parser.parse_args());record=args.pop('record')
    print(json.dumps(publish(record,**args),allow_nan=False))


if __name__=='__main__':main()
