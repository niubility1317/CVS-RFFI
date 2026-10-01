"""Publish an exclusive pushed source archive and start one prediction supervisor."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import tempfile

from run_d92_group_barrier_joint_benchmark import ROOT, CPU_ENV, SCHEMA, STARTED, read, write, validate_spec, require, oid

FLAGS = ['-F','E:/type10-7/tools/n607_ssh_config','-o','BatchMode=yes','-o','ConnectTimeout=10']
# Explicit static import closure. Native encoder/checkpoint modules, experiment
# indexes, probe artifacts and real model/data packets are never archived.
RUNTIME_PATHS = ('code/cvsrffi/__init__.py', 'code/cvsrffi/d92_ground_classifier_a.py', 'code/cvsrffi/d92_margin_qp_head.py', 'code/cvsrffi/d92_margin_joint_local_ridge.py', 'code/cvsrffi/d92_affine_joint_local_ridge.py', 'code/cvsrffi/d92_conditional_joint_local_ridge.py', 'code/cvsrffi/d92_conditional_affine_kernel.py', 'code/cvsrffi/d92_function_coordinate_residual8_local_ridge.py', 'code/cvsrffi/d92_margin_constrained_residual8_local_ridge.py', 'code/cvsrffi/d92_joint_channel_local_ridge.py', 'code/cvsrffi/d92_prototype_transport_local_ridge.py', 'code/cvsrffi/d92_branch_support_probe.py', 'code/cvsrffi/d92_branch_ridge.py', 'code/cvsrffi/d92_branch_interaction.py', 'code/cvsrffi/d92_branch_local_ridge.py', 'tools/cvs_native_artifacts.py', 'tools/export_d92_branch_support_features.py', 'tools/export_d92_branch_features.py', 'tools/export_d92_mv_kme_features.py', 'tools/d92_orbit_feature_cache.py', 'tools/export_d92_ground_classifier_a_packet.py', 'tools/evaluate_d92_margin_joint_probe.py', 'tools/run_d92_margin_joint_probe.py', 'tools/evaluate_d92_branch_support_probe.py', 'tools/evaluate_d92_branch_local_ridge_probe.py', 'tools/evaluate_d92_registration_diagnostic.py', 'tools/run_d92_registration_diagnostic.py', 'tools/d92_registration_score_diagnostics.py', 'configs/d92_branch_local_ridge_frozen_20260929.json', 'tools/run_d92_branch_support_probe.py', 'tools/run_d92_group_barrier_joint_probe.py', 'tools/evaluate_d92_group_barrier_joint_probe.py', 'tools/preflight_d92_group_barrier_joint_probe.py', 'code/cvsrffi/d92_group_barrier_joint_local_ridge.py', 'code/cvsrffi/d92_group_barrier_gate.py', 'tools/evaluate_d92_group_barrier_joint_benchmark.py', 'tools/run_d92_group_barrier_joint_benchmark.py', 'tools/score_d92_group_barrier_joint_benchmark.py')
PATHS = RUNTIME_PATHS + ('tools/publish_d92_group_barrier_joint_benchmark.py',)
ENDPOINTS=('run_d92_group_barrier_joint_benchmark','evaluate_d92_group_barrier_joint_benchmark','score_d92_group_barrier_joint_benchmark')

IMPORT_PROGRAM = r'''
import importlib,json,sys
from pathlib import Path
root=Path(sys.argv[1]).resolve();paths=set(json.loads(sys.argv[2]))
sys.path[:0]=[str(root/'tools'),str(root/'code')]
import numpy as np
import torch
def forbidden(*args,**kwargs):raise RuntimeError('Source import attempted data/model access')
np.load=forbidden;torch.load=forbidden
for name in ('run_d92_group_barrier_joint_benchmark','evaluate_d92_group_barrier_joint_benchmark',
             'score_d92_group_barrier_joint_benchmark'):
 importlib.import_module(name)
found=[]
for name,module in list(sys.modules.items()):
 file=getattr(module,'__file__',None)
 if file:
  file=Path(file).resolve()
  if file.is_relative_to(root):
   relative=file.relative_to(root).as_posix()
   if relative.endswith('.py'):assert relative in paths,(name,relative)
   found.append(relative)
 assert not name.startswith(('baseline_origin_sat_view','cvsrffi.checkpoint_loading',
     'cvsrffi.identity_only_forward','cvsrffi.muse_ssdg')),'Native encoder import: '+name
for endpoint in ('run_d92_group_barrier_joint_benchmark','evaluate_d92_group_barrier_joint_benchmark',
                 'score_d92_group_barrier_joint_benchmark'):
 assert Path(sys.modules[endpoint].__file__).resolve().is_relative_to(root)
print(json.dumps(dict(status='VERIFIED',module_files=sorted(set(found)),data_access=False,native_encoder_loaded=False)))
'''


def release_paths(spec):
    validate_spec(spec)
    return list(PATHS)+[spec['spec_path']]


def readiness(spec, root=ROOT):
    paths = release_paths(spec)
    missing = [p for p in paths if not (Path(root)/p).is_file()]
    return dict(status='SOURCE_BUNDLE_AVAILABLE_NOT_LAUNCHED' if not missing else 'INCOMPLETE_SOURCE_BUNDLE',
                paths=paths,missing=missing,launched=False,scope='Source presence only; no runtime/data/scoring claim')


def verify_bundle_imports(root=ROOT):
    """Exact whitelist in an isolated interpreter; no packet or encoder access."""
    with tempfile.TemporaryDirectory(prefix='d92-group-barrier-query-import-') as directory:
        bundle = Path(directory)
        for relative in RUNTIME_PATHS:
            target = bundle/relative; target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(Path(root)/relative,target)
        import os
        result = subprocess.run([sys.executable,'-I','-s','-c',IMPORT_PROGRAM,str(bundle),json.dumps(RUNTIME_PATHS)],
            cwd=bundle,env=dict(os.environ,**CPU_ENV),text=True,encoding='utf-8',capture_output=True,timeout=120)
        if result.returncode:
            raise RuntimeError('Isolated source import failed: '+result.stderr[-4000:])
        value = json.loads(result.stdout)
        require(value.get('status')=='VERIFIED' and value.get('data_access') is False
                and value.get('native_encoder_loaded') is False, 'Isolated source import did not verify')
        return dict(value,scope='LOCAL_ISOLATED_SOURCE_IMPORT_NOT_REMOTE_RUNTIME_OR_NUMERICAL_VALIDATION')


def git_binding(spec, root=ROOT, *, check_output=subprocess.check_output, check_call=subprocess.check_call):
    paths = release_paths(spec)
    def git(*args):
        return check_output(['git',*args],cwd=root,text=True,encoding='utf-8').strip()
    commit = git('rev-parse','HEAD'); branch = git('branch','--show-current')
    require(oid(commit) and branch, 'Exact committed branch required')
    remote = git('ls-remote','origin','refs/heads/'+branch).split()
    require(len(remote)==2 and remote[0]==commit, 'Current branch commit is not independently verified pushed')
    require(not git('status','--porcelain','--',*paths), 'Source whitelist/spec has uncommitted changes')
    check_call(['git','merge-base','--is-ancestor',spec['code']['commit'],commit],cwd=root)
    # New prediction source follows its fixed preparation parent; pushed HEAD owns the archive.
    return dict(runtime_commit=commit,preparation_commit=spec['code']['commit'],branch=branch,remote_oid=remote[0])


REMOTE = r'''
import hashlib,json,os,subprocess,tarfile
from pathlib import Path,PurePosixPath
c=CONFIG
release,archive,run_root=map(Path,(c['release'],c['archive'],c['run_root']))
if release.exists() or run_root.exists():raise FileExistsError('Existing release/run; reconcile, no repeat launch')
if hashlib.sha256(archive.read_bytes()).hexdigest()!=c['sha256']:raise ValueError('Transferred archive mismatch')
expected={release.name+'/'+p for p in c['paths']}
with tarfile.open(archive) as tar:
 members=tar.getmembers();files=set()
 for member in members:
  name=PurePosixPath(member.name)
  if name.is_absolute() or '..' in name.parts or not name.parts or name.parts[0]!=release.name:raise ValueError('Unsafe archive path')
  if not member.isdir() and not member.isfile():raise ValueError('Unsafe archive member type')
  if not (release.parent/member.name).resolve().is_relative_to(release.resolve()):raise ValueError('Archive escaped release')
  if member.isfile():
   if member.name in files:raise ValueError('Duplicate archive member')
   files.add(member.name)
 if files!=expected:raise ValueError('Archive differs from explicit source whitelist')
 release.mkdir(exist_ok=False)
 tar.extractall(release.parent)
env=dict(os.environ,**c['environment'])
imports=subprocess.run([c['python'],'-I','-s','-c',c['import_program'],str(release),json.dumps(c['runtime_paths'])],
 cwd=release,env=env,capture_output=True,text=True)
for suffix,value in (('stdout',imports.stdout),('stderr',imports.stderr)):
 with (release/('isolated_import.'+suffix)).open('x',encoding='utf-8') as f:f.write(value)
if imports.returncode:raise RuntimeError('Remote source import failed; preserve release, no launch')
verified=json.loads(imports.stdout)
if verified.get('status')!='VERIFIED':raise ValueError('Remote isolated source import unverified')
argv=[c['python'],'-u',str(release/'tools/run_d92_group_barrier_joint_benchmark.py'),'--spec',str(release/c['spec_path']),'--commit',c['commit']]
with (release/'supervisor.log').open('x',encoding='utf-8') as log:
 process=subprocess.Popen(argv,cwd=release,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
landing=dict(status='DISPATCHED_AWAITING_READBACK',pid=process.pid,argv=argv,cwd=str(release),run_root=str(run_root),
 runtime_commit=c['commit'],launch_owner='root',environment=c['environment'],truth_read=False,scorer_invoked=False,automatic_retry=False)
with (release/'launch.json').open('x',encoding='utf-8') as f:json.dump(landing,f,indent=2)
print(json.dumps(landing))
'''


def remote_config(spec, binding, archive, digest):
    return dict(release=spec['code']['cwd'],archive=archive,run_root=spec['execution']['remote_run_root'],
        python=spec['code']['environment'],spec_path=spec['spec_path'],commit=binding['runtime_commit'],sha256=digest,
        paths=release_paths(spec),runtime_paths=list(RUNTIME_PATHS),environment=CPU_ENV,import_program=IMPORT_PROGRAM)


def reconcile_program(spec, *, wait_seconds=0):
    """Read only launch/startup/state/complete metadata; no retry or score read."""
    paths = dict(release=spec['code']['cwd'],run_root=spec['execution']['remote_run_root'],wait_seconds=wait_seconds)
    return '''import json,os,time,hashlib
from pathlib import Path
c='''+repr(paths)+'''
r=Path(c['release']);s=Path(c['run_root'])
end=time.monotonic()+c['wait_seconds']
while not (s/'startup.json').exists() and time.monotonic()<end:time.sleep(.5)
archive=r.parent/(r.name+'.tar')
value=dict(release_exists=r.exists(),run_exists=s.exists(),archive_exists=archive.exists(),metadata={},metadata_errors={})
if archive.is_file():value['archive_sha256']=hashlib.sha256(archive.read_bytes()).hexdigest()
for key,path in (('launch',r/'launch.json'),('startup',s/'startup.json'),('state',s/'state.json'),('complete',s/'complete.json')):
 if path.is_file():
  try:value['metadata'][key]=json.loads(path.read_text(encoding='utf-8'))
  except (json.JSONDecodeError,OSError) as error:value['metadata_errors'][key]=str(error)
launch=value['metadata'].get('launch',{});pid=launch.get('pid')
alive=False
if type(pid) is int:
 try:os.kill(pid,0);alive=True
 except ProcessLookupError:pass
 except PermissionError:alive=None
value['supervisor_pid_observed_alive']=alive
print(json.dumps(value))
'''


def verify_landing(value, spec, binding):
    meta = value.get('metadata',{}); launch_meta = meta.get('launch',{}); startup = meta.get('startup',{})
    commit = binding['runtime_commit']
    require(launch_meta.get('runtime_commit')==commit and launch_meta.get('cwd')==spec['code']['cwd']
            and launch_meta.get('run_root')==spec['execution']['remote_run_root'] and launch_meta.get('launch_owner')=='root',
            'Launch readback binding differs')
    require(startup.get('runtime_commit')==commit and startup.get('code_commit')==spec['code']['commit']
            and startup.get('resolved_spec')==spec and startup.get('run_id')==spec['run_id']
            and startup.get('group_id')==spec['group_id'] and startup.get('schema')==SCHEMA and startup.get('status')==STARTED
            and startup.get('pid')==launch_meta.get('pid') and startup.get('truth_read') is False
            and startup.get('scorer_invoked') is False, 'Independent startup readback missing/different')
    return dict(status='VERIFIED',scope='LAUNCH_AND_SUPERVISOR_STARTUP_NOT_PREDICTION_COMPLETION',evidence=value)


def publish(spec_path, root=ROOT, *, host='N607', artifact_root=None, import_check=verify_bundle_imports,
            git_check=git_binding, call=subprocess.run):
    spec = read(Path(root)/spec_path); validate_spec(spec)
    require(spec_path==spec['spec_path'], 'Publication/spec path differs')
    require(PurePosixPath(spec['code']['cwd']).is_absolute() and ':' not in spec['code']['cwd'], 'Remote release must be POSIX absolute')
    available = readiness(spec,root); require(not available['missing'], 'Incomplete explicit source whitelist')
    verified = import_check(root); require(verified.get('status')=='VERIFIED','Exact source import not verified')
    binding = git_check(spec,root)
    name = PurePosixPath(spec['code']['cwd']).name
    folder = Path(artifact_root or 'E:/type10-7/local_artifacts/d92_group_barrier_joint_query_publication')/name
    folder.mkdir(parents=True,exist_ok=False)
    archive = folder/(name+'.tar'); remote_archive = str(PurePosixPath(spec['code']['cwd']).parent/archive.name)
    checked = [spec['code']['cwd'],spec['execution']['remote_run_root'],remote_archive]
    probe = 'from pathlib import Path\nassert not any(Path(p).exists() for p in '+repr(checked)+'), "Existing output; reconcile"\n'
    phase = 'REMOTE_COLLISION_CHECK'
    try:
        call(['ssh',*FLAGS,'-T',host,'python3 -'],input=probe.encode('utf-8'),check=True)
        phase = 'LOCAL_EXCLUSIVE_ARCHIVE'
        call(['git','archive','--format=tar','--prefix='+name+'/','--output='+str(archive),binding['runtime_commit'],
              *release_paths(spec)],cwd=root,check=True)
        digest = hashlib.sha256(archive.read_bytes()).hexdigest()
        write(folder/'binding.json',dict(binding,archive=str(archive),sha256=digest,paths=release_paths(spec),spec=spec))
        phase = 'TRANSFER'
        call(['scp',*FLAGS,str(archive),host+':'+remote_archive],check=True)
        phase = 'REMOTE_EXTRACT_IMPORT_DISPATCH'
        program = REMOTE.replace('CONFIG',repr(remote_config(spec,binding,remote_archive,digest)))
        result = call(['ssh',*FLAGS,'-T',host,'python3 -'],input=program.encode('utf-8'),capture_output=True)
        for suffix,data in (('stdout',result.stdout),('stderr',result.stderr)):
            with (folder/('landing.'+suffix)).open('xb') as stream:stream.write(data)
        require(result.returncode==0,'Remote landing returned failure; preserve artifacts and reconcile')
        phase = 'INDEPENDENT_READBACK'
        evidence = call(['ssh',*FLAGS,'-T',host,'python3 -'],input=reconcile_program(spec,wait_seconds=30).encode('utf-8'),capture_output=True)
        require(evidence.returncode==0,'Launch readback unavailable; reconcile before any mutation')
        observed = json.loads(evidence.stdout)
        verdict = verify_landing(observed,spec,binding)
        write(folder/'publication.json',dict(verdict,phase=phase,binding=binding,automatic_retry=False))
        return verdict
    except Exception as exc:
        write(folder/'publication.json',dict(status='UNKNOWN' if phase in ('TRANSFER','REMOTE_EXTRACT_IMPORT_DISPATCH','INDEPENDENT_READBACK') else 'FAILED',
            phase=phase,error_type=type(exc).__name__,error=str(exc),binding=binding,automatic_retry=False,
            next_action='Read-only reconcile; never repeat publication or launch automatically'))
        raise


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--spec',required=True)
    parser.add_argument('--reconcile',action='store_true'); parser.add_argument('--host',default='N607')
    args = parser.parse_args()
    if args.reconcile:
        value = read(ROOT/args.spec); validate_spec(value)
        result = subprocess.run(['ssh',*FLAGS,'-T',args.host,'python3 -'],input=reconcile_program(value).encode('utf-8'),capture_output=True,check=True)
        print(result.stdout.decode('utf-8'))
    else:
        publish(args.spec,host=args.host)
