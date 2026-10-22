"""Publish execution acceleration for only unstarted original G rows."""
import argparse,hashlib,io,json,subprocess,tarfile
from pathlib import Path

RUN='20261022-phase1-DURA-teacher-cache-control-r01'
BASE='3e8806900449e97f3e8fa62822322932168c6447'
REMOTE=r'''
import hashlib,json,os,subprocess,tarfile
from pathlib import Path
c=__PACKAGE__;rt=c['runtime'];release=Path(rt['release_root'])
if release.exists() or Path(rt['control_root']).exists():raise FileExistsError('Existing acceleration owner; reconcile before retry')
if subprocess.check_output(['whoami'],text=True).strip()!=c['expected_user']:raise ValueError('Ordinary account mismatch')
for key in ('previous_state','previous_runtime','source_jobs','external_pipeline_state'):
 if not Path(rt[key]).is_file():raise FileNotFoundError(rt[key])
if not Path(rt['worker_release']).is_dir():raise FileNotFoundError('Original worker release missing')
if hashlib.sha256(Path(c['archive_remote']).read_bytes()).hexdigest()!=c['sha256']:raise ValueError('Archive transfer differs')
release.mkdir()
with tarfile.open(c['archive_remote']) as archive:
 for member in archive.getmembers():
  if not member.isfile() or release.resolve() not in (release/member.name).resolve().parents:raise ValueError('Invalid archive member')
 archive.extractall(release)
Path(rt['runtime_file']).write_text(json.dumps(rt,indent=2)+'\n',encoding='utf-8')
env=dict(os.environ,PYTHONPATH=str(release),CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
subprocess.run([rt['python'],'-m','compileall','-q',str(release)],cwd=release,env=env,check=True)
command=[rt['python'],'-u','-m','experiments.cric_dura_comparison.multiprocess_control','--config',rt['runtime_file']]
with (release/'continuation.log').open('x') as log:
 child=subprocess.Popen(command,cwd=release,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
receipt=dict(status='SUBMITTED_NOT_YET_VERIFIED',pid=child.pid,cwd=str(release),argv=command,commit=rt['code_commit'])
(release/'submit.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
print(json.dumps(receipt))
'''


def runtime_for(previous,commit):
    if previous['worker_commit']!=BASE or previous['run_id']!='20261022-phase1-T1-DURA-multiprocess-control-r03':
        raise ValueError('Unexpected prior owner or baseline worker')
    project=previous['project'];control=project+'/runs/'+RUN
    return dict(previous,run_id=RUN,release_root=project+'/releases/'+RUN,control_root=control,
        runtime_file=project+'/releases/'+RUN+'/runtime.json',code_commit=commit,
        launch_owner='codex-dura-teacher-cache-20261022',source_jobs=previous['jobs'],jobs=control+'/resolved_jobs.json',
        previous_state=previous['control_root']+'/pipeline_state.json',previous_runtime=previous['runtime_file'],
        previous_control_root=previous['control_root'],previous_controller_module='experiments.cric_dura_comparison.multiprocess_control',
        previous_controller_cwd=previous['release_root'],G_teacher_acceleration=True)


def publish(repo,output,previous_file,ssh_config,runtime_factory=None):
    repo=Path(repo);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    def git(*args):return subprocess.check_output(['git','-C',str(repo),*args])
    commit=git('rev-parse','HEAD').decode().strip();branch=git('branch','--show-current').decode().strip()
    if git('ls-remote','origin','refs/heads/'+branch).decode().split()[0]!=commit:raise ValueError('Remote OID differs')
    previous=json.loads(Path(previous_file).read_text(encoding='utf-8'));rt=(runtime_factory or runtime_for)(previous,commit)
    paths=[p for p in git('ls-tree','-r','--name-only','HEAD','experiments/cric_dura_comparison').decode().splitlines()
           if p.endswith('.py') or '/vendor/' in p]
    if rt.get('supervised_extension'):
        dependency='experiments/phase1_external_dg_aug/augmentation.py'
        if not git('ls-tree','--name-only','HEAD',dependency).strip():raise ValueError('Committed default concat dependency missing')
        paths.append(dependency)
    required=('bootstrap.py','g_cache.py','g_lineage.py','g_train.py','pipeline.py','acceleration_smoke.py','teacher_targets.py')
    if rt.get('F_source_binding_recovery'):
        required+=('source_iq.py','f_binding_recovery.py','publish_F_binding_repair.py')
    if any('experiments/cric_dura_comparison/'+name not in paths for name in required):
        raise ValueError('Committed baseline execution dependencies missing')
    if git('status','--porcelain','--',*paths).strip():raise ValueError('Uncommitted execution sources')
    archive=output/(rt['run_id']+'.tar.gz')
    if archive.exists():raise FileExistsError('Existing release archive; reconcile before retry')
    with tarfile.open(archive,'w:gz') as tar:
        for name,blob in [(p,git('show',commit+':'+p)) for p in paths]+[('release_commit.txt',(commit+'\n').encode())]:
            entry=tarfile.TarInfo(name);entry.size=len(blob);entry.mode=0o644;tar.addfile(entry,io.BytesIO(blob))
    package=dict(runtime=rt,expected_user=rt['project'].split('/')[2],archive_remote=rt['project']+'/releases/'+archive.name,
        sha256=hashlib.sha256(archive.read_bytes()).hexdigest())
    (output/'runtime.json').write_text(json.dumps(rt,indent=2)+'\n',encoding='utf-8')
    subprocess.run(['scp','-F',str(ssh_config),str(archive),'N607:'+package['archive_remote']],check=True,timeout=90)
    result=subprocess.run(['ssh','-F',str(ssh_config),'-T','-o','BatchMode=yes','N607',rt['python']+' -'],
        input=REMOTE.replace('__PACKAGE__',repr(package)),text=True,encoding='utf-8',capture_output=True,timeout=90)
    result.check_returncode();receipt=json.loads(result.stdout)
    (output/'submit.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8');print(json.dumps(receipt))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ('repo','output','previous','ssh-config'):p.add_argument('--'+key,required=True)
    a=p.parse_args();publish(a.repo,a.output,a.previous,a.ssh_config)
