"""Publish one committed data-only release over the already verified N607 route."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile
from .publish import connection_options


REMOTE=r'''
import json,os,hashlib,subprocess,tarfile
from pathlib import Path
p=PACKAGE;rt=p['runtime'];release=Path(rt['release_root']);root=Path(rt['run_root'])
assert not release.exists() and not root.exists(), 'Reconcile existing release/run before retry'
assert os.geteuid()!=0 and Path(rt['project']).parts[2]==__import__('getpass').getuser()
for key in ['raw_directory','legacy_directory','phase1_release']:
 assert Path(rt[key]).is_dir(),key
assert len(list(Path(rt['raw_directory']).glob('*.zip')))==20
assert hashlib.sha256(Path(p['archive_remote']).read_bytes()).hexdigest()==p['archive_sha256']
release.mkdir()
with tarfile.open(p['archive_remote']) as z:
 for m in z.getmembers():
  assert m.isfile() and release.resolve() in (release/m.name).resolve().parents
 z.extractall(release)
Path(rt['runtime_file']).write_text(json.dumps(rt,indent=2)+'\n')
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONPATH=str(release),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
subprocess.run([rt['python'],'-m','compileall','-q',str(release)],cwd=release,env=env,check=True)
command=[rt['python'],'-u','-m',rt.get('controller_module','experiments.pamf_phase2_practical.full_runner'),'--runtime',rt['runtime_file']]
with (release/'controller.log').open('x') as f:
 child=subprocess.Popen(command,cwd=release,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
out=dict(status='SUBMITTED_AWAITING_READBACK',pid=child.pid,argv=command,cwd=str(release),run_root=str(root),
         code_commit=rt['code_commit'],archive_transfer='VERIFIED',remote_compile='PASS')
(release/'submit.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
'''


def run(a):
    repo=Path(a.repo);out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    rt=json.loads(Path(a.runtime).read_text(encoding='utf-8'))
    def git(*x):return subprocess.check_output(['git','-C',str(repo),*x])
    commit=git('rev-parse','HEAD').decode().strip();branch=git('branch','--show-current').decode().strip()
    assert git('ls-remote','origin','refs/heads/'+branch).decode().split()[0]==commit
    paths=['experiments/pamf_phase2_practical',rt.get('configuration_path','configs/pamf_phase2_full_eq_data_r08_20261029.json'),
           'configs/pamf_phase2_practical_mixture_20261028.json','automation_reports/CV-SincNet/'+rt['run_id']]
    assert not git('status','--porcelain','--',*paths).strip(),'Uncommitted release input'
    rt['code_commit']=commit
    archive=out/(rt['run_id']+'-release.tar')
    raw=git('archive','--format=tar',commit,'--',*paths)
    with tarfile.open(fileobj=io.BytesIO(raw)) as source,tarfile.open(archive,'w') as target:
        for member in source:
            if member.isfile():
                target.addfile(member,source.extractfile(member))
    rt['runtime_file']=rt['release_root']+'/runtime.json'
    package=dict(runtime=rt,archive_remote=rt['project']+'/releases/'+archive.name,
                 archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest())
    (out/'runtime.json').write_text(json.dumps(rt,indent=2),encoding='utf-8')
    (out/'package.json').write_text(json.dumps(package,indent=2),encoding='utf-8')
    connection=connection_options(repo,a.ssh_config)
    subprocess.run(['scp',*connection,str(archive),'N607:'+package['archive_remote']],check=True,timeout=60)
    code=REMOTE.replace('PACKAGE',repr(package));compile(code,'remote','exec')
    result=subprocess.run(['ssh',*connection,'-T','N607','python3 -'],input=code.encode(),capture_output=True,timeout=60)
    (out/'submission.stdout.txt').write_bytes(result.stdout);(out/'submission.stderr.txt').write_bytes(result.stderr)
    if result.returncode:raise RuntimeError(result.stderr.decode(errors='replace')+result.stdout.decode(errors='replace'))
    value=json.loads(result.stdout);(out/'submit.json').write_text(json.dumps(value,indent=2),encoding='utf-8')
    print(json.dumps(value))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('repo','runtime','ssh-config','output'):p.add_argument('--'+key,required=True)
    run(p.parse_args())
