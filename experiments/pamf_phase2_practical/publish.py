"""Single committed archive, one transfer comparison, compile and detach."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile

REMOTE=r'''
import hashlib,json,os,subprocess,tarfile
from pathlib import Path
p=PACKAGE;rt=p['runtime'];release=Path(rt['release_root']);root=Path(rt['run_root'])
if os.geteuid()==0 or str(Path(rt['project'])).split('/')[2]!=subprocess.check_output(['whoami'],text=True).strip():raise ValueError('Ordinary account mismatch')
if release.exists() or root.exists():raise FileExistsError('Reconcile existing release/run before retry')
if hashlib.sha256(Path(p['archive_remote']).read_bytes()).hexdigest()!=p['sha256']:raise ValueError('Release transfer mismatch')
for path in [rt['phase1_release'],rt['source_contract'],rt['source_cache'],rt['DURA']['checkpoint'],*rt['PAMF_sources'].values(),*rt['frozen_sources'].values()]:
 if not Path(path).exists():raise FileNotFoundError(path)
release.mkdir()
with tarfile.open(p['archive_remote']) as archive:
 for member in archive.getmembers():
  if not member.isfile() or release.resolve() not in (release/member.name).resolve().parents:raise ValueError('Invalid archive member')
 archive.extractall(release)
Path(rt['runtime_file']).write_text(json.dumps(rt,indent=2)+'\n')
env=dict(os.environ,PYTHONPATH=str(release),CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
subprocess.run([rt['python'],'-m','compileall','-q',str(release)],cwd=release,env=env,check=True)
command=[rt['python'],'-u','-m','experiments.pamf_phase2_practical.runner','dispatch','--runtime',rt['runtime_file']]
log=release/'controller.log'
with log.open('x') as f:child=subprocess.Popen(command,cwd=release,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
result=dict(status='SUBMITTED_AWAITING_READBACK',pid=child.pid,cwd=str(release),argv=command,log=str(log),code_commit=rt['code_commit'],archive_transfer='VERIFIED',remote_compile='PASS')
(release/'submit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
'''

def run(repo,runtime,output):
    repo=Path(repo);output=Path(output);output.mkdir(parents=True,exist_ok=True);rt=json.loads(Path(runtime).read_text(encoding='utf-8'))
    def git(*args):return subprocess.check_output(['git','-C',str(repo),*args])
    commit=git('rev-parse','HEAD').decode().strip();branch=git('branch','--show-current').decode().strip()
    if git('ls-remote','origin','refs/heads/'+branch).decode().split()[0]!=commit:raise ValueError('Current upstream OID differs')
    paths=['experiments/pamf_phase2_practical','configs/pamf_phase2_practical_mixture_20261028.json','automation_reports/CV-SincNet/'+rt['run_id']]
    if git('status','--porcelain','--',*paths).strip():raise ValueError('Release paths are uncommitted')
    rt['code_commit']=commit
    archive=output/(rt['run_id']+'-release.tar.gz')
    if archive.exists():raise FileExistsError('Reconcile prior publication before retry')
    source=git('archive','--format=tar',commit,'--',*paths)
    with tarfile.open(fileobj=io.BytesIO(source)) as old,tarfile.open(archive,'w:gz') as target:
        for member in old:
            if not member.isfile():continue
            data=old.extractfile(member).read();item=tarfile.TarInfo(member.name);item.size=len(data);item.mode=0o644;target.addfile(item,io.BytesIO(data))
        data=(commit+'\n').encode();item=tarfile.TarInfo('release_commit.txt');item.size=len(data);target.addfile(item,io.BytesIO(data))
    connection=['-F',str(repo/'tools/n607_ssh_config'),'-o','BatchMode=yes','-o','ConnectTimeout=10']
    package=dict(runtime=rt,archive_remote=rt['project']+'/releases/'+archive.name,sha256=hashlib.sha256(archive.read_bytes()).hexdigest())
    (output/'runtime.json').write_text(json.dumps(rt,indent=2),encoding='utf-8');(output/'package.json').write_text(json.dumps(package,indent=2),encoding='utf-8')
    subprocess.run(['scp',*connection,str(archive),'N607:'+package['archive_remote']],check=True,timeout=60)
    script=REMOTE.replace('PACKAGE',repr(package));compile(script,'remote','exec')
    result=subprocess.run(['ssh',*connection,'-T','N607','python3 -'],input=script.encode(),capture_output=True,timeout=60)
    (output/'submission.stdout.txt').write_bytes(result.stdout);(output/'submission.stderr.txt').write_bytes(result.stderr)
    if result.returncode:raise RuntimeError(result.stdout.decode(errors='replace')+result.stderr.decode(errors='replace'))
    value=json.loads(result.stdout);(output/'submit.json').write_text(json.dumps(value,indent=2),encoding='utf-8');print(json.dumps(value))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--repo',required=True);p.add_argument('--runtime',required=True);p.add_argument('--output',required=True);a=p.parse_args();run(a.repo,a.runtime,a.output)
