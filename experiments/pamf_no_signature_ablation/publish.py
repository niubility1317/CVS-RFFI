import argparse
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile
from experiments.cric_dura_comparison.publish import ssh,CONNECTION
from .method import RUN

REMOTE=r'''
import hashlib,json,os,subprocess,tarfile
from pathlib import Path
p=PACKAGE
rt=p['runtime'];release=Path(rt['release_root']);root=Path(rt['run_root'])
if release.exists() or root.exists():raise FileExistsError('Reconcile existing release/run before any retry')
if hashlib.sha256(Path(p['archive_remote']).read_bytes()).hexdigest()!=p['sha256']:raise ValueError('Archive transfer mismatch')
release.mkdir()
with tarfile.open(p['archive_remote']) as tar:
 for m in tar.getmembers():
  if not m.isfile() or release.resolve() not in (release/m.name).resolve().parents:raise ValueError('Invalid archive member')
 tar.extractall(release)
(release/'runtime.json').write_text(json.dumps(rt,indent=2)+'\n')
env=dict(os.environ,PYTHONPATH=str(release),CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
subprocess.run([rt['python'],'-m','compileall','-q',str(release/'experiments/pamf_no_signature_ablation')],cwd=release,env=env,check=True)
cmd=[rt['python'],'-u','-m','experiments.pamf_no_signature_ablation.runner','dispatch','--config',rt['runtime_file']]
log=release/'controller.log'
with log.open('x') as f:child=subprocess.Popen(cmd,cwd=release,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
result=dict(status='SUBMITTED_NOT_YET_VERIFIED',pid=child.pid,cwd=str(release),argv=cmd,log=str(log),archive_transfer='VERIFIED',compile='PASS',code_commit=rt['code_commit'])
(release/'submit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
'''


def publish(repo,output):
    repo,out=Path(repo),Path(output)
    def git(*a):return subprocess.check_output(['git','-C',str(repo),*a])
    commit=git('rev-parse','HEAD').decode().strip();branch=git('branch','--show-current').decode().strip()
    if git('ls-remote','origin','refs/heads/'+branch).decode().split()[0]!=commit:raise ValueError('Remote branch OID mismatch')
    if git('status','--porcelain','--','experiments/pamf_no_signature_ablation').strip():raise ValueError('Commit adapter before publishing')
    context=json.loads((out/'context.json').read_text(encoding='utf-8'));project=context['project'];release=project+'/releases/'+RUN
    from experiments.practical_full_2x2.plan import VIEWS
    rt=dict(run_id=RUN,project=project,python=context['python'],frozen_sources=context['frozen_sources'],code_commit=commit,
            release_root=release,run_root=project+'/runs/'+RUN,runtime_file=release+'/runtime.json',context=release+'/context.json',
            allowed_gpus=[1,5,6,7],max_own_GPU_workers=4,max_own_per_GPU=1,max_total_per_GPU=2,
            launch_owner='codex-PAMF-noSIG-supervised-20261026',views=list(VIEWS))
    archive=out/(RUN+'.tar.gz')
    if archive.exists():raise FileExistsError('Reconcile prior submission; never resubmit blindly')
    tracked=set(git('ls-files','-z','--','experiments','code','baselines').decode().split('\0'))
    src=git('archive','--format=tar',commit,'--','experiments','code','baselines')
    with tarfile.open(fileobj=io.BytesIO(src)) as source,tarfile.open(archive,'w:gz') as dest:
        for m in source:
            if m.name not in tracked or not m.isfile() or Path(m.name).suffix not in ('.py','.json','.txt','.yaml','.yml','.toml'):continue
            dest.addfile(m,source.extractfile(m))
        for name,data in [('context.json',(out/'context.json').read_bytes()),('release_commit.txt',(commit+'\n').encode()),
                          ('experiment.json',git('show',commit+':automation_reports/CV-SincNet/'+RUN+'/experiment.json'))]:
            m=tarfile.TarInfo(name);m.size=len(data);dest.addfile(m,io.BytesIO(data))
    package=dict(runtime=rt,archive_remote=project+'/releases/'+archive.name,sha256=hashlib.sha256(archive.read_bytes()).hexdigest())
    for name,obj in [('runtime.json',rt),('package.json',package)]:
        (out/name).write_text(json.dumps(obj,indent=2)+'\n',encoding='utf-8')
    subprocess.run(['scp',*CONNECTION,str(archive),'N607:'+package['archive_remote']],check=True,timeout=60)
    result=ssh(REMOTE.replace('PACKAGE',repr(package)))
    (out/'submit.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--repo',required=True);p.add_argument('--output',required=True);a=p.parse_args();publish(a.repo,a.output)
