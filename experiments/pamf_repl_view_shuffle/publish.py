"""Publish exactly one committed archive to the ordinary N607 project account."""
import argparse,hashlib,io,json,subprocess,tarfile
from pathlib import Path
from experiments.cric_dura_comparison.publish import ssh,CONNECTION
from .plan import RUN,RELEASE

REMOTE=r'''
import hashlib,json,os,shutil,subprocess,tarfile
from pathlib import Path
p=PACKAGE;rt=p['runtime'];release=Path(rt['release_root']);root=Path(rt['run_root']);project=Path(rt['project'])
if os.geteuid()==0 or subprocess.check_output(['whoami'],text=True).strip()!=str(project).split('/')[2]:raise ValueError('Ordinary project account required')
if release.exists() or root.exists():raise FileExistsError('Reconcile existing release/run before retry')
if shutil.disk_usage(project).free<10*1024**3:raise ValueError('Insufficient artifact space')
if hashlib.sha256(Path(p['archive_remote']).read_bytes()).hexdigest()!=p['sha256']:raise ValueError('Archive transfer mismatch')
release.mkdir()
with tarfile.open(p['archive_remote']) as archive:
 for member in archive.getmembers():
  if not member.isfile() or release.resolve() not in (release/member.name).resolve().parents:raise ValueError('Invalid release member')
 archive.extractall(release)
Path(rt['runtime_file']).write_text(json.dumps(rt,indent=2)+'\n')
ctx=json.loads(Path(rt['context']).read_text())
h=ctx['handles']['F0']
for k in ('source_cache','source_contract'):
 if not Path(h[k]).exists():raise FileNotFoundError(h[k])
contract=json.loads(Path(h['source_contract']).read_text())
for old in list(ctx['historical_shuffle'].values())+[ctx['historical_single_smoke']]:
 d=json.loads((Path(old['source'])/'complete.json').read_text())
 if d['status']!='SOURCE_FROZEN' or d['target_access'] or d['U_truth_read'] or json.loads(Path(d['source_contract']).read_text())!=contract:raise ValueError('Historical source-only smoke input differs')
env=dict(os.environ,PYTHONPATH=str(release),CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
subprocess.run([rt['python'],'-m','compileall','-q',str(release/'experiments/pamf_repl_view_shuffle')],cwd=release,env=env,check=True)
cmd=[rt['python'],'-u','-m','experiments.pamf_repl_view_shuffle.runner','dispatch','--config',rt['runtime_file']]
log=release/'controller.log'
with log.open('x') as f:child=subprocess.Popen(cmd,cwd=release,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
result=dict(status='SUBMITTED_NOT_YET_VERIFIED',pid=child.pid,cwd=str(release),argv=cmd,log=str(log),code_commit=rt['code_commit'],archive_transfer='VERIFIED',remote_compile='PASS')
(release/'submit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
'''

def publish(repo,output,context):
    repo=Path(repo);out=Path(output);out.mkdir(parents=True,exist_ok=True)
    def git(*a):return subprocess.check_output(['git','-C',str(repo),*a])
    commit=git('rev-parse','HEAD').decode().strip();branch=git('branch','--show-current').decode().strip()
    if git('ls-remote','origin','refs/heads/'+branch).decode().split()[0]!=commit:raise ValueError('Upstream commit not verified')
    ctx=json.loads(Path(context).read_text());prior=ctx['runtime'];project=prior['project'];release=project+'/releases/'+RELEASE
    rt=dict(run_id=RUN,project=project,release_root=release,run_root=project+'/runs/'+RUN,context=release+'/context.json',runtime_file=release+'/runtime.json',
        python=prior['python'],code_commit=commit,frozen_sources=prior['frozen_sources'],allowed_gpus=list(range(8)),max_total_per_GPU=2,
        skip_foreign_user_cards=True,launch_owner='codex-pamf-repl-view-shuffle-20261027')
    archive=out/(RELEASE+'.tar.gz')
    if archive.exists():raise FileExistsError('Reconcile prior publication before retry')
    paths={p for p in git('ls-files','-z','--','experiments','code','baselines').decode().split('\0') if p and Path(p).suffix in ('.py','.json','.txt','.yaml','.yml','.toml')}
    source=git('archive','--format=tar',commit,'--','experiments','code','baselines')
    with tarfile.open(fileobj=io.BytesIO(source)) as old,tarfile.open(archive,'w:gz') as target:
        for member in old:
            if member.name not in paths:continue
            data=old.extractfile(member).read();item=tarfile.TarInfo(member.name);item.size=len(data);item.mode=0o644;target.addfile(item,io.BytesIO(data))
        for name,data in {'context.json':Path(context).read_bytes(),'release_commit.txt':(commit+'\n').encode(),
            'experiment.json':git('show',commit+':automation_reports/CV-SincNet/'+RUN+'/experiment.json')}.items():
            item=tarfile.TarInfo(name);item.size=len(data);item.mode=0o644;target.addfile(item,io.BytesIO(data))
    package=dict(runtime=rt,archive_remote=project+'/releases/'+archive.name,sha256=hashlib.sha256(archive.read_bytes()).hexdigest())
    for name,data in [('runtime.json',rt),('package.json',package)]:
        (out/name).write_text(json.dumps(data,indent=2)+'\n')
    subprocess.run(['scp',*CONNECTION,str(archive),'N607:'+package['archive_remote']],check=True,timeout=60)
    result=ssh(REMOTE.replace('PACKAGE',repr(package)));(out/'submit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('status','pid','code_commit','archive_transfer','remote_compile')}))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--repo',required=True);p.add_argument('--output',required=True);p.add_argument('--context',required=True);a=p.parse_args();publish(a.repo,a.output,a.context)
