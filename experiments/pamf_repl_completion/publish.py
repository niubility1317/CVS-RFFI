"""Publish one committed immutable PAMF release, reusing all existing DURA/data."""
import argparse,hashlib,io,json,subprocess,tarfile
from pathlib import Path
from .plan import RUN
from experiments.cric_dura_comparison.publish import ssh,CONNECTION
ROOT=Path(__file__).resolve().parents[2]

REMOTE=r'''
import hashlib,json,os,subprocess,tarfile
from pathlib import Path
p=PACKAGE;rt=p['runtime'];release=Path(rt['release_root'])
if subprocess.check_output(['whoami'],text=True).strip()!=rt['project'].split('/')[2]:raise ValueError('Ordinary account mismatch')
if release.exists() or Path(rt['run_root']).exists():raise FileExistsError('Reconcile existing release/run before retry')
for value in rt['frozen_sources'].values():
 if not Path(value).exists():raise FileNotFoundError(value)
if hashlib.sha256(Path(p['archive_remote']).read_bytes()).hexdigest()!=p['sha256']:raise ValueError('Archive transfer mismatch')
release.mkdir()
with tarfile.open(p['archive_remote']) as tar:
 for member in tar.getmembers():
  if not member.isfile() or release.resolve() not in (release/member.name).resolve().parents:raise ValueError('Invalid release member')
 tar.extractall(release)
Path(rt['runtime_file']).write_text(json.dumps(rt,indent=2)+'\n')
ctx=json.loads(Path(rt['context']).read_text())
for h in ctx['handles'].values():
 for k in ('source_cache','source_contract'):
  if not Path(h[k]).exists():raise FileNotFoundError(h[k])
for pair in ctx['caches'].values():
 for cache in pair.values():
  done=json.loads((Path(cache['path'])/'complete.json').read_text())
  if done['status']!='DURA_IQ_CACHE_COMPLETE' or done['truth_read']:raise ValueError('Existing restored cache incomplete')
env=dict(os.environ,PYTHONPATH=str(release),CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
subprocess.run([rt['python'],'-m','compileall','-q',str(release/'experiments/pamf_repl_completion')],cwd=release,env=env,check=True)
probe="import json;from experiments.pamf_repl_completion.runner import activate;c=json.load(open("+repr(rt['runtime_file'])+"));activate(c);from experiments.pamf_repl_completion.plan import make_plan;from experiments.pamf_repl_completion import source,prepare,evaluation,objectives,timing;assert len(make_plan()['rows'])==201;print('BOOTSTRAP_IMPORT_PASS')"
probe+=';from experiments.practical_full_2x2 import cache,pamf_model,augmentation;from experiments.cric_dura_comparison import pipeline;from experiments.pamf_ssl_research import hard_pl_worker;print("RUNTIME_DEPENDENCIES_PASS")'
check=subprocess.run([rt['python'],'-c',probe],cwd=release,env=env,capture_output=True,text=True)
if check.returncode:raise RuntimeError(check.stderr)
cmd=[rt['python'],'-u','-m','experiments.pamf_repl_completion.runner','dispatch','--config',rt['runtime_file']]
log=release/'controller.log'
with log.open('x') as stream:child=subprocess.Popen(cmd,cwd=release,env=env,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
r=dict(status='SUBMITTED_NOT_YET_VERIFIED',pid=child.pid,cwd=str(release),argv=cmd,log=str(log),execution_commit=rt['code_commit'],archive_transfer='VERIFIED',remote_compile='PASS',isolated_import='PASS')
(release/'submit.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))
'''

def publish(repo,output,release_id):
    repo=Path(repo);out=Path(output);out.mkdir(parents=True,exist_ok=True)
    def git(*a):return subprocess.check_output(['git','-C',str(repo),*a])
    commit=git('rev-parse','HEAD').decode().strip();branch=git('branch','--show-current').decode().strip()
    if git('ls-remote','origin','refs/heads/'+branch).decode().split()[0]!=commit:raise ValueError('Remote OID not verified')
    context=json.loads((out/'context.json').read_text(encoding='utf-8'));prior=context['runtime'];project=prior['project']
    release=project+'/releases/'+release_id;run=project+'/runs/'+RUN
    rt=dict(run_id=RUN,project=project,release_root=release,run_root=run,code_commit=commit,python=prior['python'],
        frozen_sources=prior['frozen_sources'],context=release+'/context.json',matrix=release+'/experiment.json',runtime_file=release+'/runtime.json',
        allowed_gpus=list(range(8)),max_per_GPU=3,max_total_GPU_compute=24,new_G=0,new_restored_IQ_cache=0,
        launch_owner='codex-pamf-repl-completion-20261024')
    archive=out/(release_id+'.tar.gz')
    if archive.exists():raise FileExistsError('Reconcile prior publication before retry')
    paths=git('ls-files','-z','--','experiments','code','baselines').decode().split('\0')
    paths={p for p in paths if p and Path(p).suffix in ('.py','.json','.txt','.yaml','.yml','.toml')}
    owned=[p for p in paths if p.startswith('experiments/pamf_repl_completion/')]
    if len(owned)<10 or git('status','--porcelain','--',*owned).strip():raise ValueError('Adapter must be committed and clean')
    source=git('archive','--format=tar',commit,'--','experiments','code','baselines')
    with tarfile.open(fileobj=io.BytesIO(source)) as prior_tar,tarfile.open(archive,'w:gz') as tar:
        for member in prior_tar:
            if member.name not in paths:continue
            value=prior_tar.extractfile(member).read();item=tarfile.TarInfo(member.name);item.size=len(value);item.mode=0o644;tar.addfile(item,io.BytesIO(value))
        blobs={'experiment.json':git('show',commit+':automation_reports/CV-SincNet/'+RUN+'/experiment.json'),
            'context.json':(out/'context.json').read_bytes(),'release_commit.txt':(commit+'\n').encode()}
        for name,value in blobs.items():
            item=tarfile.TarInfo(name);item.size=len(value);item.mode=0o644;tar.addfile(item,io.BytesIO(value))
    package=dict(runtime=rt,archive_remote=project+'/releases/'+archive.name,sha256=hashlib.sha256(archive.read_bytes()).hexdigest())
    for name,obj in [('runtime.json',rt),('package.json',package)]:
        (out/name).write_text(json.dumps(obj,indent=2)+'\n',encoding='utf-8')
    subprocess.run(['scp',*CONNECTION,str(archive),'N607:'+package['archive_remote']],check=True,timeout=60)
    result=ssh(REMOTE.replace('PACKAGE',repr(package)))
    (out/'submit.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps(result),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--repo',required=True);p.add_argument('--output',required=True);p.add_argument('--release-id',required=True);a=p.parse_args();publish(a.repo,a.output,a.release_id)
