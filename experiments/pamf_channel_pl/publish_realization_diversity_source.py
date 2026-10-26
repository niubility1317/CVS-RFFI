"""Publish one immutable, source-only frozen F/G diagnostic on one available GPU."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile

from experiments.cric_dura_comparison.publish import ssh, CONNECTION
from .restored_source import validate_context
from .realization_diversity_source import RUN,RELEASE

REMOTE = r'''
import hashlib,json,os,shutil,subprocess,tarfile
from pathlib import Path
p=PACKAGE;rt=p['runtime'];release=Path(rt['release_root'])
if os.geteuid()==0 or subprocess.check_output(['whoami'],text=True).strip()!=rt['project'].split('/')[2]:
 raise ValueError('Ordinary account required')
if release.exists() or Path(rt['run_root']).exists():raise FileExistsError('Reconcile existing run before resubmission')
if shutil.disk_usage(rt['project']).free<2*1024**3:raise ValueError('Insufficient release/output space')
for path in rt['frozen_sources'].values():
 if not Path(path).is_dir():raise FileNotFoundError(path)
pending={}
for path in Path('/proc').iterdir():
 if not path.name.isdigit():continue
 try:
  if path.stat().st_uid!=os.getuid() or path.joinpath('stat').read_text().split()[2]=='Z':continue
  args=path.joinpath('cmdline').read_bytes()
  if b'python' not in args or b'experiments.' not in args:continue
  env=dict(x.split(b'=',1) for x in path.joinpath('environ').read_bytes().split(b'\0') if b'=' in x)
  value=env.get(b'CUDA_VISIBLE_DEVICES',b'').decode()
  if value in map(str,range(8)):pending[int(path.name)]=int(value)
 except (PermissionError,FileNotFoundError,ProcessLookupError):pass
available=[]
for candidate in range(8):
 apps=subprocess.check_output(['nvidia-smi','-i',str(candidate),'--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).splitlines()
 compute={int(x.strip()) for x in apps if x.strip().isdigit()}
 free=int(subprocess.check_output(['nvidia-smi','-i',str(candidate),'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
 load=len(compute|{pid for pid,g in pending.items() if g==candidate})
 if load<2 and free>=10000:available.append((load,-free,candidate))
if not available:raise ValueError('No GPU capacity;preserve other owners')
gpu=min(available)[2]
if hashlib.sha256(Path(p['archive_remote']).read_bytes()).hexdigest()!=p['sha256']:raise ValueError('Archive transfer mismatch')
release.mkdir()
with tarfile.open(p['archive_remote']) as archive:
 for m in archive.getmembers():
  if not m.isfile() or release.resolve() not in (release/m.name).resolve().parents:raise ValueError('Invalid release member')
 archive.extractall(release)
Path(rt['runtime_file']).write_text(json.dumps(rt,indent=2)+'\n')
env=dict(os.environ,PYTHONPATH=str(release),CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
subprocess.run([rt['python'],'-m','compileall','-q',str(release/'experiments/pamf_channel_pl')],cwd=release,env=env,check=True)
command=[rt['python'],'-u','-m',rt['entrypoint'],'dispatch','--config',rt['runtime_file']]
with (release/'controller.log').open('x') as stream:
 child=subprocess.Popen(command,cwd=release,env=env,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
value=dict(status='SUBMITTED_NOT_YET_VERIFIED',pid=child.pid,cwd=str(release),argv=command,log=str(release/'controller.log'),
 code_commit=rt['code_commit'],archive_transfer='VERIFIED',remote_compile='PASS',gpu=gpu)
(release/'submit.json').write_text(json.dumps(value,indent=2)+'\n');print(json.dumps(value))
'''


def publish(repo, output, context_path, release_id=RELEASE):
    if release_id!=RELEASE:raise ValueError('Registered immutable realization release required')
    repo=Path(repo);out=Path(output);out.mkdir(parents=True,exist_ok=True)
    def git(*args):return subprocess.check_output(['git','-C',str(repo),*args])
    commit=git('rev-parse','HEAD').decode().strip();branch=git('branch','--show-current').decode().strip()
    if git('ls-remote','origin','refs/heads/'+branch).decode().split()[0]!=commit:raise ValueError('Upstream OID differs')
    c=json.loads(Path(context_path).read_text(encoding='utf-8'));validate_context(c)
    run_id=RUN;entrypoint='experiments.pamf_channel_pl.realization_diversity_source'
    project=c['project'];release=project+'/releases/'+release_id
    rt=dict(run_id=run_id,project=project,python=c['python'],release_root=release,run_root=project+'/runs/'+run_id,
        context=release+'/context.json',runtime_file=release+'/runtime.json',code_commit=commit,
        frozen_sources=c['frozen_sources'],allowed_gpus=list(range(8)),launch_owner='codex-pamf-realizations-source-20261026',entrypoint=entrypoint)
    paths=git('ls-files','-z','--','experiments','code','baselines').decode().split('\0')
    paths={p for p in paths if p and Path(p).suffix in ('.py','.json','.txt','.yaml','.yml','.toml')}
    if git('status','--porcelain','--','experiments/pamf_channel_pl').strip():raise ValueError('Committed diagnostic required')
    archive=out/(release_id+'.tar.gz')
    if archive.exists():raise FileExistsError('Reconcile earlier publication before retry')
    data=git('archive','--format=tar',commit,'--','experiments','code','baselines')
    with tarfile.open(fileobj=io.BytesIO(data)) as previous,tarfile.open(archive,'w:gz') as tar:
        for member in previous:
            if member.name not in paths:continue
            value=previous.extractfile(member).read();item=tarfile.TarInfo(member.name);item.size=len(value);item.mode=0o644
            tar.addfile(item,io.BytesIO(value))
        blobs={'context.json':Path(context_path).read_bytes(),'release_commit.txt':(commit+'\n').encode(),
            'experiment.json':git('show',commit+':automation_reports/CV-SincNet/'+run_id+'/experiment.json')}
        for name,value in blobs.items():
            item=tarfile.TarInfo(name);item.size=len(value);item.mode=0o644;tar.addfile(item,io.BytesIO(value))
    package=dict(runtime=rt,archive_remote=project+'/releases/'+archive.name,sha256=hashlib.sha256(archive.read_bytes()).hexdigest())
    for name,value in [('runtime.json',rt),('package.json',package)]:
        (out/name).write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')
    subprocess.run(['scp',*CONNECTION,str(archive),'N607:'+package['archive_remote']],check=True,timeout=60)
    result=ssh(REMOTE.replace('PACKAGE',repr(package)))
    (out/'submit.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:result[k] for k in ('status','pid','code_commit','archive_transfer','remote_compile','gpu')}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--repo',required=True);p.add_argument('--output',required=True)
    p.add_argument('--context',required=True);p.add_argument('--release-id',default=RELEASE)
    a=p.parse_args();publish(a.repo,a.output,a.context,a.release_id)
