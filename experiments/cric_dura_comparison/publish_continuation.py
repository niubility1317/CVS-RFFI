"""One committed continuation release, preserving every healthy source worker."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile
from .publish import ROOT,CONNECTION,ssh
from .plan import make_plan
from .continuation_setup import G_RUN

RELEASE='20261021-phase1-CRIC-DURA-continuation-r02'

REMOTE=r'''
import hashlib,json,os,subprocess,tarfile
from pathlib import Path
c=CONFIG;rt=c['runtime'];release=Path(rt['release_root'])
if release.exists() or Path(rt['run_root']).exists():raise FileExistsError('Reconcile existing continuation; never duplicate submit')
if subprocess.check_output(['whoami'],text=True).strip()!=c['expected_user']:raise ValueError('Ordinary account mismatch')
for k in ('source_runtime','source_state','original_G_checkpoint','original_R65_checkpoint'):
 if not Path(rt[k]).is_file():raise FileNotFoundError(rt[k])
gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True)
if hashlib.sha256(Path(c['archive_remote']).read_bytes()).hexdigest()!=c['sha256']:raise ValueError('Archive transfer mismatch')
release.mkdir()
with tarfile.open(c['archive_remote']) as tar:
 for m in tar.getmembers():
  if not m.isfile() or release.resolve() not in (release/m.name).resolve().parents:raise ValueError('Invalid release member')
 tar.extractall(release)
Path(rt['runtime_file']).write_text(json.dumps(rt,indent=2)+'\n')
env=dict(os.environ,PYTHONPATH=str(release),CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
subprocess.run([rt['python'],'-m','compileall','-q',str(release)],cwd=release,env=env,check=True)
command=[rt['python'],'-u','-m','experiments.cric_dura_comparison.start_continuation','--config',rt['runtime_file']]
with (release/'continuation.log').open('x') as stream:
 child=subprocess.Popen(command,cwd=release,env=env,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
receipt=dict(status='SUBMITTED_NOT_YET_VERIFIED',pid=child.pid,cwd=str(release),argv=command,log=str(release/'continuation.log'),commit=rt['code_commit'],preflight_GPU=gpu)
(release/'submit.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt))
'''


def publish(repo,output):
    repo=Path(repo);output=Path(output)
    def git(*a):return subprocess.check_output(['git','-C',str(repo),*a])
    commit=git('rev-parse','HEAD').decode().strip();branch=git('branch','--show-current').decode().strip()
    if git('ls-remote','origin','refs/heads/'+branch).decode().split()[0]!=commit:raise ValueError('Upstream OID differs')
    source=json.loads((output/'runtime.json').read_text(encoding='utf-8'));p=source['project']
    rt=dict(project=p,main_run=source['run_id'],release_root=p+'/releases/'+RELEASE,run_root=p+'/runs/'+G_RUN,
        source_release=source['release_root'],source_runtime=source['release_root']+'/runtime.json',
        source_state=source['run_root']+'/source_pipeline_state.json',source_run_root=source['run_root'],source_log_root=source['log_root'],source_commit=source['code_commit'],
        runtime_file=p+'/releases/'+RELEASE+'/runtime.json',control_root=p+'/runs/'+G_RUN+'/control-r01',
        jobs=p+'/runs/'+G_RUN+'/jobs.json',matrix=p+'/releases/'+RELEASE+'/matrix.json',python=source['python'],
        frozen_sources=source['frozen_sources'],allowed_gpus=list(range(8)),code_commit=commit,launch_owner='codex-root-cric-dura-continuation-20261021',
        original_G_action_cache=p+'/runs/20261019-phase1-restoration-learned-action-source-s2026092701-r07/cache',
        original_G_coarse_cache=p+'/runs/20261019-phase1-reception-restoration-selfsup-manysig-m6-r02/cache',
        original_G_checkpoint=p+'/runs/20261021-phase1-restoration-R65-four-route-s2026092701-m4-r69/stage2/native_map_prior/final.pt',
        original_R65_checkpoint=p+'/runs/20261021-phase1-restoration-R46-delayed-prune-s2026092701-m2-r65/prune_after10/native_map_prior/final.pt',
        external_release=p+'/releases/20261019-phase1-external-dg-concat-pl-m219-r03',external_run_root=p+'/runs/20261019-phase1-external-dg-concat-pl-m219-r03')
    paths=[q.relative_to(ROOT).as_posix() for q in (ROOT/'experiments/cric_dura_comparison').glob('*.py')]
    paths += [q.relative_to(ROOT).as_posix() for q in (ROOT/'experiments/cric_dura_comparison/vendor').rglob('*') if q.is_file() and (q.suffix in ('.py','.txt') or q.name=='LICENSE')]
    paths+=['experiments/phase1_external_dg_aug/__init__.py','experiments/phase1_external_dg_aug/augmentation.py']
    if git('status','--porcelain','--',*paths).strip():raise ValueError('Uncommitted release sources')
    archive=output/(RELEASE+'.tar.gz')
    if archive.exists():raise FileExistsError('Existing archive: reconcile before retry')
    with tarfile.open(archive,'w:gz') as tar:
        blobs={path:git('show',commit+':'+path) for path in paths}
        blobs['matrix.json']=(json.dumps(make_plan(),indent=2)+'\n').encode();blobs['release_commit.txt']=(commit+'\n').encode()
        for name,blob in blobs.items():
            item=tarfile.TarInfo(name);item.size=len(blob);item.mode=0o644;tar.addfile(item,io.BytesIO(blob))
    package=dict(runtime=rt,expected_user=p.split('/')[2],archive_remote=p+'/releases/'+archive.name,sha256=hashlib.sha256(archive.read_bytes()).hexdigest())
    for name,value in [('continuation_runtime.json',rt),('continuation_package.json',package)]:
        (output/name).write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')
    subprocess.run(['scp',*CONNECTION,str(archive),'N607:'+package['archive_remote']],check=True,timeout=60)
    receipt=ssh(REMOTE.replace('CONFIG',repr(package)))
    (output/'continuation_submit.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(receipt))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--repo',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();publish(a.repo,a.output)
