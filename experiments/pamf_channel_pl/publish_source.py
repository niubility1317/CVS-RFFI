"""One committed archive and transfer check for the immutable source diagnostic."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile
from experiments.cric_dura_comparison.publish import ssh,CONNECTION
from .plan import SOURCE_RUN,validate_context

REMOTE=r'''
import hashlib,json,os,shutil,subprocess,tarfile
from pathlib import Path
p=PACKAGE;rt=p['runtime'];release=Path(rt['release_root'])
if subprocess.check_output(['whoami'],text=True).strip()!=rt['project'].split('/')[2]:raise ValueError('Ordinary account mismatch')
if release.exists() or Path(rt['run_root']).exists():raise FileExistsError('Reconcile existing source run before resubmission')
if shutil.disk_usage(rt['project']).free<2*1024**3:raise ValueError('Insufficient release/output space')
for path in rt['frozen_sources'].values():
 if not Path(path).is_dir():raise FileNotFoundError(path)
if hashlib.sha256(Path(p['archive_remote']).read_bytes()).hexdigest()!=p['sha256']:raise ValueError('Archive transfer mismatch')
release.mkdir()
with tarfile.open(p['archive_remote']) as archive:
 for m in archive.getmembers():
  if not m.isfile() or release.resolve() not in (release/m.name).resolve().parents:raise ValueError('Invalid release member')
 archive.extractall(release)
Path(rt['runtime_file']).write_text(json.dumps(rt,indent=2)+'\n')
env=dict(os.environ,PYTHONPATH=str(release),CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
subprocess.run([rt['python'],'-m','compileall','-q',str(release.joinpath(*rt['entrypoint'].split('.')[:-1]))],cwd=release,env=env,check=True)
probe='import json,importlib;from experiments.pamf_repl_completion.runner import activate;activate(json.load(open('+repr(rt['runtime_file'])+')));importlib.import_module('+repr(rt['entrypoint'])+');from experiments.cric_dura_comparison.frozen_inference import reader_from_payload;print("ENTRYPOINT_IMPORT_PASS")'
result=subprocess.run([rt['python'],'-c',probe],cwd=release,env=env,capture_output=True,text=True)
if result.returncode:raise RuntimeError(result.stderr)
command=[rt['python'],'-u','-m',rt['entrypoint'],'dispatch','--config',rt['runtime_file']]
log=release/'controller.log'
with log.open('x') as stream:
 child=subprocess.Popen(command,cwd=release,env=env,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
value=dict(status='SUBMITTED_NOT_YET_VERIFIED',pid=child.pid,cwd=str(release),argv=command,log=str(log),
 code_commit=rt['code_commit'],archive_transfer='VERIFIED',remote_compile='PASS',isolated_import='PASS')
(release/'submit.json').write_text(json.dumps(value,indent=2)+'\n');print(json.dumps(value))
'''


def publish(repo,output,context_path,release_id,*,run_id=SOURCE_RUN,
            entrypoint='experiments.pamf_channel_pl.source_runner',launch_owner='codex-channel-pl-source-20261025',
            allowed_gpus=(5,6,7)):
    repo=Path(repo);out=Path(output);out.mkdir(parents=True,exist_ok=True)
    def git(*a):return subprocess.check_output(['git','-C',str(repo),*a])
    commit=git('rev-parse','HEAD').decode().strip();branch=git('branch','--show-current').decode().strip()
    if git('ls-remote','origin','refs/heads/'+branch).decode().split()[0]!=commit:raise ValueError('Remote OID not verified')
    context=json.loads(Path(context_path).read_text(encoding='utf-8'));validate_context(context)
    project=context['project'];release=project+'/releases/'+release_id
    rt=dict(run_id=run_id,project=project,python=context['python'],release_root=release,
        run_root=project+'/runs/'+run_id,context=release+'/context.json',runtime_file=release+'/runtime.json',
        code_commit=commit,frozen_sources=context['frozen_sources'],allowed_gpus=list(allowed_gpus),
        launch_owner=launch_owner,entrypoint=entrypoint)
    paths=git('ls-files','-z','--','experiments','code','baselines').decode().split('\0')
    paths={p for p in paths if p and Path(p).suffix in ('.py','.json','.txt','.yaml','.yml','.toml')}
    owned=[p for p in paths if p.startswith('experiments/pamf_channel_pl/')]
    if len(owned)<7 or git('status','--porcelain','--',*owned).strip():raise ValueError('Committed clean adapter required')
    archive=out/(release_id+'.tar.gz')
    if archive.exists():raise FileExistsError('Reconcile previous archive/publication before retry')
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
    print(json.dumps(dict(status=result['status'],pid=result['pid'],code_commit=commit,
        archive_transfer=result['archive_transfer'],remote_compile=result['remote_compile'])))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--repo',required=True);p.add_argument('--output',required=True)
    p.add_argument('--context',required=True);p.add_argument('--release-id',required=True);a=p.parse_args()
    publish(a.repo,a.output,a.context,a.release_id)
