"""Six matched single-view controls with frozen raw/G target evaluation and independent scoring."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile
from experiments.cric_dura_comparison.publish import ssh,CONNECTION
from .restored_source import validate_context
from .sig1_matched_baselines import RUN,RELEASE,PREFIX_RUN,SEEDS,TASKS

REMOTE=r'''
import hashlib,json,os,shutil,subprocess,tarfile
from pathlib import Path
p=PACKAGE;rt=p['runtime'];release=Path(rt['release_root']);project=Path(rt['project'])
if os.geteuid()==0 or subprocess.check_output(['whoami'],text=True).strip()!=str(project).split('/')[2]:raise ValueError('Ordinary account required')
if release.exists() or Path(rt['run_root']).exists():raise FileExistsError('Reconcile existing source run before resubmission')
if shutil.disk_usage(project).free<5*1024**3:raise ValueError('Insufficient source artifact space')
for path in p['required']:
 if not Path(path).exists():raise FileNotFoundError(path)
e=p['evaluation']
cache=json.loads((Path(e['restored_caches']['LU'])/'complete.json').read_text())
if cache['status']!='DURA_IQ_CACHE_COMPLETE' or cache['protocol_id']!='F0' or cache['truth_read'] or cache['G']!=e['same_G']:raise ValueError('Same original G-LU cache differs')
for task in ('cross_rx','cross_both'):
 manifest=json.loads((Path(e['tests'][task]['views_root'])/'manifest.json').read_text())
 if manifest['status']!='VALIDATED_ONCE' or manifest['count']!={'cross_rx':126000,'cross_both':42000}[task] or manifest['classes']!=e['classes']:raise ValueError('Validated F0 test handle differs')
pending={}
for path in Path('/proc').iterdir():
 if not path.name.isdigit():continue
 try:
  if path.stat().st_uid!=os.getuid() or path.joinpath('stat').read_text().split()[2]=='Z':continue
  args=path.joinpath('cmdline').read_bytes()
  if b'python' not in args or b'experiments.' not in args:continue
  env=dict(v.split(b'=',1) for v in path.joinpath('environ').read_bytes().split(b'\0') if b'=' in v)
  gpu=env.get(b'CUDA_VISIBLE_DEVICES',b'').decode()
  if gpu in map(str,range(8)):pending[int(path.name)]=int(gpu)
 except (PermissionError,FileNotFoundError,ProcessLookupError):pass
available=[]
for gpu in range(8):
 apps=subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).splitlines()
 compute={int(x.strip()) for x in apps if x.strip().isdigit()}
 free=int(subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
 if len(compute|{pid for pid,g in pending.items() if g==gpu})<2 and free>=10000:available.append(gpu)
if not available:raise ValueError('No capacity;preserve other owners')
if hashlib.sha256(Path(p['archive_remote']).read_bytes()).hexdigest()!=p['sha256']:raise ValueError('Archive transfer mismatch')
release.mkdir()
with tarfile.open(p['archive_remote']) as archive:
 for m in archive.getmembers():
  if not m.isfile() or release.resolve() not in (release/m.name).resolve().parents:raise ValueError('Invalid release member')
 archive.extractall(release)
Path(rt['runtime_file']).write_text(json.dumps(rt,indent=2)+'\n')
env=dict(os.environ,PYTHONPATH=str(release),CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
subprocess.run([rt['python'],'-m','compileall','-q',str(release/'experiments/pamf_channel_pl')],cwd=release,env=env,check=True)
command=[rt['python'],'-u','-m','experiments.pamf_channel_pl.sig1_matched_baselines','dispatch','--config',rt['runtime_file']]
with (release/'controller.log').open('x') as f:
 child=subprocess.Popen(command,cwd=release,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
result=dict(status='SUBMITTED_NOT_YET_VERIFIED',pid=child.pid,cwd=str(release),argv=command,log=str(release/'controller.log'),
 code_commit=rt['code_commit'],archive_transfer='VERIFIED',remote_compile='PASS',available_gpus=available)
(release/'submit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
'''


def publish(repo,output,context,evaluation_context):
    repo=Path(repo);out=Path(output);out.mkdir(parents=True,exist_ok=True)
    def git(*args):return subprocess.check_output(['git','-C',str(repo),*args])
    commit=git('rev-parse','HEAD').decode().strip();branch=git('branch','--show-current').decode().strip()
    if git('ls-remote','origin','refs/heads/'+branch).decode().split()[0]!=commit:raise ValueError('Upstream differs')
    if git('status','--porcelain','--','experiments/pamf_channel_pl').strip():raise ValueError('Committed source implementation required')
    c=json.loads(Path(context).read_text(encoding='utf-8'));validate_context(c)
    e=json.loads(Path(evaluation_context).read_text(encoding='utf-8'))
    if set(e['tests'])!=set(TASKS) or e['same_G']!=c['G']:raise ValueError('Fixed F0 raw/G handles required')
    project=c['project'];release=project+'/releases/'+RELEASE
    rt=dict(run_id=RUN,project=project,python=c['python'],release_root=release,run_root=project+'/runs/'+RUN,
        context=release+'/context.json',evaluation_context=release+'/evaluation_context.json',runtime_file=release+'/runtime.json',code_commit=commit,
        frozen_sources=c['frozen_sources'],allowed_gpus=list(range(8)),launch_owner='codex-pamf-sig1-matched-baselines-20261026')
    archive=out/(RELEASE+'.tar.gz')
    if archive.exists():raise FileExistsError('Reconcile earlier publication before retry')
    names={p for p in git('ls-files','-z','--','experiments','code','baselines').decode().split('\0') if p and Path(p).suffix in ('.py','.json','.txt','.yaml','.yml','.toml')}
    data=git('archive','--format=tar',commit,'--','experiments','code','baselines')
    with tarfile.open(fileobj=io.BytesIO(data)) as source,tarfile.open(archive,'w:gz') as target:
        for member in source:
            if member.name not in names:continue
            value=source.extractfile(member).read();info=tarfile.TarInfo(member.name);info.size=len(value);info.mode=0o644
            target.addfile(info,io.BytesIO(value))
        for name,value in {'context.json':Path(context).read_bytes(),'evaluation_context.json':Path(evaluation_context).read_bytes(),'release_commit.txt':(commit+'\n').encode(),
            'experiment.json':git('show',commit+':automation_reports/CV-SincNet/'+RUN+'/experiment.json')}.items():
            info=tarfile.TarInfo(name);info.size=len(value);info.mode=0o644;target.addfile(info,io.BytesIO(value))
    required=[*c['frozen_sources'].values(),c['G']['checkpoint'],c['G']['source_contract']]
    for seed in SEEDS:
        folder=project+'/runs/'+PREFIX_RUN+'/prefixes/F0-A1B1-SIG_TIME-F21-s'+str(seed)+'/segment20'
        required.extend([folder+'/complete.json',folder+'/final_ssdg.pth'])
        required.append(next(v['expected_source_contract'] for v in c['cases'] if v['model_seed']==seed))
    required.append(e['restored_caches']['LU']+'/complete.json')
    for test in e['tests'].values():required.extend([test['views_root']+'/manifest.json',test['truth']])
    package=dict(runtime=rt,required=required,evaluation=e,archive_remote=project+'/releases/'+archive.name,sha256=hashlib.sha256(archive.read_bytes()).hexdigest())
    for name,value in [('runtime.json',rt),('package.json',package)]:
        (out/name).write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')
    subprocess.run(['scp',*CONNECTION,str(archive),'N607:'+package['archive_remote']],check=True,timeout=60)
    result=ssh(REMOTE.replace('PACKAGE',repr(package)))
    (out/'submit.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:result[k] for k in ('status','pid','code_commit','archive_transfer','remote_compile','available_gpus')}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--repo',required=True);p.add_argument('--output',required=True);p.add_argument('--context',required=True);p.add_argument('--evaluation-context',required=True)
    a=p.parse_args();publish(a.repo,a.output,a.context,a.evaluation_context)
