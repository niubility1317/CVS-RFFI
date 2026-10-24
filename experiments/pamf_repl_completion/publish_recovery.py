"""Transfer one CPU-entry fix; retire only the blocked owner and adopt its workers."""
import argparse,hashlib,io,json,subprocess,tarfile
from pathlib import Path
from .plan import RUN
from experiments.cric_dura_comparison.publish import ssh,CONNECTION
ROOT=Path(__file__).resolve().parents[2]
RELEASE='20261025-phase1-PAMF-REPL-control-r03'

REMOTE=r'''
import hashlib,json,os,signal,subprocess,time,tarfile,sys
from pathlib import Path
p=PACKAGE;rt=p['runtime'];release=Path(rt['release_root']);oldrt=json.loads(Path(rt['recover_previous_runtime']).read_text());oldroot=Path(oldrt['run_root'])
read=lambda p:json.loads(Path(p).read_text())
if subprocess.check_output(['whoami'],text=True).strip()!=rt['project'].split('/')[2]:raise ValueError('Ordinary account mismatch')
if release.exists() or Path(rt['run_root']).exists() or (oldroot/'handoff.json').exists():raise FileExistsError('Reconcile existing recovery before retry')
if hashlib.sha256(Path(p['archive_remote']).read_bytes()).hexdigest()!=p['sha256']:raise ValueError('Archive transfer mismatch')
release.mkdir()
with tarfile.open(p['archive_remote']) as tar:
 for m in tar:
  if not m.isfile() or release.resolve() not in (release/m.name).resolve().parents:raise ValueError('Invalid archive path')
 tar.extractall(release)
Path(rt['runtime_file']).write_text(json.dumps(rt,indent=2)+'\n')
env=dict(os.environ,PYTHONPATH=str(release),CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
subprocess.run([rt['python'],'-m','compileall','-q',str(release/'experiments/pamf_repl_completion')],cwd=release,env=env,check=True)
probe="from unittest.mock import patch;from experiments.pamf_repl_completion import runner,recovery;from experiments.pamf_repl_completion import evaluation; c={'prediction_output':'frozen','output_root':'score'}\nwith patch.object(runner,'activate',side_effect=AssertionError('MODEL_BOOTSTRAP_FORBIDDEN')),patch.object(evaluation,'score',return_value='CPU_ONLY'): assert runner.worker(c,'score')=='CPU_ONLY'\nprint('CPU_SCORE_ENTRY_PASS')"
check=subprocess.run([rt['python'],'-c',probe],cwd=release,env=env,capture_output=True,text=True)
if check.returncode:raise RuntimeError(check.stderr)
s=read(oldroot/'pipeline_state.json');pid=s['pid'];proc=Path('/proc')/str(pid)
if s['code_commit']!=rt['previous_execution_commit'] or not s['systemic_stop_new_launches'] or s['failures']!={"KeyError: 'frozen_sources'":2}:raise ValueError('Blocked owner/fingerprint changed')
owner_alive=proc.exists() and proc.joinpath('stat').read_text().split()[2]!='Z'
if owner_alive:
 sys.path.insert(0,str(release))
 from experiments.pamf_repl_completion.recovery import validate_owner
 validate_owner(dict(cwd=str(proc.joinpath('cwd').resolve()),argv=[v.decode() for v in proc.joinpath('cmdline').read_bytes().split(b'\0') if v]),oldrt)
before={k:v for k,v in s['jobs'].items() if v['status']=='RUNNING'}
handoff=dict(status='PREPARED',previous_owner_pid=pid,previous_runtime=rt['recover_previous_runtime'],replacement_runtime=rt['runtime_file'],replacement_root=rt['run_root'],preserved_workers=before,reason='CPU scorer entry KeyError; no model failure or recipe change')
(release/'handoff_preflight.json').write_text(json.dumps(handoff,indent=2)+'\n')
if owner_alive:
 os.kill(pid,signal.SIGTERM)
 for _ in range(20):
  if not proc.exists() or proc.joinpath('stat').read_text().split()[2]=='Z':break
  time.sleep(.5)
 else:raise RuntimeError('Blocked controller did not retire; no duplicate owner launched')
cmd=[rt['python'],'-u','-m','experiments.pamf_repl_completion.runner','dispatch','--config',rt['runtime_file']]
with (release/'controller.log').open('x') as log:child=subprocess.Popen(cmd,cwd=release,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
result=dict(status='SUBMITTED_NOT_YET_VERIFIED',pid=child.pid,retired_owner_pid=pid,cwd=str(release),argv=cmd,execution_commit=rt['code_commit'],archive_transfer='VERIFIED',remote_compile='PASS',CPU_score_entry='PASS')
(release/'submit.json').write_text(json.dumps(result,indent=2)+'\n')
(oldroot/'handoff.json').write_text(json.dumps(dict(handoff,status='HANDED_OFF',replacement_owner_pid=child.pid),indent=2)+'\n')
print(json.dumps(result))
'''

def publish(repo,output):
    repo=Path(repo);out=Path(output);old=json.loads((out/'runtime.json').read_text(encoding='utf-8'))
    if old['release_root'].split('/')[-1]!='20261024-phase1-PAMF-REPL-control-r02':raise ValueError('Reconcile latest owner before recovery')
    def git(*a):return subprocess.check_output(['git','-C',str(repo),*a])
    commit=git('rev-parse','HEAD').decode().strip();branch=git('branch','--show-current').decode().strip()
    if git('ls-remote','origin','refs/heads/'+branch).decode().split()[0]!=commit:raise ValueError('Remote OID not verified')
    project=old['project'];release=project+'/releases/'+RELEASE
    rt=dict(old,release_root=release,run_root=project+'/runs/'+RELEASE,runtime_file=release+'/runtime.json',context=release+'/context.json',matrix=release+'/experiment.json',
        code_commit=commit,recover_previous_runtime=old['runtime_file'],previous_execution_commit=old['code_commit'])
    archive=out/(RELEASE+'.tar.gz')
    if archive.exists():raise FileExistsError('Reconcile possibly published recovery')
    paths={p for p in git('ls-files','-z','--','experiments','code','baselines').decode().split('\0') if p and Path(p).suffix in ('.py','.json','.txt','.yaml','.yml','.toml')}
    owned=[p for p in paths if p.startswith('experiments/pamf_repl_completion/')]
    if git('status','--porcelain','--',*owned).strip():raise ValueError('Uncommitted recovery')
    source=git('archive','--format=tar',commit,'--','experiments','code','baselines')
    with tarfile.open(fileobj=io.BytesIO(source)) as src,tarfile.open(archive,'w:gz') as tar:
        for m in src:
            if m.name not in paths:continue
            value=src.extractfile(m).read();item=tarfile.TarInfo(m.name);item.size=len(value);item.mode=0o644;tar.addfile(item,io.BytesIO(value))
        blobs={'experiment.json':git('show',commit+':automation_reports/CV-SincNet/'+RUN+'/experiment.json'),'context.json':(out/'context.json').read_bytes(),'release_commit.txt':(commit+'\n').encode()}
        for name,value in blobs.items():
            item=tarfile.TarInfo(name);item.size=len(value);item.mode=0o644;tar.addfile(item,io.BytesIO(value))
    package=dict(runtime=rt,archive_remote=project+'/releases/'+archive.name,sha256=hashlib.sha256(archive.read_bytes()).hexdigest())
    (out/'recovery_package.json').write_text(json.dumps(package,indent=2)+'\n',encoding='utf-8')
    subprocess.run(['scp',*CONNECTION,str(archive),'N607:'+package['archive_remote']],check=True,timeout=60)
    result=ssh(REMOTE.replace('PACKAGE',repr(package)))
    for name,obj in [('runtime_r02.json',old),('runtime.json',rt),('recovery_submit.json',result)]:
        (out/name).write_text(json.dumps(obj,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--repo',required=True);p.add_argument('--output',required=True);a=p.parse_args();publish(a.repo,a.output)
