"""Publish committed source-only CVS matrix; refuse duplicate landing or launch."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

RUN = '20260927-phase1-cvs-daot-rc4-practical-manysig-m5-r01'
RELEASE = 'cvs_rc4_matched_20260927_r01'
REMOTE = r'''
import hashlib,json,os,subprocess,sys,tarfile,time
from pathlib import Path
c=CONFIG
project=Path(c['project']);release=project/'releases'/c['release'];archive=project/'releases'/c['archive']
run=project/'runs'/c['run_id'];logs=project/'logs'/c['run_id']
if release.exists() or run.exists() or logs.exists():raise FileExistsError('Existing landing/run; reconcile before any retry')
if hashlib.sha256(archive.read_bytes()).hexdigest()!=c['sha256']:raise ValueError('Transfer mismatch')
with tarfile.open(archive) as tar:
    for member in tar.getmembers():
        if not (project/'releases'/member.name).resolve().is_relative_to(release.resolve()) or member.issym() or member.islnk():raise ValueError('Unsafe member')
    tar.extractall(project/'releases')
python='/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python'
env=dict(os.environ,PYTHONPATH=str(release/'code')+os.pathsep+str(release),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONUNBUFFERED='1')
subprocess.run([python,'-m','compileall','-q',str(release/'code')],cwd=release,env=env,check=True)
sys.path.insert(0,str(release/'code'))
from scripts.dispatch_xuc_full import available_gpu,occupancy
spec=json.loads((release/'configs/matched_20260927/experiment_spec.json').read_text())
source=Path(spec['data']['contract_ref']);dataset=Path(spec['data']['dataset'])
for path in [source,dataset]:
    if not path.is_file():raise FileNotFoundError(path)
if spec['run_id']!=c['run_id'] or len(spec['rows'])!=5:raise ValueError('Unexpected matrix')
run.mkdir();logs.mkdir()
(run/'release.json').write_text(json.dumps(c,indent=2))
active={};receipts=[]
gpu=available_gpu(active)
if gpu is None:raise RuntimeError('No legal GPU slot; no training launched')
env['CUDA_VISIBLE_DEVICES']=str(gpu)
first=release/'configs/matched_20260927/cvs-daot-rc4-s392005.json'
with (logs/'startup_smoke.log').open('x') as log:
    subprocess.run([python,'-u',str(release/'code/scripts/check_rc4_matched.py'),'--config',str(first),'--output',str(run/'startup_smoke'),'--device','cuda:0'],cwd=release,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
for row in spec['rows']:
    gpu=available_gpu(active)
    if gpu is None:raise RuntimeError('No legal GPU slot; preserve existing receipts')
    slots=occupancy(active)
    if len(slots[gpu]['pids'])>=2 or slots[gpu]['free_mb']<12000:raise RuntimeError('GPU capacity changed')
    env['CUDA_VISIBLE_DEVICES']=str(gpu)
    rowid=row['row_id'];config=release/'configs/matched_20260927'/(rowid+'.json')
    (run/('gpu_preflight_'+rowid+'.json')).write_text(json.dumps({g:{**v,'pids':sorted(v['pids'])} for g,v in slots.items()},indent=2))
    command=[python,'-u',str(release/'code/scripts/train_rc4_matched.py'),'--config',str(config),'--dataset',str(dataset),
        '--output',str(run/rowid),'--source-contract',str(source),'--run-id',c['run_id'],'--commit',c['commit']]
    log_path=logs/(rowid+'.train.log')
    with log_path.open('x') as log:
        child=subprocess.Popen(command,cwd=release,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    active[rowid]={'process':child,'gpu':gpu}
    receipt=dict(status='LAUNCH_SUBMITTED',pid=child.pid,gpu=gpu,argv=command,cwd=str(release),run_id=c['run_id'],row_id=rowid,
        run_root=str(run),output=str(run/rowid),log=str(log_path),commit=c['commit'],source_contract=str(source),timestamp=time.time())
    (run/('launch_'+rowid+'.json')).write_text(json.dumps(receipt,indent=2)+'\n')
    receipts.append(receipt)
    (run/'launch.json').write_text(json.dumps({'run_id':c['run_id'],'commit':c['commit'],'rows':receipts},indent=2))
    print(json.dumps({'launched':rowid,'pid':child.pid,'gpu':gpu}),flush=True)
print(json.dumps({'run_id':c['run_id'],'commit':c['commit'],'rows':receipts}))
'''


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();repo=Path(__file__).resolve().parents[3]
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
    branch=subprocess.check_output(['git','branch','--show-current'],cwd=repo,text=True).strip()
    if subprocess.check_output(['git','ls-remote','origin','refs/heads/'+branch],cwd=repo,text=True).split()[0]!=commit:
        raise ValueError('Remote branch differs from HEAD')
    if subprocess.check_output(['git','status','--porcelain','--','experiments/adv3b02_xuc'],cwd=repo,text=True).strip():
        raise ValueError('Uncommitted release files')
    a.output.mkdir(parents=True,exist_ok=True)
    archive=a.output/(RELEASE+'.tar')
    if archive.exists():raise FileExistsError('Existing archive: reconcile delivery before retry')
    subprocess.run(['git','archive','--format=tar','--prefix='+RELEASE+'/','--output='+str(archive),commit+':experiments/adv3b02_xuc'],cwd=repo,check=True)
    project='/home/szu2070436088/2510044040/CV-SincNet'
    c=dict(project=project,release=RELEASE,archive=archive.name,sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),run_id=RUN,commit=commit)
    connection=['-F','E:/type10-7/tools/n607_ssh_config','-o','BatchMode=yes','-o','ConnectTimeout=10']
    subprocess.run(['scp',*connection,str(archive),'N607:'+project+'/releases/'+archive.name],check=True)
    script=REMOTE.replace('CONFIG',repr(c))
    compile(script,'remote','exec')
    result=subprocess.run(['ssh',*connection,'-T','N607','python3 -'],input=script.encode('utf-8'),capture_output=True)
    (a.output/'landing.stdout').write_bytes(result.stdout);(a.output/'landing.stderr').write_bytes(result.stderr)
    print(result.stdout.decode());print(result.stderr.decode());result.check_returncode()
    (a.output/'launch.json').write_text(json.dumps(json.loads(result.stdout.decode().splitlines()[-1]),indent=2),encoding='utf-8')

if __name__=='__main__':main()
