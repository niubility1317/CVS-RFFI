"""Publish committed files once; read back the remote state independently."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile
from experiments.cvs_coherence_identity.prepare import RUN,RELEASE,PROJECT

ROOT=Path(__file__).resolve().parents[2]
CONNECTION=['-F','E:/type10-7/tools/n607_ssh_config','-o','BatchMode=yes','-o','ConnectTimeout=10']
REMOTE=r'''
import hashlib,json,os,subprocess,sys,tarfile
from pathlib import Path
c=CONFIG
project=Path(c['project']);release=project/'releases'/c['release']
run=project/'runs'/c['run'];logs=project/'logs'/c['run']
if release.exists() or run.exists() or logs.exists(): raise FileExistsError('Reconcile existing delivery; no resubmit')
archive=project/'releases'/c['archive']
if hashlib.sha256(archive.read_bytes()).hexdigest()!=c['sha256']: raise ValueError('Archive transfer mismatch')
release.mkdir()
with tarfile.open(archive) as tar:
    for member in tar.getmembers():
        if not (release/member.name).resolve().is_relative_to(release.resolve()) or not member.isfile():
            raise ValueError('Unsafe archive member')
    tar.extractall(release)
python='/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python'
env=dict(os.environ,PYTHONPATH=str(release)+os.pathsep+str(release/'code'),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
subprocess.run([python,'-m','compileall','-q',str(release/'experiments/cvs_coherence_identity'),str(release/'experiments/cvs_stability_identity'),str(release/'experiments/cvs_balanced_identity'),str(release/'experiments/cvs_residual_identity'),str(release/'experiments/cvs_clean_design'),str(release/'experiments/adv3b02_xuc/code'),str(release/'baselines/common'),str(release/'code')],cwd=release,env=env,check=True)
subprocess.run([python,'-m','experiments.cvs_coherence_identity.source','--help'],cwd=release,env=env,stdout=subprocess.DEVNULL,check=True)
spec=release/'experiments/cvs_coherence_identity/configs/launch_spec.json'
d=json.loads(spec.read_text())
if d['run_id']!=c['run'] or len(d['rows'])!=8: raise ValueError('Unexpected matrix')
for row in d['rows']:
    cfg=json.loads(Path(row['source_config']).read_text())
    for key in ('dataset','source_contract'):
        if not Path(cfg[key]).is_file(): raise FileNotFoundError(cfg[key])
smoke='import torch;from experiments.cvs_coherence_identity.model import VARIANTS,build;torch.set_num_threads(2);losses=[torch.nn.functional.cross_entropy(build(v)(torch.randn(2,2,256)),torch.tensor([0,1])) for v in VARIANTS];assert all(torch.isfinite(v) for v in losses);[v.backward() for v in losses]'
subprocess.run([python,'-c',smoke],cwd=release,env=env,stdout=subprocess.DEVNULL,check=True)
command=[python,'-u','-m','experiments.cvs_coherence_identity.dispatch','--spec',str(spec)]
logpath=release/'dispatcher.stdout.log'
with logpath.open('x') as log:
    child=subprocess.Popen(command,cwd=release,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
receipt=dict(status='SUBMITTED',pid=child.pid,cwd=str(release),argv=command,log=str(logpath),commit=c['commit'],sha256=c['sha256'])
(release/'submit.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt))
'''

INSPECT=r'''
import json,subprocess,time
from pathlib import Path
p=Path(PROJECT);run=p/'runs'/RUN;release=p/'releases'/RELEASE
def read(path):return json.loads(path.read_text()) if path.exists() else None
state=read(run/'pipeline_state.json')
result=dict(read_at=time.time(),run_id=RUN,identity=dict(user=subprocess.check_output(['whoami'],text=True).strip(),host=subprocess.check_output(['hostname'],text=True).strip()),submit=read(release/'submit.json'),pipeline=state,selection=read(run/'source_selection.json'),rows=[])
def proc(pid):
    q=Path('/proc')/str(pid)
    try:return dict(pid=pid,cwd=str((q/'cwd').resolve()),argv=(q/'cmdline').read_bytes().decode().split('\0')) if q.exists() else None
    except OSError:return dict(pid=pid,state='UNKNOWN')
result['dispatcher_process']=proc(state['pid']) if state else None
for rid,row in (state or {}).get('rows',{}).items():
    folder=Path(row['source_output']) if row.get('source_output') else None
    ef=folder/'epoch_metrics.jsonl' if folder else None
    epoch=json.loads(ef.read_text().splitlines()[-1]) if ef and ef.exists() and ef.stat().st_size else None
    log=Path(row['log']) if row.get('log') else None
    result['rows'].append(dict(row_id=rid,status=row['status'],gpu=row.get('gpu'),process=proc(row.get('pid')),epoch=epoch,
        resolved=read(folder/'resolved_config.json') if folder else None,completion=read(folder/'completion.json') if folder else None,
        profile=read(folder/'resource_profile.json') if folder else None,log_bytes=log.stat().st_size if log and log.exists() else None,
        log_tail=log.read_text(errors='replace').splitlines()[-2:] if log and log.exists() else []))
result['gpu']=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_memory','--format=csv,noheader'],text=True)
print(json.dumps(result))
'''


def ssh(script):
    compile(script,'remote','exec')
    result=subprocess.run(['ssh',*CONNECTION,'-T','N607','python3 -'],input=script.encode('utf-8'),capture_output=True)
    if result.returncode: raise RuntimeError(result.stderr.decode('utf-8',errors='replace')+'\n'+result.stdout.decode('utf-8',errors='replace'))
    return result.stdout


def inspect(output):
    script=INSPECT.replace('PROJECT',repr(PROJECT)).replace('RELEASE',repr(RELEASE)).replace('RUN',repr(RUN))
    data=json.loads(ssh(script))
    output.mkdir(parents=True,exist_ok=True)
    (output/'readback.json').write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(run_id=RUN,status=data['pipeline']['status'] if data['pipeline'] else None,rows=[dict(row_id=r['row_id'],status=r['status'],gpu=r['gpu'],alive=bool(r['process']),epoch=r['epoch']['epoch'] if r['epoch'] else None,log_bytes=r['log_bytes']) for r in data['rows']]),ensure_ascii=False))



def publish(output):
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    branch=subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip()
    remote=subprocess.check_output(['git','-c','http.version=HTTP/1.1','ls-remote','origin','refs/heads/'+branch],cwd=ROOT,text=True,timeout=30).split()[0]
    if remote!=commit: raise ValueError('Remote branch differs from HEAD')
    names=subprocess.check_output(['git','ls-files'],cwd=ROOT,text=True).splitlines()
    selected=[n for n in names if (n.startswith('experiments/cvs_coherence_identity/') or n.startswith('experiments/cvs_stability_identity/') or n.startswith('experiments/cvs_balanced_identity/') or n.startswith('experiments/cvs_residual_identity/') or n.startswith('experiments/cvs_clean_design/') or n.startswith('experiments/cvs_identity_ce/') or n.startswith('baselines/cvcnn_ce/') or
        n.startswith('experiments/adv3b02_xuc/code/') or n.startswith('baselines/common/') or
        n.startswith('code/leo_practical/') or n in {'code/dataset_wisig.py','comparison_suite/score.py','comparison_suite/__init__.py','baselines/__init__.py'})
        and Path(n).suffix in {'.py','.json'}]
    dirty=subprocess.check_output(['git','status','--porcelain','--','experiments/cvs_coherence_identity/','experiments/cvs_stability_identity/','experiments/cvs_balanced_identity/','experiments/cvs_residual_identity/','experiments/cvs_clean_design/','experiments/cvs_identity_ce/','experiments/adv3b02_xuc/code/','baselines/common/','baselines/cvcnn_ce/','code/leo_practical/','code/dataset_wisig.py','comparison_suite/score.py','comparison_suite/__init__.py','baselines/__init__.py'],cwd=ROOT,text=True)
    if dirty.strip(): raise ValueError('Uncommitted release code/config')
    output.mkdir(parents=True,exist_ok=True);archive=output/(RELEASE+'.tar.gz')
    if archive.exists(): raise FileExistsError('Existing archive; reconcile delivery before retry')
    with tarfile.open(archive,'w:gz') as tar:
        for name in selected:
            blob=subprocess.check_output(['git','show',commit+':'+name],cwd=ROOT)
            item=tarfile.TarInfo(name);item.size=len(blob);item.mode=0o644;tar.addfile(item,io.BytesIO(blob))
        blob=(commit+'\n').encode();item=tarfile.TarInfo('release_commit.txt');item.size=len(blob);tar.addfile(item,io.BytesIO(blob))
    sha=hashlib.sha256(archive.read_bytes()).hexdigest()
    cfg=dict(project=PROJECT,release=RELEASE,run=RUN,archive=archive.name,sha256=sha,commit=commit)
    (output/'package.json').write_text(json.dumps(dict(cfg,files=len(selected)),indent=2),encoding='utf-8')
    subprocess.run(['scp',*CONNECTION,str(archive),'N607:'+PROJECT+'/releases/'+archive.name],check=True)
    receipt=json.loads(ssh(REMOTE.replace('CONFIG',repr(cfg))))
    (output/'submit.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(receipt))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--inspect',action='store_true')
    a=p.parse_args()
    if a.inspect:inspect(a.output)
    else:publish(a.output)
