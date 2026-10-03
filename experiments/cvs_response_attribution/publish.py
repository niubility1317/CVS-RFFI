"""Publish a committed source-only diagnosis and independently inspect its state."""
import argparse,hashlib,io,json,subprocess,tarfile
from pathlib import Path
from experiments.cvs_channel_response_identity.publish import ssh,CONNECTION
from experiments.cvs_channel_response_identity.prepare import PROJECT,RUN as SOURCE_RUN

ROOT=Path(__file__).resolve().parents[2]
RUN='20261003-diagnostic-cvs-response-attribution-source-manysig-m12-r01'
RELEASE='cvs_response_attribution_source_20261003_r01'
PREFIX='experiments/cvs_response_attribution/'
REMOTE='''
import hashlib,json,os,subprocess,tarfile,shutil
from pathlib import Path
c=CONFIG
p=Path(c['project']);release=p/'releases'/c['release'];run=p/'runs'/c['run'];logs=p/'logs'/c['run']
if subprocess.check_output(['whoami'],text=True).strip()!='szu2070436088' or subprocess.check_output(['hostname'],text=True).strip()!='dell-DSS8440':raise ValueError('Wrong remote identity')
if release.exists() or run.exists() or logs.exists():raise FileExistsError('Reconcile prior delivery; no relaunch')
source=p/'runs'/c['source_run'];state=json.loads((source/'pipeline_state.json').read_text())
if not state['status'].startswith('SOURCE_RESEARCH_COMPLETE') or len(state['rows'])!=12 or any(r['status']!='SOURCE_TRAINED' for r in state['rows'].values()):raise ValueError('Source is not fully frozen')
selection=json.loads((source/'source_selection.json').read_text())
if selection['target_access'] or selection['target_score_used']:raise ValueError('Invalid source selection')
gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True)
apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_memory','--format=csv,noheader'],text=True)
disk_free=shutil.disk_usage(p).free
if disk_free<2*1024**3:raise ValueError('Insufficient diagnostic output disk space')
archive=p/'releases'/c['archive']
if hashlib.sha256(archive.read_bytes()).hexdigest()!=c['sha256']:raise ValueError('Transfer mismatch')
release.mkdir()
with tarfile.open(archive) as tar:
    for member in tar.getmembers():
        if not member.isfile() or not (release/member.name).resolve().is_relative_to(release.resolve()):raise ValueError('Unsafe member')
    tar.extractall(release)
python='/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python'
subprocess.run([python,'-m','compileall','-q',str(release)],check=True)
env=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
command=[python,'-u',str(release/'dispatch.py'),'--spec',str(release/'launch_spec.json')]
logpath=release/'dispatcher.log'
with logpath.open('x') as log:
    child=subprocess.Popen(command,cwd=release,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
receipt=dict(status='SUBMITTED',pid=child.pid,cwd=str(release),argv=command,log=str(logpath),commit=c['commit'],sha256=c['sha256'],preflight_gpu=gpu,preflight_apps=apps,disk_free_bytes=disk_free)
(release/'submit.json').write_text(json.dumps(receipt,indent=2)+'\\n')
print(json.dumps(receipt))
'''

INSPECT='''
import json,subprocess,time
from pathlib import Path
c=CONFIG;p=Path(c['project']);run=p/'runs'/c['run'];release=p/'releases'/c['release']
def read(p):return json.loads(p.read_text()) if p.exists() else None
def proc(pid):
    p=Path('/proc')/str(pid)
    try:return dict(pid=pid,cwd=str((p/'cwd').resolve()),argv=(p/'cmdline').read_bytes().decode().split('\\0')) if p.exists() else None
    except OSError:return dict(pid=pid,status='UNKNOWN')
state=read(run/'pipeline_state.json');rows=[]
for rid,r in (state or {}).get('rows',{}).items():
    out=Path(r['output_root']);log=Path(r['log'])
    text=log.read_text(errors='replace') if log.exists() else ''
    rows.append(dict(row_id=rid,state=r,process=proc(r['pid']),resolved=read(out/'resolved_config.json'),completion=read(out/'completion.json'),log_bytes=len(text.encode()),conditions_done=text.count('CONDITION '),errors=[x for x in ('Traceback (most recent call last)','CUDA out of memory','ValueError:') if x in text.split('RESOLVED_CONFIG',1)[-1]]))
print(json.dumps(dict(read_at=time.time(),identity=dict(user=subprocess.check_output(['whoami'],text=True).strip(),host=subprocess.check_output(['hostname'],text=True).strip()),submit=read(release/'submit.json'),pipeline=state,dispatcher_process=proc(state['pid']) if state else None,rows=rows)))
'''

def inspect(output):
    data=json.loads(ssh(INSPECT.replace('c=CONFIG', 'c='+repr(dict(project=PROJECT,run=RUN,release=RELEASE)), 1)))
    output.mkdir(parents=True,exist_ok=True);(output/'readback.json').write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(status=data['pipeline']['status'] if data['pipeline'] else 'UNKNOWN',rows=[dict(row_id=r['row_id'],alive=bool(r['process']),conditions=r['conditions_done'],status=r['state']['status'],errors=r['errors']) for r in data['rows']])))
    return data

def publish(output):
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    branch=subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip()
    remote=subprocess.check_output(['git','-c','http.sslBackend=openssl','-c','http.version=HTTP/1.1','ls-remote','origin','refs/heads/'+branch],cwd=ROOT,text=True).split()[0]
    if remote!=commit:raise ValueError('Local/remote Git mismatch')
    if subprocess.check_output(['git','status','--porcelain','--',PREFIX],cwd=ROOT,text=True).strip():raise ValueError('Uncommitted diagnostic code')
    output.mkdir(parents=True,exist_ok=True);archive=output/(RELEASE+'.tar.gz')
    if archive.exists():raise FileExistsError('Existing delivery package: reconcile first')
    with tarfile.open(archive,'w:gz') as tar:
        for name in ('run.py','dispatch.py','launch_spec.json'):
            blob=subprocess.check_output(['git','show',commit+':'+PREFIX+name],cwd=ROOT)
            item=tarfile.TarInfo(name);item.size=len(blob);item.mode=0o644;tar.addfile(item,io.BytesIO(blob))
        blob=(commit+'\n').encode();item=tarfile.TarInfo('release_commit.txt');item.size=len(blob);tar.addfile(item,io.BytesIO(blob))
    cfg=dict(project=PROJECT,run=RUN,release=RELEASE,source_run=SOURCE_RUN,archive=archive.name,commit=commit,sha256=hashlib.sha256(archive.read_bytes()).hexdigest())
    (output/'package.json').write_text(json.dumps(cfg,indent=2)+'\n',encoding='utf-8')
    subprocess.run(['scp',*CONNECTION,str(archive),'N607:'+PROJECT+'/releases/'+archive.name],check=True)
    receipt=json.loads(ssh(REMOTE.replace('CONFIG',repr(cfg))))
    (output/'submit.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8');print(json.dumps(receipt))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--inspect',action='store_true');a=p.parse_args()
    if a.inspect:inspect(a.output)
    else:publish(a.output)
