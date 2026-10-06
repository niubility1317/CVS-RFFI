"""Publish a fresh recovery run and its seven-view evaluation dependency."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile
from experiments.cvs_phase1_stack.design import PROJECT,RUN,RELEASE,REUSED_R2_RUN
from experiments.cvs_phase1_stack.publish import ROOT,CONNECTION,ssh
from experiments.cvs_phase1_stack.sixscene_after import RUN as EVAL_RUN

PATHS=['experiments/'+name for name in ('cvs_phase1_stack','cvs_phase1_overlay','cvs_selected_concat','cvs_energy_identity','cvs_equivariant_identity','cvs_coordinate_identity','cvs_synchronized_identity','cvs_gauge_identity','cvs_rff_physics','cvs_reference_identity','cvs_residual_identity','cvs_clean_design','cvs_identity_ce','adv3b02_xuc/code')]
PATHS+=['experiments/adv3b02_xuc/configs/matched_20260927/cvs-daot-rc4-s2026092701.json','baselines/common','baselines/cvcnn_ce','baselines/__init__.py','code/leo_practical','code/dataset_wisig.py','comparison_suite/score.py','comparison_suite/__init__.py']

REMOTE=r'''
from pathlib import Path
import json,os,subprocess,tarfile,hashlib
c=CONFIG;p=Path(c['project']);release=p/'releases'/c['release'];old=p/'runs'/c['old_run']
if subprocess.check_output(['whoami'],text=True).strip()!='szu2070436088' or subprocess.check_output(['hostname'],text=True).strip()!='dell-DSS8440':raise ValueError('Wrong host/user')
for path in [release,p/'runs'/c['run'],p/'runs'/c['eval_run'],p/'logs'/c['run'],p/'logs'/c['eval_run']]:
    if path.exists():raise FileExistsError('Existing recovery path; reconcile before any retry: '+str(path))
failure=json.loads((old/'failure.json').read_text())
if failure['error']!="RuntimeError('Adopted process identity changed')":raise ValueError('Unexpected failure scope')
archive=p/'releases'/c['archive']
if hashlib.sha256(archive.read_bytes()).hexdigest()!=c['sha256']:raise ValueError('Transfer mismatch')
release.mkdir()
with tarfile.open(archive) as t:
    for m in t.getmembers():
        if not m.isfile() or not (release/m.name).resolve().is_relative_to(release.resolve()):raise ValueError('Unsafe archive')
    t.extractall(release)
python='/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python'
env=dict(os.environ,PYTHONPATH=str(release)+os.pathsep+str(release/'code'),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONUNBUFFERED='1')
subprocess.run([python,'-m','compileall','-q',str(release)],env=env,check=True)
preflight="from experiments.cvs_phase1_stack import design as d, recover as r, capacity16 as q; from pathlib import Path; import json; old=Path(d.PROJECT)/'runs'/d.REUSED_R2_RUN; launches=d.read(old/'launch_r2.json')['rows']; assert len(launches)==16; assert all(q.proc(x['pid']) is None for x in launches); assert q.proc(d.read(old/'dispatcher_active.json')['pid']) is None; assert not list(old.glob('launch_r[3-6].json')); records=r.reused_r2([d.config(x,['leo']) for x in d.rows() if x['stage']=='r2']); d.write(d.ROOT/'recovery_preflight.json',dict(status='VERIFIED',rows=records,target_read=False,new_training=False))"
with (release/'preflight.stdout.log').open('x') as log:
    subprocess.run([python,'-c',preflight],cwd=release,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    subprocess.run([python,str(release/'experiments/cvs_phase1_stack/speed_checks.py'),'--worker-root',str(release),'--output',str(release/'compatibility.json')],cwd=release,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
if __import__('shutil').disk_usage(p).free<100*1024**3:raise RuntimeError('Insufficient output space')
result=dict(status='SUBMITTED',commit=c['commit'],release=str(release),run=c['run'],eval_run=c['eval_run'],children=[])
for mode,module,args in [('source','experiments.cvs_phase1_stack.recover',[]),('evaluation','experiments.cvs_phase1_stack.sixscene_after',['--mode','dispatch'])]:
    cmd=[python,'-u','-m',module,*args];path=release/(mode+'.stdout.log')
    with path.open('x') as log:
        child=subprocess.Popen(cmd,cwd=release,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    result['children'].append(dict(mode=mode,pid=child.pid,cwd=str(release),argv=cmd,log=str(path)))
    with (release/('submit_'+mode+'.json')).open('x') as f:json.dump(result['children'][-1],f,indent=2)
with (release/'submit.json').open('x') as f:json.dump(result,f,indent=2)
print(json.dumps(result))
'''

def publish(output):
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    branch=subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip()
    remote=subprocess.check_output(['git','-c','http.sslBackend=openssl','-c','http.version=HTTP/1.1','ls-remote','origin','refs/heads/'+branch],cwd=ROOT,text=True).split()[0]
    if remote!=commit:raise ValueError('Remote branch differs')
    if subprocess.check_output(['git','status','--porcelain','--',*PATHS],cwd=ROOT,text=True).strip():raise ValueError('Uncommitted recovery code')
    output.mkdir(parents=True,exist_ok=True);archive=output/(RELEASE+'.tar.gz')
    if archive.exists():raise FileExistsError('Existing delivery; reconcile before retry')
    source=subprocess.check_output(['git','archive','--format=tar',commit,'--',*PATHS],cwd=ROOT)
    with tarfile.open(fileobj=io.BytesIO(source)) as src,tarfile.open(archive,'w:gz') as target:
        for item in src.getmembers():
            if item.isfile() and Path(item.name).suffix in {'.py','.json'}:target.addfile(item,src.extractfile(item))
        blob=(commit+'\n').encode();item=tarfile.TarInfo('release_commit.txt');item.size=len(blob);target.addfile(item,io.BytesIO(blob))
    cfg=dict(project=PROJECT,run=RUN,old_run=REUSED_R2_RUN,eval_run=EVAL_RUN,release=RELEASE,commit=commit,archive=archive.name,sha256=hashlib.sha256(archive.read_bytes()).hexdigest())
    (output/'package.json').write_text(json.dumps(cfg,indent=2)+'\n',encoding='utf-8')
    subprocess.run(['scp',*CONNECTION,str(archive),'N607:'+PROJECT+'/releases/'+archive.name],check=True)
    receipt=json.loads(ssh(REMOTE.replace('CONFIG',repr(cfg))))
    (output/'submit.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8');print(json.dumps(receipt))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();publish(a.output)
