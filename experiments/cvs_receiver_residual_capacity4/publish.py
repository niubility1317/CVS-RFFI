"""Publish committed scheduling control and hand off only our dispatcher parents."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile
from experiments.cvs_receiver_residual_v2.publish import ssh, CONNECTION
from experiments.cvs_receiver_residual_v2 import design as d

RELEASE = 'cvs_receiver_residual_capacity4_20261008_r02'
RUNS = [dict(run='20261008-phase1-receiver-residual-manysig-m16-r01', package='cvs_receiver_residual',
    release='cvs_receiver_residual_20261008_r01', commit='3712e8e4f004f21b8ffdc74ce0459ad8a2020f3d'),
    dict(run=d.RUN, package='cvs_receiver_residual_v2', release=d.RELEASE,
    commit='8a88b3a2e369644ea1d746c5e02006218ee00df6')]

REMOTE = r'''
import hashlib,json,os,signal,subprocess,sys,tarfile,time
from pathlib import Path
c=CONFIG
project=Path(c['project']);release=project/'releases'/c['release']
if subprocess.check_output(['whoami'],text=True).strip()!='szu2070436088' or subprocess.check_output(['hostname'],text=True).strip()!='dell-DSS8440': raise ValueError('Host/user differs')
if release.exists(): raise FileExistsError('Control release exists: reconcile before retry')
archive=project/'releases'/c['archive']
if hashlib.sha256(archive.read_bytes()).hexdigest()!=c['sha256']: raise ValueError('Archive differs')
release.mkdir()
with tarfile.open(archive) as tar:
    for m in tar.getmembers():
        if not m.isfile() or not (release/m.name).resolve().is_relative_to(release.resolve()): raise ValueError('Unsafe member')
    tar.extractall(release)
sys.path.insert(0,str(release))
from experiments.cvs_receiver_residual_capacity4.control import process,running,read,write,lock
control=release/'experiments/cvs_receiver_residual_capacity4/control.py'
python='/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python'
subprocess.run([python,'-m','experiments.cvs_receiver_residual_capacity4.smoke'],cwd=release,check=True,stdout=subprocess.DEVNULL)
plans=[]
for r in c['runs']:
    base=project/'runs'/r['run'];oldrelease=project/'releases'/r['release']
    if (oldrelease/'release_commit.txt').read_text().strip()!=r['commit']: raise ValueError('Worker release commit differs')
    if (base/'failure.json').exists() or (base/'completion.json').exists(): raise ValueError('Terminal run: no scheduling intervention')
    prior=project/'releases/cvs_receiver_residual_capacity4_20261008_r01'
    before=read(prior/'handoff_before.json')
    snapshot=next(x for x in before['plans'] if x['config']['run']==r['run'])
    marker=base/'dispatcher_active.json'
    owner=read(marker) if marker.exists() else read(base/'dispatcher.json')
    actual=process(owner['pid'])
    if marker.exists():
        if owner['control_release']!=str(prior) or owner['control_commit']!='3c4c60535c0c03eeadda8bdefc24465fa3e393f8': raise ValueError('Unexpected prior control owner')
        if not actual or actual['start_ticks']!=owner['start_ticks'] or actual['cwd']!=str(oldrelease) or actual['argv']!=owner['argv']:
            raise ValueError('Prior capacity4 dispatcher identity differs')
    else:
        # A first handoff stopped R2 at the post-SIGTERM /proc read race.
        # Reconcile its independent preflight receipt, absent original PID,
        # absent active marker, and no terminal artifact before resuming.
        submit=read(oldrelease/'submit.json')
        if actual or running(owner['pid']) or owner['pid']!=submit['pid'] or owner['pid']!=snapshot['actual']['pid']:
            raise ValueError('Partial handoff old owner not independently absent')
        if (base/'capacity4_handoff.json').exists(): raise ValueError('Unexpected partial handoff marker')
    workers=[]
    launches={}
    for kind in ('source','predict'):
        p=base/('launch_'+kind+'.json');launches[kind]=read(p) if p.exists() else None
        for item in (launches[kind] or {}).get('rows',[]):
            state=process(item['pid'])
            if state and state['cwd']==str(oldrelease): workers.append(state)
    plans.append(dict(config=r,owner=owner,actual=actual,original_preflight=snapshot['actual'],workers_before=workers,launches_before=launches))
write(release/'handoff_before.json',dict(status='PREFLIGHT_VERIFIED',plans=plans,at=time.time(),authorization='用户：每张卡4个实验进程'))
receipts=[]
for plan in plans:
    r=plan['config'];base=project/'runs'/r['run'];oldrelease=project/'releases'/r['release'];old=plan['actual']
    with lock(base/'capacity4_handoff.lock',nonblocking=True):
        if old:
            current=process(old['pid'])
            if not current or current['start_ticks']!=old['start_ticks'] or current['argv']!=old['argv'] or current['cwd']!=old['cwd']:
                raise ValueError('Dispatcher changed during handoff preflight')
            # Signal exactly the scheduling parent PID, never its process group or workers.
            os.kill(old['pid'],signal.SIGTERM)
            for _ in range(50):
                if not running(old['pid']): break
                time.sleep(.1)
            if running(old['pid']): raise RuntimeError('Old dispatcher remains alive; no second owner launched')
        elif (base/'dispatcher_active.json').exists() or running(plan['owner']['pid']):
            raise ValueError('A new owner appeared during partial handoff reconciliation')
        workers_after=[process(w['pid']) for w in plan['workers_before']]
        for before,after in zip(plan['workers_before'],workers_after):
            if after and (before['start_ticks']!=after['start_ticks'] or before['cwd']!=after['cwd']): raise ValueError('Worker identity changed')
        handoff=dict(status='OLD_DISPATCHER_EXIT_VERIFIED',authorization='用户：每张卡4个实验进程',
            old_dispatcher=old or plan['original_preflight'],previous_capacity4_owner=plan['owner'] if old else None,
            partial_handoff_reconciled=old is None,workers_before=plan['workers_before'],workers_after=workers_after,
            training_workers_signaled=False,worker_release_commit=r['commit'],control_commit=c['commit'],at=time.time())
        write(base/'capacity4_handoff_v2.json',handoff)
        if not (base/'capacity4_handoff.json').exists():write(base/'capacity4_handoff.json',handoff)
        with lock(base/'capacity4_owner.lock',nonblocking=True):
            command=[python,'-u',str(control),'--release',str(oldrelease),'--package',r['package']]
            env=dict(os.environ,PYTHONPATH=str(oldrelease)+os.pathsep+str(oldrelease/'code'),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
            env.pop('CUDA_VISIBLE_DEVICES',None)
            logpath=release/(r['run']+'.dispatcher.stdout.log')
            with logpath.open('x') as log:
                child=subprocess.Popen(command,cwd=oldrelease,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            actual=process(child.pid)
            if not actual: raise RuntimeError('New owner absent; reconcile handoff')
            receipt=dict(status='SUBMITTED',pid=child.pid,start_ticks=actual['start_ticks'],cwd=str(oldrelease),argv=command,
                log=str(logpath),control_release=str(release),control_commit=c['commit'],worker_release_commit=r['commit'],
                run_id=r['run'],launch_owner=plan['owner']['launch_owner'],per_gpu_limit=4,max_active=16,at=time.time())
            write(base/'dispatcher_active.json',receipt);receipts.append(receipt)
write(release/'submit.json',dict(status='SUBMITTED',owners=receipts,control_commit=c['commit']))
print(json.dumps(dict(status='SUBMITTED',owners=receipts)))
'''


def publish(output):
    prefix = 'experiments/cvs_receiver_residual_capacity4/'
    head = subprocess.check_output(['git','rev-parse','HEAD'],cwd=d.ROOT,text=True).strip()
    branch = subprocess.check_output(['git','branch','--show-current'],cwd=d.ROOT,text=True).strip()
    remote = subprocess.check_output(['git','-c','http.sslBackend=openssl','-c','http.version=HTTP/1.1','ls-remote',
        'origin','refs/heads/'+branch],cwd=d.ROOT,text=True).split()[0]
    if remote != head: raise ValueError('Local/remote OID differs')
    if subprocess.check_output(['git','status','--porcelain','--',prefix],cwd=d.ROOT,text=True).strip():
        raise ValueError('Uncommitted control code')
    output.mkdir(parents=True,exist_ok=True);archive=output/(RELEASE+'.tar.gz')
    if archive.exists(): raise FileExistsError('Existing archive: reconcile before retry')
    names=subprocess.check_output(['git','ls-files',prefix],cwd=d.ROOT,text=True).splitlines()
    with tarfile.open(archive,'w:gz') as tar:
        for n in names:
            if not n.endswith('.py'): continue
            blob=subprocess.check_output(['git','show',head+':'+n],cwd=d.ROOT)
            item=tarfile.TarInfo(n);item.size=len(blob);tar.addfile(item,io.BytesIO(blob))
        blob=(head+'\n').encode();item=tarfile.TarInfo('release_commit.txt');item.size=len(blob);tar.addfile(item,io.BytesIO(blob))
    config=dict(project=d.PROJECT,release=RELEASE,runs=RUNS,commit=head,archive=archive.name,
        sha256=hashlib.sha256(archive.read_bytes()).hexdigest())
    (output/'package.json').write_text(json.dumps(config,indent=2)+'\n',encoding='utf-8')
    subprocess.run(['scp',*CONNECTION,str(archive),'N607:'+d.PROJECT+'/releases/'+archive.name],check=True)
    result=json.loads(ssh(REMOTE.replace('CONFIG',repr(config))))
    (output/'submit.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result))


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);publish(p.parse_args().output)
