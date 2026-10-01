"""Publish committed files once; read back the remote state independently."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile
from experiments.cvs_identity_ce.prepare import RUN,RELEASE,PROJECT

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
subprocess.run([python,'-m','compileall','-q',str(release/'experiments/cvs_identity_ce'),str(release/'experiments/adv3b02_xuc/code'),str(release/'baselines/common'),str(release/'code')],cwd=release,env=env,check=True)
subprocess.run([python,'-m','experiments.cvs_identity_ce.source','--help'],cwd=release,env=env,stdout=subprocess.DEVNULL,check=True)
spec=release/'experiments/cvs_identity_ce/configs/launch_spec.json'
d=json.loads(spec.read_text())
if d['run_id']!=c['run'] or len(d['rows'])!=5: raise ValueError('Unexpected matrix')
for row in d['rows']:
    cfg=json.loads(Path(row['source_config']).read_text())
    for key in ('dataset','source_contract'):
        if not Path(cfg[key]).is_file(): raise FileNotFoundError(cfg[key])
    pred=json.loads(Path(row['config']).read_text())
    if not (Path(pred['p1_capsule'])/'manifest.json').is_file(): raise FileNotFoundError(pred['p1_capsule'])
if not Path(d['p1_truth']).is_file(): raise FileNotFoundError(d['p1_truth'])
command=[python,'-u','-m','experiments.cvs_identity_ce.dispatch','--spec',str(spec)]
logpath=release/'dispatcher.stdout.log'
with logpath.open('x') as log:
    child=subprocess.Popen(command,cwd=release,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
receipt=dict(status='SUBMITTED',pid=child.pid,cwd=str(release),argv=command,log=str(logpath),commit=c['commit'],sha256=c['sha256'])
(release/'submit.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt))
'''

INSPECT=r'''
import json,os,subprocess,time
from pathlib import Path
p=Path(PROJECT); run=p/'runs'/RUN;release=p/'releases'/RELEASE
result=dict(read_at=time.time(),run_id=RUN,release_exists=release.exists(),run_exists=run.exists(),rows=[])
def read(path): return json.loads(path.read_text()) if path.exists() else None
result['submit']=read(release/'submit.json'); result['dispatcher']=read(run/'dispatcher.json')
result['completion']=read(run/'completion.json');result['failure']=read(run/'failure.json')
result['scoring']=read(run/'scoring_p1_complete.json')
launch=read(run/'launch.json')
def proc(pid):
    path=Path('/proc')/str(pid)
    if not path.exists():return None
    try:return dict(pid=pid,cwd=str((path/'cwd').resolve()),argv=(path/'cmdline').read_bytes().decode().split('\0'),status=(path/'status').read_text().splitlines()[:6])
    except (OSError,UnicodeError):return dict(pid=pid,state='UNKNOWN')
result['dispatcher_process']=proc(result['submit']['pid']) if result['submit'] else None
for row in (launch or {}).get('rows',[]):
    folder=run/row['row_id']; source=folder/'source';log=Path(row['log'])
    epochs=source/'epoch_metrics.jsonl'
    latest=json.loads(epochs.read_text().splitlines()[-1]) if epochs.exists() and epochs.stat().st_size else None
    children=[]
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit():continue
        try:
            args=(entry/'cmdline').read_bytes().decode().split('\0')
            if any(str(release) in a for a in args) and any('source-s'+str(row['row_id'].split('-s')[-1])+'.json' in a or 'predict-s'+str(row['row_id'].split('-s')[-1])+'.json' in a for a in args):
                children.append(proc(int(entry.name)))
        except (OSError,UnicodeError):continue
    initial=read(source/'initialization.json');resolved=read(source/'resolved_config.json')
    result['rows'].append(dict(**row,process=proc(row['pid']),children=children,initialization=initial,
        effective=({k:resolved.get(k) for k in ['pid','cwd','python','hardware','total_parameters','trainable_parameters',
            'source_counts','steps_per_epoch','domain_backbone','extra_losses','selection','model_seed','augmentation_seed','target_access','commit']} if resolved else None),
        epoch=latest,source_completion=read(source/'completion.json'),prediction_complete=read(folder/'prediction/phase1_complete.json'),
        log_bytes=log.stat().st_size if log.exists() else 0,log_tail=log.read_text(errors='replace').splitlines()[-2:] if log.exists() else []))
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
    print(json.dumps(dict(run_id=RUN,completion=data['completion'],failure=data['failure'],
        rows=[dict(row_id=r['row_id'],gpu=r['gpu'],alive=bool(r['process']),
            epoch=r['epoch']['epoch'] if r['epoch'] else None,source_complete=bool(r['source_completion']),
            prediction_complete=bool(r['prediction_complete']),log_bytes=r['log_bytes'],log_tail=r['log_tail']) for r in data['rows']]),ensure_ascii=False))


def publish(output):
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    branch=subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip()
    remote=subprocess.check_output(['git','ls-remote','origin','refs/heads/'+branch],cwd=ROOT,text=True).split()[0]
    if remote!=commit: raise ValueError('Remote branch differs from HEAD')
    names=subprocess.check_output(['git','ls-files'],cwd=ROOT,text=True).splitlines()
    selected=[n for n in names if (n.startswith('experiments/cvs_identity_ce/') or
        n.startswith('experiments/adv3b02_xuc/code/') or n.startswith('baselines/common/') or
        n.startswith('code/leo_practical/') or n in {'code/dataset_wisig.py','comparison_suite/score.py','comparison_suite/__init__.py','baselines/__init__.py'})
        and Path(n).suffix in {'.py','.json'}]
    dirty=subprocess.check_output(['git','status','--porcelain','--',*selected],cwd=ROOT,text=True)
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
