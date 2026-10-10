"""Committed immutable release; short SSH; independent process readback."""
import argparse
import hashlib
import io
import json
import subprocess
import tarfile
from pathlib import Path
from experiments.standard_ce_baselines.common import ROOT,RUN,RELEASE,PROJECT,read,write

CONNECTION=['-F','E:/type10-7/tools/n607_ssh_config','-o','BatchMode=yes','-o','ConnectTimeout=10']

def ssh(script):
    compile(script,'remote','exec')
    result=subprocess.run(['ssh',*CONNECTION,'-T','N607','python3 -'],input=script.encode('utf-8'),capture_output=True,timeout=60)
    if result.returncode:raise RuntimeError(result.stderr.decode('utf-8',errors='replace')+'\n'+result.stdout.decode('utf-8',errors='replace'))
    return result.stdout

def inspect(output):
    script=r'''
import json,subprocess,time
from pathlib import Path
project=Path(PROJECT);release=project/'releases'/RELEASE;run=project/'runs'/RUN
def read(p):return json.loads(p.read_text()) if p.exists() else None
def proc(pid):
    p=Path('/proc')/str(pid)
    try:return dict(pid=pid,cwd=str((p/'cwd').resolve()),argv=(p/'cmdline').read_bytes().decode().split('\0')) if p.exists() else None
    except OSError:return dict(pid=pid,status='UNKNOWN')
state=read(run/'pipeline_state.json');rows=[]
for rid,row in (state or {}).get('rows',{}).items():
    folder=Path(row['source_output']) if row.get('source_output') else None
    ef=folder/'epoch_metrics.jsonl' if folder else None
    log=Path(row['log']) if row.get('log') else None
    rows.append(dict(row_id=rid,status=row['status'],gpu=row.get('gpu'),process=proc(row.get('pid')),
        epoch=json.loads(ef.read_text().splitlines()[-1]) if ef and ef.exists() and ef.stat().st_size else None,
        resolved=read(folder/'resolved_config.json') if folder else None,
        log_bytes=log.stat().st_size if log and log.exists() else None,
        log_tail=log.read_text(errors='replace').splitlines()[-2:] if log and log.exists() else [],error=row.get('error')))
result=dict(read_at=time.time(),user=subprocess.check_output(['whoami'],text=True).strip(),host=subprocess.check_output(['hostname'],text=True).strip(),
    release_exists=release.exists(),run_exists=run.exists(),submit=read(release/'submit.json'),state=state,rows=rows,
    dispatcher=proc(state['pid']) if state else None,summary=read(run/'summary.json'),
    gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_memory','--format=csv,noheader'],text=True))
print(json.dumps(result))
'''
    script=script.replace('PROJECT',repr(PROJECT)).replace('RELEASE',repr(RELEASE)).replace('RUN',repr(RUN))
    value=json.loads(ssh(script));write(Path(output)/'readback.json',value)
    print(json.dumps(dict(status=value['state']['status'] if value['state'] else None,dispatcher=value['dispatcher'],
        rows=[dict(row_id=r['row_id'],status=r['status'],gpu=r['gpu'],alive=bool(r['process']),epoch=r['epoch']['epoch'] if r['epoch'] else None,log_bytes=r['log_bytes'],error=r['error']) for r in value['rows']]),ensure_ascii=False))
    return value

REMOTE=r'''
import hashlib,json,os,subprocess,tarfile
from pathlib import Path
c=CONFIG;project=Path(c['project']);release=project/'releases'/c['release'];run=project/'runs'/c['run'];logs=project/'logs'/c['run']
if subprocess.check_output(['whoami'],text=True).strip()!='szu2070436088' or subprocess.check_output(['hostname'],text=True).strip()!='dell-DSS8440':raise ValueError('Host/user mismatch')
if release.exists() or run.exists() or logs.exists():raise FileExistsError('Existing delivery: reconcile, never resubmit')
archive=project/'releases'/c['archive']
if hashlib.sha256(archive.read_bytes()).hexdigest()!=c['sha256']:raise ValueError('Transfer mismatch')
release.mkdir()
with tarfile.open(archive) as tar:
    for member in tar.getmembers():
        if not member.isfile() or not (release/member.name).resolve().is_relative_to(release.resolve()):raise ValueError('Unsafe archive member')
    tar.extractall(release)
python='/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python'
env=dict(os.environ,PYTHONPATH=str(release)+os.pathsep+str(release/'code'),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
subprocess.run([python,'-m','compileall','-q',str(release/'experiments'),str(release/'baselines'),str(release/'code'),str(release/'comparison_suite')],cwd=release,env=env,check=True)
spec=release/'experiments/standard_ce_baselines/configs/launch_spec.json';d=json.loads(spec.read_text())
if d['run_id']!=c['run'] or len(d['rows'])!=56:raise ValueError('Wrong matrix')
for row in d['rows']:
    config=json.loads(Path(row['source_config']).read_text())
    for key in ('dataset','source_contract'):
        if not Path(config[key]).is_file():raise FileNotFoundError(config[key])
    for key in ('source_output','prediction_output','score_output'):
        if Path(row[key]).exists():raise FileExistsError('Output collision')
if not (Path(d['capsule'])/'manifest.json').is_file() or not Path(d['truth']).is_file():raise FileNotFoundError('Test inputs')
if not (Path(d['capsule'])/'clean.npy').is_file():raise FileNotFoundError('Clean IQ')
# CPU synthetic real-checkpoint smoke, no source/target samples, one pass for all architectures.
with (release/'smoke.log').open('x') as log:
    subprocess.run([python,'-m','experiments.standard_ce_baselines.smoke','--output',str(release/'smoke.json')],cwd=release,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
command=[python,'-u','-m','experiments.standard_ce_baselines.dispatch','--spec',str(spec)]
logpath=release/'dispatcher.stdout.log'
with logpath.open('x') as log:
    child=subprocess.Popen(command,cwd=release,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
receipt=dict(status='SUBMITTED',pid=child.pid,cwd=str(release),argv=command,log=str(logpath),commit=c['commit'],sha256=c['sha256'])
(release/'submit.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))
'''


def publish(output):
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    branch=subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip()
    remote=subprocess.check_output(['git','ls-remote','origin','refs/heads/'+branch],cwd=ROOT,text=True).split()[0]
    if remote!=commit:raise ValueError('Remote branch differs from HEAD')
    folders=('standard_ce_baselines','cvs_sixscene_eval','cvs_selected_concat','cvs_energy_identity','cvs_equivariant_identity',
        'cvs_coordinate_identity','cvs_synchronized_identity','cvs_gauge_identity','cvs_rff_physics','cvs_reference_identity',
        'cvs_residual_identity','cvs_clean_design','cvs_identity_ce','adv3b02_xuc/code')
    prefixes=['experiments/'+f+'/' for f in folders]+['baselines/common/','baselines/cvcnn_ce/','code/leo_practical/']
    singles={'code/dataset_wisig.py','comparison_suite/score.py','comparison_suite/__init__.py','baselines/__init__.py'}
    names=subprocess.check_output(['git','ls-files'],cwd=ROOT,text=True).splitlines()
    selected=[n for n in names if (any(n.startswith(p) for p in prefixes) or n in singles) and Path(n).suffix in {'.py','.json'}]
    if subprocess.check_output(['git','status','--porcelain','--',*prefixes,*singles],cwd=ROOT,text=True).strip():raise ValueError('Uncommitted release changes')
    output=Path(output);output.mkdir(parents=True,exist_ok=True);archive=output/(RELEASE+'.tar.gz')
    if archive.exists():raise FileExistsError('Existing archive: inspect delivery before retry')
    # All selected files are clean and tracked at HEAD; avoid thousands of git subprocesses.
    with tarfile.open(archive,'w:gz') as tar:
        for name in selected:
            blob=(ROOT/name).read_bytes();member=tarfile.TarInfo(name);member.size=len(blob);member.mode=0o644;tar.addfile(member,io.BytesIO(blob))
        blob=(commit+'\n').encode();member=tarfile.TarInfo('release_commit.txt');member.size=len(blob);tar.addfile(member,io.BytesIO(blob))
    config=dict(project=PROJECT,release=RELEASE,run=RUN,archive=archive.name,sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),commit=commit)
    write(output/'package.json',dict(config,files=len(selected)))
    subprocess.run(['scp',*CONNECTION,str(archive),'N607:'+PROJECT+'/releases/'+archive.name],check=True,timeout=60)
    receipt=json.loads(ssh(REMOTE.replace('CONFIG',repr(config))));write(output/'submit.json',receipt);print(receipt)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--inspect',action='store_true');a=p.parse_args()
    inspect(a.output) if a.inspect else publish(a.output)
