"""Ordinary-account immutable release, GPU0-only diagnostic and compact readback."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile
from experiments.standard_ce_baselines.common import PROJECT,ROOT,write
from experiments.standard_ce_baselines.publish import CONNECTION,ssh
from experiments.standard_ce_deployment.benchmark import BENCH_RUN

RELEASE='standard_ce_deployment_20261012_r02'

REMOTE=r'''
import hashlib,json,os,subprocess,tarfile
from pathlib import Path
c=CONFIG;p=Path(c['project']);release=p/'releases'/c['release'];run=p/'runs'/c['run']
if subprocess.check_output(['whoami'],text=True).strip()!='szu2070436088' or subprocess.check_output(['hostname'],text=True).strip()!='dell-DSS8440':raise ValueError('Wrong ordinary account/host')
if release.exists() or run.exists():raise FileExistsError('Existing run; inspect instead of relaunching')
uuid=subprocess.check_output(['nvidia-smi','--id=0','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
if uuid in subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader'],text=True):raise RuntimeError('GPU0 occupied')
archive=p/'releases'/c['archive']
if hashlib.sha256(archive.read_bytes()).hexdigest()!=c['sha256']:raise ValueError('Archive transfer mismatch')
release.mkdir()
with tarfile.open(archive) as tar:
    for member in tar.getmembers():
        if not member.isfile() or not (release/member.name).resolve().is_relative_to(release.resolve()):raise ValueError('Unsafe archive path')
    tar.extractall(release)
python='/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python'
env=dict(os.environ,PYTHONPATH=str(release)+os.pathsep+str(release/'code'),CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
check=subprocess.run([python,'-c','import py_compile,numpy,torch;from pathlib import Path;assert Path("/proc/self/statm").is_file();py_compile.compile("experiments/standard_ce_deployment/benchmark.py",doraise=True);print(torch.__version__)'],cwd=release,env=env,capture_output=True,text=True)
if check.returncode:raise RuntimeError(check.stderr)
command=[python,'-m','experiments.standard_ce_deployment.benchmark','--output',str(run)]
with (release/'stdout.log').open('x') as log:
    child=subprocess.Popen(command,cwd=release,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
result=dict(status='SUBMITTED',pid=child.pid,cwd=str(release),argv=command,commit=c['commit'],archive_sha_match=True,compile='PASS',environment=check.stdout.strip())
(release/'submit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
'''

def publish(output):
    output=Path(output);output.mkdir(exist_ok=True,parents=True)
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    branch=subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip()
    if subprocess.check_output(['git','ls-remote','origin','refs/heads/'+branch],cwd=ROOT,text=True).split()[0]!=commit:raise ValueError('Upstream mismatch')
    folders=('standard_ce_deployment','standard_ce_baselines','cvs_sixscene_eval','cvs_selected_concat','cvs_energy_identity','cvs_equivariant_identity',
        'cvs_coordinate_identity','cvs_synchronized_identity','cvs_gauge_identity','cvs_rff_physics','cvs_reference_identity',
        'cvs_residual_identity','cvs_clean_design','cvs_identity_ce','adv3b02_xuc/code')
    prefixes=['experiments/'+f+'/' for f in folders]+['baselines/common/','baselines/cvcnn_ce/','code/leo_practical/']
    singles={'code/dataset_wisig.py','comparison_suite/score.py','comparison_suite/__init__.py','baselines/__init__.py'}
    files=[n for n in subprocess.check_output(['git','ls-files'],cwd=ROOT,text=True).splitlines() if (any(n.startswith(p) for p in prefixes) or n in singles) and Path(n).suffix in {'.py','.json'}]
    if subprocess.check_output(['git','status','--porcelain','--',*prefixes,*singles],cwd=ROOT,text=True).strip():raise ValueError('Uncommitted release files')
    archive=output/(RELEASE+'.tar.gz')
    if archive.exists():raise FileExistsError('Existing archive; reconcile first')
    with tarfile.open(archive,'w:gz') as tar:
        for name in files:
            blob=(ROOT/name).read_bytes();member=tarfile.TarInfo(name);member.size=len(blob);tar.addfile(member,io.BytesIO(blob))
        blob=(commit+'\n').encode();member=tarfile.TarInfo('release_commit.txt');member.size=len(blob);tar.addfile(member,io.BytesIO(blob))
    config=dict(project=PROJECT,release=RELEASE,run=BENCH_RUN,archive=archive.name,sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),commit=commit)
    write(output/'package.json',config)
    subprocess.run(['scp',*CONNECTION,str(archive),'N607:'+PROJECT+'/releases/'+archive.name],check=True,timeout=60)
    result=json.loads(ssh(REMOTE.replace('CONFIG',repr(config))));write(output/'submit.json',result);print(json.dumps(result))

def inspect(output):
    script='''
import json,subprocess
from pathlib import Path
p=Path(PROJECT);run=p/'runs'/RUN;release=p/'releases'/RELEASE
def read(f):return json.loads(f.read_text()) if f.exists() else None
def proc(pid):
    if not pid:return None
    p=Path('/proc')/str(pid)
    try:return dict(pid=pid,cwd=str((p/'cwd').resolve()),argv=(p/'cmdline').read_bytes().decode().split('\\0')) if p.exists() else None
    except OSError:return None
state=read(run/'pipeline_state.json');log=release/'stdout.log'
print(json.dumps(dict(state=state,dispatcher=proc((state or {}).get('pid')),worker=proc(((state or {}).get('current_row') or {}).get('pid')),
 gpu=subprocess.check_output(['nvidia-smi','--id=0','--query-gpu=utilization.gpu,memory.used','--format=csv,noheader'],text=True),
 log_bytes=log.stat().st_size if log.exists() else None,log_tail=log.read_text(errors='replace').splitlines()[-4:] if log.exists() else [],
 results=read(run/'results.json'))))
'''
    script=script.replace('PROJECT',repr(PROJECT)).replace('RUN',repr(BENCH_RUN)).replace('RELEASE',repr(RELEASE))
    result=json.loads(ssh(script));write(Path(output)/'readback.json',result)
    if result['results']:write(Path(output)/'results.json',result['results'])
    print(json.dumps({k:v for k,v in result.items() if k not in ('results','state')}|dict(status=(result['state'] or {}).get('status'),completed=(result['state'] or {}).get('completed'))))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--inspect',action='store_true');a=p.parse_args()
    inspect(a.output) if a.inspect else publish(a.output)
