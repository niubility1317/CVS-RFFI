"""Immutable CPU-only source diagnostic release with independent readback."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile
from experiments.cvs_residual_identity.publish import ssh,CONNECTION

ROOT=Path(__file__).resolve().parents[2]
PROJECT='/home/szu2070436088/2510044040/CV-SincNet'
RUN='20261002-diagnostic-cvs-cfo-information-source-manysig-m1-r01'
RELEASE='cvs_cfo_information_source_20261002_r01'
PYTHON='/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python'
CONFIG='experiments/cvs_cfo_information/configs/source.json'
PREFIXES=['experiments/cvs_cfo_information/','experiments/cvs_identity_ce/','experiments/cvs_residual_identity/',
          'experiments/cvs_clean_design/','experiments/adv3b02_xuc/code/','baselines/common/',
          'baselines/cvcnn_ce/','code/leo_practical/']

REMOTE=r'''
import hashlib,json,os,subprocess,tarfile
from pathlib import Path
c=CONFIG
project=Path(c['project']);release=project/'releases'/c['release']
run=project/'runs'/c['run'];logs=project/'logs'/c['run']
if any(p.exists() for p in (release,run,logs)):raise FileExistsError('Reconcile existing release;do not resubmit')
if subprocess.check_output(['whoami'],text=True).strip()!='szu2070436088':raise ValueError('Identity mismatch')
if subprocess.check_output(['hostname'],text=True).strip()!='dell-DSS8440':raise ValueError('Host mismatch')
archive=project/'releases'/c['archive']
if hashlib.sha256(archive.read_bytes()).hexdigest()!=c['sha256']:raise ValueError('Transfer mismatch')
release.mkdir()
with tarfile.open(archive) as tar:
    for m in tar.getmembers():
        if not m.isfile() or not (release/m.name).resolve().is_relative_to(release.resolve()):raise ValueError('Unsafe archive')
    tar.extractall(release)
python=c['python'];cfgpath=release/c['config'];cfg=json.loads(cfgpath.read_text())
for key in ('dataset','source_contract'):
    if not Path(cfg[key]).is_file():raise FileNotFoundError(cfg[key])
if cfg['output_root']!=str(run/'source'):raise ValueError('Output mismatch')
env=dict(os.environ,PYTHONPATH=str(release)+os.pathsep+str(release/'code'),CUDA_VISIBLE_DEVICES='',
         OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
subprocess.run([python,'-m','compileall','-q',str(release)],cwd=release,env=env,check=True)
subprocess.run([python,'-m','experiments.cvs_cfo_information.audit','--help'],cwd=release,env=env,check=True,stdout=subprocess.DEVNULL)
smoke='import numpy as np;from experiments.cvs_cfo_information.audit import packet_features;n=np.arange(256);z=np.exp(2j*np.pi*n/20+2j*np.pi*130000*n/25000000);f,t=packet_features(np.stack([z.real,z.imag])[None]);assert abs(t["hz"][0,1]-130000)<1e-6;assert f["cfo80"].shape==(1,2)'
subprocess.run([python,'-c',smoke],cwd=release,env=env,check=True,stdout=subprocess.DEVNULL)
run.mkdir();logs.mkdir();logpath=logs/'source.log'
command=[python,'-u','-m','experiments.cvs_cfo_information.audit','--config',str(cfgpath)]
with logpath.open('x') as f:
    child=subprocess.Popen(command,cwd=release,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
receipt=dict(status='SUBMITTED',pid=child.pid,cwd=str(release),argv=command,log=str(logpath),commit=c['commit'],cpu_only=True)
(release/'submit.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt))
'''

INSPECT=r'''
import json,time
from pathlib import Path
c=CONFIG
project=Path(c['project']);release=project/'releases'/c['release'];out=project/'runs'/c['run']/'source'
def read(p):return json.loads(p.read_text()) if p.exists() else None
receipt=read(release/'submit.json');proc=Path('/proc')/str(receipt['pid']) if receipt else None
live=None
if proc and proc.exists():
    try:live=dict(pid=receipt['pid'],cwd=str((proc/'cwd').resolve()),argv=(proc/'cmdline').read_bytes().decode().split('\0'))
    except OSError:live=dict(state='UNKNOWN')
log=Path(receipt['log']) if receipt else None
print(json.dumps(dict(read_at=time.time(),submit=receipt,process=live,resolved=read(out/'resolved_config.json'),
                     completion=read(out/'completion.json'),statistics=read(out/'information_statistics.json'),
                     log_bytes=log.stat().st_size if log and log.exists() else None,
                     log_text=log.read_text(errors='replace') if log and log.exists() else None)))
'''

def inspect(output):
    d=json.loads(ssh(INSPECT.replace('CONFIG',repr(dict(project=PROJECT,release=RELEASE,run=RUN)))))
    output.mkdir(parents=True,exist_ok=True)
    (output/'readback.json').write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(status='VERIFIED' if d['completion'] else 'OBSERVED',alive=bool(d['process']),
                         completion=d['completion'],log_bytes=d['log_bytes'])))
    return d


def publish(output):
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    branch=subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip()
    remote=subprocess.check_output(['git','-c','http.sslBackend=openssl','-c','http.version=HTTP/1.1','ls-remote','origin','refs/heads/'+branch],cwd=ROOT,text=True,timeout=30).split()[0]
    if remote!=commit:raise ValueError('Remote/local commit mismatch')
    dirty=subprocess.check_output(['git','status','--porcelain','--',*PREFIXES,'code/dataset_wisig.py'],cwd=ROOT,text=True)
    if dirty.strip():raise ValueError('Uncommitted diagnostic inputs')
    names=subprocess.check_output(['git','ls-files'],cwd=ROOT,text=True).splitlines()
    names=[n for n in names if (any(n.startswith(p) for p in PREFIXES) or n in {'code/dataset_wisig.py','baselines/__init__.py'}) and Path(n).suffix in {'.py','.json'}]
    output.mkdir(parents=True,exist_ok=True);archive=output/(RELEASE+'.tar.gz')
    if archive.exists():raise FileExistsError('Existing package;reconcile delivery')
    with tarfile.open(archive,'w:gz') as tar:
        for name in names:
            blob=subprocess.check_output(['git','show',commit+':'+name],cwd=ROOT)
            m=tarfile.TarInfo(name);m.size=len(blob);m.mode=0o644;tar.addfile(m,io.BytesIO(blob))
        blob=(commit+'\n').encode();m=tarfile.TarInfo('release_commit.txt');m.size=len(blob);tar.addfile(m,io.BytesIO(blob))
    c=dict(project=PROJECT,run=RUN,release=RELEASE,archive=archive.name,sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
           commit=commit,python=PYTHON,config=CONFIG)
    (output/'package.json').write_text(json.dumps(c,indent=2)+'\n',encoding='utf-8')
    subprocess.run(['scp',*CONNECTION,str(archive),'N607:'+PROJECT+'/releases/'+archive.name],check=True)
    receipt=json.loads(ssh(REMOTE.replace('CONFIG',repr(c))))
    (output/'submit.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(receipt))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);parser.add_argument('--inspect',action='store_true')
    args=parser.parse_args();inspect(args.output) if args.inspect else publish(args.output)
