"""Ordinary-account, one-archive release of the bounded source-only probe."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile

ROOT=Path(__file__).resolve().parents[2]
RUN='20261019-diagnostic-FcrossRX-sameIQ-experts-s2026092701-m2-r02'
RELEASE='cvs_FcrossRX_sameIQ_experts_20261019_r02'


def remote(code, runtime):
    args=['ssh','-F',runtime['ssh_config'],'-o','BatchMode=yes','-o','ConnectTimeout=10','-T','N607',runtime['python']+' -']
    r=subprocess.run(args,input=code.encode('utf8'),capture_output=True,timeout=60)
    if r.returncode:raise RuntimeError(r.stderr.decode('utf8',errors='replace'))
    return json.loads(r.stdout)


def main(runtime_path):
    runtime_path=Path(runtime_path);rt=json.loads(runtime_path.read_text(encoding='utf8'))
    if subprocess.check_output(['git','status','--porcelain','--','experiments/cvs_sameiq_experts'],cwd=ROOT,text=True).strip():raise ValueError('Commit actual experiment code first')
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    release=rt['project']+'/releases/'+RELEASE
    c=dict(rt,commit=commit,release=release,archive=release+'.tar.gz',run=rt['project']+'/runs/'+RUN)
    code='''
import json,os,shutil,subprocess
from pathlib import Path
c=CONFIG
if os.geteuid()==0 or subprocess.check_output(['whoami'],text=True).strip()!=c['user'] or subprocess.check_output(['hostname'],text=True).strip()!=c['host']:raise ValueError('Expected ordinary N607 account')
if any(Path(c[k]).exists() for k in ('release','archive','run')):raise FileExistsError('Immutable output exists; reconcile before resubmitting')
if shutil.disk_usage(c['project']).free<2*1024**3:raise ValueError('Insufficient output space')
gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader,nounits'],text=True).strip()
if gpu:raise ValueError('Bounded diagnostic expects idle GPUs; defer instead of interfering')
print(json.dumps(dict(preflight='PASS',GPU=0,existing_compute_processes=0)))
'''
    preflight=remote(code.replace('CONFIG',repr(c)),rt)
    raw=subprocess.check_output(['git','archive','--format=tar',commit,'experiments/cvs_sameiq_experts'],cwd=ROOT)
    buf=io.BytesIO()
    with tarfile.open(fileobj=buf,mode='w:gz') as out:
        with tarfile.open(fileobj=io.BytesIO(raw)) as src:
            for member in src.getmembers():
                if member.isfile():out.addfile(member,src.extractfile(member))
        data=(commit+'\n').encode();info=tarfile.TarInfo('release_commit.txt');info.size=len(data);out.addfile(info,io.BytesIO(data))
    archive=runtime_path.parent/(RELEASE+'.tar.gz');archive.write_bytes(buf.getvalue());c['sha256']=hashlib.sha256(buf.getvalue()).hexdigest()
    subprocess.run(['scp','-F',rt['ssh_config'],str(archive),'N607:'+c['archive']],check=True,capture_output=True,timeout=60)
    code='''
import hashlib,json,os,subprocess,tarfile
from pathlib import Path
c=CONFIG
if hashlib.sha256(Path(c['archive']).read_bytes()).hexdigest()!=c['sha256']:raise ValueError('Single archive transfer mismatch')
release=Path(c['release']);release.mkdir()
with tarfile.open(c['archive']) as t:t.extractall(release)
env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONPATH=str(release))
subprocess.run([c['python'],'-m','compileall','-q',str(release/'experiments/cvs_sameiq_experts')],env=env,cwd=release,check=True)
argv=[c['python'],'-u',str(release/'experiments/cvs_sameiq_experts/runner.py'),'--project',c['project'],'--device','cuda:0']
log=release/'stdout.log'
with log.open('x') as f:child=subprocess.Popen(argv,cwd=release,env=env,stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
result=dict(status='SUBMITTED',pid=child.pid,cwd=str(release),argv=argv,log=str(log),commit=c['commit'],GPU=0,archive_transfer='VERIFIED',remote_compile='VERIFIED')
(release/'submit.json').write_text(json.dumps(result,indent=2)+'\\n')
print(json.dumps(result))
'''
    result=remote(code.replace('CONFIG',repr(c)),rt)
    (runtime_path.parent/'submit.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
    print(json.dumps(dict(preflight=preflight,**result)))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--runtime',required=True)
    main(parser.parse_args().runtime)
