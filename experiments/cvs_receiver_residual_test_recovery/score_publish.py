"""Publish committed CPU-only scorer; never regenerate predictions."""
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile
from experiments.cvs_receiver_residual_test_recovery import design as d
from experiments.cvs_receiver_residual_v2.publish import ssh,CONNECTION


REMOTE=r'''
import hashlib,json,os,subprocess,tarfile
from pathlib import Path
c=CONFIG;p=Path(c['project']);release=p/'releases'/c['release']
if subprocess.check_output(['whoami'],text=True).strip()!='szu2070436088' or subprocess.check_output(['hostname'],text=True).strip()!='dell-DSS8440':raise ValueError('Wrong host/user')
if release.exists() or any((p/'runs'/r['run_id']).exists() for r in c['runs']):raise FileExistsError('Existing scoring delivery: reconcile before resubmit')
archive=p/'releases'/c['archive']
if hashlib.sha256(archive.read_bytes()).hexdigest()!=c['sha256']:raise ValueError('Transfer differs')
release.mkdir()
with tarfile.open(archive) as tar:
    for member in tar.getmembers():
        if not member.isfile() or not (release/member.name).resolve().is_relative_to(release.resolve()):raise ValueError('Unsafe archive')
    tar.extractall(release)
python='/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python'
env=dict(os.environ,PYTHONPATH=str(release),CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
subprocess.run([python,'-m','compileall','-q',str(release/'experiments')],cwd=release,env=env,check=True)
subprocess.run([python,'-m','experiments.cvs_receiver_residual_test_recovery.score_smoke'],cwd=release,env=env,check=True)
command=[python,'-u','-m','experiments.cvs_receiver_residual_test_recovery.score_only']
with (release/'dispatcher.stdout.log').open('x') as log:
    child=subprocess.Popen(command,cwd=release,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
receipt=dict(status='SUBMITTED',pid=child.pid,cwd=str(release),argv=command,commit=c['commit'],training=False,predictions_reused=True,gpu=False)
(release/'submit.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt))
'''


def main():
    prefix='experiments/cvs_receiver_residual_test_recovery/'
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=d.ROOT,text=True).strip()
    branch=subprocess.check_output(['git','branch','--show-current'],cwd=d.ROOT,text=True).strip()
    remote=subprocess.check_output(['git','-c','http.sslBackend=openssl','-c','http.version=HTTP/1.1','ls-remote','origin','refs/heads/'+branch],cwd=d.ROOT,text=True).split()[0]
    if remote!=head or subprocess.check_output(['git','status','--porcelain','--',prefix],cwd=d.ROOT,text=True).strip():raise ValueError('Uncommitted/unpushed scorer')
    names=subprocess.check_output(['git','ls-files','--',prefix],cwd=d.ROOT,text=True).splitlines()
    output=d.ROOT/'local_artifacts'/d.SCORE_RELEASE;output.mkdir(exist_ok=True)
    archive=output/(d.SCORE_RELEASE+'.tar.gz')
    if archive.exists():raise FileExistsError('Existing transfer: reconcile first')
    with tarfile.open(archive,'w:gz') as tar:
        for name in names:
            if not name.endswith('.py'):continue
            blob=subprocess.check_output(['git','show',head+':'+name],cwd=d.ROOT)
            item=tarfile.TarInfo(name);item.size=len(blob);tar.addfile(item,io.BytesIO(blob))
        blob=(head+'\n').encode();item=tarfile.TarInfo('release_commit.txt');item.size=len(blob);tar.addfile(item,io.BytesIO(blob))
    config=dict(project=d.PROJECT,release=d.SCORE_RELEASE,commit=head,runs=d.SCORE_RUNS,
        archive=archive.name,sha256=hashlib.sha256(archive.read_bytes()).hexdigest())
    (output/'package.json').write_text(json.dumps(config,indent=2)+'\n',encoding='utf-8')
    subprocess.run(['scp',*CONNECTION,str(archive),'N607:'+d.PROJECT+'/releases/'+archive.name],check=True)
    response=ssh(REMOTE.replace('CONFIG',repr(config))).decode('utf-8')
    (output/'submit_stdout.txt').write_text(response,encoding='utf-8')
    receipt=json.loads(response.strip().splitlines()[-1])
    (output/'submit.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8');print(json.dumps(receipt))


if __name__=='__main__':main()
