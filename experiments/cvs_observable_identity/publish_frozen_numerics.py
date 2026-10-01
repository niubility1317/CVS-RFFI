"""Single immutable supplemental diagnostic delivery, with independent readback."""
import argparse,hashlib,io,json,subprocess,tarfile
from pathlib import Path
from experiments.cvs_observable_identity.publish import ROOT,CONNECTION,ssh
from experiments.cvs_observable_identity.prepare import PROJECT,RUN
RELEASE='cvs_observable_numerics_20261002_r01'
SOURCE_COMMIT='202aed46748856c16b015d94595ecd5364dca394'
REMOTE=r'''
import hashlib,json,os,subprocess,tarfile
from pathlib import Path
c=CONFIG
project=Path(c['project']);release=project/'releases'/c['release'];source=project/'runs'/c['run'];out=source/'post_training_numerics'
if release.exists() or out.exists():raise FileExistsError('Reconcile existing diagnostic;no overwrite/resubmit')
state=json.loads((source/'pipeline_state.json').read_text())
if state['status']!='SOURCE_RESEARCH_COMPLETE_AWAITING_FROZEN_CLEAN_TEST':raise ValueError('Source training must be terminal')
archive=project/'releases'/c['archive']
if hashlib.sha256(archive.read_bytes()).hexdigest()!=c['sha256']:raise ValueError('Transfer mismatch')
release.mkdir()
with tarfile.open(archive) as tar:
    for member in tar.getmembers():
        if not member.isfile() or not (release/member.name).resolve().is_relative_to(release.resolve()):raise ValueError('Unsafe archive')
    tar.extractall(release)
python='/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python'
env=dict(os.environ,PYTHONPATH=str(release),CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
subprocess.run([python,'-m','compileall','-q',str(release/'experiments/cvs_observable_identity')],cwd=release,env=env,check=True)
command=[python,'-m','experiments.cvs_observable_identity.frozen_numerics','--source-root',str(source),'--expected-contract',str(source/'observable_phase-s2026092701/source/source_contract.json'),'--expected-commit',c['source_commit'],'--output',str(out),'--gpu']
with (release/'diagnostic.stdout.log').open('x') as log:
    subprocess.run(command,cwd=release,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
receipt=dict(status='DIAGNOSTIC_RETURNED',commit=c['commit'],source_commit=c['source_commit'],argv=command,cwd=str(release),output=str(out),log=str(release/'diagnostic.stdout.log'))
(release/'submit.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt))
'''


def inspect(output):
    script="import json\nfrom pathlib import Path\np=Path("+repr(PROJECT)+")\nr=p/'releases'/"+repr(RELEASE)+"\no=p/'runs'/"+repr(RUN)+"/'post_training_numerics'\ndef read(q):return json.loads(q.read_text()) if q.exists() else None\nprint(json.dumps(dict(receipt=read(r/'submit.json'),diagnostic=read(o/'precision_diagnostics.json'),log=(r/'diagnostic.stdout.log').read_text() if (r/'diagnostic.stdout.log').exists() else None)))"
    d=json.loads(ssh(script));output.mkdir(parents=True,exist_ok=True)
    (output/'readback.json').write_text(json.dumps(d,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(status='VERIFIED' if d['diagnostic'] and len(d['diagnostic']['rows'])==8 else 'UNKNOWN',rows=len(d['diagnostic']['rows']) if d['diagnostic'] else None)))


def publish(output):
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    branch=subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip()
    remote=subprocess.check_output(['git','-c','http.sslBackend=openssl','-c','http.version=HTTP/1.1','ls-remote','origin','refs/heads/'+branch],cwd=ROOT,text=True,timeout=30).split()[0]
    if remote!=commit:raise ValueError('Remote OID mismatch')
    files=['experiments/cvs_observable_identity/model.py','experiments/cvs_observable_identity/frozen_numerics.py']
    dirty=subprocess.check_output(['git','status','--porcelain','--',*files,'experiments/cvs_observable_identity/publish_frozen_numerics.py'],cwd=ROOT,text=True)
    if dirty.strip():raise ValueError('Uncommitted diagnostic code')
    output.mkdir(parents=True,exist_ok=True);archive=output/(RELEASE+'.tar.gz')
    if archive.exists():raise FileExistsError('Existing package;reconcile before resubmit')
    with tarfile.open(archive,'w:gz') as tar:
        for name in files+['release_commit.txt']:
            blob=(commit+'\n').encode() if name=='release_commit.txt' else subprocess.check_output(['git','show',commit+':'+name],cwd=ROOT)
            item=tarfile.TarInfo(name);item.size=len(blob);item.mode=0o644;tar.addfile(item,io.BytesIO(blob))
    cfg=dict(project=PROJECT,run=RUN,release=RELEASE,archive=archive.name,sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),commit=commit,source_commit=SOURCE_COMMIT)
    (output/'package.json').write_text(json.dumps(cfg,indent=2)+'\n',encoding='utf-8')
    subprocess.run(['scp',*CONNECTION,str(archive),'N607:'+PROJECT+'/releases/'+archive.name],check=True)
    receipt=json.loads(ssh(REMOTE.replace('CONFIG',repr(cfg))))
    (output/'submit.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8');print(json.dumps(receipt))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--inspect',action='store_true');a=p.parse_args()
    if a.inspect:inspect(a.output)
    else:publish(a.output)
