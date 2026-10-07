"""One-shot delivery of an immutable local commit after recorded GitHub HTTP500 failures."""
from pathlib import Path
import sys,subprocess,json,hashlib,tarfile,io
wt=Path('E:/type10-7/code/snapshots/daot_practical_three_20260918_wt');sys.path.insert(0,str(wt))
from experiments.cvs_phase1_stack.publish_completed_20261007 import REMOTE,PATHS,ROOT,CONNECTION,ssh,RUN,RELEASE,PROJECT
commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=wt,text=True).strip()
assert commit=='fadf4be41eb60e956d18530bd291c69ca435facc'
remote=subprocess.check_output(['git','-c','http.sslBackend=openssl','-c','http.version=HTTP/1.1','ls-remote','origin','refs/heads/codex/daot-practical-three-20260918'],cwd=wt,text=True).split()[0]
assert remote=='68c8e8d54b2f50908b1e019b51a61e7948299b55'
subprocess.run(['git','merge-base','--is-ancestor',remote,commit],cwd=wt,check=True)
assert not subprocess.check_output(['git','status','--porcelain','--',*PATHS],cwd=wt,text=True).strip()
output=wt/'local_artifacts'/RELEASE;archive=output/(RELEASE+'.tar.gz');assert not archive.exists()
blob=subprocess.check_output(['git','archive','--format=tar',commit,'--',*PATHS],cwd=wt)
with tarfile.open(fileobj=io.BytesIO(blob)) as src,tarfile.open(archive,'w:gz') as dst:
    for item in src.getmembers():
        if item.isfile() and Path(item.name).suffix in {'.py','.json'}:dst.addfile(item,src.extractfile(item))
    snapshot=subprocess.check_output(['git','show',commit+':experiments/cvs_phase1_stack/completed_snapshot_20261007.json'],cwd=wt)
    item=tarfile.TarInfo('frozen_rows.json');item.size=len(snapshot);dst.addfile(item,io.BytesIO(snapshot))
    content=(commit+'\n').encode();item=tarfile.TarInfo('release_commit.txt');item.size=len(content);dst.addfile(item,io.BytesIO(content))
cfg=dict(project=PROJECT,run=RUN,release=RELEASE,commit=commit,archive=archive.name,sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),git_delivery='FAILED_GITHUB_HTTP500',verified_remote_oid=remote,local_commit_fixed=True)
(output/'package.json').write_text(json.dumps(cfg,indent=2)+'\n',encoding='utf-8')
subprocess.run(['scp',*CONNECTION,str(archive),'N607:'+PROJECT+'/releases/'+archive.name],check=True)
receipt=json.loads(ssh(REMOTE.replace('CONFIG',repr(cfg))));(output/'submit.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8');print(json.dumps(receipt))
