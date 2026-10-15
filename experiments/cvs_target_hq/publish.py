"""One immutable ordinary-account CPU release, followed by independent observation."""
import argparse,hashlib,io,json,subprocess,tarfile
from pathlib import Path
from . import design as d
from experiments.cvs_urban_sigtime.operations import remote,SSH
from experiments.cvs_urban_sigtime50.publish import REMOTE as ORIGINAL

REMOTE=ORIGINAL.replace('experiments/cvs_urban_sigtime50','experiments/cvs_target_hq').replace(
    'experiments.cvs_urban_sigtime50.dispatch','experiments.cvs_target_hq.predict')

def publish(output):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=d.ROOT,text=True).strip()
    dirty=subprocess.check_output(['git','status','--porcelain','--','experiments','code','baselines','comparison_suite'],cwd=d.ROOT,text=True)
    if dirty.strip():raise ValueError('Commit this isolated release code/config first')
    names=subprocess.check_output(['git','ls-files'],cwd=d.ROOT,text=True).splitlines()
    names=[p for p in names if p.startswith(('experiments/','code/','baselines/','comparison_suite/'))
        and Path(p).suffix in ('.py','.json') and '/snapshots/' not in p and '/local_artifacts/' not in p and '/tests/' not in p]
    archive=output/(d.RELEASE+'.tar.gz')
    with archive.open('xb') as f:
        with tarfile.open(fileobj=f,mode='w:gz') as tar:
            for name in names:tar.add(d.ROOT/name,arcname=name,recursive=False)
            blob=(commit+'\n').encode();entry=tarfile.TarInfo('release_commit.txt');entry.size=len(blob);tar.addfile(entry,io.BytesIO(blob))
    sha=hashlib.sha256(archive.read_bytes()).hexdigest()
    subprocess.run(['scp',*SSH,str(archive),'N607:'+d.PROJECT+'/releases/'+archive.name],check=True,timeout=60)
    config=dict(project=d.PROJECT,release=d.RELEASE,run=d.RUN,analysis=False,archive=archive.name,sha256=sha,commit=commit)
    value=remote(REMOTE.replace('CONFIG',repr(config)));d.write(output/'submission.json',value);print(json.dumps(value));return value

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();publish(a.output)
