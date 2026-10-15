"""Single archive transfer/preflight for the immutable semantic pilot."""
import argparse,hashlib,io,json,subprocess,tarfile
from pathlib import Path
from . import design as d
from experiments.cvs_urban_sigtime50.publish import REMOTE as ORIGINAL,INSPECT
from experiments.cvs_urban_sigtime.operations import remote,SSH

REMOTE=ORIGINAL.replace('experiments/cvs_urban_sigtime50','experiments/cvs_urban_semantic50').replace(
    'experiments.cvs_urban_sigtime50.dispatch','experiments.cvs_urban_semantic50.dispatch')


def configuration():return dict(project=d.PROJECT,release=d.RELEASE,run=d.RUN,analysis=False)


def inspect(output):
    result=remote(INSPECT.replace('CONFIG',repr(configuration())));d.write(Path(output)/'readback.json',result)
    print(json.dumps(result));return result


def publish(output):
    output=Path(output);output.mkdir(parents=True,exist_ok=True);commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=d.ROOT,text=True).strip()
    dirty=subprocess.check_output(['git','status','--porcelain','--','experiments','code','baselines','comparison_suite'],cwd=d.ROOT,text=True)
    if dirty.strip():raise ValueError('Commit intended code/config first; preserve unrelated edits')
    names=subprocess.check_output(['git','ls-files'],cwd=d.ROOT,text=True).splitlines()
    names=[n for n in names if n.startswith(('experiments/','code/','baselines/','comparison_suite/')) and Path(n).suffix in ('.py','.json')
        and '/snapshots/' not in n and '/local_artifacts/' not in n and '/tests/' not in n]
    archive=output/(d.RELEASE+'.tar.gz')
    with archive.open('xb') as stream:
        with tarfile.open(fileobj=stream,mode='w:gz') as tar:
            for name in names:tar.add(d.ROOT/name,arcname=name,recursive=False)
            blob=(commit+'\n').encode();entry=tarfile.TarInfo('release_commit.txt');entry.size=len(blob);tar.addfile(entry,io.BytesIO(blob))
    sha=hashlib.sha256(archive.read_bytes()).hexdigest()
    subprocess.run(['scp',*SSH,str(archive),'N607:'+d.PROJECT+'/releases/'+archive.name],check=True,timeout=60)
    result=remote(REMOTE.replace('CONFIG',repr(dict(configuration(),archive=archive.name,sha256=sha,commit=commit))))
    d.write(output/'submission.json',result);print(json.dumps(result));return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--inspect',action='store_true');a=p.parse_args()
    inspect(a.output) if a.inspect else publish(a.output)
