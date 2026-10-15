"""Read-only post-freeze source mechanism release, separate from running training."""
import argparse,hashlib,io,json,subprocess,tarfile
from pathlib import Path
from . import design as d
from .publish import REMOTE,INSPECT
from experiments.cvs_urban_sigtime.operations import remote,SSH


def configuration():
    return dict(project=d.PROJECT,release='cvs_sigtime_joint160_identity_source_20261015_r01',
        run=d.RUN,analysis=True,analysis_module='experiments.cvs_urban_identity50.mechanism',analysis_output='source_identity_mechanism')


def inspect(output):
    value=remote(INSPECT.replace('CONFIG',repr(configuration())))
    d.write(Path(output)/'readback.json',value)
    print(json.dumps(value,ensure_ascii=False));return value


def publish(output):
    # Read-only eligibility first; never contact query truth or create an output
    # until all registered training/prediction/scoring rows are already complete.
    state=remote("import json;from pathlib import Path;p=Path("+repr(d.BASE.as_posix())+
        ");v=json.loads((p/'complete.json').read_text());print(json.dumps(v))")
    if state.get('status')!='ANALYZED' or state.get('actual_model_count')!=2:
        raise ValueError('Wait for the existing two rows; never modify or restart training')
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=d.ROOT,text=True).strip()
    names=subprocess.check_output(['git','ls-files'],cwd=d.ROOT,text=True).splitlines()
    names=[n for n in names if n.startswith(('experiments/','code/','baselines/','comparison_suite/')) and Path(n).suffix in ('.py','.json')
        and '/snapshots/' not in n and '/local_artifacts/' not in n and '/tests/' not in n]
    dirty=subprocess.check_output(['git','status','--porcelain','--','experiments','code','baselines','comparison_suite'],cwd=d.ROOT,text=True)
    if dirty.strip():raise ValueError('Commit intended analysis code; preserve unrelated changes')
    archive=output/(configuration()['release']+'.tar.gz')
    with archive.open('xb') as raw:
        with tarfile.open(fileobj=raw,mode='w:gz') as tar:
            for name in names:tar.add(d.ROOT/name,arcname=name,recursive=False)
            blob=(commit+'\n').encode();entry=tarfile.TarInfo('release_commit.txt');entry.size=len(blob);tar.addfile(entry,io.BytesIO(blob))
    sha=hashlib.sha256(archive.read_bytes()).hexdigest()
    subprocess.run(['scp',*SSH,str(archive),'N607:'+d.PROJECT+'/releases/'+archive.name],check=True,timeout=60)
    value=remote(REMOTE.replace('CONFIG',repr(dict(configuration(),archive=archive.name,sha256=sha,commit=commit))))
    d.write(output/'submission.json',value);print(json.dumps(value));return value


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--inspect',action='store_true');a=p.parse_args()
    inspect(a.output) if a.inspect else publish(a.output)
