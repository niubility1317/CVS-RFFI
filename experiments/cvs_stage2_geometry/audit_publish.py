"""Immutable publication and independent observation of source-only audits."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile

from . import design as d
from experiments.cvs_stage2_decouple import publish as base
from .audit_control import RUN

RELEASE = 'cvs_stage2_geometry_source_20261011_r01'
PATHS = [*base.PATHS, *['experiments/cvs_stage2_geometry/'+name+'.py' for name in (
    '__init__','design','features','geometry_audit','geometry_checks','source_audit',
    'objectives','actions','mechanism_checks','receiver','receiver_checks',
    'physics_audit','physics_checks','audit_control','audit_checks','audit_publish')]]
REGISTRATION = 'automation_reports/CV-SincNet/' + RUN

REMOTE = base.REMOTE.replace(
    "'experiments.cvs_stage2_decouple.control_checks','--remote-preflight','--output',str(release/'preflight.json')",
    "'experiments.cvs_stage2_geometry.audit_checks'").replace(
    "command=[python,'-u','-m','experiments.cvs_stage2_decouple.dispatch']",
    "command=[python,'-u','-m','experiments.cvs_stage2_geometry.audit_control']")

INSPECT = r'''
import json,sys,time
from pathlib import Path
c=CONFIG;p=Path(c['project']);release=p/'releases'/c['release'];base=p/'runs'/c['run']
sys.path.insert(0,str(release));sys.path.insert(0,str(release/'code'))
from experiments.cvs_receiver_residual_capacity4.control import process
from experiments.cvs_multi_state_action.dispatch import capacity
def read(path):return json.loads(path.read_text()) if path.is_file() else None
submit=read(release/'submit.json');launch=read(base/'launch.json') or {'rows':[]}
result=dict(run_id=c['run'],read_at=time.time(),submit=submit,controller=read(base/'controller.json'),
 controller_process=process(submit['pid']) if submit else None,queue=read(base/'queue_state.json'),
 failure=read(base/'failure.json'),completion=read(base/'completion.json'),workers=[])
for item in launch['rows']:
 row=base/('s'+str(item['seed']));log=Path(item['log'])
 result['workers'].append(dict(item,process=process(item['pid']),failure=read(row/'failure.json'),
  artifact_exists=(row/'audit.json').is_file(),log_tail=log.read_text()[-2500:] if log.is_file() else None))
result['capacity']={str(g):dict(pids=sorted(v['pids']),free_mb=v['free_mb']) for g,v in capacity().items()}
actual=result['controller_process'];done=result['completion']
if result['failure']:result['status']='FAILED'
elif done and done['status']=='ANALYZED' and done['rows']==3 and not done['target_access'] and done['identity_updates']==0:
 from experiments.cvs_stage2_geometry.audit_control import validate_result
 for seed in (2026092701,2026092702,2026092703):validate_result(base/('s'+str(seed))/'audit.json',seed)
 result['status']='VERIFIED'
elif actual and actual['cwd']==str(release) and actual['argv']==submit['argv']:result['status']='VERIFIED'
else:result['status']='UNKNOWN'
print(json.dumps(result))
'''


def package(output):
    output = Path(output)
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=d.ROOT, text=True).strip()
    branch = subprocess.check_output(['git', 'branch', '--show-current'], cwd=d.ROOT, text=True).strip()
    remote = subprocess.check_output(['git', '-c', 'http.sslBackend=openssl', '-c', 'http.version=HTTP/1.1',
        'ls-remote', 'origin', 'refs/heads/' + branch], cwd=d.ROOT, text=True).split()[0]
    if remote != commit:
        raise ValueError('Remote release commit differs')
    paths = [*PATHS, REGISTRATION]
    if subprocess.check_output(['git', 'status', '--porcelain', '--', *paths], cwd=d.ROOT, text=True).strip():
        raise ValueError('Uncommitted release paths')
    output.mkdir(parents=True, exist_ok=True)
    archive = output / (RELEASE + '.tar.gz')
    if archive.exists():
        raise FileExistsError('Existing package; inspect outcome before retry')
    blob = subprocess.check_output(['git', 'archive', '--format=tar', commit, '--', *PATHS], cwd=d.ROOT)
    with tarfile.open(fileobj=io.BytesIO(blob)) as source, tarfile.open(archive, 'w:gz') as target:
        for member in source.getmembers():
            if member.isfile() and Path(member.name).suffix in {'.py', '.json', '.md'}:
                target.addfile(member, source.extractfile(member))
        additions = {'release_commit.txt': (commit + '\n').encode()}
        for name in ('experiment.json', 'report.md', 'events.jsonl'):
            additions['audit_registration/' + name] = subprocess.check_output(
                ['git', 'show', commit + ':' + REGISTRATION + '/' + name], cwd=d.ROOT)
        for name, value in additions.items():
            member = tarfile.TarInfo(name); member.size = len(value)
            target.addfile(member, io.BytesIO(value))
    config = dict(project=d.PROJECT, run=RUN, release=RELEASE, commit=commit,
        archive=archive.name, sha256=hashlib.sha256(archive.read_bytes()).hexdigest())
    d.write(output / 'package.json', config)
    return config, archive


def inspect(output):
    output = Path(output); config = d.read(output / 'package.json')
    payload = INSPECT.replace('CONFIG', repr(config))
    wrapper = "import subprocess\nr=subprocess.run(['/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python','-'],input=" + repr(payload) + ",text=True,capture_output=True)\nprint(r.stdout)\nif r.returncode: raise RuntimeError(r.stderr)\n"
    result = json.loads(base.ssh(wrapper))
    d.write(output / 'readback.json', result)
    return result


def publish(output):
    output = Path(output); config, archive = package(output)
    before = "from pathlib import Path\nimport json\nc=CONFIG\np=Path(c['project'])\npaths=[p/'releases'/c['release'],p/'runs'/c['run'],p/'releases'/c['archive']]\nprint(json.dumps(dict(existing=[str(x) for x in paths if x.exists()])))\n"
    prior = json.loads(base.ssh(before.replace('CONFIG', repr(config))))
    d.write(output / 'remote_before.json', prior)
    if prior['existing']:
        raise FileExistsError('Reconcile existing remote artifacts: ' + repr(prior))
    subprocess.run(['scp', *base.CONNECTION, str(archive), 'N607:' + d.PROJECT + '/releases/' + archive.name], check=True)
    try:
        receipt = json.loads(base.ssh(REMOTE.replace('CONFIG', repr(config))))
        d.write(output / 'submit.json', receipt)
    except Exception as exc:
        d.write(output / 'submission_unknown.json', dict(status='UNKNOWN', error=repr(exc), automatic_retry=False))
        raise
    return inspect(output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--inspect', action='store_true')
    args = parser.parse_args()
    print(json.dumps(inspect(args.output) if args.inspect else publish(args.output)))
