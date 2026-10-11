"""Committed immutable release; root is the sole caller authorized to publish."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile

from . import design as d
from experiments.cvs_multi_disentangle.publish import PATHS as BASE_PATHS
from experiments.cvs_phase1_stack.publish import CONNECTION, ssh

PATHS = list(dict.fromkeys([*BASE_PATHS, 'experiments/cvs_multi_action_audit',
    'experiments/cvs_multi_state_action', 'experiments/cvs_legacy_no_sat',
    'experiments/cvs_multi_action_risk', 'experiments/cvs_feature_disentangle',
    'experiments/cvs_stage2_decouple', 'experiments/cvs_stage2_geometry', 'code/sat_channel.py', 'code/training_controls.py']))

REMOTE = r'''
from pathlib import Path
import json,os,subprocess,tarfile,hashlib,shutil
c=CONFIG;p=Path(c['project']);release=p/'releases'/c['release']
if subprocess.check_output(['whoami'],text=True).strip()!='szu2070436088' or subprocess.check_output(['hostname'],text=True).strip()!='dell-DSS8440':raise ValueError('Wrong host/user')
for path in [release,p/'runs'/c['run'],p/'logs'/c['run']]:
 if path.exists():raise FileExistsError('Existing path; reconcile rather than repeat launch: '+str(path))
archive=p/'releases'/c['archive']
if hashlib.sha256(archive.read_bytes()).hexdigest()!=c['sha256']:raise ValueError('Archive transport differs')
if shutil.disk_usage(p).free<30*1024**3:raise RuntimeError('Insufficient output space')
release.mkdir()
with tarfile.open(archive) as tar:
 for member in tar.getmembers():
  if not member.isfile() or not (release/member.name).resolve().is_relative_to(release.resolve()):raise ValueError('Unsafe archive member')
 tar.extractall(release)
python='/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python'
env=dict(os.environ,PYTHONPATH=str(release)+os.pathsep+str(release/'code'),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONUNBUFFERED='1')
subprocess.run([python,'-m','compileall','-q',str(release)],cwd=release,env=env,check=True)
with (release/'preflight.log').open('x') as log:
 subprocess.run([python,'-u','-m','experiments.cvs_stage2_geometry.control_checks','--remote-preflight','--output',str(release/'preflight.json')],cwd=release,env=dict(env,CUDA_VISIBLE_DEVICES=''),stdout=log,stderr=subprocess.STDOUT,check=True)
command=[python,'-u','-m','experiments.cvs_stage2_geometry.dispatch'];logpath=release/'dispatcher.stdout.log'
with logpath.open('x') as log:
 child=subprocess.Popen(command,cwd=release,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
receipt=dict(status='SUBMITTED',pid=child.pid,cwd=str(release),argv=command,log=str(logpath),commit=c['commit'],run_id=c['run'])
with (release/'submit.json').open('x') as stream:json.dump(receipt,stream,indent=2)
print(json.dumps(receipt))
'''

INSPECT = r'''
import json,sys,time
from pathlib import Path
c=CONFIG;p=Path(c['project']);release=p/'releases'/c['release'];base=p/'runs'/c['run']
sys.path.insert(0,str(release));sys.path.insert(0,str(release/'code'))
from experiments.cvs_receiver_residual_capacity4.control import process
from experiments.cvs_multi_state_action.dispatch import capacity
def read(path):return json.loads(path.read_text()) if path.is_file() else None
submit=read(release/'submit.json');dispatcher=read(base/'dispatcher.json');launch=read(base/'launch.json') or {'rows':[]}
result=dict(run_id=c['run'],read_at=time.time(),submit=submit,dispatcher=dispatcher,
 dispatcher_process=process(submit['pid']) if submit else None,
 queue=read(base/'queue_state.json'),failure=read(base/'failure.json'),completion=read(base/'completion.json'),
 preflight=read(release/'preflight.json'),workers=[])
for item in launch['rows']:
 value=dict(item,process=process(item['pid']))
 if item['kind']=='train':
  source=base/item['identifier']/'source';value['resolved']=read(source/'resolved_config.json')
  value['source_completion']=read(source/'completion.json');value['completion']=read(base/item['identifier']/'completion.json')
 else:value['preparation']=read(base/'prepared'/('s'+item['identifier'])/'preparation_complete.json')
 result['workers'].append(value)
result['capacity']={str(g):dict(pids=sorted(v['pids']),free_mb=v['free_mb']) for g,v in capacity().items()}
actual=result['dispatcher_process']
result['status']='VERIFIED' if actual and actual['cwd']==str(release) and actual['argv']==submit['argv'] and not result['failure'] else 'FAILED' if result['failure'] else 'UNKNOWN'
if result['completion'] and not result['failure']:
 from experiments.cvs_stage2_geometry.dispatch import require_scored
 done=result['completion']
 if done['trained_models']!=12 or done['tested_models']!=12:raise ValueError('Incomplete geometry matrix')
 require_scored(read(base/'scoring_complete.json'),12)
 result['status']='VERIFIED'
print(json.dumps(result))
'''


def package(output):
    output = Path(output)
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=d.ROOT, text=True).strip()
    branch = subprocess.check_output(['git', 'branch', '--show-current'], cwd=d.ROOT, text=True).strip()
    remote = subprocess.check_output(['git', '-c', 'http.sslBackend=openssl', '-c', 'http.version=HTTP/1.1',
        'ls-remote', 'origin', 'refs/heads/' + branch], cwd=d.ROOT, text=True).split()[0]
    if remote != commit:
        raise ValueError('Published upstream differs from local release commit')
    registration='automation_reports/CV-SincNet/'+d.RUN
    dirty = subprocess.check_output(['git', 'status', '--porcelain', '--', *PATHS,registration], cwd=d.ROOT, text=True)
    if dirty.strip():
        raise ValueError('Release contains uncommitted paths')
    output.mkdir(parents=True, exist_ok=True)
    archive = output / (d.RELEASE + '.tar.gz')
    if archive.exists():
        raise FileExistsError('Existing delivery artifact; reconcile outcome before repeating')
    blob = subprocess.check_output(['git', 'archive', '--format=tar', commit, '--', *PATHS], cwd=d.ROOT)
    with tarfile.open(fileobj=io.BytesIO(blob)) as source, tarfile.open(archive, 'w:gz') as target:
        for member in source.getmembers():
            if member.isfile() and Path(member.name).suffix in {'.py', '.json', '.md'}:
                target.addfile(member, source.extractfile(member))
        value = (commit + '\n').encode()
        member = tarfile.TarInfo('release_commit.txt'); member.size = len(value)
        target.addfile(member, io.BytesIO(value))
        for name in ('experiment.json','report.md','events.jsonl'):
            value=subprocess.check_output(['git','show',commit+':'+registration+'/'+name],cwd=d.ROOT)
            member=tarfile.TarInfo('registration/'+name);member.size=len(value)
            target.addfile(member,io.BytesIO(value))
    config = dict(project=d.PROJECT, run=d.RUN, release=d.RELEASE, commit=commit,
        archive=archive.name, sha256=hashlib.sha256(archive.read_bytes()).hexdigest())
    d.write(output / 'package.json', config)
    return config, archive


def inspect(output, config=None):
    output = Path(output)
    config = config or d.read(output / 'package.json')
    # The SSH transport uses system Python; capacity imports require the
    # project's torch runtime. Inspect only, never resubmit a running release.
    payload=INSPECT.replace('CONFIG', repr(config))
    wrapper="import subprocess\nr=subprocess.run(['/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python','-'],input="+repr(payload)+",text=True,capture_output=True)\nprint(r.stdout)\nif r.returncode: raise RuntimeError(r.stderr)\n"
    result = json.loads(ssh(wrapper))
    d.write(output / 'readback.json', result)
    return result


def metadata(output):
    """Read contract/provenance metadata only; never read prediction values or scores."""
    script = r'''
import json
from pathlib import Path
c=CONFIG
def read(path):return json.loads(Path(path).read_text()) if Path(path).is_file() else None
result={'capsule_manifest':read(Path(c['capsule'])/'manifest.json'),'parents':[]}
for row in c['rows']:
 prediction=Path(row['parent_prediction']);source=Path(row['parent_checkpoint']).parent
 result['parents'].append(dict(model_seed=row['model_seed'],checkpoint_exists=(source/'final_ssdg.pth').is_file(),
 prediction_exists=prediction.is_file(),prediction_complete=read(prediction.parent/'complete.json'),
 prediction_provenance=read(prediction.parent/'provenance.json')))
views=Path(c['project'])/'runs/20261002-phase1-cvs-residual-sixscene-manysig-m8-r01/received_views'
result['views_manifest']=read(views/'manifest.json')
print(json.dumps(result))
'''
    config = dict(project=d.PROJECT, capsule=d.CAPSULE,
                  rows=[d.config(r) for r in d.rows() if r['arm'] == 'B0'])
    result = json.loads(ssh(script.replace('CONFIG', repr(config))))
    d.write(Path(output) / 'parent_metadata.json', result)
    return result


def publish(output):
    output = Path(output)
    config, archive = package(output)
    # Read before SCP: a prior archive or submission is evidence to reconcile.
    precheck = "from pathlib import Path\nimport json\nc=CONFIG\np=Path(c['project'])\npaths=[p/'releases'/c['release'],p/'runs'/c['run'],p/'releases'/c['archive']]\nprint(json.dumps(dict(existing=[str(x) for x in paths if x.exists()])))\n"
    prior = json.loads(ssh(precheck.replace('CONFIG', repr(config))))
    d.write(output / 'remote_before.json', prior)
    if prior['existing']:
        raise FileExistsError('Remote release artifacts exist; reconcile: ' + repr(prior['existing']))
    subprocess.run(['scp', *CONNECTION, str(archive), 'N607:' + d.PROJECT + '/releases/' + archive.name], check=True)
    try:
        receipt = json.loads(ssh(REMOTE.replace('CONFIG', repr(config))))
        d.write(output / 'submit.json', receipt)
    except Exception as exc:
        d.write(output / 'submission_unknown.json', dict(status='UNKNOWN', error=repr(exc), no_automatic_retry=True))
        raise
    result = inspect(output, config)
    print(json.dumps(dict(status=result['status'], submit=receipt, readback=str(output / 'readback.json'))))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--inspect', action='store_true')
    parser.add_argument('--metadata', action='store_true')
    args = parser.parse_args()
    if args.metadata:
        result = metadata(args.output)
        print(json.dumps(dict(capsule_manifest=result['capsule_manifest'], parents=[
            dict(model_seed=p['model_seed'], checkpoint_exists=p['checkpoint_exists'],
                 prediction_exists=p['prediction_exists'],
                 prediction_status=p['prediction_complete']['status'] if p['prediction_complete'] else None)
            for p in result['parents']], evidence=str(args.output / 'parent_metadata.json'))))
    elif args.inspect:
        print(json.dumps(inspect(args.output)))
    else:
        publish(args.output)
