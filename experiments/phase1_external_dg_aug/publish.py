"""One Git archive, one transfer check, then the performance and cost owners."""
import argparse
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import subprocess
import tarfile

from experiments.phase1_dg_nativebudget.publish import ssh, CONNECTION
from .plan import RUN, COST_RUN

ROOT=Path(__file__).resolve().parents[2]

REMOTE=r"""
import hashlib,json,os,subprocess,tarfile
from pathlib import Path
package=PACKAGE
rt=package['runtime'];cost=package['cost_runtime'];release=Path(rt['release_root'])
if subprocess.check_output(['whoami'],text=True).strip()!=package['expected_user']:raise ValueError('Wrong ordinary account')
for p in (release,Path(rt['run_root']),Path(rt['log_root']),Path(cost['run_root']),Path(cost['log_root'])):
 if p.exists():raise FileExistsError('Existing delivery; reconcile before retry')
archive=Path(package['archive_remote'])
if hashlib.sha256(archive.read_bytes()).hexdigest()!=package['sha256']:raise ValueError('Archive transfer mismatch')
release.mkdir()
with tarfile.open(archive) as tar:
 for member in tar.getmembers():
  target=(release/member.name).resolve()
  if not member.isfile() or release.resolve() not in target.parents:raise ValueError('Archive path differs')
 tar.extractall(release)
runtime=release/'runtime.json';runtime.write_text(json.dumps(rt,indent=2)+'\n')
env=dict(os.environ,PYTHONPATH=os.pathsep.join([str(release),str(release/'code'),rt['numpy_deps']]),
 CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
subprocess.run([rt['python'],'-m','compileall','-q',str(release)],cwd=release,env=env,check=True)
command=[rt['python'],'-u','-m','experiments.phase1_external_dg_aug.driver','dispatch','--config',str(runtime)]
with (release/'dispatcher.stdout.log').open('x') as log:
 child=subprocess.Popen(command,cwd=release,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
receipt=dict(status='SUBMITTED',pid=child.pid,cwd=str(release),argv=command,commit=package['commit'],allowed_gpus=[0,1],expected_training=219)
(release/'submit.json').write_text(json.dumps(receipt,indent=2)+'\n')
cost.update(suite_pid=child.pid)
cost_path=release/'cost_runtime.json';cost_path.write_text(json.dumps(cost,indent=2)+'\n')
cost_command=[cost['python'],'-u','-m','experiments.phase1_external_dg_aug.cost','dispatch','--config',str(cost_path)]
with (release/'cost_dispatcher.stdout.log').open('x') as log:
 cost_child=subprocess.Popen(cost_command,cwd=release,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
cost_receipt=dict(status='SUBMITTED',pid=cost_child.pid,cwd=str(release),argv=cost_command,commit=package['commit'],allowed_gpus=[0,1],expected_rows=96)
(release/'cost_submit.json').write_text(json.dumps(cost_receipt,indent=2)+'\n')
print(json.dumps(dict(performance=receipt,cost=cost_receipt,cost_runtime=cost,archive_transfer='MATCH',remote_compile='PASS')))
"""


def publish(repo,output):
    repo=Path(repo);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    commit=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
    branch=subprocess.check_output(['git','-C',str(repo),'branch','--show-current'],text=True).strip()
    remote=subprocess.check_output(['git','-C',str(repo),'ls-remote','origin','refs/heads/'+branch],text=True).split()[0]
    if remote!=commit:raise ValueError('Push verification required')
    old=json.loads((ROOT/'local_artifacts/phase1_external_dg_20261019/runtime.json').read_text(encoding='utf-8-sig'))
    project=str(PurePosixPath(old['run_root']).parent.parent)
    release=project+'/releases/'+RUN
    rt=dict(old,run_id=RUN,release_root=release,run_root=project+'/runs/'+RUN,log_root=project+'/logs/'+RUN,
        matrix=release+'/experiments/phase1_external_dg_aug/matrix.json',runtime_file=release+'/runtime.json',
        code_commit=commit,reused_suite_root=old['run_root'],launch_owner='codex-root-phase1-concat-pl-20261019')
    cost=dict(run_id=COST_RUN,release_root=release,run_root=project+'/runs/'+COST_RUN,log_root=project+'/logs/'+COST_RUN,
        python=rt['python'],numpy_deps=rt['numpy_deps'],code_commit=commit,allowed_gpus=[0,1],
        suite_release_root=release,suite_run_root=rt['run_root'],batches=[1,128],warmup=100,repeats=1000,
        launch_owner='codex-root-phase1-concat-pl-cost-20261019')
    all_files=subprocess.check_output(['git','-C',str(repo),'ls-files'],text=True).splitlines()
    selected=[f for f in all_files if f.startswith(('experiments/phase1_external_dg/','experiments/phase1_external_dg_aug/')) and Path(f).suffix in ('.py','.json')]
    selected+=['experiments/phase1_dg_nativebudget/__init__.py','experiments/phase1_dg_nativebudget/pipeline.py',
        'experiments/phase1_dg_nativebudget/riei_step.py','experiments/standard_ce_baselines/__init__.py',
        'experiments/standard_ce_baselines/model.py','baselines/__init__.py','code/baseline_origin_sat_view.py',
        'code/sat_channel.py','code/training_controls.py']
    if 'experiments/__init__.py' in all_files:selected.append('experiments/__init__.py')
    for family in ('riei_fd','drift'):
        selected+=['baselines/'+family+'/'+f for f in ('__init__.py','model.py','architecture.py','losses.py')]
    selected.append('baselines/drift/grl.py');selected=sorted(set(selected))
    if any(f not in all_files for f in selected):raise ValueError('Release dependency not committed')
    if subprocess.check_output(['git','-C',str(repo),'status','--porcelain','--',*selected],text=True).strip():
        raise ValueError('Release code/config must be committed')
    archive=output/(RUN+'.tar.gz')
    if archive.exists():raise FileExistsError('Existing package; reconcile before resubmitting')
    archive_remote=project+'/releases/'+archive.name
    probe="""import json,os
from pathlib import Path
rt=RUNTIME;cost=COST
paths=[rt['release_root'],rt['run_root'],rt['log_root'],cost['run_root'],cost['log_root'],ARCHIVE]
if any(Path(p).exists() for p in paths):raise FileExistsError('Existing immutable run root')
required=[rt['python'],rt['reused_suite_root']+'/data_handles.json',rt['datasets']['ManySig'],
 rt['ibrff_model_root']+'/backbones/FAN.py',rt['sdg_model_root']+'/main.py']
if any(not Path(p).is_file() for p in required):raise FileNotFoundError('Missing declared input')
old_state=json.loads((Path(rt['reused_suite_root'])/'pipeline_state.json').read_text())
if old_state['rows']:raise ValueError('Withdrawn old suite unexpectedly started GPU rows')
old_proc=Path('/proc')/str(old_state['pid'])
if old_proc.exists() and old_proc.joinpath('stat').read_text().rsplit(') ',1)[1].split()[0]!='Z':raise ValueError('Old suite still alive')
print(json.dumps(dict(status='INPUTS_PRESENT',ordinary_user=os.getuid()!=0)))
""".replace('RUNTIME',repr(rt)).replace('COST',repr(cost)).replace('ARCHIVE',repr(archive_remote))
    preflight=ssh(probe)
    if not preflight['ordinary_user']:raise ValueError('Privileged account not authorized')
    with tarfile.open(archive,'w:gz') as tar:
        for file in selected:
            blob=subprocess.check_output(['git','-C',str(repo),'show',commit+':'+file])
            member=tarfile.TarInfo(file);member.size=len(blob);member.mode=0o644;tar.addfile(member,io.BytesIO(blob))
        blob=(commit+'\n').encode();member=tarfile.TarInfo('release_commit.txt');member.size=len(blob);tar.addfile(member,io.BytesIO(blob))
    package=dict(runtime=rt,cost_runtime=cost,commit=commit,archive_remote=archive_remote,
        expected_user=PurePosixPath(project).parts[2],sha256=hashlib.sha256(archive.read_bytes()).hexdigest())
    (output/'runtime.json').write_text(json.dumps(rt,indent=2)+'\n',encoding='utf-8')
    (output/'package.json').write_text(json.dumps(package,indent=2)+'\n',encoding='utf-8')
    subprocess.run(['scp',*CONNECTION,str(archive),'N607:'+archive_remote],check=True,timeout=60)
    result=ssh(REMOTE.replace('PACKAGE',repr(package)))
    for filename,value in [('submit.json',result['performance']),('cost_submit.json',result['cost']),('cost_runtime.json',result['cost_runtime'])]:
        (output/filename).write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')
    (output/'release_result.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(status='SUBMITTED',performance_pid=result['performance']['pid'],cost_pid=result['cost']['pid'],
        commit=commit,training=219,cost_rows=96,archive_transfer=result['archive_transfer'],remote_compile=result['remote_compile'])))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--repo',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();publish(args.repo,args.output)
