"""Publish one committed archive and start exactly one immutable owner."""
import argparse
import hashlib
import io
import json
from pathlib import Path,PurePosixPath
import subprocess
import tarfile
from experiments.phase1_dg_nativebudget.publish import ssh,CONNECTION
from .plan import RUN,make_plan,PARENT

ROOT=Path(__file__).resolve().parents[2]

REMOTE=r'''
import hashlib,json,os,subprocess,tarfile
from pathlib import Path
c=PACKAGE
rt=c['runtime'];release=Path(rt['release_root'])
if os.getuid()==0 or subprocess.check_output(['whoami'],text=True).strip()!=c['expected_user']:
 raise ValueError('Wrong ordinary account')
for p in (release,Path(rt['run_root']),Path(rt['log_root'])):
 if p.exists():raise FileExistsError('Existing delivery; reconcile before retry')
archive=Path(c['archive_remote'])
if hashlib.sha256(archive.read_bytes()).hexdigest()!=c['sha256']:raise ValueError('Archive transfer mismatch')
release.mkdir()
with tarfile.open(archive) as tar:
 for m in tar.getmembers():
  if not m.isfile() or release.resolve() not in (release/m.name).resolve().parents:raise ValueError('Invalid archive member')
 tar.extractall(release)
runtime=release/'runtime.json';runtime.write_text(json.dumps(rt,indent=2)+'\n')
env=dict(os.environ,PYTHONPATH=os.pathsep.join([str(release),str(release/'code'),rt['numpy_deps']]),
 CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
subprocess.run([rt['python'],'-m','compileall','-q',str(release)],cwd=release,env=env,check=True)
command=[rt['python'],'-u','-m','experiments.phase1_dg_sourcecount6.driver','dispatch','--config',str(runtime)]
with (release/'dispatcher.stdout.log').open('x') as log:
 child=subprocess.Popen(command,cwd=release,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
receipt=dict(status='SUBMITTED',pid=child.pid,cwd=str(release),argv=command,commit=c['commit'],
 allowed_gpus=rt['allowed_gpus'],expected_rows=432,new_training=384,reused_rows=48,
 archive_transfer='MATCH',remote_compile='PASS')
(release/'submit.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt))
'''


def publish(repo,output):
    repo=Path(repo);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    commit=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
    branch=subprocess.check_output(['git','-C',str(repo),'branch','--show-current'],text=True).strip()
    oid=subprocess.check_output(['git','-C',str(repo),'ls-remote','origin','refs/heads/'+branch],text=True).split()[0]
    if oid!=commit:raise ValueError('Push verification required')
    old=json.loads((ROOT/'local_artifacts/phase1_external_dg_aug_pl_gpu23_r3_20261019/runtime.json').read_text(encoding='utf-8-sig'))
    project=str(PurePosixPath(old['run_root']).parent.parent);release=project+'/releases/'+RUN
    rt=dict(old,run_id=RUN,release_root=release,run_root=project+'/runs/'+RUN,log_root=project+'/logs/'+RUN,
        runtime_file=release+'/runtime.json',matrix=release+'/experiments/phase1_dg_sourcecount6/matrix.json',
        code_commit=commit,allowed_gpus=make_plan()['allowed_gpus'],parent_run_root=old['run_root'],
        existing_observations_root=project+'/runs/20261019-phase1-external-dg-suite-m168-r01/data/observations/ManySig',
        launch_owner='codex-root-sourcecount6-20261021')
    files=subprocess.check_output(['git','-C',str(repo),'ls-files'],text=True).splitlines()
    selected=[f for f in files if f.startswith(('experiments/phase1_external_dg/','experiments/phase1_external_dg_aug/',
        'experiments/phase1_dg_sourcecount6/')) and Path(f).suffix in ('.py','.json')]
    selected+=['experiments/phase1_dg_nativebudget/__init__.py','experiments/phase1_dg_nativebudget/pipeline.py',
        'experiments/phase1_dg_nativebudget/riei_step.py','experiments/standard_ce_baselines/__init__.py',
        'experiments/standard_ce_baselines/model.py','baselines/__init__.py','code/baseline_origin_sat_view.py',
        'code/sat_channel.py','code/training_controls.py','baselines/drift/grl.py']
    for family in ('riei_fd','drift'):
        selected+=['baselines/'+family+'/'+f for f in ('__init__.py','model.py','architecture.py','losses.py')]
    if 'experiments/__init__.py' in files:selected.append('experiments/__init__.py')
    selected=sorted(set(selected))
    if any(f not in files for f in selected):raise ValueError('Uncommitted release dependency')
    if subprocess.check_output(['git','-C',str(repo),'status','--porcelain','--',*selected],text=True).strip():
        raise ValueError('Release code/config dirty')
    archive=output/(RUN+'.tar.gz');remote_archive=project+'/releases/'+archive.name
    if archive.exists():raise FileExistsError('Existing package; reconcile before resubmission')
    probe=r'''
import json,os,subprocess,shutil
from pathlib import Path
rt=RUNTIME
paths=[rt['release_root'],rt['run_root'],rt['log_root'],ARCHIVE]
if any(Path(p).exists() for p in paths):raise FileExistsError('Existing immutable delivery')
required=[rt['python'],rt['datasets']['ManySig'],rt['parent_run_root']+'/completion.json',
 rt['existing_observations_root']+'/index.npz',rt['ibrff_model_root']+'/backbones/FAN.py',rt['sdg_model_root']+'/main.py']
if any(not Path(p).is_file() for p in required):raise FileNotFoundError('Missing declared input')
done=json.loads((Path(rt['parent_run_root'])/'completion.json').read_text())
if done.get('status')!='ANALYZED' or not done.get('complete'):raise ValueError('Parent matrix incomplete')
gpus=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,utilization.gpu,memory.used','--format=csv,noheader'],text=True)
print(json.dumps(dict(status='INPUTS_PRESENT',ordinary_user=os.getuid()!=0,gpus=gpus,disk_free=shutil.disk_usage(Path(rt['run_root']).parent).free)))
'''.replace('RUNTIME',repr(rt)).replace('ARCHIVE',repr(remote_archive))
    preflight=ssh(probe)
    if not preflight['ordinary_user']:raise ValueError('Privileged account denied')
    with tarfile.open(archive,'w:gz') as tar:
        for file in selected:
            blob=subprocess.check_output(['git','-C',str(repo),'show',commit+':'+file])
            m=tarfile.TarInfo(file);m.size=len(blob);m.mode=0o644;tar.addfile(m,io.BytesIO(blob))
        blob=(commit+'\n').encode();m=tarfile.TarInfo('release_commit.txt');m.size=len(blob);tar.addfile(m,io.BytesIO(blob))
    package=dict(runtime=rt,commit=commit,archive_remote=remote_archive,expected_user=PurePosixPath(project).parts[2],
        sha256=hashlib.sha256(archive.read_bytes()).hexdigest())
    for name,value in [('runtime.json',rt),('package.json',package),('release_preflight.json',preflight)]:
        (output/name).write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')
    subprocess.run(['scp',*CONNECTION,str(archive),'N607:'+remote_archive],check=True,timeout=60)
    result=ssh(REMOTE.replace('PACKAGE',repr(package)))
    (output/'submit.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--repo',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();publish(a.repo,a.output)
