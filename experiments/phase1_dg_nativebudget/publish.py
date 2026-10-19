"""Publish one committed archive, compare it once, compile once, then dispatch."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile

RUN='20261019-phase1-pamf-dg-nativebudget-m24-r01'
ROOT=Path(__file__).resolve().parents[2]
CONNECTION=['-F',str(ROOT/'tools/n607_ssh_config'),'-o','BatchMode=yes','-o','ConnectTimeout=10']


def ssh(script):
    compile(script,'remote','exec')
    result=subprocess.run(['ssh',*CONNECTION,'-T','N607','python3 -'],input=script.encode('utf-8'),
                          capture_output=True,timeout=60)
    if result.returncode:raise RuntimeError(result.stderr.decode(errors='replace')+'\n'+result.stdout.decode(errors='replace'))
    return json.loads(result.stdout)


REMOTE=r'''
import hashlib,json,os,subprocess,tarfile
from pathlib import Path
c=CONFIG
rt=c['runtime'];release=Path(rt['release_root']);run=Path(rt['run_root']);logs=Path(rt['log_root'])
if release.exists() or run.exists() or logs.exists():raise FileExistsError('Existing delivery; reconcile before any retry')
if subprocess.check_output(['whoami'],text=True).strip()!=c['expected_user']:raise ValueError('Ordinary account mismatch')
archive=Path(c['archive_remote'])
if hashlib.sha256(archive.read_bytes()).hexdigest()!=c['sha256']:raise ValueError('Archive transfer mismatch')
release.mkdir()
with tarfile.open(archive) as tar:
 for member in tar.getmembers():
  resolved=(release/member.name).resolve()
  if not member.isfile() or release.resolve() not in resolved.parents:raise ValueError('Unsafe archive member')
 tar.extractall(release)
runtime_file=release/'runtime.json'
runtime_file.write_text(json.dumps(rt,indent=2)+'\n')
env=dict(os.environ,PYTHONPATH=str(release)+os.pathsep+rt['numpy_deps'],CUDA_VISIBLE_DEVICES='',
         OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
subprocess.run([rt['python'],'-m','compileall','-q',str(release)],cwd=release,env=env,check=True)
for path in (rt['source_contract'],rt['dataset'],rt['truth'],rt['views_root']+'/manifest.json',
             rt['data_code_root']+'/code/dataset_wisig.py',rt['ibrff_model_root']+'/backbones/FAN.py',rt['sdg_model_root']+'/main.py'):
 if not Path(path).is_file():raise FileNotFoundError(path)
manifest=json.loads((Path(rt['views_root'])/'manifest.json').read_text())
if manifest['status']!='VALIDATED_ONCE' or manifest['count']!=168000:raise ValueError('Existing view handles differ')
command=[rt['python'],'-u',str(release/'pipeline.py'),'dispatch','--config',str(runtime_file)]
logpath=release/'dispatcher.stdout.log'
with logpath.open('x') as log:
 child=subprocess.Popen(command,cwd=release,env=env,stdin=subprocess.DEVNULL,
                        stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
receipt=dict(status='SUBMITTED',pid=child.pid,cwd=str(release),argv=command,log=str(logpath),commit=c['commit'],allowed_gpus=[0,1])
(release/'submit.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt))
'''


def publish(repo,output):
    repo=Path(repo);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    commit=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
    branch=subprocess.check_output(['git','-C',str(repo),'branch','--show-current'],text=True).strip()
    remote=subprocess.check_output(['git','-C',str(repo),'ls-remote','origin','refs/heads/'+branch],text=True).split()[0]
    if remote!=commit:raise ValueError('Push verification required')
    metadata=json.loads((output/'preflight_metadata.json').read_text(encoding='utf-8'))
    project=metadata['root'];p=Path(project)
    oldroot=ROOT/'local_artifacts/rffi_dg_research_20261019'
    ib=json.loads((oldroot/'20261019-phase1-ibrff-author-manysig-s2023-r02/runtime.json').read_text())
    sdg=json.loads((oldroot/'20261019-phase1-sdg-author-wisig-s2023-r01/runtime.json').read_text())
    release=project+'/releases/'+RUN
    rt=dict(run_id=RUN,release_root=release,run_root=project+'/runs/'+RUN,
        log_root=project+'/logs/'+RUN,runtime_file=release+'/runtime.json',
        matrix=release+'/matrix.json',allowed_gpus=[0,1],
        source_cache=project+'/runs/'+RUN+'/source_cache',
        source_contract=project+'/runs/phase1_daot_rc4_pure_game_m3_20260917_r2/source_contract.json',
        dataset=project+'/Dataset_WigSig/ManySig.pkl',
        data_code_root=project+'/releases/sigtime_purece_e200_20261014_r01',
        data_code_commit='6408f29ee8ff9b355d12d88271d460d4ad082d15',
        views_root=project+'/runs/20261002-phase1-cvs-residual-sixscene-manysig-m8-r01/received_views',
        truth=project+'/runs/20260927-phase1-baselines-final-clean-satellite-m5-r01/data/truth.json',
        ibrff_model_root=ib['model_root'],sdg_model_root=sdg['model_root'],
        python=sdg['python'],numpy_deps=sdg['release_root']+'/deps',
        launch_owner='codex-root-phase1-dg-nativebudget-20261019',code_commit=commit)
    paths={'experiments/phase1_dg_nativebudget/'+name:name for name in ('pipeline.py','verify.py','matrix.json','riei_step.py')}
    for name in ('baselines/__init__.py','baselines/riei_fd/__init__.py','baselines/riei_fd/model.py',
        'baselines/riei_fd/architecture.py','baselines/riei_fd/losses.py',
        'baselines/drift/__init__.py','baselines/drift/model.py','baselines/drift/architecture.py',
        'baselines/drift/losses.py','baselines/drift/grl.py'):paths[name]=name
    dirty=subprocess.check_output(['git','-C',str(repo),'status','--porcelain','--',*paths],text=True)
    if dirty.strip():raise ValueError('Release paths have uncommitted changes')
    archive=output/(RUN+'.tar.gz')
    if archive.exists():raise FileExistsError('Existing archive; inspect remote state before retry')
    with tarfile.open(archive,'w:gz') as tar:
        for original,target in paths.items():
            blob=subprocess.check_output(['git','-C',str(repo),'show',commit+':'+original])
            item=tarfile.TarInfo(target);item.size=len(blob);item.mode=0o644;tar.addfile(item,io.BytesIO(blob))
        blob=(commit+'\n').encode();item=tarfile.TarInfo('release_commit.txt');item.size=len(blob);tar.addfile(item,io.BytesIO(blob))
    config=dict(runtime=rt,commit=commit,expected_user=Path(project).parts[2],
        archive_remote=project+'/releases/'+archive.name,sha256=hashlib.sha256(archive.read_bytes()).hexdigest())
    (output/'runtime.json').write_text(json.dumps(rt,indent=2)+'\n',encoding='utf-8')
    (output/'package.json').write_text(json.dumps(config,indent=2)+'\n',encoding='utf-8')
    subprocess.run(['scp',*CONNECTION,str(archive),'N607:'+config['archive_remote']],check=True,timeout=60)
    receipt=ssh(REMOTE.replace('CONFIG',repr(config)))
    (output/'submit.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(status='SUBMITTED',pid=receipt['pid'],allowed_gpus=receipt['allowed_gpus'],commit=commit)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--repo',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();publish(a.repo,a.output)
