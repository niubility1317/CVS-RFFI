"""Publish one Git-backed archive and submit the non-PAMF CPU/GPU dependency queue."""
import argparse
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import subprocess
import tarfile

from .plan import RUN
from experiments.phase1_dg_nativebudget.publish import ssh, CONNECTION

ROOT = Path(__file__).resolve().parents[2]


REMOTE = r"""
import hashlib,json,os,subprocess,tarfile
from pathlib import Path
package=PACKAGE
rt=package['runtime'];release=Path(rt['release_root'])
for path in [release,Path(rt['run_root']),Path(rt['log_root'])]:
 if path.exists():raise FileExistsError('Existing output; reconcile instead of relaunch')
if subprocess.check_output(['whoami'],text=True).strip()!=package['expected_user']:raise ValueError('Wrong ordinary account')
archive=Path(package['archive_remote'])
if hashlib.sha256(archive.read_bytes()).hexdigest()!=package['sha256']:raise ValueError('Archive transfer mismatch')
release.mkdir()
with tarfile.open(archive) as tar:
 for member in tar.getmembers():
  target=(release/member.name).resolve()
  if not member.isfile() or release.resolve() not in target.parents:raise ValueError('Invalid archive path')
 tar.extractall(release)
runtime=release/'runtime.json';runtime.write_text(json.dumps(rt,indent=2)+'\n')
env=dict(os.environ,PYTHONPATH=str(release)+os.pathsep+rt['numpy_deps'],
 CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
subprocess.run([rt['python'],'-m','compileall','-q',str(release)],cwd=release,env=env,check=True)
command=[rt['python'],'-u','-m','experiments.phase1_external_dg.driver','dispatch','--config',str(runtime)]
with (release/'dispatcher.stdout.log').open('x') as log:
 child=subprocess.Popen(command,cwd=release,env=env,stdin=subprocess.DEVNULL,
  stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
receipt=dict(status='SUBMITTED',pid=child.pid,cwd=str(release),argv=command,
 commit=package['commit'],allowed_gpus=[0,1],expected_new_training=162,expected_frozen_eval=6)
(release/'submit.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt))
"""


def publish(repo, output):
    repo = Path(repo); output = Path(output); output.mkdir(parents=True, exist_ok=True)
    commit = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
    branch = subprocess.check_output(['git', '-C', str(repo), 'branch', '--show-current'], text=True).strip()
    remote = subprocess.check_output(['git', '-C', str(repo), 'ls-remote', 'origin',
        'refs/heads/' + branch], text=True).split()[0]
    if remote != commit: raise ValueError('Push verification required')
    parent = json.loads((ROOT / 'local_artifacts/phase1_dg_nativebudget_20261019/runtime.json').read_text(encoding='utf-8'))
    parent_receipt = json.loads((ROOT / 'local_artifacts/phase1_dg_nativebudget_20261019/submit.json').read_text(encoding='utf-8'))
    project = str(PurePosixPath(parent['dataset']).parents[1])
    release = project + '/releases/' + RUN
    rt = dict(run_id=RUN, release_root=release, run_root=project + '/runs/' + RUN,
        log_root=project + '/logs/' + RUN, matrix=release + '/experiments/phase1_external_dg/matrix.json',
        runtime_file=release + '/runtime.json', code_commit=commit, allowed_gpus=[0, 1],
        parent_pid=parent_receipt['pid'], parent_run_root=parent['run_root'],
        parent_release_root=parent['release_root'],
        existing_source_cache=parent['source_cache'], existing_source_contract=parent['source_contract'],
        existing_views_root=parent['views_root'], existing_truth=parent['truth'],
        datasets=dict(ManySig=parent['dataset'], ManyRx=project + '/Dataset_WigSig/ManyRx.pkl'),
        ibrff_model_root=parent['ibrff_model_root'], sdg_model_root=parent['sdg_model_root'],
        python=parent['python'], numpy_deps=parent['numpy_deps'],
        legacy_python=str(PurePosixPath(parent['python']).parents[2] / 'CVS-RFFI/bin/python'),
        legacy_run_root=project + '/runs/20261010-phase1-standard-ce-baselines-manysig-m56-r01',
        channel_engine_root=project + '/releases/cvs_residual_sixscene_20261002_r01',
        channel_engine_commit='7669a6de973959290089a04293cc542db26e5fe1',
        launch_owner='codex-root-phase1-external-dg-20261019')
    all_files = subprocess.check_output(['git', '-C', str(repo), 'ls-files'], text=True).splitlines()
    selected = [f for f in all_files if f.startswith('experiments/phase1_external_dg/') and
                Path(f).suffix in ('.py', '.json')]
    selected += ['experiments/phase1_dg_nativebudget/__init__.py',
        'experiments/phase1_dg_nativebudget/pipeline.py', 'experiments/phase1_dg_nativebudget/riei_step.py',
        'experiments/standard_ce_baselines/__init__.py', 'experiments/standard_ce_baselines/model.py',
        'baselines/__init__.py']
    if 'experiments/__init__.py' in all_files: selected.append('experiments/__init__.py')
    for family in ('riei_fd', 'drift'):
        for name in ('__init__.py', 'model.py', 'architecture.py', 'losses.py'):
            selected.append('baselines/' + family + '/' + name)
    selected.append('baselines/drift/grl.py')
    selected = sorted(set(selected))
    if any(f not in all_files for f in selected): raise ValueError('Release dependency not committed')
    if subprocess.check_output(['git', '-C', str(repo), 'status', '--porcelain', '--', *selected], text=True).strip():
        raise ValueError('Uncommitted release paths')
    archive = output / (RUN + '.tar.gz')
    if archive.exists(): raise FileExistsError('Existing package: reconcile before retry')
    probe = """
import json,os
from pathlib import Path
rt=RUNTIME
paths=[rt['release_root'],rt['run_root'],rt['log_root'],ARCHIVE]
existing=[p for p in paths if Path(p).exists()]
if existing:raise FileExistsError('Existing delivery; reconcile: '+str(existing))
required=[rt['python'],rt['legacy_python'],*rt['datasets'].values(),
rt['existing_source_contract'],rt['existing_source_cache']+'/metadata.json',
rt['existing_views_root']+'/manifest.json',rt['existing_truth'],
rt['ibrff_model_root']+'/backbones/FAN.py',rt['sdg_model_root']+'/main.py',
rt['channel_engine_root']+'/code/leo_practical/batch.py']
if any(not Path(p).is_file() for p in required):raise FileNotFoundError('Missing declared input/runtime')
if (Path(rt['channel_engine_root'])/'release_commit.txt').read_text().strip()!=rt['channel_engine_commit']:
 raise ValueError('Historical channel engine version differs')
print(json.dumps(dict(status='INPUTS_PRESENT',ordinary_user=os.getuid()!=0)))
""".replace('RUNTIME', repr(rt)).replace('ARCHIVE', repr(project + '/releases/' + archive.name))
    preflight = ssh(probe)
    if not preflight['ordinary_user']: raise ValueError('Privileged execution not authorized')
    with tarfile.open(archive, 'w:gz') as tar:
        for f in selected:
            blob = subprocess.check_output(['git', '-C', str(repo), 'show', commit + ':' + f])
            member = tarfile.TarInfo(f); member.size = len(blob); member.mode = 0o644
            tar.addfile(member, io.BytesIO(blob))
        value = (commit + '\n').encode()
        member = tarfile.TarInfo('release_commit.txt'); member.size = len(value)
        tar.addfile(member, io.BytesIO(value))
    package = dict(runtime=rt, commit=commit, archive_remote=project + '/releases/' + archive.name,
        expected_user=PurePosixPath(project).parts[2], sha256=hashlib.sha256(archive.read_bytes()).hexdigest())
    (output / 'runtime.json').write_text(json.dumps(rt, indent=2) + '\n', encoding='utf-8')
    (output / 'package.json').write_text(json.dumps(package, indent=2) + '\n', encoding='utf-8')
    subprocess.run(['scp', *CONNECTION, str(archive), 'N607:' + package['archive_remote']], check=True, timeout=60)
    receipt = ssh(REMOTE.replace('PACKAGE', repr(package)))
    (output / 'submit.json').write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(dict(status='SUBMITTED', pid=receipt['pid'], commit=commit,
        allowed_gpus=[0, 1], new_training=162, frozen_eval=6)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--repo', required=True); parser.add_argument('--output', required=True)
    args = parser.parse_args(); publish(args.repo, args.output)
