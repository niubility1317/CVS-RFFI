"""One immutable Git archive for deferred frozen-model costs."""
import argparse
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import subprocess
import tarfile

from experiments.phase1_dg_nativebudget.publish import ssh, CONNECTION
from .cost import COST_RUN, BATCHES, WARMUP, REPEATS

ROOT = Path(__file__).resolve().parents[2]

REMOTE = r'''
import json,hashlib,tarfile,subprocess,os
from pathlib import Path
package=PACKAGE
rt=package['runtime'];release=Path(rt['release_root'])
for p in [release,Path(rt['run_root']),Path(rt['log_root'])]:
 if p.exists():raise FileExistsError('Existing cost output; reconcile')
if subprocess.check_output(['whoami'],text=True).strip()!=package['expected_user']:
 raise ValueError('Wrong ordinary account')
archive=Path(package['archive_remote'])
if hashlib.sha256(archive.read_bytes()).hexdigest()!=package['sha256']:
 raise ValueError('Archive transfer mismatch')
release.mkdir()
with tarfile.open(archive) as tar:
 for member in tar.getmembers():
  target=(release/member.name).resolve()
  if not member.isfile() or release.resolve() not in target.parents:raise ValueError('Invalid archive member')
 tar.extractall(release)
runtime=release/'runtime.json';runtime.write_text(json.dumps(rt,indent=2)+'\n')
env=dict(os.environ,PYTHONPATH=str(release)+os.pathsep+rt['numpy_deps'],CUDA_VISIBLE_DEVICES='',
 OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
subprocess.run([rt['python'],'-m','compileall','-q',str(release)],cwd=release,env=env,check=True)
command=[rt['python'],'-u','-m','experiments.phase1_external_dg.cost','dispatch','--config',str(runtime)]
with (release/'dispatcher.stdout.log').open('x') as log:
 child=subprocess.Popen(command,cwd=release,env=env,stdin=subprocess.DEVNULL,
  stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
receipt=dict(status='SUBMITTED',pid=child.pid,cwd=str(release),argv=command,
 commit=package['commit'],allowed_gpus=[0,1],frozen_cost_rows=69,additional_training=0)
(release/'submit.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt))
'''


def runtime(output):
    suite_folder = ROOT / 'local_artifacts/phase1_external_dg_20261019'
    suite = json.loads((suite_folder / 'runtime.json').read_text(encoding='utf-8'))
    receipt = json.loads((suite_folder / 'submit.json').read_text(encoding='utf-8'))
    project = str(PurePosixPath(suite['run_root']).parent.parent)
    return dict(run_id=COST_RUN, release_root=project + '/releases/' + COST_RUN,
        run_root=project + '/runs/' + COST_RUN, log_root=project + '/logs/' + COST_RUN,
        allowed_gpus=[0, 1], batches=list(BATCHES), warmup=WARMUP, repeats=REPEATS,
        suite_pid=receipt['pid'], suite_release_root=suite['release_root'], suite_run_root=suite['run_root'],
        parent_run_root=suite['parent_run_root'], parent_matrix=suite['parent_release_root'] + '/matrix.json',
        legacy_run_root=suite['legacy_run_root'], source_contract_F0=suite['existing_source_contract'],
        python=suite['python'], numpy_deps=suite['numpy_deps'],
        launch_owner='codex-root-phase1-external-dg-cost-20261019')


def publish(repo, output):
    repo = Path(repo); output = Path(output); output.mkdir(parents=True, exist_ok=True)
    commit = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
    remote = subprocess.check_output(['git', '-C', str(repo), 'ls-remote', 'origin',
        'refs/heads/work/cvs-active'], text=True).split()[0]
    if commit != remote:
        raise ValueError('Commit push not verified')
    rt = runtime(output); rt['code_commit'] = commit
    available = subprocess.check_output(['git', '-C', str(repo), 'ls-files'], text=True).splitlines()
    files = [p for p in available if p.startswith('experiments/phase1_external_dg/') and
        Path(p).suffix in ('.py', '.json')]
    files += ['experiments/phase1_dg_nativebudget/' + p for p in ('__init__.py', 'pipeline.py', 'riei_step.py')]
    files += ['experiments/standard_ce_baselines/' + p for p in ('__init__.py', 'model.py')]
    files.append('baselines/__init__.py')
    for family in ('riei_fd', 'drift'):
        files += ['baselines/' + family + '/' + p for p in ('__init__.py', 'model.py', 'architecture.py', 'losses.py')]
    files.append('baselines/drift/grl.py')
    if 'experiments/__init__.py' in available:
        files.append('experiments/__init__.py')
    files = sorted(set(files))
    if any(p not in available for p in files) or not any(p.endswith('/cost.py') for p in files):
        raise ValueError('Uncommitted cost dependencies')
    if subprocess.check_output(['git', '-C', str(repo), 'status', '--porcelain', '--', *files], text=True).strip():
        raise ValueError('Dirty release code paths')
    archive = output / (COST_RUN + '.tar.gz')
    if archive.exists():
        raise FileExistsError('Existing cost package; reconcile')
    archive_remote = str(PurePosixPath(rt['release_root']).parent / archive.name)
    probe = '''
import os,json
from pathlib import Path
rt=RUNTIME
for p in [rt['release_root'],rt['run_root'],rt['log_root'],ARCHIVE]:
 if Path(p).exists():raise FileExistsError('Existing cost delivery; reconcile')
for p in [rt['python'],rt['parent_matrix'],rt['source_contract_F0'],rt['suite_run_root']+'/data_handles.json']:
 if not Path(p).is_file():raise FileNotFoundError('Declared cost input absent')
print(json.dumps(dict(ordinary_user=os.getuid()!=0,status='INPUTS_PRESENT')))
'''.replace('RUNTIME', repr(rt)).replace('ARCHIVE', repr(archive_remote))
    if not ssh(probe)['ordinary_user']:
        raise ValueError('Privileged cost execution is not authorized')
    with tarfile.open(archive, 'w:gz') as tar:
        for path in files:
            value = subprocess.check_output(['git', '-C', str(repo), 'show', commit + ':' + path])
            member = tarfile.TarInfo(path); member.size = len(value); member.mode = 0o644
            tar.addfile(member, io.BytesIO(value))
        value = (commit + '\n').encode(); member = tarfile.TarInfo('release_commit.txt'); member.size = len(value)
        tar.addfile(member, io.BytesIO(value))
    package = dict(runtime=rt, commit=commit, archive_remote=archive_remote,
        expected_user=PurePosixPath(rt['release_root']).parts[2], sha256=hashlib.sha256(archive.read_bytes()).hexdigest())
    (output / 'runtime.json').write_text(json.dumps(rt, indent=2) + '\n', encoding='utf-8')
    (output / 'package.json').write_text(json.dumps(package, indent=2) + '\n', encoding='utf-8')
    subprocess.run(['scp', *CONNECTION, str(archive), 'N607:' + archive_remote], check=True, timeout=60)
    receipt = ssh(REMOTE.replace('PACKAGE', repr(package)))
    (output / 'submit.json').write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(dict(status='SUBMITTED', pid=receipt['pid'], commit=commit,
        frozen_cost_rows=69, additional_training=0)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--repo', required=True); parser.add_argument('--output', required=True)
    args = parser.parse_args(); publish(args.repo, args.output)
