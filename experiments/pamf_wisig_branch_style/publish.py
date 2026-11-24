"""Package the committed comparison and release it to the ordinary N607 account."""

import argparse
import hashlib
import io
import json
import subprocess
import tarfile
from pathlib import Path

from .plan import ORIGINAL_ROW, ORIGINAL_RUN, RUN, TASKS

SSH = ['-F', 'E:/type10-7/tools/n607_ssh_config', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10']
MODULE = 'experiments.pamf_wisig_branch_style.runner'
PROJECT = '/home/szu2070436088/2510044040/CV-SincNet'
PYTHON = '/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python'
OWNER = 'codex-pamf-wisig-branch-style-20261124'


def ssh(source):
    compile(source, '<N607>', 'exec')
    result = subprocess.run(['ssh', *SSH, '-T', 'N607', 'python3 -'],
                            input=source.encode('utf-8'), capture_output=True, timeout=90)
    if result.returncode:
        raise RuntimeError(result.stderr.decode('utf-8', errors='replace'))
    return json.loads(result.stdout)


def context():
    return ssh(f"""
import json
from pathlib import Path
project = Path({PROJECT!r})
run = project / 'runs' / {ORIGINAL_RUN!r}
release = project / 'releases' / ({ORIGINAL_RUN!r} + '-release')
source = run / 'F' / {ORIGINAL_ROW!r} / 'source'
finished = json.loads((source / 'complete.json').read_text())
prefix = Path(finished['segments'][0]['folder'])
old = json.loads((release / 'context.json').read_text())['handles']['F0']
runtime = json.loads((release / 'runtime.json').read_text())
tests = {{key: old['tests'][key] for key in {TASKS!r}}}
prediction = run / 'readouts' / {ORIGINAL_ROW!r} / 'predictions'
if finished['status'] != 'SOURCE_FROZEN' or finished['target_access'] or finished['U_truth_read']:
    raise ValueError('Historical control source permissions differ')
for key, count in {{'cross_rx': 126000, 'cross_both': 42000}}.items():
    manifest = json.loads((Path(tests[key]['views_root']) / 'manifest.json').read_text())
    if manifest['status'] != 'VALIDATED_ONCE' or manifest['count'] != count:
        raise ValueError('Historical Main population handle differs')
    if not (prediction / key / 'predictions.npz').is_file():
        raise FileNotFoundError('Historical paired predictions unavailable')
print(json.dumps(dict(original_source_config=finished['config'], prefix=str(prefix),
                      original_prediction=str(prediction), classes=old['classes'], tests=tests,
                      frozen_sources=runtime['frozen_sources'])))
""")


LAND = '''
import hashlib
import json
import os
import shutil
import subprocess
import tarfile
from pathlib import Path

package = PACKAGE
rt = package['runtime']
release, run = Path(rt['release_root']), Path(rt['run_root'])
project = Path(rt['project'])
if subprocess.check_output(['whoami'], text=True).strip() != project.parts[2]:
    raise ValueError('Ordinary N607 account required')
if release.exists() or run.exists():
    raise FileExistsError('Existing release/run. Independently reconcile before retry.')
if shutil.disk_usage(project).free < 10 * 1024**3:
    raise ValueError('Insufficient space for source and prediction artifacts')
archive_path = Path(package['archive_remote'])
if hashlib.sha256(archive_path.read_bytes()).hexdigest() != package['sha256']:
    raise ValueError('Archive transfer mismatch')
release.mkdir()
with tarfile.open(archive_path) as archive:
    for member in archive.getmembers():
        if not member.isfile() or release not in (release / member.name).resolve().parents:
            raise ValueError('Invalid release member')
    archive.extractall(release)
Path(rt['runtime_file']).write_text(json.dumps(rt, indent=2) + chr(10), encoding='utf-8')
env = dict(os.environ, PYTHONPATH=str(release), CUDA_VISIBLE_DEVICES='',
           OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
subprocess.run([rt['python'], '-m', 'compileall', '-q',
                str(release / 'experiments/pamf_wisig_branch_style'),
                str(release / 'experiments/pamf_repl_completion/objectives.py'),
                str(release / 'experiments/pamf_repl_view_shuffle/evaluation.py'),
                str(release / 'experiments/cric_dura_comparison/evaluation.py')],
               cwd=release, env=env, check=True)
cmd = [rt['python'], '-u', '-m', package['module'], 'dispatch', '--config', rt['runtime_file']]
log = release / 'controller.log'
with log.open('x') as stream:
    child = subprocess.Popen(cmd, cwd=release, env=env, stdin=subprocess.DEVNULL,
                             stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
result = dict(status='SUBMITTED_NOT_YET_VERIFIED', pid=child.pid, cwd=str(release),
              argv=cmd, log=str(log), code_commit=rt['code_commit'], archive_transfer='VERIFIED',
              remote_compile='PASS')
(release / 'submit.json').write_text(json.dumps(result, indent=2) + chr(10), encoding='utf-8')
print(json.dumps(result))
'''


def publish(repo, output, context_path):
    def git(*args):
        return subprocess.check_output(['git', '-C', str(repo), *args])

    commit = git('rev-parse', 'HEAD').decode().strip()
    branch = git('branch', '--show-current').decode().strip()
    upstream = git('ls-remote', 'origin', 'refs/heads/' + branch).decode().split()
    if not upstream or upstream[0] != commit:
        raise ValueError('Published Git commit must match the remote branch')
    context_data = Path(context_path).read_bytes()
    release = PROJECT + '/releases/' + RUN + '-release'
    runtime = dict(run_id=RUN, project=PROJECT, release_root=release, context=release + '/context.json',
                   run_root=PROJECT + '/runs/' + RUN, runtime_file=release + '/runtime.json', python=PYTHON,
                   code_commit=commit, frozen_sources=json.loads(context_data)['frozen_sources'],
                   allowed_gpus=list(range(8)), launch_owner=OWNER)
    output.mkdir(parents=True, exist_ok=True)
    archive_path = output / (RUN + '-release.tar.gz')
    if archive_path.exists():
        raise FileExistsError('Independently reconcile existing submission before retry')
    sources = ('experiments', 'code', 'baselines')
    allowed = {name for name in git('ls-files', '-z', '--', *sources).decode().split(chr(0))
               if Path(name).suffix in ('.py', '.json', '.yaml', '.yml', '.txt', '.toml')}
    content = git('archive', '--format=tar', commit, '--', *sources)
    with tarfile.open(fileobj=io.BytesIO(content)) as raw, tarfile.open(archive_path, 'w:gz') as packed:
        for item in raw:
            if item.isfile() and item.name in allowed:
                packed.addfile(item, raw.extractfile(item))
        extra = {'context.json': context_data, 'release_commit.txt': (commit + chr(10)).encode(),
                 'experiment.json': git('show', f'{commit}:automation_reports/CV-SincNet/{RUN}/experiment.json')}
        for name, data in extra.items():
            item = tarfile.TarInfo(name)
            item.size = len(data)
            packed.addfile(item, io.BytesIO(data))
    package = dict(runtime=runtime, module=MODULE, archive_remote=PROJECT + '/releases/' + archive_path.name,
                   sha256=hashlib.sha256(archive_path.read_bytes()).hexdigest())
    (output / 'package.json').write_text(json.dumps(package, indent=2), encoding='utf-8')
    subprocess.run(['scp', *SSH, str(archive_path), 'N607:' + package['archive_remote']],
                   check=True, timeout=90)
    result = ssh(LAND.replace('PACKAGE', repr(package)))
    (output / 'submit.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['context', 'publish'])
    parser.add_argument('--repo', type=Path, default=Path.cwd())
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--context', type=Path)
    args = parser.parse_args()
    if args.action == 'context':
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('x', encoding='utf-8') as stream:
            json.dump(context(), stream, indent=2)
    else:
        publish(args.repo, args.output, args.context)
