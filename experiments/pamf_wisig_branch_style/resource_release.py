"""Land a resource-only controller for the existing WiSig job graph."""

import argparse
import hashlib
import io
import json
import subprocess
import tarfile
from pathlib import Path

from .publish import PROJECT, PYTHON, RUN, SSH, ssh

RELEASE = f'{RUN}-shared-memory-r01'
MEMBER = 'runner.py'
SOURCE = 'experiments/pamf_wisig_branch_style/runner.py'

INSTALL = """
import hashlib, json, os, subprocess, tarfile
from pathlib import Path

p = PACKAGE
release = Path(p['release'])
archive = Path(p['archive'])
runtime_path = Path(p['runtime'])
runtime = json.loads(runtime_path.read_text())
if os.geteuid() == 0 or runtime['run_id'] != p['run'] or runtime['python'] != p['python']:
    raise ValueError('Expected the registered ordinary-account launch')
if release.exists() or not (Path(runtime['run_root']) / 'pipeline_state.json').is_file():
    raise ValueError('Release collision or missing existing queue')
if hashlib.sha256(archive.read_bytes()).hexdigest() != p['archive_sha']:
    raise ValueError('Resource controller archive transfer differs')
release.mkdir()
with tarfile.open(archive) as tar:
    with (release / 'runner.py').open('xb') as stream:
        stream.write(tar.extractfile('runner.py').read())
controller = release / 'runner.py'
subprocess.run([p['python'], '-m', 'py_compile', str(controller)], check=True)
env = dict(os.environ, PYTHONPATH=runtime['release_root'], CUDA_VISIBLE_DEVICES='')
args = [p['python'], '-u', str(controller), 'resume', '--config', str(runtime_path),
        '--previous-pid', str(p['previous_pid']), '--controller-commit', p['commit']]
log = release / 'controller.log'
with log.open('x') as stream:
    child = subprocess.Popen(args, cwd=release, env=env, stdin=subprocess.DEVNULL,
                             stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
receipt = dict(pid=child.pid, argv=args, log=str(log), cwd=str(release), controller_commit=p['commit'],
               archive_transfer='VERIFIED', remote_compile='PASS')
(release / 'submit.json').write_text(json.dumps(receipt))
print(json.dumps(receipt))
"""


def publish(output, previous_pid):
    def git(*args):
        return subprocess.check_output(['git', *args])

    commit = git('rev-parse', 'HEAD').decode().strip()
    branch = git('branch', '--show-current').decode().strip()
    upstream = git('ls-remote', 'origin', 'refs/heads/' + branch).decode().split()
    if not upstream or upstream[0] != commit:
        raise ValueError('Resource controller commit has not reached the upstream')
    content = git('show', f'{commit}:{SOURCE}')
    archive = output / (RELEASE + '.tar.gz')
    with archive.open('xb') as stream, tarfile.open(fileobj=stream, mode='w:gz') as tar:
        item = tarfile.TarInfo(MEMBER)
        item.size = len(content)
        tar.addfile(item, io.BytesIO(content))

    releases = f'{PROJECT}/releases'
    data = dict(release=f'{releases}/{RELEASE}', archive=f'{releases}/{archive.name}',
                runtime=f'{releases}/{RUN}-release/runtime.json', python=PYTHON, run=RUN,
                commit=commit, archive_sha=hashlib.sha256(archive.read_bytes()).hexdigest(),
                previous_pid=previous_pid)
    subprocess.run(['scp', *SSH, str(archive), 'N607:' + data['archive']], check=True, timeout=60)
    result = ssh(INSTALL.replace('PACKAGE', repr(data)))
    (output / 'shared_memory_submit.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--previous-pid', type=int, required=True)
    args = parser.parse_args()
    publish(args.output, args.previous_pid)
