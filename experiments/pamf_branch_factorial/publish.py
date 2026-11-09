"""Package and submit exactly one immutable preregistered run."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile

from experiments.cric_dura_comparison import publish as transport
from .method import RUN

REMOTE = r'''
import hashlib,json,os,subprocess,tarfile
from pathlib import Path
p=PACKAGE
rt=p['runtime'];release=Path(rt['release_root']);root=Path(rt['run_root'])
if release.exists() or root.exists():raise FileExistsError('Reconcile existing release/run before any retry')
if hashlib.sha256(Path(p['archive_remote']).read_bytes()).hexdigest()!=p['sha256']:raise ValueError('Archive transfer mismatch')
release.mkdir()
with tarfile.open(p['archive_remote']) as tar:
 for m in tar.getmembers():
  if not m.isfile() or release.resolve() not in (release/m.name).resolve().parents:raise ValueError('Invalid archive member')
 tar.extractall(release)
(release/'runtime.json').write_text(json.dumps(rt,indent=2)+'\n')
env=dict(os.environ,PYTHONPATH=str(release),CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
subprocess.run([rt['python'],'-m','compileall','-q',str(release/'experiments/pamf_branch_factorial')],cwd=release,env=env,check=True)
cmd=[rt['python'],'-u','-m','experiments.pamf_branch_factorial.runner','dispatch','--config',rt['runtime_file']]
with (release/'controller.log').open('x') as f:child=subprocess.Popen(cmd,cwd=release,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
result=dict(status='SUBMITTED_NOT_YET_VERIFIED',pid=child.pid,cwd=str(release),argv=cmd,log=str(release/'controller.log'),archive_transfer='VERIFIED',compile='PASS',code_commit=rt['code_commit'])
(release/'submit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
'''


def publish(repo, output):
    repo, out = Path(repo), Path(output)
    # This checkout is nested under the project, whereas the SSH config lives
    # at the project root; never infer it from the checkout location.
    transport.CONNECTION = ['-F', 'E:/type10-7/tools/n607_ssh_config', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10']
    def git(*args):
        return subprocess.check_output(['git', '-C', str(repo), *args])
    commit = git('rev-parse', 'HEAD').decode().strip()
    branch = git('branch', '--show-current').decode().strip()
    if git('ls-remote', 'origin', 'refs/heads/' + branch).decode().split()[0] != commit:
        raise ValueError('Remote branch OID mismatch')
    if git('status', '--porcelain', '--', 'experiments/pamf_branch_factorial',
           'automation_reports/CV-SincNet/' + RUN).strip():
        raise ValueError('Commit experiment and adapter before publishing')
    context = json.loads((out / 'context.json').read_text(encoding='utf-8'))
    project = context['project']
    release = project + '/releases/' + RUN
    rt = dict(run_id=RUN, project=project, python=context['python'], frozen_sources=context['frozen_sources'],
              code_commit=commit, release_root=release, run_root=project + '/runs/' + RUN,
              runtime_file=release + '/runtime.json', context=release + '/context.json',
              allowed_gpus=list(range(8)), max_own_GPU_workers=16, max_own_per_GPU=2, max_total_per_GPU=2,
              launch_owner='codex-PAMF-branch-factorial-20261109')
    archive = out / (RUN + '.tar.gz')
    if archive.exists():
        raise FileExistsError('Reconcile prior submission before retrying')
    roots = ('experiments', 'code', 'baselines', 'comparison_suite')
    tracked = set(git('ls-files', '-z', '--', *roots).decode().split('\0'))
    src = git('archive', '--format=tar', commit, '--', *roots)
    with tarfile.open(fileobj=io.BytesIO(src)) as source, tarfile.open(archive, 'w:gz') as dest:
        for member in source:
            if member.name in tracked and member.isfile() and Path(member.name).suffix in ('.py', '.json', '.txt', '.yaml', '.yml', '.toml'):
                dest.addfile(member, source.extractfile(member))
        items = [('context.json', (out / 'context.json').read_bytes()),
                 ('release_commit.txt', (commit + '\n').encode()),
                 ('experiment.json', git('show', commit + ':automation_reports/CV-SincNet/' + RUN + '/experiment.json'))]
        for name, data in items:
            member = tarfile.TarInfo(name)
            member.size = len(data)
            dest.addfile(member, io.BytesIO(data))
    package = dict(runtime=rt, archive_remote=project + '/releases/' + archive.name,
                   sha256=hashlib.sha256(archive.read_bytes()).hexdigest())
    for name, obj in [('runtime.json', rt), ('package.json', package)]:
        (out / name).write_text(json.dumps(obj, indent=2) + '\n', encoding='utf-8')
    subprocess.run(['scp', *transport.CONNECTION, str(archive), 'N607:' + package['archive_remote']], check=True, timeout=120)
    result = transport.ssh(REMOTE.replace('PACKAGE', repr(package)))
    (out / 'submit.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    publish(args.repo, args.output)
