"""Publish one committed archive over the project's ordinary N607 connection."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile

from .evaluation import RUN

REMOTE = r'''
import hashlib,json,os,subprocess,tarfile
from pathlib import Path
c=CONFIG
release=Path(c['release']); root=Path(c['project'])/'runs'/c['run']
if release.exists() or root.exists(): raise FileExistsError('Existing run; reconcile before retry')
if subprocess.check_output(['whoami'],text=True).strip()!=c['user']: raise ValueError('Wrong N607 account')
if hashlib.sha256(Path(c['archive']).read_bytes()).hexdigest()!=c['sha256']: raise ValueError('Archive transfer mismatch')
release.mkdir()
with tarfile.open(c['archive']) as archive:
 for member in archive.getmembers():
  if not member.isfile() or release.resolve() not in (release/member.name).resolve().parents: raise ValueError('Invalid release member')
 archive.extractall(release)
env=dict(os.environ,PYTHONPATH=str(release),CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
subprocess.run([c['python'],'-m','compileall','-q',str(release/'experiments/pamf_trisig_dura_baseline')],cwd=release,env=env,check=True)
cmd=[c['python'],'-u','-m','experiments.pamf_trisig_dura_baseline.runner','dispatch','--project',c['project'],'--commit',c['commit']]
with (release/'controller.log').open('x') as stream:
 child=subprocess.Popen(cmd,cwd=release,env=env,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps(dict(status='SUBMITTED_NOT_YET_VERIFIED',pid=child.pid,cwd=str(release),argv=cmd,archive_transfer='VERIFIED',remote_compile='PASS')))
'''


def publish(repo, output, ssh_config, project, python):
    repo, output = Path(repo), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    def git(*args):
        return subprocess.check_output(['git', '-C', str(repo), *args])
    commit = git('rev-parse', 'HEAD').decode().strip()
    branch = git('branch', '--show-current').decode().strip()
    remote = git('ls-remote', 'origin', 'refs/heads/' + branch).decode().split()
    if not remote or remote[0] != commit:
        raise ValueError('Current upstream OID not verified')
    archive_path = output / (RUN + '.tar.gz')
    if archive_path.exists():
        raise FileExistsError('Existing archive; reconcile before retry')
    tracked = {p for p in git('ls-files', '-z', '--', 'experiments', 'code', 'baselines').decode().split('\0')
               if p and Path(p).suffix in ('.py', '.json', '.txt', '.yaml', '.yml', '.toml')}
    source = git('archive', '--format=tar', commit, '--', 'experiments', 'code', 'baselines')
    with tarfile.open(fileobj=io.BytesIO(source)) as old, tarfile.open(archive_path, 'w:gz') as new:
        for member in old:
            if member.name in tracked:
                blob = old.extractfile(member).read()
                info = tarfile.TarInfo(member.name); info.size = len(blob); info.mode = 0o644
                new.addfile(info, io.BytesIO(blob))
        blob = (commit + '\n').encode()
        info = tarfile.TarInfo('release_commit.txt'); info.size = len(blob); info.mode = 0o644
        new.addfile(info, io.BytesIO(blob))
    remote_archive = project + '/releases/' + archive_path.name
    package = dict(run=RUN, project=project, release=project + '/releases/' + RUN + '-release',
                   archive=remote_archive, user=project.split('/')[2], python=python, commit=commit,
                   sha256=hashlib.sha256(archive_path.read_bytes()).hexdigest())
    connection = ['-F', str(ssh_config), '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10']
    subprocess.run(['scp', *connection, str(archive_path), 'N607:' + remote_archive], check=True, timeout=120)
    script = REMOTE.replace('CONFIG', repr(package))
    compile(script, '<remote>', 'exec')
    result = subprocess.run(['ssh', *connection, '-T', 'N607', 'python3 -'], input=script.encode(),
                            capture_output=True, check=True, timeout=120)
    submission = json.loads(result.stdout)
    (output / 'submit.json').write_text(json.dumps(submission, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(submission))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--repo', required=True); p.add_argument('--output', required=True)
    p.add_argument('--ssh-config', required=True); p.add_argument('--project', required=True)
    p.add_argument('--python', required=True)
    a = p.parse_args(); publish(a.repo, a.output, a.ssh_config, a.project, a.python)
