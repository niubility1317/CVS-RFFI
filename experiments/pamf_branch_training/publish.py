"""One archive transfer and compile, then one ordinary-account launch owner."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile
from .plan import RUN

REMOTE = r'''
import hashlib,json,os,shutil,subprocess,tarfile
from pathlib import Path
p=PACKAGE
rt=p['runtime']; project=Path(rt['project']); release=Path(rt['release_root']); root=Path(rt['run_root'])
if os.geteuid()==0 or subprocess.check_output(['whoami'],text=True).strip()!=str(project).split('/')[2]:raise ValueError('Ordinary account required')
if release.exists() or root.exists():raise FileExistsError('Existing owner: reconcile before retry')
if shutil.disk_usage(project).free<15*1024**3:raise ValueError('Insufficient artifact space')
if hashlib.sha256(Path(p['archive_remote']).read_bytes()).hexdigest()!=p['sha256']:raise ValueError('Archive transfer mismatch')
release.mkdir()
with tarfile.open(p['archive_remote']) as archive:
 for member in archive.getmembers():
  if not member.isfile() or release.resolve() not in (release/member.name).resolve().parents:raise ValueError('Invalid release member')
 archive.extractall(release)
ctx=json.loads(Path(rt['context']).read_text())
for path in [rt['python'],rt['weak_root']+'/manifest.json',rt['weak_truth'],*rt['frozen_sources'].values()]:
 if not Path(path).exists():raise FileNotFoundError(path)
h=ctx['handles']['F0']
for path in [h['source_cache'],h['source_contract'],*[h['tests'][t]['views_root'] for t in ('cross_date','cross_rx','cross_both')]]:
 if not Path(path).exists():raise FileNotFoundError(path)
for kind in ('F20','F21'):
 path=Path(ctx['sources']['2026092701'][kind])/'segment20'
 if not (path/'final_ssdg.pth').is_file() or not (path/'complete.json').is_file():raise FileNotFoundError(path)
for v in ctx['caches']['F0'].values():
 d=json.loads((Path(v['path'])/'complete.json').read_text())
 if d['status']!='DURA_IQ_CACHE_COMPLETE' or d['truth_read']:raise ValueError('Frozen cache missing')
gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True)
Path(rt['runtime_file']).write_text(json.dumps(rt,indent=2)+'\n')
env=dict(os.environ,PYTHONPATH=str(release),CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
subprocess.run([rt['python'],'-m','compileall','-q',str(release/'experiments/pamf_branch_training')],cwd=release,env=env,check=True)
cmd=[rt['python'],'-u','-m','experiments.pamf_branch_training.runner','dispatch','--config',rt['runtime_file']]
with (release/'controller.log').open('x') as f:
 child=subprocess.Popen(cmd,cwd=release,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
result=dict(status='SUBMITTED_NOT_YET_VERIFIED',pid=child.pid,cwd=str(release),argv=cmd,
            code_commit=rt['code_commit'],archive_transfer='VERIFIED',remote_compile='PASS',gpu_free=gpu)
(release/'submit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
'''


def publish(repo, output, context, ssh_config, launch_owner='codex-pamf-branch-training-20261110'):
    repo, out = Path(repo), Path(output)
    out.mkdir(parents=True, exist_ok=True)
    def git(*args):
        return subprocess.check_output(['git', '-C', str(repo), *args])
    commit = git('rev-parse', 'HEAD').decode().strip()
    branch = git('branch', '--show-current').decode().strip()
    if git('ls-remote', 'origin', 'refs/heads/' + branch).decode().split()[0] != commit:
        raise ValueError('Pushed current branch OID differs from release commit')
    ctx = json.loads(Path(context).read_text(encoding='utf-8-sig'))
    old = ctx['runtime']
    project = old['project']
    release = project + '/releases/' + RUN + '-release'
    rt = dict(run_id=RUN, project=project, release_root=release, run_root=project + '/runs/' + RUN,
              context=release + '/context.json', runtime_file=release + '/runtime.json', python=old['python'],
              code_commit=commit, frozen_sources=old['frozen_sources'], allowed_gpus=[0, 1],
              launch_owner=launch_owner,
              weak_root=project + '/runs/20261010-phase1-state-action-completed-test-manysig-m24-r01/weak_views',
              weak_truth=project + '/runs/20260927-phase1-baselines-final-clean-satellite-m5-r01/data/truth.json')
    archive = out / (RUN + '-release.tar.gz')
    if archive.exists():
        raise FileExistsError('Reconcile earlier release before retry')
    paths = {p for p in git('ls-files', '-z', '--', 'experiments', 'code', 'baselines', 'comparison_suite').decode().split('\0')
             if p and Path(p).suffix in ('.py', '.json', '.txt', '.yaml', '.yml', '.toml')}
    source = git('archive', '--format=tar', commit, '--', 'experiments', 'code', 'baselines', 'comparison_suite')
    with tarfile.open(fileobj=io.BytesIO(source)) as old_tar, tarfile.open(archive, 'w:gz') as target:
        for member in old_tar:
            if member.name not in paths:
                continue
            value = old_tar.extractfile(member).read()
            info = tarfile.TarInfo(member.name); info.size = len(value); info.mode = 0o644
            target.addfile(info, io.BytesIO(value))
        for name, value in {'context.json': Path(context).read_bytes(),
                'experiment.json': git('show', commit + ':automation_reports/CV-SincNet/' + RUN + '/experiment.json')}.items():
            info = tarfile.TarInfo(name); info.size = len(value); info.mode = 0o644
            target.addfile(info, io.BytesIO(value))
    package = dict(runtime=rt, archive_remote=project + '/releases/' + archive.name,
                   sha256=hashlib.sha256(archive.read_bytes()).hexdigest())
    connection = ['-F', str(ssh_config), '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10']
    subprocess.run(['scp', *connection, str(archive), 'N607:' + package['archive_remote']], check=True, timeout=90)
    script = REMOTE.replace('PACKAGE', repr(package))
    result = subprocess.run(['ssh', *connection, '-T', 'N607', 'python3 -'], input=script.encode(),
                            capture_output=True, timeout=60)
    if result.returncode:
        raise RuntimeError((result.stderr + result.stdout).decode(errors='replace'))
    value = json.loads(result.stdout)
    (out / 'submit.json').write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(value))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for option in ('repo', 'output', 'context', 'ssh_config'):
        parser.add_argument('--' + option.replace('_', '-'), required=True)
    args = parser.parse_args()
    publish(args.repo, args.output, args.context, args.ssh_config)
