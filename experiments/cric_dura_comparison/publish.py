"""Publish the committed source stage; all run and release roots are immutable."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile

ROOT = Path(__file__).resolve().parents[2]
CONNECTION = ['-F', str(ROOT / 'tools/n607_ssh_config'), '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10']


def ssh(script):
    compile(script, 'remote', 'exec')
    result = subprocess.run(['ssh', *CONNECTION, '-T', 'N607', 'python3 -'], input=script.encode('utf-8'), capture_output=True, timeout=60)
    if result.returncode:
        raise RuntimeError(result.stderr.decode(errors='replace') + result.stdout.decode(errors='replace'))
    return json.loads(result.stdout)


REMOTE = r'''
import hashlib,json,os,subprocess,tarfile
from pathlib import Path
c=CONFIG
rt=c['runtime'];release=Path(rt['release_root'])
if any(Path(rt[k]).exists() for k in ('release_root','run_root','log_root')):
 raise FileExistsError('Existing submission: reconcile independently before retry')
if subprocess.check_output(['whoami'],text=True).strip()!=c['expected_user']:
 raise ValueError('Ordinary account mismatch')
for path in [*rt['frozen_sources'].values(),rt['old_data_handles'],rt['count_data_handles'],*rt['datasets'].values()]:
 if not Path(path).exists():raise FileNotFoundError(path)
if hashlib.sha256(Path(c['archive_remote']).read_bytes()).hexdigest()!=c['sha256']:
 raise ValueError('Archive transfer mismatch')
release.mkdir()
with tarfile.open(c['archive_remote']) as tar:
 for m in tar.getmembers():
  if not m.isfile() or release.resolve() not in (release/m.name).resolve().parents:raise ValueError('Invalid archive member')
 tar.extractall(release)
runtime=release/'runtime.json';runtime.write_text(json.dumps(rt,indent=2)+'\n')
env=dict(os.environ,PYTHONPATH=str(release),CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
subprocess.run([rt['python'],'-m','compileall','-q',str(release)],cwd=release,env=env,check=True)
command=[rt['python'],'-u','-m','experiments.cric_dura_comparison.source_queue','dispatch','--config',str(runtime)]
log=release/'source_dispatcher.log'
with log.open('x') as stream:
 child=subprocess.Popen(command,cwd=release,env=env,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
receipt=dict(status='SUBMITTED_NOT_YET_VERIFIED',pid=child.pid,cwd=str(release),argv=command,log=str(log),commit=rt['code_commit'])
(release/'submit.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt))
'''


def publish(repo, output):
    from .plan import RUN, make_plan
    repo, output = Path(repo), Path(output)
    output.mkdir(exist_ok=True, parents=True)
    def git(*a):
        return subprocess.check_output(['git', '-C', str(repo), *a])
    commit = git('rev-parse', 'HEAD').decode().strip()
    branch = git('branch', '--show-current').decode().strip()
    if git('ls-remote', 'origin', 'refs/heads/' + branch).decode().split()[0] != commit:
        raise ValueError('Current upstream OID not verified')
    dep = json.loads((output / 'dependencies_inspected.json').read_text(encoding='utf-8'))
    old = json.loads((ROOT / 'local_artifacts/phase1_dg_sourcecount6_20261021/runtime.json').read_text(encoding='utf-8'))
    project = old['run_root'].rsplit('/runs/', 1)[0]
    release = project + '/releases/' + RUN
    roots = dict(dep['paths'], G=project + '/releases/pamf_stage2_inference_20261021_r01')
    rt = dict(run_id=RUN, project=project, release_root=release, run_root=project + '/runs/' + RUN,
        log_root=project + '/logs/' + RUN, matrix=release + '/matrix.json', code_commit=commit,
        python=old['legacy_python'], frozen_sources=roots, datasets=old['datasets'], allowed_gpus=list(range(8)),
        old_data_handles=old['parent_run_root'] + '/data_handles.json', count_data_handles=old['run_root'] + '/data_handles.json',
        launch_owner='codex-root-cric-dura-source-20261021', stage='F_source',
        GPU_policy='one own/card; at most two total compute apps; free>=12000MiB; no healthy process changes')
    if RUN.endswith('-r03'):
        rt['reused_source_handles'] = project + '/runs/20261021-phase1-CRIC-DURA-comparison-m153-r02/data/handles.json'
    # Only this source stage is released here. G/prediction workers are separate
    # immutable releases; frozen source outputs must never be relabelled scored.
    names = ['__init__.py', 'bootstrap.py', 'io.py', 'data.py', 'f_data.py', 'f_model.py',
             'f_optimizer.py', 'structural.py', 'f_train.py', 'source_queue.py', 'smoke.py']
    paths = ['experiments/cric_dura_comparison/' + n for n in names]
    # DefaultConcat is the unchanged project implementation. Its channel code
    # resolves through the immutable native BASE/code path set by bootstrap.
    paths += ['experiments/phase1_external_dg_aug/__init__.py', 'experiments/phase1_external_dg_aug/augmentation.py']
    if git('status', '--porcelain', '--', *paths).strip():
        raise ValueError('Uncommitted release paths')
    archive = output / (RUN + '.tar.gz')
    if archive.exists():
        raise FileExistsError('Reconcile existing archive/submission before retry')
    with tarfile.open(archive, 'w:gz') as tar:
        blobs = {p: git('show', commit + ':' + p) for p in paths}
        blobs['matrix.json'] = (json.dumps(make_plan(), indent=2) + '\n').encode()
        blobs['release_commit.txt'] = (commit + '\n').encode()
        for name, blob in blobs.items():
            item = tarfile.TarInfo(name); item.size = len(blob); item.mode = 0o644
            tar.addfile(item, io.BytesIO(blob))
    package = dict(runtime=rt, expected_user=project.split('/')[2], archive_remote=project + '/releases/' + archive.name,
        sha256=hashlib.sha256(archive.read_bytes()).hexdigest())
    for name, value in [('runtime.json', rt), ('package.json', package)]:
        (output / name).write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')
    subprocess.run(['scp', *CONNECTION, str(archive), 'N607:' + package['archive_remote']], check=True, timeout=60)
    receipt = ssh(REMOTE.replace('CONFIG', repr(package)))
    (output / 'submit.json').write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(receipt))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo', required=True); parser.add_argument('--output', required=True)
    args = parser.parse_args(); publish(args.repo, args.output)
