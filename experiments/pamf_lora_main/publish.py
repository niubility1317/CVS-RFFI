"""Release the committed LoRa experiment and its TX-blind target IQ to N607."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile

from experiments.cric_dura_comparison.io import read, write


RELEASE_CODE = '''
import hashlib, json, os, shutil, subprocess, tarfile
from pathlib import Path
plan = __PLAN__
rt = plan['runtime']
release = Path(rt['release_root'])
archive = Path(plan['archive'])
if os.geteuid() == 0 or release.exists() or Path(rt['run_root']).exists():
    raise ValueError('Ordinary account and unused immutable release/run roots required')
if shutil.disk_usage(release.parent).free < 8 * 1024**3:
    raise ValueError('Insufficient experiment disk space')
for tool in (rt['python'], *rt['frozen_sources'].values()):
    if not Path(tool).exists():
        raise FileNotFoundError(tool)
digest = hashlib.sha256()
with archive.open('rb') as stream:
    for block in iter(lambda: stream.read(1024 * 1024), b''):
        digest.update(block)
if digest.hexdigest() != plan['sha256']:
    raise ValueError('Release transfer did not preserve SHA256')
release.mkdir()
with tarfile.open(archive) as source:
    for member in source:
        dest = release / member.name
        if not member.isfile() or release.resolve() not in dest.resolve().parents:
            raise ValueError('Invalid release member')
        dest.parent.mkdir(parents=True, exist_ok=True)
        with source.extractfile(member) as src, dest.open('xb') as dst:
            shutil.copyfileobj(src, dst)
runtime = release/'runtime.json'
runtime.write_text(json.dumps(rt, indent=2) + chr(10), encoding='utf-8')
env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='2', PYTHONPATH=str(release))
subprocess.run([rt['python'], '-m', 'compileall', '-q', 'experiments/pamf_lora_main'],
               cwd=release, env=env, check=True)
cmd = [rt['python'], '-u', '-m', 'experiments.pamf_lora_main.runner', 'dispatch', str(runtime)]
with (release/'controller.log').open('x') as log:
    p = subprocess.Popen(cmd, cwd=release, env=env, stdin=subprocess.DEVNULL,
                         stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
result = dict(pid=p.pid, cwd=str(release), code_commit=rt['code_commit'], archive_transfer='VERIFIED',
              remote_compile='PASS', status='SUBMITTED_NOT_YET_VERIFIED', run_root=rt['run_root'])
(release/'submit.json').write_text(json.dumps(result, indent=2) + chr(10))
print(json.dumps(result))
'''


def publish(repo, output, inputs, context, ssh_config, run_id):
    def git(*args):
        return subprocess.check_output(['git', '-C', str(repo), *args])

    commit = git('rev-parse', 'HEAD').decode().strip()
    branch = git('branch', '--show-current').decode().strip()
    if git('ls-remote', 'origin', f'refs/heads/{branch}').decode().split()[0] != commit:
        raise ValueError('Published upstream code differs from HEAD')
    data, output = Path(inputs), Path(output)
    src, target = read(data/'source/contract.json'), read(data/'target/contract.json')
    if (src['protocol'] != 'lora_rx5_main_v1' or target['truth_read']
            or src['counts'] != dict(L_s=2800, U_s=25200, V=12000) or target['count'] != 30000
            or set(src['receiver_names']) & set(target['receivers'])):
        raise ValueError('LoRa dataset contract differs')
    ctx = read(context)
    project = ctx['project']
    release = f'{project}/releases/{run_id}-release'
    rt = dict(ctx, run_id=run_id, code_commit=commit, release_root=release,
              run_root=f'{project}/runs/{run_id}', allowed_gpus=list(range(7)),
              source_root=f'{release}/data/source', target_root=f'{release}/data/target')
    output.mkdir(parents=True, exist_ok=True)
    package = output / f'{run_id}-release.tar.gz'
    dirs = ('experiments', 'code', 'baselines', 'comparison_suite')
    committed = git('archive', '--format=tar', commit, '--', *dirs)
    with tarfile.open(fileobj=io.BytesIO(committed)) as source, tarfile.open(package, 'x:gz', compresslevel=3) as dest:
        for member in source:
            if member.isfile() and Path(member.name).suffix in ('.py', '.json', '.txt', '.yaml', '.yml', '.toml'):
                dest.addfile(member, source.extractfile(member))
        for part in ('source', 'target'):
            for path in sorted((data/part).iterdir()):
                if path.suffix not in ('.npy', '.npz', '.json') or not path.is_file():
                    raise ValueError('Unexpected LoRa data packaging input')
                dest.add(path, arcname=f'data/{part}/{path.name}', recursive=False)
    remote = f'{project}/releases/{package.name}'
    digest = hashlib.sha256()
    with package.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    connection = ['-F', str(ssh_config), '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10']
    # The verified ordinary account and immutable destination are used once.
    subprocess.run(['scp', *connection, str(package), 'N607:'+remote], check=True, timeout=240)
    script = RELEASE_CODE.replace('__PLAN__', repr(dict(runtime=rt, archive=remote, sha256=digest.hexdigest())))
    result = subprocess.run(['ssh', *connection, '-T', 'N607', rt['python']+' -'],
        input=script.encode('utf-8'), capture_output=True, timeout=120)
    if result.returncode:
        raise RuntimeError(result.stderr.decode(errors='replace'))
    reply = json.loads(result.stdout)
    write(output/'submit.json', reply)
    print(json.dumps(reply), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('repo', 'output', 'inputs', 'context', 'ssh-config', 'run-id'):
        parser.add_argument('--'+name, required=True)
    args = parser.parse_args()
    publish(args.repo, args.output, args.inputs, args.context, args.ssh_config, args.run_id)
