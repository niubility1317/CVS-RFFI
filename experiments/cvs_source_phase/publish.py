"""Immutable two-thread CPU release; no model or target dependency."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile
from . import design as d

SSH = ['-F', 'E:/type10-7/tools/n607_ssh_config', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10']
PYTHON = '/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python'


def remote(script):
    p = subprocess.run(['ssh', *SSH, 'N607', 'python3 -'], input=script, text=True, encoding='utf-8',
        capture_output=True, timeout=60, check=True)
    return json.loads(p.stdout.strip().splitlines()[-1])


REMOTE = r'''
import hashlib,json,os,shutil,subprocess,tarfile,time
from pathlib import Path
c=CONFIG
project=Path(c['project']); release=project/'releases'/c['release']; run=Path(c['config']['output_root'])
if release.exists() or run.exists():raise FileExistsError('Reconcile existing immutable output; never resubmit')
if subprocess.check_output(['whoami'],text=True).strip()!='szu2070436088' or subprocess.check_output(['hostname'],text=True).strip()!='dell-DSS8440':raise ValueError('Ordinary account/host mismatch')
if shutil.disk_usage(project).free<2*1024**3:raise ValueError('Insufficient output space')
for path in (c['config']['source_contract'], c['config']['preparation']+'/source_cache.pt',c['config']['preparation']+'/preparation_complete.json'):
 if not Path(path).is_file():raise FileNotFoundError(path)
archive=project/'releases'/c['archive']
if hashlib.sha256(archive.read_bytes()).hexdigest()!=c['sha256']:raise ValueError('Single release archive transfer mismatch')
release.mkdir()
with tarfile.open(archive) as f:
 for m in f.getmembers():
  if not m.isfile() or not (release/m.name).resolve().is_relative_to(release.resolve()):raise ValueError('Unsafe archive destination')
 f.extractall(release)
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONPATH=str(release),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
subprocess.run([c['python'],'-m','compileall','-q',str(release/'experiments/cvs_source_phase')],cwd=release,env=env,check=True)
preflight=dict(status='VERIFIED',archive_transfer='VERIFIED',remote_compile='PASS',ordinary_account=True,CPU_threads=2,
 GPU='none',disk_free_bytes=shutil.disk_usage(project).free,commit=c['commit'],read_at=time.time())
(release/'preflight.json').write_text(json.dumps(preflight,indent=2)+'\n')
command=[c['python'],'-u','-m','experiments.cvs_source_phase.run']; log=release/'phase_source.log'
with log.open('x') as stream:child=subprocess.Popen(command,cwd=release,env=env,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
receipt=dict(status='SUBMITTED',pid=child.pid,cwd=str(release),argv=command,log=str(log),run_root=str(run),commit=c['commit'])
(release/'submit.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))
'''


def publish(output):
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    folder = 'experiments/cvs_source_phase'
    if subprocess.check_output(['git', 'status', '--porcelain', '--', folder], cwd=d.ROOT, text=True).strip():
        raise ValueError('Commit actual source code/config before release')
    d.validate(d.read(d.ROOT / folder / 'registration.json')['rows'][0]['config'])
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=d.ROOT, text=True).strip()
    archive = output / (d.RELEASE + '.tar.gz')
    names = subprocess.check_output(['git', 'ls-files', folder], cwd=d.ROOT, text=True).splitlines()
    with archive.open('xb') as raw:
        with tarfile.open(fileobj=raw, mode='w:gz') as tar:
            for name in names:
                if Path(name).suffix in ('.py', '.json'): tar.add(d.ROOT / name, arcname=name, recursive=False)
            blob = (commit + '\n').encode(); member = tarfile.TarInfo('release_commit.txt'); member.size = len(blob); tar.addfile(member, io.BytesIO(blob))
    settings = dict(project=d.PROJECT, release=d.RELEASE, run=d.RUN, config=d.config(), archive=archive.name,
        sha256=hashlib.sha256(archive.read_bytes()).hexdigest(), commit=commit, python=PYTHON)
    d.write(output / 'package.json', settings)
    subprocess.run(['scp', *SSH, str(archive), 'N607:' + d.PROJECT + '/releases/' + archive.name], check=True, timeout=60)
    receipt = remote(REMOTE.replace('CONFIG', repr(settings)))
    d.write(output / 'submission.json', receipt); print(json.dumps(receipt))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--output', required=True)
    publish(parser.parse_args().output)
