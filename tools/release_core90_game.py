"""Reviewed N607 bootstrap for one authorized immutable source experiment.

Upload this committed file and a git-archive first. No remote code patching,
installation, deletion, overwrite, target evaluation, or process termination.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive',required=True)
    p.add_argument('--sha256',required=True)
    p.add_argument('--project',required=True)
    p.add_argument('--release',required=True)
    p.add_argument('--runs-root',required=True)
    p.add_argument('--commit',required=True)
    p.add_argument('--seeds',default='392002,392003')
    p.add_argument('--rows',default='B0,B1,B2,B4,B5,B6,S1,S3,S4,C1,C2')
    p.add_argument('--max-processes-per-gpu',type=int,default=2)
    a=p.parse_args()
    project=Path(a.project).resolve()
    release=Path(a.release).resolve()
    runs=Path(a.runs_root).resolve()
    for path in (release,runs):
        if project not in path.parents or path.exists():
            raise ValueError('Expected fresh child directory of verified project: '+str(path))
    archive=Path(a.archive)
    if hashlib.sha256(archive.read_bytes()).hexdigest()!=a.sha256:
        raise ValueError('Transfer checksum mismatch')
    dataset=project/'Dataset_WigSig/ManySig.pkl'
    if not dataset.is_file(): raise FileNotFoundError(dataset)
    with zipfile.ZipFile(archive) as z:
        for info in z.infolist():
            destination=(release/info.filename).resolve()
            if release not in destination.parents:
                raise ValueError('Archive path escapes release')
        release.mkdir(parents=True)
        z.extractall(release)
    runs.mkdir(parents=True)
    (release/'release.json').write_text(json.dumps(dict(commit=a.commit,archive_sha256=a.sha256,source_only=True,project=str(project)),indent=2))
    subprocess.run([sys.executable,'-m','compileall','-q','code/cvsrffi/game_tracking','code/SSDG/train_core90_game.py','code/scripts/dispatch_core90_game.py'],cwd=release,check=True)
    subprocess.run([sys.executable,'code/scripts/build_core90_game_matrix.py','--output-dir',str(runs/'matrix'),
                    '--runs-root',str(runs/'runs'),'--dataset-path',str(dataset),'--seeds',a.seeds,'--rows',a.rows],cwd=release,check=True)
    log=(runs/'dispatcher.stdout.log').open('x')
    argv=[sys.executable,'code/scripts/dispatch_core90_game.py','--matrix',str(runs/'matrix/matrix.json'),
          '--release',str(release),'--status',str(runs/'queue_status.json'),'--max-processes-per-gpu',str(a.max_processes_per_gpu)]
    process=subprocess.Popen(argv,cwd=release,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,
                             env=dict(os.environ,OMP_NUM_THREADS='4',MKL_NUM_THREADS='4'))
    log.close()
    launch=dict(status='DISPATCH_REQUESTED_NOT_YET_VERIFIED',pid=process.pid,argv=argv,release=str(release),runs_root=str(runs),commit=a.commit)
    (runs/'launch.json').write_text(json.dumps(launch,indent=2))
    print(json.dumps(launch))


if __name__=='__main__': main()
