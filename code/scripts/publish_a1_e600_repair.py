"""Immutable release and single-owner dispatch of the authorized E600 pair."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
PROJECT = '/home/szu2070436088/2510044040/CV-SincNet'
PYTHON = '/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python'
RUN = 'a1_e600_repair_s392005_20260911_r1'
SSH_CONFIG = 'E:/type10-7/tools/n607_ssh_config'


def remote(source):
    compile(source, '<remote-release>', 'exec')
    result = subprocess.run(['ssh', '-F', SSH_CONFIG, '-o', 'BatchMode=yes',
        '-o', 'ConnectTimeout=10', 'N607', 'python3 -'], input=source, text=True,
        encoding='utf-8', capture_output=True, check=True, timeout=60)
    print(result.stdout, end='')
    return result.stdout


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['land', 'launch'])
    parser.add_argument('--release', required=True)
    args = parser.parse_args()
    if not args.release.replace('_', '').isalnum():
        raise ValueError('Invalid release name')
    release = PROJECT+'/releases/'+args.release
    evidence = ROOT/'analysis/e600_repair_20260911/release.json'
    if args.action == 'land':
        if evidence.exists():
            raise FileExistsError('Reconcile existing landing evidence before retry')
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
        archive = ROOT.parent/(args.release+'.tar.gz')
        if archive.exists():
            raise FileExistsError(archive)
        subprocess.run(['git', 'archive', '--format=tar.gz', '-o', str(archive), commit, 'code', 'configs'], cwd=ROOT, check=True)
        digest = hashlib.sha256(archive.read_bytes()).hexdigest()
        remote(f"""import json,shutil,subprocess
from pathlib import Path
p=Path({PROJECT!r});r=Path({release!r})
assert not r.exists() and not (p/'runs'/{RUN!r}).exists() and not (p/'logs'/{RUN!r}).exists()
assert not (p/'releases'/{archive.name!r}).exists()
assert shutil.disk_usage(p).free > 20*1024**3
print(json.dumps({{'free_bytes':shutil.disk_usage(p).free,'paths_absent':True,'gpu':subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)}}))
""")
        subprocess.run(['scp', '-F', SSH_CONFIG, '-o', 'BatchMode=yes', str(archive),
            'N607:'+PROJECT+'/releases/'+archive.name], check=True, timeout=180)
        result = json.loads(remote(f"""import hashlib,json,py_compile,tarfile
from pathlib import Path
archive=Path({(PROJECT+'/releases/'+archive.name)!r});release=Path({release!r})
assert hashlib.sha256(archive.read_bytes()).hexdigest()=={digest!r}
assert not release.exists()
with tarfile.open(archive) as tf:
    assert all(not m.name.startswith('/') and '..' not in Path(m.name).parts and not m.issym() and not m.islnk() for m in tf.getmembers())
    release.mkdir();tf.extractall(release)
for name in ['code/SSDG/train_ssdg.py','code/scripts/run_a1_fast_v2.py','code/scripts/run_a1_e600_repair.py','code/scripts/check_a1_e600_repair.py']:
    py_compile.compile(str(release/name),doraise=True)
(release/'release_commit.txt').write_text({commit!r}+'\\n',encoding='utf-8')
print(json.dumps({{'status':'LANDED','release':str(release),'commit':{commit!r},'sha256':{digest!r}}}))
"""))
        evidence.parent.mkdir(parents=True, exist_ok=True)
        evidence.write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    else:
        landed = json.loads(evidence.read_text(encoding='utf-8'))
        assert landed['release'] == release and landed['status'] == 'LANDED'
        remote(f"""import os,subprocess
from pathlib import Path
r=Path({release!r});p=Path({PROJECT!r})
assert (r/'release_commit.txt').read_text().strip()=={landed['commit']!r}
assert not (p/'runs'/{RUN!r}).exists() and not (p/'logs'/{RUN!r}).exists()
assert not (r/'dispatcher_process.json').exists() and not (r/'dispatcher.log').exists()
env=dict(os.environ,PYTHONPATH=str(r/'code')+os.pathsep+str(r))
subprocess.run([{PYTHON!r},'-u',str(r/'code/scripts/run_a1_e600_repair.py'),'--project-root',str(p),'--run-id',{RUN!r},'--detach'],cwd=r,env=env,check=True)
""")


if __name__ == '__main__':
    main()
