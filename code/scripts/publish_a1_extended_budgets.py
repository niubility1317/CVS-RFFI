"""Local-first immutable release of this authorized extension matrix."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
PROJECT='/home/szu2070436088/2510044040/CV-SincNet'
PYTHON='/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python'
RUN='a1_extended_s392005_20260910_r1'


def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['land','launch'])
    parser.add_argument('--release',required=True);args=parser.parse_args()
    if not args.release.replace('_','').isalnum():raise ValueError('Invalid release name')
    release=PROJECT+'/releases/'+args.release
    evidence=ROOT/'analysis/a1_extended_release.json'
    if args.action=='land':
        commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
        archive=ROOT.parent/(args.release+'.tar.gz')
        if archive.exists():raise FileExistsError(archive)
        subprocess.run(['git','archive','--format=tar.gz','-o',str(archive),commit,'code','configs'],cwd=ROOT,check=True)
        digest=hashlib.sha256(archive.read_bytes()).hexdigest()
        preflight=f"""import json,shutil
from pathlib import Path
p=Path({PROJECT!r});r=Path({release!r})
assert not r.exists() and not (p/'runs'/{RUN!r}).exists() and not (p/'logs'/{RUN!r}).exists()
assert not (p/'releases'/{archive.name!r}).exists()
print(json.dumps({{'free_bytes':shutil.disk_usage(p).free,'release_absent':True,'run_absent':True}}))
"""
        remote(preflight)
        subprocess.run(['scp','-o','BatchMode=yes',str(archive),'N607:'+PROJECT+'/releases/'+archive.name],check=True,timeout=180)
        payload=f"""import hashlib,json,py_compile,tarfile
from pathlib import Path
archive=Path({(PROJECT+'/releases/'+archive.name)!r}); release=Path({release!r})
assert hashlib.sha256(archive.read_bytes()).hexdigest()=={digest!r}
assert not release.exists()
with tarfile.open(archive) as tf:
    assert all(not m.name.startswith('/') and '..' not in Path(m.name).parts and not m.issym() and not m.islnk() for m in tf.getmembers())
    release.mkdir(); tf.extractall(release)
files=['code/SSDG/train_ssdg.py','code/cvsrffi/a1_budget_schedule.py','code/cvsrffi/a1_periodic_target.py',
       'code/scripts/run_a1_fast_v2.py','code/scripts/run_a1_extended_budgets.py','code/scripts/check_a1_extended_budgets.py']
for name in files:py_compile.compile(str(release/name),doraise=True)
(release/'release_commit.txt').write_text({commit!r}+'\\n',encoding='utf-8')
print(json.dumps({{'status':'LANDED','release':str(release),'commit':{commit!r},'sha256':{digest!r},'compiled':files}}))
"""
        result=json.loads(remote(payload));result['local_archive']=str(archive)
        evidence.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    else:
        landed=json.loads(evidence.read_text(encoding='utf-8'))
        assert landed['release']==release and landed['status']=='LANDED'
        payload=f"""import json,os,subprocess
from pathlib import Path
r=Path({release!r});p=Path({PROJECT!r})
assert (r/'release_commit.txt').read_text().strip()=={landed['commit']!r}
assert not (p/'runs'/{RUN!r}).exists() and not (p/'logs'/{RUN!r}).exists()
env=dict(os.environ,PYTHONPATH=str(r/'code')+os.pathsep+str(r))
subprocess.run([{PYTHON!r},'-u',str(r/'code/scripts/run_a1_extended_budgets.py'),
    '--project-root',str(p),'--run-id',{RUN!r},'--detach'],cwd=r,env=env,check=True)
"""
        print(remote(payload))


def remote(source):
    result=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10','N607','python3 -'],
        input=source,text=True,encoding='utf-8',capture_output=True,check=True,timeout=60)
    print(result.stdout, end='');return result.stdout


if __name__=='__main__':main()
