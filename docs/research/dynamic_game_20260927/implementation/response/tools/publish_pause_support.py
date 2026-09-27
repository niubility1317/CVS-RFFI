"""Land committed recovery support without launching any training or stopping jobs."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

REMOTE = r'''
import hashlib,json,subprocess,tarfile
from pathlib import Path
c=CONFIG
root=Path(c['project'])/'releases';release=root/c['release'];archive=root/c['archive']
if release.exists():raise FileExistsError(release)
if hashlib.sha256(archive.read_bytes()).hexdigest()!=c['sha256']:raise ValueError('transfer mismatch')
with tarfile.open(archive) as t:
 for m in t.getmembers():
  if not (root/m.name).resolve().is_relative_to(release.resolve()) or m.issym() or m.islnk():raise ValueError('unsafe archive')
 t.extractall(root)
subprocess.run(['/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python','-m','compileall','-q',str(release/'code'),str(release/'tools')],check=True)
print(json.dumps(dict(status='LANDED_COMPILED_NO_TRAINING',release=str(release),commit=c['commit'])))
'''


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    repo=Path(__file__).resolve().parents[3]
    def git(*args):return subprocess.check_output(['git',*args],cwd=repo,text=True).strip()
    commit=git('rev-parse','HEAD');branch=git('branch','--show-current')
    if git('ls-remote','origin','refs/heads/'+branch).split()[0]!=commit:raise ValueError('unpublished HEAD')
    if git('status','--porcelain','--','experiments/adv3b02_xuc'):raise ValueError('uncommitted release')
    a.output.mkdir(parents=True,exist_ok=True)
    name='adv3b02_pause_'+commit[:10];archive=a.output/(name+'.tar')
    subprocess.run(['git','archive','--format=tar','--prefix='+name+'/','--output='+str(archive),commit+':experiments/adv3b02_xuc'],cwd=repo,check=True)
    project='/home/szu2070436088/2510044040/CV-SincNet'
    c=dict(project=project,release=name,archive=archive.name,commit=commit,sha256=hashlib.sha256(archive.read_bytes()).hexdigest())
    conn=['-F','E:/type10-7/tools/n607_ssh_config','-o','BatchMode=yes','-o','ConnectTimeout=10']
    subprocess.run(['scp',*conn,str(archive),'N607:'+project+'/releases/'+archive.name],check=True)
    result=subprocess.run(['ssh',*conn,'-T','N607','python3 -'],input=REMOTE.replace('CONFIG',repr(c)).encode(),capture_output=True,timeout=55)
    (a.output/'landing.stdout').write_bytes(result.stdout);(a.output/'landing.stderr').write_bytes(result.stderr)
    print(result.stdout.decode());print(result.stderr.decode());result.check_returncode()


if __name__=='__main__':main()
