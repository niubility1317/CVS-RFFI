"""Native Windows release transport: one archive comparison, compile, then launch."""
import argparse,hashlib,io,json,os,subprocess,tarfile
from pathlib import Path
def main():
 p=argparse.ArgumentParser();p.add_argument('--stage',choices=['transfer','launch'],required=True);p.add_argument('--commit',required=True);p.add_argument('--source-root',type=Path,required=True);p.add_argument('--artifact-root',type=Path,required=True);p.add_argument('--project-root',required=True);p.add_argument('--ssh-config',required=True);a=p.parse_args()
 O=a.artifact_root;W=a.source_root;P=a.project_root
 release='cvs_FcrossRX_ampm_source_20261018_r01';run='20261018-diagnostic-FcrossRX-ampm-LtoV-manysig-m1-r01'
 R=P+'/releases/'+release;B=P+'/runs/'+run;L=B+'-launch';archive=O/(release+'.tar.gz')
 env=dict(os.environ,SystemRoot='C:/Windows',windir='C:/Windows',PATHEXT='.COM;.EXE;.BAT;.CMD')
 ssh=['C:/Program Files/Git/usr/bin/ssh.exe','-F',a.ssh_config,'-o','BatchMode=yes','-o','ConnectTimeout=10','-T','N607','python3 -']
 if a.stage=='transfer':
  if archive.exists():raise FileExistsError('Archive exists; independently reconcile before retry')
  with tarfile.open(archive,'w:gz') as t:
   for f in sorted((W/'experiments/cvs_ampm_source48').rglob('*.py')):t.add(f,arcname=f.relative_to(W).as_posix(),recursive=False)
   marker=tarfile.TarInfo('experiments/__init__.py');marker.size=0;t.addfile(marker,io.BytesIO(b''))
   value=(a.commit+'\n').encode();info=tarfile.TarInfo('release_commit.txt');info.size=len(value);t.addfile(info,io.BytesIO(value))
  sha=hashlib.sha256(archive.read_bytes()).hexdigest();A=P+'/releases/'+archive.name
  subprocess.run(['C:/Program Files/Git/usr/bin/scp.exe','-S','C:/Program Files/Git/usr/bin/ssh.exe','-F',a.ssh_config,'-o','BatchMode=yes','-o','ConnectTimeout=10',str(archive),'N607:'+A],env=env,check=True,timeout=40)
  script="import hashlib,json,subprocess,tarfile\nfrom pathlib import Path\nR=Path("+repr(R)+");B=Path("+repr(B)+");L=Path("+repr(L)+");A=Path("+repr(A)+")\nif R.exists() or B.exists() or L.exists():raise FileExistsError('Own output/release collision;reconcile')\nactual=hashlib.sha256(A.read_bytes()).hexdigest()\nif actual!="+repr(sha)+":raise ValueError('One archive transport comparison failed')\nR.mkdir()\nwith tarfile.open(A) as t:t.extractall(R)\nsubprocess.run(['/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python','-m','compileall','-q',str(R/'experiments')],check=True)\nprint(json.dumps(dict(status='VERIFIED',archive_SHA=actual,compile='PASS',release=str(R),run_absent=not B.exists(),log_absent=not L.exists())))\n"
 else:
  script="import json,os,subprocess,time\nfrom pathlib import Path\nR=Path("+repr(R)+");B=Path("+repr(B)+");L=Path("+repr(L)+")\nif B.exists() or L.exists():raise FileExistsError('Own run collision,no duplicate')\nif (R/'release_commit.txt').read_text().strip()!="+repr(a.commit)+":raise ValueError('Released code commit differs')\nL.mkdir();env=dict(os.environ,PYTHONPATH=str(R),CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')\nargs=['/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python','-u','-m','experiments.cvs_ampm_source48.run']\nwith (L/'stdout.log').open('xb') as out,(L/'stderr.log').open('xb') as err:p=subprocess.Popen(args,cwd=R,env=env,stdout=out,stderr=err,start_new_session=True)\nq=dict(status='DISPATCHED',pid=p.pid,start_ticks=int(Path('/proc',str(p.pid),'stat').read_text().split()[21]),cwd=str(R),argv=args,CUDA_VISIBLE_DEVICES=env['CUDA_VISIBLE_DEVICES'],read_at=time.time())\n(L/'launch.json').write_text(json.dumps(q,indent=2)+'\\n');print(json.dumps(q))\n"
 r=subprocess.run(ssh,input=script.encode(),capture_output=True,env=env,timeout=50);(O/(a.stage+'.stdout.log')).write_bytes(r.stdout);(O/(a.stage+'.stderr.log')).write_bytes(r.stderr)
 if r.returncode:raise RuntimeError(r.stderr.decode(errors='replace'))
 result=json.loads(r.stdout);(O/(a.stage+'.json')).write_text(json.dumps(result,indent=2)+'\n',encoding='utf8');print(json.dumps(result))
if __name__=='__main__':main()
