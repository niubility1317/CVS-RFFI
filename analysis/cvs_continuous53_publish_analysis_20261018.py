"""Native Windows transport for the independent, CPU-only analysis owner."""
import argparse,hashlib,io,json,os,subprocess,tarfile
from pathlib import Path
FILES=['cvs_continuous53_results_20261018.py','cvs_continuous52_results_20261018.py','cvs_continuous52_coverage_20261018.py','cvs_continuous53_pipeline_20261018.py','cvs_continuous53_publish_analysis_20261018.py','check_continuous53_analysis.py','cvs_impulse_factorial_E30_results_20261018.py','cvs_fullsource_E30_results_20261017.py','cvs_crossviewSupCon_state_coverage_20261018.py']
PROJECT='/home/szu2070436088/2510044040/CV-SincNet'
RELEASE=PROJECT+'/releases/cvs_FcrossRX_continuous_recovery_analysis_20261018_r01'
OUTPUT=PROJECT+'/runs/20261018-phase1-FcrossRX-continuous-recovery-e30-s2026092701-m1-r01/analysis_auto_r01'
TRUTH=PROJECT+'/runs/20260927-phase1-baselines-final-clean-satellite-m5-r01/data/truth.json'
PYTHON='/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python'
def main():
 p=argparse.ArgumentParser();p.add_argument('--stage',choices=['transfer','launch'],required=True);p.add_argument('--commit',required=True);p.add_argument('--source-root',type=Path,required=True);p.add_argument('--artifact-root',type=Path,required=True);p.add_argument('--ssh-config',required=True);a=p.parse_args()
 R=RELEASE;B=OUTPUT;L=B+'-launch';O=a.artifact_root;archive=O/'continuous53_analysis_r01.tar.gz';env=dict(os.environ,SystemRoot='C:/Windows',windir='C:/Windows')
 ssh=['C:/Program Files/Git/usr/bin/ssh.exe','-F',a.ssh_config,'-o','BatchMode=yes','-o','ConnectTimeout=10','-T','N607','python3 -']
 if a.stage=='transfer':
  if archive.exists():raise FileExistsError('Reconcile existing archive before another transfer')
  with tarfile.open(archive,'x:gz') as t:
   for name in FILES:t.add(a.source_root/'analysis'/name,arcname=name,recursive=False)
   value=(a.commit+'\n').encode();info=tarfile.TarInfo('release_commit.txt');info.size=len(value);t.addfile(info,io.BytesIO(value))
  sha=hashlib.sha256(archive.read_bytes()).hexdigest();remote_archive=PROJECT+'/releases/continuous53_analysis_r01.tar.gz'
  subprocess.run(['C:/Program Files/Git/usr/bin/scp.exe','-S','C:/Program Files/Git/usr/bin/ssh.exe','-F',a.ssh_config,'-o','BatchMode=yes','-o','ConnectTimeout=10',str(archive),'N607:'+remote_archive],env=env,check=True,timeout=40)
  script="import hashlib,json,subprocess,tarfile,os\nfrom pathlib import Path\nR=Path("+repr(R)+");B=Path("+repr(B)+");L=Path("+repr(L)+");A=Path("+repr(remote_archive)+")\nif R.exists() or B.exists() or L.exists():raise FileExistsError('Owned analysis output/release collision')\nactual=hashlib.sha256(A.read_bytes()).hexdigest()\nif actual!="+repr(sha)+":raise ValueError('One archive transport comparison failed')\nR.mkdir()\nwith tarfile.open(A) as t:t.extractall(R)\nsubprocess.run(["+repr(PYTHON)+",'-m','compileall','-q',str(R)],check=True)\nsubprocess.run(["+repr(PYTHON)+",'-c','import cvs_continuous53_results_20261018,cvs_continuous52_coverage_20261018,cvs_continuous53_pipeline_20261018'],cwd=R,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONPATH=str(R)),check=True)\nprint(json.dumps(dict(status='VERIFIED',archive_SHA=actual,compile='PASS',imports='PASS',analysis_output_absent=not B.exists())))\n"
 else:
  script="import json,os,subprocess,time\nfrom pathlib import Path\nR=Path("+repr(R)+");B=Path("+repr(B)+");L=Path("+repr(L)+")\nif B.exists() or L.exists():raise FileExistsError('Existing analysis owner;reconcile')\nif (R/'release_commit.txt').read_text().strip()!="+repr(a.commit)+":raise ValueError('Actual analysis code commit differs')\nL.mkdir();env=dict(os.environ,PYTHONPATH=str(R),CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')\nargs=["+repr(PYTHON)+",'-u',str(R/'cvs_continuous53_pipeline_20261018.py'),'--output',str(B),'--truth',"+repr(TRUTH)+",'--code-commit',"+repr(a.commit)+"]\nwith (L/'stdout.log').open('xb') as out,(L/'stderr.log').open('xb') as err:p=subprocess.Popen(args,cwd=R,env=env,stdout=out,stderr=err,start_new_session=True)\nrecord=dict(status='DISPATCHED',pid=p.pid,start_ticks=int(Path('/proc',str(p.pid),'stat').read_text().split()[21]),cwd=str(R),argv=args,CUDA_VISIBLE_DEVICES=env['CUDA_VISIBLE_DEVICES'],read_at=time.time())\n(L/'launch.json').write_text(json.dumps(record,indent=2)+'\\n');print(json.dumps(record))\n"
 r=subprocess.run(ssh,input=script.encode(),capture_output=True,env=env,timeout=50);(O/(a.stage+'.stdout.log')).write_bytes(r.stdout);(O/(a.stage+'.stderr.log')).write_bytes(r.stderr)
 if r.returncode:raise RuntimeError(r.stderr.decode(errors='replace'))
 result=json.loads(r.stdout);(O/(a.stage+'.json')).write_text(json.dumps(result,indent=2)+'\n',encoding='utf8');print(json.dumps(result))
if __name__=='__main__':main()
