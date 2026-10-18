"""One immutable zero-NN CPU analysis release;never alters training processes."""
import argparse,hashlib,io,json,subprocess,tarfile
from pathlib import Path
RUN='20261018-phase1-FcrossRX-satelliteDropout-e30-s2026092701-m2-r01'
RELEASE='cvs_F_satellite_dropout_analysis_20261018_r03'
OUTPUT='analysis_auto_r03'
FILES=['cvs_satellite_dropout_results_20261018.py','cvs_satellite_dropout_logs_20261018.py','cvs_satellite_dropout_figures_20261018.py','cvs_satellite_dropout_pipeline_20261018.py','check_satellite_dropout_results_20261018.py','check_satellite_dropout_logs_20261018.py','check_satellite_dropout_pipeline_20261018.py','cvs_satellite_dropout_publish_analysis_20261018.py']

def analysis_config(rt,commit):
 output=rt['project']+'/runs/'+RUN+'/'+OUTPUT
 return dict(project=rt['project'],release=rt['project']+'/releases/'+RELEASE,archive=rt['project']+'/releases/'+RELEASE+'.tar.gz',output=output,launch=output+'-launch',python=rt['python'],user=rt['user'],host=rt['host'],commit=commit,truth=rt['project']+'/runs/20260927-phase1-baselines-final-clean-satellite-m5-r01/data/truth.json')

def remote(script,rt):
 r=subprocess.run(['ssh','-F',rt['ssh_config'],'-o','BatchMode=yes','-o','ConnectTimeout=10','-T','N607',rt['python']+' -'],input=script.encode('utf8'),capture_output=True,timeout=55)
 if r.returncode:raise RuntimeError(r.stderr.decode('utf8',errors='replace'))
 return json.loads(r.stdout)
def main():
 p=argparse.ArgumentParser();p.add_argument('--stage',choices=['transfer','launch'],required=True);p.add_argument('--runtime',type=Path,required=True);p.add_argument('--source-root',type=Path,required=True);p.add_argument('--artifact-root',type=Path,required=True);p.add_argument('--commit',required=True);args=p.parse_args();rt=json.loads(args.runtime.read_text(encoding='utf8'));root=args.source_root;local=args.artifact_root;local.mkdir(parents=True,exist_ok=True)
 c=analysis_config(rt,args.commit)
 if args.stage=='transfer':
  script=r'''
import json,os,subprocess
from pathlib import Path
c=CONFIG
if os.geteuid()==0 or subprocess.check_output(['whoami'],text=True).strip()!=c['user'] or subprocess.check_output(['hostname'],text=True).strip()!=c['host']:raise ValueError('Expected ordinary N607 account required')
if any(Path(c[k]).exists() for k in ('release','archive','output','launch')):raise FileExistsError('Existing analysis;reconcile,never resubmit')
if not (Path(c['project'])/'runs/20261018-phase1-FcrossRX-satelliteDropout-e30-s2026092701-m2-r01/dispatcher.json').is_file():raise ValueError('Registered real training run missing')
print(json.dumps(dict(status='VERIFIED',ordinary_account=True,training_processes_untouched=True,target_truth_read=False,CP=0,NN=0)))
'''
  preflight=remote(script.replace('CONFIG',repr(c)),rt);(local/'preflight.json').write_text(json.dumps(preflight,indent=2)+'\n',encoding='utf8')
  if subprocess.check_output(['git','status','--porcelain','--',*['analysis/'+f for f in FILES]],cwd=root,text=True).strip():raise ValueError('Commit actual analysis code before release')
  raw=subprocess.check_output(['git','archive',args.commit,*['analysis/'+f for f in FILES]],cwd=root);archive=local/(RELEASE+'.tar.gz')
  with tarfile.open(fileobj=io.BytesIO(raw)) as source,tarfile.open(archive,'w:gz') as target:
   for member in source.getmembers():
    if member.isfile():target.addfile(member,source.extractfile(member))
   data=(args.commit+'\n').encode();info=tarfile.TarInfo('release_commit.txt');info.size=len(data);target.addfile(info,io.BytesIO(data))
  c['sha256']=hashlib.sha256(archive.read_bytes()).hexdigest();subprocess.run(['scp','-F',rt['ssh_config'],'-o','BatchMode=yes','-o','ConnectTimeout=10',str(archive),'N607:'+c['archive']],capture_output=True,check=True,timeout=55)
  script=r'''
import hashlib,json,os,re,subprocess,tarfile
from pathlib import Path
c=CONFIG;release=Path(c['release'])
if release.exists() or Path(c['output']).exists() or Path(c['launch']).exists():raise FileExistsError('Existing mutation;reconcile')
if hashlib.sha256(Path(c['archive']).read_bytes()).hexdigest()!=c['sha256']:raise ValueError('Single archive transfer mismatch')
release.mkdir()
with tarfile.open(c['archive']) as source:
 for member in source.getmembers():
  if not member.isfile() or not (release/member.name).resolve().is_relative_to(release.resolve()):raise ValueError('Invalid archive member')
 source.extractall(release)
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2');cwd=release/'analysis'
subprocess.run([c['python'],'-m','compileall','-q',str(cwd)],cwd=cwd,env=env,check=True)
with (release/'checks.stdout.log').open('x') as out,(release/'checks.stderr.log').open('x') as err:subprocess.run([c['python'],'-m','unittest','check_satellite_dropout_results_20261018','check_satellite_dropout_pipeline_20261018'],cwd=cwd,env=env,stdout=out,stderr=err,check=True)
checks=re.search(r'Ran (\d+) tests?',(release/'checks.stderr.log').read_text())
if checks is None:raise ValueError('Missing unittest result count')
result=dict(status='TRANSFER_VERIFIED',commit=c['commit'],archive_SHA256=c['sha256'],remote_compile='PASS',remote_imports='PASS',remote_unit_tests='PASS_'+checks.group(1),IQ=0,CP=0,NN=0,updates=0,target_truth_read=False)
(release/'transfer.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
'''
  result=remote(script.replace('CONFIG',repr(c)),rt);(local/'transfer.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8');print(json.dumps(result))
 else:
  script=r'''
import json,os,subprocess,time
from pathlib import Path
c=CONFIG;release=Path(c['release']);out=Path(c['output']);launch=Path(c['launch']);cwd=release/'analysis'
if out.exists() or launch.exists():raise FileExistsError('Existing owner/output;reconcile,no duplicate')
if (release/'release_commit.txt').read_text().strip()!=c['commit'] or json.loads((release/'transfer.json').read_text())['status']!='TRANSFER_VERIFIED':raise ValueError('Analysis release identity differs')
launch.mkdir();env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2');cmd=[c['python'],'-u',str(cwd/'cvs_satellite_dropout_pipeline_20261018.py'),'--project',c['project'],'--output',str(out),'--truth',c['truth'],'--code-commit',c['commit']]
with (launch/'stdout.log').open('x') as stdout,(launch/'stderr.log').open('x') as stderr:child=subprocess.Popen(cmd,cwd=cwd,env=env,stdin=subprocess.DEVNULL,stdout=stdout,stderr=stderr,start_new_session=True)
result=dict(status='SUBMITTED',pid=child.pid,cwd=str(cwd),argv=cmd,analysis_code_commit=c['commit'],training_processes_untouched=True,NN=0)
(launch/'launch.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
'''
  result=remote(script.replace('CONFIG',repr(c)),rt);(local/'launch.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8');print(json.dumps({k:v for k,v in result.items() if k not in ('cwd','argv')}))
if __name__=='__main__':main()
