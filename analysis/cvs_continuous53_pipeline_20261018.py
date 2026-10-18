"""One-shot post-training analysis owner; never alters a training process."""
import argparse,json,os,subprocess,sys,time,traceback
from pathlib import Path
import cvs_continuous53_results_20261018 as a
ROOT=a.ROOT
TRAINING_RELEASE=a.PROJECT/'releases/cvs_FcrossRX_continuous_recovery_E30_20261018_r01'
HERE=Path(__file__).resolve().parent

class IncompleteRun(RuntimeError):pass

def read(path):return json.loads(Path(path).read_text(encoding='utf8'))
def write(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf8')

def metadata_ready(paths):
 for path in paths:
  try:read(path)
  except (FileNotFoundError,json.JSONDecodeError):return False
 return True

def source_metadata():
 return [r['root']/name for r in a.rows() for name in ['source/completion.json','source_qualification/complete.json']]+[ROOT/'source_matrix_decision.json']

def formal_metadata():
 return [r['root']/name for r in a.rows() for name in ['prediction/complete.json','scoring/complete.json','complete.json']]

def owned_handles():
 handles=[];unknown=False
 for path,key in [(ROOT/'dispatcher.json','dispatcher'),(ROOT/'launch.json','workers')]:
  try:q=read(path)
  except (FileNotFoundError,json.JSONDecodeError):unknown=True;continue
  ids=[q['pid']] if key=='dispatcher' else [r['pid'] for r in q['rows']]
  for pid in ids:
   p=Path('/proc',str(pid))
   try:
    if os.readlink(p/'cwd')==str(TRAINING_RELEASE) and 'cvs_continuous52/runner.py' in (p/'cmdline').read_bytes().replace(b'\0',b' ').decode():handles.append(pid)
   except FileNotFoundError:continue
   except OSError:unknown=True
 return sorted(set(handles)),unknown

def technical_failure():
 paths=[ROOT/'failure.json']+[r['root']/'source_qualification_failure.json' for r in a.rows() if not r['reused']]
 return next((str(p) for p in paths if p.exists()),None)

def wait_until(stage,paths,out,ready=metadata_ready,probe=owned_handles,failure=technical_failure,sleep=time.sleep):
 loops=0
 while not ready(paths):
  reason=failure()
  if reason:raise IncompleteRun('Known technical artifact: '+reason)
  alive,unknown=probe()
  if not alive and not unknown:raise IncompleteRun('All owned handles absent before '+stage+' metadata complete')
  state=dict(status='WAIT_'+stage,live_owned_PIDs=alive,observation_unknown=unknown,metadata_files_expected=len(paths),updated_at=time.time(),NN=0,target_scores_read=False)
  write(out/'state.json',state)
  if loops%4==0:print('WAIT_'+stage+' live='+str(len(alive))+' observation_unknown='+str(unknown),flush=True)
  sleep(15);loops+=1

def command(stage,out,truth):
 if stage=='source':return [sys.executable,str(HERE/'cvs_continuous53_results_20261018.py'),'--mode','source','--output',str(out/'source')]
 if stage=='coverage':return [sys.executable,str(HERE/'cvs_continuous53_results_20261018.py'),'--mode','coverage','--source-analysis',str(out/'source'),'--output',str(out/'coverage')]
 if stage=='final':return [sys.executable,str(HERE/'cvs_continuous53_results_20261018.py'),'--mode','final','--source-analysis',str(out/'source'),'--truth',str(truth),'--output',str(out/'final')]
 raise ValueError('Only fixed analysis stages')

def run_stage(stage,out,truth,env):
 args=command(stage,out,truth);print('START_'+stage.upper(),flush=True)
 with (out/(stage+'.stdout.log')).open('xb') as log,(out/(stage+'.stderr.log')).open('xb') as err:subprocess.run(args,cwd=HERE,env=env,stdout=log,stderr=err,check=True)
 print('DONE_'+stage.upper(),flush=True)

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--truth',type=Path,required=True);p.add_argument('--code-commit',required=True);args=p.parse_args();out=args.output
 if out.parent!=ROOT or out.name!='analysis_auto_r01':raise ValueError('Only owned immutable analysis output')
 if out.exists():raise FileExistsError('Reconcile existing analysis owner;never duplicate')
 out.mkdir();started=time.perf_counter();env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
 write(out/'active.json',dict(pid=os.getpid(),cwd=os.getcwd(),argv=sys.argv,analysis_code_commit=args.code_commit,training_run=a.RUN,new_model_updates=0,new_model_NN=0,truth_opened=False,scope='posttraining only;no process mutation',read_at=time.time()))
 coverage=None
 try:
  # Synthetic tests contain no real IQ/model/query inputs and never train.
  subprocess.run([sys.executable,str(HERE/'check_continuous53_analysis.py')],cwd=HERE,env=env,check=True)
  wait_until('SOURCE',source_metadata(),out)
  run_stage('source',out,args.truth,env)
  # Coverage is independent source metadata work; final scoring never waits on it.
  cvlog=(out/'coverage.stdout.log').open('xb');cverr=(out/'coverage.stderr.log').open('xb')
  try:coverage=subprocess.Popen(command('coverage',out,args.truth),cwd=HERE,env=env,stdout=cvlog,stderr=cverr,start_new_session=True)
  finally:cvlog.close();cverr.close()
  write(out/'coverage_launch.json',dict(pid=coverage.pid,NN=0,IQ_reads=0,source_only=True))
  wait_until('FORMAL',formal_metadata(),out)
  run_stage('final',out,args.truth,env)
  coverage_code=coverage.wait()
  write(out/'complete.json',dict(status='ANALYZED' if coverage_code==0 else 'PARTIAL_MECHANISM_COVERAGE',coverage_exit_code=coverage_code,source_and_final_complete=True,model_seed_count=1,NN=0,IQ_reads=0,checkpoint_reads=0,training_updates=0,analysis_code_commit=args.code_commit,pipeline_wall_including_wait_seconds=time.perf_counter()-started,goal_complete=False))
  print('PIPELINE_COMPLETE coverage_exit='+str(coverage_code),flush=True)
 except BaseException as error:
  # Training and any healthy source coverage child are preserved.
  write(out/'failure.json',dict(error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc(),training_processes_untouched=True,coverage_PID=coverage.pid if coverage is not None else None,no_automatic_retry=True,NN=0))
  raise
if __name__=='__main__':main()
