"""Two declared frozen models, sequential CPU; no training/target or relaunch."""
import os,subprocess,sys,time
from pathlib import Path
from . import design as d

def run():
 if sys.argv[1:] or os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('No role/operator/seed overrides')
 out=d.resolve(d.BASE);out.mkdir(parents=True,exist_ok=False);d.validate(d.config());d.write(out/'active.json',dict(pid=os.getpid(),cwd=os.getcwd(),argv=sys.argv,owner=d.OWNER));d.write(out/'request.json',d.config());rows=[]
 try:
  for seed in d.SEEDS:
   log=out/('s'+str(seed)+'.stdout.log');command=[sys.executable,'-u',str(d.ROOT/'experiments/cvs_isolated_awgn17/runner.py'),'--seed',str(seed)]
   with log.open('x') as stream:child=subprocess.Popen(command,cwd=d.ROOT,env=dict(os.environ),stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT)
   row=dict(model_seed=seed,pid=child.pid,cwd=str(d.ROOT),argv=command,log=str(log));rows.append(row);d.write(out/'launch.json',dict(rows=rows));code=child.wait()
   if code:raise RuntimeError('Source guard worker failed;preservepartial;noautomaticretry seed'+str(seed)+' exit'+str(code))
   if not (out/('s'+str(seed))/'completion.json').is_file():raise ValueError('Missing source worker completion')
  results={str(seed):d.read(out/('s'+str(seed))/'result.json') for seed in d.SEEDS};decision='PASS' if all(v['decision']=='PASS' for v in results.values()) else 'FAIL';cost=sum(v['actual_NN_packets'] for v in results.values())
  if cost!=50402:raise ValueError('Total actual NN cost differs')
  d.write(out/'result.json',dict(status='ANALYZED_SOURCE_ISOLATED_NOISE_ONLY',decision=decision,models={k:v['qualification'] for k,v in results.items()},actual_NN_packets=cost,unique_source_L=5040,source_decisions=50400,native_noise_generations=40320,synthetic_canonical_native_noise_generations=128,old_ordinary_generated=0,PA_generated=0,As_generated=0,checkpoint_loads=2,optimizer_updates=0,head_fit_updates=0,target_access=False,Ag_efficacy_proved=False,joint_goal_complete=False,automatic_training=False))
  d.write(out/'completion.json',dict(status='ANALYZED_SOURCE_ISOLATED_NOISE_ONLY',decision=decision,models=2,source_decisions=50400,actual_NN_packets=cost,optimizer_updates=0,target_access=False,automatic_training=False));print('COMPLETE '+decision,flush=True)
 except Exception as error:d.write(out/'failure.json',dict(status='FAILED',error=repr(error),no_automatic_retry=True));raise
if __name__=='__main__':run()
