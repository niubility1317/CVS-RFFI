import os,sys,time,traceback,subprocess
from pathlib import Path
import numpy as np
import torch
from . import design as d
from .core import iq_complex,bound_from_L,curves,cell_means,evaluate

def load_source():
 expected=d.read(d.SOURCE);packed=torch.load(d.CACHE,map_location='cpu',weights_only=False)
 if any(packed['contract'].get(k)!=v for k,v in expected.items()) or packed['contract'].get('native_role_comparison')!='EXACT_MATCH':raise ValueError('Unchanged source carrier contract mismatch')
 roles={}
 for name,key,n,count in [('L_s','labeled',6300,70),('V','validation',27000,300)]:
  source=packed[key];out={k:source[k] for k in ('x','y','rx','day','ids')}
  if out['x'].shape!=(n,2,256) or len(set(out['ids']))!=n or set(out['ids'])!=set(expected['role_ids'][name]):raise ValueError('Exact source role ID mismatch')
  for k in ('y','rx','day'):out[k]=np.asarray(out[k].tolist(),np.int64)
  if any(int(((out['y']==t)&(out['rx']==r)&(out['day']==day)).sum())!=count for t in range(6) for r in d.RX for day in (1,2,3)):raise ValueError('All90 source cells required')
  roles[name]=out
 if set(roles['L_s']['ids'])&set(roles['V']['ids']) or (set(roles['L_s']['ids'])|set(roles['V']['ids']))&set(expected['role_ids']['U_s']):raise ValueError('Source roles overlap')
 del packed
 return roles

def main():
 # Stage consumes no model/checkpoint/query; synthetic source-statistic smoke is sufficient.
 subprocess.run([sys.executable,'-m','experiments.cvs_ampm_source48.checks'],check=True,stdout=subprocess.DEVNULL)
 if d.BASE.exists():raise FileExistsError('Immutable output root already exists')
 d.BASE.mkdir();started=time.perf_counter();torch.set_num_threads(2)
 d.write(d.BASE/'checks.json',dict(status='PASS',tests=7,real_checkpoint='NOT_APPLICABLE_NO_MODEL_CONSUMED',source_IQ=0,NN=0,target_access=False))
 d.write(d.BASE/'active.json',dict(pid=os.getpid(),start_ticks=int(Path('/proc',str(os.getpid()),'stat').read_text().split()[21]),cwd=os.getcwd(),argv=sys.argv,CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES'),read_at=time.time()))
 print('SOURCE_STATISTIC_SMOKE PASS;load unchanged legal L/V carrier',flush=True)
 try:
  roles=load_source();L=roles['L_s'];zl=iq_complex(np.asarray(L['x'].tolist()));bound,quantiles=bound_from_L(zl,L['rx']);del zl
  beta=np.linspace(-bound,bound,21);beta[10]=0
  d.write(d.BASE/'resolved_config.json',dict(config=d.config(),actual_beta_bound=bound,L_each_RX_power_quantile=quantiles.tolist(),beta=beta.tolist(),u=[-.5,0,.5],code_commit=(d.ROOT/'release_commit.txt').read_text().strip(),source_roles='EXACT_MATCH',cache_contains_only_source_roles=True,cache_other_derived_state_deserialized_not_used=True,U_TX_hidden=True,python=sys.executable,CWD=os.getcwd(),GPU=None,NN=0,leo_practical_file='NOT_IMPORTED_NO_CHANNEL_GENERATED'))
  obs={};cells={}
  for role,count in [('L_s',70),('V',300)]:
   r=roles[role];f=np.empty((len(r['ids']),21,3),np.float64)
   for start in range(0,len(f),64):f[start:start+64]=curves(iq_complex(np.asarray(r['x'][start:start+64].tolist())),beta)
   cells[role]=cell_means(f,r['y'],r['rx'],r['day'],count)
   prefix='L' if role=='L_s' else 'V'
   obs.update({prefix+'_features':f,prefix+'_y':r['y'],prefix+'_rx':r['rx'],prefix+'_day':r['day'],prefix+'_ids':np.asarray(r['ids'])})
   d.write(d.BASE/'progress.json',dict(completed_role=role,source_packets=len(f),NN=0,training_updates=0,target_access=False))
   print('EXTRACTED '+role+' packets='+str(len(f)),flush=True)
  result,fitted=evaluate(cells['L_s'],cells['V'],beta);wall=time.perf_counter()-started
  result.update(wall_seconds=wall,unique_IQ=33300,L=6300,V=27000,feature_values=33300*21*3,adjacent_terms=33300*21*3*255,model_forwards=0,training_updates=0,teacher_forwards=0,head_fits=0,source_RX_holdout=None,source_fold_fits=72,target_access=False)
  np.savez_compressed(d.BASE/'observations.npz',**obs,beta=beta,L_cells=cells['L_s'],V_cells=cells['V'],**fitted)
  d.write(d.BASE/'result.json',result);print(result['decision'],flush=True)
  subprocess.run([sys.executable,str(Path(__file__).with_name('recount.py')),str(d.BASE)],check=True)
  d.write(d.BASE/'completion.json',dict(status='SOURCE_ANALYZED',decision=result['decision'],source_qualified=result['source_qualified'],unique_IQ=33300,NN=0,training_updates=0,target_access=False,source_wall_seconds=wall,wall_seconds=time.perf_counter()-started,independent_recount='PASS'))
 except Exception as exc:
  d.write(d.BASE/'failure.json',dict(error_type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc(),preserve_outputs=True));raise
if __name__=='__main__':main()
