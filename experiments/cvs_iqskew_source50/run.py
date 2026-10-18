import os,sys,time,traceback,subprocess
from pathlib import Path
import numpy as np
from . import design as d
from .core import features,cells,evaluate,DELAYS,OMEGA
from .source import load_source

def main():
 subprocess.run([sys.executable,'-m','experiments.cvs_iqskew_source50.checks'],check=True,stdout=subprocess.DEVNULL)
 if d.BASE.exists():raise FileExistsError('Immutable output root exists;reconcile,never overwrite')
 d.BASE.mkdir();started=time.perf_counter()
 d.write(d.BASE/'checks.json',dict(status='PASS',tests=9,real_checkpoint='NOT_APPLICABLE_NO_MODEL_CONSUMED',source_IQ_used=0,NN=0,target_access=False))
 d.write(d.BASE/'active.json',dict(pid=os.getpid(),start_ticks=int(Path('/proc',str(os.getpid()),'stat').read_text().split()[21]),cwd=os.getcwd(),argv=sys.argv,CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES'),read_at=time.time()))
 try:
  import torch
  torch.set_num_threads(2);roles=load_source();obs={};coh={}
  d.write(d.BASE/'resolved_config.json',dict(config=d.config(),code_commit=(d.ROOT/'release_commit.txt').read_text().strip(),source_roles='EXACT_MATCH',cache_other_derived_state_deserialized_not_used=True,U_TX_not_accessed=True,python=sys.executable,CWD=os.getcwd(),GPU=None,NN=0,leo_practical_file='NOT_IMPORTED_NO_CHANNEL_OR_AUGMENTED_IQ_GENERATED'))
  for role,n in [('L_s',70),('V',300)]:
   r=roles[role];f=np.empty((len(r['ids']),4,127),np.float64)
   for j in range(0,len(f),64):f[j:j+64]=features(np.asarray(r['x'][j:j+64].tolist()))
   key='L' if role=='L_s' else 'V';coh[key]=cells(f,r['y'],r['rx'],r['day'],n)
   if role=='L_s':
    for h in (0,1):coh['L'+str(h)]=cells(f,r['y'],r['rx'],r['day'],n,h)
   obs.update({key+'_features':f,key+'_y':r['y'],key+'_rx':r['rx'],key+'_day':r['day'],key+'_ids':np.asarray(r['ids'])})
   d.write(d.BASE/'progress.json',dict(completed_role=role,source_packets_used=len(f),NN=0,training_updates=0,target_access=False));print('EXTRACTED '+role+' '+str(len(f)),flush=True)
  result,fitted=evaluate(coh['L'],coh['V'],coh['L0'],coh['L1']);wall=time.perf_counter()-started
  result.update(wall_seconds=wall,unique_IQ_packets_used=33300,L=6300,V=27000,feature_real_values=33300*4*127,model_forwards=0,training_updates=0,teacher_forwards=0,head_fits=0,new_IQ_views=0,source_RX_holdout=None,source_fold_fits=72,target_access=False,finite_window_law_is_necessary_statistical_model_not_a_physical_delay_estimate=True)
  np.savez_compressed(d.BASE/'observations.npz',**obs,delays=DELAYS,omega=OMEGA,**fitted);d.write(d.BASE/'result.json',result);print(result['decision'],flush=True)
  subprocess.run([sys.executable,str(Path(__file__).with_name('recount.py')),str(d.BASE)],check=True)
  d.write(d.BASE/'completion.json',dict(status='SOURCE_ANALYZED',decision=result['decision'],source_qualified=result['source_qualified'],unique_IQ_packets_used=33300,NN=0,training_updates=0,target_access=False,source_wall_seconds=wall,wall_seconds=time.perf_counter()-started,independent_recount='PASS'))
 except Exception as exc:
  d.write(d.BASE/'failure.json',dict(error_type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc(),preserve_outputs=True));raise
if __name__=='__main__':main()
