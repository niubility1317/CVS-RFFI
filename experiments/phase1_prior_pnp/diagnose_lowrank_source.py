"""Fixed source-only low-rank posterior approximation diagnostic."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
import torch
import pipeline_unroll as pipeline
from phase1_unroll_cpu import load_fixed,LS
from lowrank_posterior import factors_from_covariance,lowrank_posterior

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--commit',required=True);a=p.parse_args()
 if a.out.exists():raise FileExistsError('Immutable source diagnostic output required')
 a.out.mkdir(parents=True);torch.set_num_threads(8);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;models=load_fixed(a.root,'cpu');recovery,teacher=models;cache,clean,rx,training,pv,ph=pipeline.s.source_contract(a.root);received,scene=pipeline.s.epoch_data(cache,19)
 with np.load(a.root/'runs'/LS/'predictions.npz',allow_pickle=False) as data:ids=data['indices'].copy()
 ids=ids[scene[ids]!='clean'][:160]
 if len(ids)!=160:raise ValueError('160 fixed impaired source records required')
 prior_cov=recovery[2][1];factor={k:factors_from_covariance(prior_cov,k) for k in (64,128,256)};original=pipeline.posterior;decisions=[];waves=[];times={}
 with torch.no_grad():
  for name in ('full','64','128','256'):
   if name=='full':pipeline.posterior=original
   else:
    def approximate(y,A,noise,means,cov,priors,*,f=factor[int(name)]):
     var=noise[:,0,0];eye=torch.eye(y.shape[1],dtype=y.dtype,device=y.device)
     if not torch.equal(noise,eye[None]*var[:,None,None]):raise ValueError('Isotropic observation noise required')
     return lowrank_posterior(y,A,var,means,f,priors)
    pipeline.posterior=approximate
   chunks=[];out=[];timing=[]
   for start in range(0,160,32):
    x=pipeline.core.tensor(received[ids[start:start+32]],'cpu');tick=time.perf_counter();raw=teacher.classify_features(teacher.features(x)).argmax(1);parent,wave,active=pipeline.evaluate_wave(x,recovery);pred=teacher.classify_features(teacher.features(wave)).argmax(1);seconds=time.perf_counter()-tick;chunks.append(torch.stack((raw,pred,active.long()),1).cpu().numpy());out.append(wave.cpu().numpy())
    if start:timing.append(seconds)
   decisions.append(np.concatenate(chunks));waves.append(np.concatenate(out));times[name]=timing;print('SOURCE_LOWRANK_DONE',name,round(sum(timing),3),flush=True)
 pipeline.posterior=original;decisions=np.stack(decisions);waves=np.stack(waves);np.savez_compressed(a.out/'predictions.npz',indices=ids,decisions=decisions,waves=waves,scene=scene[ids]);sha=hashlib.sha256((a.out/'predictions.npz').read_bytes()).hexdigest();result={'status':'SOURCE_PREDICTIONS_FIXED','commit':a.commit,'count':len(ids),'settings':['full','64','128','256'],'timings_seconds':times,'sha256':sha,'source_only':True,'TX_truth_reads':0,'U_s_truth_reads':0,'V_reads':0,'test_reads':0,'model_updates':0};(a.out/'prediction_complete.json').write_text(json.dumps(result,indent=2)+'\n');print('LOWRANK_SOURCE_COMPLETE',json.dumps(result),flush=True)
if __name__=='__main__':main()
