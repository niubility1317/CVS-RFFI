"""Time fixed source-only unrolled predictions; never access ground test or TX labels."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
import torch
from phase1_unroll_cpu import load_fixed,predict_batch,s,core,LS

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--threads',type=int,choices=[1,2,4,8],required=True);a=p.parse_args()
 if a.out.exists():raise FileExistsError('Immutable benchmark output required')
 torch.set_num_threads(a.threads);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;models=load_fixed(a.root,'cpu');cache,clean,rx,training,pv,ph=s.source_contract(a.root);received,scene=s.epoch_data(cache,19)
 with np.load(a.root/'runs'/LS/'predictions.npz',allow_pickle=False) as data:ids=data['indices'].copy()
 ids=ids[scene[ids]!='clean'][:160]
 if len(ids)!=160:raise ValueError('160 fixed impaired source records required')
 chunks=[];elapsed=[]
 with torch.no_grad():
  for start in range(0,160,32):
   x=core.tensor(received[ids[start:start+32]],'cpu');tick=time.perf_counter();raw,pred,active=predict_batch(x,models);dt=time.perf_counter()-tick;chunks.append(torch.stack((raw,pred,active.long()),1).cpu().numpy())
   if start:elapsed.append(dt)
 arr=np.concatenate(chunks);a.out.mkdir(parents=True);np.save(a.out/'indices.npy',ids,allow_pickle=False);np.save(a.out/'decisions.npy',arr,allow_pickle=False);summary=dict(status='PASS',threads=a.threads,batch_size=32,warmup_records=32,timed_records=128,seconds=elapsed,records_per_second=128/sum(elapsed),active=int(arr[32:,2].sum()),decision_sha256=hashlib.sha256(arr.tobytes()).hexdigest(),source_only=True,TX_labels_read=0,U_s_truth_reads=0,V_reads=0,test_reads=0,caveat='Shared host timing, previously explored source records, source decisions only.')
 (a.out/'result.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary),flush=True)
if __name__=='__main__':main()
