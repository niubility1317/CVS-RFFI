"""Independently recount fixed source pseudo metrics without identity truth."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import torch

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--source',type=Path,required=True);ap.add_argument('--score',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
 if a.out.exists():raise FileExistsError(a.out)
 m=json.loads((a.source/'result.json').read_text());sc=json.loads(a.score.read_text());p=a.source/'source_predictions.npz';ck=a.source/'checkpoint.pt'
 if sc['status']!='SOURCE_RAW_FLOW_SCORE_COMPLETE' or sha(p)!=m['prediction_sha256'] or sha(p)!=sc['prediction_sha256'] or sha(ck)!=m['checkpoint_sha256'] or sha(ck)!=sc['checkpoint_sha256']:raise ValueError('Frozen artifact hashes differ')
 with np.load(p,allow_pickle=False) as z:d={k:z[k].copy() for k in z.files}
 state=torch.load(ck,map_location='cpu',weights_only=False);fit=np.asarray(state['fit_ids']);ids=d['indices']
 if state['commit']!=m['commit'] or len(np.unique(fit))!=4096 or len(np.unique(ids))!=672 or np.intersect1d(fit,ids).size:raise ValueError('Disjoint source IDs or checkpoint binding failed')
 fields=0
 for group in ('RX4_6','RX8'):
  for view in ['clean','practical_high','practical_mid','practical_low_suburban','practical_high_urban','practical_mid_urban','practical_low_urban','all_impaired']:
   ix=[i for i in range(len(ids)) if d['group'][i]==group and (d['scene'][i]!='clean' if view=='all_impaired' else d['scene'][i]==view)];v=d['values'];trusted=[i for i in ix if v[i,0]>=.95];raw={i for i in trusted if v[i,2]==v[i,1]};flow={i for i in trusted if v[i,4]==v[i,1]};res=sc['results'][group][view];expected={'count':len(ix),'trusted_count':len(trusted),'raw_correct':len(raw),'flow_correct':len(flow),'repairs':len(flow-raw),'damages':len(raw-flow),'raw_aligned_iq_error':float(np.mean(v[ix,3])),'flow_aligned_iq_error':float(np.mean(v[ix,5]))}
   for key,value in expected.items():
    if abs(res[key]-value)>1e-7:raise ValueError('Mismatch '+group+'/'+view+'/'+key)
    fields+=1
 out=dict(status='PASS',prediction_sha256=sha(p),checkpoint_sha256=sha(ck),source_records=len(ids),fit_records=len(fit),independent_fields=fields,identity_truth_reads=0,method='independent sets and means over immutable source predictions')
 a.out.parent.mkdir(parents=True,exist_ok=True)
 with a.out.open('x',encoding='utf8') as f:json.dump(out,f,indent=2);f.write('\n')
 print(json.dumps(out),flush=True)
if __name__=='__main__':main()
