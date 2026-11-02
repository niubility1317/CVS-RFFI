"""Independent set-based audit of the fixed source flow comparison."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--old',type=Path,required=True);ap.add_argument('--new',type=Path,required=True);ap.add_argument('--score',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
 if a.out.exists():raise FileExistsError(a.out)
 score=json.loads(a.score.read_text());paths=[a.old/'source_predictions.npz',a.new/'source_predictions.npz']
 if score.get('status')!='SOURCE_IDENTITY_PAIR_COMPLETE' or sha(paths[0])!=score['old_prediction_sha256'] or sha(paths[1])!=score['new_prediction_sha256']:raise ValueError('Fixed source result hashes required')
 data=[]
 for p in paths:
  with np.load(p,allow_pickle=False) as z:data.append({k:z[k].copy() for k in z.files})
 old,new=data
 for k in ('indices','rx','scene','group'):
  if not np.array_equal(old[k],new[k]):raise ValueError('Source identity mismatch '+k)
 if len(np.unique(old['indices']))!=672:raise ValueError('Unique complete source cohort required')
 fields=0
 for g in ('RX4_6','RX8'):
  for v in ['clean','practical_high','practical_mid','practical_low_suburban','practical_high_urban','practical_mid_urban','practical_low_urban','all_impaired']:
   ix=[i for i in range(672) if old['group'][i]==g and (old['scene'][i]!='clean' if v=='all_impaired' else old['scene'][i]==v)];a0=old['values'];a1=new['values'];trusted=[i for i in ix if a0[i,0]>=.95];first={i for i in trusted if a0[i,4]==a0[i,1]};second={i for i in trusted if a1[i,4]==a0[i,1]};expected={'count':len(ix),'trusted_count':len(trusted),'old_correct':len(first),'new_correct':len(second),'repairs':len(second-first),'damages':len(first-second),'old_iq_error':float(np.mean(a0[ix,5])),'new_iq_error':float(np.mean(a1[ix,5]))};cell=score['results'][g][v]
   for k,val in expected.items():
    if abs(cell[k]-val)>1e-7:raise ValueError('Independent mismatch '+g+'/'+v+'/'+k)
    fields+=1
 result=dict(status='PASS',records=672,independent_fields=fields,old_prediction_sha256=sha(paths[0]),new_prediction_sha256=sha(paths[1]),identity_truth_reads=0,method='independent per-record sets and means')
 a.out.parent.mkdir(parents=True,exist_ok=True)
 with a.out.open('x',encoding='utf8') as f:json.dump(result,f,indent=2);f.write('\n')
 print(json.dumps(result),flush=True)
if __name__=='__main__':main()
