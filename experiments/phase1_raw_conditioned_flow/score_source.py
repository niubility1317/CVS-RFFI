"""Score frozen source flow decisions against clean-F pseudo targets."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np

def summarize(v):
 if v.ndim!=2 or v.shape[1]!=6 or not np.isfinite(v).all():raise ValueError('Finite six-column source predictions required')
 trust=v[:,0]>=.95;target=v[:,1];raw=v[:,2]==target;flow=v[:,4]==target
 return dict(count=len(v),trusted_count=int(trust.sum()),raw_correct=int((trust&raw).sum()),flow_correct=int((trust&flow).sum()),repairs=int((trust&~raw&flow).sum()),damages=int((trust&raw&~flow).sum()),raw_aligned_iq_error=float(v[:,3].mean()) if len(v) else None,flow_aligned_iq_error=float(v[:,5].mean()) if len(v) else None)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--source',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
 if a.out.exists():raise FileExistsError(a.out)
 m=json.loads((a.source/'result.json').read_text());p=a.source/'source_predictions.npz';ck=a.source/'checkpoint.pt';sha=lambda x:hashlib.sha256(x.read_bytes()).hexdigest()
 if m.get('status')!='SOURCE_RAW_FLOW_TRAINING_COMPLETE' or m.get('commit')!='476d045e942ba0011bb8cf908d035624370c3668' or m.get('evaluation_count')!=672 or sha(p)!=m.get('prediction_sha256') or sha(ck)!=m.get('checkpoint_sha256') or any(m.get(k)!=0 for k in ('L_s_truth_reads','U_s_truth_reads','V_reads','test_reads')):raise ValueError('Fixed source flow artifacts required')
 with np.load(p,allow_pickle=False) as z:ids=z['indices'];rx=z['rx'];scene=z['scene'];group=z['group'];v=z['values']
 if ids.shape!=(672,) or len(np.unique(ids))!=672 or v.shape!=(672,6) or not np.isfinite(v).all() or not np.all(np.isin(v[:,[1,2,4]],np.arange(6))) or not np.all((v[:,0]>=0)&(v[:,0]<=1)) or not np.all(v[:,[3,5]]>=0):raise ValueError('Valid source predictions required')
 views=['clean','practical_high','practical_mid','practical_low_suburban','practical_high_urban','practical_mid_urban','practical_low_urban'];result={}
 for name,receivers in [('RX4_6',(4,6)),('RX8',(8,))]:
  if not np.array_equal(group==name,np.isin(rx,receivers)):raise ValueError('Source receiver grouping differs')
  result[name]={}
  for view in views+['all_impaired']:
   sel=(group==name)&((scene!='clean') if view=='all_impaired' else (scene==view));n=int(sel.sum())
   if n!=(288 if view=='all_impaired' else 48):raise ValueError('Exact predeclared source sample counts required')
   result[name][view]=summarize(v[sel])
 out=dict(status='SOURCE_RAW_FLOW_SCORE_COMPLETE',prediction_sha256=sha(p),checkpoint_sha256=sha(ck),truth_reads=0,results=result,caveat='Clean F pseudo-target agreement, not identity truth; repeated source records, no target result.')
 a.out.parent.mkdir(parents=True,exist_ok=True)
 with a.out.open('x',encoding='utf8') as f:json.dump(out,f,indent=2);f.write('\n')
 print(json.dumps(out),flush=True)
if __name__=='__main__':main()
