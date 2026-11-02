"""Paired source pseudo-target comparison of frozen raw and identity flows."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def paired(old,new):
 if old.shape!=new.shape or old.ndim!=2 or old.shape[1]!=6 or not np.isfinite(old).all() or not np.isfinite(new).all():raise ValueError('Matching finite source rows required')
 if not np.array_equal(old[:,[1,2]],new[:,[1,2]]) or not np.allclose(old[:,[0,3]],new[:,[0,3]],rtol=1e-5,atol=1e-7):raise ValueError('Common frozen clean F, raw F and source IQ required')
 trusted=old[:,0]>=.95;a=old[:,4]==old[:,1];b=new[:,4]==old[:,1]
 return dict(count=len(old),trusted_count=int(trusted.sum()),old_correct=int((trusted&a).sum()),new_correct=int((trusted&b).sum()),repairs=int((trusted&~a&b).sum()),damages=int((trusted&a&~b).sum()),old_iq_error=float(old[:,5].mean()) if len(old) else None,new_iq_error=float(new[:,5].mean()) if len(new) else None)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--old',type=Path,required=True);ap.add_argument('--new',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
 if a.out.exists():raise FileExistsError(a.out)
 paths=[];metadata=[];arrays=[]
 for folder,status,commit,ls_reads in [(a.old,'SOURCE_IDENTITY_FLOW_TRAINING_COMPLETE','4fa488be513e5a116ee1cb88411af2a82e17ccda',0),(a.new,'SOURCE_GLOBAL_FLOW_TRAINING_COMPLETE','a11a61dbd2dee6d484daf78773b613127cf9435a',6300)]:
  m=json.loads((folder/'result.json').read_text());p=folder/'source_predictions.npz';ck=folder/'checkpoint.pt'
  if m.get('status')!=status or m.get('commit')!=commit or m.get('evaluation_count')!=672 or m.get('prediction_sha256')!=sha(p) or m.get('checkpoint_sha256')!=sha(ck) or (m.get('L_s_truth_reads')!=ls_reads or any(m.get(k)!=0 for k in ('U_s_truth_reads','V_reads','test_reads'))):raise ValueError('Fixed source artifacts required')
  with np.load(p,allow_pickle=False) as z:arrays.append({k:z[k].copy() for k in z.files})
  paths.append(p);metadata.append(m)
 first,second=arrays
 for k in ('indices','rx','scene','group'):
  if not np.array_equal(first[k],second[k]):raise ValueError('Source IDs and strata must match: '+k)
 if len(np.unique(first['indices']))!=672 or first['values'].shape!=(672,6):raise ValueError('Complete unique source cohort required')
 views=['clean','practical_high','practical_mid','practical_low_suburban','practical_high_urban','practical_mid_urban','practical_low_urban'];out={}
 for g in ('RX4_6','RX8'):
  out[g]={}
  for v in views+['all_impaired']:
   sel=(first['group']==g)&((first['scene']!='clean') if v=='all_impaired' else (first['scene']==v));expected=288 if v=='all_impaired' else 48
   if int(sel.sum())!=expected:raise ValueError('Exact predeclared source strata required')
   out[g][v]=paired(first['values'][sel],second['values'][sel])
 result=dict(status='SOURCE_GLOBAL_PAIR_COMPLETE',old_prediction_sha256=sha(paths[0]),new_prediction_sha256=sha(paths[1]),L_s_truth_reads=0,U_s_truth_reads=0,V_reads=0,test_reads=0,results=out,caveat='Source scoring reads no identity truth; global model used only fit L_s labels during training. Previously explored source IDs and pseudo targets are not ground performance.')
 a.out.parent.mkdir(parents=True,exist_ok=True)
 with a.out.open('x',encoding='utf8') as f:json.dump(result,f,indent=2);f.write('\n')
 print(json.dumps(result),flush=True)
if __name__=='__main__':main()
