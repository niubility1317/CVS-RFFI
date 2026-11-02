"""Independently audit fixed source replay with index/set based counting."""
import argparse,hashlib,json,math
from pathlib import Path
import numpy as np

def main():
 p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--prior',type=Path,required=True);p.add_argument('--score',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 if a.output.exists():raise FileExistsError('Immutable independent audit required')
 m=json.loads((a.source/'prediction_complete.json').read_text());pm=json.loads((a.prior/'prediction_complete.json').read_text());score=json.loads(a.score.read_text());path=a.source/'source_predictions.npz';original=a.prior/'source_predictions.npz'
 if m.get('status')!='SOURCE_PREDICTIONS_FIXED' or m.get('count')!=11761 or hashlib.sha256(path.read_bytes()).hexdigest()!=m.get('sha256') or score.get('prediction_sha256')!=m['sha256'] or hashlib.sha256(original.read_bytes()).hexdigest()!=pm.get('sha256') or m.get('prior_sha256')!=pm.get('sha256'):raise ValueError('Source prediction and prior hashes required')
 with np.load(path,allow_pickle=False) as z:ids=z['indices'];rx=z['rx'];scene=z['scene'];prior=z['prior_values'];approx=z['lowrank_values']
 with np.load(original,allow_pickle=False) as z:
  if not np.array_equal(ids,z['indices']) or not np.array_equal(rx,z['rx']) or not np.array_equal(scene,z['scene']) or not np.array_equal(prior,z['values']):raise ValueError('Prior source replay mismatch')
 if ids.shape!=(11761,) or prior.shape!=(len(ids),15) or approx.shape!=(len(ids),4) or not np.isfinite(approx).all():raise ValueError('Complete finite results required')
 if not np.isin(approx[:,[0,2]],np.arange(6)).all() or np.any(approx[:,[1,3]]<0):raise ValueError('Valid fixed predictions and IQ errors required')
 inactive=prior[:,3]<=.8
 if not np.array_equal(approx[inactive,0],prior[inactive,5]) or not np.array_equal(approx[inactive,2],prior[inactive,5]):raise ValueError('Inactive gate must preserve raw F decisions')
 names=['full','rank128','rank256','DURA'];views=['clean','practical_high','practical_mid','practical_low_suburban','practical_high_urban','practical_mid_urban','practical_low_urban','all_impaired'];cells=0;fields=0
 for group,receivers in [('RX4_6',{4,6}),('RX8',{8})]:
  for view in views:
   indices=[i for i in range(len(ids)) if int(rx[i]) in receivers and (scene[i]!='clean' if view=='all_impaired' else scene[i]==view)];valid=[i for i in indices if prior[i,1]>=.95];cell=score['results'][group][view]
   assert cell['count']==len(indices) and cell['trusted_count']==len(valid) and cell['active_count']==sum(prior[i,3]>.8 for i in indices);fields+=3;cells+=1
   def full(i):return int(prior[i,13] if prior[i,3]>.8 else prior[i,5])
   def decision(name,i):return full(i) if name=='full' else int(prior[i,2]) if name=='DURA' else int(approx[i,0 if name=='rank128' else 2])
   for name in names:
    value=cell[name];ok={i for i in valid if decision(name,i)==int(prior[i,0])};base={i for i in valid if full(i)==int(prior[i,0])};assert value['pseudo_correct']==len(ok) and value['repairs_vs_full']==len(ok-base) and value['damages_vs_full']==len(base-ok);fields+=3
    if name!='DURA':
     values=[float(prior[i,14] if prior[i,3]>.8 else prior[i,6]) if name=='full' else float(approx[i,1 if name=='rank128' else 3]) for i in indices];assert math.isclose(value['aligned_iq_error_mean'],sum(values)/len(values),rel_tol=1e-6,abs_tol=1e-8);fields+=1
 out=dict(status='PASS',source_only=True,records=len(ids),groups_and_scenes=cells,fields_recomputed=fields,prediction_sha256=m['sha256'],prior_sha256=pm['sha256'],truth_reads=0,U_s_truth_reads=0,V_reads=0,test_reads=0,caveat='Independent set/count audit of fixed source outputs, not a new inference replay.');a.output.parent.mkdir(parents=True,exist_ok=True)
 with a.output.open('x',encoding='utf8') as f:json.dump(out,f,indent=2);f.write('\n')
 print(json.dumps(out),flush=True)
if __name__=='__main__':main()
