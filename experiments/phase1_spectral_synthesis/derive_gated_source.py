"""Post-hoc replay of the previously fixed 0.8 source corruption gate."""
import argparse,hashlib,json,shutil
from pathlib import Path
import numpy as np

def route(values,probability):
 if values.ndim!=2 or values.shape[1]!=8 or probability.shape!=(len(values),):raise ValueError('Fixed source result and one gate probability per record required')
 result=values.copy();active=probability>.8
 for pred,err in [(4,5),(6,7)]:result[:,pred]=np.where(active,values[:,pred],values[:,2]);result[:,err]=np.where(active,values[:,err],values[:,3])
 return result

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--commit',required=True);a=p.parse_args()
 if a.out.exists():raise FileExistsError('Immutable derived source predictions required')
 runs=a.root/'runs';source=runs/'20261102-phase1-spectral-synthesis-pilot-s2026092701-r01';prior=runs/'20261102-phase1-unroll-source-replay-s2026092701-r01';m=json.loads((source/'prediction_complete.json').read_text());pm=json.loads((prior/'prediction_complete.json').read_text());sp=source/'source_predictions.npz';pp=prior/'source_predictions.npz';sh=lambda x:hashlib.sha256(x.read_bytes()).hexdigest()
 if m.get('status')!='SOURCE_PREDICTIONS_FIXED' or m.get('commit')!='059987ed75b568b4ce052d7638ee0262bd5652c3' or m.get('count')!=672 or m.get('prediction_sha256')!=sh(sp) or pm.get('status')!='SOURCE_PREDICTIONS_FIXED' or pm.get('count')!=11761 or pm.get('sha256')!='fa90fd400bed527ca6f0409ccd09a0f3ad6a979492f77e8fcf35fdf7d0b6d6a5' or pm.get('sha256')!=sh(pp) or any(m.get(k)!=0 for k in ('U_s_truth_reads','V_reads','test_reads','model_updates_after_prediction')):raise ValueError('Exact fixed parent predictions required')
 with np.load(sp,allow_pickle=False) as z:ids=z['indices'].copy();split=z['split'].copy();rx=z['rx'].copy();scene=z['scene'].copy();values=z['values'].copy()
 with np.load(pp,allow_pickle=False) as z:
  lookup={int(x):i for i,x in enumerate(z['indices'])}
  if len(lookup)!=11761:raise ValueError('Unique prior physical IDs required')
  rows=np.array([lookup[int(x)] for x in ids]);old=z['values'][rows].copy()
  if not np.array_equal(rx,z['rx'][rows]) or not np.array_equal(scene,z['scene'][rows]):raise ValueError('Prior source physical scenes must match')
 active=old[:,3]>.8
 if len(ids)!=672 or not np.array_equal(values[:,1],old[:,0]) or not np.allclose(values[:,0],old[:,1],rtol=1e-6,atol=1e-7) or not np.array_equal(values[:,2],old[:,5]) or not np.allclose(values[:,3],old[:,6],rtol=1e-6,atol=1e-7) or not np.array_equal(values[active,4],old[active,7]) or not np.allclose(values[active,5],old[active,8],rtol=1e-6,atol=1e-7):raise ValueError('Common raw and active physical parent must match exactly')
 gated=route(values,old[:,3]);a.out.mkdir(parents=True);checkpoint=source/'checkpoint.pt';shutil.copyfile(checkpoint,a.out/'checkpoint.pt');path=a.out/'source_predictions.npz'
 with path.open('xb') as f:np.savez(f,indices=ids,split=split,rx=rx,scene=scene,values=gated,gate_probability=old[:,3])
 result=dict(status='SOURCE_PREDICTIONS_FIXED',commit=m['commit'],derivation_commit=a.commit,posthoc=True,description='Fixed previously trained .8 source gate selects entire raw IQ output or entire fixed spectral waveform; no new inference.',count=len(ids),prediction_sha256=sh(path),source_prediction_sha256=m['prediction_sha256'],prior_prediction_sha256=pm['sha256'],checkpoint_sha256=sh(checkpoint),gate_threshold=.8,gate_active_count=int(active.sum()),F_frozen=True,DURA_loaded=False,U_s_truth_reads=0,V_reads=0,test_reads=0,model_updates_after_prediction=0)
 with (a.out/'prediction_complete.json').open('x',encoding='utf8') as f:json.dump(result,f,indent=2);f.write('\n')
 print(json.dumps(result),flush=True)
if __name__=='__main__':main()
