"""Independently audit fixed full source candidate arrays without labels."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import torch
from source_pilot import s

PRIOR='20261102-phase1-unroll-source-replay-s2026092701-r01';PRIOR_SHA='fa90fd400bed527ca6f0409ccd09a0f3ad6a979492f77e8fcf35fdf7d0b6d6a5'
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--kind',choices=('complementary','free'),required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
 if a.output.exists():raise FileExistsError('Immutable source prediction audit required')
 runs=a.root/'runs';n='20261102-phase1-complementary-full-source-s2026092701-r01' if a.kind=='complementary' else '20261102-phase1-free-mixture-full-source-s2026092701-r01';ref='20261102-phase1-complementary-source-replay-s2026092701-r02' if a.kind=='complementary' else '20261102-phase1-free-mixture-labeled-source-s2026092701-r01';refname='source_predictions.npz' if a.kind=='complementary' else 'predictions.npz';folder=runs/n;m=json.loads((folder/'prediction_complete.json').read_text());path=folder/'source_predictions.npz';prior=runs/PRIOR/'source_predictions.npz';reference=runs/ref/refname;sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();commits={'complementary':'5d39fc51d7b93aca9f846238c26836e10fe0b1a4','free':'a69b3755b72be7fd23ae78e00ff05283b191d6b1'}
 if m.get('status')!='SOURCE_PREDICTIONS_FIXED' or m.get('commit')!=commits[a.kind] or m.get('count')!=11761 or m.get('sha256')!=sha(path) or m.get('prior_sha256')!=PRIOR_SHA or sha(prior)!=PRIOR_SHA or m.get('ls_reference_sha256')!=sha(reference) or m.get('DURA_in_candidate') is not False or any(m.get(k)!=0 for k in ('TX_truth_reads','U_s_truth_reads','V_reads','test_reads','model_updates')):raise ValueError('Exact frozen source candidate provenance required')
 with np.load(path,allow_pickle=False) as z:ids=z['indices'];rx=z['rx'];scene=z['scene'];prior_values=z['prior_values'];pred=z['candidate_predictions'] if a.kind=='complementary' else z['candidate_prediction'][:,None]
 with np.load(prior,allow_pickle=False) as z:
  if not np.array_equal(ids,z['indices']) or not np.array_equal(rx,z['rx']) or not np.array_equal(scene,z['scene']) or not np.array_equal(prior_values,z['values']):raise ValueError('Exact original source physical IDs, raw F and gate required')
 k=3 if a.kind=='complementary' else 1
 if ids.shape!=(11761,) or len(set(ids.tolist()))!=11761 or pred.shape!=(11761,k) or not np.isin(pred,np.arange(6)).all():raise ValueError('Complete valid candidate predictions required')
 inactive=prior_values[:,3]<=.8
 if not np.all(pred[inactive]==prior_values[inactive,5,None]):raise ValueError('Inactive gate must preserve raw F decision')
 with np.load(reference,allow_pickle=False) as z:
  order=np.argsort(ids);sorted_ids=ids[order];pos=np.searchsorted(sorted_ids,z['indices']);rows=order[pos]
  if not np.array_equal(sorted_ids[pos],z['indices']) or not np.array_equal(rx[rows],z['rx']) or not np.array_equal(scene[rows],z['scene']):raise ValueError('Reference L_s IDs and views must match')
  old=z['candidate_predictions'] if a.kind=='complementary' else z['predictions'][:,4,None]
  if not np.array_equal(pred[rows],old):raise ValueError('All 1047 prior L_s candidate decisions must match')
 cache,clean,allrx,training,pv,ph=s.source_contract(a.root);validation,heldout=s.new_splits(cache,allrx,training,pv,ph)
 if not np.array_equal(ids,np.r_[validation,heldout]) or not np.array_equal(rx,allrx[ids]) or np.intersect1d(ids,training).size:raise ValueError('Exact source heldout cohort required')
 if a.kind=='complementary':
  modelruns={'attention':'20261102-phase1-attention-waveform-pilot-s2026092701-r02','structured_prior':'20261102-phase1-structured-iq-prior-s2026092701-r01','white_prior':'20261102-phase1-independent-iq-prior-s2026092701-r01'}
  for key,run in modelruns.items():
   checkpoint=runs/run/'checkpoint.pt';state=torch.load(checkpoint,map_location='cpu',weights_only=False)
   if sha(checkpoint)!=m['model_sha256'][key] or np.intersect1d(ids,state['fit_ids']).size:raise ValueError('Exact disjoint candidate checkpoint required: '+key)
 else:
  checkpoint=runs/'20261102-phase1-free-mixture-pilot-s2026092701-r01/checkpoint.pt';state=torch.load(checkpoint,map_location='cpu',weights_only=False);prior_fit=np.sort(np.random.default_rng(2026120355).choice(training,size=8192,replace=False));pool=np.setdiff1d(training,prior_fit,assume_unique=True);fit=np.sort(np.random.default_rng(2026121855).choice(pool,size=4096,replace=False))
  if sha(checkpoint)!=m['model_sha256'] or state.get('commit')!='25fe47b9101033b4343ee7b7825c0f7ad79885bc' or np.intersect1d(ids,np.r_[prior_fit,fit]).size:raise ValueError('Exact disjoint free model checkpoint required')
 result=dict(status='PASS',kind=a.kind,count=len(ids),candidate_count=k,matched_source_L_s_records=len(rows),matched_L_s_decisions=1047*k,inactive_records=int(inactive.sum()),prediction_sha256=m['sha256'],prior_sha256=PRIOR_SHA,method='independent sorted source-ID join, fixed checkpoint hashes and inactive-gate checks',TX_truth_reads=0,U_s_truth_reads=0,V_reads=0,test_reads=0)
 a.output.parent.mkdir(parents=True,exist_ok=True)
 with a.output.open('x',encoding='utf8') as f:json.dump(result,f,indent=2);f.write('\n')
 print(json.dumps(result),flush=True)
if __name__=='__main__':main()
