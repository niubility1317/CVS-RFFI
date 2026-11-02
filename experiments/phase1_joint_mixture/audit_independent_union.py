"""Post-hoc oracle union of fixed DURA-independent Phase1 predictions."""
import argparse,json
from pathlib import Path
import numpy as np
VIEWS=['clean','practical_high','practical_mid','practical_low_suburban','practical_high_urban','practical_mid_urban','practical_low_urban']
CANDIDATES={
 'direct_axial':'20261102-phase1-direct-axial-iq-s2026092701-r01/source',
 'blind_gate':'20261102-phase1-blind-gate-s2026092701-r01/source',
 'curriculum_gate':'20261102-phase1-curriculum-blind-gate-s2026092701-r01/source',
 'attention':'20261102-phase1-attention-waveform-ground-test-s2026092701-r01',
 'full_hard':'20261102-phase1-free-full-hard-ground-test-s2026092701-r01',
 'free_mixture':'20261102-phase1-free-mixture-ground-test-s2026092701-r01',
 'hierarchical':'20261102-phase1-hierarchical-prior-ground-test-s2026092701-r01',
 'white_prior':'20261102-phase1-independent-prior-pnp-ground-test-s2026092701-r02',
 'structured_prior':'20261102-phase1-structured-prior-ground-test-s2026092701-r01',
 'joint_mixture':'20261102-phase1-joint-mixture-ground-test-s2026092701-r01',
 'confidence_router':'20261102-phase1-confidence-router-ground-test-s2026092701-r01'}

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--commit',required=True);a=p.parse_args()
 if a.out.exists():raise FileExistsError('Immutable diagnostic output required')
 r=a.root;base=r/'runs/20261101-phase1-dura-trisigtime-baseline-s2026092701-r01/flow/test_prediction';ids=np.load(base/'physical_ids.npy',allow_pickle=False)
 if ids.shape!=(168000,) or len(set(ids.tolist()))!=len(ids):raise ValueError('Exact unique Phase1 physical IDs required')
 predictions={v:{} for v in VIEWS};raw={}
 for v in VIEWS:
  raw[v]=np.load(base/(v+'.raw.npy'),allow_pickle=False)
  if raw[v].shape!=(len(ids),) or raw[v].dtype!=np.int64 or np.any((raw[v]<0)|(raw[v]>=6)):raise ValueError('Frozen raw F predictions required')
 for name,run in CANDIDATES.items():
  root=r/'runs'/run/'flow';folder=root/'test_prediction';m=json.loads((folder/'complete.json').read_text());summary=json.loads((root/'test_summary.json').read_text())
  if m.get('status')!='PREDICTIONS_COMPLETE' or m.get('count')!=len(ids) or m.get('views')!=VIEWS or m.get('TX_truth_reads')!=0 or m.get('model_updates')!=0 or summary.get('status')!='ANALYZED' or not summary.get('all_predictions_before_truth') or summary.get('target_feedback') or not np.array_equal(np.load(folder/'physical_ids.npy',allow_pickle=False),ids):raise ValueError('Exact completed frozen candidate contract required: '+name)
  for view in VIEWS:
   observed=np.load(folder/(view+'.raw.npy'),allow_pickle=False);candidate=np.load(folder/(view+'.restored.npy'),allow_pickle=False)
   if not np.array_equal(observed,raw[view]) or candidate.shape!=(len(ids),) or candidate.dtype!=np.int64 or np.any((candidate<0)|(candidate>=6)):raise ValueError('Exact common raw F and valid candidate predictions required: '+name+' '+view)
   predictions[view][name]=candidate
 # Ground identity is accessed only after all candidate outputs and raw F are verified.
 truth=json.loads((r/'runs/20260927-phase1-baselines-final-clean-satellite-m5-r01/data/truth.json').read_text());y=np.asarray([truth[str(i)]['label'] for i in ids]);day=np.asarray([truth[str(i)]['day'] for i in ids]);clean_accuracy=float(np.mean(raw['clean']==y)*100);names=['raw']+list(CANDIDATES);results={}
 for view in VIEWS:
  correct=np.stack([raw[view]==y]+[predictions[view][name]==y for name in CANDIDATES]);union=correct.any(0);base2=correct[0]|correct[names.index('free_mixture')];single={}
  for i,name in enumerate(names):
   others=np.any(np.delete(correct,i,axis=0),axis=0);single[name]={'correct':int(correct[i].sum()),'accuracy':float(correct[i].mean()*100),'exclusive_correct':int(np.count_nonzero(correct[i]&~others)),'unique_vs_raw_free':int(np.count_nonzero(correct[i]&~base2))}
  subsets={}
  for group,select in [('cross_rx_source_dates',day!='2021_03_01'),('cross_rx_and_new_day',day=='2021_03_01')]:
   n=int(select.sum());subsets[group]={'count':n,'oracle_correct':int(np.count_nonzero(union&select)),'oracle_accuracy':float(union[select].mean()*100),'raw_clean_accuracy':float(np.mean((raw['clean']==y)[select])*100)}
  classes={}
  for k in range(6):
   select=y==k;classes[str(k)]={'count':int(select.sum()),'oracle_correct':int(np.count_nonzero(union&select)),'oracle_accuracy':float(union[select].mean()*100)}
  results[view]={'count':len(ids),'oracle_correct':int(union.sum()),'oracle_accuracy':float(union.mean()*100),'oracle_gap_pp':clean_accuracy-float(union.mean()*100),'all_candidates_wrong':int((~union).sum()),'raw_free_union_correct':int(base2.sum()),'per_candidate':single,'cross_domain':subsets,'classes':classes}
 a.out.mkdir(parents=True);out=dict(status='POSTHOC_ORACLE_UNION_COMPLETE',commit=a.commit,candidate_runs=CANDIDATES,count=len(ids),raw_clean_accuracy=clean_accuracy,baseline='Frozen Tri-SIG-TIME SSL',ground_truth_used_after_predictions_fixed=True,U_s_truth_reads=0,V_reads=0,model_updates=0,results=results,caveat='Oracle selects per record using true ground labels and cannot be deployed; only upper bound for these fixed candidates on a reused single-seed simulated Phase1 development set. No DURA outputs or teacher-derived candidates included.');
 with (a.out/'independent_oracle_union.json').open('x',encoding='utf8') as f:json.dump(out,f,indent=2);f.write('\n')
 print('INDEPENDENT_ORACLE_UNION_COMPLETE',json.dumps({k:{x:v[x] for x in ('oracle_correct','oracle_accuracy','all_candidates_wrong','raw_free_union_correct')} for k,v in results.items()}),flush=True)
if __name__=='__main__':main()
