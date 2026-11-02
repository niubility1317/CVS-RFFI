"""Post-hoc Phase1 oracle bounds for fixed complementary candidate subsets."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
VIEWS=['clean','practical_high','practical_mid','practical_low_suburban','practical_high_urban','practical_mid_urban','practical_low_urban'];CANDIDATES={'attention':'20261102-phase1-attention-waveform-ground-test-s2026092701-r01','structured':'20261102-phase1-structured-prior-ground-test-s2026092701-r01','white':'20261102-phase1-independent-prior-pnp-ground-test-s2026092701-r02','free':'20261102-phase1-free-mixture-ground-test-s2026092701-r01','full_hard':'20261102-phase1-free-full-hard-ground-test-s2026092701-r01','joint':'20261102-phase1-joint-mixture-ground-test-s2026092701-r01'};SUBSETS={'three':['attention','structured','white'],'four':['attention','structured','white','free'],'six':['attention','structured','white','free','full_hard','joint']}
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--commit',required=True);a=ap.parse_args()
 if a.out.exists():raise FileExistsError('Immutable oracle diagnostic required')
 runs=a.root/'runs';base=runs/'20261101-phase1-dura-trisigtime-baseline-s2026092701-r01/flow/test_prediction';ids=np.load(base/'physical_ids.npy',allow_pickle=False);sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
 if ids.shape!=(168000,) or len(set(ids.tolist()))!=len(ids):raise ValueError('Unique fixed Phase1 physical IDs required')
 raw={v:np.load(base/(v+'.raw.npy'),allow_pickle=False) for v in VIEWS};pred={v:{} for v in VIEWS};hashes={}
 for name,n in CANDIDATES.items():
  flow=runs/n/'flow';folder=flow/'test_prediction';m=json.loads((folder/'complete.json').read_text());score=json.loads((flow/'test_summary.json').read_text())
  if m.get('status')!='PREDICTIONS_COMPLETE' or m.get('count')!=168000 or m.get('views')!=VIEWS or m.get('TX_truth_reads')!=0 or m.get('model_updates')!=0 or score.get('status')!='ANALYZED' or not score.get('all_predictions_before_truth') or score.get('target_feedback') or not np.array_equal(np.load(folder/'physical_ids.npy',allow_pickle=False),ids):raise ValueError('Exact frozen candidate contract required: '+name)
  hashes[name]={}
  for v in VIEWS:
   z=np.load(folder/(v+'.raw.npy'),allow_pickle=False);path=folder/(v+'.restored.npy');x=np.load(path,allow_pickle=False)
   if not np.array_equal(z,raw[v]) or x.shape!=(168000,) or x.dtype!=np.int64 or not np.all((x>=0)&(x<6)):raise ValueError('Common raw and valid candidate predictions required: '+name+' '+v)
   pred[v][name]=x;hashes[name][v]=sha(path)
 # These truth labels were previously observed in development; all current inputs are fixed.
 truth=json.loads((runs/'20260927-phase1-baselines-final-clean-satellite-m5-r01/data/truth.json').read_text());y=np.asarray([truth[str(i)]['label'] for i in ids]);day=np.asarray([truth[str(i)]['day'] for i in ids]);groups={'all':np.ones(len(ids),dtype=bool),'cross_rx':day!='2021_03_01','cross_both':day=='2021_03_01'}
 if int(groups['cross_rx'].sum())!=126000 or int(groups['cross_both'].sum())!=42000:raise ValueError('Fixed Phase1 test groups required')
 result={}
 for v in VIEWS:
  result[v]={}
  for subset,names in SUBSETS.items():
   union=np.logical_or.reduce([raw[v]==y]+[pred[v][name]==y for name in names]);cells={}
   for group,mask in groups.items():
    cells[group]={'count':int(mask.sum()),'oracle_correct':int(np.count_nonzero(union&mask)),'oracle_accuracy':float(union[mask].mean()*100),'raw_clean_accuracy':float((raw['clean'][mask]==y[mask]).mean()*100)}
   result[v][subset]=cells
 a.out.mkdir(parents=True);out=dict(status='POSTHOC_SUBSET_ORACLE_COMPLETE',commit=a.commit,candidate_runs=CANDIDATES,subsets=SUBSETS,prediction_sha256=hashes,physical_ids_sha256=sha(base/'physical_ids.npy'),count=len(ids),ground_truth_used_after_fixed_predictions=True,U_s_truth_reads=0,V_reads=0,model_updates=0,results=result,caveat='Oracle uses unavailable identity truth to choose output; this is not deployable or an independent blind ground test.')
 with (a.out/'subset_oracles.json').open('x',encoding='utf8') as f:json.dump(out,f,indent=2);f.write('\n')
 print(json.dumps({'status':out['status'],'low_urban':result['practical_low_urban']}),flush=True)
if __name__=='__main__':main()
