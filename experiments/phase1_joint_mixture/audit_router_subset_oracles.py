"""Independently audit fixed Phase1 oracle subset counts using set unions."""
import argparse,hashlib,json,math
from pathlib import Path
import numpy as np

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--source',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--commit',required=True);a=ap.parse_args()
 if a.output.exists():raise FileExistsError('Immutable oracle audit required')
 d=json.loads((a.source/'subset_oracles.json').read_text());runs=a.root/'runs';base=runs/'20261101-phase1-dura-trisigtime-baseline-s2026092701-r01/flow/test_prediction';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();ids=np.load(base/'physical_ids.npy',allow_pickle=False)
 if d.get('status')!='POSTHOC_SUBSET_ORACLE_COMPLETE' or d.get('commit')!=a.commit or d.get('count')!=168000 or d.get('physical_ids_sha256')!=sha(base/'physical_ids.npy') or any(d.get(k)!=0 for k in ('U_s_truth_reads','V_reads','model_updates')):raise ValueError('Fixed oracle inputs required')
 views=['clean','practical_high','practical_mid','practical_low_suburban','practical_high_urban','practical_mid_urban','practical_low_urban'];candidates={'attention':'20261102-phase1-attention-waveform-ground-test-s2026092701-r01','structured':'20261102-phase1-structured-prior-ground-test-s2026092701-r01','white':'20261102-phase1-independent-prior-pnp-ground-test-s2026092701-r02','free':'20261102-phase1-free-mixture-ground-test-s2026092701-r01','full_hard':'20261102-phase1-free-full-hard-ground-test-s2026092701-r01','joint':'20261102-phase1-joint-mixture-ground-test-s2026092701-r01'};subsets={'three':['attention','structured','white'],'four':['attention','structured','white','free'],'six':['attention','structured','white','free','full_hard','joint']}
 if d.get('candidate_runs')!=candidates or d.get('subsets')!=subsets:raise ValueError('Exact prespecified candidate subsets required')
 arrays={};raw={}
 for view in views:
  raw[view]=np.load(base/(view+'.raw.npy'),allow_pickle=False);arrays[view]={}
  for name,run in candidates.items():
   folder=runs/run/'flow/test_prediction';path=folder/(view+'.restored.npy');m=json.loads((folder/'complete.json').read_text())
   if m.get('status')!='PREDICTIONS_COMPLETE' or m.get('TX_truth_reads')!=0 or m.get('model_updates')!=0 or sha(path)!=d['prediction_sha256'][name][view] or not np.array_equal(np.load(folder/'physical_ids.npy',allow_pickle=False),ids) or not np.array_equal(np.load(folder/(view+'.raw.npy'),allow_pickle=False),raw[view]):raise ValueError('Candidate source binding failed: '+name+' '+view)
   arrays[view][name]=np.load(path,allow_pickle=False)
 truth=json.loads((runs/'20260927-phase1-baselines-final-clean-satellite-m5-r01/data/truth.json').read_text());y=np.asarray([truth[str(i)]['label'] for i in ids]);day=np.asarray([truth[str(i)]['day'] for i in ids]);groups={'all':set(range(len(ids))),'cross_rx':set(np.flatnonzero(day!='2021_03_01').tolist()),'cross_both':set(np.flatnonzero(day=='2021_03_01').tolist())};checks=0;fields=0
 for view in views:
  correct={'raw':set(np.flatnonzero(raw[view]==y).tolist())}
  for name in candidates:correct[name]=set(np.flatnonzero(arrays[view][name]==y).tolist())
  clean=set(np.flatnonzero(raw['clean']==y).tolist())
  for subset,names in subsets.items():
   union=correct['raw'].union(*(correct[name] for name in names))
   for group,members in groups.items():
    cell=d['results'][view][subset][group];assert cell['count']==len(members) and cell['oracle_correct']==len(union&members) and math.isclose(cell['oracle_accuracy'],100*len(union&members)/len(members),abs_tol=1e-9) and math.isclose(cell['raw_clean_accuracy'],100*len(clean&members)/len(members),abs_tol=1e-9);checks+=1;fields+=4
 out=dict(status='PASS',count=len(ids),subset_group_view_cells=checks,fields_recomputed=fields,method='independent set union and group intersections',ground_truth_posthoc=True,U_s_truth_reads=0,V_reads=0,model_updates=0)
 if checks!=63:raise ValueError('63 subset/group/view cells required')
 with a.output.open('x',encoding='utf8') as f:json.dump(out,f,indent=2);f.write('\n')
 print(json.dumps(out),flush=True)
if __name__=='__main__':main()
