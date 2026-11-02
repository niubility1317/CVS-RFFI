"""Post-hoc ideal union for independent candidates with matched source outputs."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
VIEWS=['clean','practical_high','practical_mid','practical_low_suburban','practical_high_urban','practical_mid_urban','practical_low_urban']
RUNS={'confidence_router':('20261102-phase1-confidence-router-labeled-source-s2026092701-r01','20261102-phase1-confidence-router-ground-test-s2026092701-r01'),'full_hard':('20261102-phase1-free-full-hard-labeled-source-s2026092701-r01','20261102-phase1-free-full-hard-ground-test-s2026092701-r01'),'free_mixture':('20261102-phase1-free-mixture-labeled-source-s2026092701-r01','20261102-phase1-free-mixture-ground-test-s2026092701-r01'),'hierarchical':('20261102-phase1-hierarchical-prior-labeled-source-s2026092701-r01','20261102-phase1-hierarchical-prior-ground-test-s2026092701-r01'),'joint_mixture':('20261102-phase1-joint-mixture-labeled-source-s2026092701-r01','20261102-phase1-joint-mixture-ground-test-s2026092701-r01')}

def analyze(pred,truth,select):
 values=np.stack([p[select]==truth[select] for p in pred.values()]);union=values.any(0);return dict(count=int(select.sum()),oracle_correct=int(union.sum()),oracle_accuracy=float(union.mean()*100) if len(union) else None,all_wrong=int((~union).sum()),per_candidate_correct={name:int(values[i].sum()) for i,name in enumerate(pred)})

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--commit',required=True);a=p.parse_args()
 if a.out.exists():raise FileExistsError('Immutable diagnostic output required')
 r=a.root;base=r/'runs/20261101-phase1-dura-trisigtime-baseline-s2026092701-r01/flow/test_prediction';ids=np.load(base/'physical_ids.npy',allow_pickle=False);ground={v:{} for v in VIEWS};raw={v:np.load(base/(v+'.raw.npy'),allow_pickle=False) for v in VIEWS};source=None;srcpred={};src_hash={}
 if ids.shape!=(168000,) or len(set(ids.tolist()))!=len(ids) or any(x.shape!=(len(ids),) or x.dtype!=np.int64 or np.any((x<0)|(x>=6)) for x in raw.values()):raise ValueError('Exact unique Phase1 baseline predictions required')
 for name,(src_run,ground_run) in RUNS.items():
  sf=r/'runs'/src_run;sm=json.loads((sf/'prediction_complete.json').read_text());path=sf/'predictions.npz'
  if sm.get('status')!='SOURCE_PREDICTIONS_FIXED' or sm.get('role')!='L_s' or sm.get('count')!=1047 or sm.get('TX_truth_reads')!=0 or hashlib.sha256(path.read_bytes()).hexdigest()!=sm.get('predictions_sha256'):raise ValueError('Fixed source L_s predictions required: '+name)
  with np.load(path,allow_pickle=False) as z:data={k:z[k].copy() for k in ('indices','physical_ids','rx','scene','part','predictions','active')}
  if source is None:source=data
  elif any(not np.array_equal(data[k],source[k]) for k in ('indices','physical_ids','rx','scene','part','active')) or not np.array_equal(data['predictions'][:,:3],source['predictions'][:,:3]):raise ValueError('Source physical IDs, common raw F and gate must match')
  srcpred[name]=data['predictions'][:,4];src_hash[name]=sm['predictions_sha256']
  gf=r/'runs'/ground_run/'flow';folder=gf/'test_prediction';gm=json.loads((folder/'complete.json').read_text());summary=json.loads((gf/'test_summary.json').read_text())
  if gm.get('status')!='PREDICTIONS_COMPLETE' or gm.get('count')!=168000 or gm.get('views')!=VIEWS or gm.get('TX_truth_reads')!=0 or gm.get('model_updates')!=0 or not summary.get('all_predictions_before_truth') or summary.get('target_feedback') or not np.array_equal(np.load(folder/'physical_ids.npy',allow_pickle=False),ids):raise ValueError('Fixed ground candidate required: '+name)
  for v in VIEWS:
   if not np.array_equal(np.load(folder/(v+'.raw.npy'),allow_pickle=False),raw[v]):raise ValueError('Common raw F ground predictions required')
   candidate=np.load(folder/(v+'.restored.npy'),allow_pickle=False)
   if candidate.shape!=(len(ids),) or candidate.dtype!=np.int64 or np.any((candidate<0)|(candidate>=6)):raise ValueError('Valid fixed candidate predictions required')
   ground[v][name]=candidate
 cache=r/'runs/20261019-phase1-restoration-learned-action-source-s2026092701-r07/cache/train';physical=np.load(cache/'physical_ids.npy',allow_pickle=False);roles=np.load(cache/'roles.npy',allow_pickle=False);ix=source['indices']
 if not np.all(roles[ix]=='L_s') or not np.array_equal(source['physical_ids'],physical[ix]):raise ValueError('Exact source L_s role and physical IDs required')
 # Predictions verified before opening either source L_s or ground identity truth.
 f0=r/'runs/20261021-phase1-CRIC-DURA-comparison-m153-r02/data/F0';contract=json.loads((f0/'source_contract.json').read_text())
 with np.load(f0/'L_s_metadata.npz',allow_pickle=False) as z:k=z['ids'];values=z['y']
 if list(k)!=contract['role_ids']['L_s'] or len(k)!=6300:raise ValueError('Exact L_s label contract required')
 mapping={str(x):int(y) for x,y in zip(k,values)};ys=np.array([mapping[str(x)] for x in source['physical_ids']]);truth=json.loads((r/'runs/20260927-phase1-baselines-final-clean-satellite-m5-r01/data/truth.json').read_text());yt=np.array([truth[str(i)]['label'] for i in ids]);src={'raw':source['predictions'][:,1],**srcpred};src_results={}
 for group,rxs in [('RX4_6',(4,6)),('RX8',(8,))]:
  src_results[group]={}
  for v in VIEWS+['all_impaired']:
   select=np.isin(source['rx'],rxs)&((source['scene']!='clean') if v=='all_impaired' else source['scene']==v);src_results[group][v]=analyze(src,ys,select)
 target={}
 for v in VIEWS:target[v]=analyze({'raw':raw[v],**ground[v]},yt,np.ones(len(yt),dtype=bool))
 a.out.mkdir(parents=True);out=dict(status='POSTHOC_SUBSET_ORACLE_COMPLETE',commit=a.commit,candidate_runs=RUNS,source_prediction_sha256=src_hash,source_L_s_count=len(ix),ground_count=len(ids),ground_predictions_fixed_before_truth=True,L_s_metadata_labels_read=6300,U_s_truth_reads=0,V_reads=0,model_updates=0,source=src_results,ground=target,caveat='True-label ideal selection is not deployable; candidate subset chosen after exploratory analysis of previously observed development results; only these outputs are bounded.');
 with (a.out/'subset_oracle.json').open('x',encoding='utf8') as f:json.dump(out,f,indent=2);f.write('\n')
 print('SUBSET_ORACLE_COMPLETE',json.dumps({'source_RX4_6':src_results['RX4_6']['all_impaired'],'source_RX8':src_results['RX8']['all_impaired'],'ground_mid':target['practical_mid_urban'],'ground_low':target['practical_low_urban']}),flush=True)
if __name__=='__main__':main()
