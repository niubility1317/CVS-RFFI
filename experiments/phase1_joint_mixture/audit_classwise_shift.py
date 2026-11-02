"""Post-hoc Phase1 source/ground classwise audit of fixed independent recovery."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
VIEWS=['clean','practical_high','practical_mid','practical_low_suburban','practical_high_urban','practical_mid_urban','practical_low_urban']

def cell(truth,raw,pred,dura,clean):
 n=len(truth);a=raw==truth;b=pred==truth;d=dura==truth;c=clean==truth;return dict(count=n,clean_correct=int(c.sum()),raw_correct=int(a.sum()),restored_correct=int(b.sum()),dura_correct=int(d.sum()),restored_repairs_vs_raw=int(np.count_nonzero(b&~a)),restored_damages_vs_raw=int(np.count_nonzero(~b&a)),wrong_restored_predictions=np.bincount(pred[~b],minlength=6).tolist(),raw_prediction_distribution=np.bincount(raw,minlength=6).tolist(),restored_prediction_distribution=np.bincount(pred,minlength=6).tolist())

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--commit',required=True);a=p.parse_args()
 if a.out.exists():raise FileExistsError('Immutable audit output required')
 r=a.root;gr=r/'runs/20261102-phase1-free-mixture-ground-test-s2026092701-r01/flow';gd=gr/'test_prediction';base=r/'runs/20261101-phase1-dura-trisigtime-baseline-s2026092701-r01/flow/test_prediction';sm=r/'runs/20261102-phase1-free-mixture-labeled-source-s2026092701-r01';gm=json.loads((gd/'complete.json').read_text());summary=json.loads((gr/'test_summary.json').read_text());source_manifest=json.loads((sm/'prediction_complete.json').read_text());srcpath=sm/'predictions.npz'
 if gm.get('status')!='PREDICTIONS_COMPLETE' or gm.get('count')!=168000 or gm.get('views')!=VIEWS or gm.get('TX_truth_reads')!=0 or gm.get('model_updates')!=0 or summary.get('status')!='ANALYZED' or not summary.get('all_predictions_before_truth') or summary.get('target_feedback') or source_manifest.get('status')!='SOURCE_PREDICTIONS_FIXED' or source_manifest.get('role')!='L_s' or source_manifest.get('count')!=1047 or hashlib.sha256(srcpath.read_bytes()).hexdigest()!=source_manifest.get('predictions_sha256'):raise ValueError('Fixed source and ground predictions required')
 ids=np.load(gd/'physical_ids.npy',allow_pickle=False)
 if ids.shape!=(168000,) or len(set(ids.tolist()))!=len(ids) or not np.array_equal(ids,np.load(base/'physical_ids.npy',allow_pickle=False)):raise ValueError('Exact ground physical IDs required')
 ground={}
 for view in VIEWS:
  raw=np.load(gd/(view+'.raw.npy'),allow_pickle=False);candidate=np.load(gd/(view+'.restored.npy'),allow_pickle=False);dura=np.load(base/(view+'.restored.npy'),allow_pickle=False);raw_base=np.load(base/(view+'.raw.npy'),allow_pickle=False)
  if any(x.shape!=(len(ids),) or x.dtype!=np.int64 or np.any((x<0)|(x>=6)) for x in (raw,candidate,dura,raw_base)) or not np.array_equal(raw,raw_base):raise ValueError('Valid common raw and fixed predictions required')
  ground[view]=(raw,candidate,dura)
 with np.load(srcpath,allow_pickle=False) as z:source={k:z[k] for k in z.files}
 cache=r/'runs/20261019-phase1-restoration-learned-action-source-s2026092701-r07/cache/train';physical=np.load(cache/'physical_ids.npy',allow_pickle=False);roles=np.load(cache/'roles.npy',allow_pickle=False);idx=source['indices']
 if idx.shape!=(1047,) or not np.all(roles[idx]=='L_s') or not np.array_equal(source['physical_ids'],physical[idx]) or source['predictions'].shape!=(1047,5):raise ValueError('Exact source roles and IDs required')
 # All source and ground predictions fixed and verified before accessing labels.
 truth=json.loads((r/'runs/20260927-phase1-baselines-final-clean-satellite-m5-r01/data/truth.json').read_text());yt=np.asarray([truth[str(i)]['label'] for i in ids]);f0=r/'runs/20261021-phase1-CRIC-DURA-comparison-m153-r02/data/F0';contract=json.loads((f0/'source_contract.json').read_text())
 with np.load(f0/'L_s_metadata.npz',allow_pickle=False) as z:src_ids=z['ids'];labels=z['y']
 if list(src_ids)!=contract['role_ids']['L_s'] or labels.shape!=(6300,) or len(set(src_ids.tolist()))!=6300:raise ValueError('Authorized L_s metadata required')
 order=np.argsort(src_ids);keys=src_ids[order];pos=np.searchsorted(keys,physical[idx]);
 if not np.array_equal(keys[pos],physical[idx]):raise ValueError('Source labels must match physical IDs')
 ys=labels[order][pos];sp=source['predictions'];groups={(x['view'],x['dimension'],x['value']):x for x in summary['groups']};target={}
 for view in VIEWS:
  raw,pred,dura=ground[view];target[view]={}
  for k in range(6):
   select=yt==k;z=cell(yt[select],raw[select],pred[select],dura[select],ground['clean'][0][select]);g=groups[(view,'TX',str(k))]
   if z['count']!=g['count'] or abs(100*z['raw_correct']/z['count']-g['raw'])>1e-9 or abs(100*z['restored_correct']/z['count']-g['restored'])>1e-9:raise ValueError('Ground group score mismatch')
   target[view][str(k)]=z
 source_result={}
 for group,receivers in [('RX4_6',(4,6)),('RX8',(8,)),('all',(4,6,8))]:
  source_result[group]={}
  for view in VIEWS+['all_impaired']:
   select=np.isin(source['rx'],receivers)&((source['scene']!='clean') if view=='all_impaired' else (source['scene']==view));source_result[group][view]={}
   for k in range(6):
    ix=select&(ys==k)
    if np.any(ix):source_result[group][view][str(k)]=cell(ys[ix],sp[ix,1],sp[ix,4],sp[ix,2],sp[ix,0])
 a.out.mkdir(parents=True);result=dict(status='POSTHOC_CLASSWISE_COMPLETE',commit=a.commit,source_prediction_sha256=source_manifest['predictions_sha256'],ground_count=len(ids),source_L_s_count=len(idx),L_s_metadata_labels_read=6300,U_s_truth_reads=0,V_reads=0,predictions_fixed_before_ground_truth=True,source=source_result,ground=target,caveat='Post-hoc analysis of repeatedly observed development ground test and explored source records. No target-based model fitting and no blind or real-satellite claim.');
 with (a.out/'classwise_shift.json').open('x',encoding='utf8') as f:json.dump(result,f,indent=2);f.write('\n')
 print('CLASSWISE_AUDIT_COMPLETE',json.dumps({'source_low_urban':source_result['all']['practical_low_urban'],'ground_low_urban':target['practical_low_urban']}),flush=True)
if __name__=='__main__':main()
