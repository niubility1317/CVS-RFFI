"""Independently replay sample routes, selected IQ decisions and source counts."""
import argparse,hashlib,json,math,sys
from pathlib import Path
import numpy as np

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def verify_routes(prob,logweights,choices,pred):
 if prob.ndim!=3 or logweights.shape!=prob.shape[:2] or choices.shape!=(len(prob),2) or pred.shape!=(len(prob),4) or not np.isfinite(prob).all() or not np.isfinite(logweights).all() or not np.allclose(prob.sum(-1),1,rtol=1e-5,atol=1e-5) or np.any((choices<0)|(choices>=prob.shape[1])):raise ValueError('Valid source probabilities and choices required')
 w=np.exp(logweights-logweights.max(1,keepdims=True));w=w/w.sum(1,keepdims=True);mass=(w[:,:,None]*prob).sum(1);preferred=mass.argmax(1);support=np.take_along_axis(prob,preferred[:,None,None],axis=2).squeeze(-1);cons=w*support;conf=prob.max(-1);row=np.arange(len(prob))
 if np.any(cons[row,choices[:,0]]<cons.max(1)-1e-6) or np.any(conf[row,choices[:,1]]<conf.max(1)-1e-6):raise ValueError('Independent selection scores differ')
 classes=prob.argmax(-1)
 if not np.array_equal(pred[:,2],classes[row,choices[:,0]]) or not np.array_equal(pred[:,3],classes[row,choices[:,1]]):raise ValueError('Selected class differs from candidate')

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--source',type=Path,required=True);ap.add_argument('--score',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
 if a.out.exists():raise FileExistsError('Immutable source audit required')
 m=json.loads((a.source/'prediction_complete.json').read_text());sc=json.loads(a.score.read_text());path=a.source/'source_predictions.npz';h=sha(path);prior=a.root/'runs/20261102-phase1-unroll-source-replay-s2026092701-r01/source_predictions.npz';ref=a.root/'runs/20261102-phase1-spectral-synthesis-scaled-s2026092701-r01/source_predictions.npz';ck=a.root/'runs/20261102-phase1-blind-bayes-action-full-s2026092701-r01/checkpoint_epoch10.pt';src=a.root/'runs/20261102-phase1-blind-bayes-source-validation-s2026092701-r01/source_prior.pt'
 if m.get('status')!='SOURCE_PREDICTIONS_FIXED' or m.get('commit')!='f6072ab0935cd6d4a40cb4e50a230f8e463f466b' or m.get('count')!=672 or m.get('sha256')!=h or m.get('prior_sha256')!=sha(prior) or m.get('reference_sha256')!=sha(ref) or m.get('blind_checkpoint_sha256')!=sha(ck) or m.get('source_prior_sha256')!=sha(src) or sc.get('status')!='SOURCE_SAMPLE_SCORE_COMPLETE' or sc.get('prediction_sha256')!=h or sc.get('fixed_predictions_before_truth') is not True or any(sc.get(k)!=0 for k in ('U_s_truth_reads','V_reads','test_reads')):raise ValueError('Fixed source sample and score required')
 with np.load(path,allow_pickle=False) as z:ids=z['indices'];rx=z['rx'];scene=z['scene'];split=z['split'];old=z['prior_values'];pred=z['predictions'];err=z['aligned_errors'];prob=z['probabilities'];lw=z['logweights'];choices=z['choices'];active=z['active'];waves=z['selected_waveforms']
 if ids.shape!=(672,) or len(set(ids.tolist()))!=672 or old.shape!=(672,15) or pred.shape!=(672,4) or err.shape!=(672,4) or prob.shape!=(672,6,6) or lw.shape!=(672,6) or choices.shape!=(672,2) or active.shape!=(672,) or waves.shape!=(672,2,2,256) or not np.isfinite(waves).all() or not np.isfinite(err).all():raise ValueError('Complete finite source samples required')
 with np.load(ref,allow_pickle=False) as z:
  if not np.array_equal(ids,z['indices']) or not np.array_equal(rx,z['rx']) or not np.array_equal(scene,z['scene']) or not np.array_equal(split,z['split']):raise ValueError('Fixed source cohort differs')
 with np.load(prior,allow_pickle=False) as z:
  order=np.argsort(z['indices']);keys=z['indices'][order];pos=np.searchsorted(keys,ids)
  if np.any(pos>=len(keys)) or not np.array_equal(keys[pos],ids) or not np.array_equal(old,z['values'][order[pos]]):raise ValueError('Prior source predictions differ')
 if not np.array_equal(pred[:,0],old[:,5]) or not np.array_equal(pred[:,1],old[:,7]) or not np.array_equal(active,old[:,3]>.8):raise ValueError('Source raw, physical, gate differ')
 verify_routes(prob,lw,choices,pred)
 import torch
 sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'phase1_joint_mixture'))
 from source_pilot import s,core
 torch.set_num_threads(2);teacher,_=core.teacher(a.root,'cpu');teacher.eval().requires_grad_(False);actual=[]
 with torch.no_grad():
  for j in range(2):
   chunks=[]
   for start in range(0,len(ids),8):
    x=core.tensor(waves[start:start+8,j],'cpu');chunks.append(teacher.classify_features(teacher.features(x)).argmax(1).cpu().numpy())
   actual.append(np.concatenate(chunks))
 if not np.array_equal(actual[0],pred[:,2]) or not np.array_equal(actual[1],pred[:,3]):raise ValueError('Replayed selected IQ classification differs')
 cache,clean,allrx,training,pv,ph=s.source_contract(a.root);received,allscene=s.epoch_data(cache,19)
 if not np.array_equal(allrx[ids],rx) or not np.array_equal(allscene[ids],scene) or not np.array_equal(waves[~active,0],received[ids[~active]]) or not np.array_equal(waves[~active,1],received[ids[~active]]):raise ValueError('Inactive source input preservation failed')
 physical=np.load(cache/'train/physical_ids.npy',allow_pickle=False);roles=np.load(cache/'train/roles.npy',allow_pickle=False);meta=a.root/'runs/20261021-phase1-CRIC-DURA-comparison-m153-r02/data/F0';contract=json.loads((meta/'source_contract.json').read_text())
 with np.load(meta/'L_s_metadata.npz',allow_pickle=False) as z:keys=z['ids'];labels=z['y']
 if keys.tolist()!=contract['role_ids']['L_s'] or len(keys)!=6300:raise ValueError('Exact source L_s labels required')
 k=keys.astype(str);order=np.argsort(k);ks=k[order];query=physical[ids].astype(str);pos=np.searchsorted(ks,query);hit=(pos<len(ks))&(ks[np.minimum(pos,len(ks)-1)]==query);known=roles[ids]=='L_s'
 if len(set(k.tolist()))!=6300 or not np.array_equal(hit,known) or not np.all(known|(roles[ids]=='U_s')) or sc.get('L_s_scored')!=int(known.sum()):raise ValueError('Independent source label join failed')
 truth=np.full(len(ids),-1,dtype=np.int64);truth[known]=labels[order[pos[known]]];sample_pred=prob.argmax(-1);fields=0;cells=0;names=['raw','physical','consensus','max_confidence']
 for group in ('RX4_6','RX8'):
  for view,cell in sc['results'][group].items():
   ix=[i for i in range(len(ids)) if split[i]==group and (scene[i]!='clean' if view=='all_impaired' else scene[i]==view)];ps=[i for i in ix if old[i,1]>=.95];ls=[i for i in ix if known[i]]
   if (cell['count'],cell['pseudo_count'],cell['L_s_count'],cell['active_count'])!=(len(ix),len(ps),len(ls),sum(bool(active[i]) for i in ix)):raise ValueError('Source sample counts differ')
   fields+=4
   for label,rows,target in [('pseudo',ps,old[:,0].astype(int)),('L_s',ls,truth)]:
    oracle=sum(int(pred[i,1]==target[i] or target[i] in sample_pred[i]) for i in rows)
    if cell[label+'_oracle_correct']!=oracle:raise ValueError('Oracle count differs')
    fields+=1;base={i for i in rows if int(pred[i,1])==target[i]}
    for j,name in enumerate(names):
     good={i for i in rows if int(pred[i,j])==target[i]};item=cell[name]
     if (item[label+'_correct'],item[label+'_repairs_vs_physical'],item[label+'_damages_vs_physical'])!=(len(good),len(good-base),len(base-good)):raise ValueError('Source paired counts differ')
     fields+=3
   for j,name in enumerate(names):
    mean=sum(float(err[i,j]) for i in ix)/len(ix)
    if not math.isclose(cell[name]['iq_error_mean'],mean,rel_tol=1e-6,abs_tol=1e-8):raise ValueError('Source IQ error mean differs')
    fields+=1
   cells+=1
 if cells!=16 or fields!=544:raise ValueError('Complete source audit required')
 out=dict(status='PASS',source_only=True,records=len(ids),L_s_scored=int(known.sum()),cells=cells,fields_recomputed=fields,selected_iq_predictions_replayed=2*len(ids),prediction_sha256=h,method='independent routes, frozen F selected-IQ replay, sorted physical-ID join and set counts',U_s_truth_reads=0,V_reads=0,test_reads=0)
 a.out.parent.mkdir(parents=True,exist_ok=True)
 with a.out.open('x',encoding='utf8') as f:json.dump(out,f,indent=2);f.write('\n')
 print(json.dumps(out),flush=True)
if __name__=='__main__':main()
