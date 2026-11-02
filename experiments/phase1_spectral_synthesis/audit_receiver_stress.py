"""Audit source receiver stress counts and the frozen clean-F guard."""
import argparse,hashlib,json,math,sys
from pathlib import Path
import numpy as np
from fit_sampling import select_fit

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--source',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--commit',required=True);a=ap.parse_args()
 if a.output.exists():raise FileExistsError('Immutable stress audit required')
 sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'phase1_prior_pnp'));sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'phase1_joint_mixture'));from source_pilot import s
 m=json.loads((a.source/'result.json').read_text());path=a.source/'source_predictions.npz';sha=hashlib.sha256(path.read_bytes()).hexdigest()
 if m.get('status')!='SOURCE_STRESS_COMPLETE' or m.get('commit')!=a.commit or m.get('count')!=288 or m.get('sha256')!=sha or m.get('bounds')!=[0.,.15,.3,.5] or any(m.get(k)!=0 for k in ('U_s_truth_reads','V_reads','test_reads')):raise ValueError('Exact fixed source stress predictions required')
 with np.load(path,allow_pickle=False) as z:ids=z['indices'];rx_saved=z['rx'];epochs=z['epochs'];labels=z['labels'];coeff=z['coefficients'];values=z['values']
 if ids.shape!=(288,) or len(set(ids.tolist()))!=288 or coeff.shape!=(288,4) or not np.all(np.abs(coeff)<=1) or values.shape!=(288,4,6) or not np.isfinite(values).all() or not np.isin(values[:,:,:3],np.arange(6)).all() or not np.isin(values[:,:,3:5],[0,1]).all() or np.any(values[:,:,5]<0):raise ValueError('Valid unique stress source arrays required')
 cache,clean,rx,training,pv,ph=s.source_contract(a.root);validation,heldout=s.new_splits(cache,rx,training,pv,ph);prior=np.sort(np.random.default_rng(2026120355).choice(training,size=8192,replace=False));fit=select_fit(training,prior,8192);roles=np.load(cache/'train/roles.npy',allow_pickle=False)
 if not np.isin(ids,fit).all() or np.intersect1d(ids,np.r_[prior,validation,heldout]).size or not np.all(roles[ids]=='L_s') or not np.array_equal(rx_saved,rx[ids]):raise ValueError('Only disjoint fit L_s records required')
 physical=np.load(cache/'train/physical_ids.npy',allow_pickle=False);meta=a.root/'runs/20261021-phase1-CRIC-DURA-comparison-m153-r02/data/F0/L_s_metadata.npz';contract=json.loads((meta.parent/'source_contract.json').read_text())
 with np.load(meta,allow_pickle=False) as z:ls_ids=z['ids'];ls_truth=z['y']
 if ls_ids.tolist()!=contract['role_ids']['L_s'] or len(ls_ids)!=6300:raise ValueError('Exact legal L_s metadata required')
 order=np.argsort(ls_ids);sorted_ids=ls_ids[order];pos=np.searchsorted(sorted_ids,physical[ids])
 if not np.array_equal(sorted_ids[pos],physical[ids]) or not np.array_equal(ls_truth[order[pos]],labels):raise ValueError('Independent physical-ID label join failed')
 fresh=a.root/'runs/20261031-phase1-DURA-freshViews-E30-s2026092701-r01/source_cache'
 for epoch in np.unique(epochs):
  source=fresh if epoch>=21 else cache;_,scene=s.epoch_data(source,int(epoch));sel=epochs==epoch
  if not np.all(scene[ids[sel]]=='practical_low_urban'):raise ValueError('Fixed low-city source views required')
 checks=0;fields=0;guard_checks=0
 for j,bound in enumerate([0.,.15,.3,.5]):
  for i in range(len(ids)):
   expected=0 if j==0 else int(values[i,j,2]==labels[i]);assert values[i,j,3]==expected
   if values[i,j,4]==0:assert values[i,j,0]==values[i,j,1]
   guard_checks+=1
  for label in range(6):
   ix=[i for i in range(len(ids)) if labels[i]==label];cell=m['results'][str(bound)][str(label)];raw={i for i in ix if int(values[i,j,0])==label};physical_correct={i for i in ix if int(values[i,j,1])==label};assert len(ix)==48 and cell['count']==len(ix) and cell['applied']==sum(int(values[i,j,3]) for i in ix) and cell['active']==sum(int(values[i,j,4]) for i in ix) and cell['raw_correct']==len(raw) and cell['physical_correct']==len(physical_correct) and cell['repairs']==len(physical_correct-raw) and cell['damages']==len(raw-physical_correct);assert math.isclose(cell['aligned_iq_error_mean'],sum(float(values[i,j,5]) for i in ix)/len(ix),rel_tol=1e-6,abs_tol=1e-8);fields+=8;checks+=1
 out=dict(status='PASS',source_only=True,fit_L_s_scored=len(ids),class_bound_cells=checks,fields_recomputed=fields,guard_checks=guard_checks,prediction_sha256=sha,method='independent sorted source-ID join and paired set counts; checks stored F guard decisions',U_s_truth_reads=0,V_reads=0,test_reads=0)
 with a.output.open('x',encoding='utf8') as f:json.dump(out,f,indent=2);f.write('\n')
 print(json.dumps(out),flush=True)
if __name__=='__main__':main()
