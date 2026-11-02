"""Audit fixed noise-bin source counts using sorted physical-ID joins and sets."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--source',type=Path,required=True);ap.add_argument('--score',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
 if a.out.exists():raise FileExistsError('Immutable audit required')
 m=json.loads((a.source/'prediction_complete.json').read_text());score=json.loads(a.score.read_text());path=a.source/'source_noise.npz';prior=a.root/'runs/20261102-phase1-unroll-source-replay-s2026092701-r01/source_predictions.npz';ck=a.root/'runs/20261102-phase1-blind-bayes-action-full-s2026092701-r01/checkpoint_epoch10.pt';h=sha(path)
 if m.get('status')!='SOURCE_NOISE_FIXED' or m.get('commit')!='6cab9dde4a0b9b66f4462c046a927893de1a8b44' or m.get('count')!=11761 or m.get('sha256')!=h or m.get('prior_sha256')!=sha(prior) or m.get('blind_checkpoint_sha256')!=sha(ck) or score.get('status')!='SOURCE_NOISE_SCORE_COMPLETE' or score.get('noise_sha256')!=h or score.get('prior_sha256')!=sha(prior) or score.get('fixed_predictions_before_truth') is not True or score.get('bin_edges')!=[-6.,-5.,-4.,-3.,-2.,-1.] or any(score.get(k)!=0 for k in ('U_s_truth_reads','V_reads','test_reads')):raise ValueError('Fixed source predictions and score required')
 with np.load(path,allow_pickle=False) as z:ids=z['indices'];rx=z['rx'];scene=z['scene'];nv=z['values']
 with np.load(prior,allow_pickle=False) as z:
  if not np.array_equal(ids,z['indices']) or not np.array_equal(rx,z['rx']) or not np.array_equal(scene,z['scene']):raise ValueError('Prior source IDs differ')
  v=z['values']
 if ids.shape!=(11761,) or len(set(ids.tolist()))!=11761 or nv.shape!=(11761,2) or v.shape!=(11761,15) or not np.isfinite(nv).all():raise ValueError('Complete finite source records required')
 cache=a.root/'runs/20261019-phase1-restoration-learned-action-source-s2026092701-r07/cache/train';physical=np.load(cache/'physical_ids.npy',allow_pickle=False);roles=np.load(cache/'roles.npy',allow_pickle=False);meta=a.root/'runs/20261021-phase1-CRIC-DURA-comparison-m153-r02/data/F0';contract=json.loads((meta/'source_contract.json').read_text())
 with np.load(meta/'L_s_metadata.npz',allow_pickle=False) as z:keys=z['ids'];labels=z['y']
 if keys.tolist()!=contract['role_ids']['L_s'] or len(keys)!=6300:raise ValueError('Exact L_s label contract required')
 k=keys.astype(str);order=np.argsort(k);ks=k[order];query=physical[ids].astype(str);pos=np.searchsorted(ks,query);hit=(pos<len(ks))&(ks[np.minimum(pos,len(ks)-1)]==query);known=roles[ids]=='L_s'
 if len(set(k.tolist()))!=6300 or int(known.sum())!=1047 or not np.array_equal(hit,known) or not np.all(known|(roles[ids]=='U_s')) or score.get('L_s_scored')!=1047:raise ValueError('Independent source label join failed')
 truth=np.full(len(ids),-1,dtype=np.int64);truth[known]=labels[order[pos[known]]];edges=[-6.,-5.,-4.,-3.,-2.,-1.];cells=0;fields=0
 for group,receivers in [('RX4_6',{4,6}),('RX8',{8})]:
  for b,cell in enumerate(score['results'][group]):
   def in_bin(x):return (b==0 or x>=edges[b-1]) and (b==6 or x<edges[b])
   ix=[i for i in range(len(ids)) if int(rx[i]) in receivers and scene[i]!='clean' and in_bin(float(nv[i,1]))];pseudo=[i for i in ix if v[i,1]>=.95];ls=[i for i in ix if known[i]]
   if (cell['count'],cell['pseudo_count'],cell['L_s_count'])!=(len(ix),len(pseudo),len(ls)):raise ValueError('Noise bin counts differ')
   fields+=3
   for key,rows,target in [('pseudo',pseudo,v[:,0].astype(int)),('L_s',ls,truth)]:
    oracle=sum(any(int(v[i,c])==target[i] for c in (5,7,13)) for i in rows)
    if cell[key+'_oracle_correct']!=oracle:raise ValueError('Oracle count differs')
    fields+=1;base={i for i in rows if int(v[i,7])==target[i]}
    for name,col in [('raw',5),('physical',7),('unroll1',13)]:
     good={i for i in rows if int(v[i,col])==target[i]};item=cell[name][key]
     if (item['correct'],item['repairs_vs_physical'],item['damages_vs_physical'])!=(len(good),len(good-base),len(base-good)):raise ValueError('Paired source score differs')
     fields+=3
   cells+=1
 if cells!=14 or fields!=322:raise ValueError('Complete source bins required')
 out=dict(status='PASS',records=len(ids),L_s_scored=1047,cells=cells,fields_recomputed=fields,noise_sha256=h,method='sorted source physical-ID join and independent set counts',U_s_truth_reads=0,V_reads=0,test_reads=0)
 a.out.parent.mkdir(parents=True,exist_ok=True)
 with a.out.open('x',encoding='utf8') as f:json.dump(out,f,indent=2);f.write('\n')
 print(json.dumps(out),flush=True)
if __name__=='__main__':main()
