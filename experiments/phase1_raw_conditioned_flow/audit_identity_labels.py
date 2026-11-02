"""Independently join legal source labels and recount paired flow results."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--score',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
 if a.out.exists():raise FileExistsError(a.out)
 runs=a.root/'runs';score=json.loads(a.score.read_text());paths=[runs/'20261102-phase1-raw-conditioned-flow-source-s2026092701-r01/source_predictions.npz',runs/'20261102-phase1-identity-flow-source-s2026092701-r01/source_predictions.npz'];pair=runs/'20261102-phase1-identity-flow-source-pair-s2026092701-r01/independent_audit.json'
 if score.get('status')!='SOURCE_IDENTITY_LS_SCORE_COMPLETE' or sha(paths[0])!=score['old_prediction_sha256'] or sha(paths[1])!=score['new_prediction_sha256'] or sha(pair)!=score['pair_audit_sha256']:raise ValueError('Fixed source prediction bindings required')
 ds=[]
 for p in paths:
  with np.load(p,allow_pickle=False) as z:ds.append({k:z[k].copy() for k in z.files})
 old,new=ds;ids=old['indices']
 if len(np.unique(ids))!=672 or any(not np.array_equal(old[k],new[k]) for k in ('indices','rx','scene','group')):raise ValueError('Common fixed source IDs required')
 cache=runs/'20261019-phase1-restoration-learned-action-source-s2026092701-r07/cache/train';physical=np.load(cache/'physical_ids.npy',allow_pickle=False);roles=np.load(cache/'roles.npy',allow_pickle=False);meta=runs/'20261021-phase1-CRIC-DURA-comparison-m153-r02/data/F0/L_s_metadata.npz';contract=json.loads((meta.parent/'source_contract.json').read_text())
 with np.load(meta,allow_pickle=False) as z:keys=z['ids'].copy();labels=z['y'].copy()
 if keys.tolist()!=contract['role_ids']['L_s'] or len(keys)!=6300:raise ValueError('Exact L_s source contract required')
 key=np.asarray([str(x) for x in keys]);order=np.argsort(key);sorted_keys=key[order];query=np.asarray([str(x) for x in physical[ids]]);pos=np.searchsorted(sorted_keys,query);present=(pos<len(sorted_keys))&(sorted_keys[np.minimum(pos,len(sorted_keys)-1)]==query);known=roles[ids]=='L_s'
 if len(set(key.tolist()))!=6300 or not np.array_equal(present,known) or not np.all(known|(roles[ids]=='U_s')) or int(known.sum())!=score['L_s_scored']:raise ValueError('Sorted role and identity join failed')
 truth=np.full(len(ids),-1,dtype=np.int64);truth[known]=labels[order[pos[known]]];v0=old['values'];v1=new['values'];fields=0
 for g in ('RX4_6','RX8'):
  for v in ['clean','practical_high','practical_mid','practical_low_suburban','practical_high_urban','practical_mid_urban','practical_low_urban','all_impaired']:
   ix=[i for i in range(len(ids)) if known[i] and old['group'][i]==g and (old['scene'][i]!='clean' if v=='all_impaired' else old['scene'][i]==v)];raw={i for i in ix if v0[i,2]==truth[i]};before={i for i in ix if v0[i,4]==truth[i]};after={i for i in ix if v1[i,4]==truth[i]};want={'count':len(ix),'raw_correct':len(raw),'old_correct':len(before),'new_correct':len(after),'old_repairs_vs_raw':len(before-raw),'old_damages_vs_raw':len(raw-before),'new_repairs_vs_raw':len(after-raw),'new_damages_vs_raw':len(raw-after),'repairs_vs_old':len(after-before),'damages_vs_old':len(before-after)};cell=score['results'][g][v]
   for k,val in want.items():
    if cell[k]!=val:raise ValueError('Independent mismatch '+g+'/'+v+'/'+k)
    fields+=1
 out=dict(status='PASS',records=len(ids),L_s_scored=int(known.sum()),independent_fields=fields,old_prediction_sha256=sha(paths[0]),new_prediction_sha256=sha(paths[1]),U_s_truth_reads=0,V_reads=0,test_reads=0,method='sorted physical-ID role join and independent set counts')
 a.out.parent.mkdir(parents=True,exist_ok=True)
 with a.out.open('x',encoding='utf8') as f:json.dump(out,f,indent=2);f.write('\n')
 print(json.dumps(out),flush=True)
if __name__=='__main__':main()
