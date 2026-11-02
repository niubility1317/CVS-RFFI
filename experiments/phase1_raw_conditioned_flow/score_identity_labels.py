"""Truth-last L_s identity scoring of two fixed source flow predictions."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def label_stats(y,raw,old,new):
 if any(x.shape!=y.shape for x in (raw,old,new)):raise ValueError('Paired label arrays required')
 a=raw==y;b=old==y;c=new==y
 return dict(count=len(y),raw_correct=int(a.sum()),old_correct=int(b.sum()),new_correct=int(c.sum()),old_repairs_vs_raw=int((~a&b).sum()),old_damages_vs_raw=int((a&~b).sum()),new_repairs_vs_raw=int((~a&c).sum()),new_damages_vs_raw=int((a&~c).sum()),repairs_vs_old=int((~b&c).sum()),damages_vs_old=int((b&~c).sum()))

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
 if a.out.exists():raise FileExistsError(a.out)
 runs=a.root/'runs';names=['20261102-phase1-raw-conditioned-flow-source-s2026092701-r01','20261102-phase1-identity-flow-source-s2026092701-r01'];pair=runs/'20261102-phase1-identity-flow-source-pair-s2026092701-r01';audit=json.loads((pair/'independent_audit.json').read_text());score=json.loads((pair/'source_pair.json').read_text());paths=[runs/n/'source_predictions.npz' for n in names]
 if audit.get('status')!='PASS' or score.get('status')!='SOURCE_IDENTITY_PAIR_COMPLETE' or sha(paths[0])!=audit.get('old_prediction_sha256') or sha(paths[1])!=audit.get('new_prediction_sha256'):raise ValueError('Independently fixed pair required')
 data=[]
 for p in paths:
  with np.load(p,allow_pickle=False) as z:data.append({k:z[k].copy() for k in z.files})
 old,new=data;ids=old['indices']
 if ids.shape!=(672,) or len(np.unique(ids))!=672 or any(not np.array_equal(old[k],new[k]) for k in ('indices','rx','scene','group')) or not np.array_equal(old['values'][:,2],new['values'][:,2]):raise ValueError('Exact common fixed source records required')
 cache=runs/'20261019-phase1-restoration-learned-action-source-s2026092701-r07/cache/train';physical=np.load(cache/'physical_ids.npy',allow_pickle=False);roles=np.load(cache/'roles.npy',allow_pickle=False);known=roles[ids]=='L_s';unlabeled=roles[ids]=='U_s'
 if not np.all(known|unlabeled):raise ValueError('Only source L_s and U_s roles allowed')
 meta=runs/'20261021-phase1-CRIC-DURA-comparison-m153-r02/data/F0/L_s_metadata.npz';contract=json.loads((meta.parent/'source_contract.json').read_text())
 with np.load(meta,allow_pickle=False) as z:lsids=z['ids'];labels=z['y']
 if lsids.tolist()!=contract['role_ids']['L_s'] or len(lsids)!=6300 or np.any((labels<0)|(labels>=6)):raise ValueError('Exact authorized L_s labels required')
 mapping={str(k):int(v) for k,v in zip(lsids,labels)}
 if len(mapping)!=6300 or not all(str(k) in mapping for k in physical[ids[known]]) or not all(str(k) not in mapping for k in physical[ids[unlabeled]]):raise ValueError('Source identity role binding failed')
 truth=np.full(len(ids),-1,dtype=np.int64);truth[known]=[mapping[str(k)] for k in physical[ids[known]]];raw=old['values'][:,2].astype(np.int64);before=old['values'][:,4].astype(np.int64);after=new['values'][:,4].astype(np.int64);views=['clean','practical_high','practical_mid','practical_low_suburban','practical_high_urban','practical_mid_urban','practical_low_urban'];result={}
 for g in ('RX4_6','RX8'):
  result[g]={}
  for v in views+['all_impaired']:
   sel=known&(old['group']==g)&((old['scene']!='clean') if v=='all_impaired' else (old['scene']==v));result[g][v]=label_stats(truth[sel],raw[sel],before[sel],after[sel])
 out=dict(status='SOURCE_IDENTITY_LS_SCORE_COMPLETE',old_prediction_sha256=sha(paths[0]),new_prediction_sha256=sha(paths[1]),pair_audit_sha256=sha(pair/'independent_audit.json'),L_s_scored=int(known.sum()),L_s_metadata_labels_read=6300,U_s_truth_reads=0,V_reads=0,test_reads=0,results=result,caveat='Previously explored source records, F trained on source; no blind or ground claim.')
 a.out.parent.mkdir(parents=True,exist_ok=True)
 with a.out.open('x',encoding='utf8') as f:json.dump(out,f,indent=2);f.write('\n')
 print(json.dumps(out),flush=True)
if __name__=='__main__':main()
