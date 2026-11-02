"""Independently audit source score with a physical-ID label join and set counts."""
import argparse,hashlib,json,math
from pathlib import Path
import numpy as np

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--source',type=Path,required=True);p.add_argument('--score',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 if a.output.exists():raise FileExistsError('Immutable audit required')
 manifest=json.loads((a.source/'prediction_complete.json').read_text());score=json.loads(a.score.read_text());path=a.source/'source_predictions.npz';sha=hashlib.sha256(path.read_bytes()).hexdigest()
 if manifest.get('status')!='SOURCE_PREDICTIONS_FIXED' or manifest.get('commit')!='d67348d3312a24917681c66d4e3074e3f3ff86c1' or manifest.get('count')!=672 or manifest.get('prediction_sha256')!=sha or score.get('prediction_sha256')!=sha or score.get('status')!='SOURCE_SCORE_COMPLETE' or not score.get('fixed_predictions_before_truth') or any(score.get(k)!=0 for k in ('U_s_truth_reads','V_reads','test_reads')):raise ValueError('Complete fixed predictions required')
 with np.load(path,allow_pickle=False) as z:ids=z['indices'];split=z['split'];scene=z['scene'];values=z['values']
 if ids.shape!=(672,) or len(set(ids.tolist()))!=672 or values.shape!=(672,8):raise ValueError('Complete source cohort required')
 cache=a.root/'runs/20261019-phase1-restoration-learned-action-source-s2026092701-r07/cache';physical=np.load(cache/'train/physical_ids.npy',allow_pickle=False);roles=np.load(cache/'train/roles.npy',allow_pickle=False);meta=a.root/'runs/20261021-phase1-CRIC-DURA-comparison-m153-r02/data/F0/L_s_metadata.npz';contract=json.loads((meta.parent/'source_contract.json').read_text())
 with np.load(meta,allow_pickle=False) as z:physical_ls=z['ids'];truth_ls=z['y']
 if physical_ls.tolist()!=contract['role_ids']['L_s'] or len(physical_ls)!=6300 or not np.all((truth_ls>=0)&(truth_ls<6)):raise ValueError('Invalid authorized source labels')
 lookup={str(k):int(v) for k,v in zip(physical_ls,truth_ls)};known=roles[ids]=='L_s';unlabeled=roles[ids]=='U_s'
 if len(lookup)!=6300 or not np.all(known|unlabeled) or not all(str(v) in lookup for v in physical[ids[known]]) or not all(str(v) not in lookup for v in physical[ids[unlabeled]]):raise ValueError('Source physical-ID join failed')
 truth={i:lookup[str(physical[ids[i]])] for i in np.flatnonzero(known)}
 if score.get('L_s_scored')!=len(truth):raise ValueError('Source truth count mismatch')
 cells=0;fields=0
 for group in ('RX4_6','RX8'):
  for view,cell in score['results'][group].items():
   ix=[i for i in range(len(ids)) if split[i]==group and (scene[i]!='clean' if view=='all_impaired' else scene[i]==view)];pseudo=[i for i in ix if values[i,0]>=.95];ls=[i for i in ix if i in truth]
   assert cell['count']==len(ix) and cell['pseudo_count']==len(pseudo) and cell['L_s_count']==len(ls);fields+=3
   for name,col in [('raw',2),('physical',4),('synthesis',6)]:
    item=cell[name];iq=sum(float(values[i,col+1]) for i in ix)/len(ix);assert math.isclose(item['iq_error_mean'],iq,rel_tol=1e-6,abs_tol=1e-8);fields+=1
    for key,subset in [('pseudo',pseudo),('L_s',ls)]:
     correct=lambda i:int(values[i,1]) if key=='pseudo' else truth[i];good={i for i in subset if int(values[i,col])==correct(i)};base={i for i in subset if int(values[i,4])==correct(i)}
     assert item[key+'_correct']==len(good) and item[key+'_repairs_vs_physical']==len(good-base) and item[key+'_damages_vs_physical']==len(base-good);fields+=3
   cells+=1
 if cells!=16:raise ValueError('Expected 16 source cells')
 out=dict(status='PASS',source_only=True,records=len(ids),L_s_scored=len(truth),cells=cells,fields_recomputed=fields,prediction_sha256=sha,method='independent source physical-ID join and set-based counts',U_s_truth_reads=0,V_reads=0,test_reads=0)
 with a.output.open('x',encoding='utf8') as f:json.dump(out,f,indent=2);f.write('\n')
 print(json.dumps(out),flush=True)
if __name__=='__main__':main()
