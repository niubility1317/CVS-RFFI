"""Post-hoc L_s classwise paired damage across fixed source restorers."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
VIEWS=['clean','practical_high','practical_mid','practical_low_suburban','practical_high_urban','practical_mid_urban','practical_low_urban','all_impaired']

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--commit',required=True);a=ap.parse_args()
 if a.out.exists():raise FileExistsError('Immutable source diagnostic required')
 runs=a.root/'runs';names=['20261102-phase1-unroll-source-replay-s2026092701-r01','20261102-phase1-supervised-unroll-source-replay-s2026092701-r01','20261102-phase1-spectral-synthesis-full-source-s2026092701-r02'];expected=['fa90fd400bed527ca6f0409ccd09a0f3ad6a979492f77e8fcf35fdf7d0b6d6a5','58df544e480a35b13f5c70af551bfe72e3693a21727495caf49dd347c34b25b7','265c7cc357a1fbd12560b0effa745ae1d95598c9eae5e69d9911b34aed53e74b'];sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();paths=[]
 for n,h in zip(names,expected):
  d=runs/n;m=json.loads((d/'prediction_complete.json').read_text());p=d/'source_predictions.npz'
  if m.get('status')!='SOURCE_PREDICTIONS_FIXED' or m.get('count')!=11761 or m.get('sha256')!=h or sha(p)!=h or any(m.get(k)!=0 for k in ('TX_truth_reads','U_s_truth_reads','V_reads','test_reads')):raise ValueError('Exact fixed source predictions required')
  paths.append(p)
 with np.load(paths[0],allow_pickle=False) as a0,np.load(paths[1],allow_pickle=False) as a1,np.load(paths[2],allow_pickle=False) as a2:
  ids=a0['indices'].copy();rx=a0['rx'].copy();scene=a0['scene'].copy();prior=a0['values'].copy();sup=a1['candidate_values'].copy();spectral=a2['spectral_values'].copy()
  if any(not np.array_equal(a0[k],z[k]) for z in (a1,a2) for k in ('indices','rx','scene')) or not np.array_equal(prior,a1['prior_values']) or not np.array_equal(prior,a2['prior_values']):raise ValueError('Same fixed source cohort and physical parent required')
 if ids.shape!=(11761,) or len(set(ids.tolist()))!=11761 or prior.shape!=(11761,15) or sup.shape!=(11761,4) or spectral.shape!=(11761,2):raise ValueError('Complete source predictions required')
 cache=runs/'20261019-phase1-restoration-learned-action-source-s2026092701-r07/cache';physical=np.load(cache/'train/physical_ids.npy',allow_pickle=False);roles=np.load(cache/'train/roles.npy',allow_pickle=False);meta=runs/'20261021-phase1-CRIC-DURA-comparison-m153-r02/data/F0/L_s_metadata.npz';contract=json.loads((meta.parent/'source_contract.json').read_text())
 with np.load(meta,allow_pickle=False) as z:physical_ls=z['ids'];truth_ls=z['y']
 if physical_ls.tolist()!=contract['role_ids']['L_s'] or len(physical_ls)!=6300 or not np.all((truth_ls>=0)&(truth_ls<6)):raise ValueError('Authorized L_s identities required')
 lookup={str(k):int(y) for k,y in zip(physical_ls,truth_ls)};known=roles[ids]=='L_s';unlabeled=roles[ids]=='U_s'
 if len(lookup)!=6300 or int(known.sum())!=1047 or not np.all(known|unlabeled) or not all(str(x) in lookup for x in physical[ids[known]]) or not all(str(x) not in lookup for x in physical[ids[unlabeled]]):raise ValueError('Exact physical-ID source label roles required')
 truth=np.full(len(ids),-1,dtype=np.int64);truth[known]=[lookup[str(x)] for x in physical[ids[known]]];pred={'raw':prior[:,5],'physical':prior[:,7],'unroll':prior[:,13],'single':sup[:,0],'multiview':sup[:,2],'spectral':spectral[:,0]}
 if any(not np.isin(v,np.arange(6)).all() for v in pred.values()):raise ValueError('Six-class predictions required')
 cells={}
 for group,receivers in [('RX4_6',(4,6)),('RX8',(8,))]:
  for view in VIEWS:
   for label in range(6):
    select=known&np.isin(rx,receivers)&(truth==label)&((scene!='clean') if view=='all_impaired' else (scene==view));n=int(select.sum());item={'count':n,'methods':{}}
    for name,v in pred.items():
     correct=v==truth;raw=pred['raw']==truth;parent=pred['physical']==truth;item['methods'][name]={'correct':int(np.count_nonzero(select&correct)),'repairs_vs_raw':int(np.count_nonzero(select&correct&~raw)),'damages_vs_raw':int(np.count_nonzero(select&~correct&raw)),'repairs_vs_physical':int(np.count_nonzero(select&correct&~parent)),'damages_vs_physical':int(np.count_nonzero(select&~correct&parent))}
    cells[f'{group}:{view}:{label}']=item
 a.out.mkdir(parents=True);result=dict(status='SOURCE_CLASS_DIAGNOSTIC_COMPLETE',commit=a.commit,parent_sha256=expected,source_only=True,L_s_scored=int(known.sum()),L_s_metadata_labels_read=6300,U_s_truth_reads=0,V_reads=0,test_reads=0,group_cells=len(cells),cells=cells,caveat='Post-hoc source development analysis; true-label counts in individual cells may be small; not ground performance or blind evidence.')
 with (a.out/'source_class_counts.json').open('x',encoding='utf8') as f:json.dump(result,f,indent=2);f.write('\n')
 print(json.dumps({'status':result['status'],'cells':len(cells),'RX4_6_low_urban_class5':cells['RX4_6:practical_low_urban:5'],'RX8_low_urban_class5':cells['RX8:practical_low_urban:5']}),flush=True)
if __name__=='__main__':main()
