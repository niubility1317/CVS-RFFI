"""Truth-last scoring of fixed low-rank predictions on legal Phase1 L_s."""
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'phase1_joint_mixture'))
from source_pilot import s,core

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--commit',required=True);a=p.parse_args()
 if a.out.exists():raise FileExistsError('Immutable scoring output required')
 src=a.root/'runs/20261102-phase1-lowrank-source-diagnostic-s2026092701-r01';m=core.read(src/'prediction_complete.json');path=src/'predictions.npz'
 if m.get('status')!='SOURCE_PREDICTIONS_FIXED' or m.get('commit')!='8e10e44610526de53c4507d976c59bedb73b431f' or m.get('count')!=160 or m.get('settings')!=['full','64','128','256'] or any(m.get(x)!=0 for x in ('TX_truth_reads','U_s_truth_reads','V_reads','test_reads','model_updates')) or hashlib.sha256(path.read_bytes()).hexdigest()!=m.get('sha256'):raise ValueError('Fixed source predictions required before labels')
 with np.load(path,allow_pickle=False) as d:ids=d['indices'].copy();dec=d['decisions'].copy();scene=d['scene'].copy()
 if ids.shape!=(160,) or len(set(ids.tolist()))!=160 or dec.shape!=(4,160,3):raise ValueError('Complete unique predictions required')
 cache,clean,rx,training,pv,ph=s.source_contract(a.root);validation,heldout=s.new_splits(cache,rx,training,pv,ph);roles=np.load(cache/'train/roles.npy',allow_pickle=False);physical=np.load(cache/'train/physical_ids.npy',allow_pickle=False);_,current_scene=s.epoch_data(cache,19)
 if not np.all(roles[ids]=='L_s') or np.intersect1d(ids,training).size or not np.all(np.isin(ids,np.r_[validation,heldout])) or not np.array_equal(scene,current_scene[ids]):raise ValueError('Fixed heldout L_s role and scene required')
 with np.load(a.root/'runs/20261102-phase1-unroll-source-replay-s2026092701-r01/source_predictions.npz',allow_pickle=False) as prior:
  oldids=prior['indices'];values=prior['values'];lookup={int(v):i for i,v in enumerate(oldids)};pos=np.array([lookup[int(v)] for v in ids]);reference=np.where(values[pos,3]>.8,values[pos,13],values[pos,5])
 if not np.array_equal(reference,dec[0,:,1]):raise ValueError('Full source model replay mismatch')
 complete=core.read(a.root/'runs'/core.F_RUN/'F'/core.F_ROW/'source/complete.json');source=Path(complete['config']['source_cache']);contract=core.read(source/'source_contract.json')
 # Only now open legitimate L_s class labels; U_s labels are never opened.
 with np.load(source/'L_s_metadata.npz',allow_pickle=False) as z:source_ids=z['ids'].copy();labels=z['y'].copy()
 if list(source_ids)!=contract['role_ids']['L_s'] or labels.shape!=(6300,) or np.any((labels<0)|(labels>=6)) or len(set(source_ids.tolist()))!=6300:raise ValueError('Exact authorized L_s metadata required')
 lpos=np.flatnonzero(roles=='L_s');phys_l=physical[lpos]
 if set(source_ids.tolist())!=set(phys_l.tolist()):raise ValueError('Exact L_s physical IDs required')
 path_iq=(source/'L_s_x.npy').resolve();index=path_iq.parent/'L_s_index.npz'
 if not index.is_file():index=path_iq.parent/'L_s_metadata.npz'
 with np.load(index,allow_pickle=False) as v:stored=v['ids'].copy()
 if len(set(stored.tolist()))!=6300 or set(stored.tolist())!=set(source_ids.tolist()):raise ValueError('Exact L_s IQ index required')
 row={str(v):i for i,v in enumerate(stored)};iq=np.load(path_iq,mmap_mode='r',allow_pickle=False);ix=np.array([row[str(physical[i])] for i in ids])
 if iq.shape!=(6300,2,256) or not np.allclose(iq[ix],clean[ids],rtol=1e-5,atol=1e-6):raise ValueError('L_s IQ and source physical IDs must match')
 by_id={str(k):int(v) for k,v in zip(source_ids,labels)};truth=np.array([by_id[str(physical[i])] for i in ids]);correct=dec[:,:,1]==truth[None,:];results={}
 for k,name in enumerate(m['settings']):
  select=correct[k];reference=correct[0];results[name]={'correct':int(select.sum()),'count':160,'repairs_vs_full':int(np.sum(select&~reference)),'damages_vs_full':int(np.sum(~select&reference)),'by_scene':{str(q):{'count':int(np.sum(scene==q)),'correct':int(np.sum(select&(scene==q)))} for q in np.unique(scene)}}
 a.out.mkdir(parents=True);result={'status':'SOURCE_LS_TRUTH_LAST_COMPLETE','commit':a.commit,'prediction_sha256':m['sha256'],'truth_role':'L_s only','L_s_metadata_labels_read':6300,'L_s_scored':160,'U_s_truth_reads':0,'V_reads':0,'test_reads':0,'results':results,'caveat':'Post-hoc source L_s truth; previously explored IDs and views; 160 records, not ground accuracy or proof of generalization.'};core.write(a.out/'source_ls_truth_score.json',result);print(json.dumps(result),flush=True)
if __name__=='__main__':main()
