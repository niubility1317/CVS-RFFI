"""Frozen source-only attention diagnosis on paired fit IDs and heldout receivers."""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from attention_restorer import AttentionRestorer
from validate_free_full import s,core,source_prior,VIEWS

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--commit',required=True);p.add_argument('--device',default='cuda:0');a=p.parse_args()
 if a.out.exists():raise FileExistsError('Immutable output required')
 a.out.mkdir(parents=True);torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 state=torch.load(a.root/'runs/20261102-phase1-attention-waveform-pilot-s2026092701-r02/checkpoint.pt',map_location='cpu',weights_only=False)
 if state.get('commit')!='52df24a8215001bbd3e2e3bba158396ac33e270d':raise ValueError('Fixed checkpoint required')
 net=AttentionRestorer().to(a.device);net.load_state_dict(state['model']);net.eval().requires_grad_(False);cache,clean,rx,training,pv,ph=s.source_contract(a.root);val,hold=s.new_splits(cache,rx,training,pv,ph);fit=state['fit_ids'];fresh=a.root/'runs/20261031-phase1-DURA-freshViews-E30-s2026092701-r01/source_cache';old,scene_old=s.epoch_data(fresh,21);new,scene_new=s.epoch_data(cache,19)
 rng=np.random.default_rng(2026123155);fit=fit[(scene_old[fit]!='clean')&(scene_new[fit]!='clean')];val=val[scene_new[val]!='clean'];hold=hold[scene_new[hold]!='clean']
 if min(len(fit),len(val),len(hold))<1024 or np.intersect1d(fit,np.r_[val,hold]).size:raise ValueError('Disjoint complete cohorts required')
 fit=np.sort(rng.choice(fit,1024,replace=False));val=np.sort(rng.choice(val,1024,replace=False));hold=np.sort(rng.choice(hold,1024,replace=False));teacher,_=core.teacher(a.root,a.device);teacher.eval().requires_grad_(False);_,gate=s.initial_models(a.root,a.device);gate.eval().requires_grad_(False);dura=source_prior.frozen_dura(a.root,a.device);results={}
 for name,ids,received,scene in (('fit_trained_view',fit,old,scene_old),('fit_other_view',fit,new,scene_new),('validation_other_view',val,new,scene_new),('holdout_other_view',hold,new,scene_new)):
  arrays=[]
  with torch.no_grad():
   for start in range(0,len(ids),32):
    ix=ids[start:start+32];x=core.tensor(received[ix],a.device);y=core.tensor(clean[ix],a.device);conf,target=F.softmax(teacher.classify_features(teacher.features(y)),1).max(1);enabled=gate(x).sigmoid()>.8;full=net(x);signals=[x,dura(x),torch.where(enabled[:,None,None],x+.25*(full-x),x),torch.where(enabled[:,None,None],full,x)];den=y.square().sum((1,2)).clamp_min(1e-12);cols=[target.float(),conf]
    for z in signals:
     pred=teacher.classify_features(teacher.features(z)).argmax(1);err=(z-y).square().sum((1,2))/den;cols.extend((pred.float(),err))
    arrays.append(torch.stack(cols,1).cpu())
  v=np.from_dlpack(torch.cat(arrays).contiguous());trusted=v[:,1]>=.95;cell={'count':len(ids),'valid_count':int(trusted.sum())}
  for j,method in enumerate(('raw','dura','attention_025','attention_1')):cell[method]={'pseudo_correct':int(np.count_nonzero(trusted&(v[:,2+2*j]==v[:,0]))),'direct_iq_nmse':float(v[:,3+2*j].mean())}
  cell['scene_counts']={view:int(np.count_nonzero(scene[ids]==view)) for view in VIEWS if view!='clean'};results[name]=cell;print('ATTENTION_DOMAIN_CELL',name,json.dumps(cell),flush=True)
 core.write(a.out/'result.json',dict(status='SOURCE_DOMAIN_DIAGNOSTIC_COMPLETE',commit=a.commit,results=results,paired_fit_ids=True,source_only=True,F_frozen=True,TX_truth_reads=0,U_s_truth_reads=0,V_reads=0,test_reads=0,caveat='Fit physical IDs were trained in views 21/22/23; clean F pseudo labels are not truth; source IDs have been explored.'))
if __name__=='__main__':main()
