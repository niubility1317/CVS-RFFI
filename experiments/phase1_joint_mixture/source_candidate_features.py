"""Extract fixed per-candidate source confidence and waveform consistency features."""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from source_pilot import s,core,source_prior,BlindActionNet,physical,action_matrix,posterior,SOURCE,PRIOR_RUN
from mixture_controller import MixtureController
from free_mixture import FreeMixtureController
from source_union_audit import RUNS

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--commit',required=True);p.add_argument('--device',default='cuda:0');p.add_argument('--smoke',action='store_true');a=p.parse_args()
 if a.out.exists():raise FileExistsError('Immutable source feature output required')
 a.out.mkdir(parents=True);torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 bs=torch.load(a.root/'runs'/SOURCE/'checkpoint_epoch10.pt',map_location='cpu',weights_only=False);p=torch.load(a.root/'runs'/PRIOR_RUN/'source_prior.pt',map_location='cpu',weights_only=False);hier=torch.load(a.root/'runs/20261102-phase1-hierarchical-prior-pilot-s2026092701-r01/source_hierarchical_prior.pt',map_location='cpu',weights_only=False);joint=torch.load(a.root/'runs/20261102-phase1-joint-mixture-pilot-s2026092701-r01/checkpoint.pt',map_location='cpu',weights_only=False);free=torch.load(a.root/'runs/20261102-phase1-free-mixture-pilot-s2026092701-r01/checkpoint.pt',map_location='cpu',weights_only=False)
 if bs.get('commit')!='94693a342bb2daa684c0f21415d7b949f9fb2107' or bs.get('epoch')!=10 or p.get('source_ids')!=8192 or hier.get('fit_ids')!=8192 or len(hier['counts'])!=12 or joint.get('commit')!='c67dc6d7b6a3880ffe2fe4e49357ee4da18eb4ef' or free.get('commit')!='25fe47b9101033b4343ee7b7825c0f7ad79885bc':raise ValueError('Fixed source model contract required')
 blind=BlindActionNet().to(a.device);blind.load_state_dict(bs['model']);blind.eval().requires_grad_(False);prior=(p['means'].to(a.device),p['covariances'].to(a.device),p['priors'].to(a.device));hprior=(hier['means'].to(a.device),hier['covariances'].to(a.device),hier['priors'].to(a.device));jnet=MixtureController().to(a.device);jnet.load_state_dict(joint['model']);jnet.eval().requires_grad_(False);fnet=FreeMixtureController().to(a.device);fnet.load_state_dict(free['model']);fnet.eval().requires_grad_(False);teacher,_=core.teacher(a.root,a.device);teacher.eval().requires_grad_(False);_,gate=s.initial_models(a.root,a.device);gate.eval().requires_grad_(False);dura=source_prior.frozen_dura(a.root,a.device);eye=torch.eye(512,device=a.device)
 data={}
 for name,run in RUNS.items():
  folder=a.root/'runs'/run;done=core.read(folder/'prediction_complete.json')
  if done.get('status')!='SOURCE_PREDICTIONS_FIXED' or done.get('role')!='L_s' or done.get('TX_truth_reads')!=0 or done.get('count')!=1047:raise ValueError('Fixed source L_s predictions required')
  with np.load(folder/'predictions.npz',allow_pickle=False) as z:data[name]={k:z[k] for k in z.files}
 base=data['hierarchical'];ids=base['indices'][:32] if a.smoke else base['indices'];n=len(ids)
 for d in data.values():
  for k in ('indices','physical_ids','rx','scene','part','active'):
   if not np.array_equal(d[k],base[k]):raise ValueError('Aligned source cohorts required')
  if not np.array_equal(d['predictions'][:,:4],base['predictions'][:,:4]):raise ValueError('Common fixed source baseline required')
 cache,clean,rx,training,pv,ph=s.source_contract(a.root);roles=np.load(cache/'train/roles.npy',allow_pickle=False);physical_ids=np.load(cache/'train/physical_ids.npy',allow_pickle=False);received,scene=s.epoch_data(cache,19)
 if np.any(roles[ids]!='L_s') or not np.array_equal(physical_ids[ids],base['physical_ids'][:n]):raise ValueError('Fixed source L_s cohort required')
 features=[];predictions=[];tick=time.monotonic()
 with torch.no_grad():
  for start in range(0,n,8):
   ix=ids[start:start+8];x=core.tensor(received[ix],a.device);active=gate(x).sigmoid()>.8;parent=x.clone();hwave=x.clone();jwave=x.clone();fwave=x.clone()
   if active.any():
    xx=x[active];states,lp,v,z=physical(blind,teacher,xx,prior,eye);parent[active]=z.reshape(-1,2,256);jwave[active]=jnet(v,lp,states,scale=1.)[0];fwave[active]=fnet(v,lp,states,scale=1.)[0];h,lv=blind(xx);A=action_matrix(h);hs,hp=posterior(xx.reshape(len(xx),512),A,eye[None]*lv.exp()[:,None,None],*hprior);hwave[active]=(hs*hp.exp().unsqueeze(-1)).sum(1).reshape(-1,2,256)
   waves=[dura(x),parent,hwave,jwave,fwave];emb=[teacher.features(w) for w in waves];logits=[teacher.classify_features(e) for e in emb];probs=torch.stack([F.softmax(z,1) for z in logits],1);pred=probs.argmax(2);top=torch.topk(probs,2,dim=2).values;entropy=-(probs*probs.clamp_min(1e-12).log()).sum(2);norm=torch.stack([F.normalize(e,dim=1) for e in emb],1);cos=torch.einsum('nif,njf->nij',norm,norm);eq=(pred[:,:,None]==pred[:,None,:]).float();flat=torch.stack([w.reshape(len(x),-1) for w in waves],1);basepower=flat[:,0].square().sum(1).clamp_min(1e-12);distance=(flat[:,:,None,:]-flat[:,None,:,:]).square().sum(3)/basepower[:,None,None];norm_input=(flat-x.reshape(len(x),1,-1)).square().sum(2)/x.reshape(len(x),-1).square().sum(1).clamp_min(1e-12)[:,None]
   v=torch.cat((top[:,:,0:1],(top[:,:,0]-top[:,:,1])[:,:,None],entropy[:,:,None],cos,eq,torch.log1p(distance),torch.log1p(norm_input)[:,:,None]),2)
   if not torch.isfinite(v).all():raise ValueError('Finite source features required')
   expected=np.column_stack((base['predictions'][start:start+len(ix),2],base['predictions'][start:start+len(ix),3],base['predictions'][start:start+len(ix),4],data['joint']['predictions'][start:start+len(ix),4],data['free']['predictions'][start:start+len(ix),4]));actual=np.from_dlpack(pred.cpu().contiguous())
   if not np.array_equal(actual,expected):raise ValueError('Fixed source candidate predictions must replay exactly')
   features.append(v.cpu());predictions.append(pred.cpu())
   if (start+len(ix))%256==0:print('CANDIDATE_SOURCE_FEATURE_PROGRESS',start+len(ix),'seconds',round(time.monotonic()-tick,1),flush=True)
 val=np.from_dlpack(torch.cat(features).contiguous());pred=np.from_dlpack(torch.cat(predictions).contiguous())
 if a.smoke:print('CANDIDATE_SOURCE_FEATURE_SMOKE PASS',len(ids),'shape',val.shape,flush=True);return
 with (a.out/'features.npz').open('xb') as f:np.savez_compressed(f,indices=ids,physical_ids=base['physical_ids'],rx=base['rx'],scene=base['scene'],predictions=pred,features=val)
 result={'status':'SOURCE_CANDIDATE_FEATURES_FIXED','commit':a.commit,'count':n,'candidate_count':5,'features_per_candidate':int(val.shape[2]),'source_only':True,'TX_truth_reads':0,'U_s_truth_reads':0,'V_reads':0,'test_reads':0,'caveat':'Features from previously explored source L_s records and fixed candidate models; no target or independent-blind claim.'}
 with (a.out/'result.json').open('x',encoding='utf8') as f:json.dump(result,f,indent=2);f.write('\n')
 print('CANDIDATE_SOURCE_FEATURES_FIXED',json.dumps(result),flush=True)
if __name__=='__main__':main()
