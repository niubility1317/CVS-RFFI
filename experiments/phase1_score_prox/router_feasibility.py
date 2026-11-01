"""Source-only fixed-candidate complementarity and input-confidence routing."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from train_source import s,core,GroundDenoiser,BlindActionNet,action_matrix,posterior,SOURCE,SHA,PRIOR_RUN,PRIOR_SHA,SCENES
from noise_prox import reconstruct_noise_adaptive
from router_selection import select_confidence,oracle_reachable
RUN='20261102-phase1-score-prox-pilot-s2026092701-r01';NET_SHA='5f23bc370ea445dbffe21e1a125f1066321a9df46a1c90e0b617648255323642'

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--commit',required=True);p.add_argument('--device',default='cuda:0');p.add_argument('--smoke',action='store_true');a=p.parse_args()
    if a.out.exists():raise FileExistsError('Immutable source-only output required')
    a.out.mkdir(parents=True);torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    ckpt=a.root/'runs'/RUN/'checkpoint_epoch03.pt';blind_ckpt=a.root/'runs'/SOURCE/'checkpoint_epoch10.pt';prior_file=a.root/'runs'/PRIOR_RUN/'source_prior.pt'
    for path,expected in ((ckpt,NET_SHA),(blind_ckpt,SHA),(prior_file,PRIOR_SHA)):
        if hashlib.sha256(path.read_bytes()).hexdigest()!=expected:raise ValueError('Exact fixed source checkpoint required')
    netstate=torch.load(ckpt,map_location='cpu',weights_only=False);blindstate=torch.load(blind_ckpt,map_location='cpu',weights_only=False);prior=torch.load(prior_file,map_location='cpu',weights_only=False);net=GroundDenoiser().to(a.device);net.load_state_dict(netstate['model']);net.eval().requires_grad_(False);blind=BlindActionNet().to(a.device);blind.load_state_dict(blindstate['model']);blind.eval().requires_grad_(False);means=prior['means'].to(a.device);covs=prior['covariances'].to(a.device);priors=prior['priors'].to(a.device);eye=torch.eye(512,device=a.device)
    cache,clean,rx,training,pv,ph=s.source_contract(a.root);validation,heldout=s.new_splits(cache,rx,training,pv,ph);received,scene=s.epoch_data(cache,19);rng=np.random.default_rng(2026120355);prior_ids=np.sort(rng.choice(training,size=8192,replace=False));ids=training[:4] if a.smoke else np.concatenate([np.sort(rng.choice(heldout[scene[heldout]==v],size=64,replace=False)) for v in SCENES]);teacher,_=core.teacher(a.root,a.device);teacher.eval().requires_grad_(False);out=[];tick=time.monotonic()
    with torch.no_grad():
        for start in range(0,len(ids),8):
            idx=ids[start:start+8];x=core.tensor(received[idx],a.device);y=core.tensor(clean[idx],a.device);conf,target=F.softmax(teacher.classify_features(teacher.features(y)),1).max(1);h,lv=blind(x);A=action_matrix(h);variance=lv.exp();states,w=posterior(x.reshape(len(x),512),A,eye[None]*variance[:,None,None],means,covs,priors);z=(states*w.exp().unsqueeze(-1)).sum(1).reshape_as(x);signals=[z]+[reconstruct_noise_adaptive(net,A,x,z,variance,k) for k in (.25,1.,4.)];den=y.square().sum((1,2)).clamp_min(1e-12);preds=[];confs=[];iq=[]
            for signal in signals:
                confidence,pred=F.softmax(teacher.classify_features(teacher.features(signal)),1).max(1);preds.append(pred);confs.append(confidence);iq.append((signal-y).square().sum((1,2))/den)
            preds=torch.stack(preds,1);confs=torch.stack(confs,1);iq=torch.stack(iq,1);cols=[conf,target.float()]+[preds[:,i].float() for i in range(4)]+[confs[:,i] for i in range(4)]+[iq[:,i] for i in range(4)];batch=torch.stack(cols,1).cpu()
            if not torch.isfinite(batch).all():raise ValueError('Finite source predictions required')
            out.append(batch)
            if not a.smoke and (start+len(idx))%64==0:print('ROUTER_FEASIBILITY_PROGRESS',start+len(idx),'seconds',round(time.monotonic()-tick,1),flush=True)
    if a.smoke:
        with (a.out/'result.json').open('x',encoding='utf8') as f:json.dump(dict(status='PASS',records=len(ids),source_only=True,TX_labels_read=0,U_s_truth_reads=0,V_reads=0,test_reads=0),f,indent=2)
        print('ROUTER_FEASIBILITY_SMOKE PASS',flush=True);return
    values=torch.cat(out);results={}
    for j,view in enumerate(SCENES):
        rows=values[j*64:(j+1)*64];valid=rows[:,0]>=.95;target=rows[:,1].long();preds=rows[:,2:6].long();confs=rows[:,6:10];iq=rows[:,10:14];parent=preds[:,0]==target;oracle=oracle_reachable(preds,target);cell=dict(count=len(rows),valid_count=int(valid.sum().item()),parent_correct=int((valid&parent).sum().item()),oracle_union_correct=int((valid&oracle).sum().item()),oracle_extra=int((valid&oracle&~parent).sum().item()),parent_mean_iq_nmse=float(iq[:,0].mean().item()))
        for margin in (0.,.05,.15):
            choice=select_confidence(confs,margin);pred=preds.gather(1,choice[:,None]).squeeze(1);q=iq.gather(1,choice[:,None]).squeeze(1);correct=pred==target;cell[str(margin)]=dict(correct=int((valid&correct).sum().item()),repaired=int((valid&correct&~parent).sum().item()),harmed=int((valid&~correct&parent).sum().item()),selected_candidate=int((choice>0).sum().item()),mean_iq_nmse=float(q.mean().item()))
        results[view]=cell
    result=dict(status='COMPLETE_SOURCE_ONLY_ROUTER_FEASIBILITY',commit=a.commit,model_sha256=NET_SHA,scenes=results,probe_ids=len(ids),margins=[0.,.05,.15],oracle_uses_clean_ground_F_pseudo=True,oracle_not_deployable=True,selectors_use_candidate_confidence_only=True,source_only=True,TX_labels_read=0,U_s_truth_reads=0,V_reads=0,test_reads=0,caveat='Oracle union uses clean F pseudo-target, not available in deployment; sample previously explored; no test accuracy or general bound.')
    with (a.out/'result.json').open('x',encoding='utf8') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    print('ROUTER_FEASIBILITY_COMPLETE',json.dumps(result),flush=True)
if __name__=='__main__':main()
