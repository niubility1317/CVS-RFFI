"""Source-only learned mixture of six physical posterior IQ hypotheses."""
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from mixture_controller import MixtureController

DEP=Path('/home/szu2070436088/2510044040/CV-SincNet/releases/20261102-phase1-posterior-utility-pilot-s2026092701-r02-release')
if (DEP/'release_commit.txt').read_text().strip()!='fa7bfabf08a04ccf3ff21c4552890bf03f911b0b':
    raise ValueError('Fixed source utility interface required')
sys.path.insert(0,str(DEP/'experiments/phase1_posterior_utility'))
from train_utility import s, core, labels_source, source_prior, BlindActionNet, action_matrix, posterior, SOURCE, PRIOR_RUN, SCENES
from utility_model import build_features

@torch.no_grad()
def physical(blind,teacher,x,prior,eye):
    h,lv=blind(x); A=action_matrix(h)
    states,lp=posterior(x.reshape(len(x),512),A,eye[None]*lv.exp()[:,None,None],*prior)
    parent=(states*lp.exp().unsqueeze(-1)).sum(1)
    pf=teacher.features(parent.reshape_as(x));pl=teacher.classify_features(pf)
    cf=teacher.features(states.reshape(-1,2,256)).reshape(len(x),6,160)
    cl=teacher.classify_features(cf.reshape(-1,160)).reshape(len(x),6,6)
    v=build_features(pf,cf,pl,cl,lp,lv,x.reshape(len(x),512),A,parent,states)
    return states,lp,v,parent

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--commit',required=True);ap.add_argument('--device',default='cuda:0');ap.add_argument('--smoke',action='store_true');a=ap.parse_args()
    if a.out.exists(): raise FileExistsError('Immutable output required')
    a.out.mkdir(parents=True);torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.manual_seed(2026121854)
    blindstate=torch.load(a.root/'runs'/SOURCE/'checkpoint_epoch10.pt',map_location='cpu',weights_only=False);p=torch.load(a.root/'runs'/PRIOR_RUN/'source_prior.pt',map_location='cpu',weights_only=False)
    if blindstate.get('commit')!='94693a342bb2daa684c0f21415d7b949f9fb2107' or blindstate.get('epoch')!=10 or p.get('source_ids')!=8192:raise ValueError('Matching source checkpoint contract required')
    blind=BlindActionNet().to(a.device);blind.load_state_dict(blindstate['model']);blind.eval().requires_grad_(False)
    prior=(p['means'].to(a.device),p['covariances'].to(a.device),p['priors'].to(a.device));eye=torch.eye(512,device=a.device)
    cache,clean,rx,training,pv,ph=s.source_contract(a.root);validation,heldout=s.new_splits(cache,rx,training,pv,ph)
    rng=np.random.default_rng(2026120355);prior_ids=np.sort(rng.choice(training,size=8192,replace=False));pool=np.setdiff1d(training,prior_ids,assume_unique=True)
    fit=np.sort(np.random.default_rng(2026121855).choice(pool,size=4096,replace=False))
    if np.intersect1d(fit,np.r_[prior_ids,validation,heldout]).size:raise ValueError('Disjoint source IDs required')
    roles=np.load(cache/'train/roles.npy',allow_pickle=False);teacher,contract=core.teacher(a.root,a.device);teacher.eval().requires_grad_(False)
    truth=labels_source.source_labels(a.root,cache,clean,roles,contract);labels=np.full(len(roles),-1,dtype=np.int64);labels[fit]=truth[fit];del truth
    if np.any(labels[roles=='U_s']!=-1) or np.any(labels[np.r_[validation,heldout]]!=-1):raise ValueError('Only L_s fit labels allowed')
    fresh=a.root/'runs/20261031-phase1-DURA-freshViews-E30-s2026092701-r01/source_cache'
    if (fresh/'train/physical_ids.npy').resolve()!=(cache/'train/physical_ids.npy').resolve():raise ValueError('Fixed source IDs required')
    epochs=(21,) if a.smoke else (21,22,23);samples=[];counts=dict(exposures=0,eligible=0,labeled=0);tick=time.monotonic()
    for epoch in epochs:
        info=core.read(fresh/'epochs'/('%02d'%epoch)/'complete.json')
        if info.get('count')!=63000 or info.get('namespace')!=f'learned-action/train/epoch{epoch:02d}' or any(info.get(k)!=0 for k in ('TX_labels_read','F_outputs_read','target_reads')):raise ValueError('Approved source views required')
        received,scene=s.epoch_data(fresh,epoch);ids=fit[scene[fit]!='clean'];ids=ids[:64] if a.smoke else ids
        for start in range(0,len(ids),8):
            idx=ids[start:start+8];x=core.tensor(received[idx],a.device);y=core.tensor(clean[idx],a.device);st,lp,v,base=physical(blind,teacher,x,prior,eye)
            with torch.no_grad():
                conf,pseudo=F.softmax(teacher.classify_features(teacher.features(y)),1).max(1);known=torch.tensor((labels[idx]>=0).tolist(),device=a.device,dtype=torch.bool);lab=torch.tensor(labels[idx].tolist(),device=a.device);target=torch.where(known,lab,pseudo);keep=known|(conf>=.95)
            if keep.any():samples.append(tuple(z[keep].detach().cpu() for z in (v,lp,st,y,target)))
            counts['exposures']+=len(idx);counts['eligible']+=int(keep.sum());counts['labeled']+=int(known.sum())
        print('JOINT_MIXTURE_SOURCE',epoch,json.dumps(counts),'seconds',round(time.monotonic()-tick,1),flush=True)
    if not samples:raise ValueError('Eligible fit examples required')
    v,lp,states,y,targets=(torch.cat(z) for z in zip(*samples));del samples
    net=MixtureController().to(a.device);opt=torch.optim.AdamW(net.parameters(),lr=.0003,weight_decay=.0001);gen=torch.Generator().manual_seed(2026121856);losses=[]
    for epoch in range(1,(2 if a.smoke else 9)):
        order=torch.randperm(len(targets),generator=gen);total=0.
        for start in range(0,len(order),32):
            idx=order[start:start+32];vv=v[idx].to(a.device);ll=lp[idx].to(a.device);ss=states[idx].to(a.device);yy=y[idx].to(a.device);tt=targets[idx].to(a.device).long()
            pred,w,c=net(vv,ll,ss); logits=teacher.classify_features(teacher.features(pred));ce=F.cross_entropy(logits,tt);den=yy.flatten(1).square().sum(1).clamp_min(1e-12);iq=((pred-yy).flatten(1).square().sum(1)/den).mean();kl=(w*(w.clamp_min(1e-9).log()-ll)).sum(1).mean();loss=ce+.2*iq+.02*kl
            opt.zero_grad(set_to_none=True);loss.backward()
            if net.network[-1].weight.grad is None or not torch.isfinite(net.network[-1].weight.grad).all():raise ValueError('Finite controller gradient required')
            if epoch==1 and start==0 and net.network[-1].weight.grad.abs().sum()==0:raise ValueError('Nonzero initial controller gradient required')
            torch.nn.utils.clip_grad_norm_(net.parameters(),1.,error_if_nonfinite=True)
            if not torch.isfinite(loss) or any(p.grad is not None or p.requires_grad for p in list(teacher.parameters())+list(blind.parameters())):raise ValueError('Finite gradient and frozen upstream models required')
            opt.step();total+=float(loss.detach())*len(idx)
        losses.append(total/len(targets));print('JOINT_MIXTURE_EPOCH',epoch,'loss',losses[-1],flush=True)
    if a.smoke:
        with (a.out/'result.json').open('x',encoding='utf8') as f:json.dump(dict(status='PASS',counts=counts,F_frozen=True,blind_frozen=True,test_reads=0,U_s_truth_reads=0,V_reads=0),f,indent=2)
        print('JOINT_MIXTURE_SMOKE PASS',flush=True);return
    with (a.out/'checkpoint.pt').open('xb') as f:torch.save(dict(commit=a.commit,model=net.state_dict(),counts=counts,losses=losses),f)
    net.eval().requires_grad_(False);dura=source_prior.frozen_dura(a.root,a.device);received,scene=s.epoch_data(cache,19)
    ids=np.concatenate([np.sort(rng.choice(heldout[scene[heldout]==view],size=64,replace=False)) for view in SCENES]);out=[]
    with torch.no_grad():
        for start in range(0,len(ids),8):
            idx=ids[start:start+8];x=core.tensor(received[idx],a.device);ground=core.tensor(clean[idx],a.device);conf,target=F.softmax(teacher.classify_features(teacher.features(ground)),1).max(1);st,lp,v,base=physical(blind,teacher,x,prior,eye)
            wave=[x,dura(x),base.reshape_as(x)]+[net(v,lp,st,scale=t)[0] for t in (.5,1.,2.)];den=ground.square().sum((1,2)).clamp_min(1e-12);cols=[conf,target.float()]
            for sig in wave:
                pred=teacher.classify_features(teacher.features(sig)).argmax(1);iq=(sig-ground).square().sum((1,2))/den;cols.extend((pred.float(),iq))
            row=torch.stack(cols,1).cpu()
            if not torch.isfinite(row).all():raise ValueError('Finite source outputs required')
            out.append(row)
    values=np.from_dlpack(torch.cat(out).contiguous());result={}
    for j,view in enumerate(SCENES):
        rows=values[64*j:64*(j+1)];valid=rows[:,0]>=.95;cell=dict(count=len(rows),valid_count=int(valid.sum()))
        for i,name in enumerate(('raw','dura','parent','joint_0.5','joint_1','joint_2')):cell[name]=dict(correct=int(np.count_nonzero(valid&(rows[:,2+2*i]==rows[:,1]))),mean_iq_nmse=float(rows[:,3+2*i].mean()))
        result[view]=cell
    result=dict(status='SOURCE_PILOT_COMPLETE',commit=a.commit,training_counts=counts,losses=losses,scenes=result,source_only=True,F_frozen=True,DURA_used_in_training=False,DURA_used_in_candidate=False,U_s_truth_reads=0,V_reads=0,test_reads=0,caveat='Previously observed source IDs; clean F pseudo labels are not actual test accuracy.')
    with (a.out/'result.json').open('x',encoding='utf8') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    print('JOINT_MIXTURE_COMPLETE',json.dumps(result),flush=True)
if __name__=='__main__':main()
