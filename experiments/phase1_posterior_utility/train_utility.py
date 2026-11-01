"""Source-only pairwise selector for fixed class-conditional posterior IQ."""
import argparse,hashlib,json,sys,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from utility_model import UtilityNet,build_features,select_candidate,pair_targets
DEP=Path('/home/szu2070436088/2510044040/CV-SincNet/releases/20261102-phase1-bayes-residual-full-s2026092701-r01-release')
if (DEP/'release_commit.txt').read_text().strip()!='848c4b4c47b054f43c8303638e5aa549154ebfd4':raise ValueError('Exact source interface release required')
sys.path.insert(0,str(DEP/'experiments/phase1_bayes_residual_full'))
from bayes_residual_full import s,core,labels_source,source_prior,BlindActionNet,action_matrix,posterior,SOURCE,SHA,PRIOR_RUN,PRIOR_SHA,SCENES

@torch.no_grad()
def compute(blind,teacher,x,means,covs,priors,eye):
    h,lv=blind(x);A=action_matrix(h);states,lp=posterior(x.reshape(len(x),512),A,eye[None]*lv.exp()[:,None,None],means,covs,priors);parent=(states*lp.exp().unsqueeze(-1)).sum(1);pf=teacher.features(parent.reshape_as(x));pl=teacher.classify_features(pf);cf=teacher.features(states.reshape(-1,2,256)).reshape(len(x),6,160);cl=teacher.classify_features(cf.reshape(-1,160)).reshape(len(x),6,6);features=build_features(pf,cf,pl,cl,lp,lv,x.reshape(len(x),512),A,parent,states)
    return parent,states,pl,cl,features

def training_pairs(teacher,ground,labels,parent,states,pl,cl,features):
    with torch.no_grad():
        conf,pseudo=F.softmax(teacher.classify_features(teacher.features(ground)),1).max(1);known=labels>=0;target=torch.where(known,labels,pseudo);eligible=known|(conf>=.95);bp=pl.argmax(1);cp=cl.argmax(2);den=ground.flatten(1).square().sum(1).clamp_min(1e-12);base_iq=(parent-ground.flatten(1)).square().sum(1)/den;cand_iq=(states-ground.flatten(1)[:,None,:]).square().sum(2)/den[:,None];mask,positive=pair_targets(target,eligible,bp,cp,base_iq,cand_iq)
    return features[mask],positive[mask].float(),dict(eligible=int(eligible.sum().item()),labeled=int(known.sum().item()),pairs=int(mask.sum().item()),positive=int((positive&mask).sum().item()),negative=int((~positive&mask).sum().item()))

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--commit',required=True);p.add_argument('--device',default='cuda:0');p.add_argument('--smoke',action='store_true');p.add_argument('--balanced-pairs',action='store_true');a=p.parse_args()
    if a.out.exists():raise FileExistsError('Immutable source output required')
    a.out.mkdir(parents=True);torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.manual_seed(2026121354)
    cp=a.root/'runs'/SOURCE/'checkpoint_epoch10.pt';pp=a.root/'runs'/PRIOR_RUN/'source_prior.pt'
    if hashlib.sha256(cp.read_bytes()).hexdigest()!=SHA or hashlib.sha256(pp.read_bytes()).hexdigest()!=PRIOR_SHA:raise ValueError('Exact fixed blind model and source prior required')
    state=torch.load(cp,map_location='cpu',weights_only=False);prior=torch.load(pp,map_location='cpu',weights_only=False);blind=BlindActionNet().to(a.device);blind.load_state_dict(state['model']);blind.eval().requires_grad_(False);means=prior['means'].to(a.device);covs=prior['covariances'].to(a.device);priors=prior['priors'].to(a.device);eye=torch.eye(512,device=a.device)
    cache,clean,rx,training,pv,ph=s.source_contract(a.root);validation,heldout=s.new_splits(cache,rx,training,pv,ph);rng=np.random.default_rng(2026120355);prior_ids=np.sort(rng.choice(training,size=8192,replace=False));pool=np.setdiff1d(training,prior_ids,assume_unique=True);frng=np.random.default_rng(2026121355);fit=np.sort(frng.choice(pool,size=4096,replace=False))
    if np.intersect1d(fit,np.r_[prior_ids,validation,heldout]).size:raise ValueError('Disjoint source fit and holdout IDs required')
    roles=np.load(cache/'train/roles.npy',allow_pickle=False);teacher,contract=core.teacher(a.root,a.device);teacher.eval().requires_grad_(False);source_truth=labels_source.source_labels(a.root,cache,clean,roles,contract);labels=np.full(len(roles),-1,dtype=np.int64);labels[fit]=source_truth[fit];del source_truth
    if np.any(labels[roles=='U_s']!=-1) or np.any(labels[np.r_[validation,heldout]]!=-1):raise ValueError('Only L_s fit truth may enter source training')
    fresh=a.root/'runs/20261031-phase1-DURA-freshViews-E30-s2026092701-r01/source_cache'
    if (fresh/'train/physical_ids.npy').resolve()!=(cache/'train/physical_ids.npy').resolve():raise ValueError('Original source IDs required')
    for epoch in (21,22,23):
        info=core.read(fresh/'epochs'/('%02d'%epoch)/'complete.json')
        if info.get('count')!=63000 or info.get('namespace')!=f'learned-action/train/epoch{epoch:02d}' or any(info.get(k)!=0 for k in ('TX_labels_read','F_outputs_read','target_reads')) or info.get('native_engine_modified') is not False:raise ValueError('Fresh source view provenance required')
    features=[];targets=[];counts={'eligible':0,'labeled':0,'pairs':0,'positive':0,'negative':0};tick=time.monotonic();epochs=(21,) if a.smoke else (21,22,23)
    for epoch in epochs:
        received,scene=s.epoch_data(fresh,epoch);ids=fit[scene[fit]!='clean'];ids=ids[:256] if a.smoke else ids
        for start in range(0,len(ids),8):
            idx=ids[start:start+8];x=core.tensor(received[idx],a.device);y=core.tensor(clean[idx],a.device);t=torch.tensor(labels[idx].tolist(),device=a.device,dtype=torch.long);parent,states,pl,cl,v=compute(blind,teacher,x,means,covs,priors,eye);xx,yy,stats=training_pairs(teacher,y,t,parent,states,pl,cl,v)
            if len(xx):features.append(xx.cpu());targets.append(yy.cpu())
            for key in counts:counts[key]+=stats[key]
        print('UTILITY_FEATURES',epoch,'records',len(ids),'counts',json.dumps(counts),'seconds',round(time.monotonic()-tick,1),flush=True)
    if not features or counts['positive']<1 or counts['negative']<1:raise ValueError('Source preflight requires both real repair and harm examples')
    sample=torch.cat(features);target=torch.cat(targets);net=UtilityNet().to(a.device);first=sample[:min(128,len(sample))].to(a.device);first_y=target[:len(first)].to(a.device);loss=F.binary_cross_entropy_with_logits(net.net(first).squeeze(-1),first_y);loss.backward()
    if not torch.isfinite(loss) or net.net[0].weight.grad is None or not torch.isfinite(net.net[0].weight.grad).all() or net.net[0].weight.grad.abs().sum()==0 or any(p.grad is not None or p.requires_grad for p in list(blind.parameters())+list(teacher.parameters())):raise ValueError('Finite source pair training and frozen F/parent required')
    net.zero_grad(set_to_none=True);smoke=dict(status='PASS',source_fit_ids=len(fit),L_s_metadata_labels=6300,fit_L_s_labels=int(np.count_nonzero(labels[fit]>=0)),pair_counts=counts,F_frozen=True,blind_parent_frozen=True,DURA_loaded=False,U_s_truth_reads=0,V_reads=0,test_reads=0);print('UTILITY_SOURCE_SMOKE',json.dumps(smoke),flush=True)
    if a.smoke:
        with (a.out/'result.json').open('x',encoding='utf8') as f:json.dump(smoke,f,indent=2)
        return
    if counts['pairs']<100 or counts['positive']<10 or counts['negative']<10:raise ValueError('Insufficient source repair/harm pairs for selector pilot')
    pos_weight=torch.tensor(counts['negative']/counts['positive'],device=a.device) if a.balanced_pairs else None
    opt=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.001);g=torch.Generator().manual_seed(2026121356);losses=[];net.train()
    for epoch in range(1,11):
        order=torch.randperm(len(sample),generator=g);total=0.
        for start in range(0,len(order),256):
            ix=order[start:start+256];v=sample[ix].to(a.device);y=target[ix].to(a.device);loss=F.binary_cross_entropy_with_logits(net.net(v).squeeze(-1),y,pos_weight=pos_weight)
            if not torch.isfinite(loss):raise FloatingPointError('Finite selector training loss required')
            opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),1.,error_if_nonfinite=True);opt.step();total+=float(loss.detach())*len(ix)
        losses.append(total/len(sample));print('UTILITY_EPOCH',epoch,'pairs',len(sample),'loss',losses[-1],flush=True)
    if any(p.grad is not None for p in list(blind.parameters())+list(teacher.parameters())):raise ValueError('Ground F and blind parent must remain frozen')
    with (a.out/'checkpoint_epoch10.pt').open('xb') as f:torch.save(dict(commit=a.commit,epoch=10,model=net.state_dict(),parent_sha256=SHA,prior_sha256=PRIOR_SHA),f)
    net.eval().requires_grad_(False);dura=source_prior.frozen_dura(a.root,a.device);received,scene=s.epoch_data(cache,19);ids=np.concatenate([np.sort(rng.choice(heldout[scene[heldout]==v],size=64,replace=False)) for v in SCENES]);out=[]
    with torch.no_grad():
        for start in range(0,len(ids),8):
            idx=ids[start:start+8];x=core.tensor(received[idx],a.device);y=core.tensor(clean[idx],a.device);conf,truth=F.softmax(teacher.classify_features(teacher.features(y)),1).max(1);parent,states,pl,cl,v=compute(blind,teacher,x,means,covs,priors,eye);score=net(v);bp=pl.argmax(1);pred=cl.argmax(2);den=y.flatten(1).square().sum(1).clamp_min(1e-12);base_iq=(parent-y.flatten(1)).square().sum(1)/den;d=dura(x);d_pred=teacher.classify_features(teacher.features(d)).argmax(1);d_iq=(d-y).flatten(1).square().sum(1)/den;union=(pred==truth[:,None]).any(1)|(bp==truth);cols=[conf,truth.float(),bp.float(),base_iq,d_pred.float(),d_iq,union.float()]
            for threshold in (.5,.75,.9):
                choice=select_candidate(score,bp,pred,threshold);use=choice>=0;selected=parent.clone();cpred=bp.clone()
                if use.any():selected[use]=states[use,choice[use]];cpred[use]=pred[use,choice[use]]
                iq=(selected-y.flatten(1)).square().sum(1)/den;cols.extend((cpred.float(),iq,use.float()))
            batch=torch.stack(cols,1).cpu()
            if not torch.isfinite(batch).all():raise ValueError('Finite source predictions required')
            out.append(batch)
    values=torch.cat(out);results={}
    for j,view in enumerate(SCENES):
        rows=values[j*64:(j+1)*64];valid=rows[:,0]>=.95;y=rows[:,1];base=rows[:,2]==y;cell=dict(count=len(rows),valid_count=int(valid.sum().item()),parent_correct=int((valid&base).sum().item()),parent_mean_iq_nmse=float(rows[:,3].mean().item()),dura_correct=int((valid&(rows[:,4]==y)).sum().item()),dura_mean_iq_nmse=float(rows[:,5].mean().item()),oracle_union_correct=int((valid&(rows[:,6]>0)).sum().item()))
        for i,threshold in enumerate((.5,.75,.9)):
            good=rows[:,7+3*i]==y;cell[str(threshold)]=dict(correct=int((valid&good).sum().item()),repairs=int((valid&good&~base).sum().item()),damages=int((valid&~good&base).sum().item()),selected=int((rows[:,9+3*i]>0).sum().item()),mean_iq_nmse=float(rows[:,8+3*i].mean().item()))
        results[view]=cell
    result=dict(status='COMPLETE_SOURCE_ONLY_POSTERIOR_UTILITY_PILOT',commit=a.commit,balanced_pairs=a.balanced_pairs,positive_weight=float(pos_weight.item()) if pos_weight is not None else None,fit_ids=len(fit),fit_L_s_labels=int(np.count_nonzero(labels[fit]>=0)),L_s_metadata_labels=6300,pair_counts=counts,epoch_losses=losses,scenes=results,probe_ids=len(ids),thresholds=[.5,.75,.9],F_frozen=True,blind_parent_frozen=True,DURA_loaded_after_training=True,U_s_truth_reads=0,V_reads=0,test_reads=0,caveat='Source L_s fit truth and U_s high-confidence clean-F pseudo labels; source physical IDs previously explored, pseudo target diagnostic not test truth; output is a conditional waveform and class hallucination remains possible.')
    core.write(a.out/'result.json',result);print('UTILITY_COMPLETE',json.dumps(result),flush=True)
if __name__=='__main__':main()
