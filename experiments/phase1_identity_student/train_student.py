"""Source-only amortization of bounded identity-guided correction."""
import argparse,hashlib,json,sys,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from identity_student_model import IdentityStudent
DEP=Path('/home/szu2070436088/2510044040/CV-SincNet/releases/20261102-phase1-bayes-residual-full-s2026092701-r01-release')
if (DEP/'release_commit.txt').read_text().strip()!='848c4b4c47b054f43c8303638e5aa549154ebfd4':raise ValueError('Exact Phase1 source interface release required')
sys.path.insert(0,str(DEP/'experiments/phase1_bayes_residual_full'))
from bayes_residual_full import s,core,labels_source,source_prior,BlindActionNet,prior_output,SOURCE,SHA,PRIOR_RUN,PRIOR_SHA,SCENES

def student_loss(net,teacher,x,z,b,ground,labels):
    with torch.no_grad():
        clean_probs=F.softmax(teacher.classify_features(teacher.features(ground)),1);confidence,pseudo=clean_probs.max(1);known=labels>=0;target=torch.where(known,labels,pseudo);eligible=known|(confidence>=.95)
    if not eligible.any():raise ValueError('No eligible source training targets in batch')
    query=z.detach().requires_grad_(True);base_logits=teacher.classify_features(teacher.features(query));per=F.cross_entropy(base_logits,target,reduction='none');gradient=torch.autograd.grad((per*eligible.float()).sum(),query)[0].detach();length=gradient.flatten(1).norm(dim=1).clamp_min(1e-12);radius=.02*z.flatten(1).norm(dim=1).clamp_min(1e-8);desired=-gradient*(radius/length)[:,None,None];hard=(base_logits.detach().argmax(1)!=target)&eligible;weight=eligible.float()*(1+3*hard.float());den=weight.sum().clamp_min(1.)
    out=net(x,z,b);delta=out-z;direction=((delta-desired).flatten(1).square().sum(1)/radius.square()).mul(weight).sum()/den;logits=teacher.classify_features(teacher.features(out));ce=(F.cross_entropy(logits,target,reduction='none')*weight).sum()/den;iq=((out-ground).flatten(1).square().sum(1)/ground.flatten(1).square().sum(1).clamp_min(1e-8)).mul(eligible.float()).sum()/eligible.float().sum();return .5*direction+.03*ce+.2*iq,dict(eligible=int(eligible.sum().item()),hard=int(hard.sum().item()),labeled=int(known.sum().item()))

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--commit',required=True);p.add_argument('--device',default='cuda:0');p.add_argument('--smoke',action='store_true');a=p.parse_args()
    if a.out.exists():raise FileExistsError('Immutable source output required')
    a.out.mkdir(parents=True);torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.manual_seed(2026121154)
    cp=a.root/'runs'/SOURCE/'checkpoint_epoch10.pt';pp=a.root/'runs'/PRIOR_RUN/'source_prior.pt'
    if hashlib.sha256(cp.read_bytes()).hexdigest()!=SHA or hashlib.sha256(pp.read_bytes()).hexdigest()!=PRIOR_SHA:raise ValueError('Exact blind parent and ground prior required')
    state=torch.load(cp,map_location='cpu',weights_only=False);prior=torch.load(pp,map_location='cpu',weights_only=False);blind=BlindActionNet().to(a.device);blind.load_state_dict(state['model']);blind.eval().requires_grad_(False);means=prior['means'].to(a.device);covs=prior['covariances'].to(a.device);priors=prior['priors'].to(a.device);eye=torch.eye(512,device=a.device)
    cache,clean,rx,training,pv,ph=s.source_contract(a.root);validation,heldout=s.new_splits(cache,rx,training,pv,ph);rng=np.random.default_rng(2026120355);prior_ids=np.sort(rng.choice(training,size=8192,replace=False));pool=np.setdiff1d(training,prior_ids,assume_unique=True);frng=np.random.default_rng(2026121155);fit=np.sort(frng.choice(pool,size=4096,replace=False))
    if np.intersect1d(fit,np.r_[prior_ids,validation,heldout]).size:raise ValueError('Disjoint source training and heldout IDs required')
    roles=np.load(cache/'train/roles.npy',allow_pickle=False);teacher,contract=core.teacher(a.root,a.device);teacher.eval().requires_grad_(False);source_truth=labels_source.source_labels(a.root,cache,clean,roles,contract);labels=np.full(len(roles),-1,dtype=np.int64);labels[fit]=source_truth[fit];del source_truth
    if np.any(labels[roles=='U_s']!=-1) or np.any(labels[np.r_[validation,heldout]]!=-1):raise ValueError('Only source L_s fit labels may enter loss')
    fresh=a.root/'runs/20261031-phase1-DURA-freshViews-E30-s2026092701-r01/source_cache'
    if (fresh/'train/physical_ids.npy').resolve()!=(cache/'train/physical_ids.npy').resolve():raise ValueError('Original source physical IDs required')
    for epoch in (21,22,23):
        info=core.read(fresh/'epochs'/('%02d'%epoch)/'complete.json')
        if info.get('count')!=63000 or info.get('namespace')!=f'learned-action/train/epoch{epoch:02d}' or any(info.get(k)!=0 for k in ('TX_labels_read','F_outputs_read','target_reads')) or info.get('native_engine_modified') is not False:raise ValueError('Fresh source view provenance required')
    received,scene=s.epoch_data(fresh,21);ls=fit[(roles[fit]=='L_s')&(scene[fit]!='clean')][:4];us=fit[(roles[fit]=='U_s')&(scene[fit]!='clean')][:4];ids=np.r_[ls,us]
    if len(ls)!=4 or len(us)!=4:raise ValueError('Mixed source smoke required')
    net=IdentityStudent().to(a.device);x=core.tensor(received[ids],a.device);y=core.tensor(clean[ids],a.device);z,b=prior_output(blind,x,means,covs,priors,eye);target=torch.tensor(labels[ids].tolist(),device=a.device,dtype=torch.long);loss,counts=student_loss(net,teacher,x,z,b,y,target)
    if not torch.isfinite(loss):raise ValueError('Finite source loss required')
    loss.backward()
    if net.output.weight.grad is None or not torch.isfinite(net.output.weight.grad).all() or net.output.weight.grad.abs().sum()==0 or any(p.requires_grad or p.grad is not None for p in list(teacher.parameters())+list(blind.parameters())):raise ValueError('Nonzero student gradient and frozen F/parent required')
    net.zero_grad(set_to_none=True);smoke=dict(status='PASS',records=8,L_s_records=4,U_s_records=4,eligible=counts['eligible'],hard=counts['hard'],F_frozen=True,blind_parent_frozen=True,DURA_loaded=False,L_s_metadata_labels=6300,fit_L_s_labels=int(np.count_nonzero(labels[fit]>=0)),U_s_truth_reads=0,V_reads=0,test_reads=0);print('IDENTITY_STUDENT_SMOKE',json.dumps(smoke),flush=True)
    if a.smoke:
        with (a.out/'result.json').open('x',encoding='utf8') as f:json.dump(smoke,f,indent=2)
        return
    opt=torch.optim.AdamW(net.parameters(),lr=.0003,weight_decay=.001);losses=[];counts_epoch=[];updates=0;eligible_total=0;hard_total=0;labeled_total=0;tick=time.monotonic();net.train()
    for epoch in (21,22,23):
        received,scene=s.epoch_data(fresh,epoch);order=frng.permutation(fit);total=0.;n=0
        for start in range(0,len(order),16):
            ids=order[start:start+16];ids=ids[scene[ids]!='clean']
            if not len(ids):continue
            x=core.tensor(received[ids],a.device);ground=core.tensor(clean[ids],a.device);z,b=prior_output(blind,x,means,covs,priors,eye);target=torch.tensor(labels[ids].tolist(),device=a.device,dtype=torch.long);loss,counts=student_loss(net,teacher,x,z,b,ground,target)
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite source student loss')
            opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),1.,error_if_nonfinite=True);opt.step();n+=len(ids);total+=float(loss.detach())*len(ids);eligible_total+=counts['eligible'];hard_total+=counts['hard'];labeled_total+=counts['labeled'];updates+=1
        losses.append(total/n);counts_epoch.append(n);print('IDENTITY_STUDENT_EPOCH',epoch,'used',n,'eligible',eligible_total,'hard',hard_total,'labeled',labeled_total,'updates',updates,'loss',losses[-1],'seconds',round(time.monotonic()-tick,1),flush=True)
    if updates<700 or labeled_total<500 or any(p.grad is not None for p in list(teacher.parameters())+list(blind.parameters())):raise ValueError('Source budget or frozen model invariant failed')
    with (a.out/'checkpoint_epoch03.pt').open('xb') as f:torch.save(dict(commit=a.commit,updates=updates,model=net.state_dict(),parent_sha256=SHA,prior_sha256=PRIOR_SHA),f)
    net.eval().requires_grad_(False);dura=source_prior.frozen_dura(a.root,a.device);received,scene=s.epoch_data(cache,19);ids=np.concatenate([np.sort(rng.choice(heldout[scene[heldout]==v],size=64,replace=False)) for v in SCENES]);values=[]
    with torch.no_grad():
        for start in range(0,len(ids),8):
            idx=ids[start:start+8];x=core.tensor(received[idx],a.device);ground=core.tensor(clean[idx],a.device);conf,target=F.softmax(teacher.classify_features(teacher.features(ground)),1).max(1);z,b=prior_output(blind,x,means,covs,priors,eye);candidate=net(x,z,b);den=ground.square().sum((1,2)).clamp_min(1e-12);base_norm=z.flatten(1).norm(dim=1).clamp_min(1e-8);ratio=(candidate-z).flatten(1).norm(dim=1)/base_norm
            if (ratio>.02001).any():raise ValueError('2% source correction budget required')
            cols=[conf,target.float(),ratio]
            for signal in (x,dura(x),z,candidate):
                pred=teacher.classify_features(teacher.features(signal)).argmax(1);iq=(signal-ground).square().sum((1,2))/den;cols.extend((pred.float(),iq))
            values.append(torch.stack(cols,1).cpu())
    values=torch.cat(values);results={}
    for j,view in enumerate(SCENES):
        rows=values[j*64:(j+1)*64];valid=rows[:,0]>=.95;cell=dict(count=len(rows),valid_count=int(valid.sum().item()),mean_relative_perturbation=float(rows[:,2].mean().item()))
        for i,name in enumerate(('raw','dura','parent','student')):cell[name]=dict(correct=int((valid&(rows[:,3+2*i]==rows[:,1])).sum().item()),mean_iq_nmse=float(rows[:,4+2*i].mean().item()))
        results[view]=cell
    result=dict(status='COMPLETE_SOURCE_ONLY_IDENTITY_STUDENT_PILOT',commit=a.commit,fit_ids=len(fit),fit_L_s_labels=int(np.count_nonzero(labels[fit]>=0)),L_s_metadata_labels=6300,updates=updates,epoch_losses=losses,epoch_counts=counts_epoch,eligible=eligible_total,hard=hard_total,labeled=labeled_total,scenes=results,probe_ids=len(ids),F_frozen=True,blind_parent_frozen=True,DURA_loaded_after_training=True,U_s_truth_reads=0,V_reads=0,test_reads=0,caveat='Training uses input gradients of F and L_s source labels or high-confidence U_s clean-F pseudo labels. IQ bound does not ensure physical fidelity; source IDs previously explored; source F pseudo targets not test truth.')
    with (a.out/'result.json').open('x',encoding='utf8') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    print('IDENTITY_STUDENT_COMPLETE',json.dumps(result),flush=True)
if __name__=='__main__':main()
