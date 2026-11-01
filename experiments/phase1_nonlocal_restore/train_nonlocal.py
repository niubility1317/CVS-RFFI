"""Phase1 source-only noise-conditioned nonlocal posterior correction."""
import argparse,hashlib,json,sys,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from nonlocal_model import NoiseNonlocalRestorer
DEP=Path('/home/szu2070436088/2510044040/CV-SincNet/releases/20261102-phase1-bayes-residual-full-s2026092701-r01-release')
if (DEP/'release_commit.txt').read_text().strip()!='848c4b4c47b054f43c8303638e5aa549154ebfd4':raise ValueError('Exact Phase1 source interface release required')
sys.path.insert(0,str(DEP/'experiments/phase1_bayes_residual_full'))
from bayes_residual_full import s,core,labels_source,source_prior,BlindActionNet,prior_output,SOURCE,SHA,PRIOR_RUN,PRIOR_SHA,SCENES

def objective(net,teacher,x,z,b,logvar,ground,labels):
    with torch.no_grad():
        clean_f=teacher.features(ground);clean_logits=teacher.classify_features(clean_f);prob=F.softmax(clean_logits,1);conf,pseudo=prob.max(1);soft=F.softmax(clean_logits/2,1);known=labels>=0;target=torch.where(known,labels,pseudo);eligible=known|(conf>=.95);base=teacher.classify_features(teacher.features(z));hard=(base.argmax(1)!=target)&eligible;weight=eligible.float()*(1+3*hard.float())
    out=net(x,z,b,logvar);features=teacher.features(out);logits=teacher.classify_features(features);iq=((out-ground).flatten(1).square().sum(1)/ground.flatten(1).square().sum(1).clamp_min(1e-8)).mean();feature=(1-F.cosine_similarity(features,clean_f,dim=1,eps=1e-8)).mean();kl=F.kl_div(F.log_softmax(logits/2,dim=1),soft,reduction='none').sum(1)*4;kl=(kl*weight).sum()/weight.sum().clamp_min(1.)
    ce=F.cross_entropy(logits[known],labels[known]) if known.any() else logits.sum()*0
    return .7*iq+.2*feature+.1*kl+.15*ce,dict(eligible=int(eligible.sum().item()),hard=int(hard.sum().item()),labeled=int(known.sum().item()))

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--commit',required=True);p.add_argument('--device',default='cuda:0');p.add_argument('--smoke',action='store_true');a=p.parse_args()
    if a.out.exists():raise FileExistsError('Immutable source output required')
    a.out.mkdir(parents=True);torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.manual_seed(2026121254)
    cp=a.root/'runs'/SOURCE/'checkpoint_epoch10.pt';pp=a.root/'runs'/PRIOR_RUN/'source_prior.pt'
    if hashlib.sha256(cp.read_bytes()).hexdigest()!=SHA or hashlib.sha256(pp.read_bytes()).hexdigest()!=PRIOR_SHA:raise ValueError('Exact blind parent and source prior required')
    state=torch.load(cp,map_location='cpu',weights_only=False);prior=torch.load(pp,map_location='cpu',weights_only=False);blind=BlindActionNet().to(a.device);blind.load_state_dict(state['model']);blind.eval().requires_grad_(False);means=prior['means'].to(a.device);covs=prior['covariances'].to(a.device);priors=prior['priors'].to(a.device);eye=torch.eye(512,device=a.device)
    cache,clean,rx,training,pv,ph=s.source_contract(a.root);validation,heldout=s.new_splits(cache,rx,training,pv,ph);rng=np.random.default_rng(2026120355);prior_ids=np.sort(rng.choice(training,size=8192,replace=False));pool=np.setdiff1d(training,prior_ids,assume_unique=True);frng=np.random.default_rng(2026121255);fit=np.sort(frng.choice(pool,size=4096,replace=False))
    if np.intersect1d(fit,np.r_[prior_ids,validation,heldout]).size:raise ValueError('Disjoint source fit and validation roles required')
    roles=np.load(cache/'train/roles.npy',allow_pickle=False);teacher,contract=core.teacher(a.root,a.device);teacher.eval().requires_grad_(False);source_truth=labels_source.source_labels(a.root,cache,clean,roles,contract);labels=np.full(len(roles),-1,dtype=np.int64);labels[fit]=source_truth[fit];del source_truth
    if np.any(labels[roles=='U_s']!=-1) or np.any(labels[np.r_[validation,heldout]]!=-1):raise ValueError('Only L_s fit truth may enter optimizer')
    fresh=a.root/'runs/20261031-phase1-DURA-freshViews-E30-s2026092701-r01/source_cache'
    if (fresh/'train/physical_ids.npy').resolve()!=(cache/'train/physical_ids.npy').resolve():raise ValueError('Original Phase1 source IDs required')
    for epoch in (21,22,23):
        info=core.read(fresh/'epochs'/('%02d'%epoch)/'complete.json')
        if info.get('count')!=63000 or info.get('namespace')!=f'learned-action/train/epoch{epoch:02d}' or any(info.get(k)!=0 for k in ('TX_labels_read','F_outputs_read','target_reads')) or info.get('native_engine_modified') is not False:raise ValueError('Fresh source view provenance required')
    received,scene=s.epoch_data(fresh,21);ls=fit[(roles[fit]=='L_s')&(scene[fit]!='clean')][:4];us=fit[(roles[fit]=='U_s')&(scene[fit]!='clean')][:4];ids=np.r_[ls,us]
    if len(ls)!=4 or len(us)!=4:raise ValueError('Mixed source smoke required')
    net=NoiseNonlocalRestorer().to(a.device);x=core.tensor(received[ids],a.device);y=core.tensor(clean[ids],a.device);z,b=prior_output(blind,x,means,covs,priors,eye)
    with torch.no_grad():_,lv=blind(x)
    target=torch.tensor(labels[ids].tolist(),device=a.device,dtype=torch.long);loss,counts=objective(net,teacher,x,z,b,lv,y,target)
    if not torch.isfinite(loss):raise ValueError('Finite source loss required')
    loss.backward()
    if net.output.weight.grad is None or not torch.isfinite(net.output.weight.grad).all() or net.output.weight.grad.abs().sum()==0 or any(p.requires_grad or p.grad is not None for p in list(teacher.parameters())+list(blind.parameters())):raise ValueError('Finite nonzero student gradient and frozen F/parent required')
    net.zero_grad(set_to_none=True);smoke=dict(status='PASS',records=8,L_s_records=4,U_s_records=4,eligible=counts['eligible'],hard=counts['hard'],F_frozen=True,blind_parent_frozen=True,DURA_loaded=False,L_s_metadata_labels=6300,fit_L_s_labels=int(np.count_nonzero(labels[fit]>=0)),U_s_truth_reads=0,V_reads=0,test_reads=0);print('NONLOCAL_SMOKE',json.dumps(smoke),flush=True)
    if a.smoke:
        with (a.out/'result.json').open('x',encoding='utf8') as f:json.dump(smoke,f,indent=2)
        return
    opt=torch.optim.AdamW(net.parameters(),lr=.0003,weight_decay=.001);losses=[];counts_epoch=[];updates=0;eligible_total=0;hard_total=0;labeled_total=0;tick=time.monotonic();net.train()
    for epoch in (21,22,23):
        received,scene=s.epoch_data(fresh,epoch);order=frng.permutation(fit);total=0.;n=0
        for start in range(0,len(order),16):
            ids=order[start:start+16];ids=ids[scene[ids]!='clean']
            if not len(ids):continue
            x=core.tensor(received[ids],a.device);ground=core.tensor(clean[ids],a.device);z,b=prior_output(blind,x,means,covs,priors,eye)
            with torch.no_grad():_,lv=blind(x)
            target=torch.tensor(labels[ids].tolist(),device=a.device,dtype=torch.long);loss,counts=objective(net,teacher,x,z,b,lv,ground,target)
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite source objective')
            opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),1.,error_if_nonfinite=True);opt.step();n+=len(ids);total+=float(loss.detach())*len(ids);eligible_total+=counts['eligible'];hard_total+=counts['hard'];labeled_total+=counts['labeled'];updates+=1
        losses.append(total/n);counts_epoch.append(n);print('NONLOCAL_EPOCH',epoch,'used',n,'eligible',eligible_total,'hard',hard_total,'labeled',labeled_total,'updates',updates,'loss',losses[-1],'seconds',round(time.monotonic()-tick,1),flush=True)
    if updates<700 or labeled_total<500 or any(p.grad is not None for p in list(teacher.parameters())+list(blind.parameters())):raise ValueError('Source training or freeze invariant failed')
    with (a.out/'checkpoint_epoch03.pt').open('xb') as f:torch.save(dict(commit=a.commit,updates=updates,model=net.state_dict(),parent_sha256=SHA,prior_sha256=PRIOR_SHA),f)
    net.eval().requires_grad_(False);dura=source_prior.frozen_dura(a.root,a.device);received,scene=s.epoch_data(cache,19);ids=np.concatenate([np.sort(rng.choice(heldout[scene[heldout]==v],size=64,replace=False)) for v in SCENES]);values=[]
    with torch.no_grad():
        for start in range(0,len(ids),8):
            idx=ids[start:start+8];x=core.tensor(received[idx],a.device);ground=core.tensor(clean[idx],a.device);conf,target=F.softmax(teacher.classify_features(teacher.features(ground)),1).max(1);z,b=prior_output(blind,x,means,covs,priors,eye);_,lv=blind(x);candidate=net(x,z,b,lv);den=ground.square().sum((1,2)).clamp_min(1e-12);base_norm=z.flatten(1).norm(dim=1).clamp_min(1e-8);ratio=(candidate-z).flatten(1).norm(dim=1)/base_norm
            if (ratio>.20001).any():raise ValueError('20% source correction budget required')
            cols=[conf,target.float(),ratio]
            for signal in (x,dura(x),z,candidate):
                pred=teacher.classify_features(teacher.features(signal)).argmax(1);iq=(signal-ground).square().sum((1,2))/den;cols.extend((pred.float(),iq))
            values.append(torch.stack(cols,1).cpu())
    values=torch.cat(values);results={}
    for j,view in enumerate(SCENES):
        rows=values[j*64:(j+1)*64];valid=rows[:,0]>=.95;cell=dict(count=len(rows),valid_count=int(valid.sum().item()),mean_relative_perturbation=float(rows[:,2].mean().item()))
        for i,name in enumerate(('raw','dura','parent','nonlocal')):cell[name]=dict(correct=int((valid&(rows[:,3+2*i]==rows[:,1])).sum().item()),mean_iq_nmse=float(rows[:,4+2*i].mean().item()))
        results[view]=cell
    result=dict(status='COMPLETE_SOURCE_ONLY_NONLOCAL_PILOT',commit=a.commit,fit_ids=len(fit),fit_L_s_labels=int(np.count_nonzero(labels[fit]>=0)),L_s_metadata_labels=6300,updates=updates,epoch_losses=losses,epoch_counts=counts_epoch,eligible=eligible_total,hard=hard_total,labeled=labeled_total,scenes=results,probe_ids=len(ids),F_frozen=True,blind_parent_frozen=True,DURA_loaded_after_training=True,U_s_truth_reads=0,V_reads=0,test_reads=0,caveat='Source L_s truth only fit IDs; U_s pseudo targets; attention correction limited by 20% L2 but physical fidelity not guaranteed; source IDs previously explored.')
    with (a.out/'result.json').open('x',encoding='utf8') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    print('NONLOCAL_COMPLETE',json.dumps(result),flush=True)
if __name__=='__main__':main()
