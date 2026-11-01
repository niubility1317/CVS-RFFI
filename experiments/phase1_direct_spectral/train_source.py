"""Train direct spectral-temporal restorer on Phase1 source pairs only."""
import argparse,json,sys,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from direct_spectral_model import DirectSpectralRestorer
DEP=Path('/home/szu2070436088/2510044040/CV-SincNet/releases/20261102-phase1-bayes-residual-full-s2026092701-r01-release')
if (DEP/'release_commit.txt').read_text().strip()!='848c4b4c47b054f43c8303638e5aa549154ebfd4':raise ValueError('Exact Phase1 source interface release required')
sys.path.insert(0,str(DEP/'experiments/phase1_bayes_residual_full'))
from bayes_residual_full import s,core,labels_source,source_prior,SCENES

def objective(net,teacher,x,y,labels):
    with torch.no_grad():
        clean_f=teacher.features(y);soft=F.softmax(teacher.classify_features(clean_f)/2,1)
    out=net(x);features=teacher.features(out);logits=teacher.classify_features(features)
    denom=y.square().sum((1,2)).clamp_min(1e-8)
    iq=((out-y).square().sum((1,2))/denom).mean()
    feature=(1-F.cosine_similarity(features,clean_f,dim=1,eps=1e-8)).mean()
    kl=F.kl_div(F.log_softmax(logits/2,dim=1),soft,reduction='batchmean')*4
    valid=labels>=0
    ce=F.cross_entropy(logits[valid],labels[valid]) if valid.any() else logits.sum()*0
    return .7*iq+.2*feature+.1*kl+.15*ce

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--commit',required=True);p.add_argument('--device',default='cuda:0');p.add_argument('--smoke',action='store_true');a=p.parse_args()
    if a.out.exists():raise FileExistsError('Immutable output required')
    a.out.mkdir(parents=True);torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.manual_seed(2026120954)
    cache,clean,rx,training,pv,ph=s.source_contract(a.root);validation,heldout=s.new_splits(cache,rx,training,pv,ph)
    rng=np.random.default_rng(2026120955);fit=np.sort(rng.choice(training,size=4096,replace=False))
    if np.intersect1d(fit,np.r_[validation,heldout]).size:raise ValueError('Source roles must be disjoint')
    roles=np.load(cache/'train/roles.npy',allow_pickle=False);teacher,contract=core.teacher(a.root,a.device);teacher.eval().requires_grad_(False)
    source_truth=labels_source.source_labels(a.root,cache,clean,roles,contract);labels=np.full(len(roles),-1,dtype=np.int64);labels[fit]=source_truth[fit];del source_truth
    if np.any(labels[roles=='U_s']!=-1) or np.any(labels[np.r_[validation,heldout]]!=-1):raise ValueError('Only labeled source training may enter optimization')
    fresh=a.root/'runs/20261031-phase1-DURA-freshViews-E30-s2026092701-r01/source_cache'
    if (fresh/'train/physical_ids.npy').resolve()!=(cache/'train/physical_ids.npy').resolve():raise ValueError('Original source physical IDs required')
    for epoch in (21,22,23):
        info=core.read(fresh/'epochs'/('%02d'%epoch)/'complete.json')
        if info.get('count')!=63000 or info.get('namespace')!=f'learned-action/train/epoch{epoch:02d}' or any(info.get(k)!=0 for k in ('TX_labels_read','F_outputs_read','target_reads')) or info.get('native_engine_modified') is not False:raise ValueError('Source view provenance mismatch')
    rec,sc=s.epoch_data(fresh,21);ls=fit[(roles[fit]=='L_s')&(sc[fit]!='clean')][:4];us=fit[(roles[fit]=='U_s')&(sc[fit]!='clean')][:4];idx=np.r_[ls,us]
    if len(ls)!=4 or len(us)!=4:raise ValueError('Need 4 source labeled and 4 unlabeled records')
    net=DirectSpectralRestorer().to(a.device);x=core.tensor(rec[idx],a.device);y=core.tensor(clean[idx],a.device);label=torch.tensor(labels[idx].tolist(),device=a.device,dtype=torch.long);loss=objective(net,teacher,x,y,label)
    if not torch.isfinite(loss):raise ValueError('Finite source loss required')
    loss.backward()
    if net.spectrum.weight.grad is None or net.output.weight.grad is None or not torch.isfinite(net.spectrum.weight.grad).all() or not torch.isfinite(net.output.weight.grad).all() or net.spectrum.weight.grad.abs().sum()==0 or net.output.weight.grad.abs().sum()==0:raise ValueError('Finite nonzero source gradients required')
    if any(p.requires_grad or p.grad is not None for p in teacher.parameters()):raise ValueError('Frozen F required')
    net.zero_grad(set_to_none=True);smoke=dict(status='PASS',records=8,L_s=4,U_s=4,F_frozen=True,DURA_loaded=False,blind_parent_loaded=False,L_s_metadata_labels=6300,train_L_s_labels=int(np.count_nonzero(labels[fit]>=0)),U_s_truth_reads=0,V_reads=0,test_reads=0);print('DIRECT_SPECTRAL_SMOKE',json.dumps(smoke),flush=True)
    if a.smoke:
        with (a.out/'result.json').open('x',encoding='utf8') as f:json.dump(smoke,f,indent=2)
        return
    opt=torch.optim.AdamW(net.parameters(),lr=.0003,weight_decay=.001);counts=[];losses=[];updates=0;used_labeled=0;tick=time.monotonic();net.train()
    for epoch in (21,22,23):
        rec,sc=s.epoch_data(fresh,epoch);order=rng.permutation(fit);n=0;total=0.
        for start in range(0,len(order),16):
            idx=order[start:start+16];idx=idx[sc[idx]!='clean']
            if not len(idx):continue
            x=core.tensor(rec[idx],a.device);y=core.tensor(clean[idx],a.device);label=torch.tensor(labels[idx].tolist(),device=a.device,dtype=torch.long);loss=objective(net,teacher,x,y,label)
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite training objective')
            opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),1.,error_if_nonfinite=True);opt.step();n+=len(idx);total+=float(loss.detach())*len(idx);used_labeled+=int((label>=0).sum().item());updates+=1
        counts.append(n);losses.append(total/n);print('DIRECT_SPECTRAL_EPOCH',epoch,'used',n,'labeled_total',used_labeled,'updates',updates,'loss',losses[-1],'seconds',round(time.monotonic()-tick,1),flush=True)
    if updates<700 or used_labeled<500 or any(p.grad is not None for p in teacher.parameters()):raise ValueError('Source training contract failed')
    with (a.out/'checkpoint_epoch03.pt').open('xb') as f:torch.save(dict(commit=a.commit,updates=updates,model=net.state_dict()),f)
    net.eval().requires_grad_(False);dura=source_prior.frozen_dura(a.root,a.device);received,scene=s.epoch_data(cache,19);r_probe=np.random.default_rng(2026120355);ids=np.concatenate([np.sort(r_probe.choice(heldout[scene[heldout]==v],size=64,replace=False)) for v in SCENES]);out=[]
    with torch.no_grad():
        for start in range(0,len(ids),8):
            idx=ids[start:start+8];x=core.tensor(received[idx],a.device);y=core.tensor(clean[idx],a.device);conf,target=F.softmax(teacher.classify_features(teacher.features(y)),1).max(1);den=y.square().sum((1,2)).clamp_min(1e-12);cols=[conf,target.float()]
            for signal in (x,dura(x),net(x)):
                pred=teacher.classify_features(teacher.features(signal)).argmax(1);iq=(signal-y).square().sum((1,2))/den;cols.extend((pred.float(),iq))
            out.append(torch.stack(cols,1).cpu())
    values=torch.cat(out);results={}
    for j,view in enumerate(SCENES):
        rows=values[j*64:(j+1)*64];mask=rows[:,0]>=.95;cell=dict(count=len(rows),valid_count=int(mask.sum().item()))
        for i,name in enumerate(('raw','dura','direct')):cell[name]=dict(correct=int((mask&(rows[:,2+2*i]==rows[:,1])).sum().item()),mean_iq_nmse=float(rows[:,3+2*i].mean().item()))
        results[view]=cell
    result=dict(status='COMPLETE_SOURCE_ONLY_DIRECT_SPECTRAL_PILOT',commit=a.commit,fit_ids=len(fit),train_L_s_labels=int(np.count_nonzero(labels[fit]>=0)),used_labeled=used_labeled,updates=updates,epoch_losses=losses,epoch_counts=counts,scenes=results,probe_ids=len(ids),F_frozen=True,DURA_loaded_after_training=True,blind_parent_loaded=False,L_s_metadata_labels=6300,U_s_truth_reads=0,V_reads=0,test_reads=0,caveat='Source pseudo targets not truth; physical IDs previously seen in other views; no independent blind claim.')
    with (a.out/'result.json').open('x',encoding='utf8') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    print('DIRECT_SPECTRAL_COMPLETE',json.dumps(result),flush=True)
if __name__=='__main__':main()
