"""Phase1 source-only learned clean prior with blind data-prox reconstruction."""
import argparse,hashlib,json,sys,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from score_prox_model import GroundDenoiser,data_prox
DEP=Path('/home/szu2070436088/2510044040/CV-SincNet/releases/20261102-phase1-bayes-residual-full-s2026092701-r01-release')
if (DEP/'release_commit.txt').read_text().strip()!='848c4b4c47b054f43c8303638e5aa549154ebfd4':raise ValueError('Exact source interface release required')
sys.path.insert(0,str(DEP/'experiments/phase1_bayes_residual_full'))
from bayes_residual_full import s,core,source_prior,BlindActionNet,action_matrix,posterior,SOURCE,SHA,PRIOR_RUN,PRIOR_SHA,SCENES

def loss_on_clean(net,teacher,clean,sigma):
    rms=clean.square().sum(1).mean(1).sqrt().clamp_min(1e-8);noise_rms=sigma*rms;noisy=clean+torch.randn_like(clean)*(noise_rms[:,None,None]/2**.5)
    with torch.no_grad():cf=teacher.features(clean);soft=F.softmax(teacher.classify_features(cf)/2,1)
    out=net(noisy,noise_rms);of=teacher.features(out);logits=teacher.classify_features(of)
    iq=((out-clean).square().sum((1,2))/clean.square().sum((1,2)).clamp_min(1e-8)).mean();feat=(1-F.cosine_similarity(of,cf,dim=1,eps=1e-8)).mean();kl=F.kl_div(F.log_softmax(logits/2,dim=1),soft,reduction='batchmean')*4
    return iq+.15*feat+.05*kl

def reconstruct(net,A,received,parent,strength):
    current=parent
    for frac in (.20,.10,.05):
        rms=current.square().sum(1).mean(1).sqrt().clamp_min(1e-8);prior=net(current,frac*rms);current=data_prox(A,received.reshape(len(received),-1),prior.reshape(len(received),-1),strength).reshape_as(received)
    return current

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--commit',required=True);p.add_argument('--device',default='cuda:0');p.add_argument('--smoke',action='store_true');a=p.parse_args()
    if a.out.exists():raise FileExistsError('Immutable output required')
    a.out.mkdir(parents=True);torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.manual_seed(2026121054)
    cache,clean,rx,training,pv,ph=s.source_contract(a.root);validation,heldout=s.new_splits(cache,rx,training,pv,ph);rng=np.random.default_rng(2026120355);prior_ids=np.sort(rng.choice(training,size=8192,replace=False));pool=np.setdiff1d(training,prior_ids,assume_unique=True);frng=np.random.default_rng(2026121055);fit=np.sort(frng.choice(pool,size=4096,replace=False))
    if np.intersect1d(fit,np.r_[prior_ids,validation,heldout]).size:raise ValueError('Disjoint source role IDs required')
    teacher,_=core.teacher(a.root,a.device);teacher.eval().requires_grad_(False);net=GroundDenoiser().to(a.device);ids=fit[:8];ground=core.tensor(clean[ids],a.device);sigma=torch.linspace(.05,.4,len(ids),device=a.device);loss=loss_on_clean(net,teacher,ground,sigma)
    if not torch.isfinite(loss):raise ValueError('Finite source loss required')
    loss.backward()
    if net.output.weight.grad is None or not torch.isfinite(net.output.weight.grad).all() or net.output.weight.grad.abs().sum()==0 or any(p.requires_grad or p.grad is not None for p in teacher.parameters()):raise ValueError('Finite prior gradient and frozen classifier required')
    net.zero_grad(set_to_none=True);smoke=dict(status='PASS',records=8,F_frozen=True,DURA_loaded=False,blind_parent_loaded=False,TX_labels_read=0,U_s_truth_reads=0,V_reads=0,test_reads=0);print('SCORE_PROX_SOURCE_SMOKE',json.dumps(smoke),flush=True)
    if a.smoke:
        with (a.out/'result.json').open('x',encoding='utf8') as f:json.dump(smoke,f,indent=2)
        return
    opt=torch.optim.AdamW(net.parameters(),lr=.0003,weight_decay=.001);losses=[];counts=[];updates=0;tick=time.monotonic();net.train();choices=torch.tensor([.05,.1,.2,.4],device=a.device)
    for epoch in range(1,4):
        order=frng.permutation(fit);total=0.;n=0
        for start in range(0,len(order),16):
            idx=order[start:start+16];ground=core.tensor(clean[idx],a.device);sigma=choices[torch.randint(len(choices),(len(idx),),device=a.device)];loss=loss_on_clean(net,teacher,ground,sigma)
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite prior training loss')
            opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),1.,error_if_nonfinite=True);opt.step();total+=float(loss.detach())*len(idx);n+=len(idx);updates+=1
        losses.append(total/n);counts.append(n);print('SCORE_PROX_EPOCH',epoch,'count',n,'loss',losses[-1],'updates',updates,'seconds',round(time.monotonic()-tick,1),flush=True)
    if updates!=768 or any(p.grad is not None for p in teacher.parameters()):raise ValueError('Source budget or freeze contract violated')
    with (a.out/'checkpoint_epoch03.pt').open('xb') as f:torch.save(dict(commit=a.commit,updates=updates,model=net.state_dict()),f)
    net.eval().requires_grad_(False);checkpoint=a.root/'runs'/SOURCE/'checkpoint_epoch10.pt';prior_path=a.root/'runs'/PRIOR_RUN/'source_prior.pt'
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest()!=SHA or hashlib.sha256(prior_path.read_bytes()).hexdigest()!=PRIOR_SHA:raise ValueError('Exact fixed source blind model and prior required')
    saved=torch.load(checkpoint,map_location='cpu',weights_only=False);prior=torch.load(prior_path,map_location='cpu',weights_only=False);blind=BlindActionNet().to(a.device);blind.load_state_dict(saved['model'],strict=True);blind.eval().requires_grad_(False);means=prior['means'].to(a.device);covs=prior['covariances'].to(a.device);priors=prior['priors'].to(a.device);eye=torch.eye(512,device=a.device);dura=source_prior.frozen_dura(a.root,a.device);received,scene=s.epoch_data(cache,19);ids=np.concatenate([np.sort(rng.choice(heldout[scene[heldout]==view],size=64,replace=False)) for view in SCENES]);values=[];strengths=(.1,1.,10.)
    with torch.no_grad():
        for start in range(0,len(ids),8):
            idx=ids[start:start+8];x=core.tensor(received[idx],a.device);y=core.tensor(clean[idx],a.device);conf,target=F.softmax(teacher.classify_features(teacher.features(y)),1).max(1);h,lv=blind(x);A=action_matrix(h);N=eye[None]*lv.exp()[:,None,None];states,weights=posterior(x.reshape(len(x),512),A,N,means,covs,priors);parent=(states*weights.exp().unsqueeze(-1)).sum(1).reshape_as(x);variants=[x,dura(x),parent]+[reconstruct(net,A,x,parent,k) for k in strengths];den=y.square().sum((1,2)).clamp_min(1e-12);cols=[conf,target.float()]
            for signal in variants:
                pred=teacher.classify_features(teacher.features(signal)).argmax(1);iq=(signal-y).square().sum((1,2))/den;cols.extend((pred.float(),iq))
            batch=torch.stack(cols,1).cpu()
            if not torch.isfinite(batch).all():raise ValueError('Finite source evaluation required')
            values.append(batch)
            if (start+len(idx))%64==0:print('SCORE_PROX_EVALUATED',start+len(idx),flush=True)
    values=torch.cat(values);results={}
    for j,view in enumerate(SCENES):
        rows=values[j*64:(j+1)*64];valid=rows[:,0]>=.95;cell=dict(count=len(rows),valid_count=int(valid.sum().item()))
        for i,name in enumerate(('raw','dura','parent','prox_0.1','prox_1.0','prox_10.0')):cell[name]=dict(correct=int((valid&(rows[:,2+2*i]==rows[:,1])).sum().item()),mean_iq_nmse=float(rows[:,3+2*i].mean().item()))
        results[view]=cell
    result=dict(status='COMPLETE_SOURCE_ONLY_SCORE_PROX_PILOT',commit=a.commit,fit_ids=len(fit),prior_fit_ids=len(prior_ids),updates=updates,epoch_losses=losses,epoch_counts=counts,scenes=results,probe_ids=len(ids),strengths=strengths,sigma_schedule=[.2,.1,.05],F_frozen=True,DURA_loaded_after_training=True,blind_parent_loaded_after_training=True,TX_labels_read=0,U_s_truth_reads=0,V_reads=0,test_reads=0,caveat='Clean F pseudo targets not truth; physical IDs previously used in other views; no independent blind claim.')
    with (a.out/'result.json').open('x',encoding='utf8') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    print('SCORE_PROX_COMPLETE',json.dumps(result),flush=True)
if __name__=='__main__':main()
