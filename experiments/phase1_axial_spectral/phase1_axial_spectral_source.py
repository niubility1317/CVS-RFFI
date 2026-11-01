"""Phase1 axial spectral restoration: source IQ training and selection only."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from phase1_axial_spectral_mask import AxialSpectralMask
from phase1_complex_invariant_loss import complex_gain_invariant_error
from experiments.pamf_phase1_flow import experiment as core

EPOCHS = (2, 5, 10)
BLENDS = (.25, .5, 1.)
VIEWS = ('clean', 'practical_high', 'practical_mid', 'practical_low_suburban', 'practical_high_urban', 'practical_mid_urban', 'practical_low_urban')

@torch.no_grad()
def align(wave, reference):
    """Least-squares complex gain and ±64 sample shift on central 128."""
    if wave.shape != reference.shape or wave.ndim != 3 or wave.shape[1:] != (2, 256):
        raise ValueError('matching IQ required')
    pad = F.pad(wave, (64, 64), mode='reflect')
    full = torch.complex(pad[:,0], pad[:,1]).unfold(-1, 256, 1)
    part = full[:,:,64:-64]
    target = torch.complex(reference[:,0,64:-64], reference[:,1,64:-64])[:,None,:]
    gain = (part.conj()*target).sum(-1)/part.abs().square().sum(-1).clamp_min(1e-20)
    error = (gain[:,:,None]*part-target).abs().square().mean(-1)
    choice = error.argmin(1); row = torch.arange(len(wave),device=wave.device)
    value = gain[row,choice,None]*full[row,choice]
    return torch.stack((value.real,value.imag),1), error[row,choice]


def source_contract(root):
    from experiments.cric_dura_comparison.bootstrap import activate
    runtime=core.read(root/'releases'/(core.F_RUN+'-release')/'runtime.json')
    if runtime['code_commit']!='3ff8f4b7e85b5608f5b45350aa18081499a7ef1d':
        raise ValueError('Exact Phase1 runtime required')
    activate(runtime)
    complete=core.read(root/'runs'/core.F_RUN/'F'/core.F_ROW/'source/complete.json')
    contract=core.read(complete['config']['source_contract'])
    cache,clean,roles=core.data(root,contract)
    ids=np.load(cache/'train/physical_ids.npy',allow_pickle=False)
    rx=np.load(cache/'train/rx.npy',allow_pickle=False)
    day=np.load(cache/'train/day.npy',allow_pickle=False)
    rng=np.random.default_rng(2026110551)
    fixed=np.concatenate([np.sort(rng.choice(np.flatnonzero(roles==role),size=take,replace=False)) for role,take in (('L_s',1024),('U_s',3072))])
    if len(np.unique(fixed))!=4096 or len(np.unique(ids))!=63000: raise ValueError('Source ID contract violated')
    is_train=~(np.isin(rx,[4,6,8]) & (day==3));is_train[fixed]=False
    valid=fixed[np.isin(rx[fixed],[4,6]) & (day[fixed]==3)]
    hold=fixed[(rx[fixed]==8) & (day[fixed]==3)]
    if len(valid)==0 or len(hold)==0 or np.any(is_train & np.isin(rx,[4,6,8]) & (day==3)) or np.intersect1d(np.flatnonzero(is_train),np.concatenate((valid,hold))).size:
        raise ValueError('Disjoint source splits required')
    return cache,clean,rx,np.flatnonzero(is_train),valid,hold


def epoch_data(cache, epoch):
    folder=cache/'epochs'/('%02d'%epoch)
    received=np.load(folder/'received.npy',mmap_mode='r',allow_pickle=False)
    with np.load(folder/'groups.npz',allow_pickle=False) as group:
        scene=group['scene']
    if received.shape!=(63000,2,256) or received.dtype!=np.float32 or scene.shape!=(63000,) or set(np.unique(scene))!=set(VIEWS):
        raise ValueError('Complete seven-scene source cache required')
    return received,scene


def smoke(model,clean,received,indices,device):
    x=core.tensor(received[indices],device);y=core.tensor(clean[indices],device)
    with torch.no_grad():target,_=align(y,x)
    model.train();out=model(x);loss=(complex_gain_invariant_error(out,target)+.5*(out-target).square().mean((1,2))/target.square().mean((1,2)).clamp_min(1e-8)).mean()
    if not torch.isfinite(loss):raise ValueError('Finite source smoke loss required')
    loss.backward()
    for grad in (model.output.weight.grad,model.temporal_output.weight.grad):
        if grad is None or not torch.isfinite(grad).all() or not (grad.abs().sum()>0):raise ValueError('Source smoke needs nonzero finite gradients')
    model.zero_grad(set_to_none=True)
    return dict(status='PASS',records=len(indices),finite_loss=float(loss),source_only=True,DURA_used=False,F_loaded=False,TX_labels_read=0,V_reads=0,test_reads=0)

def evaluate(model,clean,received,scene,rx,valid,hold,device):
    ids=np.concatenate((valid,hold)); group=np.concatenate((np.ones(len(valid),dtype=np.int64),np.full(len(hold),2,dtype=np.int64)))
    result={}; model.eval()
    with torch.no_grad():
        chunks=[]
        for start in range(0,len(ids),32):
            index=ids[start:start+32];x=core.tensor(received[index],device);y=core.tensor(clean[index],device);a=x
            pred=model(x)
            den=y[:,:,64:-64].square().sum(1).mean(1).clamp_min(1e-12)
            _,baseline=align(a,y); cols=[baseline/den]
            for alpha in BLENDS:
                _,err=align(a+alpha*(pred-a),y);cols.append(err/den)
            chunks.append(torch.stack(cols,1).cpu())
        values=np.from_dlpack(torch.cat(chunks).contiguous())
    for part in (1,2):
        for receiver in ((4,6) if part==1 else (8,)):
            for view in VIEWS:
                mask=(group==part)&(rx[ids]==receiver)&(scene[ids]==view)
                if not mask.any():raise ValueError('Every selected source receiver/view cell must be present')
                v=values[mask]
                result[f'{part}:{receiver}:{view}']=dict(count=int(mask.sum()),baseline=float(v[:,0].mean()),candidate=[float(v[:,j].mean()) for j in (1,2,3)])
    return result


def choose(candidates):
    best=dict(kind='raw',validation_improvement=0.)
    for epoch,groups in candidates.items():
        for col,alpha in enumerate(BLENDS):
            validation=[g for k,g in groups.items() if k.startswith('1:')]
            improvement=sum(g['baseline']-g['candidate'][col] for g in validation)
            guard=all(g['candidate'][col]<=g['baseline']+(1e-4 if k.endswith(':clean') else 1e-6) for k,g in groups.items() if k.startswith('1:'))
            if guard and improvement>best['validation_improvement']+1e-8:
                best=dict(kind='axial_direct_iq',epoch=int(epoch),alpha=alpha,validation_improvement=improvement)
    return best


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--commit',required=True);p.add_argument('--device',default='cuda:0');p.add_argument('--smoke',action='store_true');args=p.parse_args()
    if args.out.exists():raise FileExistsError('Immutable output root exists')
    args.out.mkdir(parents=True);torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.manual_seed(2026110754);rng=np.random.default_rng(2026110755)
    cache,clean,rx,training,valid,hold=source_contract(args.root)
    model=AxialSpectralMask().to(args.device)
    received,_=epoch_data(cache,1)
    if args.smoke:
        report=smoke(model,clean,received,training[:32],args.device)
        with (args.out/'result.json').open('x',encoding='utf8') as f:json.dump(report,f,indent=2,allow_nan=False);f.write('\n')
        print('AXIAL_SOURCE_SMOKE',json.dumps(report),flush=True);return
    opt=torch.optim.AdamW(model.parameters(),lr=.0003,weight_decay=.001)
    states={};losses=[];counts=[]
    for epoch in range(1,11):
        received,_=epoch_data(cache,epoch);order=rng.permutation(training);model.train();total=0.;count=0
        for start in range(0,len(order),32):
            index=order[start:start+32];x=core.tensor(received[index],args.device)
            with torch.no_grad():
                a=x;y=core.tensor(clean[index],args.device);target,_=align(y,a)
            out=model(x)
            norm=target.square().mean((1,2)).clamp_min(1e-8)
            loss=(complex_gain_invariant_error(out,target)+.5*(out-target).square().mean((1,2))/norm+.01*(out-a).square().mean((1,2))/norm).mean()
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite source IQ objective')
            opt.zero_grad(set_to_none=True);loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step()
            total+=float(loss.detach())*len(a);count+=len(a)
        if count<100:raise ValueError('Insufficient gated source training records')
        losses.append(total/count);counts.append(count)
        if epoch in EPOCHS:
            states[epoch]={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            with (args.out/('checkpoint_epoch%02d.pt'%epoch)).open('xb') as f:torch.save(dict(commit=args.commit,epoch=epoch,model=states[epoch]),f)
        print('AXIAL_SOURCE_EPOCH',epoch,'trained',count,'loss',losses[-1],flush=True)
    received,scene=epoch_data(cache,20);candidates={}
    for epoch in EPOCHS:
        model.load_state_dict(states[epoch]);candidates[epoch]=evaluate(model,clean,received,scene,rx,valid,hold,args.device)
        print('AXIAL_SOURCE_CANDIDATES_EVALUATED',epoch,flush=True)
    selected=choose(candidates)
    if selected['kind']!='raw':
        with (args.out/'selected.pt').open('xb') as f:torch.save(dict(commit=args.commit,selected=selected,model=states[selected['epoch']]),f)
    report=dict(selected=selected,validation=candidates,train_ids=len(training),validation_ids=len(valid),heldout_ids=len(hold),losses=losses,training_counts=counts,source_only=True,DURA_used=False,F_loaded=False,TX_labels_read=0,V_reads=0,test_reads=0)
    with (args.out/'source_selection.json').open('x',encoding='utf8') as f:json.dump(report,f,indent=2,allow_nan=False);f.write('\n')
    print('AXIAL_IQ_SELECTION_FIXED',json.dumps(selected),flush=True)

if __name__=='__main__':main()
