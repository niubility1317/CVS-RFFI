"""Phase1 direct restoration with frozen-teacher source training and selection."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from phase1_axial_spectral_mask import AxialSpectralMask
from phase1_complex_invariant_loss import complex_gain_invariant_error
from experiments.pamf_phase1_flow import experiment as core

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


EPOCHS=(1,2,4)


def objective(model,teacher,x,y,targets):
    with torch.no_grad():
        target,_=align(y,x);clean_features=teacher.features(y);clean_logits=teacher.classify_features(clean_features)
        soft=F.softmax(clean_logits/2.,dim=1);clean_prob=F.softmax(clean_logits,dim=1)
        labeled=targets>=0;hard_target=torch.where(labeled,targets,clean_prob.argmax(1));trusted=labeled|(clean_prob.max(1).values>=.95)
        raw_prediction=teacher.classify_features(teacher.features(x)).argmax(1)
        weight=trusted.float()*(1.+2.*(raw_prediction!=hard_target).float())
    output=x+.5*(model(x)-x);features=teacher.features(output);logits=teacher.classify_features(features)
    hard=F.cross_entropy(logits,hard_target,reduction='none');weighted=(hard*weight).sum()/weight.sum().clamp_min(1.)
    distill=F.kl_div(F.log_softmax(logits/2.,dim=1),soft,reduction='none').sum(1)*4.
    fingerprint=1.-F.cosine_similarity(features,clean_features,dim=1,eps=1e-8)
    norm=target.square().mean((1,2)).clamp_min(1e-8)
    fixed=(output-target).square().mean((1,2))/norm;deviation=(output-x).square().mean((1,2))/norm
    return weighted+(.3*distill+.2*fingerprint+.4*complex_gain_invariant_error(output,target)+.2*fixed+.01*deviation).mean()


def restrict_training_labels(labels,roles,training):
    if labels.shape!=roles.shape or np.any(labels[roles=='U_s']!=-1):raise ValueError('U_s truth must never be materialized')
    result=np.full(len(labels),-1,dtype=np.int64);result[training]=labels[training]
    if np.any(result[np.setdiff1d(np.arange(len(labels)),training)]!=-1):raise ValueError('Training label isolation failed')
    return result

def smoke(model,teacher,clean,received,indices,targets,device):
    x=core.tensor(received[indices],device);y=core.tensor(clean[indices],device)
    model.train();loss=objective(model,teacher,x,y,torch.tensor(targets[indices].tolist(),device=device,dtype=torch.long))
    if not torch.isfinite(loss):raise ValueError('Finite semantic source loss required')
    loss.backward()
    if any(p.requires_grad or p.grad is not None for p in teacher.parameters()):raise ValueError('Teacher must remain frozen')
    for grad in (model.output.weight.grad,model.temporal_output.weight.grad):
        if grad is None or not torch.isfinite(grad).all() or not (grad.abs().sum()>0):raise ValueError('Nonzero finite reconstruction gradients required')
    model.zero_grad(set_to_none=True)
    return dict(status='PASS',records=len(indices),finite_loss=float(loss.detach()),source_only=True,DURA_used=False,F_frozen=True,L_s_labels_used=int((targets[indices]>=0).sum()),U_s_labels_read=0,V_reads=0,test_reads=0)


def evaluate(model,teacher,clean,received,scene,rx,valid,hold,device):
    ids=np.concatenate((valid,hold));group=np.concatenate((np.ones(len(valid),dtype=np.int64),np.full(len(hold),2,dtype=np.int64)));model.eval()
    batches=[]
    with torch.no_grad():
        for start in range(0,len(ids),32):
            index=ids[start:start+32];x=core.tensor(received[index],device);y=core.tensor(clean[index],device)
            p=F.softmax(teacher.classify_features(teacher.features(y)),1);truth=p.argmax(1);confidence=p.max(1).values
            raw=teacher.classify_features(teacher.features(x)).argmax(1);prediction=model(x)
            den=y[:,:,64:-64].square().sum(1).mean(1).clamp_min(1e-12);_,raw_err=align(x,y)
            cols=[truth.float(),confidence,raw.float(),raw_err/den]
            for alpha in BLENDS:
                wave=x+alpha*(prediction-x);pred=teacher.classify_features(teacher.features(wave)).argmax(1);_,err=align(wave,y)
                cols.extend((pred.float(),err/den))
            batches.append(torch.stack(cols,1).cpu())
    values=np.from_dlpack(torch.cat(batches).contiguous());result={}
    for part in (1,2):
        for receiver in ((4,6) if part==1 else (8,)):
            for view in VIEWS:
                mask=(group==part)&(rx[ids]==receiver)&(scene[ids]==view);v=values[mask]
                if not len(v):raise ValueError('Every receiver and scene cell required')
                trusted=v[:,1]>=.95
                if not trusted.any():raise ValueError('Confident clean teacher targets required')
                correct=v[:,2]==v[:,0]
                result[f'{part}:{receiver}:{view}']=dict(count=len(v),valid_count=int(trusted.sum()),raw_correct=int((trusted&correct).sum()),candidate_correct=[int((trusted&(v[:,j]==v[:,0])).sum()) for j in (4,6,8)],raw_iq=float(v[:,3].mean()),candidate_iq=[float(v[:,j].mean()) for j in (5,7,9)])
    return result


def choose(candidates):
    best=dict(kind='raw',validation_net=0)
    for epoch,groups in candidates.items():
        for col,alpha in enumerate(BLENDS):
            validation=[(k,g) for k,g in groups.items() if k.startswith('1:')]
            net=sum(g['candidate_correct'][col]-g['raw_correct'] for k,g in validation if not k.endswith(':clean'))
            classification_guard=all(g['candidate_correct'][col]>=g['raw_correct']-(0 if k.endswith(':clean') else 1) for k,g in validation)
            iq_guard=all(g['candidate_iq'][col]<=(.001 if k.endswith(':clean') else g['raw_iq']*1.1+1e-6) for k,g in validation)
            if classification_guard and iq_guard and net>=2 and net>best['validation_net']:
                best=dict(kind='axial_direct_iq',epoch=int(epoch),alpha=alpha,validation_net=net)
    return best


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--commit',required=True);p.add_argument('--device',default='cuda:0');p.add_argument('--smoke',action='store_true');args=p.parse_args()
    if args.out.exists():raise FileExistsError('Immutable output root exists')
    args.out.mkdir(parents=True);torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.manual_seed(2026110954);rng=np.random.default_rng(2026110955)
    cache,clean,rx,training,valid,hold=source_contract(args.root);teacher,contract=core.teacher(args.root,args.device)
    from experiments.pamf_phase1_flow import feature_conditioned_experiment as labels_reader
    roles=np.load(cache/'train/roles.npy',allow_pickle=False)
    full_labels=labels_reader.source_labels(args.root,cache,clean,roles,contract)
    train_targets=restrict_training_labels(full_labels,roles,training);del full_labels
    labeled_count=int((train_targets>=0).sum())
    if labeled_count<1000:raise ValueError('Insufficient legitimate labeled training sources')
    parent=args.root/'runs/20261102-phase1-direct-axial-iq-s2026092701-r01/source/checkpoint_epoch10.pt'
    state=torch.load(parent,map_location='cpu',weights_only=False)
    if state['commit']!='bc9ac03f23271614f332fd06bd86e77a0cd2cca5' or state['epoch']!=10:raise ValueError('Exact direct source parent required')
    model=AxialSpectralMask().to(args.device);model.load_state_dict(state['model'],strict=True)
    received,_=epoch_data(cache,1)
    if args.smoke:
        report=smoke(model,teacher,clean,received,training[:16],train_targets,args.device)
        with (args.out/'result.json').open('x',encoding='utf8') as f:json.dump(report,f,indent=2,allow_nan=False);f.write('\n')
        print('CURRICULUM_SOURCE_SMOKE',json.dumps(report),flush=True);return
    opt=torch.optim.AdamW(model.parameters(),lr=.0001,weight_decay=.001);states={};losses=[]
    for epoch in range(1,5):
        received,_=epoch_data(cache,epoch+15);order=rng.permutation(training);model.train();total=0.;count=0
        for start in range(0,len(order),32):
            index=order[start:start+32];x=core.tensor(received[index],args.device);y=core.tensor(clean[index],args.device)
            loss=objective(model,teacher,x,y,torch.tensor(train_targets[index].tolist(),device=args.device,dtype=torch.long))
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite semantic source loss')
            opt.zero_grad(set_to_none=True);loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step()
            total+=float(loss.detach())*len(x);count+=len(x)
        if count!=len(training) or any(p.grad is not None for p in teacher.parameters()):raise ValueError('Source count or frozen teacher violated')
        losses.append(total/count)
        if epoch in EPOCHS:
            states[epoch]={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            with (args.out/('checkpoint_epoch%02d.pt'%epoch)).open('xb') as f:torch.save(dict(commit=args.commit,epoch=epoch,model=states[epoch]),f)
        print('SEMANTIC_SOURCE_EPOCH',epoch,'records',count,'loss',losses[-1],flush=True)
    received,scene=epoch_data(cache,20);candidates={}
    for epoch in EPOCHS:
        model.load_state_dict(states[epoch]);candidates[epoch]=evaluate(model,teacher,clean,received,scene,rx,valid,hold,args.device)
        print('SEMANTIC_SOURCE_CANDIDATES_EVALUATED',epoch,flush=True)
    selected=choose(candidates)
    if selected['kind']!='raw':
        with (args.out/'selected.pt').open('xb') as f:torch.save(dict(commit=args.commit,selected=selected,model=states[selected['epoch']]),f)
    report=dict(selected=selected,validation=candidates,train_ids=len(training),validation_ids=len(valid),heldout_ids=len(hold),losses=losses,source_only=True,DURA_used=False,F_frozen=True,training_L_s_labels_used=labeled_count,L_s_metadata_records_read=6300,U_s_labels_read=0,V_reads=0,test_reads=0,selection_targets='frozen clean Tri-SIG-TIME SSL confidence >= 0.95')
    with (args.out/'source_selection.json').open('x',encoding='utf8') as f:json.dump(report,f,indent=2,allow_nan=False);f.write('\n')
    print('SEMANTIC_SOURCE_SELECTION_FIXED',json.dumps(selected),flush=True)

if __name__=='__main__':main()
