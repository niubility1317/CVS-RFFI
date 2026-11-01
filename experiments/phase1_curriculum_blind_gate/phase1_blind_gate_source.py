"""Source-only blind corruption gate with fresh receiver/day evaluation IDs."""
import argparse,json,math
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from phase1_blind_gate_model import BlindGate
from phase1_axial_spectral_mask import AxialSpectralMask
from phase1_axial_spectral_source import source_contract,epoch_data,align,VIEWS
from experiments.pamf_phase1_flow import experiment as core

EPOCHS=(1,3,5);ALPHAS=(.25,.5);THRESHOLDS=(.8,.9,.95,.99)
PARENT='20261102-phase1-direct-semantic-s2026092701-r01';PARENT_COMMIT='c957ef168f4ff6ef5671331668481b40d111228c'


def new_splits(cache,rx,training,valid_prior,hold_prior):
    roles=np.load(cache/'train/roles.npy',allow_pickle=False);day=np.load(cache/'train/day.npy',allow_pickle=False)
    rng=np.random.default_rng(2026110551)
    old=np.concatenate([np.sort(rng.choice(np.flatnonzero(roles==role),size=take,replace=False)) for role,take in (('L_s',1024),('U_s',3072))])
    mask=np.ones(len(roles),dtype=bool);mask[old]=False
    validation=np.flatnonzero(mask & np.isin(rx,[4,6]) & (day==3))
    heldout=np.flatnonzero(mask & (rx==8) & (day==3))
    if len(old)!=4096 or len(validation)!=7840 or len(heldout)!=3921 or np.intersect1d(np.r_[validation,heldout],np.r_[training,old]).size or not set(valid_prior).issubset(set(old)) or not set(hold_prior).issubset(set(old)):
        raise ValueError('Fresh physical IDs and receiver/day holdout required')
    return validation,heldout


def gate_loss(gate,x,scene):
    y=torch.tensor((scene!='clean').astype(np.float32).tolist(),device=x.device)
    theta=torch.rand(len(x),device=x.device)*6.283185307179586;c=theta.cos()[:,None];s=theta.sin()[:,None]
    rotated=torch.stack((x[:,0]*c-x[:,1]*s,x[:,0]*s+x[:,1]*c),1)
    logit=gate(rotated);loss=F.binary_cross_entropy_with_logits(logit,y,reduction='none');weight=torch.where(y==0.,4.,1.)
    return (loss*weight).mean()


def smoke(gate,received,scene,training,device):
    clean=training[scene[training]=='clean'][:8];impaired=training[scene[training]!='clean'][:8];ids=np.r_[clean,impaired]
    if len(ids)!=16:raise ValueError('Mixed clean and impaired source records required')
    x=core.tensor(received[ids],device);loss=gate_loss(gate,x,scene[ids]);loss.backward()
    if not torch.isfinite(loss) or any(p.grad is None or not torch.isfinite(p.grad).all() for p in gate.parameters()):raise ValueError('Finite detector gradients required')
    gate.zero_grad(set_to_none=True);return dict(status='PASS',records=16,clean=8,impaired=8,TX_labels_read=0,V_reads=0,test_reads=0,F_loaded=False,DURA_used=False)


def candidate_eval(gate,teacher,model,clean,received,scene,rx,validation,heldout,device):
    ids=np.r_[validation,heldout];split=np.r_[np.ones(len(validation),dtype=np.int64),np.full(len(heldout),2,dtype=np.int64)];chunks=[]
    with torch.no_grad():
        for start in range(0,len(ids),32):
            idx=ids[start:start+32];x=core.tensor(received[idx],device);y=core.tensor(clean[idx],device)
            clean_prob=F.softmax(teacher.classify_features(teacher.features(y)),1);target=clean_prob.argmax(1);conf=clean_prob.max(1).values
            raw=teacher.classify_features(teacher.features(x)).argmax(1);prob=gate(x).sigmoid();den=y[:,:,64:-64].square().sum(1).mean(1).clamp_min(1e-12);_,err=align(x,y)
            full=model(x);cols=[target.float(),conf,raw.float(),prob,err/den]
            for alpha in ALPHAS:
                z=x+alpha*(full-x);pred=teacher.classify_features(teacher.features(z)).argmax(1);_,e=align(z,y);cols.extend((pred.float(),e/den))
            chunks.append(torch.stack(cols,1).cpu())
    values=np.from_dlpack(torch.cat(chunks).contiguous());out={}
    for alpha_idx,alpha in enumerate(ALPHAS):
        for threshold in THRESHOLDS:
            cells={}
            for part in (1,2):
                for receiver in ((4,6) if part==1 else (8,)):
                    for view in VIEWS:
                        mask=(split==part)&(rx[ids]==receiver)&(scene[ids]==view);v=values[mask]
                        if not len(v):raise ValueError('All source scene cells required')
                        trusted=v[:,1]>=.95;active=v[:,3]>threshold
                        if not trusted.any():raise ValueError('Clean teacher pseudo targets required')
                        pred=np.where(active,v[:,5+2*alpha_idx],v[:,2]);iq=np.where(active,v[:,6+2*alpha_idx],v[:,4])
                        cells[f'{part}:{receiver}:{view}']=dict(count=len(v),valid_count=int(trusted.sum()),active=int(active.sum()),raw_correct=int((trusted&(v[:,2]==v[:,0])).sum()),candidate_correct=int((trusted&(pred==v[:,0])).sum()),raw_iq=float(v[:,4].mean()),candidate_iq=float(iq.mean()))
            out[f'{alpha}:{threshold}']=cells
    return out


def choose(results):
    best=dict(kind='raw',validation_net=0)
    for epoch,combos in results.items():
        for key,cells in combos.items():
            alpha,threshold=map(float,key.split(':'));valid=[(k,v) for k,v in cells.items() if k.startswith('1:')]
            net=sum(v['candidate_correct']-v['raw_correct'] for k,v in valid if not k.endswith(':clean'))
            cls=all(v['candidate_correct']>=v['raw_correct']-max(1,math.ceil(.005*v['valid_count'])) for k,v in valid)
            iq=all(v['candidate_iq']<=(.001 if k.endswith(':clean') else 1.1*v['raw_iq']+1e-6) for k,v in valid)
            if cls and iq and net>=10 and net>best['validation_net']:best=dict(kind='gated_direct_iq',epoch=int(epoch),alpha=alpha,threshold=threshold,validation_net=net)
    return best


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--commit',required=True);ap.add_argument('--device',default='cuda:0');ap.add_argument('--smoke',action='store_true');args=ap.parse_args()
    if args.out.exists():raise FileExistsError('Immutable output root exists')
    args.out.mkdir(parents=True);torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.manual_seed(2026111054);rng=np.random.default_rng(2026111055)
    cache,clean,rx,training,prior_valid,prior_hold=source_contract(args.root);validation,heldout=new_splits(cache,rx,training,prior_valid,prior_hold)
    gate=BlindGate().to(args.device);received,scene=epoch_data(cache,1)
    if args.smoke:
        result=smoke(gate,received,scene,training,args.device)
        with (args.out/'result.json').open('x',encoding='utf8') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
        print('BLIND_GATE_SOURCE_SMOKE',json.dumps(result),flush=True);return
    opt=torch.optim.AdamW(gate.parameters(),lr=.0003,weight_decay=.001);losses=[]
    for epoch in range(1,4):
        received,scene=epoch_data(cache,epoch);order=rng.permutation(training);gate.train();total=0.;count=0
        for start in range(0,len(order),128):
            idx=order[start:start+128];x=core.tensor(received[idx],args.device);loss=gate_loss(gate,x,scene[idx])
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite detector source loss')
            opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(gate.parameters(),1.,error_if_nonfinite=True);opt.step();total+=float(loss.detach())*len(x);count+=len(x)
        if count!=len(training):raise ValueError('Source train count mismatch')
        losses.append(total/count);print('BLIND_GATE_EPOCH',epoch,'records',count,'loss',losses[-1],flush=True)
    gate.eval().requires_grad_(False);teacher,_=core.teacher(args.root,args.device);parent=args.root/'runs'/PARENT/'source'
    done=core.read(args.root/'runs'/PARENT/'completion.json')
    if done['status']!='SOURCE_SELECTION_RETAINS_RAW' or done['target_predictions'] is not False:raise ValueError('Exact parent source-only completion required')
    received,scene=epoch_data(cache,20);results={};states={};model=AxialSpectralMask().to(args.device)
    for epoch in EPOCHS:
        state=torch.load(parent/('checkpoint_epoch%02d.pt'%epoch),map_location='cpu',weights_only=False)
        if state['commit']!=PARENT_COMMIT or state['epoch']!=epoch:raise ValueError('Exact source semantic checkpoint required')
        states[epoch]=state['model'];model.load_state_dict(state['model'],strict=True);model.eval().requires_grad_(False)
        results[epoch]=candidate_eval(gate,teacher,model,clean,received,scene,rx,validation,heldout,args.device)
        print('BLIND_GATE_CANDIDATES_EVALUATED',epoch,flush=True)
    selected=choose(results)
    if selected['kind']!='raw':
        with (args.out/'selected.pt').open('xb') as f:torch.save(dict(commit=args.commit,selected=selected,model=states[selected['epoch']],gate=gate.state_dict()),f)
    result=dict(selected=selected,validation=results,train_ids=len(training),validation_ids=len(validation),heldout_ids=len(heldout),prior_sample_ids_excluded=4096,losses=losses,source_only=True,DURA_used=False,F_frozen=True,TX_labels_read=0,U_s_labels_read=0,V_reads=0,test_reads=0,gate_inputs='received IQ only',parent_selection='post-hoc clean protection explored; fresh IDs for source selection')
    with (args.out/'source_selection.json').open('x',encoding='utf8') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    print('BLIND_GATE_SOURCE_SELECTION_FIXED',json.dumps(selected),flush=True)
if __name__=='__main__':main()
