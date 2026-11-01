"""Source-only phase-factor feasibility experiment; no target data or labels."""
import argparse,json,sys
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from phase_factor_model import AdaptivePhaseFactor

GATE_RUN='20261102-phase1-blind-gate-s2026092701-r01'
GATE_COMMIT='c905f3ab3d7f0d121e2c44c3b0bfb26188fa5825'
PARENT_RUN='20261102-phase1-teacher-student-s2026092701-r01'
PARENT_COMMIT='a90a5e381c19e94455bfafd9c67321833c764491'


def target_gain(x,reference):
    """Complex gain minimizing central-sample error over delays -64..64."""
    padded=F.pad(x,(64,64),mode='reflect')
    z=torch.complex(padded[:,0],padded[:,1]).unfold(-1,256,1)[:,:,64:-64]
    target=torch.complex(reference[:,0,64:-64],reference[:,1,64:-64])[:,None,:]
    gain=(z.conj()*target).sum(-1)/z.abs().square().sum(-1).clamp_min(1e-20)
    err=(gain[:,:,None]*z-target).abs().square().mean(-1)
    index=err.argmin(1);chosen=gain.gather(1,index[:,None]).squeeze(1)
    valid=chosen.abs()>1e-6
    return chosen/chosen.abs().clamp_min(1e-8),valid


def source_deps(root):
    release=root/'releases'/(PARENT_RUN+'-release')
    if (release/'release_commit.txt').read_text().strip()!=PARENT_COMMIT:raise ValueError('Exact source dependency release required')
    sys.path.insert(0,str(release/'experiments/phase1_teacher_student'))
    from phase1_teacher_student_source import source_contract,epoch_data,new_splits,VIEWS
    from phase1_blind_gate_model import BlindGate
    from experiments.pamf_phase1_flow import experiment as core
    from experiments.pamf_phase1_flow import anchor_source_smoke as prior
    return source_contract,epoch_data,new_splits,VIEWS,BlindGate,core,prior


def initial_gate(root,device,cls):
    saved=torch.load(root/'runs'/GATE_RUN/'source/selected.pt',map_location='cpu',weights_only=False)
    if saved['commit']!=GATE_COMMIT or saved['selected']['kind']!='gated_direct_iq':raise ValueError('Exact frozen gate required')
    gate=cls().to(device);gate.load_state_dict(saved['gate'],strict=True);return gate.eval().requires_grad_(False)


def batch_targets(x,dura,gate):
    with torch.no_grad():
        active=gate(x).sigmoid()>.8
        if not active.any():return active,None
        target,valid=target_gain(x[active],dura(x[active]))
        if not valid.all():
            ids=active.nonzero().squeeze(1);active[ids[~valid]]=False;target=target[valid]
    return active,target


def loss_for(model,x,target):
    return (1.-(model(x)*target.conj()).real).mean()


def evaluate(model,dura,gate,core,received,scene,rx,validation,heldout,views,device):
    ids=np.r_[validation,heldout];split=np.r_[np.ones(len(validation),dtype=np.int64),np.full(len(heldout),2,dtype=np.int64)];rows=[]
    with torch.no_grad():
        for start in range(0,len(ids),32):
            idx=ids[start:start+32];x=core.tensor(received[idx],device);active,target=batch_targets(x,dura,gate)
            vals=torch.zeros((len(idx),3),device=device);vals[:,0]=active.float()
            if active.any():
                pred=model(x[active]);vals[active,1]=1.-target.real;vals[active,2]=1.-(pred*target.conj()).real
            rows.append(vals.cpu())
    values=np.from_dlpack(torch.cat(rows).contiguous());cells={}
    for part,receivers in ((1,(4,6)),(2,(8,))):
        for receiver in receivers:
            for view in views:
                if view=='clean':continue
                mask=(split==part)&(rx[ids]==receiver)&(scene[ids]==view)&(values[:,0]>.5)
                if not mask.any():raise ValueError('Missing active source validation scene')
                v=values[mask];cells[f'{part}:{receiver}:{view}']=dict(count=len(v),baseline=float(v[:,1].mean()),candidate=float(v[:,2].mean()))
    return cells


def choose(results):
    selected=dict(kind='identity',validation_loss=None)
    for epoch,cells in results.items():
        vals=[v for k,v in cells.items() if k.startswith('1:')];n=sum(v['count'] for v in vals)
        base=sum(v['count']*v['baseline'] for v in vals)/n;cur=sum(v['count']*v['candidate'] for v in vals)/n
        guard=all(v['candidate']<=v['baseline']+.05 for v in vals)
        if guard and cur<=.95*base and (selected['validation_loss'] is None or cur<selected['validation_loss']):selected=dict(kind='adaptive_phase_factor',epoch=int(epoch),validation_loss=cur,baseline_loss=base)
    return selected


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--commit',required=True);ap.add_argument('--device',default='cuda:0');ap.add_argument('--smoke',action='store_true');a=ap.parse_args()
    if a.out.exists():raise FileExistsError('Immutable output already exists')
    a.out.mkdir(parents=True);torch.set_num_threads(2);torch.manual_seed(2026111454);rng=np.random.default_rng(2026111455)
    source_contract,epoch_data,new_splits,views,BlindGate,core,prior=source_deps(a.root)
    cache,clean,rx,training,oldval,oldhold=source_contract(a.root);validation,heldout=new_splits(cache,rx,training,oldval,oldhold)
    gate=initial_gate(a.root,a.device,BlindGate);dura=prior.frozen_dura(a.root,a.device).eval().requires_grad_(False);model=AdaptivePhaseFactor().to(a.device)
    received,scene=epoch_data(cache,1)
    if a.smoke:
        ids=training[scene[training]!='clean'][:32];x=core.tensor(received[ids],a.device);active,target=batch_targets(x,dura,gate)
        if int(active.sum())<1:raise ValueError('Active source smoke examples required')
        loss=loss_for(model,x[active],target);loss.backward()
        if not torch.isfinite(loss) or any(p.grad is None or not torch.isfinite(p.grad).all() for p in model.parameters()):raise ValueError('Invalid phase-factor gradient')
        if any(p.requires_grad or p.grad is not None for p in list(dura.parameters())+list(gate.parameters())):raise ValueError('Teacher and gate must be frozen')
        result=dict(status='PASS',records=len(ids),active=int(active.sum()),loss=float(loss),DURA_teacher_frozen=True,gate_frozen=True,F_loaded=False,TX_labels_read=0,U_s_truth_reads=0,V_reads=0,test_reads=0)
        (a.out/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf8');print('PHASE_FACTOR_SMOKE',json.dumps(result),flush=True);return
    optimizer=torch.optim.AdamW(model.parameters(),lr=.0003,weight_decay=.001);states={};losses=[];counts=[]
    for epoch in range(1,4):
        received,scene=epoch_data(cache,epoch);order=rng.permutation(training);model.train();total=0.;count=0
        for start in range(0,len(order),32):
            idx=order[start:start+32];x=core.tensor(received[idx],a.device);active,target=batch_targets(x,dura,gate)
            if not active.any():continue
            loss=loss_for(model,x[active],target)
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite phase loss')
            optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step();total+=float(loss.detach())*int(active.sum());count+=int(active.sum())
        if count<1000:raise ValueError('Insufficient active source records')
        counts.append(count);losses.append(total/count);states[epoch]={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        with (a.out/('checkpoint_epoch%02d.pt'%epoch)).open('xb') as f:torch.save(dict(commit=a.commit,epoch=epoch,model=states[epoch]),f)
        print('PHASE_FACTOR_EPOCH',epoch,'active',count,'loss',losses[-1],flush=True)
    received,scene=epoch_data(cache,16);results={}
    for epoch in (1,2,3):
        model.load_state_dict(states[epoch]);model.eval().requires_grad_(False)
        results[epoch]=evaluate(model,dura,gate,core,received,scene,rx,validation,heldout,views,a.device);print('PHASE_FACTOR_EVALUATED',epoch,flush=True)
    selected=choose(results)
    if selected['kind']!='identity':
        with (a.out/'selected.pt').open('xb') as f:torch.save(dict(commit=a.commit,selected=selected,model=states[selected['epoch']]),f)
    result=dict(selected=selected,validation=results,train_ids=len(training),validation_ids=len(validation),heldout_ids=len(heldout),active_counts=counts,losses=losses,source_cache_epoch=16,source_only=True,F_loaded=False,DURA_source_teacher_only=True,gate_frozen=True,TX_labels_read=0,U_s_truth_reads=0,V_reads=0,test_reads=0,scope='phase of best delay/complex gain to source teacher, proxy only; source validation physical IDs previously examined under other channel views')
    with (a.out/'source_selection.json').open('x',encoding='utf8') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    print('PHASE_FACTOR_SELECTION',json.dumps(selected),flush=True)
if __name__=='__main__':main()
