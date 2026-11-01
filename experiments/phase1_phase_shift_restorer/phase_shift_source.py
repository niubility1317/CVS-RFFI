"""Source-only DURA-to-independent-network distillation and validation."""
import argparse,json,math,hashlib,sys
from pathlib import Path
DEPENDENCY=Path('/home/szu2070436088/2510044040/CV-SincNet/releases/20261102-phase1-teacher-student-s2026092701-r01-release')
if (DEPENDENCY/'release_commit.txt').read_text().strip()!='a90a5e381c19e94455bfafd9c67321833c764491':raise ValueError('Exact Phase1 source dependency required')
sys.path.insert(0,str(DEPENDENCY/'experiments/phase1_teacher_student'))
import numpy as np
import torch
from torch.nn import functional as F
from phase_shift_model import PhaseShiftRestorer
from phase_factor_model import AdaptivePhaseFactor
from phase1_blind_gate_model import BlindGate
from phase1_blind_gate_source import source_contract,epoch_data,new_splits,align,VIEWS
from phase1_complex_invariant_loss import complex_gain_invariant_error
from experiments.pamf_phase1_flow import experiment as core
from experiments.pamf_phase1_flow import anchor_source_smoke as prior

EPOCHS=(1,3,5);ALPHAS=(.5,1.);THRESHOLDS=(.8,.99)
GATE_RUN='20261102-phase1-blind-gate-s2026092701-r01';GATE_COMMIT='c905f3ab3d7f0d121e2c44c3b0bfb26188fa5825'
SEM_RUN='20261102-phase1-direct-semantic-s2026092701-r01';SEM_COMMIT='c957ef168f4ff6ef5671331668481b40d111228c'


def initial_models(root,device):
    fixed=torch.load(root/'runs'/GATE_RUN/'source/selected.pt',map_location='cpu',weights_only=False)
    if fixed['commit']!=GATE_COMMIT or fixed['selected']['kind']!='gated_direct_iq':raise ValueError('Exact frozen blind gate required')
    gate=BlindGate().to(device);gate.load_state_dict(fixed['gate'],strict=True);gate.eval().requires_grad_(False)
    phase_path=root/'runs/20261102-phase1-adaptive-phase-factor-s2026092701-r01/source/selected.pt'
    if hashlib.sha256(phase_path.read_bytes()).hexdigest()!='8e9fd8f97a57263be320b8dd57e4dd615fb015d5aab2d276d43864ec8fe8e508':raise ValueError('Exact frozen phase checkpoint hash required')
    saved=torch.load(phase_path,map_location='cpu',weights_only=False)
    if saved['commit']!='f8a2f526d67e21757c6972124403b6c3078e8d5b' or saved['selected']['epoch']!=3:raise ValueError('Exact source-only phase factor required')
    phase=AdaptivePhaseFactor();phase.load_state_dict(saved['model'],strict=True)
    model=PhaseShiftRestorer(phase).to(device)
    return model,gate


def objective(model,dura,teacher,x,y):
    with torch.no_grad():
        reference=dura(x);aligned,_=align(y,reference)
        d_logits=teacher.classify_features(teacher.features(reference));y_logits=teacher.classify_features(teacher.features(y))
        d_soft=F.softmax(d_logits/2.,1);y_soft=F.softmax(y_logits/2.,1)
    output=model(x);student=teacher.classify_features(teacher.features(output))
    norm=reference.square().mean((1,2)).clamp_min(1e-8)
    imitate=(output-reference).square().mean((1,2))/norm
    d_kl=F.kl_div(F.log_softmax(student/2.,dim=1),d_soft,reduction='none').sum(1)*4.
    y_kl=F.kl_div(F.log_softmax(student/2.,dim=1),y_soft,reduction='none').sum(1)*4.
    return (imitate+.5*d_kl+.2*y_kl+.2*complex_gain_invariant_error(output,aligned)).mean()


def smoke(model,gate,dura,teacher,clean,received,scene,training,device):
    index=training[scene[training]!='clean'][:16];x=core.tensor(received[index],device);y=core.tensor(clean[index],device)
    with torch.no_grad():active=gate(x).sigmoid()>.8
    if not active.any():raise ValueError('Active corrupted source smoke required')
    model.train();loss=objective(model,dura,teacher,x[active],y[active]);loss.backward()
    if not torch.isfinite(loss):raise ValueError('Finite source objective required')
    for p in (model.body.phase_head[2].weight,model.body.temporal_output.weight,model.body.output.weight):
        if p.grad is None or not torch.isfinite(p.grad).all() or not p.grad.abs().sum()>0:raise ValueError('Finite student gradients required')
    if any(p.requires_grad or p.grad is not None for p in list(teacher.parameters())+list(dura.parameters())+list(gate.parameters())+list(model.phase.parameters())):raise ValueError('All source teachers and gate must stay frozen')
    model.zero_grad(set_to_none=True);return dict(status='PASS',count=len(index),active=int(active.sum()),DURA_teacher_frozen=True,F_frozen=True,gate_frozen=True,TX_labels_read=0,U_s_labels_read=0,V_reads=0,test_reads=0)


def evaluate(model,gate,dura,teacher,clean,received,scene,rx,validation,heldout,device):
    ids=np.r_[validation,heldout];split=np.r_[np.ones(len(validation),dtype=np.int64),np.full(len(heldout),2,dtype=np.int64)];chunks=[]
    with torch.no_grad():
        for start in range(0,len(ids),32):
            idx=ids[start:start+32];x=core.tensor(received[idx],device);y=core.tensor(clean[idx],device);d=dura(x)
            prob=F.softmax(teacher.classify_features(teacher.features(y)),1);target=prob.argmax(1);conf=prob.max(1).values
            baseline=teacher.classify_features(teacher.features(d)).argmax(1);g=gate(x).sigmoid();den=y[:,:,64:-64].square().sum(1).mean(1).clamp_min(1e-12);_,de=align(d,y)
            full=model(x);raw=teacher.classify_features(teacher.features(x)).argmax(1);_,re=align(x,y);cols=[target.float(),conf,baseline.float(),g,de/den,raw.float(),re/den]
            for alpha in ALPHAS:
                anchor=model.anchor(x);z=anchor+alpha*(full-anchor);pred=teacher.classify_features(teacher.features(z)).argmax(1);_,ze=align(z,y);cols.extend((pred.float(),ze/den))
            chunks.append(torch.stack(cols,1).cpu())
    values=np.from_dlpack(torch.cat(chunks).contiguous());out={}
    for ai,alpha in enumerate(ALPHAS):
        for threshold in THRESHOLDS:
            cells={}
            for part in (1,2):
                for receiver in ((4,6) if part==1 else (8,)):
                    for view in VIEWS:
                        mask=(split==part)&(rx[ids]==receiver)&(scene[ids]==view);v=values[mask]
                        if not len(v):raise ValueError('All source scene cells required')
                        trusted=v[:,1]>=.95;active=v[:,3]>threshold
                        if not trusted.any():raise ValueError('Confident clean F source targets required')
                        pred=np.where(active,v[:,7+2*ai],v[:,5]);iq=np.where(active,v[:,8+2*ai],v[:,6])
                        cells[f'{part}:{receiver}:{view}']=dict(count=len(v),valid_count=int(trusted.sum()),active=int(active.sum()),baseline_correct=int((trusted&(v[:,2]==v[:,0])).sum()),candidate_correct=int((trusted&(pred==v[:,0])).sum()),baseline_iq=float(v[:,4].mean()),candidate_iq=float(iq.mean()))
            out[f'{alpha}:{threshold}']=cells
    return out


def choose(results):
    best=dict(kind='dura',validation_net=0)
    for epoch,combos in results.items():
        for key,cells in combos.items():
            alpha,threshold=map(float,key.split(':'));valid=[(k,v) for k,v in cells.items() if k.startswith('1:')]
            net=sum(v['candidate_correct']-v['baseline_correct'] for k,v in valid if not k.endswith(':clean'))
            cls=all(v['candidate_correct']>=v['baseline_correct']-max(1,math.ceil(.005*v['valid_count'])) for k,v in valid)
            iq=all(v['candidate_iq']<=(v['baseline_iq']+.001 if k.endswith(':clean') else 1.2*v['baseline_iq']+1e-6) for k,v in valid)
            if cls and iq and net>=10 and net>best['validation_net']:best=dict(kind='phase_shift_restorer',epoch=int(epoch),alpha=alpha,threshold=threshold,validation_net=net)
    return best


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--commit',required=True);ap.add_argument('--device',default='cuda:0');ap.add_argument('--smoke',action='store_true');args=ap.parse_args()
    if args.out.exists():raise FileExistsError('Immutable output root exists')
    args.out.mkdir(parents=True);torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.manual_seed(2026111554);rng=np.random.default_rng(2026111555)
    cache,clean,rx,training,prior_valid,prior_hold=source_contract(args.root);validation,heldout=new_splits(cache,rx,training,prior_valid,prior_hold)
    model,gate=initial_models(args.root,args.device);dura=prior.frozen_dura(args.root,args.device);teacher,_=core.teacher(args.root,args.device)
    received,scene=epoch_data(cache,1)
    if args.smoke:
        result=smoke(model,gate,dura,teacher,clean,received,scene,training,args.device)
        with (args.out/'result.json').open('x',encoding='utf8') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
        print('PHASE_SHIFT_STUDENT_SOURCE_SMOKE',json.dumps(result),flush=True);return
    opt=torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),lr=.0001,weight_decay=.001);losses=[];counts=[];states={}
    for epoch in range(1,6):
        received,scene=epoch_data(cache,epoch+10);order=rng.permutation(training);model.train();total=0.;count=0
        for start in range(0,len(order),32):
            idx=order[start:start+32];x=core.tensor(received[idx],args.device);y=core.tensor(clean[idx],args.device)
            with torch.no_grad():active=gate(x).sigmoid()>.8
            if not active.any():continue
            x=x[active];y=y[active];loss=objective(model,dura,teacher,x,y)
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite student source objective')
            opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step();total+=float(loss.detach())*len(x);count+=len(x)
        if count<1000 or any(p.grad is not None for p in list(gate.parameters())+list(dura.parameters())+list(teacher.parameters())):raise ValueError('Frozen teachers or source count violated')
        losses.append(total/count);counts.append(count)
        if epoch in EPOCHS:
            states[epoch]={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            with (args.out/('checkpoint_epoch%02d.pt'%epoch)).open('xb') as f:torch.save(dict(commit=args.commit,epoch=epoch,model=states[epoch]),f)
        print('PHASE_SHIFT_STUDENT_EPOCH',epoch,'active_records',count,'loss',losses[-1],flush=True)
    received,scene=epoch_data(cache,19);results={}
    for epoch in EPOCHS:
        model.load_state_dict(states[epoch]);model.eval().requires_grad_(False)
        results[epoch]=evaluate(model,gate,dura,teacher,clean,received,scene,rx,validation,heldout,args.device)
        print('PHASE_SHIFT_STUDENT_CANDIDATES_EVALUATED',epoch,flush=True)
    selected=choose(results)
    if selected['kind']!='dura':
        with (args.out/'selected.pt').open('xb') as f:torch.save(dict(commit=args.commit,selected=selected,model=states[selected['epoch']],gate=gate.state_dict()),f)
    result=dict(selected=selected,validation=results,train_ids=len(training),validation_ids=len(validation),heldout_ids=len(heldout),active_counts=counts,losses=losses,source_only=True,DURA_teacher_training_only=True,F_frozen=True,gate_frozen=True,TX_labels_read=0,U_s_labels_read=0,V_reads=0,test_reads=0,source_cache_epoch=19,source_caveat='source physical IDs were previously used in gate source evaluation; channel view is new')
    with (args.out/'source_selection.json').open('x',encoding='utf8') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    print('PHASE_SHIFT_STUDENT_SELECTION_FIXED',json.dumps(selected),flush=True)
if __name__=='__main__':main()
