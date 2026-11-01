"""Source-only blind conditional posterior candidate diagnostics."""
import argparse,hashlib,json,sys,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from hypothesis_selector import choose_hypotheses
DEP=Path('/home/szu2070436088/2510044040/CV-SincNet/releases/20261102-phase1-bayes-residual-full-s2026092701-r01-release')
if (DEP/'release_commit.txt').read_text().strip()!='848c4b4c47b054f43c8303638e5aa549154ebfd4':raise ValueError('Exact source interface release required')
sys.path.insert(0,str(DEP/'experiments/phase1_bayes_residual_full'))
from bayes_residual_full import s,core,source_prior,BlindActionNet,action_matrix,posterior,SOURCE,SHA,PRIOR_RUN,PRIOR_SHA,SCENES

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--commit',required=True);p.add_argument('--device',default='cuda:0');p.add_argument('--smoke',action='store_true');a=p.parse_args()
    if a.out.exists():raise FileExistsError('Immutable source output required')
    a.out.mkdir(parents=True);torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    cp=a.root/'runs'/SOURCE/'checkpoint_epoch10.pt';pp=a.root/'runs'/PRIOR_RUN/'source_prior.pt'
    if hashlib.sha256(cp.read_bytes()).hexdigest()!=SHA or hashlib.sha256(pp.read_bytes()).hexdigest()!=PRIOR_SHA:raise ValueError('Exact fixed source blind model and prior required')
    state=torch.load(cp,map_location='cpu',weights_only=False);prior=torch.load(pp,map_location='cpu',weights_only=False);blind=BlindActionNet().to(a.device);blind.load_state_dict(state['model']);blind.eval().requires_grad_(False);means=prior['means'].to(a.device);covs=prior['covariances'].to(a.device);priors=prior['priors'].to(a.device);eye=torch.eye(512,device=a.device)
    cache,clean,rx,training,pv,ph=s.source_contract(a.root);validation,heldout=s.new_splits(cache,rx,training,pv,ph);received,scene=s.epoch_data(cache,19);rng=np.random.default_rng(2026120355);prior_ids=np.sort(rng.choice(training,size=8192,replace=False));ids=training[:8] if a.smoke else np.concatenate([np.sort(rng.choice(heldout[scene[heldout]==v],size=64,replace=False)) for v in SCENES]);teacher,_=core.teacher(a.root,a.device);teacher.eval().requires_grad_(False);dura=source_prior.frozen_dura(a.root,a.device);out=[];tick=time.monotonic()
    with torch.no_grad():
        for start in range(0,len(ids),8):
            idx=ids[start:start+8];x=core.tensor(received[idx],a.device);y=core.tensor(clean[idx],a.device);conf,target=F.softmax(teacher.classify_features(teacher.features(y)),1).max(1);h,lv=blind(x);A=action_matrix(h);states,lp=posterior(x.reshape(len(x),512),A,eye[None]*lv.exp()[:,None,None],means,covs,priors);parent=(states*lp.exp().unsqueeze(-1)).sum(1).reshape_as(x);hyp=states.reshape(len(x),6,2,256);flat=hyp.reshape(-1,2,256);hyp_logits=teacher.classify_features(teacher.features(flat)).reshape(len(x),6,6);choices=choose_hypotheses(lp,hyp_logits);parent_pred=teacher.classify_features(teacher.features(parent)).argmax(1);d=dura(x);dura_pred=teacher.classify_features(teacher.features(d)).argmax(1);ypred=hyp_logits.argmax(2);oracle=(ypred==target[:,None]).any(1)|(parent_pred==target);den=y.square().sum((1,2)).clamp_min(1e-12);spread=(hyp-parent[:,None]).square().sum((2,3)).mean(1)/parent.square().sum((1,2)).clamp_min(1e-12);cols=[conf,target.float(),parent_pred.float(),(parent-y).square().sum((1,2))/den,dura_pred.float(),(d-y).square().sum((1,2))/den,oracle.float(),spread]
            for name in ('map','self','product'):
                chosen=choices[name];signal=hyp[torch.arange(len(x),device=a.device),chosen];pred=ypred.gather(1,chosen[:,None]).squeeze(1);iq=(signal-y).square().sum((1,2))/den;cols.extend((pred.float(),iq,chosen.float(),(pred==chosen).float()))
            batch=torch.stack(cols,1).cpu()
            if not torch.isfinite(batch).all():raise ValueError('Finite fixed hypothesis outputs required')
            out.append(batch)
            if not a.smoke and (start+len(idx))%64==0:print('HYPOTHESIS_PROGRESS',start+len(idx),'seconds',round(time.monotonic()-tick,1),flush=True)
    if a.smoke:
        core.write(a.out/'result.json',dict(status='PASS',records=len(ids),source_only=True,TX_labels_read=0,U_s_truth_reads=0,V_reads=0,test_reads=0));print('HYPOTHESIS_SMOKE PASS',flush=True);return
    values=torch.cat(out);results={}
    for j,view in enumerate(SCENES):
        rows=values[j*64:(j+1)*64];valid=rows[:,0]>=.95;target=rows[:,1];parent=rows[:,2]==target;cell=dict(count=len(rows),valid_count=int(valid.sum().item()),parent_correct=int((valid&parent).sum().item()),parent_mean_iq_nmse=float(rows[:,3].mean().item()),dura_correct=int((valid&(rows[:,4]==target)).sum().item()),dura_mean_iq_nmse=float(rows[:,5].mean().item()),oracle_union_correct=int((valid&(rows[:,6]>0)).sum().item()),oracle_extra=int((valid&(rows[:,6]>0)&~parent).sum().item()),mean_hypothesis_spread=float(rows[:,7].mean().item()))
        for i,name in enumerate(('map','self','product')):
            pred=rows[:,8+4*i];good=pred==target;chosen=rows[:,10+4*i];cell[name]=dict(correct=int((valid&good).sum().item()),repairs=int((valid&good&~parent).sum().item()),damages=int((valid&~good&parent).sum().item()),mean_iq_nmse=float(rows[:,9+4*i].mean().item()),chosen_class_matches_clean_F=int((valid&(chosen==target)).sum().item()),classifier_agrees_chosen=int((valid&(rows[:,11+4*i]>0)).sum().item()))
        results[view]=cell
    result=dict(status='COMPLETE_SOURCE_ONLY_POSTERIOR_HYPOTHESES',commit=a.commit,scenes=results,probe_ids=len(ids),fixed_blind_sha256=SHA,fixed_prior_sha256=PRIOR_SHA,input_only_selectors=['map','self','product'],oracle_union_uses_clean_F_pseudo=True,oracle_not_deployable=True,source_only=True,TX_labels_read=0,U_s_truth_reads=0,V_reads=0,test_reads=0,caveat='Source physical IDs previously explored; clean F pseudo targets not true accuracy. Oracle needs unavailable clean F target, no universal bound; class-conditioned priors may hallucinate identity.')
    core.write(a.out/'result.json',result);print('HYPOTHESIS_COMPLETE',json.dumps(result),flush=True)
if __name__=='__main__':main()
