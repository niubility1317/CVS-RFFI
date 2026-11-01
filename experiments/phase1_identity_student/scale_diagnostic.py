"""Fixed 2%-bounded scaling of source-trained identity student outputs."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from train_student import s,core,source_prior,BlindActionNet,IdentityStudent,prior_output,SOURCE,SHA,PRIOR_RUN,PRIOR_SHA,SCENES
from student_scale import scale_with_bound
RUN='20261102-phase1-identity-student-pilot-s2026092701-r01';NET_SHA='bf523ea96e8b0ba53f691401ee2d4975fa35e1fc44b8230c6598439656a3a22b'

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--commit',required=True);p.add_argument('--device',default='cuda:0');p.add_argument('--smoke',action='store_true');a=p.parse_args()
    if a.out.exists():raise FileExistsError('Immutable source-only output required')
    a.out.mkdir(parents=True);torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    ckpt=a.root/'runs'/RUN/'checkpoint_epoch03.pt';blind_ckpt=a.root/'runs'/SOURCE/'checkpoint_epoch10.pt';prior_file=a.root/'runs'/PRIOR_RUN/'source_prior.pt'
    for path,expected in ((ckpt,NET_SHA),(blind_ckpt,SHA),(prior_file,PRIOR_SHA)):
        if hashlib.sha256(path.read_bytes()).hexdigest()!=expected:raise ValueError('Exact fixed source checkpoint required')
    done=core.read(a.root/'runs'/RUN/'result.json')
    if done.get('status')!='COMPLETE_SOURCE_ONLY_IDENTITY_STUDENT_PILOT' or done.get('updates')!=768 or any(done.get(k)!=0 for k in ('U_s_truth_reads','V_reads','test_reads')):raise ValueError('Completed source training required')
    netstate=torch.load(ckpt,map_location='cpu',weights_only=False);blindstate=torch.load(blind_ckpt,map_location='cpu',weights_only=False);prior=torch.load(prior_file,map_location='cpu',weights_only=False);net=IdentityStudent().to(a.device);net.load_state_dict(netstate['model']);net.eval().requires_grad_(False);blind=BlindActionNet().to(a.device);blind.load_state_dict(blindstate['model']);blind.eval().requires_grad_(False);means=prior['means'].to(a.device);covs=prior['covariances'].to(a.device);priors=prior['priors'].to(a.device);eye=torch.eye(512,device=a.device)
    cache,clean,rx,training,pv,ph=s.source_contract(a.root);validation,heldout=s.new_splits(cache,rx,training,pv,ph);received,scene=s.epoch_data(cache,19);rng=np.random.default_rng(2026120355);prior_ids=np.sort(rng.choice(training,size=8192,replace=False));ids=training[:8] if a.smoke else np.concatenate([np.sort(rng.choice(heldout[scene[heldout]==v],size=64,replace=False)) for v in SCENES]);teacher,_=core.teacher(a.root,a.device);teacher.eval().requires_grad_(False);dura=source_prior.frozen_dura(a.root,a.device);out=[];tick=time.monotonic()
    with torch.no_grad():
        for start in range(0,len(ids),8):
            idx=ids[start:start+8];x=core.tensor(received[idx],a.device);y=core.tensor(clean[idx],a.device);conf,target=F.softmax(teacher.classify_features(teacher.features(y)),1).max(1);z,b=prior_output(blind,x,means,covs,priors,eye);student=net(x,z,b);variants=[z,dura(x)]+[scale_with_bound(z,student,k) for k in (1.,4.,8.)];den=y.square().sum((1,2)).clamp_min(1e-12);base_norm=z.flatten(1).norm(dim=1).clamp_min(1e-8);cols=[conf,target.float()]
            for signal in variants:
                pred=teacher.classify_features(teacher.features(signal)).argmax(1);iq=(signal-y).square().sum((1,2))/den;ratio=(signal-z).flatten(1).norm(dim=1)/base_norm;cols.extend((pred.float(),iq,ratio))
            batch=torch.stack(cols,1).cpu()
            if not torch.isfinite(batch).all() or any((batch[:,4+3*i]>.02001).any() for i in (2,3,4)):raise ValueError('Finite and bounded student corrections required')
            out.append(batch)
            if not a.smoke and (start+len(idx))%64==0:print('IDENTITY_SCALE_PROGRESS',start+len(idx),'seconds',round(time.monotonic()-tick,1),flush=True)
    if a.smoke:
        core.write(a.out/'result.json',dict(status='PASS',records=len(ids),source_only=True,TX_labels_read=0,U_s_truth_reads=0,V_reads=0,test_reads=0));print('IDENTITY_SCALE_SMOKE PASS',flush=True);return
    values=torch.cat(out);results={}
    for j,view in enumerate(SCENES):
        rows=values[j*64:(j+1)*64];valid=rows[:,0]>=.95;target=rows[:,1];base=rows[:,2]==target;cell=dict(count=len(rows),valid_count=int(valid.sum().item()))
        for i,name in enumerate(('parent','dura','scale_1','scale_4','scale_8')):
            correct=rows[:,2+3*i]==target;cell[name]=dict(correct=int((valid&correct).sum().item()),repairs_vs_parent=int((valid&correct&~base).sum().item()),damages_vs_parent=int((valid&~correct&base).sum().item()),mean_iq_nmse=float(rows[:,3+3*i].mean().item()),mean_relative_perturbation=float(rows[:,4+3*i].mean().item()))
        results[view]=cell
    result=dict(status='COMPLETE_SOURCE_ONLY_IDENTITY_SCALE',commit=a.commit,student_sha256=NET_SHA,scenes=results,probe_ids=len(ids),scales=[1.,4.,8.],relative_l2_cap=.02,source_only=True,TX_labels_read=0,U_s_truth_reads=0,V_reads=0,test_reads=0,caveat='Scale choices follow pilot observation; source IDs previously explored; F pseudo targets not truth; small IQ norm not physical validity.')
    core.write(a.out/'result.json',result);print('IDENTITY_SCALE_COMPLETE',json.dumps(result),flush=True)
if __name__=='__main__':main()
