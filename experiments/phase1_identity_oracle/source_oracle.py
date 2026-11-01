"""Legal source-L_s oracle: fixed blind parent then bounded label-guided probe."""
import argparse,hashlib,json,sys,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from bounded_oracle import oracle_refine
DEP=Path('/home/szu2070436088/2510044040/CV-SincNet/releases/20261102-phase1-bayes-residual-full-s2026092701-r01-release')
if (DEP/'release_commit.txt').read_text().strip()!='848c4b4c47b054f43c8303638e5aa549154ebfd4':raise ValueError('Exact source interface release required')
sys.path.insert(0,str(DEP/'experiments/phase1_bayes_residual_full'))
from bayes_residual_full import s,core,source_prior,BlindActionNet,action_matrix,posterior,SOURCE,SHA,PRIOR_RUN,PRIOR_SHA,SCENES
from experiments.pamf_phase1_flow import feature_conditioned_experiment as labels_source

def setup(root,device):
    cp=root/'runs'/SOURCE/'checkpoint_epoch10.pt';pp=root/'runs'/PRIOR_RUN/'source_prior.pt'
    if hashlib.sha256(cp.read_bytes()).hexdigest()!=SHA or hashlib.sha256(pp.read_bytes()).hexdigest()!=PRIOR_SHA:raise ValueError('Exact fixed blind model and source prior required')
    state=torch.load(cp,map_location='cpu',weights_only=False);prior=torch.load(pp,map_location='cpu',weights_only=False);blind=BlindActionNet().to(device);blind.load_state_dict(state['model']);blind.eval().requires_grad_(False);_,gate=s.initial_models(root,device);gate.eval().requires_grad_(False);teacher,contract=core.teacher(root,device);teacher.eval().requires_grad_(False);return blind,gate,teacher,contract,prior['means'].to(device),prior['covariances'].to(device),prior['priors'].to(device),torch.eye(512,device=device)

@torch.no_grad()
def parent_output(x,models):
    blind,gate,teacher,contract,means,covs,priors,eye=models;active=gate(x).sigmoid()>.8;z=x.clone()
    if active.any():
        v=x[active];h,lv=blind(v);A=action_matrix(h);N=eye[None]*lv.exp()[:,None,None];states,w=posterior(v.reshape(len(v),512),A,N,means,covs,priors);z[active]=(states*w.exp().unsqueeze(-1)).sum(1).reshape_as(v)
    return z,active

def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('smoke','predict','oracle'));p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--commit',required=True);p.add_argument('--device',default='cuda:0');a=p.parse_args();torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;cache,clean,rx,training,pv,ph=s.source_contract(a.root);validation,heldout=s.new_splits(cache,rx,training,pv,ph);roles=np.load(cache/'train/roles.npy',allow_pickle=False);physical=np.load(cache/'train/physical_ids.npy',allow_pickle=False);received,scene=s.epoch_data(cache,19);models=setup(a.root,a.device);teacher=models[2];classify=lambda z:teacher.classify_features(teacher.features(z))
    if a.mode=='smoke':
        if a.out.exists():raise FileExistsError('Immutable output required')
        a.out.mkdir(parents=True);idx=training[:4];x=core.tensor(received[idx],a.device);ground=core.tensor(clean[idx],a.device);z,_=parent_output(x,models)
        with torch.no_grad():pseudo=classify(ground).argmax(1)
        oracle=oracle_refine(classify,z,pseudo,.05,steps=2);ratio=(oracle-z).flatten(1).norm(dim=1)/z.flatten(1).norm(dim=1).clamp_min(1e-8)
        if not torch.isfinite(oracle).all() or (ratio>.05001).any() or any(q.grad is not None or q.requires_grad for q in teacher.parameters()):raise ValueError('Bounded input gradient and F freeze required')
        core.write(a.out/'smoke.json',dict(status='PASS',records=len(idx),source_only=True,source_labels_read=0,U_s_truth_reads=0,V_reads=0,test_reads=0));print('IDENTITY_ORACLE_SMOKE PASS',flush=True);return
    if a.mode=='predict':
        if a.out.exists():raise FileExistsError('Immutable output required')
        a.out.mkdir(parents=True);val=validation[(roles[validation]=='L_s')&(scene[validation]!='clean')];hold=heldout[(roles[heldout]=='L_s')&(scene[heldout]!='clean')];ids=np.r_[val,hold];part=np.r_[np.ones(len(val),dtype=np.int64),np.full(len(hold),2,dtype=np.int64)]
        if len(val)!=613 or len(hold)!=297 or np.any(roles[ids]!='L_s') or np.intersect1d(ids,training).size:raise ValueError('Exact source L_s cohorts required')
        dura=source_prior.frozen_dura(a.root,a.device);bases=[];rows=[];active=[];tick=time.monotonic()
        with torch.no_grad():
            for start in range(0,len(ids),8):
                idx=ids[start:start+8];x=core.tensor(received[idx],a.device);z,on=parent_output(x,models);raw=classify(x).argmax(1);d=classify(dura(x)).argmax(1);pred=classify(z).argmax(1);bases.append(z.cpu());rows.append(torch.stack((raw,d,pred),1).cpu());active.append(on.cpu())
                if (start+len(idx))%256==0:print('IDENTITY_ORACLE_BASE_PREDICT',start+len(idx),'seconds',round(time.monotonic()-tick,1),flush=True)
        path=a.out/'base_predictions.npz'
        with path.open('xb') as f:np.savez(f,indices=ids,physical_ids=physical[ids],rx=rx[ids],scene=scene[ids],part=part,base=np.from_dlpack(torch.cat(bases).contiguous()),predictions=np.from_dlpack(torch.cat(rows).contiguous()),active=np.from_dlpack(torch.cat(active).to(torch.uint8).contiguous()))
        core.write(a.out/'base_manifest.json',dict(status='SOURCE_BASE_FIXED',commit=a.commit,count=len(ids),base_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),source_only=True,source_labels_read=0,U_s_truth_reads=0,V_reads=0,test_reads=0));print('IDENTITY_ORACLE_BASE_FIXED',len(ids),flush=True);return
    if not a.out.exists():raise ValueError('Fixed source base required')
    manifest=core.read(a.out/'base_manifest.json');path=a.out/'base_predictions.npz'
    if manifest.get('status')!='SOURCE_BASE_FIXED' or manifest.get('commit')!=a.commit or manifest.get('source_labels_read')!=0 or hashlib.sha256(path.read_bytes()).hexdigest()!=manifest.get('base_sha256'):raise ValueError('Hash verified base predictions required before source labels')
    with np.load(path,allow_pickle=False) as f:data={k:f[k] for k in f.files}
    ids=data['indices'];expected=np.r_[validation[(roles[validation]=='L_s')&(scene[validation]!='clean')],heldout[(roles[heldout]=='L_s')&(scene[heldout]!='clean')]]
    if not np.array_equal(ids,expected) or not np.array_equal(data['physical_ids'],physical[ids]) or np.any(roles[ids]!='L_s'):raise ValueError('Exact legitimate L_s source role required')
    truth=labels_source.source_labels(a.root,cache,clean,roles,models[3])[ids]
    if np.any(truth<0) or np.any(truth>=6):raise ValueError('Valid L_s source labels required')
    output=[];tick=time.monotonic()
    for start in range(0,len(ids),16):
        idx=ids[start:start+16];base=core.tensor(data['base'][start:start+len(idx)],a.device);ground=core.tensor(clean[idx],a.device);target=torch.tensor(truth[start:start+len(idx)].tolist(),device=a.device,dtype=torch.long);den=ground.square().sum((1,2)).clamp_min(1e-12);base_norm=base.flatten(1).norm(dim=1).clamp_min(1e-8);cols=[]
        with torch.no_grad():
            pb=classify(base).argmax(1);bq=(base-ground).square().sum((1,2))/den
        if not np.array_equal(np.from_dlpack(pb.cpu().contiguous()),data['predictions'][start:start+len(idx),2]):raise ValueError('Fixed base predictions must reproduce')
        cols=[pb.float(),bq]
        for eps in (.02,.05,.10):
            z=oracle_refine(classify,base,target,eps,steps=16);ratio=(z-base).flatten(1).norm(dim=1)/base_norm
            if not torch.isfinite(z).all() or (ratio>eps+1e-5).any():raise ValueError('Bounded finite oracle output required')
            with torch.no_grad():pred=classify(z).argmax(1);iq=(z-ground).square().sum((1,2))/den
            cols.extend((pred.float(),iq,ratio))
        output.append(torch.stack(cols,1).cpu())
        if (start+len(idx))%256==0:print('IDENTITY_ORACLE_OPTIMIZED',start+len(idx),'seconds',round(time.monotonic()-tick,1),flush=True)
    values=np.from_dlpack(torch.cat(output).contiguous());opath=a.out/'oracle_predictions.npz'
    with opath.open('xb') as f:np.savez(f,indices=ids,outputs=values)
    result={}
    for part in (1,2):
        for view in SCENES:
            mask=(data['part']==part)&(data['scene']==view);y=truth[mask];v=values[mask];base=data['predictions'][mask];correct=(base[:,2]==y);cell=dict(count=len(y),raw_correct=int(np.count_nonzero(base[:,0]==y)),dura_correct=int(np.count_nonzero(base[:,1]==y)),parent_correct=int(np.count_nonzero(correct)),parent_mean_iq_nmse=float(np.mean(v[:,1])))
            if not len(y):raise ValueError('All L_s source scene cells required')
            for j,eps in enumerate((.02,.05,.10)):
                good=v[:,2+3*j]==y;repair=int(np.count_nonzero(good&~correct));harm=int(np.count_nonzero(~good&correct));cell[str(eps)]=dict(correct=int(np.count_nonzero(good)),repairs=repair,damages=harm,oracle_retain_correct=int(np.count_nonzero(correct))+repair,mean_iq_nmse=float(np.mean(v[:,3+3*j])),mean_relative_perturbation=float(np.mean(v[:,4+3*j])))
            result[f'{part}:{view}']=cell
    result=dict(status='SOURCE_LS_IDENTITY_ORACLE_COMPLETE',commit=a.commit,base_sha256=manifest['base_sha256'],oracle_predictions_sha256=hashlib.sha256(opath.read_bytes()).hexdigest(),source_counts=result,source_only=True,uses_source_true_label_for_optimization=True,oracle_not_deployable=True,U_s_truth_reads=0,V_reads=0,test_reads=0,caveat='Oracle uses true source L_s device labels and F gradients; perturbations may be adversarial rather than physically faithful; F trained on source records, no independent target claim.')
    core.write(a.out/'oracle_result.json',result);print('IDENTITY_ORACLE_COMPLETE',json.dumps(result),flush=True)
if __name__=='__main__':main()
