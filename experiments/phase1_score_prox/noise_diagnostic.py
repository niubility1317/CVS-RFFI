"""Fixed source-only noise-adaptive plug-and-play restoration diagnostic."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from train_source import s,core,source_prior,GroundDenoiser,BlindActionNet,action_matrix,posterior,reconstruct,SOURCE,SHA,PRIOR_RUN,PRIOR_SHA,SCENES
from noise_prox import reconstruct_noise_adaptive
RUN='20261102-phase1-score-prox-pilot-s2026092701-r01';NET_SHA='5f23bc370ea445dbffe21e1a125f1066321a9df46a1c90e0b617648255323642'

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--commit',required=True);p.add_argument('--device',default='cuda:0');p.add_argument('--smoke',action='store_true');a=p.parse_args()
    if a.out.exists():raise FileExistsError('Immutable source output required')
    a.out.mkdir(parents=True);torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    ckpt=a.root/'runs'/RUN/'checkpoint_epoch03.pt';blind_ckpt=a.root/'runs'/SOURCE/'checkpoint_epoch10.pt';prior_file=a.root/'runs'/PRIOR_RUN/'source_prior.pt'
    for pth,expected in ((ckpt,NET_SHA),(blind_ckpt,SHA),(prior_file,PRIOR_SHA)):
        if hashlib.sha256(pth.read_bytes()).hexdigest()!=expected:raise ValueError('Exact frozen source artifact required')
    fitrec=core.read(a.root/'runs'/RUN/'result.json')
    if fitrec.get('status')!='COMPLETE_SOURCE_ONLY_SCORE_PROX_PILOT' or fitrec.get('updates')!=768 or any(fitrec.get(k)!=0 for k in ('TX_labels_read','U_s_truth_reads','V_reads','test_reads')):raise ValueError('Exact source training record required')
    netstate=torch.load(ckpt,map_location='cpu',weights_only=False);blindstate=torch.load(blind_ckpt,map_location='cpu',weights_only=False);prior=torch.load(prior_file,map_location='cpu',weights_only=False);net=GroundDenoiser().to(a.device);net.load_state_dict(netstate['model']);net.eval().requires_grad_(False);blind=BlindActionNet().to(a.device);blind.load_state_dict(blindstate['model']);blind.eval().requires_grad_(False);means=prior['means'].to(a.device);covs=prior['covariances'].to(a.device);priors=prior['priors'].to(a.device);eye=torch.eye(512,device=a.device)
    cache,clean,rx,training,pv,ph=s.source_contract(a.root);validation,heldout=s.new_splits(cache,rx,training,pv,ph);received,scene=s.epoch_data(cache,19);rng=np.random.default_rng(2026120355);prior_ids=np.sort(rng.choice(training,size=8192,replace=False));ids=training[:4] if a.smoke else np.concatenate([np.sort(rng.choice(heldout[scene[heldout]==v],size=64,replace=False)) for v in SCENES]);teacher,_=core.teacher(a.root,a.device);teacher.eval().requires_grad_(False);out=[];tick=time.monotonic()
    with torch.no_grad():
        for start in range(0,len(ids),8):
            idx=ids[start:start+8];x=core.tensor(received[idx],a.device);y=core.tensor(clean[idx],a.device);conf,target=F.softmax(teacher.classify_features(teacher.features(y)),1).max(1);h,lv=blind(x);A=action_matrix(h);variance=lv.exp();N=eye[None]*variance[:,None,None];states,weights=posterior(x.reshape(len(x),512),A,N,means,covs,priors);z=(states*weights.exp().unsqueeze(-1)).sum(1).reshape_as(x);variants=[z,reconstruct(net,A,x,z,10.)]+[reconstruct_noise_adaptive(net,A,x,z,variance,k) for k in (.25,1.,4.)];den=y.square().sum((1,2)).clamp_min(1e-12);cols=[conf,target.float(),lv]
            for signal in variants:
                pred=teacher.classify_features(teacher.features(signal)).argmax(1);iq=(signal-y).square().sum((1,2))/den;cols.extend((pred.float(),iq))
            batch=torch.stack(cols,1).cpu()
            if not torch.isfinite(batch).all():raise ValueError('Finite source diagnostic required')
            out.append(batch)
            if not a.smoke and (start+len(idx))%64==0:print('NOISE_ADAPTIVE_PROGRESS',start+len(idx),'seconds',round(time.monotonic()-tick,1),flush=True)
    if a.smoke:
        with (a.out/'result.json').open('x',encoding='utf8') as f:json.dump(dict(status='PASS',records=len(ids),noise_from_input_only=True,source_only=True,TX_labels_read=0,U_s_truth_reads=0,V_reads=0,test_reads=0),f,indent=2)
        print('NOISE_ADAPTIVE_SMOKE PASS',flush=True);return
    values=torch.cat(out);simvar=np.load(cache/'epochs/19/variance.npy',mmap_mode='r',allow_pickle=False)[ids];truevar=core.tensor(simvar,'cpu').clamp_min(1e-10);results={}
    for j,view in enumerate(SCENES):
        rows=values[j*64:(j+1)*64];mask=rows[:,0]>=.95;truth=truevar[j*64:(j+1)*64];cell=dict(count=len(rows),valid_count=int(mask.sum().item()),mean_abs_log_noise_error=float((rows[:,2]-truth.log()).abs().mean().item()),mean_predicted_variance=float(rows[:,2].exp().mean().item()),mean_simulator_variance=float(truth.mean().item()))
        for i,name in enumerate(('parent','fixed_10.0','adaptive_0.25','adaptive_1.0','adaptive_4.0')):cell[name]=dict(correct=int((mask&(rows[:,3+2*i]==rows[:,1])).sum().item()),mean_iq_nmse=float(rows[:,4+2*i].mean().item()))
        results[view]=cell
    result=dict(status='COMPLETE_SOURCE_ONLY_NOISE_ADAPTIVE_DIAGNOSTIC',commit=a.commit,model_sha256=NET_SHA,scenes=results,probe_ids=len(ids),multipliers=[.25,1.,4.],noise_from_input_only=True,simulator_variance_used_only_after_predictions=True,source_only=True,TX_labels_read=0,U_s_truth_reads=0,V_reads=0,test_reads=0,caveat='Source pseudo targets not truth, simulation variance diagnostic only; source IDs previously explored; no independent blind claim.')
    with (a.out/'result.json').open('x',encoding='utf8') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    print('NOISE_ADAPTIVE_COMPLETE',json.dumps(result),flush=True)
if __name__=='__main__':main()
