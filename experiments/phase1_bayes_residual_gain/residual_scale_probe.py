"""Source-only frozen residual scaling diagnostic; no target truth."""
import argparse,hashlib,json,sys,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F

DEP=Path('/home/szu2070436088/2510044040/CV-SincNet/releases/20261102-phase1-bayes-residual-pilot-s2026092701-r01-release')
if (DEP/'release_commit.txt').read_text().strip()!='7580869fa8d5d048cbb5463b055b37a757debeb5':raise ValueError('Exact residual source release required')
sys.path.insert(0,str(DEP/'experiments/phase1_bayes_residual'))
from bayes_residual_pilot import ResidualCorrector,BlindActionNet,prior_output,s,core,source_prior,SOURCE,SHA,PRIOR_RUN,PRIOR_SHA
RESIDUAL_RUN='20261102-phase1-bayes-residual-pilot-s2026092701-r01';RESIDUAL_SHA='d42175ea409bb3dbe159a2009146e0e50fd863c087c4cd57d61199d2654f76f9'
VIEWS=('clean','practical_high','practical_mid','practical_low_suburban','practical_high_urban','practical_mid_urban','practical_low_urban');SCALES=(0.,1.,4.,16.)

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--commit',required=True);p.add_argument('--device',default='cuda:0');p.add_argument('--smoke',action='store_true');a=p.parse_args()
 if a.out.exists():raise FileExistsError('Immutable output required')
 a.out.mkdir(parents=True);torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 parent=a.root/'runs'/SOURCE/'checkpoint_epoch10.pt';prior_path=a.root/'runs'/PRIOR_RUN/'source_prior.pt';residual_path=a.root/'runs'/RESIDUAL_RUN/'checkpoint.pt'
 if any(hashlib.sha256(p.read_bytes()).hexdigest()!=h for p,h in ((parent,SHA),(prior_path,PRIOR_SHA),(residual_path,RESIDUAL_SHA))):raise ValueError('Exact frozen source states required')
 parent_state=torch.load(parent,map_location='cpu',weights_only=False);prior=torch.load(prior_path,map_location='cpu',weights_only=False);saved=torch.load(residual_path,map_location='cpu',weights_only=False)
 if saved['commit']!='7580869fa8d5d048cbb5463b055b37a757debeb5' or saved['updates']!=512 or saved['parent_sha256']!=SHA or saved['prior_sha256']!=PRIOR_SHA:raise ValueError('Exact residual checkpoint provenance required')
 cache,clean,rx,training,pv,ph=s.source_contract(a.root);validation,heldout=s.new_splits(cache,rx,training,pv,ph);r_prior=np.random.default_rng(2026120355);prior_ids=np.sort(r_prior.choice(training,size=8192,replace=False));pool=np.setdiff1d(training,prior_ids,assume_unique=True);r_fit=np.random.default_rng(2026120655);fit=np.sort(r_fit.choice(pool,size=4096,replace=False));free=np.setdiff1d(pool,fit,assume_unique=True);received,scene=s.epoch_data(cache,19);rng=np.random.default_rng(2026120755);per=4 if a.smoke else 64;ids=np.concatenate([np.sort(rng.choice(free[scene[free]==v],size=per,replace=False)) for v in VIEWS])
 if np.intersect1d(ids,np.r_[fit,prior_ids,validation,heldout]).size:raise ValueError('Distinct source diagnostic physical IDs required')
 model=BlindActionNet().to(a.device);model.load_state_dict(parent_state['model'],strict=True);model.eval().requires_grad_(False);net=ResidualCorrector().to(a.device);net.load_state_dict(saved['model'],strict=True);net.eval().requires_grad_(False);teacher,_=core.teacher(a.root,a.device);dura=source_prior.frozen_dura(a.root,a.device);means=prior['means'].to(a.device);covs=prior['covariances'].to(a.device);priors=prior['priors'].to(a.device);eye=torch.eye(512,device=a.device);out=[];tick=time.monotonic()
 with torch.no_grad():
  for start in range(0,len(ids),8):
   idx=ids[start:start+8];x=core.tensor(received[idx],a.device);y=core.tensor(clean[idx],a.device);prob=F.softmax(teacher.classify_features(teacher.features(y)),1);conf,target=prob.max(1);z,b=prior_output(model,x,means,covs,priors,eye);fixed=net(z,b);delta=fixed-z;den=z.square().sum((1,2)).clamp_min(1e-12);change=(delta.square().sum((1,2))/den).sqrt();d=dura(x);dp=teacher.classify_features(teacher.features(d)).argmax(1);cols=[conf,target.float(),dp.float(),change];yd=y.square().sum((1,2)).clamp_min(1e-12)
   for scale in SCALES:
    candidate=z+scale*delta;pred=teacher.classify_features(teacher.features(candidate)).argmax(1);iq=(candidate-y).square().sum((1,2))/yd;cols.extend((pred.float(),iq))
   out.append(torch.stack(cols,1).cpu())
   if (start+len(idx))%128==0:print('RESIDUAL_SCALE_PROGRESS',start+len(idx),'seconds',round(time.monotonic()-tick,1),flush=True)
 values=torch.cat(out);result={}
 for j,view in enumerate(VIEWS):
  rows=values[j*per:(j+1)*per];mask=rows[:,0]>=.95;cell=dict(count=len(rows),valid_count=int(mask.sum().item()),baseline_dura_correct=int((mask&(rows[:,2]==rows[:,1])).sum().item()),mean_relative_correction=float(rows[:,3].mean().item()),scales={})
  for i,scale in enumerate(SCALES):cell['scales'][str(scale)]=dict(correct=int((mask&(rows[:,4+2*i]==rows[:,1])).sum().item()),mean_iq_nmse=float(rows[:,5+2*i].mean().item()))
  result[view]=cell
 result=dict(status='SOURCE_RESIDUAL_SCALE_SMOKE' if a.smoke else 'SOURCE_RESIDUAL_SCALE_COMPLETE',commit=a.commit,scenes=result,source_ids=len(ids),fit_excluded=len(fit),prior_excluded=len(prior_ids),scales=list(SCALES),residual_sha256=RESIDUAL_SHA,source_only=True,TX_labels_read=0,U_s_truth_reads=0,V_reads=0,test_reads=0,caveat='Fixed residual model previously used source L_s training labels; source IDs and other views previously exposed; scale diagnostic is not a deployable selection or Phase1 test.')
 with (a.out/'result.json').open('x',encoding='utf8') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
 print('RESIDUAL_SCALE_COMPLETE',json.dumps(result),flush=True)
if __name__=='__main__':main()
