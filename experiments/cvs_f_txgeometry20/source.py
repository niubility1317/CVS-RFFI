"""Local source cache only, fixed20 from scratch, default classifier jointly trained."""
import json,math,os,random,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from . import design as d,model as api
from .mechanism import objective
from .augmentation import Satellite
from tools.local_phase1_data import SourceRoleDataset
from experiments.cvs_equivariant_identity.precision import numerical_context,FULL_FP32_POLICY
from experiments.cvs_rdrop_sinc_units.optimizer import build as optimizer_build,clip_physical
from cvsrffi.schedule import build_aug_base_cfg,make_augmentor,configure_augmentor_for_epoch

def seed(s):
 random.seed(s);np.random.seed(s);torch.manual_seed(s);torch.cuda.manual_seed_all(s)

def validate_source(profile):
 root=Path(profile['dataset']['source_cache']);m=d.read(root/'manifest.json')
 if m['status']!='READY' or m['native_role_ids']!='EXACT_MATCH' or m['class_id_to_tx']!=list(d.CLASSES) or m['source_receivers']!=[1,3,4,6,8] or m['source_days']!=[1,2,3] or m['U_TX_labels_exported'] or m['target_loaders_constructed'] or m['num_domains']!=15:raise ValueError('Existing legal source handles differ')
 if {k:v['count'] for k,v in m['roles'].items()}!={'L_s':6300,'U_s':56700,'V':27000}:raise ValueError('Source role counts differ')
 return root,m

@torch.no_grad()
def validate(model,loader,device):
 model.eval();cm=np.zeros((6,6),dtype=np.int64);rxcm={r:np.zeros((6,6),dtype=np.int64) for r in (1,3,4,6,8)};daycm={r:np.zeros((6,6),dtype=np.int64) for r in (1,2,3)};sums=np.zeros((6,160),dtype=np.float64);ce_sum=0.
 versions=[p._version for p in model.parameters()];buffers={n:v.clone() for n,v in model.named_buffers()}
 for x,y,domain,meta in loader:
  out=model.forward_identity_only(x.to(device));logits=out['tx_logits'];pred=logits.argmax(1).cpu().numpy();truth=y.numpy();z=F.normalize(out['id_feat_joint'],dim=1).cpu().numpy()
  if not torch.isfinite(logits).all() or not np.isfinite(z).all():raise FloatingPointError('Nonfinite frozen source outputs')
  np.add.at(cm,(truth,pred),1);np.add.at(sums,truth,z);ce_sum+=float(F.cross_entropy(logits,y.to(device),reduction='sum'))
  for r,mat in rxcm.items():
   mask=meta['rx_i'].numpy()==r;np.add.at(mat,(truth[mask],pred[mask]),1)
  for day,mat in daycm.items():
   mask=meta['day_i'].numpy()==day;np.add.at(mat,(truth[mask],pred[mask]),1)
 if cm.sum()!=27000 or not np.all(cm.sum(1)==4500) or any(not np.all(v.sum(1)==900) for v in rxcm.values()):raise ValueError('Source V full role coverage differs')
 if versions!=[p._version for p in model.parameters()] or any(not torch.equal(v,dict(model.named_buffers())[n]) for n,v in buffers.items()):raise ValueError('Validation updated persistent model state')
 centers=sums/4500;norms=np.linalg.norm(centers,axis=1);unit=centers/np.maximum(norms[:,None],1e-12);cos=unit@unit.T
 return dict(count=27000,accuracy=float(cm.trace()/27000),CE=ce_sum/27000,recall=(cm.diagonal()/4500).tolist(),worst_RX=min(float(v.trace()/v.sum()) for v in rxcm.values()),confusion=cm.tolist(),receiver_confusion={str(r):v.tolist() for r,v in rxcm.items()},day_confusion={str(r):v.tolist() for r,v in daycm.items()},mean_within_TX_pair_cosine=((4500*norms**2-1)/4499).tolist(),class_centroid_cosine=cos.tolist(),feature_diagnostic_fitting=False)

def train(c,profile,base,commit):
 d.validate(c);root,manifest=validate_source(profile);out=Path(base)/c['row_id']/'source'
 out.mkdir(parents=True,exist_ok=False);device='cuda:0';torch.set_num_threads(2);seed(c['model_seed']);start=time.perf_counter()
 loader=torch.utils.data.DataLoader(SourceRoleDataset(root,'L_s'),batch_size=128,shuffle=True,drop_last=True,num_workers=0,generator=torch.Generator().manual_seed(c['model_seed']))
 val=torch.utils.data.DataLoader(SourceRoleDataset(root,'V'),batch_size=256,shuffle=False,num_workers=0)
 if len(loader)!=49 or len(val.dataset)!=27000:raise ValueError('Local fixed-role sampler differs')
 a=api.args(c);a.wisig_pkl=profile['dataset']['manysig_pkl'];a.output_dir=str(out)
 aug_cfg=build_aug_base_cfg(a);augment=make_augmentor(aug_cfg);satellite=Satellite(a);steps=0;sat_seen=applied=0
 with numerical_context(FULL_FP32_POLICY):
  model=api.build(c,device);initial={n:v.detach().cpu().clone() for n,v in model.state_dict().items()}
  torch.save(dict(model=initial,config=c,scratch_only=True,ancestors=[]),out/'initial_smoke.pt')
  model.load_state_dict(torch.load(out/'initial_smoke.pt',weights_only=False,map_location=device)['model'],strict=True);model.eval()
  with torch.no_grad():z=model.forward_identity_only(torch.zeros(2,2,256,device=device))
  if z['tx_logits'].shape!=(2,6) or z['id_feat_joint'].shape!=(2,160) or not torch.isfinite(z['tx_logits']).all():raise ValueError('Real own-checkpoint no-query smoke failed')
  d.write(out/'smoke.json',dict(status='PASS',real_checkpoint=True,checkpoint='initial_smoke.pt',query_reads=0,truth_reads=0,default_head_shape=[6,160]))
  print('SMOKE_PASS '+c['row_id'],flush=True)
  optimizer=optimizer_build(model,model.parameters(),'CENTER_UNITS',lr=c['lr'],weight_decay=c['weight_decay'],eps=1e-8,foreach=False)
  d.write(out/'resolved_config.json',dict(config=c,commit=commit,pid=os.getpid(),cwd=os.getcwd(),python=__import__('sys').executable,torch=torch.__version__,gpu=torch.cuda.get_device_name(0),physical_gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),parameters=sum(p.numel() for p in model.parameters()),source_contract=profile['dataset']['source_contract'],source_cache=str(root),source_role_match='EXACT_MATCH',target_access=False,checkpoint_sources=[],native_model=api.original.config(next(r for r in api.original.training_rows() if r['arm']=='CROSS_RX' and r['model_seed']==c['model_seed'])),sampler='shuffle L6300, drop_last128, cycle until222updates/epoch; independent seeded generator',augmentation_reuse='same post-ordinary concat clean/sat IQ for both forwards',supervised_only=True))
  iterator=iter(loader);epoch_records=[]
  torch.cuda.reset_peak_memory_stats()
  with (out/'epochs.jsonl').open('x',encoding='utf-8') as log:
   for epoch in range(1,d.EPOCHS+1):
    model.train();epoch_start=time.perf_counter();lr=c['lr_min']+(c['lr']-c['lr_min'])*.5*(1+math.cos(math.pi*(epoch-1)/d.EPOCHS));optimizer.param_groups[0]['lr']=lr
    aug_state=configure_augmentor_for_epoch(augment,aug_cfg,epoch,a);accum={};nsteps=0
    for bi in range(d.STEPS):
     try:x,y,domain,meta=next(iterator)
     except StopIteration:iterator=iter(loader);x,y,domain,meta=next(iterator)
     if (y<0).any() or y.max()>=6:raise ValueError('Only original L_s visible TX allowed')
     x=x.to(device);y=y.to(device);rx=meta['rx_i'].to(device)
     x=augment(x,labels=y,no_pa=False).detach()
     if not torch.isfinite(x).all():raise FloatingPointError('Nonfinite ordinary source augmentation')
     sx,si=satellite(x,epoch,bi);joined=torch.cat((x,sx),0);sat_seen+=128;applied+=128*int(si['applied'])
     optimizer.zero_grad(set_to_none=True);total,metrics=objective(model,joined,y,rx,epoch,c)
     if not torch.isfinite(total):raise FloatingPointError('Nonfinite training loss: preserve, no score-based retry')
     total.backward();norm=clip_physical(optimizer,model.parameters(),5.)
     if not torch.isfinite(norm):raise FloatingPointError('Nonfinite source gradient')
     optimizer.step();steps+=1;nsteps+=1
     for k,v in dict(total_loss=float(total.detach()),**metrics).items():
      if isinstance(v,(int,float)):accum[k]=accum.get(k,0.)+v
    metrics=validate(model,val,device);record=dict(epoch=epoch,optimizer_steps=steps,lr=lr,train={k:v/nsteps for k,v in accum.items()},sourceV=metrics,epoch_seconds=time.perf_counter()-epoch_start,satellite_examples=sat_seen,satellite_applied_examples=applied,augmentation=aug_state)
    log.write(json.dumps(record,allow_nan=False)+'\n');log.flush();epoch_records.append(record)
    d.write(out/f'epoch{epoch:02d}.json',record)
    print(json.dumps(dict(row=c['row_id'],epoch=epoch,steps=steps,V=metrics['accuracy'],recall=metrics['recall'],seconds=record['epoch_seconds'])),flush=True)
   if steps!=d.UPDATES or optimizer._sinc_unit_steps!=steps or optimizer._sinc_physical_clip_calls!=steps:raise ValueError('Physical optimizer/update budget incomplete')
   state=model.state_dict();W='id_backbone.encoder.core.id_backbone.cls_head.weight'
   if torch.equal(initial[W],state[W].cpu()):raise ValueError('Default classifier did not jointly update')
   torch.save(dict(model=state,config=c,commit=commit,epoch=20,optimizer_steps=steps,checkpoint_selection='fixed_final20',source_roles='EXACT_MATCH',scratch_only=True,ancestors=[],target_access=False,default_head_joint_training=True,head_fit_updates=0),out/'final.pt')
   done=dict(status='SOURCE_FROZEN',config=c,commit=commit,epoch=20,optimizer_steps=steps,checkpoint='final.pt',selection='fixed_final20',sourceV=epoch_records[-1]['sourceV'],target_access=False,truth_access=False,head_fit_updates=0,U_exposure=0,satellite_examples=sat_seen,satellite_applied_examples=applied,seconds=time.perf_counter()-start,peak_cuda_MiB=torch.cuda.max_memory_allocated()/1024**2)
   d.write(out/'completion.json',done)
  return done

if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--row',required=True);p.add_argument('--profile',required=True);p.add_argument('--base',required=True);p.add_argument('--commit',required=True);a=p.parse_args();train(d.config(next(r for r in d.rows() if r['row_id']==a.row)),d.read(a.profile),a.base,a.commit)
