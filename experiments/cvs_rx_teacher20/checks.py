import copy,json
from types import SimpleNamespace
from collections import Counter
from pathlib import Path
import torch
from torch.nn import functional as F
from . import design as d,subset,model as api
from experiments.cvs_phase1_stack.runtime import installed
def reject(call):
 try:call()
 except (ValueError,RuntimeError):return
 raise AssertionError('Illegal source action accepted')
def main():
 torch.set_num_threads(2);negatives=0
 index=[SimpleNamespace(tx_i=t,rx_i=r,day_i=day,eq_i=0,sig_i=i) for t in range(6) for r in d.RX for day in (1,2,3) for i in range(70)]
 def physical(v):return tuple(getattr(v,k) for k in ('tx_i','rx_i','day_i','eq_i','sig_i'))
 for arm in d.ARMS:
  chosen=subset.select(index,arm);rev=list(reversed(index));other=subset.select(rev,arm)
  assert {physical(index[i]) for i in chosen}=={physical(rev[i]) for i in other}
  assert len(chosen)==5040 and set(Counter((index[i].tx_i,index[i].day_i) for i in chosen).values())=={280}
  assert {index[i].rx_i for i in chosen}==set(d.RX if arm=='SAMPLE_80' else d.FIT)
  reject(lambda:subset.select(index[:-1],arm));negatives+=1
  bad=copy.deepcopy(index);bad[1]=bad[0];reject(lambda:subset.select(bad,arm));negatives+=1
 models=[]
 for row in d.rows():
  c=d.config(row);a=d.make_args(c,'cpu');assert (a.epochs,a.label_epochs,a.pseudo_epochs)==(20,20,0)
  assert not any((a.use_unlabeled,a.use_ema_teacher,a.use_aug,a.use_concat_sat_channel_aug,a.baseline_ckpt,a.teacher_ckpt)) and a.label_smoothing==0 and all(v==0 for k,v in vars(a).items() if k.startswith('lambda_'))
  for key,value in [('epochs',50),('head_fit_updates',1),('test_views',['clean']),('model_seed',2026092702),('parent_checkpoint','allRX.pth')]:
   bad=dict(c);bad[key]=value;reject(lambda:d.validate(bad));negatives+=1
  with installed(c) as native:
   pairs=list(native._muse_epoch_pairs([1,2,3],subset.NoU(),use_muse=False))
  assert len(pairs)==222 and all(p[1] is None for p in pairs)
  models.append(api.build(c,'cpu'))
 initial=models[0].state_dict();assert all(torch.equal(v,models[1].state_dict()[k]) for k,v in initial.items())
 m=models[1].train();x=torch.randn(128,2,256);y=torch.arange(128)%6;ids=list(range(128))
 m._teacher_allowed_L={i:(int(y[i]),3,1) for i in ids}
 meta=dict(base_index=torch.arange(128),rx_i=torch.full((128,),3),day_i=torch.ones(128,dtype=torch.long))
 with m._phase1_labeled_feature_context():out=m.forward_identity_only(x)
 ce=F.cross_entropy(out['tx_logits'],y);args=dict(model=m,labeled_output=out,labels=y,extra=(None,meta),domains=None,optimizer_step=1,identity_loss=ce,labeled_count=128)
 zero,metrics=m._source_labeled_feature_hook(**args)
 assert float(zero.detach())==0 and m._teacher_calls==m._feature_l_steps==1 and len(m._teacher_seen)==128
 params=dict(m.named_parameters());W=api.classifier(m).weight
 names=('id_backbone.encoder.core.id_backbone.t1.pw.weight','id_backbone.encoder.core.id_backbone.t2.pw.weight','id_backbone.encoder.sequence_pool.local_projection.weight')
 selected=[params[k] for k in names]+[W]
 plain=torch.autograd.grad(ce,selected,retain_graph=True);actual=torch.autograd.grad(ce+zero,selected)
 assert all(torch.isfinite(v).all() and v.norm()>0 for v in actual) and all(torch.equal(a,b) for a,b in zip(plain,actual)) and all(p.grad is None for p in m.parameters())
 m._teacher_calls=0
 for key,value in [('rx_i',torch.ones(128,dtype=torch.long)),('base_index',torch.full((128,),99999)),('day_i',torch.full((128,),9))]:
  bad=dict(meta);bad[key]=value;reject(lambda:m._source_labeled_feature_hook(**dict(args,extra=(None,bad))));negatives+=1
 reject(lambda:m._source_labeled_feature_hook(**dict(args,labels=(y+1)%6)));negatives+=1
 reject(lambda:m._source_labeled_feature_hook(**dict(args,extra=None)));negatives+=1
 bad=dict(meta);bad['base_index']=torch.arange(127);reject(lambda:m._source_labeled_feature_hook(**dict(args,extra=(None,bad))));negatives+=1
 m.eval();reject(lambda:m._source_labeled_feature_hook(**args));negatives+=1
 from .provenance import validate_evidence,validate_checkpoint
 from .compare import qualify
 for row,constructed in zip(d.rows(),models):
  c=d.config(row);all_L=['L'+str(i) for i in range(6300)];all_V=['V'+str(i) for i in range(27000)]
  expected=dict(role_ids=dict(L_s=all_L,U_s=['U0'],V=all_V));records=[]
  for tx in range(6):
   for rx in c['teacher_exposure']['train_RX']:
    for day in (1,2,3):
     for i in range(c['teacher_exposure']['per_RX_TX_day']):
      b=len(records);records.append(dict(physical_id=all_L[b],base_index=b,TX=tx,RX=rx,day=day))
  contract=dict(role_ids=dict(L_s=all_L[:5040],U_s=[],V=all_V[:21600]),source_receivers=c['teacher_exposure']['train_RX'],teacher_exposure=c['teacher_exposure'],carrier_contract_ref=d.SOURCE,native_role_comparison='EXACT_MATCH_CARRIER_THEN_TEACHER_SUBSET')
  done=dict(status='SOURCE_TRAINED',config=c,epoch=20,optimizer_steps=4440,logged_steps=4440,L_context_steps=4440,target_access=False,target_evaluated=False,initialization='scratch',checkpoint_sources=[],ancestors=[],baseline_recipe=c['training_recipe'],label_epochs=20,pseudo_epochs=0,head_fit_updates=0,native_fitted_probes_disabled=True,native_disabled_probe_calls=1,identity_input_policy=dict(explicit_CFO_identity_input=False),feature_extractor=copy.deepcopy(constructed._feature_learning_report),mask_counts={},final_source_metrics=dict(source_val_count=21600,source_val_rx_accuracy={str(rx):1. for rx in d.FIT}))
  initial=dict(scratch_only=True,checkpoint_sources=[],ancestors=[],source_roles='EXACT_MATCH',target_contact=False,teacher_origin='disabled',seed=d.SEED)
  freeze=dict(status='SOURCE_FROZEN',config=c,checkpoint=str(Path(c['output_root'])/'final_ssdg.pth'),epoch=20,selection='final_only',target_access=False)
  resolved=dict(config=c,native_fitted_probes_disabled=True,identity_input_policy=dict(explicit_CFO_identity_input=False),actual_feature_extractor=done['feature_extractor'])
  actual=vars(d.make_args(c,'cpu')).copy();selected=dict(config=c,train_records=records,fit_V_RX_counts={str(rx):5400 for rx in d.FIT},U_iterator_calls=0,held_V_IQ_read_before_freeze=False,checkpoint_sources=[],ancestors=[])
  progress=dict(epoch=20,verified_L_steps=4440,unique_L_packets=5040,U_iterator_calls=0,EMA=False,target_access=False,head_fit_updates=0)
  packet=[c,done,initial,freeze,resolved,actual,contract,expected,selected,progress];validate_evidence(*packet)
  for position,key,value in ((1,'ancestors',['allRX.pth']),(1,'target_access',True),(1,'head_fit_updates',1),(5,'use_unlabeled',True),(6,'source_receivers',[1]),(8,'U_iterator_calls',1),(9,'verified_L_steps',11100)):
   bad=copy.deepcopy(packet);bad[position][key]=value;reject(lambda:validate_evidence(*bad));negatives+=1
  bad=copy.deepcopy(packet);bad[8]['train_records'][0]['physical_id']='outside';reject(lambda:validate_evidence(*bad));negatives+=1
  checkpoint=dict(epoch=20,run_id=d.RUN,candidate_id=c['row_id'],checkpoint_selection='final_only',a1_ema_successful_updates=4440,ema_model=None,args=actual);validate_checkpoint(checkpoint,c)
  for key,value in (('epoch',50),('ema_model',{'leak':1}),('a1_ema_successful_updates',11100)):
   bad=dict(checkpoint);bad[key]=value;reject(lambda:validate_checkpoint(bad,c));negatives+=1
 import numpy as np
 y=[0]*5400;pred=[1]+[0]*5399;base=dict(labels=y,predictions=pred,source_health=True,errors=1,mean_CE=1.)
 candidate=dict(base,predictions=y,errors=0);mask=np.ones(5400,dtype=bool)
 assert qualify(base,candidate,mask)['qualified']
 assert not qualify(base,candidate,np.zeros(5400,dtype=bool))['qualified']
 assert not qualify(base,dict(candidate,mean_CE=1.1),mask)['qualified']
 assert not qualify(base,dict(candidate,source_health=False),mask)['qualified']
 assert not qualify(candidate,base,mask)['qualified']
 reject(lambda:qualify(base,candidate,np.ones(5399,dtype=bool)));negatives+=1
 print(json.dumps(dict(status='PASS',negative_cases=negatives,models=2,matched_initial_tensors=len(models[0].state_dict()),native_dummy_epoch_steps=222,U_iterator_calls=0,synthetic_L_forward_packets=128,actual_body_and_default_W_gradients_nonzero=True,zero_hook_gradient_equals_native_CE_bitwise=True,head_fit_updates=0,checkpoint_loads=0,real_IQ_reads=0,query_reads=0,optimizer_steps=0)))
if __name__=='__main__':main()
