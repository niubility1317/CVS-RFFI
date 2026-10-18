"""One real ownAs20 no-query readout plus one actual6L bank hook probe;770NN total."""
import numpy as np
import torch
from . import design as d,model
from .source import parent_payload
from .bank import select_indices
from .losses import weighted_terms
from experiments.cvs_crossview_supcon41 import model as base

class SyntheticBank:
 def __init__(self):
  n=100;g=np.random.default_rng(6008);self.iq=g.normal(size=(n,2,256)).astype(np.float32);domain=np.arange(n)%15;rx=np.asarray(base.supcon.RX_BY_DOMAIN)[domain];groups=np.repeat(np.arange(4),[10,20,30,40]);ids=np.stack((np.ones(n,int),rx,domain%3+1,np.arange(n)+9000,np.arange(n)+10000),1)
  self.meta=dict(TX=(np.arange(n)+1)%6,RX=rx,day=domain%3+1,domain=domain,bin=groups,scene=np.asarray(['practical_low_suburban','practical_mid_urban','practical_low_urban'])[np.arange(n)%3],state=np.asarray(['LOS','PARTIAL','BLOCKED'])[np.arange(n)%3],lock=np.asarray(['locked','degraded','unlocked'])[np.arange(n)%3],signal_to_added_noise_db=np.asarray([0.,5.,10.,20.])[groups],quality_snr_db=np.asarray([0.,5.,10.,20.])[groups],cumulative_epoch=np.full(n,21),replica=np.arange(n)%4,context_step=np.full(n,4441),source_L_index=np.arange(n),sample_ids=ids)
 def draw(self,mode,epoch,step):
  indices,w,r=select_indices(self.meta['bin'],mode,epoch,step);r['cache_gather_cpu_wall_seconds']=0.;return self.iq[indices].copy(),{k:v[indices].copy() for k,v in self.meta.items()},r

def smoke():
 schedule=base.supcon.schedule;base.supcon.schedule=lambda ep:schedule(int(ep)+20);seen=[];capture=model.capture_forward
 try:
  configs=[d.config(row) for row in d.rows()]
  for cfg in configs:
   args=d.make_args(cfg,'cpu')
   if (args.label_smoothing,args.label_epochs,args.pseudo_epochs)!=(.01,0,10) or not args.use_unlabeled or not args.use_ema_teacher or not args.use_aug:raise ValueError('Original ordinary/globalLS/U/teacher flags changed')
  if len({c['parent_source'] for c in configs})!=1 or any(c['reference_row_id']!=d.REFERENCE_ROW for c in configs):raise ValueError('Both newrows require same strict originalB01As20')
  c=configs[1];parent=parent_payload(c);m=model.build(c,'cpu');m.load_state_dict(parent['model'],strict=True);m.eval();before={k:v.clone() for k,v in m.state_dict().items()}
  with torch.no_grad():z=m.forward_identity_only(torch.zeros(2,2,256))['tx_logits']
  if z.shape!=(2,6) or not torch.isfinite(z).all() or any(not torch.equal(v,m.state_dict()[k]) for k,v in before.items()):raise ValueError('Real own originalAs20 no-query smoke failed')
  real=dict(reference_row_id=d.REFERENCE_ROW,new_rows=[c['row_id'] for c in configs],NN=2,source_IQ=0,target_query=False,checkpoint=c['parent_source']+'/final_ssdg.pth',optimizer_updates=0);del m,before
  torch.manual_seed(6007);m=model.build(c,'cpu');m.load_state_dict(parent['model'],strict=True);m.train();initial={k:v.clone() for k,v in m.state_dict().items()};g=torch.Generator().manual_seed(6008);fake=SyntheticBank();m._full_channel.bank=fake
  state=torch.random.get_rng_state().clone()
  iid=fake.draw('IID',21,4441);strat=fake.draw('STRATIFIED',21,4441)
  if not torch.equal(state,torch.random.get_rng_state()) or iid[2]['weights']!=[1.]*64 or strat[2]['bin_counts']!=[16]*4 or abs(strat[2]['bank_weight_sum']-64)>1e-12:raise ValueError('Private IID/stratified bank draw or risk mass differs')
  m._crossview_smoke_capture=True;m._feature_l_steps=1;x=torch.randn((128,2,256),generator=g);ordinary=x+.01;y=torch.arange(128)%6;domains=torch.arange(128)%15;rx=torch.tensor(base.supcon.RX_BY_DOMAIN)[domains];meta=dict(eq_i=torch.ones(128,dtype=torch.long),rx_i=rx,day_i=domains%3+1,sig_i=torch.arange(128)+2000,base_index=torch.arange(128))
  first,maps=base.capture_forward(m,ordinary,domains,True);m._supcon_pending=dict(first=maps,second={});m._legacy_actual_L_input=ordinary.clone();ce=torch.nn.functional.cross_entropy(first,y,label_smoothing=.01)
  def tracked(mm,xx,dd,active):seen.append((xx.detach().clone(),dd.detach().clone()));return capture(mm,xx,dd,active)
  model.capture_forward=tracked
  extra,info=m._source_labeled_feature_hook(model=m,labeled_output=dict(tx_logits=first),labels=y,extra=(domains,meta),domains=domains,optimizer_step=1,identity_loss=ce,labeled_count=128,labeled_inputs=x)
  model.capture_forward=capture;debug=m._crossview_debug;mask=debug['mask'];rec=m._bank_records[-1]
  if len(seen)!=3 or not torch.equal(seen[1][0],seen[2][0]) or not torch.equal(seen[1][1],seen[2][1]) or not torch.equal(seen[1][0][mask],debug['bank_input']) or not torch.equal(debug['mixed_y'][mask],torch.tensor(rec['bank_TX'],dtype=torch.long)) or not torch.equal(debug['mixed_y'][~mask],y[~mask]) or rec['mixed_label_replacements']<=0:raise ValueError('Actual mixed sameIQ must use bank ownlabels and currentordinary complement')
  z0,z1=debug['mixed_logits'];ones=torch.ones(128);iid_rd,iid_ce,iid_kl,per,_=weighted_terms(z0,z1,debug['mixed_y'],ones);old_rd,old_ce,old_kl,_,_=model.terms(z0,z1,debug['mixed_y'],1.)
  if any(not torch.allclose(a,b,atol=1e-6,rtol=1e-6) for a,b in ((iid_rd,old_rd),(iid_ce,old_ce),(iid_kl,old_kl))):raise ValueError('IID perpacket implementation must recover original formula on same logits')
  ordinary_maps=tuple(debug['ordinary0'].values())+tuple(debug['ordinary1'].values());mixed_maps=tuple(debug['satellite0'].values())+tuple(debug['satellite1'].values());route=torch.autograd.grad(debug['aux'],ordinary_maps+mixed_maps,allow_unused=True,retain_graph=True)
  if any(v is None or not torch.isfinite(v).all() or float(v.abs().sum())<=0 for v in route[:4]) or any(v is not None for v in route[4:]):raise ValueError('ORD128 must use only currentL ordinary maps')
  total=ce+extra;total.backward();head=m.id_backbone.encoder.core.id_backbone.cls_head.weight;sinc=m.id_backbone.encoder.core.id_backbone.sinc;sinc_ids={id(p) for p in sinc.parameters()};backbone=sum(float(p.grad.abs().sum()) for p in m.parameters() if p is not head and id(p) not in sinc_ids and p.grad is not None)
  if sum(p.numel() for p in m.parameters())!=900722 or len(initial)!=206 or not torch.isfinite(total) or head.grad is None or float(head.grad.abs().sum())<=0 or sum(float(p.grad.abs().sum()) for p in sinc.parameters() if p.grad is not None)<=0 or backbone<=0 or any(not torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None):raise ValueError('Original defaulthead/Sinc/backbone joint gradients failed')
  r=m._full_records[-1];shape=m._shape_records[-1];expected=.5*r['common_RD']+.5*r['extra_RD']+.25*shape['shape_RD']+.1*r['SupCon']
  from .schema import INFO_KEYS
  if m.id_backbone.encoder._full_train_packets!=768 or m._shape_NN!=256 or m._full_channel.packets!=64 or m._full_channel.calls!=1 or r['L_forwards']!=6 or r['SupCon_weight']!=.1 or shape['cumulative_step']!=4441 or set(info)!=set(INFO_KEYS) or abs(r['total_L']-expected)>3e-6 or abs(r['total_L']-float(total.detach()))>3e-6 or m._supcon_pending is not None or m._supcon_handles or any(not torch.equal(v,m.state_dict()[k]) for k,v in initial.items()) or getattr(m,'_source_unlabeled_feature_hook',None) is not None:raise ValueError('Actual6L/1.25/ORD128/oldU/defaulthead budget differs')
  result=dict(status='PASS',tested_modes=['IID','STRATIFIED'],actual_hook_mode='STRATIFIED',same_logits_IID_recovers_old_terms=True,mixed_true_bank_labels_used=True,mixed_label_replacements=rec['mixed_label_replacements'],same_IQ_each_pair=True,bank_sampling_preserves_Torch_RNG=True,default_head_joint_gradient=True,Sinc_joint_gradient=True,backbone_joint_gradient=True,actual_L_NN=768,real_checkpoint_NN=2,total_NN=770,real_IQ=0,synthetic_As_IQ=64,optimizer_updates=0,extra_CE_operator_calls=4,extra_CE_logit_packets=512,extra_loss_computation='two128CE counterfactual IID plus originalterms two128CE on existing logits;no NN',head_fit_updates=0,target_access=False)
 finally:base.supcon.schedule=schedule;model.capture_forward=capture
 d.write(d.BASE/'real_checkpoint_smoke.json',dict(status='PASS',record=real,NN=2,real_IQ=0,target_query=False,optimizer_updates=0));d.write(d.BASE/'synthetic_smoke.json',result)
 print('REAL_OWN_AS20_NO_QUERY_AND_ONE_ACTUAL_QUALITYRISK_6L_HOOK PASS770NN;prepare shared bank then train',flush=True);return result
