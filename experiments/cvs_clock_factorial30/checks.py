"""New inheritance,EMA,incremental budget and truth closure checks;no real IQ."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import tempfile
import numpy as np
import torch
from . import design as d,model as api

def protocol():
 n=0
 for r in d.rows():
  c=d.config(r);d.validate(c)
  for key,value in (('epochs',30),('label_epochs',1),('cumulative_epochs',40),('parent_source','UNKNOWN'),('operator_epoch_offset',0),('classifier','linear'),('U_truth_read',True),('baseline_reuse',True)):
   bad=deepcopy(c);bad[key]=value
   try:d.validate(bad)
   except ValueError:n+=1
   else:raise AssertionError('Changed E30 contract accepted')
  for key,value in (('RX',[1,3,4,6]),('L',5580)):
   bad=deepcopy(c);bad['data_contract'][key]=value
   try:d.validate(bad)
   except ValueError:n+=1
   else:raise AssertionError('Source holdout accepted')
  p=d.parent_config(c['arm']);cp=dict(run_id=p['run_id'],candidate_id=p['row_id'],epoch=20,checkpoint_selection='final_only',a1_ema_successful_updates=4440,ema_model={'exists':True},args=dict(baseline_ckpt='',teacher_ckpt='',source_fold_held_rx=None,freeze_backbone=False))
  d.validate_parent_checkpoint(cp,c)
  for key,value in (('run_id','UNKNOWN'),('candidate_id','OTHER'),('epoch',19),('checkpoint_selection','best_target'),('a1_ema_successful_updates',4439),('ema_model',None)):
   bad=deepcopy(cp);bad[key]=value
   try:d.validate_parent_checkpoint(bad,c)
   except ValueError:n+=1
   else:raise AssertionError('Unknown/target-selected parent accepted')
 return n

def run():
 from .source import Observer
 from experiments.cvs_fullsource_concat20 import supcon
 from experiments.cvs_sigtime_heldcredit.held_credit import fresh_parameter_caches
 torch.set_num_threads(2);n=protocol();reports=[];NN=0;physics=0;clock_packets=0
 from experiments.cvs_phase1_overlay.model import native_modules
 native=native_modules()
 original_schedule=supcon.schedule
 for row in d.rows():
  c=d.config(row);a=d.make_args(c,'cpu');native._validate_a1_scratch_only(a);assert a.a1_scratch_only is False and a.from_scratch is False and a.baseline_ckpt
  bad_args=deepcopy(a);bad_args.a1_scratch_only=True
  try:native._validate_a1_scratch_only(bad_args)
  except ValueError:n+=1
  else:raise AssertionError('Scratch-only flag accepted a parent')
  torch.manual_seed(202610175501);m=api.build(c,'cpu');initial={k:v.detach().clone() for k,v in m.state_dict().items()}
  teacher=deepcopy(initial);W='id_backbone.encoder.core.id_backbone.cls_head.weight';teacher[W]=teacher[W]+.01
  parent=dict(model=initial,ema_model=teacher,args=vars(d.ref.make_args(d.clock.reference_config())))
  with tempfile.TemporaryDirectory() as folder:
   observer_config=dict(c,output_root=str(Path(folder)/'source'));o=Observer(observer_config,parent)
   assert not hasattr(o,'model') and not o.out.exists()
   fake=SimpleNamespace(deepcopy=deepcopy,_resolve_phase1_terminal_status=native._resolve_phase1_terminal_status,configure_augmentor_for_epoch=lambda obj,cfg,e,a:(e,a.label_epochs))
   torch.manual_seed(202610175502);raw=torch.randn(128,2,256);ordinary=raw*.9;y=torch.arange(128)%6;domains=torch.arange(128)%15;rx=torch.tensor(supcon.RX_BY_DOMAIN)[domains]
   meta=dict(rx_i=rx,day_i=domains%3+1,eq_i=torch.ones(128,dtype=torch.long),sig_i=torch.arange(128),base_index=torch.arange(128))
   for obj in (m._full_channel,getattr(m,'_clock',None)):
    if obj is None:continue
    count=64 if obj is getattr(m,'_clock',None) else 128
    small={k:v[:count] for k,v in meta.items()}
    for ep,st in ((1,1),(20,4440),(31,6661),(21,4663)):
     try:obj.obj.apply(raw[:count],small,ep,st)
     except ValueError:n+=1
     else:raise AssertionError('Wrong cumulative stream budget accepted')
   seen=[];original=m.id_backbone.encoder._full_features
   def traced(x):seen.append(x.detach().clone());return original(x)
   m.id_backbone.encoder._full_features=traced;m.train()
   with o.native_scope(fake):
    assert not hasattr(o,'model') and (o.out/'terminal_policy.json').is_file()
    o.initialize(m,observer_config,o.out)
    assert fake.configure_augmentor_for_epoch(None,None,0,None)==(13,13)
    terminal_args=dict(tail_stopped=False,export_failed=False,final_blocked=False,selected_checkpoint_exists=True,heldout_eval_status='COMPLETE',p0_mechanisms_ready=False,p1_mechanisms_ready=False,endpoint_export_ready=True,mechanism_gates_required=True,endpoint_export_required=False)
    assert fake._resolve_phase1_terminal_status(**terminal_args)=='COMPLETE'
    for key,value,wanted in (('tail_stopped',True,'STOPPED_TAIL'),('export_failed',True,'FAILED_EXPORT'),('final_blocked',True,'NON_PROMOTABLE_GUARD_BLOCKED'),('selected_checkpoint_exists',False,'NO_SAFE_CHECKPOINT'),('heldout_eval_status','PENDING','HELDOUT_EVAL_INCOMPLETE')):
     bad=dict(terminal_args);bad[key]=value;assert fake._resolve_phase1_terminal_status(**bad)==wanted;n+=1
    t=fake.deepcopy(m);assert all(torch.equal(v,teacher[k]) for k,v in t.state_dict().items());assert not torch.equal(t.state_dict()[W],m.state_dict()[W]);del t
    opt=torch.optim.AdamW(m.parameters(),lr=.0002,weight_decay=.0001);assert len(opt.state)==0
    with m._phase1_labeled_feature_context():z=m(ordinary,return_aux=True)
    ce=torch.nn.functional.cross_entropy(z['tx_logits'],y,label_smoothing=.01)
    corr,_=m._source_labeled_feature_hook(model=m,labeled_output=z,labels=y,extra=(domains,meta),domains=domains,optimizer_step=1,identity_loss=ce,labeled_count=128,labeled_inputs=raw)
    rec=m._full_records[0];assert rec['SupCon_weight']==.1 and len(rec['SupCon_details'])==4
    assert len(seen)==4 and torch.equal(seen[0],seen[1]) and torch.equal(seen[2],seen[3])
    full=row['arm']=='FULL_AS';ag=bool(c.get('new_Ag'));sat=bool(c.get('As'));g=(torch.arange(128)+1)%2==0
    if full:assert not torch.equal(seen[2][g],ordinary[g])
    else:assert torch.equal(seen[2][g],ordinary[g])==(not ag)
    assert torch.equal(seen[2][~g],ordinary[~g])==(not sat)
    assert rec.get('Ag_packets',0)==(64 if ag else 0) and rec['As_packets']==(128 if full else 64 if sat else 0)
    total=ce+corr;assert abs(float(total.detach())-(.5*(rec['common_RD']+rec['extra_RD'])+.1*rec['SupCon']))<2e-6
    total.backward();head=m.get_parameter(W);before=head.detach().clone();assert head.grad is not None and torch.isfinite(head.grad).all()
    assert any(p.grad is not None and bool((p.grad!=0).any()) for name,p in m.named_parameters() if name!=W)
    torch.nn.utils.clip_grad_norm_(m.parameters(),5.);opt.step();assert not torch.equal(head,before) and opt._sinc_unit_steps==1 and opt._sinc_physical_clip_calls==1
   assert supcon.schedule is original_schedule and original_schedule(1)==0
   m.id_backbone.encoder._full_features=original;m.eval()
   with torch.no_grad(),fresh_parameter_caches(m):
    f=m.id_backbone.encoder.features(raw[:2]);ef=api.export(m).eval()(raw[:2]);assert f.shape==(2,160) and torch.equal(f,ef)
   NN+=516;physics+=rec['As_packets'];clock_packets+=rec.get('Ag_packets',0)
   reports.append(dict(arm=row['arm'],parameters=sum(p.numel() for p in m.parameters()),parent_student_strict=True,own_EMA_strict_distinct_from_student=True,optimizer_reset=True,body_default_head_joint_update=True,SupCon_cumulative_weight=.1,ordinary_parent_E13=True,L_forwards=4,Ag_packets=rec.get('Ag_packets',0),As_packets=rec['As_packets'],export160_exact=True,same_IQ_pairs=True))
  del m
 from . import evaluate as ev,proof as pv
 original_read,original_ids,original_source=d.read,ev.ids,pv.source
 try:
  pv.source=lambda c:None;ev.ids=lambda:np.arange(ev.COUNT)
  valid=dict(status='PREDICTIONS_COMPLETE',config=d.config(d.rows()[0]),count=ev.COUNT,views=list(d.VIEWS),truth_read=False,query_fit=False,head_fit_updates=0)
  for key,bad in (('status','PENDING'),('count',5),('views',['clean']),('truth_read',True),('query_fit',True),('head_fit_updates',1)):
   value=dict(valid);value[key]=bad
   def fake_read(path):
    if Path(path).name=='complete.json' and Path(path).parent.name=='prediction':return value
    raise AssertionError('Truth opened before closure')
   d.read=fake_read
   try:ev.score(d.rows()[0]['row_id'])
   except ValueError:n+=1
   else:raise AssertionError('Incomplete prediction allowed truth')
 finally:d.read,ev.ids,pv.source=original_read,original_ids,original_source
 return dict(status='PASS',negative_cases=n,rows=reports,synthetic_NN=NN,synthetic_physics_packets=physics,synthetic_clock_packets=clock_packets,synthetic_optimizer_steps=5,real_IQ=0,CP=0,target_access=False)
