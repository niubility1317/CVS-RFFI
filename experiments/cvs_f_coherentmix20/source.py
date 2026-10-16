from contextlib import contextmanager
import gzip,json,sys,time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from torch.nn import functional as F
from . import design as d,model as api
from .pairing import Bank,Runtime
from experiments.cvs_additive_as20.source import Observer as ParentObserver
from experiments.cvs_rx_u_increment20 import fold as carrier_fold
from experiments.cvs_feature_sequence50.source import train as shared_train
from experiments.cvs_sigtime_heldcredit.held_credit import fresh_parameter_caches
from experiments.cvs_u_schedule50.counterfactual import _preserve_rng

def serial(v):
 if isinstance(v,np.generic):return serial(v.item())
 if isinstance(v,np.ndarray):return serial(v.tolist())
 if isinstance(v,dict):return {str(k):serial(x) for k,x in v.items()}
 if isinstance(v,(tuple,list)):return [serial(x) for x in v]
 if isinstance(v,complex):return [v.real,v.imag]
 if isinstance(v,float) and not np.isfinite(v):return None
 return v

@contextmanager
def fold_scope():
 previous=carrier_fold.d;carrier_fold.d=SimpleNamespace(RX=d.RX,FIT=d.FIT,HELD=d.HELD,STEPS=d.STEPS,SOURCE=d.SOURCE)
 try:yield
 finally:carrier_fold.d=previous

class RecordingLoader:
 def __init__(self,loader,role,stream):self.loader=loader;self.role=role;self.stream=stream
 def __getattr__(self,key):return getattr(self.loader,key)
 def __len__(self):return len(self.loader)
 def __iter__(self):
  for batch in self.loader:
   x,y,domain,meta=batch
   if set(meta['rx_i'].tolist())-set(d.FIT) or (self.role=='U' and bool((y>=0).any())):raise ValueError('HeldRX1 or U TX truth entered training')
   ids=list(zip(*(meta[k].tolist() for k in ('eq_i','rx_i','day_i','sig_i','base_index'))));self.stream.write(json.dumps(dict(role=self.role,ids=ids))+'\n');yield batch

@contextmanager
def capture_physics(channel):
 cls=channel.residual.ResidualChannel;original=cls.process;records=[]
 def observed(instance,x):
  y,m=original(instance,x);records.append(serial(m));return y,m
 cls.process=observed
 try:yield records
 finally:cls.process=original

class Observer(ParentObserver):
 def data(self,native,ctx,out):
  from cvsrffi.game_tracking.data import opaque_id
  original=ctx['train_loader'].dataset;self.held_opaque=[opaque_id(it) for it in original.index if int(it.rx_i)==d.HELD]
  with fold_scope():unused,self.held=carrier_fold.prepare(native,ctx)
  from cvsrffi.xuc_fusion.native import role_ids_from_native
  self.roles=role_ids_from_native(ctx)
  if len(self.held_opaque)!=1260 or len(set(self.held_opaque))!=1260 or any(v[1]!=d.HELD for v in self.held['ids']):raise ValueError('Complete fixed heldRX1 L only')
  d.write(Path(out)/'source_fold.json',dict(held_RX=d.HELD,fit_RX=list(d.FIT),effective_counts={k:len(v) for k,v in self.roles.items()},all_training_calibration_probe_loaders_exclude_held_RX=True,held_L_packets=1260,held_L_opaque_ids=self.held_opaque,additional_L_IQ_reads=1404,discarded_unused_anchor_packets=len(unused['y']),target_access=False,U_truth_read=False))
  self.sequence=gzip.open(Path(out)/'training_sequence.jsonl.gz','xt',encoding='utf-8',compresslevel=3)
  self._bank_dataset=ctx['train_loader'].dataset;self._bank_domain_map=dict(ctx['domain_label_map'])
  for name,role in (('train_loader','L'),('unlabeled_loader','U')):ctx[name]=RecordingLoader(ctx[name],role,self.sequence)
 def initialize(self,m,c,out):
  super().initialize(m,c,out);started=time.perf_counter()
  with _preserve_rng():self.bank=Bank.from_dataset(self._bank_dataset,self._bank_domain_map)
  self.bank_read_seconds=time.perf_counter()-started;del self._bank_dataset
  self.mix_trace=gzip.open(self.out/'coherent_pairs.jsonl.gz','xt',encoding='utf8',compresslevel=3);m._mix_runtime=Runtime(self.bank,self.mix_trace)
  d.write(self.out/'coherent_bank.json',dict(summary=self.bank.summary(),entries=[dict(physical_id=list(__import__('experiments.cvs_f_coherentmix20.pairing',fromlist=['key']).key(e[3])),TX=e[1],domain=e[2]) for e in self.bank.entries],bank_read_seconds=self.bank_read_seconds,source='Exact filtered visible L dataset only;after scratch smoke',U_truth_read=False,target_access=False))
 @contextmanager
 def native_scope(self,native):
  with super().native_scope(native):
   try:yield
   finally:
    if hasattr(self,'mix_trace'):self.mix_trace.close()
 def epoch(self,m,ctx,rows,out,device):
  super().epoch(m,ctx,rows,out,device);steps=len(rows)*222
  if m._mix_calls!=steps or m._mix_packets!=steps*128 or self.bank.calls!=steps or self.bank.packets!=steps*128 or self.teacher._mix_runtime is not None or self.reader._mix_runtime is not None:raise ValueError('One visibleL coherent view only;no teacher/reader runtime')
  self.mix_trace.flush();d.write(out/'coherent_progress.json',dict(epoch=len(rows),steps=steps,mix_packets=m._mix_packets,max_relative_L2_change=m._mix_max,nontrivial_fraction_005=m._mix_nontrivial/m._mix_packets,target_access=False))
 def source_contract(self,expected,ctx):
  from cvsrffi.xuc_fusion.native import role_ids_from_native
  if role_ids_from_native(ctx)!=self.roles or {k:len(v) for k,v in self.roles.items()}!={'L_s':5040,'U_s':45360,'V':21600}:raise ValueError('Exact four-RX role sizes required')
  if any(not set(v)<=set(expected['role_ids'][k]) for k,v in self.roles.items()) or set(self.held_opaque)&set().union(*(set(v) for v in self.roles.values())) or not set(self.held_opaque)<=set(expected['role_ids']['L_s']):raise ValueError('HeldL/fit roles overlap or escape carrier')
  if self.roles!=d.read(d.SMOKE_SOURCE/'source_contract.json')['role_ids']:raise ValueError('Actual training roles differ from matching no-query smoke contract')
  return dict(expected,role_ids=self.roles,source_receivers=list(d.FIT),held_RX=d.HELD,carrier_contract_ref=d.SOURCE,native_role_comparison='EXACT_MATCH_CARRIER_THEN_SOURCE_FOLD',checkpoint_effective_training_contract='RX3/4/6/8 only;scratch,ownEMA;no inherited state')
 def evaluate(self,state):
  self.reader.load_state_dict(state,strict=True);self.reader.eval();device=next(self.reader.parameters()).device;predictions=[];labels=[];margins=[];ce=0.
  with _preserve_rng(),torch.no_grad(),fresh_parameter_caches(self.reader):
   for start in range(0,1260,128):
    x=self.held['x'][start:start+128].to(device);y=self.held['y'][start:start+128].to(device);z=self.reader.forward_identity_only(x)['tx_logits'].double()
    if z.shape!=(len(y),6) or not bool(torch.isfinite(z).all()):raise ValueError('Finite held defaulthead logits required')
    other=z.masked_fill(F.one_hot(y,6).bool(),-torch.inf);margins+=((z.gather(1,y[:,None]).flatten()-other.max(1).values)/30).cpu().tolist();ce+=float(F.cross_entropy(z,y,reduction='sum'));predictions+=z.argmax(1).cpu().tolist();labels+=y.cpu().tolist()
  errors=sum(p!=y for p,y in zip(predictions,labels));return dict(count=1260,mean_CE=ce/1260,errors=errors,accuracy=1-errors/1260,mean_margin=sum(margins)*30/1260,margin_over30=margins,physical_ids=self.held['ids'],labels=labels,predictions=predictions,head_fit_updates=0)
 def scene(self,scene):
  reader=self.reader;device=next(reader.parameters()).device;labels=[];predictions=[];margins=[];physics=[];ce=0.;count=0
  with _preserve_rng(),torch.no_grad(),fresh_parameter_caches(reader):
   for start in range(0,1260,128):
    raw=self.held['x'][start:start+128].to(device);y=self.held['y'][start:start+128].to(device);ids=self.held['ids'][start:start+128];keys=('eq_i','rx_i','day_i','sig_i','base_index');meta={k:torch.tensor([v[i] for v in ids],dtype=torch.long) for i,k in enumerate(keys)}
    with capture_physics(reader._joint_channel) as full:x,observed=reader._joint_channel.apply(raw,meta,namespace='source_eval/RX1/final20',scene=scene)
    if len(full)!=len(y) or len(observed)!=len(y):raise ValueError('Exactly one full physical draw per held packet')
    physics.extend({**a,**b} for a,b in zip(full,serial(observed)))
    z=reader.forward_identity_only(x)['tx_logits'].double()
    if z.shape!=(len(y),6) or not bool(torch.isfinite(z).all()):raise ValueError('Complete finite held source prediction required')
    ce+=float(F.cross_entropy(z,y,reduction='sum'));predictions+=z.argmax(1).cpu().tolist();labels+=y.cpu().tolist();count+=len(y);other=z.masked_fill(F.one_hot(y,6).bool(),-torch.inf);margins+=((z.gather(1,y[:,None]).flatten()-other.max(1).values)/30).cpu().tolist()
  result=dict(count=count,accuracy=sum(a==b for a,b in zip(labels,predictions))/count,mean_CE=ce/count,macro_F1=d.macro_f1(labels,predictions),labels=labels,predictions=predictions,margin_over30=margins,physics=serial(physics),physical_ids=self.held['ids'],head_fit_updates=0);d.write(self.out/'source_readouts'/(scene+'.json'),result);return result

 def finish(self):
  done=d.read(self.out/'completion.json');payload=torch.load(self.out/'final_ssdg.pth',map_location='cpu',weights_only=False);m=self.model
  if done['epoch']!=20 or done['optimizer_steps']!=4440 or done['initialization']!='scratch' or done['checkpoint_sources'] or payload['args']['baseline_ckpt'] or payload['args']['teacher_ckpt'] or not payload['args']['use_aug'] or payload['args'].get('source_fold_held_rx')!=d.HELD or payload['checkpoint_selection']!='final_only' or payload['a1_ema_successful_updates']!=4440 or payload.get('ema_model') is None:raise ValueError('Full scratch20/defaulthead/originalU provenance required')
  initial=torch.load(self.out/'initial_smoke.pt',map_location='cpu',weights_only=False)['model'];W='id_backbone.encoder.core.id_backbone.cls_head.weight'
  if payload['model'][W].shape!=(6,160) or torch.equal(initial[W],payload['model'][W]):raise ValueError('Default head must train jointly')
  if any(not torch.equal(v,m.state_dict()[n].detach().cpu()) for n,v in payload['model'].items()) or len(m._joint_details)!=4440 or m._supcon_applied!=3330:raise ValueError('Frozen live state or full execution differs')
  clean=self.evaluate(payload['model']);clean['macro_F1']=d.macro_f1(clean['labels'],clean['predictions']);d.write(self.out/'source_readouts/clean.json',clean)
  # evaluate() loaded the exact final state in reader. Same named physical
  # draws apply to B00 and B01; source labels never select draws or rows.
  views={'clean':clean}
  for scene in d.SCENES:views[scene]=self.scene(scene)
  with gzip.open(self.out/'satmix_steps.jsonl.gz','xt',encoding='utf8') as f:
   for record in m._joint_details:f.write(json.dumps(record,allow_nan=False)+'\n')
  with gzip.open(self.out/'unit_steps.jsonl.gz','xt',encoding='utf8') as f:
   for record in self.optimizer._sinc_unit_lr_records:f.write(json.dumps(record,allow_nan=False)+'\n')
  return self._final_record(done,payload,views)

 def _final_record(self,done,payload,views):
  result=super()._final_record(done,payload,views);m=self.model
  expected={(t,r,other,day) for t in range(6) for r in d.FIT for other in d.FIT for day in (1,2,3) if (r==other)==(self.c['arm']=='SAME_RX')}
  if set(self.bank.coverage)!=expected or self.bank.unique!=set(self.bank.index) or m._mix_calls!=4440 or self.teacher._mix_calls or self.teacher._mix_packets:raise ValueError('Complete actual source partner coverage/no teacher interpolation required')
  mix=dict(self.bank.summary(),max_relative_L2_change=m._mix_max,nontrivial_fraction_005=m._mix_nontrivial/m._mix_packets,IQ_mixing=True,hardware_replacement_claim=False,bank_read_seconds=self.bank_read_seconds)
  result.update(held_RX=d.HELD,fit_RX=list(d.FIT),Ag=True,As=False,mix=mix,teacher_mix_calls=0);result['views']={name:{k:v for k,v in value.items() if k not in ('physical_ids','labels','predictions','physics','margin_over30')} for name,value in views.items()};d.write(self.out/'satmix_result.json',result);return result

def train(c):
 observer=Observer(c);shared_train(c,design=d,model_api=api,observer=observer);return observer.finish()
