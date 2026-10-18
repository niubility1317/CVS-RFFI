"""Focused mask/routing/native-boundary negatives;CPU NumPy mocks,no IQ/CP/NN."""
import ast,copy,json,random,sys,types
from pathlib import Path
import numpy as np
from . import design as d
from .protocol import cumulative,check_checkpoint,check_default_head
from .schema import INFO_KEYS,CONCAT_FIELDS
from .u_routing import select,metadata,Router,NAMESPACE

_NATIVE_FEATURE_CODE=None
_NATIVE_NONE_GUARD_CODE=None

def frozen_boundary():
 """Compile the actual frozen model feature-hook call and its None-adjustment guard."""
 global _NATIVE_FEATURE_CODE,_NATIVE_NONE_GUARD_CODE
 local=d.ROOT/'local_artifacts/F_UPractical_E30_20261019/frozen_native_train_ssdg.py'
 if local.is_file():p=local
 else:
  from experiments.cvs_phase1_overlay.model import native_modules
  p=Path(native_modules().__file__)
 source=p.read_text(encoding='utf8');tree=ast.parse(source)
 assert '_trusted_pseudo_loss_hook' not in source
 found=[];guards=[];base_masks=[]
 for node in ast.walk(tree):
  for field in ('body','orelse'):
   body=getattr(node,field,None)
   if not isinstance(body,list):continue
   for i,stmt in enumerate(body[:-1]):
    if isinstance(stmt,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='feature_hook' for t in stmt.targets) and isinstance(stmt.value,ast.Call) and isinstance(stmt.value.func,ast.Name) and stmt.value.func.id=='getattr' and len(stmt.value.args)>=2 and isinstance(stmt.value.args[0],ast.Name) and stmt.value.args[0].id=='model' and isinstance(stmt.value.args[1],ast.Constant) and stmt.value.args[1].value=='_source_unlabeled_feature_hook':found.append((stmt,body[i+1]))
   for stmt in body:
    if isinstance(stmt,ast.If) and isinstance(stmt.test,ast.Compare) and isinstance(stmt.test.left,ast.Name) and stmt.test.left.id=='source_unlabeled_feature_loss':guards.append(stmt)
    if isinstance(stmt,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='base_mask' for t in stmt.targets) and ast.unparse(stmt.value)=='conf_mask & domain_mask & temporal_mask':base_masks.append(stmt)
 assert len(found)==1 and len(guards)==1 and len(base_masks)==1
 assignment,condition=found[0];call=next(n for n in ast.walk(condition) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='feature_hook');keys=[k.arg for k in call.keywords]
 assert keys==['model','teacher','student_output','teacher_output','extra','domains','optimizer_step','identity_loss','pseudo_loss','entropy_loss']
 segment=source[source.index('elif legacy_unlabeled_active:'):source.index('elif legacy_unlabeled_active:')+9000]
 assert segment.index('base_mask = conf_mask & domain_mask & temporal_mask')<segment.index('x_s = _strong_augment(x_u, float(args.strong_noise_std))')<segment.index('out_s = model(x_s')<segment.index('loss_u = F.cross_entropy(out_s["tx_logits"][mask], pseudo[mask])')
 assert 'domain_mask = torch.ones_like(conf_mask)' in segment and 'strong_mask = torch.ones_like(base_mask)' in segment and 'loss_u = out_s["tx_logits"].sum() * 0.0' in segment and 'loss_ent = entropy_per_sample.mean()' in segment
 _NATIVE_FEATURE_CODE=compile(ast.fix_missing_locations(ast.Module(body=[assignment,condition],type_ignores=[])),str(p),'exec');_NATIVE_NONE_GUARD_CODE=compile(ast.fix_missing_locations(ast.Module(body=guards,type_ignores=[])),str(p),'exec')
 return dict(status='PASS',actual_source=str(p),actual_model_hook_line=assignment.lineno,actual_call_line=call.lineno,keyword_names=keys,existing_base_mask_line=base_masks[0].lineno,original_mask_before_strong=True,missing_old_module_hook=True,None_adjustment_guard_line=guards[0].lineno,NN=0,IQ=0,CP=0)

def routing_checks():
 old_torch=sys.modules.get('torch');old_scope=sys.modules.pop(__package__+'.u_scope',None)
 class T(np.ndarray):
  @property
  def device(self):return 'cpu'
  def clone(self):return self.copy().view(T)
  def detach(self):return self
  def cpu(self):return self
  def sum(self,*a,**kw):return np.asarray(super().sum(*a,**kw)).view(T)
 def tensor(a,device=None,dtype=None):return np.asarray(a,dtype=dtype).view(T)
 fake=types.SimpleNamespace(tensor=tensor,long=np.int64,isfinite=np.isfinite,equal=np.array_equal)
 sys.modules['torch']=fake;positive=0;negative=0
 def bad(f):
  nonlocal negative
  try:f()
  except (ValueError,TypeError,KeyError):negative+=1
  else:raise AssertionError('Invalid U routing/native boundary accepted')
 def metas(n):return dict(eq_i=tensor(np.ones(n,np.int64)),rx_i=tensor(np.asarray(d.RX)[np.arange(n)%5]),day_i=tensor(np.arange(n)%3+1),sig_i=tensor(np.arange(n)+2000),base_index=tensor(np.arange(n)),tx_label_visible=tensor(np.zeros(n,bool)))
 class Channel:
  def __init__(self):self.calls=0;self.packets=0;self.observed=[]
  def apply(self,x,meta,epoch,step):
   self.calls+=1;self.packets+=len(x);self.observed.append((x.copy(),copy.deepcopy(meta),epoch,step))
   obs=[dict(sample_id=tuple(int(meta[k][i]) for k in ('eq_i','rx_i','day_i','sig_i','base_index')),session_id='mockRXday',namespace='U_QV',scene='practical_low_urban',state_end='LOS',receiver_lock_status='LOCKED',signal_to_added_noise_db=4.,quality_snr_db=5.) for i in range(len(x))]
   return (x+7).view(T),obs
  def summary(self):return dict(calls=self.calls,packets=self.packets)
 try:
  from .u_scope import NativeUScope
  for n in (256,124,8,7):
   raw=tensor(np.arange(n*512,dtype=np.float32).reshape(n,2,256)/1000);before=raw.copy();strong=raw+.02;meta=metas(n)
   for k in ('sig_i','base_index','rx_i','day_i'):meta[k][1]=meta[k][0]
   for kind in ('empty','single','sparse','all'):
    mask=tensor(np.zeros(n,bool));mask[-1]=kind=='single'
    if kind=='sparse':mask[::3]=True
    elif kind=='all':mask[:]=True
    qualified=np.flatnonzero(mask).tolist();k=len(qualified)
    for row in d.rows():
     mode=d.config(row)['u_practical']['mode'];ps=np.random.get_state();pr=random.getstate();positions,plan=select(n,mask,21,4441,mode=mode);after=np.random.get_state()
     assert np.array_equal(ps[1],after[1]) and ps[2:]==after[2:] and pr==random.getstate() and len(set(positions))==k and plan['qualification_positions']==qualified
     c=d.config(row);ch=Channel();router=Router(c,ch);student,r=router.route(raw,strong,meta,mask,epoch=21,step=4441);ix=r['selected_positions'];com=[i for i in range(n) if i not in ix]
     assert ix==positions and r['selected_slots']==r['As_packets']==r['qualification_count']==k and r['U_packets']==n and np.array_equal(raw,before) and r['duplicate_PID_occurrences']==1 and not r['U_truth_available'] and not r['same_future_Ascount_claim']
     assert ch.calls==int(k>0) and ch.packets==k and (student is strong)==(k==0) and np.array_equal(student[com],strong[com])
     if k:assert np.array_equal(student[ix],raw[ix]+7) and np.array_equal(ch.observed[0][0],raw[ix]) and ch.observed[0][2:]==(21,4441) and ch.observed[0][1]['base_index']==meta['base_index'][ix].tolist()
     if mode=='QUALIFIED_PRACTICAL':assert ix==qualified and r['Gaussian_positions']==np.flatnonzero(~mask).tolist()
     assert router.summary()['qualification_count']==k and router.summary()['As_packets']==k
     json.dumps(r,allow_nan=False);positive+=1
  # Dynamic dosage changes with this row's actual mask, not a fixed quarter or other row.
  n=124;mask=tensor(np.arange(n)%5==0);router=Router(d.config(d.rows()[0]),Channel());x=tensor(np.zeros((n,2,256),np.float32))
  for i in range(3):router.route(x,x+.01,metas(n),mask if i!=1 else ~mask,epoch=21,step=4441+i)
  assert router.As_packets==int(mask.sum())*2+int((~mask).sum()) and router.As_packets==router.selected==router.qualified and router.calls==3
  positive+=1
  # Execute the actual frozen feature-hook call/None guard against scoped native mocks.
  for row in d.rows():
   for kind in ('empty','sparse','all'):
    n=8;c=d.config(row);raw=tensor(np.ones((n,2,256),np.float32));meta=metas(n);router=Router(c,Channel());pseudo=tensor(np.arange(n)%6,dtype=np.int64);conf=tensor(np.linspace(.93,.99,n));cm=tensor(np.ones(n,bool));tm=tensor(np.zeros(n,bool) if kind=='empty' else np.ones(n,bool) if kind=='all' else np.arange(n)%2==0);logits=tensor(np.tile(np.arange(6,dtype=np.float32),(n,1)));calls=dict(threshold=0,temporal=0,strong=0)
    def threshold(*args):calls['threshold']+=1;return cm
    def temporal(*args):calls['temporal']+=1;return tm
    def strong(x,std):calls['strong']+=1;return (x+np.random.normal(0,std,x.shape).astype(x.dtype)).view(T)
    native=types.SimpleNamespace(_threshold_mask=threshold,_temporal_mask_tensor=temporal,_strong_augment=strong);m=types.SimpleNamespace();o=types.SimpleNamespace(model=m,pending_U=dict(meta=meta,count=n,hidden_y=True,teacher_raw=raw,teacher_raw_no_grad=True),u_records=[],u_router=router);scope=NativeUScope(o,native);scope.install();args=types.SimpleNamespace(pseudo_strong_agreement=False,pseudo_domain_gate=False,pseudo_temporal_gate=True,pseudo_temporal_mode='batch_neighbor',pseudo_threshold_mode='rx_day_quantile',pseudo_quantile=.86,tau_min=.92,tau_max=.97,pseudo_temporal_window=2,pseudo_temporal_min_conf=.8,lambda_u=.16,lambda_ent=.01,label_smoothing=.01)
    try:
     assert native._threshold_mask(conf,tensor(np.arange(n)%3),args) is cm;assert native._temporal_mask_tensor(pseudo,conf,(None,meta),args,'cpu') is tm
     state=np.random.get_state();expected=strong(raw,.015);expectedstate=np.random.get_state();np.random.set_state(state);calls['strong']=0;student=native._strong_augment(raw,.015);gotstate=np.random.get_state();assert calls['strong']==1 and np.array_equal(expectedstate[1],gotstate[1]) and expectedstate[2:]==gotstate[2:]
     mask=cm&tm;r=o.pending_U['record'];assert r['qualification_positions']==np.flatnonzero(mask).tolist() and np.array_equal(student[r['Gaussian_positions']],expected[r['Gaussian_positions']])
     if c['arm']=='Q01':assert r['selected_positions']==r['qualification_positions']
     lu=tensor(1.2 if mask.any() else 0.);ent=tensor(.7)
     env=dict(model=m,pseudo_source=types.SimpleNamespace(),out_s=dict(tx_logits=logits),out_w=dict(tx_logits=logits),extra_u=(None,meta),d_u=tensor(np.arange(n)%3),r3_optimizer_steps=0,loss_tx_l=tensor(1.),loss_u=lu,loss_ent=ent,y_u=tensor(np.full(n,-1)),r3_u_logs={})
     exec(_NATIVE_FEATURE_CODE,{},env)
     assert env['source_unlabeled_feature_loss'] is None and env['feature_metrics']=={} and env['loss_u'] is lu and env['loss_ent'] is ent
     env.update(loss=tensor(1.7),scaled_closed_loss=tensor(1.7));original_loss=env['loss'];exec(_NATIVE_NONE_GUARD_CODE,{},env);assert env['loss'] is original_loss
     assert o.pending_U is None and len(o.u_records)==1 and calls==dict(threshold=1,temporal=1,strong=1)
     r=o.u_records[0];assert r['CE_denominator']==r['qualification_count']==r['As_packets']==int(mask.sum()) and r['entropy_denominator']==n and r['entropy_all_U'] and r['native_losses_preserved'] and r['native_feature_adjustment'] is None and r['CE_label_smoothing']==0 and not r['strong_agreement']
     if c['arm']=='Q01':assert r['eligible_As_slots']==int(mask.sum()) and r['eligible_Gaussian_slots']==0
     json.dumps(r,allow_nan=False);positive+=1
    finally:scope.restore()
    assert native._threshold_mask is threshold and native._strong_augment is strong and not hasattr(native,'_trusted_pseudo_loss_hook') and not hasattr(m,'_source_unlabeled_feature_hook')
  for n,e,s in [(0,21,4441),(256,20,4441),(256,22,4441),(256,31,6661),(True,21,4441)]:bad(lambda n=n,e=e,s=s:select(n,np.zeros(max(0,int(n)),bool),e,s,mode='COUNT_RANDOM'))
  for value in ([1,0]*4,np.ones((8,1),bool),np.ones(7,bool),np.asarray([np.nan]*8)):bad(lambda value=value:select(8,value,21,4441,mode='COUNT_RANDOM'))
  bad(lambda:select(8,np.ones(8,bool),21,4441,mode='QUARTER_PRACTICAL'))
  for k,v in [('tx_i',np.zeros(8,int)),('tx_label',np.zeros(8,int)),('rx_i',np.full(8,2)),('day_i',np.full(8,4)),('eq_i',np.zeros(8,int)),('base_index',np.arange(7))]:
   mm=metas(8);mm[k]=v;bad(lambda mm=mm:metadata(mm,8))
  c=d.config(d.rows()[0]);ch=Channel();router=Router(c,ch);x=tensor(np.zeros((8,2,256),np.float32));mask=tensor(np.ones(8,bool));bad(lambda:router.route(x[:7],x,metas(8),mask,epoch=21,step=4441));nonfinite=x.copy();nonfinite[0,0,0]=np.nan;bad(lambda:router.route(nonfinite,x,metas(8),mask,epoch=21,step=4441))
 finally:
  if old_torch is None:sys.modules.pop('torch',None)
  else:sys.modules['torch']=old_torch
  sys.modules.pop(__package__+'.u_scope',None)
  if old_scope is not None:sys.modules[__package__+'.u_scope']=old_scope
 return dict(status='PASS',mock_positive_cases=positive,negatives=negative,production_batches=[256,124],nonmultiple_batch_supported=7,mask_cases=['empty','single','sparse','all'],dynamic_actual_dosage=True,zero_mask_channel_calls=0,zero_mask_clone=False,original_strong_numpy_RNG_endstate_equal=True,global_private_selection_numpy_Python_RNG_preserved=True,raw_unmodified=True,selected_raw_not_Gaussian=True,original_masks_returned_by_identity=True,teacher_only_accepts_student_disagreement=True,empty_mask_native_differentiable_zero_interface=True,U_truth_used=False,same_future_Ascount_claim=False,NN=0)

def run():
 negative=0
 def bad(f):
  nonlocal negative
  try:f()
  except (ValueError,TypeError,KeyError,StopIteration):negative+=1
  else:raise AssertionError('Invalid fixed qualification-route/data/parent scope accepted')
 assert [r['arm'] for r in d.rows()]==['R01','Q01'] and NAMESPACE=='FcrossRX/UQualified/count-routing/v1' and d.ALLOCATION_SEED==20261019731
 for row in d.rows():
  c=d.config(row);d.validate(c);ref=d.reference_row(row)
  assert ref=='B01-s2026092701' and c['As'] and c['updates']==2220 and c['epochs']==10 and c['cumulative_updates']==6660 and c['u_practical']['selected_slots'] is None and c['u_practical']['U_As_packets'] is None and c['u_practical']['slot_fraction'] is None and not c['u_practical']['same_future_Ascount_claim']
  for k,v in [('epochs',50),('updates',4440),('As',False),('parent_source','OTHER'),('row_id','S01-s2026092701'),('model_seed',2026092702)]:
   z=copy.deepcopy(c);z[k]=v;bad(lambda z=z:d.validate(z))
  for k,v in [('mode','QUARTER_PRACTICAL'),('slot_fraction',.25),('U_truth_used',True),('L_changed',True),('extra_NN',1),('same_future_Ascount_claim',True)]:
   z=copy.deepcopy(c);z['u_practical'][k]=v;bad(lambda z=z:d.validate(z))
  cp=dict(epoch=20,a1_ema_successful_updates=4440,run_id=d.REFERENCE_RUN,candidate_id=ref,checkpoint_selection='final_only',ema_model={'mock':1},args=dict(seed=d.SEED,baseline_ckpt='',teacher_ckpt='',epochs=20,use_ema_teacher=True,label_smoothing=.01,batch_size=128))
  verify=lambda v:check_checkpoint(v,run=d.REFERENCE_RUN,row=ref,epoch=20,updates=4440,baseline='')
  verify(cp)
  for k,v in [('epoch',30),('a1_ema_successful_updates',6660),('run_id','other'),('candidate_id',c['row_id']),('checkpoint_selection','target_best'),('ema_model',None)]:
   z=copy.deepcopy(cp);z[k]=v;bad(lambda z=z:verify(z))
 check_default_head((6,160))
 for shape in [(5,160),(6,64),(6,160,1)]:bad(lambda shape=shape:check_default_head(shape))
 assert cumulative(1,1)==(21,4441) and cumulative(10,2220)==(30,6660)
 for ep,step in [(0,0),(1,223),(11,2221),(10,1)]:bad(lambda ep=ep,step=step:cumulative(ep,step))
 boundary=frozen_boundary();routing=routing_checks()
 model_source=Path(__file__).with_name('model.py').read_text(encoding='utf8');assert 'original.hook(self,**kw)' in model_source and '_full_channel=' not in model_source and set(INFO_KEYS)=={'concat/'+k for k in CONCAT_FIELDS} and len(INFO_KEYS)==7
 for p in Path(__file__).parent.glob('*.py'):ast.parse(p.read_text(encoding='utf8'),filename=str(p))
 return dict(status='PASS',protocol_negatives=negative,actual_frozen_native_boundary=boundary,routing=routing,emitted_native_info_keys=7,full6L_original_hook=True,actual_rows=2,unique_original_parents=1,no_new_losses=True,real_model_checkpoint_smoke='launcher first step pending1570NN',NN=0,CP=0,IQ=0,updates=0,target=False)

if __name__=='__main__':
 result=run()
 if len(sys.argv)>1:d.write(sys.argv[1],result)
 print(json.dumps(result))
