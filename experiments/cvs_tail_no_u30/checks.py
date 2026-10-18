"""Narrow protocol negatives and actual native control-flow doubles;zero NN/IQ/CP."""
import ast,copy,json,sys,types
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from . import design as d
from .policy import NoUTraversal,validate_no_u_args,U_WEIGHTS,TEACHER_WEIGHTS
from .protocol import cumulative,check_checkpoint,check_source_metadata,check_default_head
from .schema import INFO_KEYS,CONCAT_FIELDS,NO_U_FIELDS

def run():
 negative=0
 def bad(f):
  nonlocal negative
  try:f()
  except (ValueError,TypeError,KeyError,StopIteration,RuntimeError):negative+=1
  else:raise AssertionError('Invalid tailNoU scope accepted')
 assert [r['arm'] for r in d.rows()]==['B00','B01']
 for row in d.rows():
  c=d.config(row);d.validate(c)
  assert (c['epochs'],c['label_epochs'],c['pseudo_epochs'],c['updates'],c['cumulative_epochs'],c['cumulative_updates'])==(10,0,10,2220,30,6660)
  assert c['data_contract']==dict(RX=[1,3,4,6,8],days=[1,2,3],L=6300,U=56700,V=27000,split=392005,source_RX_holdout=None,internal_L_split=False,U_TX_truth_hidden=True)
  assert c['As']==(row['arm']=='B01') and c['training_mechanism']['pair_weights']==[.5,.5,.25] and c['crossview']['mode']=='ORD128' and c['features']==[]
  assert d.reference_config(row,20)['arm']==row['arm'] and Path(c['parent_source'])==Path(d.reference_config(row,30)['parent_source'])
  for k,v in [('target_access',True),('epochs',30),('updates',6660),('parent_epochs',30),('step_offset',0),('model_seed',2026092702),('baseline_reuse',True),('Ag_mode','WHITE'),('As',not c['As']),('label_epochs',10),('pseudo_epochs',0),('parent_source','unknown'),('separate_head_training',True)]:
   z=copy.deepcopy(c);z[k]=v;bad(lambda z=z:d.validate(z))
  for key,value in [('mode','MIXED128'),('additional_anchors',64)]:
   z=copy.deepcopy(c);z['crossview'][key]=value;bad(lambda z=z:d.validate(z))
 bad(lambda:d.config(d.rows()[0],20));bad(lambda:d.config(dict(arm='OTHER',row_id='OTHER-s2026092701',model_seed=d.SEED)))
 check_default_head((6,160))
 for shape in [(5,160),(6,64),(6,160,1)]:bad(lambda shape=shape:check_default_head(shape))
 assert cumulative(1,1)==(21,4441) and cumulative(10,2220)==(30,6660)
 for ep,step in [(0,0),(1,223),(11,2221),(10,1)]:bad(lambda ep=ep,step=step:cumulative(ep,step))
 def original_args():
  return SimpleNamespace(epochs=10,label_epochs=0,pseudo_epochs=10,batch_size=128,label_smoothing=.01,seed=d.SEED,use_aug=True,use_concat_sat_channel_aug=False,use_ema_teacher=True,use_unlabeled=True,lambda_u=.16,lambda_ent=.01,baseline_ckpt='parent/final_ssdg.pth',teacher_ckpt='',from_scratch=False,a1_scratch_only=False,use_muse_ssdg=False,use_adv3b02_daot_stn=False,fasttrust_rc4=False,sat_anchor_ssl=False,freeze_backbone=False,phase1_lr_schedule='cosine')
 # Execute actual new make_args around a pinned-reference factory double.
 names=('experiments.cvs_noise_shape_factorial43','experiments.cvs_noise_shape_factorial43.design');saved={k:sys.modules.get(k) for k in names};package=types.ModuleType(names[0]);base=types.ModuleType(names[1]);factory_calls=[]
 base.rows=d.rows;base.config=lambda row,segment:d.reference_config(row,segment)
 def factory(cfg,device):factory_calls.append((copy.deepcopy(cfg),device));return original_args()
 base.make_args=factory;package.design=base
 try:
  sys.modules[names[0]]=package;sys.modules[names[1]]=base
  for row in d.rows():
   c=d.config(row);args=d.make_args(c,'cpu');validate_no_u_args(args)
   assert (args.label_epochs,args.pseudo_epochs,args.use_unlabeled,args.use_ema_teacher,args.lambda_u,args.lambda_ent)==(0,10,False,False,0.,0.)
   assert factory_calls[-1]==(d.reference_config(row,30),'cpu') and args.baseline_ckpt==c['parent_source']+'/final_ssdg.pth'
  for field in ('use_aug','use_unlabeled','use_ema_teacher'):
   def drift(cfg,device,field=field):a=original_args();setattr(a,field,False);return a
   base.make_args=drift;bad(lambda:d.make_args(d.config(d.rows()[0]),'cpu'))
 finally:
  for k in names:
   if saved[k] is None:sys.modules.pop(k,None)
   else:sys.modules[k]=saved[k]
 child_args=vars(original_args()).copy();child_args.update(label_epochs=0,pseudo_epochs=10,use_unlabeled=False,use_ema_teacher=False,lambda_u=0.,lambda_ent=0.)
 for k,v in [('use_unlabeled',True),('use_ema_teacher',True),('label_epochs',10),('pseudo_epochs',0),('use_muse_ssdg',True),('use_adv3b02_daot_stn',True),('fasttrust_rc4',True),('sat_anchor_ssl',True),('teacher_ckpt','external'),('source_unlabeled_epochs',[]),('freeze_backbone',True),('use_aug',False),('from_scratch',True)]+[(k,.1) for k in U_WEIGHTS+TEACHER_WEIGHTS]:
  a=dict(child_args);a[k]=v;bad(lambda a=a:validate_no_u_args(SimpleNamespace(**a)))
 for row in d.rows():
  p_args=dict(child_args,epochs=20,label_epochs=13,pseudo_epochs=7,baseline_ckpt='',from_scratch=True,a1_scratch_only=True,use_unlabeled=True,use_ema_teacher=True)
  parent=dict(epoch=20,a1_ema_successful_updates=4440,run_id=d.REFERENCE_RUN,candidate_id=row['row_id'],checkpoint_selection='final_only',ema_model={'mock':1},args=p_args)
  def verify_parent(v):return check_checkpoint(v,run=d.REFERENCE_RUN,row=row['row_id'],epoch=20,updates=4440,baseline='')
  verify_parent(parent)
  for k,v in [('epoch',30),('a1_ema_successful_updates',6660),('run_id','other'),('candidate_id','other'),('checkpoint_selection','target_best'),('ema_model',None)]:
   z=copy.deepcopy(parent);z[k]=v;bad(lambda z=z:verify_parent(z))
  for k,v in [('seed',2026092702),('teacher_ckpt','external'),('baseline_ckpt','external'),('use_ema_teacher',False),('use_unlabeled',False),('label_epochs',10)]:
   z=copy.deepcopy(parent);z['args'][k]=v;bad(lambda z=z:verify_parent(z))
  child=dict(epoch=10,a1_ema_successful_updates=2220,run_id=d.RUN,candidate_id=row['row_id'],checkpoint_selection='final_only',ema_model=None,args=dict(child_args))
  def verify_child(v):return check_checkpoint(v,run=d.RUN,row=row['row_id'],epoch=10,updates=2220,baseline='parent/final_ssdg.pth',child=True)
  verify_child(child)
  z=copy.deepcopy(child);z['ema_model']={'not_allowed':1};bad(lambda:verify_child(z))
  z=copy.deepcopy(child);z['a1_ema_successful_updates']=0;bad(lambda:verify_child(z))
  z=copy.deepcopy(child);del z['ema_model'];bad(lambda:verify_child(z))
 expected=dict(checkpoint_sources=['parent/final_ssdg.pth'],parent_source='parent');done=dict(config=expected,epoch=10,optimizer_steps=2220,target_access=False,target_evaluated=False,head_fit_updates=0,tail_U_packets=0,tail_EMA_updates=0);frozen=dict(config=expected,selection='final_only',target_access=False);init=dict(target_contact=False,checkpoint_sources=expected['checkpoint_sources'],scratch_only=False,ancestors=['parent'],teacher_origin='disabled in child;own20 EMA provenance verified');contract=dict(role_ids={'L':['fixed']})
 def metadata(dd=done,ff=frozen,ii=init,ss=contract):check_source_metadata(dd,ff,ii,ss,expected,contract['role_ids'])
 metadata()
 for k,v in [('target_access',True),('target_evaluated',True),('head_fit_updates',1),('optimizer_steps',6660),('tail_U_packets',1),('tail_EMA_updates',1)]:
  z=copy.deepcopy(done);z[k]=v;bad(lambda z=z:metadata(dd=z))
 z=copy.deepcopy(init);z['ancestors']=['unknown'];bad(lambda:metadata(ii=z));bad(lambda:metadata(ss=dict(role_ids={'L':['different']})))
 # Execute the actual native phase/U/epoch-pair helpers;U loader has len but raises on iteration.
 path=d.ROOT/'experiments/adv3b02_xuc/code/SSDG/train_ssdg.py'
 if not path.is_file():
  from experiments.cvs_phase1_overlay.model import native_modules
  path=Path(native_modules().__file__)
 tree=ast.parse(path.read_text(encoding='utf8'));env={}
 names={'_legacy_unlabeled_active','_muse_epoch_pairs','_source_epoch_phase','_resolve_epoch_schedule'}
 exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names],type_ignores=[]),'actual_native_helpers','exec'),env)
 class L:
  def __iter__(self):yield 'L0';yield 'L1'
 class U:
  def __len__(self):return 222
  def __iter__(self):raise AssertionError('Underlying U dataset must never be fetched')
 u=NoUTraversal(U());a=SimpleNamespace(**child_args)
 assert env['_resolve_epoch_schedule'](a)==10
 for epoch in range(1,11):
  phase=env['_source_epoch_phase'](epoch,a);assert phase=='pseudo' and env['_legacy_unlabeled_active'](a,muse_state=None,phase=phase) is False
 pairs=list(env['_muse_epoch_pairs'](L(),u,use_muse=False,use_unlabeled_step_budget=True))
 assert len(pairs)==222 and pairs[:3]==[('L0',None),('L1',None),('L0',None)] and u.iteration_attempts==u.packets==0
 bad(lambda:iter(u));assert u.iteration_attempts==1
 # Actual native EMA create/update/save guards are conditional on a live model.
 native_text=path.read_text(encoding='utf8')
 assert 'if bool(args.use_ema_teacher) or muse_state is not None:' in native_text and '"ema_model": ema_model.state_dict() if ema_model is not None else None' in native_text
 create=next(n for n in ast.walk(tree) if isinstance(n,ast.If) and any(isinstance(x,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='ema_model' for t in x.targets) and isinstance(x.value,ast.Call) and isinstance(x.value.func,ast.Attribute) and x.value.func.attr=='to' for x in n.body))
 create_test=compile(ast.Expression(create.test),'actual_native_ema_create_guard','eval');assert eval(create_test,dict(args=a,muse_state=None)) is False
 # Execute actual observer scope with interface doubles:parent20 augbase/freeze13,+20 schedule,zero teacher routes.
 from contextlib import contextmanager
 @contextmanager
 def optimizer_scope(owner,units):assert units=='CENTER_UNITS';yield
 calls=[];native=SimpleNamespace(deepcopy=lambda v,*a,**kw:('copy',v),configure_augmentor_for_epoch=lambda obj,cfg,epoch,args:calls.append((obj,cfg,epoch,args.epochs)),_resolve_phase1_terminal_status=lambda **kw:kw,_update_ema_model=lambda *a,**kw:None,build_aug_base_cfg=lambda args:{'epochs':args.epochs})
 originals=dict(vars(native));sc=SimpleNamespace(schedule=lambda epoch:epoch);writes=[];owner=SimpleNamespace(parent={'args':{'epochs':20}},out=Path('SYNTHETIC'),teacher_builds=0,EMA_updates=0,model=object())
 ot=ast.parse(Path(__file__).with_name('observer.py').read_text(encoding='utf8'));cl=next(n for n in ot.body if isinstance(n,ast.ClassDef) and n.name=='Observer');fn=next(n for n in cl.body if isinstance(n,ast.FunctionDef) and n.name=='native_scope');oe=dict(contextmanager=contextmanager,SimpleNamespace=SimpleNamespace,supcon=sc,native_scope=optimizer_scope,d=SimpleNamespace(write=lambda p,v:writes.append(v)))
 fn.name='observer_scope';exec(compile(ast.Module(body=[fn],type_ignores=[]),'actual_observer_native_scope','exec'),oe)
 with oe['observer_scope'](owner,native):
  native.configure_augmentor_for_epoch('AUG',{'epochs':10},1,a);assert calls==[('AUG',{'epochs':20},13,20)] and sc.schedule(1)==21
  assert owner.teacher_builds==owner.EMA_updates==0 and writes[-1]['child_ema_model'] is None
  bad(lambda:native.deepcopy(owner.model));bad(lambda:native._update_ema_model(None,None,.999))
 assert all(getattr(native,k) is v for k,v in originals.items()) and sc.schedule(1)==1
 # Hook delegates original equations and emits exactly14 keys without a new forward.
 text=Path(__file__).with_name('model.py').read_text(encoding='utf8');assert 'delta,previous_info=original.hook(self,**kw)' in text and 'capture_forward' not in text and '.forward_identity_only(' not in text
 mt=ast.parse(text);hook=next(n for n in mt.body if isinstance(n,ast.FunctionDef) and n.name=='hook');actual=set();fieldsets=dict(CONCAT_FIELDS=CONCAT_FIELDS,NO_U_FIELDS=NO_U_FIELDS)
 for node in ast.walk(hook):
  if isinstance(node,ast.DictComp) and isinstance(node.key,ast.BinOp) and isinstance(node.key.left,ast.Constant):actual.update(node.key.left.value+k for k in fieldsets[node.generators[0].iter.id])
 assert actual==set(INFO_KEYS) and len(actual)==14 and 'shape/RD' not in actual and .5+.5+.25==1.25
 # Source rule: clean is newAs-newNoAs;midlow is newAs-originalAs;native JSON bool.
 from . import report
 assert report.PAIRS==(('TAIL_NO_U_AS30','TAIL_NO_U_NOAS30'),('TAIL_NO_U_AS30','AS30'),('TAIL_NO_U_NOAS30','NOAS30'),('AS30','NOAS30'))
 old_rows=report.logical_rows;old_read=report.d.read;old_write=report.d.write;old_load=report.np.load
 class Arrays:
  def __init__(self):
   yy=np.arange(1800)%6;rr=np.asarray([1,3,4,6,8])[np.arange(1800)%5];clean=np.eye(6,dtype=np.float32)[yy];self.values=dict(ids=np.asarray(['mock'+str(i) for i in range(1800)]),TX=yy,RX=rr,clean=clean,satellite=np.tile(clean,(2,3,1,1)))
  def __enter__(self):return self.values
  def __exit__(self,*unused):pass
 def read_mock(path):
  name=Path(path).name
  if name=='completion.json':return dict(target_access=False,final_source_metrics=dict(source_val_count=27000,source_val_accuracy=.5 if 'NOAS30' in str(path) and 'TAIL_NO_U_NOAS30' not in str(path) else .98,source_val_rx_accuracy={str(r):.98 for r in (1,3,4,6,8)}))
  if name in ('source_frozen.json','complete.json'):return dict(target_access=False)
  return dict(role_ids={'L':['mock']})
 captured=[]
 try:
  report.logical_rows=lambda:[(name,Path('SYNTHETIC')/name) for name in ('NOAS30','AS30','TAIL_NO_U_NOAS30','TAIL_NO_U_AS30')];report.d.read=read_mock;report.d.write=lambda path,value:captured.append(value);report.np.load=lambda *args,**kwargs:Arrays()
  summary=report.source_decision();json.dumps(summary,allow_nan=False);assert type(summary['promotion']['passed']) is bool and summary['promotion']['passed'] is False and summary['promotion']['newAs_minus_newNoAs_fullV_clean_pp']==0. and len(captured)==1 and summary['scientific_failure_skips_formal'] is False
 finally:report.logical_rows=old_rows;report.d.read=old_read;report.d.write=old_write;report.np.load=old_load
 for file in Path(__file__).parent.glob('*.py'):ast.parse(file.read_text(encoding='utf8'),filename=str(file))
 return dict(status='PASS',protocol_negatives=negative,actual_make_args_kwargs_two_rows=True,actual_native_U_phase_and_epoch_pairs=True,U_loader_only_len_used_222=True,EMA_creation_guard_false=True,actual_observer_parent_aug_freeze13_and_offsets=True,actual_observer_teacher_build_and_update_forbidden=True,parent_EMA_required_child_None_required=True,original_P43_ORD128_6L_delegated=True,total_L_scale=1.25,emitted_native_info_keys=len(actual),source_summary_native_JSON=True,real_Torch_or_checkpoint_smoke='launcher first step pending1540NN',NN=0,CP=0,IQ=0,updates=0,target=False)
if __name__=='__main__':print(json.dumps(run()))
