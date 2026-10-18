"""Focused new parent/budget/clock/scope checks;unchanged LRaw math is reused."""
import ast,copy,json,sys
from pathlib import Path
from types import SimpleNamespace,MethodType
from contextlib import contextmanager
from . import design as d
from .protocol import cumulative,check_checkpoint,check_source_metadata,check_parent_budget,check_default_head

def run():
 negative=0
 def bad(f):
  nonlocal negative
  try:f()
  except (ValueError,TypeError,KeyError,StopIteration):negative+=1
  else:raise AssertionError('Invalid fixed E50 scope accepted')
 assert d.PARENT_COMMIT=='d14450bf1b0d12cc281df1f4d0e79e8760e48282' and d.ORIGINAL50_COMMIT=='0f399c910324e21a8b1a60346d879be5c2964a9c'
 assert [r['arm'] for r in d.rows()]==['B00','B01']
 for row in d.rows():
  c=d.config(row);d.validate(c);parent=d.parent_config(c)
  assert (c['epochs'],c['label_epochs'],c['pseudo_epochs'],c['updates'],c['epoch_offset'],c['step_offset'],c['cumulative_epochs'],c['cumulative_updates'])==(20,0,20,4440,30,6660,50,11100)
  assert Path(c['parent_source'])==Path(parent['output_root']) and Path(c['parent_source']).parts[-2:]==(row['row_id'],'source') and parent['run_id']==d.PARENT_RUN
  assert c['data_contract']==parent['data_contract'] and c['training_mechanism']['pair_weights']==[.5,.5,.25] and c['As']==(row['arm']=='B01') and c['crossview']['mode']=='ORD128' and c['lraw']['whole_trajectory_NoAug'] is False and c['lraw']['raw_return_epochs_cumulative']==30 and 'source_promotion' not in c
  for k,v in [('epochs',50),('label_epochs',20),('pseudo_epochs',0),('updates',11100),('parent_epochs',20),('parent_source',parent['parent_source']),('model_seed',2026092702),('step_offset',4440),('epoch_offset',20),('As',not c['As']),('target_access',True),('separate_head_training',True)]:
   z=copy.deepcopy(c);z[k]=v;bad(lambda z=z:d.validate(z))
  z=copy.deepcopy(c);z['training_mechanism']['pair_weights']=[.5,.125,.375,.25];bad(lambda:d.validate(z))
 assert cumulative(1,1)==(31,6661) and cumulative(20,4440)==(50,11100)
 for ep,step in [(0,0),(1,223),(21,4441),(20,2220)]:bad(lambda ep=ep,step=step:cumulative(ep,step))
 check_default_head((6,160));bad(lambda:check_default_head((6,64)))
 def args():return SimpleNamespace(epochs=10,label_epochs=0,pseudo_epochs=10,batch_size=128,seed=d.SEED,label_smoothing=.01,use_aug=True,use_unlabeled=True,use_ema_teacher=True,use_concat_sat_channel_aug=False,baseline_ckpt='own20/final_ssdg.pth',teacher_ckpt='',lr=.0002,weight_decay=.0001,from_scratch=False,a1_scratch_only=False,a1_ema_startup_average=False,freeze_backbone=False,phase1_lr_schedule='cosine',phase1_lr_min=.000001)
 old_factory=d.old.make_args;called=[]
 try:
  d.old.make_args=lambda cfg,device:(called.append((copy.deepcopy(cfg),device)) or args())
  for row in d.rows():
   c=d.config(row);a=d.make_args(c,'cpu');assert called[-1]==(d.parent_config(c),'cpu')
   assert (a.epochs,a.label_epochs,a.pseudo_epochs,a.lr,a.weight_decay,a.phase1_lr_min)==(20,0,20,.0002,.0001,.000001) and a.baseline_ckpt==c['parent_source']+'/final_ssdg.pth' and a.use_aug and a.use_unlabeled and a.use_ema_teacher and not a.from_scratch and not a.a1_scratch_only
  for field,value in [('use_aug',False),('use_unlabeled',False),('use_ema_teacher',False),('freeze_backbone',True),('phase1_lr_min',.00001)]:
   def drift(cfg,device,field=field,value=value):a=args();setattr(a,field,value);return a
   d.old.make_args=drift;bad(lambda:d.make_args(d.config(d.rows()[0]),'cpu'))
 finally:d.old.make_args=old_factory
 for row in d.rows():
  for epoch,updates,run_id in [(10,2220,d.PARENT_RUN),(20,4440,d.RUN)]:
   a=vars(args()).copy();a.update(epochs=epoch,pseudo_epochs=epoch,baseline_ckpt='OWN/final_ssdg.pth')
   cp=dict(epoch=epoch,a1_ema_successful_updates=updates,run_id=run_id,candidate_id=row['row_id'],checkpoint_selection='final_only',ema_model={'own':1},args=a)
   def verify(v):check_checkpoint(v,run=run_id,row=row['row_id'],epoch=epoch,updates=updates,baseline='OWN/final_ssdg.pth')
   verify(cp)
   for k,v in [('epoch',50),('a1_ema_successful_updates',11100),('run_id',d.REFERENCE_RUN),('candidate_id','other'),('checkpoint_selection','target_best'),('ema_model',None)]:
    z=copy.deepcopy(cp);z[k]=v;bad(lambda z=z:verify(z))
   for k,v in [('seed',2026092702),('teacher_ckpt','external'),('baseline_ckpt','wrong/final_ssdg.pth'),('use_aug',False),('use_unlabeled',False),('use_ema_teacher',False),('pseudo_epochs',0)]:
    z=copy.deepcopy(cp);z['args'][k]=v;bad(lambda z=z:verify(z))
 parent=d.parent_config(d.config(d.rows()[0]));result=dict(cumulative_epochs=30,cumulative_optimizer_steps=6660,parent_epochs=20,parent_optimizer_steps=4440,U_truth_read=False,target_access=False,head_fit_updates=0,parent_EMA_restored=True,lraw=dict(mode='ordinary_draw_discard_return_raw'));check_parent_budget(parent,result)
 for k,v in [('cumulative_epochs',10),('cumulative_optimizer_steps',2220),('parent_epochs',30),('parent_optimizer_steps',6660),('target_access',True),('parent_EMA_restored',False)]:
  z=copy.deepcopy(result);z[k]=v;bad(lambda z=z:check_parent_budget(parent,z))
 z=copy.deepcopy(parent);z['run_id']=d.REFERENCE_RUN;bad(lambda:check_parent_budget(z,result))
 expected=dict(checkpoint_sources=['parent/final_ssdg.pth'],parent_source='parent');done=dict(config=expected,epoch=20,optimizer_steps=4440,target_access=False,target_evaluated=False,head_fit_updates=0);frozen=dict(config=expected,selection='final_only',target_access=False);init=dict(target_contact=False,checkpoint_sources=expected['checkpoint_sources'],scratch_only=False,ancestors=['parent']);contract=dict(role_ids={'L':['fixed']})
 def metadata(dd=done,ss=contract):check_source_metadata(dd,frozen,init,ss,expected,contract['role_ids'],20,4440)
 metadata()
 for k,v in [('epoch',10),('optimizer_steps',2220),('target_access',True),('head_fit_updates',1)]:
  z=copy.deepcopy(done);z[k]=v;bad(lambda z=z:metadata(dd=z))
 bad(lambda:metadata(ss=dict(role_ids={'L':['wrong']})))
 # Execute the actual new hook around the unchanged old hook interface.
 tree=ast.parse(Path(__file__).with_name('model.py').read_text(encoding='utf8'));fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='hook')
 def old_hook(owner,**kw):owner._lraw_records.append(dict(step=kw['optimizer_step'],cumulative_step=4440+kw['optimizer_step'],cumulative_epoch=21));return 1.25,{'concat/total_L':1.25}
 env=dict(old=SimpleNamespace(hook=old_hook));exec(compile(ast.Module(body=[fn],type_ignores=[]),'actual_new50_hook','exec'),env)
 for step,epoch in [(1,31),(4440,50)]:
  owner=SimpleNamespace(_lraw_records=[]);loss,info=env['hook'](owner,optimizer_step=step);assert loss==1.25 and info=={'concat/total_L':1.25} and owner._lraw_records[-1]['cumulative_step']==6660+step and owner._lraw_records[-1]['cumulative_epoch']==epoch
 # Actual scope restores ownLRaw30 EMA,but uses original20 augargs/base/frozen13 and SupCon+30.
 @contextmanager
 def opt_scope(owner,units):assert units=='CENTER_UNITS';yield
 calls=[];writes=[];loaded=[];model=object();fake_ema=SimpleNamespace(id_backbone=SimpleNamespace(encoder=SimpleNamespace(_full_train_packets=0,_full_eval_packets=2)),load_state_dict=lambda state,strict:loaded.append((state,strict)))
 native=SimpleNamespace(deepcopy=lambda v,*a,**kw:fake_ema,configure_augmentor_for_epoch=lambda obj,cfg,epoch,args:calls.append((cfg,epoch,args.epochs)),_resolve_phase1_terminal_status=lambda **kw:kw,make_augmentor=lambda cfg:cfg,build_aug_base_cfg=lambda args:{'epochs':args.epochs})
 originals=dict(vars(native));sup=SimpleNamespace(schedule=lambda epoch:epoch)
 def wrap_factory(original,records):
  def make(*a,**kw):return original(*a,**kw)
  make.calls=0;return make
 owner=SimpleNamespace(parent={'args':{'epochs':10},'ema_model':{'ownLRaw30':True}},scratch_args={'epochs':20},ordinary_records=[],model=model,out=Path('SYNTHETIC'))
 ot=ast.parse(Path(__file__).with_name('observer.py').read_text(encoding='utf8'));cls=next(n for n in ot.body if isinstance(n,ast.ClassDef) and n.name=='Observer');scope=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='native_scope');scope.name='observer_scope'
 oe=dict(contextmanager=contextmanager,SimpleNamespace=SimpleNamespace,ordinary_factory=wrap_factory,supcon=sup,native_scope=opt_scope,d=SimpleNamespace(write=lambda p,v:writes.append(v)))
 exec(compile(ast.Module(body=[scope],type_ignores=[]),'actual_new50_observer_scope','exec'),oe)
 with oe['observer_scope'](owner,native):
  native.configure_augmentor_for_epoch('AUG',{'epochs':20},1,args());native.deepcopy(model)
  assert calls==[({'epochs':20},13,20)] and sup.schedule(1)==31 and sup.schedule(20)==50 and loaded==[({'ownLRaw30':True},True)] and writes[-1]['origin']=='strict own LRaw FINAL30 EMA'
 assert all(getattr(native,k) is v for k,v in originals.items())
 # Channel differs from the known original E50 route only by this namespace import;no quota/newnoise/HW rule.
 text=Path(__file__).with_name('channel.py').read_text(encoding='utf8');assert 'epoch not in range(31,51)' in text and 'step not in range(6661,11101)' in text and "namespace=f'FcrossRX/train/e{epoch}/step{step}/L_S'" in text and "rng.choice(d.TRAIN_SCENES)" in text
 aug=Path(__file__).with_name('augmentation.py').read_text(encoding='utf8');assert 'from experiments.cvs_lraw_tail30.augmentation import Wrapper,draw_and_discard,ordinary_factory' in aug
 from . import report
 labels=[x[0] for x in report.logical_rows()];assert set(labels)=={'NOAS30','AS30','NOAS50','AS50','LRAW_NOAS30','LRAW_AS30','LRAW_NOAS50','LRAW_AS50'}
 import inspect
 assert 'promotion=' not in inspect.getsource(report.source_decision) and 'source_extension_descriptive_only=True' in inspect.getsource(report.source_decision)
 for file in Path(__file__).parent.glob('*.py'):ast.parse(file.read_text(encoding='utf8'),filename=str(file))
 return dict(status='PASS',protocol_negatives=negative,parent_native10_cumulative30_distinguished=True,only_ownLRaw30_parent_allowed=True,original20_augargs_base_frozen13=True,ownLRaw30_EMA_restored=True,new20_resetAdamW_cos20=True,clock_E31_50_step6661_11100=True,unchanged_LRaw30_factory_reused=True,unchanged_original_E50_physics_window=True,eight_logical_source_descriptive_no_new_promotion=True,real_checkpoint_factory_smoke='launcher first step pending1540NN',NN=0,CP=0,IQ=0,updates=0,target=False)
if __name__=='__main__':print(json.dumps(run()))
