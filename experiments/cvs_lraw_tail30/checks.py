"""Protocol,actual factory and no-Torch interface checks;no real model/CP/IQ."""
import ast,copy,json,sys,types,random
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from . import design as d
from .protocol import cumulative,check_checkpoint,check_source_metadata,check_default_head
from .schema import INFO_KEYS,CONCAT_FIELDS,LRAW_FIELDS

def run():
 negative=0
 def bad(f):
  nonlocal negative
  try:f()
  except (ValueError,TypeError,KeyError,StopIteration):negative+=1
  else:raise AssertionError('Invalid LRaw scope accepted')
 assert [r['arm'] for r in d.rows()]==['B00','B01']
 for row in d.rows():
  c=d.config(row);d.validate(c)
  assert (c['epochs'],c['label_epochs'],c['pseudo_epochs'],c['updates'],c['cumulative_epochs'],c['cumulative_updates'])==(10,0,10,2220,30,6660)
  assert c['data_contract']==dict(RX=[1,3,4,6,8],days=[1,2,3],L=6300,U=56700,V=27000,split=392005,source_RX_holdout=None,internal_L_split=False,U_TX_truth_hidden=True)
  assert c['As']==(row['arm']=='B01') and c['training_mechanism']['pair_weights']==[.5,.5,.25] and c['crossview']['mode']=='ORD128'
  assert d.reference_config(row,20)['arm']==row['arm'] and Path(c['parent_source'])==Path(d.reference_config(row,30)['parent_source']) and c['lraw']['whole_trajectory_NoAug'] is False
  for k,v in [('target_access',True),('epochs',30),('updates',6660),('parent_epochs',30),('step_offset',0),('model_seed',2026092702),('baseline_reuse',True),('Ag_mode','WHITE'),('As',not c['As']),('label_epochs',10),('pseudo_epochs',0),('parent_source','unknown'),('separate_head_training',True)]:
   z=copy.deepcopy(c);z[k]=v;bad(lambda z=z:d.validate(z))
  z=copy.deepcopy(c);z['crossview']['mode']='MIXED128';bad(lambda:d.validate(z))
  z=copy.deepcopy(c);z['lraw']['full_native_aug_computation']=False;bad(lambda:d.validate(z))
 bad(lambda:d.config(d.rows()[0],20));bad(lambda:d.config(dict(arm='OTHER',row_id='OTHER-s2026092701',model_seed=d.SEED)))
 check_default_head((6,160))
 for shape in [(5,160),(6,64),(6,160,1)]:bad(lambda shape=shape:check_default_head(shape))
 assert cumulative(1,1)==(21,4441) and cumulative(10,2220)==(30,6660)
 for ep,step in [(0,0),(1,223),(11,2221),(10,1)]:bad(lambda ep=ep,step=step:cumulative(ep,step))
 def original_args():return SimpleNamespace(epochs=10,label_epochs=0,pseudo_epochs=10,batch_size=128,label_smoothing=.01,seed=d.SEED,use_aug=True,use_concat_sat_channel_aug=False,use_ema_teacher=True,use_unlabeled=True,lambda_u=.16,lambda_ent=.01,baseline_ckpt='parent/final_ssdg.pth',teacher_ckpt='',from_scratch=False,a1_scratch_only=False,freeze_backbone=False,phase1_lr_schedule='cosine')
 names=('experiments.cvs_noise_shape_factorial43','experiments.cvs_noise_shape_factorial43.design');saved={k:sys.modules.get(k) for k in names};package=types.ModuleType(names[0]);base=types.ModuleType(names[1]);factory_calls=[]
 base.rows=d.rows;base.config=lambda row,segment:d.reference_config(row,segment)
 def factory(cfg,device):factory_calls.append((copy.deepcopy(cfg),device));return original_args()
 base.make_args=factory;package.design=base
 try:
  sys.modules[names[0]]=package;sys.modules[names[1]]=base
  for row in d.rows():
   c=d.config(row);args=d.make_args(c,'cpu');assert factory_calls[-1]==(d.reference_config(row,30),'cpu')
   assert (args.label_epochs,args.pseudo_epochs,args.use_aug,args.use_unlabeled,args.use_ema_teacher,args.lambda_u,args.lambda_ent)==(0,10,True,True,True,.16,.01)
  for field in ('use_aug','use_ema_teacher','use_unlabeled'):
   def drift(cfg,device,field=field):a=original_args();setattr(a,field,False);return a
   base.make_args=drift;bad(lambda:d.make_args(d.config(d.rows()[0]),'cpu'))
 finally:
  for k in names:
   if saved[k] is None:sys.modules.pop(k,None)
   else:sys.modules[k]=saved[k]
 for row in d.rows():
  for epoch,updates,run_id,baseline in [(20,4440,d.REFERENCE_RUN,''),(10,2220,d.RUN,'parent/final_ssdg.pth')]:
   a=vars(original_args()).copy();a.update(epochs=epoch,label_epochs=13 if epoch==20 else 0,pseudo_epochs=7 if epoch==20 else 10,baseline_ckpt=baseline)
   cp=dict(epoch=epoch,a1_ema_successful_updates=updates,run_id=run_id,candidate_id=row['row_id'],checkpoint_selection='final_only',ema_model={'mock':1},args=a)
   def verify(v):return check_checkpoint(v,run=run_id,row=row['row_id'],epoch=epoch,updates=updates,baseline=baseline)
   verify(cp)
   for k,v in [('epoch',30),('a1_ema_successful_updates',6660),('run_id','other'),('candidate_id','other'),('checkpoint_selection','target_best'),('ema_model',None)]:
    z=copy.deepcopy(cp);z[k]=v;bad(lambda z=z:verify(z))
   for k,v in [('seed',2026092702),('teacher_ckpt','external'),('baseline_ckpt','external'),('use_aug',False),('use_unlabeled',False),('use_ema_teacher',False)]:
    z=copy.deepcopy(cp);z['args'][k]=v;bad(lambda z=z:verify(z))
 expected=dict(checkpoint_sources=['parent/final_ssdg.pth'],parent_source='parent');done=dict(config=expected,epoch=10,optimizer_steps=2220,target_access=False,target_evaluated=False,head_fit_updates=0);frozen=dict(config=expected,selection='final_only',target_access=False);init=dict(target_contact=False,checkpoint_sources=expected['checkpoint_sources'],scratch_only=False,ancestors=['parent']);contract=dict(role_ids={'L':['fixed']})
 def metadata(dd=done,ff=frozen,ii=init,ss=contract):check_source_metadata(dd,ff,ii,ss,expected,contract['role_ids'])
 metadata()
 for k,v in [('target_access',True),('target_evaluated',True),('head_fit_updates',1),('optimizer_steps',6660)]:
  z=copy.deepcopy(done);z[k]=v;bad(lambda z=z:metadata(dd=z))
 z=copy.deepcopy(init);z['ancestors']=['unknown'];bad(lambda:metadata(ii=z));bad(lambda:metadata(ss=dict(role_ids={'L':['different']})))
 # Execute the actual wrapper with a mutating Tensor interface double;Python/NumPy RNG endstates matched.
 class Tensor(np.ndarray):
  def detach(self):return self
  def clone(self):return self.copy().view(Tensor)
  def square(self):return np.square(self).view(Tensor)
  def flatten(self,dim=0):return self.reshape((len(self),-1)).view(Tensor) if dim==1 else self.reshape(-1).view(Tensor)
 fake_torch=SimpleNamespace(float32=np.float32,isfinite=np.isfinite);old_torch=sys.modules.get('torch');aug_name=__package__+'.augmentation';old_aug=sys.modules.get(aug_name)
 calls=[];aux_rng=random.Random(901)
 class Aug:
  def __init__(self,identity=False):self.identity=identity;self.p_pa=.14
  def __call__(self,x,*,labels,no_pa):
   assert no_pa is False and labels.shape==(128,);calls.append((len(x),no_pa));value=random.random()+float(np.random.random())+aux_rng.random()
   if not self.identity:x+=value
   return x
 try:
  sys.modules['torch']=fake_torch;sys.modules.pop(aug_name,None)
  import importlib
  aug_module=importlib.import_module(aug_name);x=np.ones((128,2,256),np.float32).view(Tensor);y=(np.arange(128)%6).view(Tensor);initial=x.clone()
  for identity in (False,True):
   aug=Aug(identity);records=[];make=aug_module.ordinary_factory(lambda cfg:aug,records);wrapper=make({'mock':True});wrapper.p_pa=.2;assert aug.p_pa==.2 and make.calls==1
   random.seed(902);np.random.seed(903);aux_rng.seed(904);r0=random.getstate();n0=np.random.get_state();t0=aux_rng.getstate()
   generated=aug(x.clone(),labels=y,no_pa=False);r1=random.getstate();n1=np.random.get_state();t1=aux_rng.getstate()
   random.setstate(r0);np.random.set_state(n0);aux_rng.setstate(t0);actual=wrapper(x,labels=y,no_pa=False);n2=np.random.get_state()
   assert random.getstate()==r1 and n1[0]==n2[0] and np.array_equal(n1[1],n2[1]) and n1[2:]==n2[2:] and aux_rng.getstate()==t1
   assert np.array_equal(actual,initial) and np.array_equal(x,initial) and records[0]['returned_raw_MSE']==0. and records[0]['discardedordinary_MSE']==float((generated-initial).square().mean()) and records[0]['full_native_draws']==1 and records[0]['count']==128
   if identity:assert records[0]['discardedordinary_MSE']==0. and records[0]['discardedordinary_changed_packets']==0
   for kw in ({'labels':None,'no_pa':False},{'labels':y,'no_pa':True},{'labels':y,'no_pa':False,'return_pa_strength':True}):bad(lambda kw=kw:wrapper(x,**kw))
   bad(lambda:wrapper(x,y,no_pa=False));bad(lambda:wrapper(x[:127],labels=y[:127],no_pa=False));bad(lambda:wrapper(x,labels=(y+7).view(Tensor),no_pa=False));bad(lambda:make({'second':True}))
  # Execute actual model hook around original-hook double;no new neural computation.
  tree=ast.parse(Path(__file__).with_name('model.py').read_text(encoding='utf8'));hook=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='hook');env=dict(original=SimpleNamespace(hook=lambda self,**kw:(2.,{'old':0})),CONCAT_FIELDS=CONCAT_FIELDS,LRAW_FIELDS=LRAW_FIELDS)
  exec(compile(ast.Module(body=[hook],type_ignores=[]),'actual_LRaw_hook','exec'),env)
  rec=dict(total_L=1.25,common_CE=1.,common_KL=.01,SupCon=.2,SupCon_weight=.1,L_forwards=6,As_packets=64,crossview_mode='ORD128',ordinary_raw_MSE=0.)
  owner=SimpleNamespace(_full_records=[rec],_lraw_ordinary_records=[records[0]],_lraw_records=[]);loss,info=env['hook'](owner,optimizer_step=1)
  assert loss==2. and set(info)==set(INFO_KEYS) and len(info)==14 and info['concat/total_L']==1.25 and info['lraw/ordinary_packets']==128
  rec['ordinary_raw_MSE']=.01;bad(lambda:env['hook'](owner,optimizer_step=1))
 finally:
  if old_torch is None:sys.modules.pop('torch',None)
  else:sys.modules['torch']=old_torch
  if old_aug is None:sys.modules.pop(aug_name,None)
  else:sys.modules[aug_name]=old_aug
 # Actual native call sites are L-only;U strong uses separate function,not ordinary object.
 native_path=d.ROOT/'experiments/adv3b02_xuc/code/SSDG/train_ssdg.py'
 if not native_path.is_file():
  from experiments.cvs_phase1_overlay.model import native_modules
  native_path=Path(native_modules().__file__)
 native_tree=ast.parse(native_path.read_text(encoding='utf8'));ordinary_calls=[n for n in ast.walk(native_tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='augmentor']
 assert len(ordinary_calls)==2 and all(isinstance(n.args[0],ast.Name) and n.args[0].id=='x_l' and any(k.arg=='labels' and isinstance(k.value,ast.Name) and k.value.id=='y_l' for k in n.keywords) for n in ordinary_calls)
 strong=next(n for n in native_tree.body if isinstance(n,ast.FunctionDef) and n.name=='_strong_augment');assert not any(isinstance(n,ast.Name) and n.id=='augmentor' for n in ast.walk(strong))
 text=Path(__file__).with_name('model.py').read_text(encoding='utf8');assert 'delta,previous_info=original.hook(self,**kw)' in text and 'capture_forward' not in text and '.forward_identity_only(' not in text and .5+.5+.25==1.25
 from . import report
 assert report.PAIRS==(('LRAW_AS30','LRAW_NOAS30'),('LRAW_AS30','AS30'),('LRAW_NOAS30','NOAS30'),('AS30','NOAS30'))
 old_rows=report.logical_rows;old_read=report.d.read;old_write=report.d.write;old_load=report.np.load
 class Arrays:
  def __init__(self):
   yy=np.arange(1800)%6;rr=np.asarray([1,3,4,6,8])[np.arange(1800)%5];clean=np.eye(6,dtype=np.float32)[yy];self.values=dict(ids=np.asarray(['mock'+str(i) for i in range(1800)]),TX=yy,RX=rr,clean=clean,satellite=np.tile(clean,(2,3,1,1)))
  def __enter__(self):return self.values
  def __exit__(self,*unused):pass
 def read_mock(path):
  name=Path(path).name
  if name=='completion.json':return dict(target_access=False,final_source_metrics=dict(source_val_count=27000,source_val_accuracy=.5 if 'NOAS30' in str(path) and 'LRAW_NOAS30' not in str(path) else .98,source_val_rx_accuracy={str(r):.98 for r in (1,3,4,6,8)}))
  if name in ('source_frozen.json','complete.json'):return dict(target_access=False)
  return dict(role_ids={'L':['mock']})
 captured=[]
 try:
  report.logical_rows=lambda:[(name,Path('SYNTHETIC')/name) for name in ('NOAS30','AS30','LRAW_NOAS30','LRAW_AS30')];report.d.read=read_mock;report.d.write=lambda path,value:captured.append(value);report.np.load=lambda *args,**kwargs:Arrays()
  summary=report.source_decision();json.dumps(summary,allow_nan=False);assert type(summary['promotion']['passed']) is bool and summary['promotion']['passed'] is False and summary['promotion']['newAs_minus_newNoAs_fullV_clean_pp']==0. and len(captured)==1 and summary['scientific_failure_skips_formal'] is False
 finally:report.logical_rows=old_rows;report.d.read=old_read;report.d.write=old_write;report.np.load=old_load
 for file in Path(__file__).parent.glob('*.py'):ast.parse(file.read_text(encoding='utf8'),filename=str(file))
 return dict(status='PASS',protocol_negatives=negative,actual_make_args_U_EMA_ordinary_preserved=True,actual_wrapper_factory_and_mutating_input_clone_mock=True,wrapper_same_call_Python_NumPy_and_rng_double_endstates=True,identity_discard_zero_MSE_accepted=True,actual_model_hook14keys=True,actual_native_ordinary_L_only_callsites=True,U_strong_function_not_ordinary=True,original_P43_ORD128_6L_delegated=True,total_L_scale=1.25,source_summary_native_JSON=True,real_Torch_and_checkpoint_smoke='launcher first step pending1540NN;actual samecall Torch RNG pending',NN=0,CP=0,IQ=0,updates=0,target=False)
if __name__=='__main__':print(json.dumps(run()))
