"""Fixed scope,exact scene quotas and actual downstream interface mocks;no model/IQ."""
import ast,copy,json,sys,types
from pathlib import Path
import numpy as np
from . import design as d
from .protocol import cumulative,check_checkpoint,check_source_metadata,check_default_head
from .schema import INFO_KEYS,CONCAT_FIELDS,BALANCE_FIELDS
from .assignment import allocation,SCENES,NAMESPACE

def run():
 negative=0
 def bad(f):
  nonlocal negative
  try:f()
  except (ValueError,TypeError,KeyError,StopIteration):negative+=1
  else:raise AssertionError('Invalid fixed scene balance scope accepted')
 assert [r['arm'] for r in d.rows()]==['B01']
 for row in d.rows():
  c=d.config(row);d.validate(c)
  assert (c['epochs'],c['updates'],c['cumulative_epochs'],c['cumulative_updates'])==(10,2220,30,6660)
  assert c['data_contract']==dict(RX=[1,3,4,6,8],days=[1,2,3],L=6300,U=56700,V=27000,split=392005,source_RX_holdout=None,internal_L_split=False,U_TX_truth_hidden=True)
  assert c['As'] is True and c['training_mechanism']['pair_weights']==[.5,.5,.25] and c['crossview']['mode']=='ORD128' and c['scene_balance']['physical_ID_uniqueness_required'] is False
  assert d.reference_config(row,20)['arm']==row['arm'] and Path(c['parent_source'])==Path(d.reference_config(row,30)['parent_source'])
  for k,v in [('target_access',True),('epochs',30),('updates',6660),('parent_epochs',30),('step_offset',0),('model_seed',2026092702),('baseline_reuse',True),('Ag_mode','WHITE'),('As',not c['As']),('label_epochs',13),('parent_source','unknown'),('separate_head_training',True)]:
   z=copy.deepcopy(c);z[k]=v;bad(lambda z=z:d.validate(z))
  z=copy.deepcopy(c);z['crossview']['mode']='MIXED128';bad(lambda:d.validate(z))
 bad(lambda:d.config(d.rows()[0],20));bad(lambda:d.config(dict(arm='OTHER',row_id='OTHER-s2026092701',model_seed=d.SEED)))
 check_default_head((6,160))
 for shape in [(5,160),(6,64),(6,160,1)]:bad(lambda shape=shape:check_default_head(shape))
 assert cumulative(1,1)==(21,4441) and cumulative(10,2220)==(30,6660)
 for ep,step in [(0,0),(1,223),(11,2221),(10,1)]:bad(lambda ep=ep,step=step:cumulative(ep,step))
 for row in d.rows():
  cp=dict(epoch=20,a1_ema_successful_updates=4440,run_id=d.REFERENCE_RUN,candidate_id=row['row_id'],checkpoint_selection='final_only',ema_model={'mock':1},args=dict(seed=d.SEED,baseline_ckpt='',teacher_ckpt='',epochs=20,use_ema_teacher=True,label_smoothing=.01,batch_size=128))
  def verify(v):return check_checkpoint(v,run=d.REFERENCE_RUN,row=row['row_id'],epoch=20,updates=4440,baseline='')
  verify(cp)
  for k,v in [('epoch',30),('a1_ema_successful_updates',6660),('run_id','other'),('candidate_id','other'),('checkpoint_selection','target_best'),('ema_model',None)]:
   z=copy.deepcopy(cp);z[k]=v;bad(lambda z=z:verify(z))
  for k,v in [('seed',2026092702),('teacher_ckpt','external'),('baseline_ckpt','external')]:
   z=copy.deepcopy(cp);z['args'][k]=v;bad(lambda z=z:verify(z))
 expected=dict(checkpoint_sources=['parent/final_ssdg.pth'],parent_source='parent');done=dict(config=expected,epoch=10,optimizer_steps=2220,target_access=False,target_evaluated=False,head_fit_updates=0);frozen=dict(config=expected,selection='final_only',target_access=False);init=dict(target_contact=False,checkpoint_sources=expected['checkpoint_sources'],scratch_only=False,ancestors=['parent']);contract=dict(role_ids={'L':['fixed']})
 def metadata(dd=done,ff=frozen,ii=init,ss=contract):check_source_metadata(dd,ff,ii,ss,expected,contract['role_ids'])
 metadata()
 for k,v in [('target_access',True),('target_evaluated',True),('head_fit_updates',1),('optimizer_steps',6660)]:
  z=copy.deepcopy(done);z[k]=v;bad(lambda z=z:metadata(dd=z))
 z=copy.deepcopy(init);z['ancestors']=['unknown'];bad(lambda:metadata(ii=z));bad(lambda:metadata(ss=dict(role_ids={'L':['different']})))
 # Private permutation,exact perstep/perthree/epoch/tail totals;no globalRNG draw.
 import hashlib,random
 def stable_seed(seed,*parts):return int.from_bytes(hashlib.sha256(json.dumps([int(seed),*parts],ensure_ascii=False,separators=(',',':')).encode('utf8')).digest()[:8],'little')
 def ns(step):return f'FcrossRX/train/e{(step-1)//222+1}/step{step}/L_S'
 before=np.random.get_state();python_before=random.getstate();plans=[allocation(step,ns(step),d.CHANNEL_SEEDS['train'],stable_seed) for step in range(4441,6661)];after=np.random.get_state()
 assert before[0]==after[0] and np.array_equal(before[1],after[1]) and before[2:]==after[2:] and random.getstate()==python_before
 for step,p in zip(range(4441,6661),plans):
  counts=np.bincount(p['scene_indices'],minlength=3);expected=np.full(3,21);expected[(step-1)%3]+=1
  assert np.array_equal(counts,expected) and p['planned_counts']==expected.tolist() and sorted(p['permutation'])==list(range(64))
  assert p==allocation(step,ns(step),d.CHANNEL_SEEDS['train'],stable_seed)
 for offset in range(0,2220,3):assert np.sum([np.bincount(p['scene_indices'],minlength=3) for p in plans[offset:offset+3]],axis=0).tolist()==[64]*3
 for offset in range(0,2220,222):assert np.sum([np.bincount(p['scene_indices'],minlength=3) for p in plans[offset:offset+222]],axis=0).tolist()==[4736]*3
 assert np.sum([np.bincount(p['scene_indices'],minlength=3) for p in plans],axis=0).tolist()==[47360]*3
 for step,namespace,count in [(4440,ns(4440),64),(6661,ns(6661),64),(4441,'source_eval/x',64),(4441,ns(4441),63),(4441,ns(4441),0)]:bad(lambda step=step,namespace=namespace,count=count:allocation(step,namespace,d.CHANNEL_SEEDS['train'],stable_seed,count=count))
 # Execute the actual downstream apply function with instrumented interface doubles.
 from collections import Counter
 from dataclasses import dataclass,asdict
 from types import SimpleNamespace
 @dataclass(frozen=True)
 class Hardware:session_id:str;offset:float=0.
 class Tensor(np.ndarray):
  def detach(self):return self
  def cpu(self):return self
  def to(self,*args,**kw):return self
  @property
  def device(self):return 'cpu'
 calls=[]
 class Residual:
  def __init__(self,cfg,seed,hw):self.cfg=cfg;self.seed=seed;self.hw=hw;calls.append(('ResidualChannel',cfg,seed,hw))
  def process(self,x):calls.append(('process',x.copy()));return x,dict(channel_equalization_applied=False,state_end='LOS',receiver_lock_status='locked')
 def norm(x):calls.append(('normalize_input',x.copy()));r=float(np.sqrt(np.mean(abs(x)**2)));return x/r,r
 def hardware(cfg,seed,session):calls.append(('receiver_for_session',cfg,seed,session));return Hardware(session)
 def seeded(seed,*parts):calls.append(('stable_seed',seed,parts));return stable_seed(seed,*parts)
 fake=torch=SimpleNamespace(isfinite=np.isfinite,float32=np.float32,frombuffer=lambda b,dtype:np.frombuffer(b,dtype=dtype).view(Tensor))
 tree=ast.parse(Path(__file__).with_name('channel.py').read_text(encoding='utf8'));cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Practical');fun=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='apply')
 env=dict(np=np,torch=fake,d=d,allocation=allocation,NAMESPACE=NAMESPACE,Counter=Counter,asdict=asdict,serial=lambda v:v)
 exec(compile(ast.Module(body=[fun],type_ignores=[]),'actual_Practical_apply','exec'),env)
 def state():return SimpleNamespace(ch=SimpleNamespace(stable_seed=seeded,receiver_for_session=hardware,normalize_input=norm),residual=SimpleNamespace(ResidualChannel=Residual),configs={s:object() for s in SCENES},sessions={},coverage=Counter(),calls=0,packets=0,scene_balance_records=[])
 x=np.arange(64*2*256,dtype=float).reshape(64,2,256).view(Tensor)+1.;x=x.view(Tensor)
 meta=dict(eq_i=np.ones(64,int),rx_i=np.ones(64,int),day_i=np.ones(64,int),sig_i=np.arange(64)+100,base_index=np.arange(64));meta['sig_i'][1]=meta['sig_i'][0];meta['base_index'][1]=meta['base_index'][0]
 obj=state();result,records=env['apply'](obj,x,meta,21,4441);r=obj.scene_balance_records[-1]
 assert result.shape==(64,2,256) and obj.calls==1 and obj.packets==64 and len(records)==64 and r['sample_ids'][0]==r['sample_ids'][1] and len(set(r['sample_ids']))==63
 assert r['scene_counts']==dict(zip(SCENES,[22,21,21])) and r['input_positions']==list(range(64)) and r['extra_NN']==0
 assert sum(c[0]=='receiver_for_session' for c in calls)==sum(c[0]=='normalize_input' for c in calls)==sum(c[0]=='ResidualChannel' for c in calls)==sum(c[0]=='process' for c in calls)==64
 assert sum(c[0]=='stable_seed' for c in calls)==65
 propagation=[c for c in calls if c[0]=='ResidualChannel'];normalized=[c[1] for c in calls if c[0]=='process']
 for i,(call,signal) in enumerate(zip(propagation,normalized)):
  scene=SCENES[r['scene_indices'][i]];pid=tuple(int(meta[k][i]) for k in ('eq_i','rx_i','day_i','sig_i','base_index'));raw=x[i,0]+1j*x[i,1]
  assert call[1] is obj.configs[scene] and call[2]==stable_seed(d.CHANNEL_SEEDS['train'],ns(4441),scene,pid) and call[3]==Hardware('rx1:day1:eq1')
  assert np.array_equal(signal,raw/np.sqrt(np.mean(abs(raw)**2))) and r['channel_seeds'][i]==call[2] and r['session_ids'][i]=='rx1:day1:eq1'
 for epoch,step in [(20,4441),(21,4440),(22,4441)]:bad(lambda epoch=epoch,step=step:env['apply'](state(),x,meta,epoch,step))
 wrong={k:v.copy() for k,v in meta.items()};wrong['rx_i'][0]=2;bad(lambda:env['apply'](state(),x,wrong,21,4441))
 bad(lambda:env['apply'](state(),x[:63],{k:v[:63] for k,v in meta.items()},21,4441))
 # Original hook is delegated,not copied or replaced with MIX/weighted equations.
 text=Path(__file__).with_name('model.py').read_text(encoding='utf8');assert 'delta,previous_info=original.hook(self,**kw)' in text and "m._crossview_mode!='ORD128'" in text and 'capture_forward' not in text and '.forward_identity_only(' not in text
 mt=ast.parse(text);hook=next(n for n in mt.body if isinstance(n,ast.FunctionDef) and n.name=='hook');actual=set();fieldsets=dict(CONCAT_FIELDS=CONCAT_FIELDS,BALANCE_FIELDS=BALANCE_FIELDS)
 for node in ast.walk(hook):
  if isinstance(node,ast.DictComp) and isinstance(node.key,ast.BinOp) and isinstance(node.key.left,ast.Constant):actual.update(node.key.left.value+k for k in fieldsets[node.generators[0].iter.id])
 assert actual==set(INFO_KEYS) and len(actual)==14 and 'shape/RD' not in actual
 assert .5+.5+.25==1.25
 # Source clean comparator is originalNoAs;mid/low comparator is originalAs.
 from . import report
 assert report.PAIRS==( ('BAL_AS30','NOAS30'),('BAL_AS30','AS30'),('AS30','NOAS30') )
 old_rows=report.logical_rows;old_read=report.d.read;old_write=report.d.write;old_load=report.np.load
 class Arrays:
  def __init__(self):
   yy=np.arange(1800)%6;rr=np.asarray([1,3,4,6,8])[np.arange(1800)%5];clean=np.eye(6,dtype=np.float32)[yy];self.values=dict(ids=np.asarray(['mock'+str(i) for i in range(1800)]),TX=yy,RX=rr,clean=clean,satellite=np.tile(clean,(2,3,1,1)))
  def __enter__(self):return self.values
  def __exit__(self,*unused):pass
 def read_mock(path):
  name=Path(path).name
  if name=='completion.json':return dict(target_access=False,final_source_metrics=dict(source_val_count=27000,source_val_accuracy=.98,source_val_rx_accuracy={str(r):.98 for r in (1,3,4,6,8)}))
  if name in ('source_frozen.json','complete.json'):return dict(target_access=False)
  return dict(role_ids={'L':['mock']})
 captured=[]
 try:
  report.logical_rows=lambda:[(name,Path('SYNTHETIC')/name) for name in ('NOAS30','AS30','BAL_AS30')];report.d.read=read_mock;report.d.write=lambda path,value:captured.append(value);report.np.load=lambda *args,**kwargs:Arrays()
  summary=report.source_decision();json.dumps(summary,allow_nan=False);assert type(summary['promotion']['passed']) is bool and summary['promotion']['passed'] is False and len(captured)==1 and summary['scientific_failure_skips_formal'] is False
 finally:report.logical_rows=old_rows;report.d.read=old_read;report.d.write=old_write;report.np.load=old_load
 for file in Path(__file__).parent.glob('*.py'):ast.parse(file.read_text(encoding='utf8'),filename=str(file))
 return dict(status='PASS',protocol_negatives=negative,all2220_allocation_counts_verified=True,perstep21_21_22=True,perthree_each64=True,perepoch_each4736=True,tail10_each47360=True,private_RNG_global_numpy_and_python_unchanged=True,actual_downstream_apply_mock64=True,unchanged_cfg_seed_session_normalize_residual_call_path=True,duplicate_PID_preserved=True,original_P43_ORD128_delegated=True,emitted_native_info_keys=len(actual),full6L_total_metric=True,source_summary_native_JSON=True,real_Torch_or_checkpoint_smoke='launcher first step pending770NN',NN=0,CP=0,IQ=0,updates=0,target=False)
if __name__=='__main__':print(json.dumps(run()))
