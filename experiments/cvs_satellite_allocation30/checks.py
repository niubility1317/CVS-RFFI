"""Focused feasibleallocation/permissions/RNG/maskinterface checks;no realIQ/CP/NN."""
import ast,copy,json
from pathlib import Path
import numpy as np
from . import design as d
from .protocol import cumulative,check_checkpoint,check_source_metadata,check_default_head
from .schema import INFO_KEYS,CONCAT_FIELDS,ALLOC_FIELDS

def legacy_smoothing_source_check():
 # The frozen loader owns dependency paths;the candidate release packages no legacy model.
 import importlib.util
 try:spec=importlib.util.find_spec('experiments.cvs_sigtime_rdrop_legacy.model')
 except (ModuleNotFoundError,ValueError):spec=None
 origin=getattr(spec,'origin',None)
 p=Path(origin) if origin and origin not in ('built-in','frozen') else d.ROOT/'experiments/cvs_sigtime_rdrop_legacy/model.py'
 if p.is_file():
  assert 'smoothing!=.01' in p.read_text(encoding='utf8')
  return dict(status='PASS',source=str(p),locator='actual import spec' if origin else 'local static source')
 return dict(status='NOT_PACKAGED_NONBLOCKING',source=None,interpretation='Unchanged frozen dependency;actual args .01 and original terms remain checked by launcher smoke')

def selector_checks():
 import random
 from .allocation import select,waterfill,SEED,NAMESPACE,stable_seed
 rejected=0
 def bad(f):
  nonlocal rejected
  try:f()
  except (ValueError,TypeError):rejected+=1
  else:raise AssertionError('Invalid allocation interface accepted')
 y=np.arange(128)%6;rx=np.asarray([1,3,4,6,8])[(np.arange(128)//6)%5]
 before=np.random.get_state();python_before=random.getstate();mask,r=select(y,rx,21,4441)
 after=np.random.get_state();assert before[0]==after[0] and np.array_equal(before[1],after[1]) and before[2:]==after[2:] and random.getstate()==python_before
 def verify(m,s,yy,rr):
  assert m.dtype==bool and m.shape==(128,) and int(m.sum())==64 and sorted(s['selected_L_positions']+s['ordinary_L_positions'])==list(range(128))
  assert len(set(s['selected_L_positions']))==64 and sum(c['available'] for c in s['cells'])==128 and sum(c['selected'] for c in s['cells'])==64
  assert all(0<=c['selected']<=c['available'] for c in s['cells'])
  for c in s['cells']:
   assert all(int(yy[i])==c['TX'] and int(rr[i])==c['RX'] for i in c['input_L_positions'])
   assert c['selected_L_positions']==[i for i in c['input_L_positions'] if m[i]]
  quotas=np.asarray([c['selected'] for c in s['cells']]);capacity=np.asarray([c['available'] for c in s['cells']]);live=quotas[quotas<capacity]
  if len(live):assert int(live.max()-live.min())<=1
  json.dumps(s,allow_nan=False)
 verify(mask,r,y,rx);assert len(r['cells'])==30
 # Same physicalgroups under a class-ID rename retain theexact mask androundorders.
 renamed=np.asarray([4,1,5,0,2,3])[y];mm,ss=select(renamed,rx,21,4441);assert np.array_equal(mask,mm) and ss['rounds']==r['rounds']
 # Reordering actualbatch positions carries the samepriorities androundRNG tocore.
 seed=stable_seed(SEED,NAMESPACE,21,4441);rng=np.random.default_rng(seed);p=rng.random(128);m0,s0=waterfill(y,rx,p,rng);permutation=np.random.default_rng(6009).permutation(128)
 rng2=np.random.default_rng(seed);rng2.random(128);mp,sp=waterfill(y[permutation],rx[permutation],p[permutation],rng2);back=np.zeros(128,bool);back[permutation]=mp;assert np.array_equal(back,m0) and sp['rounds']==s0['rounds']
 cases=[(np.zeros(128,int),np.ones(128,int)),(np.r_[5,np.zeros(127,int)],np.r_[8,np.ones(127,int)]),(np.r_[np.arange(30)%6,np.zeros(98,int)],np.r_[np.repeat([1,3,4,6,8],6),np.ones(98,int)])]
 for yy,rr in cases:
  m,s=select(yy,rr,21,4441);verify(m,s,yy,rr)
 assert len(select(*cases[0],21,4441)[1]['cells'])==1 and select(*cases[1],21,4441)[1]['quota_min']==1
 # No global RNG isused for arbitraryepochs andbatchcompositions;nodata isloaded.
 for step in [4441,4442,4662,4663,5555,6439,6660]:
  epoch=(step-1)//222+1;m,s=select(y,rx,epoch,step);verify(m,s,y,rx);assert s['old_selected_L_positions']==np.flatnonzero((np.arange(128)+step)%2!=0).tolist()
 for yy,rr,pp in [(y[:127],rx[:127],p[:127]),(y.astype(float),rx,p),(np.full(128,6),rx,p),(y,np.full(128,2),p),(y,rx,np.full(128,np.nan))]:bad(lambda yy=yy,rr=rr,pp=pp:waterfill(yy,rr,pp,np.random.default_rng(1)))
 for epoch,step in [(20,4441),(21,4440),(22,4441),(31,6661)]:bad(lambda epoch=epoch,step=step:select(y,rx,epoch,step))
 # RepeatedphysicalIDs remain repeatedpositions;selectorhas noPID input or dedup.
 assert 'PID' not in __import__('inspect').signature(select).parameters
 return dict(status='PASS',negatives=rejected,full30cells=True,missing_singleton_and_singlecell=True,position_withoutreplacement64=True,feasible_waterfill=True,class_rename_exact_equivariance=True,carried_priority_core_permutation_equivariance=True,global_numpy_and_Python_RNG_unchanged=True,priority_ties_rule='originalposition tiebreak;permutation check usesdistinct drawnpriorities')

def torch_rng_checks():
 try:import torch
 except ImportError:return dict(status='NOT_AVAILABLE_LOCAL',CPU_Torch_state_test='remote focused check',CUDA_no_random_calls='selector has noTorch import/API',NN=0)
 from .allocation import select
 cpu=torch.random.get_rng_state().clone();cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
 select(np.arange(128)%6,np.asarray([1,3,4,6,8])[(np.arange(128)//6)%5],21,4441)
 assert torch.equal(cpu,torch.random.get_rng_state())
 if cuda is not None:assert all(torch.equal(a,b) for a,b in zip(cuda,torch.cuda.get_rng_state_all()))
 return dict(status='PASS',CPU_Torch_state_unchanged=True,CUDA_state_checked=cuda is not None,CUDA_state_unchanged=True if cuda is not None else None,selector_has_no_Torch_import=True,NN=0)

def mask_boundary_mock():
 from types import SimpleNamespace
 from .allocation import select
 class Tensor(np.ndarray):
  def to(self,*unused,**kw):return self
  @property
  def device(self):return 'cpu'
 def tensor(v):return np.asarray(v).view(Tensor)
 y=np.arange(128)%6;rx=np.asarray([1,3,4,6,8])[(np.arange(128)//6)%5];chosen,plan=select(y,rx,21,4441);mask=tensor(chosen)
 raw=tensor(np.arange(128*2*256,dtype=np.float32).reshape(128,2,256));before=raw.copy();meta=dict(eq_i=tensor(np.ones(128,int)),rx_i=tensor(rx),day_i=tensor(np.arange(128)%3+1),sig_i=tensor(np.arange(128)+2000),base_index=tensor(np.arange(128)),unbatched='retain')
 for key in ('eq_i','rx_i','day_i','sig_i','base_index'):meta[key][1]=meta[key][0]
 calls=[]
 class Channel:
  def apply(self,x,metadata,**kw):calls.append((x.copy(),metadata,kw));return x.copy(),[dict(sample_id=tuple(int(metadata[k][i]) for k in ('eq_i','rx_i','day_i','sig_i','base_index'))) for i in range(64)]
 tree=ast.parse(Path(__file__).with_name('model.py').read_text(encoding='utf8'));hook=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='hook')
 nodes=[n for n in hook.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='meta' or isinstance(t,ast.Tuple) and any(isinstance(e,ast.Name) and e.id=='satellite' for e in t.elts) for t in n.targets)]
 assert len(nodes)==2
 function=ast.parse('def boundary(self,raw,smask,extra,epoch,step): pass').body[0];function.body=nodes+[ast.Return(value=ast.Tuple(elts=[ast.Name(id=k,ctx=ast.Load()) for k in ('satellite','observed','meta')],ctx=ast.Load()))];module=ast.fix_missing_locations(ast.Module(body=[function],type_ignores=[]));env=dict(torch=SimpleNamespace(is_tensor=lambda x:isinstance(x,Tensor)));exec(compile(module,'actual_mask_boundary','exec'),env)
 output,observed,used=env['boundary'](SimpleNamespace(_full_channel=Channel()),raw,mask,(None,meta),1,1)
 assert len(calls)==1 and output.shape==(64,2,256) and np.array_equal(before,raw) and np.array_equal(output,raw[plan['selected_L_positions']])
 assert calls[0][2]==dict(epoch=1,step=1) and used['unbatched']=='retain'
 for key in ('eq_i','rx_i','day_i','sig_i','base_index'):assert np.array_equal(used[key],meta[key][mask])
 ids=[tuple(int(meta[k][i]) for k in ('eq_i','rx_i','day_i','sig_i','base_index')) for i in range(128)];assert ids[0]==ids[1] and [o['sample_id'] for o in observed]==[ids[i] for i in plan['selected_L_positions']]
 return dict(status='PASS',actual_source_mask_metadata_subset=True,one_channel_call64=True,original128_input_unchanged=True,repeated_PID_not_rejected=True,NN=0,real_IQ=0)

def run():
 negative=0
 def bad(f):
  nonlocal negative
  try:f()
  except (ValueError,TypeError,KeyError,StopIteration):negative+=1
  else:raise AssertionError('Invalid fixed allocation scope accepted')
 assert [r['arm'] for r in d.rows()]==['B01']
 for row in d.rows():
  c=d.config(row);d.validate(c)
  assert (c['epochs'],c['updates'],c['cumulative_epochs'],c['cumulative_updates'])==(10,2220,30,6660)
  assert c['data_contract']==dict(RX=[1,3,4,6,8],days=[1,2,3],L=6300,U=56700,V=27000,split=392005,source_RX_holdout=None,internal_L_split=False,U_TX_truth_hidden=True)
  assert c['As'] is True and c['training_mechanism']['pair_weights']==[.5,.5,.25] and c['crossview']['mode']=='ORD128' and c['satellite_allocation']['selected_packets_per_step']==64
  assert d.reference_config(row,20)['arm']==row['arm'] and Path(c['parent_source'])==Path(d.reference_config(row,30)['parent_source'])
  for k,v in [('target_access',True),('epochs',30),('updates',6660),('parent_epochs',30),('step_offset',0),('model_seed',2026092702),('baseline_reuse',True),('Ag_mode','WHITE'),('As',not c['As']),('label_epochs',13),('parent_source','unknown'),('separate_head_training',True)]:
   z=copy.deepcopy(c);z[k]=v;bad(lambda z=z:d.validate(z))
  z=copy.deepcopy(c);z['crossview']['mode']='MIXED128';bad(lambda:d.validate(z))
  for key,value in [('selected_packets_per_step',128),('augmentation_seed',7),('loader_changed',True),('extra_NN',128)]:
   z=copy.deepcopy(c);z['satellite_allocation'][key]=value;bad(lambda z=z:d.validate(z))
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
 allocation=selector_checks();torch_rng=torch_rng_checks();boundary=mask_boundary_mock()
 text=Path(__file__).with_name('model.py').read_text(encoding='utf8');tree=ast.parse(text)
 assert "m._crossview_mode!='ORD128'" in text and 'total=.5*common+.5*extra_loss+weight*aux' in text and 'smask=torch.as_tensor(selected' in text
 assert text.count('capture_forward(self,')==3 and text.count('self.forward_identity_only(view,')==2 and 'smoothing' not in text.replace('label_smoothing=.01','')
 legacy_source=legacy_smoothing_source_check()
 actual=set();fieldsets=dict(CONCAT_FIELDS=CONCAT_FIELDS,ALLOC_FIELDS=ALLOC_FIELDS)
 for node in ast.walk(tree):
  if isinstance(node,ast.DictComp) and isinstance(node.key,ast.BinOp) and isinstance(node.key.left,ast.Constant) and isinstance(node.generators[0].iter,ast.Name):actual.update(node.key.left.value+k for k in fieldsets[node.generators[0].iter.id])
 assert actual==set(INFO_KEYS) and len(actual)==14 and 'shape/RD' not in actual
 assert .5+.5+.25==1.25
 # Source clean comparator is originalNoAs;mid/low comparator is originalAs.
 from . import report
 assert report.PAIRS==( ('ALLOC_AS30','NOAS30'),('ALLOC_AS30','AS30'),('AS30','NOAS30') )
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
  report.logical_rows=lambda:[(name,Path('SYNTHETIC')/name) for name in ('NOAS30','AS30','ALLOC_AS30')];report.d.read=read_mock;report.d.write=lambda path,value:captured.append(value);report.np.load=lambda *args,**kwargs:Arrays()
  summary=report.source_decision();json.dumps(summary,allow_nan=False);assert type(summary['promotion']['passed']) is bool and summary['promotion']['passed'] is False and len(captured)==1 and summary['scientific_failure_skips_formal'] is False
 finally:report.logical_rows=old_rows;report.d.read=old_read;report.d.write=old_write;report.np.load=old_load
 for file in Path(__file__).parent.glob('*.py'):ast.parse(file.read_text(encoding='utf8'),filename=str(file))
 return dict(status='PASS',protocol_negatives=negative,allocation=allocation,Torch_RNG=torch_rng,mask_boundary=boundary,legacy_smoothing_source=legacy_source,emitted_native_info_keys=len(actual),full6L_total_metric=True,no_new_losses_orCE=True,source_summary_native_JSON=True,real_model_checkpoint_smoke='launcher first step pending770NN',NN=0,CP=0,IQ=0,updates=0,target=False)
if __name__=='__main__':print(json.dumps(run()))
