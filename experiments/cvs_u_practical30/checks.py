"""Focused hidden-U/quarter/teacher-only/loss/row-parent negatives;no real IQ/CP/NN."""
import ast,copy,json,random,sys,types
from pathlib import Path
import numpy as np
from . import design as d
from .protocol import cumulative,check_checkpoint,check_source_metadata,check_default_head
from .schema import INFO_KEYS,CONCAT_FIELDS
from .u_routing import select,metadata,Router

def legacy_smoothing_source_check():
 import importlib.util
 try:spec=importlib.util.find_spec('experiments.cvs_sigtime_rdrop_legacy.model')
 except (ModuleNotFoundError,ValueError):spec=None
 origin=getattr(spec,'origin',None);p=Path(origin) if origin and origin not in ('built-in','frozen') else d.ROOT/'experiments/cvs_sigtime_rdrop_legacy/model.py'
 if p.is_file():
  assert 'smoothing!=.01' in p.read_text(encoding='utf8');return dict(status='PASS',source=str(p))
 return dict(status='NOT_PACKAGED_NONBLOCKING',source=None,interpretation='Actual original terms and globalLS.01 checked by launcher real smoke')

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
 sys.modules['torch']=fake
 count=0;negative=0
 def bad(f):
  nonlocal negative
  try:f()
  except (ValueError,TypeError,KeyError):negative+=1
  else:raise AssertionError('Invalid U boundary accepted')
 def metas(n):return dict(eq_i=tensor(np.ones(n,np.int64)),rx_i=tensor(np.asarray(d.RX)[np.arange(n)%5]),day_i=tensor(np.arange(n)%3+1),sig_i=tensor(np.arange(n)+2000),base_index=tensor(np.arange(n)),tx_label_visible=tensor(np.zeros(n,bool)))
 class Channel:
  def __init__(self):self.calls=0;self.packets=0;self.observed=[]
  def apply(self,x,meta,epoch,step):
   self.calls+=1;self.packets+=len(x);self.observed.append((x.copy(),copy.deepcopy(meta),epoch,step))
   obs=[dict(sample_id=tuple(int(meta[k][i]) for k in ('eq_i','rx_i','day_i','sig_i','base_index')),session_id='mockRXday',namespace='U_S',scene='practical_low_urban',state_end='LOS',receiver_lock_status='LOCKED',signal_to_added_noise_db=4.,quality_snr_db=5.) for i in range(len(x))]
   return (x+7).view(T),obs
  def summary(self):return dict(calls=self.calls,packets=self.packets)
 try:
  from .u_scope import NativeUScope
  for n in (256,124,8):
   raw=tensor(np.arange(n*512,dtype=np.float32).reshape(n,2,256)/1000);before=raw.copy();strong=raw+.02;meta=metas(n);meta['sig_i'][1]=meta['sig_i'][0];meta['base_index'][1]=meta['base_index'][0];meta['rx_i'][1]=meta['rx_i'][0];meta['day_i'][1]=meta['day_i'][0]
   ps=np.random.get_state();pr=random.getstate();positions,plan=select(n,21,4441);after=np.random.get_state();assert np.array_equal(ps[1],after[1]) and ps[2:]==after[2:] and pr==random.getstate() and len(set(positions))==n//4
   for arm in ('N01','C01','S01'):
    c=d.config(next(r for r in d.rows() if r['arm']==arm));ch=Channel();router=Router(c,ch);student,r=router.route(raw,strong,meta,epoch=21,step=4441);ix=r['selected_positions'];com=[i for i in range(n) if i not in ix]
    assert ix==positions and r['selected_slots']==n//4 and r['U_packets']==n and np.array_equal(raw,before) and r['duplicate_PID_occurrences']==1 and not r['U_truth_available']
    if arm=='S01':
     assert ch.calls==1 and ch.packets==n//4 and np.array_equal(student[ix],raw[ix]+7) and np.array_equal(student[com],strong[com]) and np.array_equal(ch.observed[0][0],raw[ix]) and student is not strong
     assert ch.observed[0][2:]==(21,4441) and ch.observed[0][1]['base_index']==meta['base_index'][ix].tolist()
    else:assert student is strong and ch.calls==0 and ch.packets==0 and r['As_packets']==0 and len(r['Gaussian_positions'])==n
    json.dumps(r,allow_nan=False);count+=1
  assert sum(select(256 if i%222!=221 else 124,(i//222)+21,i+4441)[0].__len__() for i in range(2220))==141750
  assert sum(256 if i%222!=221 else 124 for i in range(2220))==567000
  # Native call wrappers invoke original qualify/strong once;student disagreement never gates.
  for arm in ('C01','S01'):
   for empty in (False,True):
    n=8;c=d.config(next(r for r in d.rows() if r['arm']==arm));raw=tensor(np.ones((n,2,256),np.float32));meta=metas(n);router=Router(c,Channel());pseudo=tensor(np.arange(n)%6,dtype=np.int64);conf=tensor(np.linspace(.93,.99,n));cm=tensor(np.ones(n,bool) if not empty else np.zeros(n,bool));tm=tensor(np.arange(n)%2==0);logits=tensor(np.tile(np.arange(6,dtype=np.float32),(n,1)));calls=dict(threshold=0,temporal=0,strong=0)
    def threshold(*args):calls['threshold']+=1;return cm
    def temporal(*args):calls['temporal']+=1;return tm
    def strong(x,std):calls['strong']+=1;return (x+np.random.normal(0,std,x.shape).astype(x.dtype)).view(T)
    native=types.SimpleNamespace(_threshold_mask=threshold,_temporal_mask_tensor=temporal,_strong_augment=strong,_trusted_pseudo_loss_hook=None)
    m=object();o=types.SimpleNamespace(model=m,pending_U=dict(meta=meta,count=n,hidden_y=True,teacher_raw=raw,teacher_raw_no_grad=True),u_records=[],u_router=router);scope=NativeUScope(o,native);scope.install();args=types.SimpleNamespace(pseudo_strong_agreement=False,pseudo_domain_gate=False,pseudo_temporal_gate=True,pseudo_temporal_mode='batch_neighbor',pseudo_threshold_mode='rx_day_quantile',pseudo_quantile=.86,tau_min=.92,tau_max=.97,pseudo_temporal_window=2,pseudo_temporal_min_conf=.8,lambda_u=.16,lambda_ent=.01,label_smoothing=.01)
    try:
     assert native._threshold_mask(conf,tensor(np.arange(n)%3),args) is cm;assert native._temporal_mask_tensor(pseudo,conf,(None,meta),args,'cpu') is tm
     state=np.random.get_state();expected=strong(raw,.015);expectedstate=np.random.get_state();np.random.set_state(state);calls['strong']=0;student=native._strong_augment(raw,.015);gotstate=np.random.get_state();assert calls['strong']==1 and np.array_equal(expectedstate[1],gotstate[1]) and expectedstate[2:]==gotstate[2:]
     selected=select(n,21,4441)[0];com=[i for i in range(n) if i not in selected];assert np.array_equal(student[com],expected[com])
     if arm=='C01':assert np.array_equal(student,expected)
     mask=cm&tm;lu=tensor(1.2 if mask.any() else 0.);ent=tensor(.7)
     returned=native._trusted_pseudo_loss_hook(model=m,args=args,epoch=1,batch_idx=0,loss_l=tensor(1.),loss_u=lu,loss_ent=ent,active=True,logits=logits,pseudo=pseudo,mask=mask)
     assert returned is lu and o.pending_U is None and len(o.u_records)==1 and calls==dict(threshold=1,temporal=1,strong=1)
     r=o.u_records[0];assert r['CE_denominator']==int(mask.sum()) and r['entropy_denominator']==n and r['entropy_all_U'] and r['native_loss_returned_unchanged'] and r['CE_label_smoothing']==0 and not r['strong_agreement'];json.dumps(r,allow_nan=False)
    finally:scope.restore()
    assert native._threshold_mask is threshold and native._strong_augment is strong and native._trusted_pseudo_loss_hook is None
    count+=1
  for n,e,s in [(0,21,4441),(125,21,4441),(256,20,4441),(256,22,4441),(256,31,6661)]:bad(lambda n=n,e=e,s=s:select(n,e,s))
  for k,v in [('tx_i',np.zeros(8,int)),('rx_i',np.full(8,2)),('day_i',np.full(8,4)),('eq_i',np.zeros(8,int)),('base_index',np.arange(7))]:
   mm=metas(8);mm[k]=v;bad(lambda mm=mm:metadata(mm,8))
 finally:
  if old_torch is None:sys.modules.pop('torch',None)
  else:sys.modules['torch']=old_torch
  sys.modules.pop(__package__+'.u_scope',None)
  if old_scope is not None:sys.modules[__package__+'.u_scope']=old_scope
 return dict(status='PASS',mock_positive_cases=count,negatives=negative,production_batches=[256,124],quarters=[64,31],U_packets=567000,U_selected_slots=141750,original_strong_numpy_RNG_endstate_equal=True,global_private_selection_numpy_Python_RNG_preserved=True,raw_unmodified=True,selected_raw_not_Gaussian=True,teacher_only_accepts_student_disagreement=True,empty_mask_differentiable_zero_interface=True,U_truth_used=False,NN=0)

def run():
 negative=0
 def bad(f):
  nonlocal negative
  try:f()
  except (ValueError,TypeError,KeyError,StopIteration):negative+=1
  else:raise AssertionError('Invalid UPractical/data/parent scope accepted')
 assert [r['arm'] for r in d.rows()]==['N01','C01','S01']
 for row in d.rows():
  c=d.config(row);d.validate(c);ref=('B00' if row['arm']=='N01' else 'B01')+'-s'+str(d.SEED)
  assert (c['epochs'],c['updates'],c['cumulative_epochs'],c['cumulative_updates'])==(10,2220,30,6660) and c['reference_row_id']==ref
  assert c['data_contract']==dict(RX=[1,3,4,6,8],days=[1,2,3],L=6300,U=56700,V=27000,split=392005,source_RX_holdout=None,internal_L_split=False,U_TX_truth_hidden=True)
  assert c['As']==(row['arm']!='N01') and c['training_mechanism']['pair_weights']==[.5,.5,.25] and c['crossview']['mode']=='ORD128'
  assert d.reference_config(row,20)['row_id']==ref and d.reference_config(row,30)['row_id']==ref and Path(c['parent_source'])==Path(d.reference_config(row,30)['parent_source'])
  for k,v in [('target_access',True),('epochs',30),('updates',6660),('step_offset',0),('model_seed',2026092702),('baseline_reuse',True),('Ag_mode','WHITE'),('As',not c['As']),('reference_row_id','B01-wrong'),('parent_source','unknown'),('label_epochs',13)]:
   z=copy.deepcopy(c);z[k]=v;bad(lambda z=z:d.validate(z))
  for k,v in [('slot_fraction',.5),('pseudo_strong_agreement',True),('teacher_only_eligibility',False),('U_truth_used',True),('extra_NN',256),('mode','RAW_CONTROL')]:
   z=copy.deepcopy(c);z['u_practical'][k]=v;bad(lambda z=z:d.validate(z))
  # Test actual make_args kwargs/mapping against a minimal original API,not a local NN.
  old=sys.modules.get('experiments.cvs_noise_shape_factorial43');pkg=types.ModuleType('experiments.cvs_noise_shape_factorial43')
  original_rows=[dict(arm=a,row_id=a+'-s'+str(d.SEED),model_seed=d.SEED) for a in ('B00','B01')]
  def cfg(rr,seg):return copy.deepcopy(d._REFERENCE['configs'][rr['arm']][str(seg)])
  calls=[]
  def make(oc,device):
   calls.append((oc,device));return types.SimpleNamespace(epochs=10,label_epochs=0,pseudo_epochs=10,batch_size=128,label_smoothing=.01,seed=d.SEED,use_aug=True,use_concat_sat_channel_aug=False,use_ema_teacher=True,use_unlabeled=True,pseudo_strong_agreement=True,pseudo_domain_gate=False,pseudo_temporal_gate=True,pseudo_temporal_mode='batch_neighbor',strong_noise_std=.015,lambda_u=.16,lambda_ent=.01)
  pkg.design=types.SimpleNamespace(config=cfg,rows=lambda:original_rows,make_args=make);sys.modules[pkg.__name__]=pkg
  try:
   a=d.make_args(c,'cpu');assert calls[0][0]['row_id']==ref and not a.pseudo_strong_agreement and a.candidate_id==c['row_id'] and a.baseline_ckpt==c['parent_source']+'/final_ssdg.pth' and a.use_unlabeled and a.use_ema_teacher and a.label_epochs==0 and a.pseudo_epochs==10
  finally:
   if old is None:sys.modules.pop(pkg.__name__,None)
   else:sys.modules[pkg.__name__]=old
  cp=dict(epoch=20,a1_ema_successful_updates=4440,run_id=d.REFERENCE_RUN,candidate_id=ref,checkpoint_selection='final_only',ema_model={'mock':1},args=dict(seed=d.SEED,baseline_ckpt='',teacher_ckpt='',epochs=20,use_ema_teacher=True,label_smoothing=.01,batch_size=128))
  def verify(v):return check_checkpoint(v,run=d.REFERENCE_RUN,row=ref,epoch=20,updates=4440,baseline='')
  verify(cp)
  for k,v in [('epoch',30),('a1_ema_successful_updates',6660),('run_id','other'),('candidate_id',c['row_id']),('checkpoint_selection','target_best'),('ema_model',None)]:
   z=copy.deepcopy(cp);z[k]=v;bad(lambda z=z:verify(z))
 check_default_head((6,160))
 for sh in [(5,160),(6,64),(6,160,1)]:bad(lambda sh=sh:check_default_head(sh))
 assert cumulative(1,1)==(21,4441) and cumulative(10,2220)==(30,6660)
 for ep,step in [(0,0),(1,223),(11,2221),(10,1)]:bad(lambda ep=ep,step=step:cumulative(ep,step))
 routing=routing_checks();legacy=legacy_smoothing_source_check()
 text=Path(__file__).with_name('model.py').read_text(encoding='utf8');assert 'original.hook(self,**kw)' in text and '_full_channel=' not in text and set(INFO_KEYS)=={'concat/'+k for k in CONCAT_FIELDS} and len(INFO_KEYS)==7
 from . import report
 assert len(report.logical_rows())==5 and len(report.PAIRS)==10 and .5+.5+.25==1.25
 assert "'UPRACTICAL30_minus_MATCHED_NOAS30'" in Path(__file__).with_name('report.py').read_text(encoding='utf8')

 # Source summary is JSON-native and enforces only S-N clean/S-C midlow.
 old_rows=report.logical_rows;old_read=report.d.read;old_write=report.d.write;old_load=report.np.load
 class Arrays:
  def __init__(self):
   yy=np.arange(1800)%6;rr=np.asarray(d.RX)[np.arange(1800)%5];clean=np.eye(6,dtype=np.float32)[yy];self.values=dict(ids=np.asarray(['mock'+str(i) for i in range(1800)]),TX=yy,RX=rr,clean=clean,satellite=np.tile(clean,(2,3,1,1)))
  def __enter__(self):return self.values
  def __exit__(self,*unused):pass
 def read_mock(path):
  name=Path(path).name
  if name=='completion.json':return dict(target_access=False,final_source_metrics=dict(source_val_count=27000,source_val_accuracy=.98,source_val_rx_accuracy={str(r):.98 for r in d.RX}))
  if name in ('source_frozen.json','complete.json'):return dict(target_access=False)
  return dict(role_ids={'L_s':['mock']})
 captured=[]
 try:
  report.logical_rows=lambda:[(name,Path('SYNTHETIC')/name) for name in report.NAMES];report.d.read=read_mock;report.d.write=lambda path,value:captured.append(value);report.np.load=lambda *args,**kwargs:Arrays()
  summary=report.source_decision();json.dumps(summary,allow_nan=False);assert type(summary['promotion']['passed']) is bool and not summary['promotion']['passed'] and summary['promotion']['S_minus_N_fullV_clean_pp']==0. and summary['promotion']['S_minus_C_midlow_pp']==0. and len(captured)==1 and not summary['scientific_failure_skips_formal']
 finally:report.logical_rows=old_rows;report.d.read=old_read;report.d.write=old_write;report.np.load=old_load
 for p in Path(__file__).parent.glob('*.py'):ast.parse(p.read_text(encoding='utf8'),filename=str(p))
 return dict(status='PASS',protocol_negatives=negative,routing=routing,legacy_smoothing_source=legacy,emitted_native_info_keys=7,full6L_original_hook=True,actual_rows=3,unique_original_parents=2,source_stage_S_minus_N=True,no_new_losses=True,real_model_checkpoint_smoke='launcher first step pending2356NN',NN=0,CP=0,IQ=0,updates=0,target=False)
if __name__=='__main__':print(json.dumps(run()))
