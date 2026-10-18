"""Fixed protocol and exact target-entropy math;no IQ/CP/model read or NN."""
import ast,copy,json,time
from pathlib import Path
import numpy as np
from . import design as d
from .protocol import cumulative,check_checkpoint,check_source_metadata,check_default_head
from .schema import INFO_KEYS,CONCAT_FIELDS,SATLS_FIELDS

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

def numpy_math():
 rng=np.random.default_rng(6001);z=[rng.normal(size=(128,6)) for _ in range(2)];y=np.arange(128)%6;mask=(np.arange(128)+1)%2!=0
 def ce(a,e):
  lp=a-np.max(a,axis=1,keepdims=True);lp-=np.log(np.exp(lp).sum(1,keepdims=True));return -(1-e)*lp[np.arange(128),y]-e*lp.mean(1)
 old=[ce(a,.01) for a in z];new=[np.where(mask,ce(a,.05),o) for a,o in zip(z,old)]
 delta=.25*sum((n-o).sum()/128 for n,o in zip(new,old));direct=.25*sum(n.mean() for n in new);base=.25*sum(o.mean() for o in old)
 assert abs(direct-base-delta)<1e-14
 expected=.25*(.05-.01)*(np.eye(6)[y]-1/6)/128;expected[~mask]=0
 # Finite differences of the correction,including selected/complement and everyclass.
 eps=1e-5;maxerr=0.
 for i in range(12):
  for k in range(6):
   oldval=z[0][i,k];z[0][i,k]=oldval+eps;a=.25*((ce(z[0],.05)-ce(z[0],.01))[mask].sum())/128;z[0][i,k]=oldval-eps;b=.25*((ce(z[0],.05)-ce(z[0],.01))[mask].sum())/128;z[0][i,k]=oldval;maxerr=max(maxerr,abs((a-b)/(2*eps)-expected[i,k]))
 assert maxerr<1e-10
 return dict(group_mean_and_delta=True,finite_difference_gradient_max_error=maxerr)

def torch_math():
 try:import torch
 except ImportError:return dict(status='NOT_AVAILABLE_LOCAL',actual_autograd='launcher remote focused check',NN=0)
 from .smoothing import correction
 from torch.nn import functional as F
 state=torch.random.get_rng_state();start=time.perf_counter();calls=[];original=F.cross_entropy
 try:
  torch.manual_seed(6002);a=torch.randn(128,6,dtype=torch.float64,requires_grad=True);b=torch.randn(128,6,dtype=torch.float64,requires_grad=True);y=torch.arange(128)%6;mask=(torch.arange(128)+1)%2!=0
  def recorded(z,*args,**kw):calls.append(dict(packets=len(z),epsilon=kw['label_smoothing']));return original(z,*args,**kw)
  base_means=tuple(original(z,y,label_smoothing=.01) for z in (a,b))
  F.cross_entropy=recorded
  delta,r=correction(a,b,y,mask,old_CE_means=base_means)
  F.cross_entropy=original
  assert calls==[dict(packets=64,epsilon=.01)]*2+[dict(packets=64,epsilon=.05)]*2
  old=[original(z,y,reduction='none',label_smoothing=.01) for z in (a,b)];new=[original(z,y,reduction='none',label_smoothing=.05) for z in (a,b)]
  direct=.25*sum(torch.where(mask,n,o).mean() for n,o in zip(new,old));via=.25*sum(o.mean() for o in old)+delta
  assert torch.allclose(direct,via,atol=1e-12,rtol=1e-12)
  gd=torch.autograd.grad(direct,(a,b),retain_graph=True);gv=torch.autograd.grad(via,(a,b),retain_graph=True)
  assert all(torch.allclose(x,y,atol=1e-12,rtol=1e-12) for x,y in zip(gd,gv))
  dg=torch.autograd.grad(delta,(a,b),retain_graph=True);assert all(torch.allclose(v[~mask],torch.zeros_like(v[~mask]),atol=1e-15,rtol=0) for v in dg)
  same,sr=correction(a,b,y,mask,epsilon=.01,old_CE_means=base_means);sg=torch.autograd.grad(same,(a,b),retain_graph=True)
  assert float(same)==0 and all(torch.count_nonzero(v)==0 for v in sg)
  none,nr=correction(a,b,y,mask,enabled=False);ng=torch.autograd.grad(none,a)[0]
  assert float(none)==0 and not torch.count_nonzero(ng) and nr['extra_CE_operator_calls']==nr['extra_CE_logit_packets']==0
  rejected=0
  for aa,bb,yy,mm,epsilon in [(a[:64],b[:64],y[:64],mask[:64],.05),(a,b,y.float(),mask,.05),(a,b,y,mask.float(),.05),(a,b,y,torch.ones(128,dtype=torch.bool),.05),(a,b,y,mask,.1),(a,b,torch.full((128,),6),mask,.05)]:
   try:correction(aa,bb,yy,mm,epsilon=epsilon,old_CE_means=base_means)
   except ValueError:rejected+=1
   else:raise AssertionError('Invalid smoothing scope accepted')
  return dict(status='PASS',actual_loss_gradient_equivalence=True,epsilon01_identity_loss_and_gradient=True,noAs_zero_loss_and_CE_calls=True,exact_extra_CE_call_shapes=True,protocol_negatives=rejected,CPU_wall_seconds=time.perf_counter()-start,synthetic_logit_tensor_only=True,NN=0)
 finally:F.cross_entropy=original;torch.random.set_rng_state(state)

def run():
 negative=0
 def bad(f):
  nonlocal negative
  try:f()
  except (ValueError,TypeError,KeyError,StopIteration):negative+=1
  else:raise AssertionError('Invalid fixed satelliteLS scope accepted')
 assert [r['arm'] for r in d.rows()]==['B01']
 for row in d.rows():
  c=d.config(row);d.validate(c)
  assert (c['epochs'],c['updates'],c['cumulative_epochs'],c['cumulative_updates'])==(10,2220,30,6660)
  assert c['data_contract']==dict(RX=[1,3,4,6,8],days=[1,2,3],L=6300,U=56700,V=27000,split=392005,source_RX_holdout=None,internal_L_split=False,U_TX_truth_hidden=True)
  assert c['As'] is True and c['training_mechanism']['pair_weights']==[.5,.5,.25] and c['crossview']['mode']=='ORD128' and c['satellite_label_smoothing']['epsilon']==.05
  assert d.reference_config(row,20)['arm']==row['arm'] and Path(c['parent_source'])==Path(d.reference_config(row,30)['parent_source'])
  for k,v in [('target_access',True),('epochs',30),('updates',6660),('parent_epochs',30),('step_offset',0),('model_seed',2026092702),('baseline_reuse',True),('Ag_mode','WHITE'),('As',not c['As']),('label_epochs',13),('parent_source','unknown'),('separate_head_training',True)]:
   z=copy.deepcopy(c);z[k]=v;bad(lambda z=z:d.validate(z))
  z=copy.deepcopy(c);z['crossview']['mode']='MIXED128';bad(lambda:d.validate(z))
  for key,value in [('epsilon',.1),('native_label_smoothing',.05),('selected_packets_per_step',128),('extra_NN',128)]:
   z=copy.deepcopy(c);z['satellite_label_smoothing'][key]=value;bad(lambda z=z:d.validate(z))
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
 # The localhook is a fixed P41/P43 copy;no model call exists in the correction.
 math=numpy_math();actual_torch=torch_math()
 text=Path(__file__).with_name('model.py').read_text(encoding='utf8');tree=ast.parse(text)
 assert "m._crossview_mode!='ORD128'" in text and 'extra_loss=extra_loss+2*smooth_delta' in text and 'extra_ce=extra_ce+2*smooth_delta' in text
 assert 'smooth_delta,sr=correction(z0,z1,labels,smask' in text and 'identity_loss,sm' not in text
 assert text.count('capture_forward(self,')==3 and text.count('self.forward_identity_only(view,')==2
 assert 'forward_identity_only' not in Path(__file__).with_name('smoothing.py').read_text(encoding='utf8')
 legacy_source=legacy_smoothing_source_check()
 actual=set();fieldsets=dict(CONCAT_FIELDS=CONCAT_FIELDS,SATLS_FIELDS=SATLS_FIELDS)
 for node in ast.walk(tree):
  if isinstance(node,ast.DictComp) and isinstance(node.key,ast.BinOp) and isinstance(node.key.left,ast.Constant) and isinstance(node.generators[0].iter,ast.Name):actual.update(node.key.left.value+k for k in fieldsets[node.generators[0].iter.id])
 assert actual==set(INFO_KEYS) and len(actual)==14 and 'shape/RD' not in actual
 assert .5+.5+.25==1.25 and .5*.5*.5==.125 and .5*.5==.25
 # Source clean comparator is originalNoAs;mid/low comparator is originalAs.
 from . import report
 assert report.PAIRS==( ('SATLS_AS30','NOAS30'),('SATLS_AS30','AS30'),('AS30','NOAS30') )
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
  report.logical_rows=lambda:[(name,Path('SYNTHETIC')/name) for name in ('NOAS30','AS30','SATLS_AS30')];report.d.read=read_mock;report.d.write=lambda path,value:captured.append(value);report.np.load=lambda *args,**kwargs:Arrays()
  summary=report.source_decision();json.dumps(summary,allow_nan=False);assert type(summary['promotion']['passed']) is bool and summary['promotion']['passed'] is False and len(captured)==1 and summary['scientific_failure_skips_formal'] is False
 finally:report.logical_rows=old_rows;report.d.read=old_read;report.d.write=old_write;report.np.load=old_load
 for file in Path(__file__).parent.glob('*.py'):ast.parse(file.read_text(encoding='utf8'),filename=str(file))
 return dict(status='PASS',protocol_negatives=negative,math=math,Torch_synthetic_logits=actual_torch,legacy_smoothing_source=legacy_source,emitted_native_info_keys=len(actual),full6L_total_metric=True,selectedAs64_only=True,native_global_smoothing_remains_0p01=True,original_P43_forward_call_count=True,source_summary_native_JSON=True,real_Torch_model_checkpoint_smoke='launcher first step pending770NN',NN=0,CP=0,IQ=0,updates=0,target=False)
if __name__=='__main__':print(json.dumps(run()))
