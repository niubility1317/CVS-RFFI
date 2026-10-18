"""Fixed scope negatives, actual grouped helper mocks and emitted metric schema."""
import ast,copy,json,sys,types
from pathlib import Path
import numpy as np
from . import design as d
from .protocol import cumulative,check_checkpoint,check_source_metadata,check_default_head
from .weighted import grouped,combine
from .schema import INFO_KEYS

def run():
 negative=0
 def bad(f):
  nonlocal negative
  try:f()
  except (ValueError,TypeError,KeyError,StopIteration):negative+=1
  else:raise AssertionError('Invalid fixed Asweight scope accepted')
 assert [r['arm'] for r in d.rows()]==['B00','B01']
 for row in d.rows():
  c=d.config(row);d.validate(c)
  assert (c['epochs'],c['updates'],c['cumulative_epochs'],c['cumulative_updates'])==(10,2220,30,6660)
  assert c['data_contract']==dict(RX=[1,3,4,6,8],days=[1,2,3],L=6300,U=56700,V=27000,split=392005,source_RX_holdout=None,internal_L_split=False,U_TX_truth_hidden=True)
  assert c['As']==(row['arm']=='B01') and c['mixed_weight']['total_L_scale']==1.25 and c['mixed_weight']['primary_ordinary_weight']==.5 and c['mixed_weight']['raw_weight']==.25
  assert d.reference_config(row,20)['arm']==row['arm'] and Path(c['parent_source'])==Path(d.reference_config(row,30)['parent_source'])
  for k,v in [('target_access',True),('epochs',30),('updates',6660),('parent_epochs',30),('step_offset',0),('model_seed',2026092702),('baseline_reuse',True),('Ag_mode','WHITE'),('As',not c['As']),('label_epochs',13),('parent_source','unknown'),('separate_head_training',True)]:
   z=copy.deepcopy(c);z[k]=v;bad(lambda z=z:d.validate(z))
  z=copy.deepcopy(c);z['mixed_weight']['selected_group_weight']=.5;bad(lambda:d.validate(z))
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
 # Execute the actual grouped helper with analytical scalar/gradient mocks, no model/Torch.
 class Scalar:
  def __init__(self,value,gradient):self.value=float(value);self.gradient=np.asarray(gradient,dtype=float)
  def __mul__(self,k):return Scalar(self.value*k,self.gradient*k)
  __rmul__=__mul__
  def __add__(self,other):return Scalar(self.value+other.value,self.gradient+other.gradient)
 def mock_terms(a,b,y,alpha):
  ids=a[:,1].astype(int);ce_grad=np.zeros(256);kl_grad=np.zeros(256);ce_grad[ids]=1/len(ids);kl_grad[128+ids]=1/len(ids)
  ce=Scalar(a[:,0].mean(),ce_grad);kl=Scalar(b[:,0].mean(),kl_grad)
  return ce+kl,ce,kl,None,None
 a=np.zeros((128,6));b=np.zeros((128,6));a[:,0]=np.linspace(.2,3,128);b[:,0]=np.linspace(.01,.5,128);a[:,1]=np.arange(128);y=np.arange(128,dtype=np.int64)%6;mask=np.arange(128)%2==0
 previous=sys.modules.get('torch');sys.modules['torch']=types.SimpleNamespace(long=np.dtype('int64'),bool=np.dtype('bool'))
 try:
  equal,_,_=grouped(a,b,y,mask,mock_terms,.25,.25);original=mock_terms(a,b,y,1.)
  weighted,ordinary,selected=grouped(a,b,y,mask,mock_terms)
  for v,o in zip(equal,original[:3]):assert abs(v.value-.5*o.value)<1e-12 and np.allclose(v.gradient,.5*o.gradient,atol=1e-15,rtol=0)
  for v,o,s in zip(weighted,ordinary,selected):assert abs(v.value-.125*o.value-.375*s.value)<1e-12 and np.allclose(v.gradient,.125*o.gradient+.375*s.gradient,atol=1e-15,rtol=0)
  assert abs(weighted[0].value-weighted[1].value-weighted[2].value)<1e-12
  for wrong in [np.zeros(128,dtype=bool),np.zeros(64,dtype=bool),mask.astype(int)]:bad(lambda wrong=wrong:grouped(a,b,y,wrong,mock_terms))
  bad(lambda:grouped(a[:64],b[:64],y[:64],mask[:64],mock_terms))
  for invalid in (-1,6):
   wrong=y.copy();wrong[0]=invalid;bad(lambda wrong=wrong:grouped(a,b,wrong,mask,mock_terms))
 finally:
  if previous is None:sys.modules.pop('torch',None)
  else:sys.modules['torch']=previous
 bad(lambda:combine((1,2,3),(4,5,6),.5,.5))
 assert .5+.125+.375+.25==1.25
 # Native info names are read from the actual hook AST, matching the shared schema.
 tree=ast.parse(Path(__file__).with_name('model.py').read_text(encoding='utf8'));hook=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='hook')
 prefixes={};raw_keys=set()
 for node in ast.walk(hook):
  if isinstance(node,ast.DictComp) and isinstance(node.key,ast.BinOp) and isinstance(node.key.left,ast.Constant) and isinstance(node.key.left.value,str):
   prefix=node.key.left.value;keys=ast.literal_eval(node.generators[0].iter);prefixes[prefix]=set(prefix+k for k in keys)
  if isinstance(node,ast.Dict):
   raw_keys.update(k.value for k in node.keys if isinstance(k,ast.Constant) and isinstance(k.value,str) and k.value.startswith('weight/'))
 actual=set().union(*prefixes.values(),raw_keys)
 assert actual==set(INFO_KEYS) and 'shape/RD' not in actual and len(actual)==21
 source=Path(__file__).with_name('model.py').read_text(encoding='utf8')
 assert "info={'concat/'+k:record[k]" in source and "record['L_forwards']=6" in source
 # Reproduce the known source-summary NumPy boolean boundary using synthetic arrays.
 from . import report
 old_rows=report.logical_rows;old_read=report.d.read;old_write=report.d.write;old_load=report.np.load
 class Arrays:
  def __init__(self):
   yy=np.arange(1800)%6;rr=np.asarray([1,3,4,6,8])[np.arange(1800)%5];clean=np.eye(6,dtype=np.float32)[yy];sat=np.tile(clean,(2,3,1,1))
   self.values=dict(ids=np.asarray(['mock'+str(i) for i in range(1800)]),TX=yy,RX=rr,clean=clean,satellite=sat)
  def __enter__(self):return self.values
  def __exit__(self,*unused):pass
 def read_mock(path):
  name=Path(path).name
  if name=='completion.json':return dict(target_access=False,final_source_metrics=dict(source_val_count=27000,source_val_accuracy=.98,source_val_rx_accuracy={str(r):.98 for r in (1,3,4,6,8)}))
  if name=='source_frozen.json':return dict(target_access=False)
  if name=='complete.json':return dict(target_access=False)
  return dict(role_ids={'L':['mock']})
 captured=[]
 try:
  report.logical_rows=lambda:[(name,Path('SYNTHETIC')/name) for name in ('NOAS_ORIGINAL','AS_ORIGINAL','NOAS_WEIGHT','AS_WEIGHT')]
  report.d.read=read_mock;report.d.write=lambda path,value:captured.append(value);report.np.load=lambda *args,**kwargs:Arrays()
  summary=report.source_decision();json.dumps(summary,allow_nan=False)
  assert type(summary['promotion']['passed']) is bool and summary['promotion']['passed'] is False and len(captured)==1
 finally:report.logical_rows=old_rows;report.d.read=old_read;report.d.write=old_write;report.np.load=old_load
 for file in Path(__file__).parent.glob('*.py'):ast.parse(file.read_text(encoding='utf8'),filename=str(file))
 return dict(status='PASS',protocol_negatives=negative,actual_grouped_helper_mock=True,equal_weights_reproduce_loss_and_gradient=True,CE_and_KL_fixed_same_weights=True,emitted_native_info_keys=len(actual),full6L_total_metric=True,source_summary_native_JSON=True,NN=0,CP=0,IQ=0,updates=0,target=False)
if __name__=='__main__':print(json.dumps(run()))
