"""Focused contract negatives and loss-component arithmetic; no NN/CP/IQ."""
import ast,copy,json
from pathlib import Path
import numpy as np
from . import design as d
from .protocol import cumulative,check_checkpoint,check_source_metadata,check_default_head
from .schema import INFO_KEYS
def run():
 negative=0
 def bad(f):
  nonlocal negative
  try:f()
  except (ValueError,TypeError,KeyError,StopIteration):negative+=1
  else:raise AssertionError('Invalid splitAdam fixed scope accepted')
 assert [r['arm'] for r in d.rows()]==['N01','C01','S01']
 for row in d.rows():
  c=d.config(row);d.validate(c);ref=d.reference_row(row)
  assert (c['epochs'],c['updates'],c['cumulative_epochs'],c['cumulative_updates'])==(10,2220,30,6660)
  assert c['data_contract']==dict(RX=[1,3,4,6,8],days=[1,2,3],L=6300,U=56700,V=27000,split=392005,source_RX_holdout=None,internal_L_split=False,U_TX_truth_hidden=True)
  assert c['As']==(row['arm']!='N01') and c['optimizer_mode']==c['moment_mode']==('SPLIT' if row['arm']=='S01' else 'MEASURE') and not c['amp']
  assert c['training_mechanism']['pair_weights']==[.5,.5,.25] and c['crossview']['mode']=='ORD128'
  assert ref['arm']==('B00' if row['arm']=='N01' else 'B01') and d.reference_config(row,20)['row_id']==ref['row_id'] and Path(c['parent_source'])==Path(d.reference_config(row,30)['parent_source'])
  assert c['split_adam']['adaptive_weights']==[.8,.2] and c['split_adam']['VJP_calls_per_update']==2 and not c['split_adam']['extra_NN'] and not c['split_adam']['new_U']
  for k,v in [('target_access',True),('epochs',30),('updates',6660),('parent_epochs',30),('step_offset',0),('model_seed',2026092702),('baseline_reuse',True),('Ag_mode','WHITE'),('As',not c['As']),('parent_source','unknown'),('parent_row_id','test_selected'),('amp',True),('optimizer_mode','ORIGINAL'),('moment_mode','PROJECT'),('separate_head_training',True)]:
   z=copy.deepcopy(c);z[k]=v;bad(lambda z=z:d.validate(z))
  z=copy.deepcopy(c);z['split_adam']['adaptive_weights']=[.9,.1];bad(lambda:d.validate(z))
  z=copy.deepcopy(c);z['crossview']['mode']='MIXED128';bad(lambda:d.validate(z))
 bad(lambda:d.config(d.rows()[0],20));bad(lambda:d.reference_row(dict(arm='OTHER',row_id='OTHER-s2026092701',model_seed=d.SEED)))
 check_default_head((6,160))
 for shape in [(5,160),(6,64),(6,160,1)]:bad(lambda shape=shape:check_default_head(shape))
 assert cumulative(1,1)==(21,4441) and cumulative(10,2220)==(30,6660)
 for ep,step in [(0,0),(1,223),(11,2221),(10,1)]:bad(lambda ep=ep,step=step:cumulative(ep,step))
 for row in d.rows():
  ref=d.reference_row(row);cp=dict(epoch=20,a1_ema_successful_updates=4440,run_id=d.REFERENCE_RUN,candidate_id=ref['row_id'],checkpoint_selection='final_only',ema_model={'mock':1},args=dict(seed=d.SEED,baseline_ckpt='',teacher_ckpt='',epochs=20,use_ema_teacher=True,label_smoothing=.01,batch_size=128))
  def verify(v):return check_checkpoint(v,run=d.REFERENCE_RUN,row=ref['row_id'],epoch=20,updates=4440,baseline='')
  verify(cp)
  for k,v in [('epoch',30),('a1_ema_successful_updates',6660),('run_id','other'),('candidate_id',row['row_id']),('checkpoint_selection','target_best'),('ema_model',None)]:
   z=copy.deepcopy(cp);z[k]=v;bad(lambda z=z:verify(z))
  for k,v in [('seed',2026092702),('teacher_ckpt','external'),('baseline_ckpt','external')]:
   z=copy.deepcopy(cp);z['args'][k]=v;bad(lambda z=z:verify(z))
  tail=copy.deepcopy(cp);tail.update(epoch=10,a1_ema_successful_updates=2220,run_id=d.RUN,candidate_id=row['row_id']);tail['args'].update(epochs=10,baseline_ckpt='own20/final_ssdg.pth',amp=False)
  check_checkpoint(tail,run=d.RUN,row=row['row_id'],epoch=10,updates=2220,baseline='own20/final_ssdg.pth');tail['args']['amp']=True;bad(lambda:check_checkpoint(tail,run=d.RUN,row=row['row_id'],epoch=10,updates=2220,baseline='own20/final_ssdg.pth'))
 expected=dict(checkpoint_sources=['parent/final_ssdg.pth'],parent_source='parent');done=dict(config=expected,epoch=10,optimizer_steps=2220,target_access=False,target_evaluated=False,head_fit_updates=0);frozen=dict(config=expected,selection='final_only',target_access=False);init=dict(target_contact=False,checkpoint_sources=expected['checkpoint_sources'],scratch_only=False,ancestors=['parent']);contract=dict(role_ids={'L':['fixed']})
 def metadata(dd=done,ff=frozen,ii=init,ss=contract):check_source_metadata(dd,ff,ii,ss,expected,contract['role_ids'])
 metadata()
 for k,v in [('target_access',True),('target_evaluated',True),('head_fit_updates',1),('optimizer_steps',6660)]:
  z=copy.deepcopy(done);z[k]=v;bad(lambda z=z:metadata(dd=z))
 z=copy.deepcopy(init);z['ancestors']=['unknown'];bad(lambda:metadata(ii=z));bad(lambda:metadata(ss=dict(role_ids={'L':['different']})))
 # The native RD uses packet-mean CE and symmetric KL. Prove the selected64
 # component is .25 RD64 inside .5 RD128, without changing any native loss.
 rng=np.random.default_rng(19731);z0=rng.normal(size=(128,6));z1=rng.normal(size=(128,6));y=np.arange(128)%6
 def logsoft(x):a=x-x.max(1,keepdims=True);return a-np.log(np.exp(a).sum(1,keepdims=True))
 def per_packet(a,b):
  la,lb=logsoft(a),logsoft(b);pa,pb=np.exp(la),np.exp(lb);ce=-.5*((1-.01)*(la[np.arange(len(y)),y]+lb[np.arange(len(y)),y])+.01*(la.mean(1)+lb.mean(1)));kl=.5*((pa*(la-lb)).sum(1)+(pb*(lb-la)).sum(1));return ce+kl
 packets=per_packet(z0,z1)
 for step in (1,2,2220):
  mask=(np.arange(128)+step)%2!=0;assert mask.sum()==64
  selected=.25*packets[mask].mean();rest=.5*packets.mean()-selected
  assert abs(rest-.25*packets[~mask].mean())<1e-12 and abs(rest+selected-.5*packets.mean())<1e-12
 # AST ties the arithmetic and native update boundary to the real implementation.
 text=Path(__file__).with_name('model.py').read_text(encoding='utf8');tree=ast.parse(text);hook=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='hook');actual=set()
 for node in ast.walk(hook):
  if isinstance(node,ast.DictComp) and isinstance(node.key,ast.BinOp) and isinstance(node.key.left,ast.Constant) and node.key.left.value=='concat/':actual.update('concat/'+k for k in ast.literal_eval(node.generators[0].iter))
  if isinstance(node,ast.Dict):actual.update(k.value for k in node.keys if isinstance(k,ast.Constant) and k.value in ('shape/RD','split/As_RD'))
 assert actual==set(INFO_KEYS) and len(actual)==9
 assert 'delta,old_info=original.hook(self,**kw)' in text and 'return delta,info' in text and 'calls==2' in text and 'calls!=3' in text and '.25*native_terms(z0[smask],z1[smask],y[smask],alpha)[0]' in text and 'result[0]*0.' in text
 assert 'core.capture(self,total_loss,component,optimizer,optimizer_step)' in text and 'post_core_auxiliary_loss' not in text and '_source_unlabeled_feature_hook' in text
 observer=Path(__file__).with_name('observer.py').read_text(encoding='utf8');assert 'total=physical_clip(params,max_norm,*args,**kwargs);core.after_clip(self.model,params,max_norm,total)' in observer and 'core.attach_optimizer(opt,self.model,amp=False)' in observer
 from . import report
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
  report.logical_rows=lambda:[(name,Path('SYNTHETIC')/name) for name in ('N01','C01','S01')];report.d.read=read_mock;report.d.write=lambda path,value:captured.append(value);report.np.load=lambda *args,**kwargs:Arrays();summary=report.source_decision();json.dumps(summary,allow_nan=False)
  assert type(summary['promotion']['passed']) is bool and summary['promotion']['passed'] is False and len(captured)==1 and summary['scientific_failure_skips_formal'] is False
 finally:report.logical_rows=old_rows;report.d.read=old_read;report.d.write=old_write;report.np.load=old_load
 for file in Path(__file__).parent.glob('*.py'):ast.parse(file.read_text(encoding='utf8'),filename=str(file))
 return dict(status='PASS',protocol_negatives=negative,three_new_matched_rows=True,original_ORD128_and_U=True,masked64_loss_arithmetic=True,native_full_L_plus_U_probe=True,physical_clip_precedes_component_clip=True,emitted_native_info_keys=len(actual),source_summary_native_JSON=True,real_Torch_or_checkpoint_smoke='launcher first step pending',NN=0,CP=0,IQ=0,updates=0,target=False)
if __name__=='__main__':print(json.dumps(run()))
