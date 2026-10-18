"""Fixed scope negatives and actual128 membership selectors; no model or IQ."""
import ast,copy,json,sys,types
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
  else:raise AssertionError('Invalid fixed mixed128 scope accepted')
 assert [r['arm'] for r in d.rows()]==['B00','B01']
 for row in d.rows():
  c=d.config(row);d.validate(c)
  assert (c['epochs'],c['updates'],c['cumulative_epochs'],c['cumulative_updates'])==(10,2220,30,6660)
  assert c['data_contract']==dict(RX=[1,3,4,6,8],days=[1,2,3],L=6300,U=56700,V=27000,split=392005,source_RX_holdout=None,internal_L_split=False,U_TX_truth_hidden=True)
  assert c['As']==(row['arm']=='B01') and c['training_mechanism']['pair_weights']==[.5,.5,.25] and c['crossview']['mode']=='MIXED128' and c['mixed_contrast']['physical_ID_uniqueness_required'] is False
  assert d.reference_config(row,20)['arm']==row['arm'] and Path(c['parent_source'])==Path(d.reference_config(row,30)['parent_source'])
  for k,v in [('target_access',True),('epochs',30),('updates',6660),('parent_epochs',30),('step_offset',0),('model_seed',2026092702),('baseline_reuse',True),('Ag_mode','WHITE'),('As',not c['As']),('label_epochs',13),('parent_source','unknown'),('separate_head_training',True)]:
   z=copy.deepcopy(c);z[k]=v;bad(lambda z=z:d.validate(z))
  z=copy.deepcopy(c);z['crossview']['mode']='SAT192';bad(lambda:d.validate(z))
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
 # Execute the actual membership/metadata functions with array mocks; no Torch/model.
 tree=ast.parse(Path(__file__).with_name('relations.py').read_text(encoding='utf8'))
 body=[n for n in tree.body if isinstance(n,ast.Assign) or isinstance(n,ast.FunctionDef) and n.name in ('select_maps','metadata')]
 ns={'torch':types.SimpleNamespace(bool=np.dtype('bool'),long=np.dtype('int64'))};exec(compile(ast.Module(body=body,type_ignores=[]),'actual_membership','exec'),ns)
 levels=ns['LEVELS'];select=ns['select_maps'];metadata=ns['metadata'];mask=np.arange(128)%2==0
 ordinary={level:np.full((128,3,7),i,dtype=float) for i,level in enumerate(levels)};mixed={level:v+1 for level,v in ordinary.items()};y=np.arange(128,dtype=np.int64)%6;rx=np.asarray([1,3,4,6,8],dtype=np.int64)[np.arange(128)%5]
 chosen=select(ordinary,mixed,mask,'MIXED128');yy,rr=metadata(y,rx,mask,'MIXED128')
 assert all(chosen[level] is mixed[level] and chosen[level].shape[0]==128 for level in levels) and yy is y and rr is rx
 assert all(select(ordinary,mixed,mask,'ORD128')[level] is ordinary[level] for level in levels)
 # Sampler positions/metadata are preserved even when source physical metadata repeats.
 duplicated_y=np.zeros(128,dtype=np.int64);duplicated_rx=np.ones(128,dtype=np.int64)
 assert metadata(duplicated_y,duplicated_rx,mask,'MIXED128')[0] is duplicated_y
 for mode in ('SAT192','DUP192','UNKNOWN'):bad(lambda mode=mode:select(ordinary,mixed,mask,mode))
 for wrong in (np.zeros(128,dtype=bool),np.zeros(64,dtype=bool),mask.astype(int)):
  bad(lambda wrong=wrong:select(ordinary,mixed,wrong,'MIXED128'));bad(lambda wrong=wrong:metadata(y,rx,wrong,'MIXED128'))
 bad(lambda:select(ordinary,{level:np.zeros((192,3,7)) for level in levels},mask,'MIXED128'))
 for invalid in (-1,6):
  wrong=y.copy();wrong[0]=invalid;bad(lambda wrong=wrong:metadata(wrong,rx,mask,'MIXED128'))
 wrong=rx.copy();wrong[0]=2;bad(lambda:metadata(y,wrong,mask,'MIXED128'))
 bad(lambda:metadata(y[:64],rx[:64],mask,'MIXED128'))
 assert .5+.5+.25==1.25
 # Exact native telemetry names come from the actual hook AST, not historical metrics.
 mt=ast.parse(Path(__file__).with_name('model.py').read_text(encoding='utf8'));hook=next(n for n in mt.body if isinstance(n,ast.FunctionDef) and n.name=='hook');actual=set()
 for node in ast.walk(hook):
  if isinstance(node,ast.DictComp) and isinstance(node.key,ast.BinOp) and isinstance(node.key.left,ast.Constant) and isinstance(node.key.left.value,str):actual.update(node.key.left.value+k for k in ast.literal_eval(node.generators[0].iter))
 assert actual==set(INFO_KEYS) and len(actual)==14 and 'shape/RD' not in actual
 text=Path(__file__).with_name('model.py').read_text(encoding='utf8')
 assert 'total=.5*common+.5*extra_loss+weight*aux' in text and "record['total_L']+=.25*float(rd.detach())" in text and "record['L_forwards']=6" in text
 assert "[('first',first_maps,sat0),('second',second_maps,sat1)]" in text and "relations.penalty(a,b,labels,rx,smask,self._crossview_mode,actual_As=self._shape_As)" in text
 assert 'grouped(' not in text and '_weight_records' not in text and "original.noise.apply(raw,step,'NONE')" in text
 # Native JSON promotion result cannot retain NumPy booleans/scalars.
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
  if name in ('source_frozen.json','complete.json'):return dict(target_access=False)
  return dict(role_ids={'L':['mock']})
 captured=[]
 try:
  report.logical_rows=lambda:[(name,Path('SYNTHETIC')/name) for name in ('NOAS30','AS30','MIXED_NOAS30','MIXED_AS30')]
  report.d.read=read_mock;report.d.write=lambda path,value:captured.append(value);report.np.load=lambda *args,**kwargs:Arrays()
  summary=report.source_decision();json.dumps(summary,allow_nan=False)
  assert type(summary['promotion']['passed']) is bool and summary['promotion']['passed'] is False and len(captured)==1 and summary['scientific_failure_skips_formal'] is False
 finally:report.logical_rows=old_rows;report.d.read=old_read;report.d.write=old_write;report.np.load=old_load
 for file in Path(__file__).parent.glob('*.py'):ast.parse(file.read_text(encoding='utf8'),filename=str(file))
 return dict(status='PASS',protocol_negatives=negative,actual128_selector_mock=True,mixed_maps_reused_without_concat=True,original_sampler_not_deduplicated=True,emitted_native_info_keys=len(actual),full6L_total_metric=True,source_summary_native_JSON=True,real_Torch_or_checkpoint_smoke='launcher first step pending',NN=0,CP=0,IQ=0,updates=0,target=False)
if __name__=='__main__':print(json.dumps(run()))
