"""Focused new FINAL30-parent/20-epoch/cumulative-window negatives; no CP/IQ/model."""
import ast,copy,json
from pathlib import Path
from . import design as d
from .protocol import cumulative,check_checkpoint,check_source_metadata,check_default_head,check_parent_budget
from .schema import INFO_KEYS

def run():
 negative=0
 def bad(f):
  nonlocal negative
  try:f()
  except (ValueError,KeyError,TypeError,StopIteration):negative+=1
  else:raise AssertionError('Invalid weighted50 scope accepted')
 for row in d.rows():
  c=d.config(row);d.validate(c)
  assert (c['epochs'],c['label_epochs'],c['pseudo_epochs'],c['updates'],c['epoch_offset'],c['step_offset'],c['cumulative_epochs'],c['cumulative_updates'])==(20,0,20,4440,30,6660,50,11100)
  assert c['data_contract']==dict(RX=[1,3,4,6,8],days=[1,2,3],L=6300,U=56700,V=27000,split=392005,source_RX_holdout=None,internal_L_split=False,U_TX_truth_hidden=True)
  assert c['mixed_weight']['ordinary_group_weight']==.125 and c['mixed_weight']['selected_group_weight']==.375 and c['mixed_weight']['total_L_scale']==1.25
  assert Path(c['parent_source'])==Path(d.parent_config(c)['output_root']) and c['As']==(row['arm']=='B01')
  for key,value in [('epochs',50),('updates',2220),('parent_epochs',20),('step_offset',4440),('epoch_offset',20),('cumulative_updates',6660),('model_seed',2026092702),('As',not c['As']),('parent_source','oldoriginal50'),('target_access',True),('baseline_reuse',True),('separate_head_training',True)]:
   z=copy.deepcopy(c);z[key]=value;bad(lambda z=z:d.validate(z))
  z=copy.deepcopy(c);z['mixed_weight']['selected_group_weight']=.5;bad(lambda:d.validate(z))
  cp=dict(epoch=10,a1_ema_successful_updates=2220,run_id=d.PARENT_RUN,candidate_id=row['row_id'],checkpoint_selection='final_only',ema_model={'mock':1},args=dict(seed=d.SEED,baseline_ckpt='ownoriginal20/final_ssdg.pth',teacher_ckpt='',epochs=10,use_ema_teacher=True,label_smoothing=.01,batch_size=128))
  def verify(v):return check_checkpoint(v,run=d.PARENT_RUN,row=row['row_id'],epoch=10,updates=2220,baseline='ownoriginal20/final_ssdg.pth')
  verify(cp)
  for key,value in [('epoch',30),('a1_ema_successful_updates',6660),('run_id',d.REFERENCE_RUN),('candidate_id','other'),('checkpoint_selection','target_best'),('ema_model',None)]:
   z=copy.deepcopy(cp);z[key]=value;bad(lambda z=z:verify(z))
  for key,value in [('seed',2026092702),('baseline_ckpt','oldoriginal50/final_ssdg.pth'),('teacher_ckpt','external'),('epochs',30)]:
   z=copy.deepcopy(cp);z['args'][key]=value;bad(lambda z=z:verify(z))
 bad(lambda:d.config(d.rows()[0],20));bad(lambda:d.config(dict(arm='B11',row_id='B11-s2026092701',model_seed=d.SEED)))
 assert cumulative(1,1)==(31,6661) and cumulative(20,4440)==(50,11100)
 for ep,step in [(0,0),(1,223),(21,4441),(10,4440)]:bad(lambda ep=ep,step=step:cumulative(ep,step))
 expected=dict(checkpoint_sources=['ownoriginal20/final_ssdg.pth'],parent_source='ownoriginal20');done=dict(config=expected,epoch=10,optimizer_steps=2220,target_access=False,target_evaluated=False,head_fit_updates=0);frozen=dict(config=expected,selection='final_only',target_access=False);init=dict(target_contact=False,checkpoint_sources=expected['checkpoint_sources'],scratch_only=False,ancestors=['ownoriginal20']);contract=dict(role_ids={'L':['fixed']})
 def metadata(dd=done,ff=frozen,ii=init,ss=contract):return check_source_metadata(dd,ff,ii,ss,expected,contract['role_ids'],10,2220)
 metadata()
 for key,value in [('epoch',30),('optimizer_steps',6660),('target_access',True),('head_fit_updates',1)]:
  z=copy.deepcopy(done);z[key]=value;bad(lambda z=z:metadata(dd=z))
 z=copy.deepcopy(init);z['ancestors']=['unknown'];bad(lambda:metadata(ii=z));bad(lambda:metadata(ss=dict(role_ids={'L':['different']})))
 config=dict(cumulative_epochs=30,cumulative_updates=6660);result=dict(cumulative_epochs=30,cumulative_optimizer_steps=6660,parent_epochs=20,parent_optimizer_steps=4440,U_truth_read=False,target_access=False,head_fit_updates=0);check_parent_budget(config,result)
 for key,value in [('cumulative_epochs',10),('cumulative_optimizer_steps',2220),('parent_epochs',30),('parent_optimizer_steps',6660),('U_truth_read',True),('target_access',True),('head_fit_updates',1)]:
  z=copy.deepcopy(result);z[key]=value;bad(lambda z=z:check_parent_budget(config,z))
 check_default_head((6,160))
 for shape in ((6,64),(5,160),(6,160,1)):bad(lambda shape=shape:check_default_head(shape))
 from .report import logical_rows
 logical=logical_rows();assert len(logical)==8 and len({label for label,_ in logical})==8 and sum(path.parent==d.BASE for _,path in logical)==2
 assert len(INFO_KEYS)==21 and 'shape/RD' not in INFO_KEYS
 tree=ast.parse(Path(__file__).with_name('model.py').read_text(encoding='utf8'));hook=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='hook');actual=set()
 for node in ast.walk(hook):
  if isinstance(node,ast.DictComp) and isinstance(node.key,ast.BinOp) and isinstance(node.key.left,ast.Constant) and isinstance(node.key.left.value,str):actual.update(node.key.left.value+k for k in ast.literal_eval(node.generators[0].iter))
  if isinstance(node,ast.Dict):actual.update(k.value for k in node.keys if isinstance(k,ast.Constant) and isinstance(k.value,str) and k.value.startswith('weight/'))
 assert actual==set(INFO_KEYS)
 for file in Path(__file__).parent.glob('*.py'):ast.parse(file.read_text(encoding='utf8'),filename=str(file))
 return dict(status='PASS',protocol_negatives=negative,parent_native_epoch10_EMA2220_cumulative30_6660=True,new_native_epoch20_updates4440_cumulative50_11100=True,cumulative_channel31_to50=True,logical_rows=8,new_rows=2,native_info_keys=21,unchanged_grouped_helper_reused=True,CP=0,IQ=0,NN=0,updates=0,target=False)
if __name__=='__main__':print(json.dumps(run()))
