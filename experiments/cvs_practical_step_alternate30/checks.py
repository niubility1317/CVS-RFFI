"""Focused data/provenance negatives plus actual mixed-view function with two-step mocks."""
import ast,copy,json
from pathlib import Path
from . import design as d
from .protocol import cumulative,slot_plan,check_checkpoint,check_source_metadata

def run():
 negative=0
 def bad(f):
  nonlocal negative
  try:f()
  except (ValueError,TypeError,KeyError,StopIteration):negative+=1
  else:raise AssertionError('Invalid fixed stepALT scope accepted')
 c=d.config(d.rows()[0]);d.validate(c)
 assert (c['epochs'],c['updates'],c['cumulative_epochs'],c['cumulative_updates'])==(10,2220,30,6660)
 assert c['data_contract']==dict(RX=[1,3,4,6,8],days=[1,2,3],L=6300,U=56700,V=27000,split=392005,source_RX_holdout=None,internal_L_split=False,U_TX_truth_hidden=True)
 for k,v in [('target_access',True),('epochs',30),('updates',6660),('parent_epochs',30),('step_offset',0),('model_seed',2026092702),('baseline_reuse',True),('Ag_mode','WHITE'),('As',False),('label_epochs',13),('parent_source','unknown')]:
  z=copy.deepcopy(c);z[k]=v;bad(lambda z=z:d.validate(z))
 bad(lambda:d.config(d.rows()[0],20));bad(lambda:d.config(dict(arm='B00',row_id='B00-s2026092701',model_seed=d.SEED)))
 assert cumulative(1,1)==(21,4441) and cumulative(10,2220)==(30,6660)
 for ep,step in [(0,0),(1,223),(11,2221),(10,1)]:bad(lambda ep=ep,step=step:cumulative(ep,step))
 total=0
 for ep in range(1,11):
  plans=[slot_plan(ep,s) for s in range((ep-1)*222+1,ep*222+1)]
  assert sum(p['As_packets'] for p in plans)==14208 and sum(p['on'] for p in plans)==111
  total+=sum(p['As_packets'] for p in plans)
 assert total==142080
 cp=dict(epoch=20,a1_ema_successful_updates=4440,run_id=d.REFERENCE_RUN,candidate_id=c['row_id'],checkpoint_selection='final_only',ema_model={'mock':1},args=dict(seed=d.SEED,baseline_ckpt='',teacher_ckpt='',epochs=20,use_ema_teacher=True,label_smoothing=.01,batch_size=128))
 def verify(v):return check_checkpoint(v,run=d.REFERENCE_RUN,row=c['row_id'],epoch=20,updates=4440,baseline='')
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
 # Execute the actual pure view organizer without importing Torch/model/native/data.
 tree=ast.parse(Path(__file__).with_name('model.py').read_text(encoding='utf8'))
 definition=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='mixed_view');scope=dict(slot_plan=slot_plan)
 exec(compile(ast.Module(body=[definition],type_ignores=[]),'actual_mixed_view','exec'),scope)
 class Batch:
  shape=(128,2,256)
  def __init__(self,token):self.token=token
  def clone(self):return Batch(self.token)
 class Channel:
  def __init__(self):self.calls=[]
  def apply(self,raw,meta,epoch,step):self.calls.append((raw.token,meta,epoch,step));return Batch('satellite'),[{} for _ in range(128)]
 channel=Channel();raw=Batch('raw');ordinary=Batch('same actual ordinary');meta={'physical':'mock only'}
 on,obs,plan=scope['mixed_view'](channel,raw,ordinary,meta,1,1)
 assert on.token=='satellite' and len(obs)==128 and plan['As_packets']==128
 off,obs,plan=scope['mixed_view'](channel,raw,ordinary,meta,1,2)
 assert off.token==ordinary.token and off is not ordinary and not obs and plan['As_packets']==0 and channel.calls==[('raw',meta,1,1)]
 malformed=Batch('invalid');malformed.shape=(0,2,256);bad(lambda:scope['mixed_view'](channel,malformed,ordinary,meta,1,1))
 for file in Path(__file__).parent.glob('*.py'):ast.parse(file.read_text(encoding='utf8'),filename=str(file))
 return dict(status='PASS',protocol_negatives=negative,actual_mixed_function_mock_steps=2,mock_channel_calls=1,mock_As=[128,0],As_per_epoch=14208,As_total=142080,NN=0,CP=0,IQ=0,updates=0,target=False)
if __name__=='__main__':print(json.dumps(run()))
