"""Focused resume/data/truth-scope negative tests; no model or IQ access."""
import copy,json
from . import design as d
from .protocol import cumulative,check_checkpoint,check_source_metadata

def run():
 negative=0
 def bad(f):
  nonlocal negative
  try:f()
  except (ValueError,TypeError,KeyError):negative+=1
  else:raise AssertionError('Invalid fixed E50 scope accepted')
 c=d.config(d.rows()[0]);d.validate(c)
 for k,v in [('target_access',True),('As',True),('epochs',50),('parent_epochs',20),('step_offset',4440),('cumulative_updates',11101),('model_seed',2026092702),('baseline_reuse',True),('Ag_mode','WHITE'),('pseudo_epochs',10),('label_epochs',13),('parent_source','unverified')]:
  z=copy.deepcopy(c);z[k]=v;bad(lambda z=z:d.validate(z))
 bad(lambda:d.config(d.rows()[0],20));bad(lambda:d.config(dict(arm='AS_DET',row_id='unknown',model_seed=d.SEED)))
 assert cumulative(1,1)==(31,6661) and cumulative(20,4440)==(50,11100)
 for ep,step in [(0,0),(1,223),(21,4441),(10,4440)]:bad(lambda ep=ep,step=step:cumulative(ep,step))
 cp=dict(epoch=10,a1_ema_successful_updates=2220,run_id=d.REFERENCE_RUN,candidate_id=c['row_id'],checkpoint_selection='final_only',ema_model={'mock':1},args=dict(baseline_ckpt='stage20/final_ssdg.pth',teacher_ckpt='',epochs=10,use_ema_teacher=True,label_smoothing=.01,batch_size=128))
 def verify(v):return check_checkpoint(v,run=d.REFERENCE_RUN,row=c['row_id'],epoch=10,updates=2220,baseline='stage20/final_ssdg.pth')
 verify(cp)
 for k,v in [('epoch',30),('a1_ema_successful_updates',6660),('run_id','other'),('checkpoint_selection','target_best'),('ema_model',None)]:
  z=copy.deepcopy(cp);z[k]=v;bad(lambda z=z:verify(z))
 z=copy.deepcopy(cp);z['args']['teacher_ckpt']='external';bad(lambda:verify(z))
 expected=dict(checkpoint_sources=['parent/final_ssdg.pth'],parent_source='parent');done=dict(config=expected,epoch=10,optimizer_steps=2220,target_access=False,target_evaluated=False,head_fit_updates=0);frozen=dict(config=expected,selection='final_only',target_access=False);init=dict(target_contact=False,checkpoint_sources=expected['checkpoint_sources'],scratch_only=False,ancestors=['parent']);contract=dict(role_ids={'L':['fixed']})
 def metadata(dd=done,ff=frozen,ii=init,ss=contract):check_source_metadata(dd,ff,ii,ss,expected,contract['role_ids'],10,2220)
 metadata()
 for k,v in [('target_access',True),('target_evaluated',True),('head_fit_updates',1),('optimizer_steps',6660)]:
  z=copy.deepcopy(done);z[k]=v;bad(lambda z=z:metadata(dd=z))
 z=copy.deepcopy(init);z['ancestors']=['unknown'];bad(lambda:metadata(ii=z))
 bad(lambda:metadata(ss=dict(role_ids={'L':['different']})))
 return dict(status='PASS',protocol_negatives=negative,cumulative_epoch31_to50=True,local_EMA_2220_not_cumulative6660=True,strict_student_ownEMA_lineage=True,CVS_NN=0,IQ=0,CP=0,target=False,updates=0)
if __name__=='__main__':print(json.dumps(run()))
