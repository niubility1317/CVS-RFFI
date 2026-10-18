"""Focused fixed-budget, lineage, source-only decision and dropout-scope checks."""
import copy,json
from . import design as d
from .protocol import cumulative,check_checkpoint,check_source_metadata,capacity_available

def run():
 negative=0
 assert capacity_available(dict(pids={1},free_mb=20000))
 assert not capacity_available(dict(pids={1,2},free_mb=20000))
 assert not capacity_available(dict(pids={1},free_mb=20000),{2})
 assert not capacity_available(dict(pids=set(),free_mb=11999))
 def bad(f):
  nonlocal negative
  try:f()
  except (ValueError,TypeError,KeyError):negative+=1
  else:raise AssertionError('Invalid fixed P62 scope accepted')
 c=d.config(d.rows()[0]);d.validate(c)
 for k,v in [('target_access',True),('As',False),('new_Ag',True),('epochs',50),('parent_epochs',20),('step_offset',4440),('cumulative_updates',11101),('model_seed',2026092702),('baseline_reuse',True),('Ag_mode','WHITE'),('pseudo_epochs',10),('label_epochs',13),('parent_source','unverified'),('satellite_mode','AS_RD')]:
  z=copy.deepcopy(c);z[k]=v;bad(lambda z=z:d.validate(z))
 bad(lambda:d.config(d.rows()[0],20));bad(lambda:d.config(dict(arm='AS_RD',row_id='unknown',model_seed=d.SEED)))
 assert cumulative(1,1)==(31,6661) and cumulative(20,4440)==(50,11100)
 for ep,step in [(0,0),(1,223),(21,4441),(10,4440),(30,6660)]:bad(lambda ep=ep,step=step:cumulative(ep,step))
 cp=dict(epoch=10,a1_ema_successful_updates=2220,run_id=d.PARENT_RUN,candidate_id=c['row_id'],checkpoint_selection='final_only',ema_model={'mock':1},args=dict(baseline_ckpt='stage20/final_ssdg.pth',teacher_ckpt='',epochs=10,use_ema_teacher=True,label_smoothing=.01,batch_size=128))
 def verify(v):return check_checkpoint(v,run=d.PARENT_RUN,row=c['row_id'],epoch=10,updates=2220,baseline='stage20/final_ssdg.pth')
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
 from .report import decision_from_rows
 rates={'NOAS50':{'full_V_clean':.9},'DET50':{'full_V_clean':.899,'mid_low_mean':.605,'panel':{'practical_low_urban':{'accuracy':.55}}},'CE50':{'mid_low_mean':.6,'panel':{'practical_low_urban':{'accuracy':.55}}}}
 v=decision_from_rows(rates);assert v['source_qualified'] is True
 rates['DET50']['full_V_clean']=.8989;assert decision_from_rows(rates)['source_qualified'] is False
 rates['DET50']['full_V_clean']=.9;rates['DET50']['mid_low_mean']=.604;assert decision_from_rows(rates)['source_qualified'] is False
 rates['DET50']['mid_low_mean']=.605;rates['DET50']['panel']['practical_low_urban']['accuracy']=.549;assert decision_from_rows(rates)['source_qualified'] is False
 assert len(d.logical_rows())==8 and sum(not q[2] for q in d.logical_rows())==2
 json.dumps(v,allow_nan=False)
 return dict(status='PASS',protocol_negatives=negative,source_bool_JSON=True,cumulative_epoch31_to50=True,local_EMA_4440_not_cumulative11100=True,strict_student_ownEMA_lineage=True,all_eight_logical_rows=True,CVS_NN=0,IQ=0,CP=0,target=False,updates=0)
if __name__=='__main__':print(json.dumps(run()))
