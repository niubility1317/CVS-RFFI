"""Fixed source/default-head/teacher protocol negatives and tensor-only teaching."""
import copy,json
from . import design as d
from .protocol import final_metadata,checkpoint

def run():
 negative=0
 def bad(f):
  nonlocal negative
  try:f()
  except (ValueError,TypeError,KeyError):negative+=1
  else:raise AssertionError('Invalid fixed U-quality66 scope accepted')
 assert tuple(r['arm'] for r in d.rows())==('MASS_UNIFORM','OBSERVATION_WEIGHTED')
 parents=[]
 for row in d.rows():
  c=d.config(row);d.validate(c);parents.append(c['parent_source'])
  for k,v in [('target_access',True),('As',False),('new_Ag',True),('epochs',20),('updates',4440),('model_seed',2026092702),('parent_source','unknown'),('label_epochs',13),('Ag_mode','WHITE'),('teacher_ckpt','unverified'),('checkpoint_sources',[])]:
   z=copy.deepcopy(c);z[k]=v;bad(lambda z=z:d.validate(z))
  bad(lambda row=row:d.config(row,20))
 assert len(set(parents))==1 and len(c['checkpoint_sources'])==1
 c=d.config(d.rows()[0]);bad(lambda:d.config(dict(arm='other',row_id='x',model_seed=d.SEED)))
 roles={'L':['toyL'],'U':['toyU'],'V':['toyV']};done=dict(config=c,epoch=10,optimizer_steps=2220,target_access=False,head_fit_updates=0);frozen=dict(config=c,selection='final_only',target_access=False);init=dict(scratch_only=False,target_contact=False,checkpoint_sources=c['checkpoint_sources'],source_roles='EXACT_MATCH');contract=dict(role_ids=roles)
 final_metadata(c,done,frozen,init,contract,roles)
 for key,val in [('target_access',True),('epoch',20),('optimizer_steps',4440),('head_fit_updates',1)]:
  z=copy.deepcopy(done);z[key]=val;bad(lambda z=z:final_metadata(c,z,frozen,init,contract,roles))
 z=copy.deepcopy(init);z['checkpoint_sources']=[];bad(lambda:final_metadata(c,done,frozen,z,contract,roles))
 cp=dict(epoch=10,a1_ema_successful_updates=2220,run_id=d.RUN,candidate_id=c['row_id'],checkpoint_selection='final_only',ema_model={},args=dict(baseline_ckpt=c['parent_source']+'/final_ssdg.pth',teacher_ckpt=''))
 checkpoint(c,cp)
 for key,val in [('run_id',d.REFERENCE_RUN),('candidate_id','B01-s2026092701'),('epoch',20),('ema_model',None)]:
  z=copy.deepcopy(cp);z[key]=val;bad(lambda z=z:checkpoint(c,z))
 z=copy.deepcopy(cp);z['args']['teacher_ckpt']='unverified';bad(lambda:checkpoint(c,z))
 from .test_teaching import run as teaching_checks
 teaching=teaching_checks()
 return dict(status='PASS',protocol_negatives=negative,teaching_checks=teaching,both_students_same_As20_parent=True,one_unique_checkpoint_source=True,CVS_NN=0,IQ=0,CP=0,target=False,optimizer_updates=0)
if __name__=='__main__':print(json.dumps(run()))
