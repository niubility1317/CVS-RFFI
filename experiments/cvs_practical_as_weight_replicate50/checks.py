"""Focused fixed seed/stage/parent/true-factory/conditional eligibility and paths."""
import ast,copy,json,sys,types
from pathlib import Path
import numpy as np
from . import design as d
from .protocol import cumulative,check_checkpoint,check_metadata,check_factory,eligibility
from .schema import INFO_KEYS

def run():
 negative=0
 def bad(f):
  nonlocal negative
  try:f()
  except (ValueError,KeyError,TypeError,StopIteration):negative+=1
  else:raise AssertionError('Invalid fixed seed/stage scope accepted')
 assert len(d.rows())==4 and {r['model_seed'] for r in d.rows()}=={2026092702,2026092703}
 for row in d.rows():
  for budget in (30,50):
   c=d.config(row,budget);d.validate(c);n=10 if budget==30 else 20;offset=4440 if budget==30 else 6660
   assert (c['epochs'],c['updates'],c['step_offset'],c['cumulative_epochs'],c['cumulative_updates'])==(n,n*222,offset,budget,6660 if budget==30 else 11100)
   assert Path(c['output_root'])==d.stage_root(row,budget)/'source' and c['seed_binding']['F_factory_model_seed']==row['model_seed']
   if budget==50:assert Path(c['parent_source'])==d.stage_root(row,30)/'source'
   else:assert Path(c['parent_source'])==Path(d.PROJECT)/'runs'/d.PARENT_RUN/row['row_id']/'stage20'
   check_factory(row['model_seed'],c);bad(lambda:check_factory(2026092701,c))
   for key,value in [('model_seed',2026092701),('epochs',50),('updates',1),('step_offset',0),('parent_source','unknown'),('target_budget',20),('target_access',True),('As',not c['As'])]:
    z=copy.deepcopy(c);z[key]=value;bad(lambda z=z:d.validate(z))
   done=dict(config=c,epoch=n,optimizer_steps=n*222,target_access=False,target_evaluated=False,head_fit_updates=0,cumulative_epochs=budget,cumulative_optimizer_steps=c['cumulative_updates']);frozen=dict(config=c,selection='final_only',target_access=False);init=dict(target_contact=False,scratch_only=False,checkpoint_sources=c['checkpoint_sources'],ancestors=[c['parent_source']]);contract=dict(role_ids={'L':['fixed']})
   def metadata(v=done):return check_metadata(v,frozen,init,contract,c,contract['role_ids'])
   metadata()
   for key,value in [('cumulative_epochs',n),('cumulative_optimizer_steps',n*222),('optimizer_steps',6660),('target_access',True),('head_fit_updates',1)]:
    z=copy.deepcopy(done);z[key]=value;bad(lambda z=z:metadata(z))
   cp=dict(epoch=n,a1_ema_successful_updates=n*222,run_id=d.RUN,candidate_id=row['row_id'],checkpoint_selection='final_only',ema_model={'mock':1},args=dict(seed=row['model_seed'],baseline_ckpt=c['parent_source']+'/final_ssdg.pth',teacher_ckpt='',epochs=n,use_ema_teacher=True,label_smoothing=.01,batch_size=128))
   def verify(v):return check_checkpoint(v,seed=row['model_seed'],run=d.RUN,row=row['row_id'],epoch=n,updates=n*222,baseline=c['parent_source']+'/final_ssdg.pth')
   verify(cp)
   for key,value in [('run_id','other'),('candidate_id','seed01'),('epoch',50),('a1_ema_successful_updates',6660),('checkpoint_selection','target_best')]:
    z=copy.deepcopy(cp);z[key]=value;bad(lambda z=z:verify(z))
   z=copy.deepcopy(cp);z['args']['seed']=2026092701;bad(lambda:verify(z))
 for budget,end_ep,end_step,first,last in ((30,10,2220,(21,4441),(30,6660)),(50,20,4440,(31,6661),(50,11100))):
  assert cumulative(1,1,budget)==first and cumulative(end_ep,end_step,budget)==last
  for ep,step in ((0,0),(1,223),(end_ep+1,end_step+1)):bad(lambda ep=ep,step=step: cumulative(ep,step,budget))
 summary=dict(run_id=d.SEED01_WEIGHT50_RUN,target_scores_read=False,NN=0,IQ_reads=0,checkpoint_reads=0,updates=0,decision=dict(source_criteria_pass=True,fixed_02_03_source_eligible=True,clean_nonnegative=True,mid_low_gain_at_least_0p5pp=True,newAs50_minus_ownAs30_fullV_clean_pp=0.,newAs50_minus_ownAs30_mid_low_mean_pp=.5));eligibility(summary,d.SEED01_WEIGHT50_RUN)
 for key,value in [('target_scores_read',True),('NN',1),('run_id','other')]:
  z=copy.deepcopy(summary);z[key]=value;bad(lambda z=z:eligibility(z,d.SEED01_WEIGHT50_RUN))
 for key,value in [('fixed_02_03_source_eligible',False),('newAs50_minus_ownAs30_fullV_clean_pp',-.001),('newAs50_minus_ownAs30_mid_low_mean_pp',.49)]:
  z=copy.deepcopy(summary);z['decision'][key]=value;bad(lambda z=z:eligibility(z,d.SEED01_WEIGHT50_RUN))
 # Execute stage-aware prediction preflight on synthetic logits only, no formal arrays/truth.
 from . import evaluate as ev
 old_count,old_obs,old_read,old_load=ev.COUNT,ev.observations,ev.d.read,np.load;ids=np.arange(8);visited=[]
 class Loaded:
  def __init__(self,p):self.p=Path(p)
  def __enter__(self):
   if self.p.name=='predictions.npz':return dict(ids=ids,**{v:np.zeros(8,dtype=np.int64) for v in d.VIEWS})
   return dict(ids=ids,logits=np.tile(np.asarray([[1.,0,0,0,0,0]]),(8,1)))
  def __exit__(self,*unused):pass
 class ArrayFile(dict):
  @property
  def files(self):return list(self)
 try:
  ev.COUNT=8;ev.observations=lambda:(ids,{})
  class Data(Loaded):
   def __enter__(self):return ArrayFile(super().__enter__())
  for row in d.rows():
   for budget in (30,50):
    c=d.config(row,budget)
    def read(path,c=c):visited.append(Path(path));return dict(status='PREDICTIONS_COMPLETE',config=c,count=8,views=list(d.VIEWS),truth_read=False,query_fit=False,all_parameters_frozen=True,head_fit_updates=0)
    ev.d.read=read;np.load=lambda path,**kw:Data(path);assert ev.preflight(row['row_id'],budget)[0]==c
 finally:ev.COUNT=old_count;ev.observations=old_obs;ev.d.read=old_read;np.load=old_load
 assert len(visited)==8 and len(set(visited))==8
 # True native seed argument is applied to every stage, independent of seed01 recipe template.
 name='experiments.cvs_As_replicate49';prior=sys.modules.get(name);fake=types.ModuleType(name)
 fake.design=types.SimpleNamespace(rows=d.rows,config=lambda row,segment:dict(row,segment=segment),make_args=lambda c,device:types.SimpleNamespace(seed=2026092701,batch_size=128,label_smoothing=.01,use_aug=True,use_concat_sat_channel_aug=False,use_ema_teacher=True))
 sys.modules[name]=fake
 try:
  for row in d.rows():
   for budget in (30,50):
    c=d.config(row,budget);a=d.make_args(c);assert a.seed==row['model_seed'] and a.epochs==c['epochs'] and a.baseline_ckpt==c['parent_source']+'/final_ssdg.pth'
 finally:
  if prior is None:sys.modules.pop(name,None)
  else:sys.modules[name]=prior
 assert len(INFO_KEYS)==21
 for file in Path(__file__).parent.glob('*.py'):ast.parse(file.read_text(encoding='utf8'),filename=str(file))
 return dict(status='PASS',protocol_negatives=negative,new_trajectories=4,new_stage_rows=8,true_seed_arguments=8,staged_prediction_paths=8,new_model_epochs=120,new_updates=26640,conditional_source_rule=True,intermediate_scientific_scores_never_select_continuation=True,NN=0,CP=0,IQ=0,updates=0,remote_actions=0,target=False)
if __name__=='__main__':print(json.dumps(run()))
