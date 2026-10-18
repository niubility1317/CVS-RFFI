"""Focused matrix/seed/lineage/truth-negative checks; no Torch, CP or IQ access."""
import ast,copy,json,sys,types
from pathlib import Path
from types import SimpleNamespace
from . import design as d
from .protocol import cumulative,check_checkpoint,check_source_metadata

def run():
 negative=0
 def bad(f):
  nonlocal negative
  try:f()
  except (ValueError,TypeError,KeyError,StopIteration):negative+=1
  else:raise AssertionError('Invalid fixed E50 replication scope accepted')
 assert [r['row_id'] for r in d.rows()]==[a+'-s'+str(s) for s in (2026092702,2026092703) for a in ('B00','B01')]
 assert len(d.rows())==4 and all(r['model_seed']!=2026092701 for r in d.rows())
 for row in d.rows():
  c=d.config(row);d.validate(c)
  assert c['parent_source']==str(Path(d.PROJECT)/'runs'/d.REFERENCE_RUN/row['row_id']/'source') and list(map(Path,c['checkpoint_sources']))==[Path(c['parent_source'])/'final_ssdg.pth']
  assert (c['epochs'],c['updates'],c['cumulative_epochs'],c['cumulative_updates'])==(20,4440,50,11100)
  assert c['data_contract']==dict(RX=[1,3,4,6,8],days=[1,2,3],L=6300,U=56700,V=27000,split=392005,source_RX_holdout=None,internal_L_split=False,U_TX_truth_hidden=True)
  assert c['seed_binding']['F_factory_model_seed']==row['model_seed'] and c['training_mechanism']['same_IQ_within_RDpair'] and c['training_mechanism']['augmentation_redraws_in_second_forward']==0
  assert c['As']==(row['arm']=='B01') and not c['new_Ag'] and c['training_mechanism']['L_forwards_per_sample']==6
  for k,v in [('target_access',True),('epochs',50),('parent_epochs',20),('step_offset',4440),('cumulative_updates',11101),('model_seed',2026092701),('baseline_reuse',True),('Ag_mode','WHITE'),('pseudo_epochs',10),('label_epochs',13),('parent_source','unverified')]:
   z=copy.deepcopy(c);z[k]=v;bad(lambda z=z:d.validate(z))
  z=copy.deepcopy(c);z['seed_binding']['F_factory_model_seed']=2026092701;bad(lambda z=z:d.validate(z))
 bad(lambda:d.config(d.rows()[0],20));bad(lambda:d.config(dict(arm='B00',row_id='B00-s2026092701',model_seed=2026092701)))
 assert cumulative(1,1)==(31,6661) and cumulative(20,4440)==(50,11100)
 for ep,step in [(0,0),(1,223),(21,4441),(10,4440)]:bad(lambda ep=ep,step=step:cumulative(ep,step))
 for row in d.rows():
  cp=dict(epoch=10,a1_ema_successful_updates=2220,run_id=d.REFERENCE_RUN,candidate_id=row['row_id'],checkpoint_selection='final_only',ema_model={'mock':1},args=dict(seed=row['model_seed'],baseline_ckpt='stage20/final_ssdg.pth',teacher_ckpt='',epochs=10,use_ema_teacher=True,label_smoothing=.01,batch_size=128))
  def verify(v):return check_checkpoint(v,seed=row['model_seed'],run=d.REFERENCE_RUN,row=row['row_id'],epoch=10,updates=2220,baseline='stage20/final_ssdg.pth')
  verify(cp)
  for k,v in [('epoch',30),('a1_ema_successful_updates',6660),('run_id','other'),('candidate_id','B00-s2026092701'),('checkpoint_selection','target_best'),('ema_model',None)]:
   z=copy.deepcopy(cp);z[k]=v;bad(lambda z=z:verify(z))
  for k,v in [('seed',2026092701),('teacher_ckpt','external'),('baseline_ckpt','other_seed/final_ssdg.pth')]:
   z=copy.deepcopy(cp);z['args'][k]=v;bad(lambda z=z:verify(z))
 expected=dict(checkpoint_sources=['parent/final_ssdg.pth'],parent_source='parent');done=dict(config=expected,epoch=10,optimizer_steps=2220,target_access=False,target_evaluated=False,head_fit_updates=0);frozen=dict(config=expected,selection='final_only',target_access=False);init=dict(target_contact=False,checkpoint_sources=expected['checkpoint_sources'],scratch_only=False,ancestors=['parent']);contract=dict(role_ids={'L':['fixed']})
 def metadata(dd=done,ff=frozen,ii=init,ss=contract):check_source_metadata(dd,ff,ii,ss,expected,contract['role_ids'],10,2220)
 metadata()
 for k,v in [('target_access',True),('target_evaluated',True),('head_fit_updates',1),('optimizer_steps',6660)]:
  z=copy.deepcopy(done);z[k]=v;bad(lambda z=z:metadata(dd=z))
 z=copy.deepcopy(init);z['ancestors']=['unknown'];bad(lambda:metadata(ii=z))
 bad(lambda:metadata(ss=dict(role_ids={'L':['different']})))
 # The parent factory supplies recipe flags; the continuation must override real native seed.
 name='experiments.cvs_As_replicate49';prior=sys.modules.get(name);fake=types.ModuleType(name)
 fake.design=SimpleNamespace(rows=d.rows,config=lambda row,segment:dict(row,segment=segment),make_args=lambda c,device:SimpleNamespace(seed=2026092701,label_smoothing=.01,use_aug=True,use_concat_sat_channel_aug=False,use_ema_teacher=True,batch_size=128))
 sys.modules[name]=fake
 try:
  for row in d.rows():
   c=d.config(row);a=d.make_args(c)
   assert a.seed==row['model_seed'] and a.candidate_id==row['row_id'] and a.run_id==d.RUN and a.baseline_ckpt==c['parent_source']+'/final_ssdg.pth' and (a.epochs,a.label_epochs,a.pseudo_epochs)==(20,0,20) and not a.a1_ema_startup_average
 finally:
  if prior is None:sys.modules.pop(name,None)
  else:sys.modules[name]=prior
 from .report import logical_rows,pairs_for,ALL_SEEDS
 logical=logical_rows();assert len(logical)==12 and len({label for label,_ in logical})==12 and tuple(ALL_SEEDS)==(2026092701,2026092702,2026092703)
 assert sum(root.parent==d.BASE for _,root in logical)==4
 for seed in ALL_SEEDS:
  assert len(pairs_for(seed))==4 and all(a.endswith(str(seed)) and b.endswith(str(seed)) for _,a,b in pairs_for(seed))
 for file in Path(__file__).parent.glob('*.py'):ast.parse(file.read_text(encoding='utf8'),filename=str(file))
 # The registered namespace actually supplies unchanged helpers used by all four rows.
 from . import evaluate
 assert evaluate.__package__==__package__
 return dict(status='PASS',protocol_negatives=negative,positive_native_seed_bindings=4,registered_new_rows=4,total_report_model_seeds=3,seed01_retrained=False,cumulative_epoch31_to50=True,local_EMA_2220_not_cumulative6660=True,strict_student_ownEMA_lineage=True,CVS_NN=0,IQ=0,CP=0,target=False,updates=0)
if __name__=='__main__':print(json.dumps(run()))
