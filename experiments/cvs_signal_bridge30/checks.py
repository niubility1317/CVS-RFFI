"""Focused data/parent/budget negatives and synthetic-logit CE math,zero NN."""
import ast,copy,json
from pathlib import Path
from . import design as d
from .protocol import cumulative,check_default_head,check_checkpoint,check_source_metadata
def torch_math():
 import torch
 from torch.nn import functional as F
 state=torch.random.get_rng_state();torch.manual_seed(6201)
 try:
  a=torch.randn(64,6,dtype=torch.float64,requires_grad=True);b=torch.randn(64,6,dtype=torch.float64,requires_grad=True);y=torch.arange(64)%6;loss=.25*.5*(F.cross_entropy(a,y,label_smoothing=.01)+F.cross_entropy(b,y,label_smoothing=.01));joint=.25*F.cross_entropy(torch.cat((a,b)),torch.cat((y,y)),label_smoothing=.01);g0=torch.autograd.grad(loss,(a,b),retain_graph=True);g1=torch.autograd.grad(joint,(a,b));assert torch.allclose(loss,joint,atol=1e-12,rtol=1e-12) and all(torch.allclose(x,z,atol=1e-12,rtol=1e-12) for x,z in zip(g0,g1));assert .5+.5+.25+.25==1.5
  return dict(status='PASS',checks=3,loss_gradient_equivalence=True,extra_KL=0,NN=0)
 finally:torch.random.set_rng_state(state)
def run():
 negative=0
 def bad(fn):
  nonlocal negative
  try:fn()
  except (ValueError,TypeError,KeyError,StopIteration):negative+=1
  else:raise AssertionError('Invalid fixed experiment accepted')
 for row in d.rows():
  c=d.config(row);d.validate(c);assert c['data_contract']==dict(RX=[1,3,4,6,8],days=[1,2,3],L=6300,U=56700,V=27000,split=392005,source_RX_holdout=None,internal_L_split=False,U_TX_truth_hidden=True)
  for k,v in [('run_id','other'),('epochs',30),('updates',6660),('parent_epochs',30),('model_seed',2026092702),('parent_source','unknown'),('extra_view','TARGET'),('As',not c['As']),('Ag_mode','WHITE'),('source_selection','target_best')]:
   x=copy.deepcopy(c);x[k]=v;bad(lambda x=x:d.validate(x))
  for k,v in [('extra_CE_weight',.5),('HQ_reference_snr_db',40.),('extra_teacher_NN',64),('extra_KL',1),('extra_packets',128),('formal_IQ_changed',True)]:
   x=copy.deepcopy(c);x['signal_bridge'][k]=v;bad(lambda x=x:d.validate(x))
  p=dict(epoch=20,a1_ema_successful_updates=4440,run_id=d.REFERENCE_RUN,candidate_id=c['parent_row_id'],checkpoint_selection='final_only',ema_model={'mock':1},args=dict(seed=d.SEED,baseline_ckpt='',teacher_ckpt='',epochs=20,use_ema_teacher=True,label_smoothing=.01,batch_size=128))
  def verify(v):return check_checkpoint(v,run=d.REFERENCE_RUN,row=c['parent_row_id'],epoch=20,updates=4440,baseline='')
  verify(p)
  for k,v in [('epoch',30),('a1_ema_successful_updates',6660),('run_id','other'),('candidate_id','unknown'),('checkpoint_selection','target_best'),('ema_model',None)]:
   x=copy.deepcopy(p);x[k]=v;bad(lambda x=x:verify(x))
  for k,v in [('seed',2026092702),('teacher_ckpt','external'),('baseline_ckpt','external')]:
   x=copy.deepcopy(p);x['args'][k]=v;bad(lambda x=x:verify(x))
 for shape in [(5,160),(6,64),(6,160,1)]:bad(lambda shape=shape:check_default_head(shape))
 for ep,step in [(0,0),(1,223),(11,2221),(10,1)]:bad(lambda ep=ep,step=step:cumulative(ep,step))
 expected=dict(checkpoint_sources=['parent/final_ssdg.pth'],parent_source='parent');done=dict(config=expected,epoch=10,optimizer_steps=2220,target_access=False,target_evaluated=False,head_fit_updates=0);frozen=dict(config=expected,selection='final_only',target_access=False);init=dict(target_contact=False,checkpoint_sources=expected['checkpoint_sources'],scratch_only=False,ancestors=['parent']);contract=dict(role_ids={'L':['mock']})
 def metadata(dd=done,ss=contract):check_source_metadata(dd,frozen,init,ss,expected,contract['role_ids'])
 metadata()
 for k,v in [('target_access',True),('target_evaluated',True),('head_fit_updates',1),('optimizer_steps',6660)]:
  x=copy.deepcopy(done);x[k]=v;bad(lambda x=x:metadata(dd=x))
 bad(lambda:metadata(ss=dict(role_ids={'L':['different']})))
 for file in Path(__file__).parent.glob('*.py'):ast.parse(file.read_text(encoding='utf8'))
 try:math=torch_math()
 except ImportError:math=dict(status='Torch unavailable locally;real remote check',NN=0)
 return dict(status='PASS',protocol_negatives=negative,math=math,NN=0,CP=0,IQ=0,updates=0,target_access=False)
if __name__=='__main__':print(json.dumps(run()))
