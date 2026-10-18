"""Focused fixed-scope negatives and actual autograd math,zero NN/CP/IQ."""
import copy,json,ast
from pathlib import Path
from . import design as d
from .protocol import check_checkpoint,check_default_head,cumulative,check_source_metadata

def torch_math():
 import torch
 from torch.nn import functional as F
 from .risk import correction,packet_weights
 state=torch.random.get_rng_state();torch.manual_seed(7001);count=0
 def ok(v):
  nonlocal count
  if not v:raise AssertionError('WithinTX risk math mismatch')
  count+=1
 try:
  a=torch.randn(128,6,dtype=torch.float64,requires_grad=True);b=torch.randn(128,6,dtype=torch.float64,requires_grad=True);y=(torch.arange(128)//2)%6;mask=(torch.arange(128)+1)%2!=0;means=tuple(F.cross_entropy(v,y,label_smoothing=.01) for v in (a,b));ce=[F.cross_entropy(v,y,reduction='none',label_smoothing=.01) for v in (a,b)];score=.5*(ce[0][mask].detach()+ce[1][mask].detach());rng_before=torch.random.get_rng_state().clone()
  wh,gh=packet_weights(score,y[mask],'HARD',1);wr,gr=packet_weights(score,y[mask],'SHUFFLE',1)
  ok(torch.equal(torch.random.get_rng_state(),rng_before));ok(not wh.requires_grad and not wr.requires_grad);ok(float(wh.min())>=.75 and float(wh.max())<=1.25 and abs(float(wh.sum())-64)<1e-12)
  for cls in range(6):
   ii=y[mask]==cls;n=int(ii.sum());ok(abs(float(wh[ii].sum())-n)<1e-12);ok(torch.equal(wh[ii].sort().values,wr[ii].sort().values))
  ok(float((wh*score).mean())>=float(score.mean())-1e-12)
  for mode in ('HARD','SHUFFLE'):
   delta,r=correction(a,b,y,mask,mode,1,old_CE_means=means);w,_=packet_weights(score,y[mask],mode,1);allw=torch.ones(128,dtype=a.dtype);allw[mask]=w;direct=.25*sum((allw*v).mean() for v in ce);via=.25*sum(v.mean() for v in ce)+delta
   ok(torch.allclose(direct,via,atol=1e-12,rtol=1e-12));gd=torch.autograd.grad(direct,(a,b),retain_graph=True);gv=torch.autograd.grad(via,(a,b),retain_graph=True);ok(all(torch.allclose(x,z,atol=1e-12,rtol=1e-12) for x,z in zip(gd,gv)));gdelta=torch.autograd.grad(delta,(a,b),retain_graph=True);ok(all(int(torch.count_nonzero(v[~mask]))==0 for v in gdelta));ok(r['extra_CE_operator_calls']==2 and r['extra_CE_logit_packets']==128 and r['extra_NN']==0)
  singleton=torch.zeros(64);singlelabels=torch.zeros(64,dtype=torch.long);singlelabels[:5]=torch.arange(1,6);ws,_=packet_weights(singleton,singlelabels,'HARD',1);ok(all(float(ws[i])==1 for i in range(5)))
  rejected=0
  for aa,bb,yy,mm,mode,step in [(a[:64],b[:64],y[:64],mask[:64],'HARD',1),(a,b,y.float(),mask,'HARD',1),(a,b,y,mask.float(),'HARD',1),(a,b,y,torch.ones(128,dtype=torch.bool),'HARD',1),(a,b,y,mask,'SCAN',1),(a,b,y,mask,'HARD',0),(a,b,torch.full((128,),6),mask,'HARD',1)]:
   try:correction(aa,bb,yy,mm,mode,step,old_CE_means=means)
   except ValueError:rejected+=1
   else:raise AssertionError('Invalid input accepted')
  return dict(status='PASS',checks=count,protocol_negatives=rejected,loss_and_gradient_exact=True,class_mass_preserved=True,same_histogram_control=True,RNG_preserved=True,NN=0,CP=0,IQ=0,updates=0)
 finally:torch.random.set_rng_state(state)

def run():
 count=0
 def bad(fn):
  nonlocal count
  try:fn()
  except (ValueError,TypeError,KeyError,StopIteration):count+=1
  else:raise AssertionError('Invalid data/budget/method accepted')
 for row in d.rows():
  c=d.config(row);d.validate(c);assert c['data_contract']==dict(RX=[1,3,4,6,8],days=[1,2,3],L=6300,U=56700,V=27000,split=392005,source_RX_holdout=None,internal_L_split=False,U_TX_truth_hidden=True)
  for k,v in [('run_id','other'),('epochs',30),('updates',6660),('parent_epochs',30),('model_seed',2026092702),('parent_source','unknown'),('risk_mode','SCAN'),('As',False),('Ag_mode','WHITE'),('source_selection','target_best')]:
   z=copy.deepcopy(c);z[k]=v;bad(lambda z=z:d.validate(z))
  for k,v in [('gamma',.5),('top_fraction',.25),('extra_NN',64),('KL_unchanged',False),('selected_packets',128)]:
   z=copy.deepcopy(c);z['within_TX_risk'][k]=v;bad(lambda z=z:d.validate(z))
  cp=dict(epoch=20,a1_ema_successful_updates=4440,run_id=d.REFERENCE_RUN,candidate_id='B01-s2026092701',checkpoint_selection='final_only',ema_model={'mock':1},args=dict(seed=d.SEED,baseline_ckpt='',teacher_ckpt='',epochs=20,use_ema_teacher=True,label_smoothing=.01,batch_size=128))
  def verify(v):return check_checkpoint(v,run=d.REFERENCE_RUN,row='B01-s2026092701',epoch=20,updates=4440,baseline='')
  verify(cp)
  for k,v in [('epoch',30),('a1_ema_successful_updates',6660),('run_id','other'),('candidate_id','B00-s2026092701'),('checkpoint_selection','target_best'),('ema_model',None)]:
   z=copy.deepcopy(cp);z[k]=v;bad(lambda z=z:verify(z))
  for k,v in [('seed',2026092702),('teacher_ckpt','external'),('baseline_ckpt','external')]:
   z=copy.deepcopy(cp);z['args'][k]=v;bad(lambda z=z:verify(z))
 check_default_head((6,160))
 for shape in [(5,160),(6,64),(6,160,1)]:bad(lambda shape=shape:check_default_head(shape))
 for ep,step in [(0,0),(1,223),(11,2221),(10,1)]:bad(lambda ep=ep,step=step:cumulative(ep,step))
 expected=dict(checkpoint_sources=['parent/final_ssdg.pth'],parent_source='parent');done=dict(config=expected,epoch=10,optimizer_steps=2220,target_access=False,target_evaluated=False,head_fit_updates=0);frozen=dict(config=expected,selection='final_only',target_access=False);init=dict(target_contact=False,checkpoint_sources=expected['checkpoint_sources'],scratch_only=False,ancestors=['parent']);contract=dict(role_ids={'L':['mock']})
 def metadata(dd=done,ff=frozen,ii=init,ss=contract):check_source_metadata(dd,ff,ii,ss,expected,contract['role_ids'])
 metadata()
 for k,v in [('target_access',True),('target_evaluated',True),('head_fit_updates',1),('optimizer_steps',6660)]:
  z=copy.deepcopy(done);z[k]=v;bad(lambda z=z:metadata(dd=z))
 bad(lambda:metadata(ss=dict(role_ids={'L':['different']})))
 for file in Path(__file__).parent.glob('*.py'):ast.parse(file.read_text(encoding='utf8'))
 try:math=torch_math()
 except ImportError:math=dict(status='TORCH_UNAVAILABLE_LOCAL;remote uses actualTorch',NN=0)
 return dict(status='PASS',protocol_negatives=count,Torch_synthetic_logits=math,NN=0,CP=0,IQ=0,updates=0,target=False)
if __name__=='__main__':print(json.dumps(run()))
