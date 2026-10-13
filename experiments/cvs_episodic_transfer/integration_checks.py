"""Focused producer checks on the actual backbone and source loss hook."""
from copy import deepcopy
from pathlib import Path
import torch
from . import design as d
from .model import build
from .episode import Sampler,Objective

def fake_L():
 rx=torch.tensor(d.SOURCE_RX).repeat_interleave(1260);y=torch.arange(6).repeat_interleave(210).repeat(5);ids=[f'fixture-{i}' for i in range(6300)]
 g=torch.Generator().manual_seed(13)
 L=dict(x=torch.randn(6300,2,256,generator=g),y=y,rx=rx,ids=ids)
 return L,dict(role_ids=dict(L_s=ids,U_s=['not-L-U'],V=['not-L-V']))

def checks():
 torch.set_num_threads(2);L,contract=fake_L();report={};negative=0
 for mode in ('same_rx','cross_rx'):
  sampler=Sampler(L,mode,2026092701,contract);rx_counts={r:0 for r in d.SOURCE_RX};first=[]
  for index in range(d.EPISODES):
   s,q,rx=sampler.draw();rx_counts[rx]+=1
   assert torch.equal(torch.bincount(L['y'][s],minlength=6),torch.full((6,),2)) and torch.equal(torch.bincount(L['y'][q],minlength=6),torch.full((6,),4))
   if mode=='same_rx':assert (L['rx'][s]==rx).all()
   else:assert (L['rx'][s]!=rx).all() and len(L['rx'][s].unique())==2
   if index<2:first.append((s.tolist(),q.tolist()))
  replay=Sampler(L,mode,2026092701,contract)
  for expected in first:
   s,q,_=replay.draw();assert (s.tolist(),q.tolist())==expected
  assert set(rx_counts.values())=={555}
  c=d.config(next(r for r in d.training_rows() if r['arm']==('EPISODE_SAME_RX' if mode=='same_rx' else 'EPISODE_CROSS_RX')))
  torch.manual_seed(c['model_seed']);model=build(c).train();obj=Objective(c,L,contract)
  for step in (1,2,3):
   loss,metrics=obj(model,step);assert float(loss)==0 and metrics['train/episode_active']==0
  rng=torch.get_rng_state().clone();loss,metrics=obj(model,4);assert torch.equal(rng,torch.get_rng_state()) and metrics['train/episode_active']==1
  W=model.id_backbone.encoder.core.id_backbone.cls_head.weight
  assert torch.autograd.grad(loss,W,retain_graph=True,allow_unused=True)[0] is None
  loss.backward();gradient=sum(float(p.grad.square().sum()) for p in model.id_backbone.parameters() if p.grad is not None)
  assert gradient>0 and all(torch.isfinite(p.grad).all() for p in model.id_backbone.parameters() if p.grad is not None)
  assert obj.calls==4 and obj.episodes==1
  try:obj.finish()
  except ValueError:negative+=1
  else:raise AssertionError('Incomplete auxiliary budget accepted')
  try:obj(model,6)
  except ValueError:negative+=1
  else:raise AssertionError('Wrong update clock accepted')
  report[mode]=dict(episode_loss=float(loss),backbone_gradient_l2=gradient**.5,old_class_W_gradient=None,episodes=d.EPISODES,RX_episodes=rx_counts,main_rng_unchanged=True)
 native=Path(d.ROOT)/'experiments/adv3b02_xuc/code/SSDG/train_ssdg.py';text=native.read_text(encoding='utf-8')
 start=text.index('            feature_auxiliary_metrics = {}');end=text.index('            loss_is_finite =',start);block='\n'.join(line[12:] for line in text[start:end].splitlines())
 a=torch.tensor(2.,requires_grad=True);b=torch.tensor(3.,requires_grad=True)
 env=dict(loss=a*a,scaled_closed_loss=a*a,model=None,r3_optimizer_steps=3,_source_feature_auxiliary_hook=lambda model,step:(b*b,dict(actual_step=step)))
 exec(compile(block,str(native),'exec'),env);assert env['feature_auxiliary_metrics']['actual_step']==4 and env['loss'].item()==13 and env['scaled_closed_loss'].item()==13
 env['loss'].backward();assert a.grad.item()==4 and b.grad.item()==6
 a=torch.tensor(2.,requires_grad=True);env=dict(loss=a*a,scaled_closed_loss=a*a,_source_feature_auxiliary_hook=None);exec(compile(block,str(native),'exec'),env);assert env['loss'].item()==4 and env['feature_auxiliary_metrics']=={}
 for change in ('weight','interval','L_only','V_training','main_rng_preserved'):
  c=d.config(d.training_rows()[0]);c['source_episode'][change]=not c['source_episode'][change] if isinstance(c['source_episode'][change],bool) else 99
  try:d.validate(c)
  except ValueError:negative+=1
  else:raise AssertionError('Changed episode contract accepted')
 c=d.config(d.rows()[0])
 try:d.make_args(c)
 except ValueError:negative+=1
 else:raise AssertionError('Baseline retraining accepted')
 bad=deepcopy(contract);bad['role_ids']['L_s']=bad['role_ids']['L_s'][:-1]
 try:Sampler(L,'same_rx',1,bad)
 except ValueError:negative+=1
 else:raise AssertionError('Wrong source roles accepted')
 return dict(status='PASS',report=report,negative_cases=negative,native_total_loss_hook_verified=True,no_data_or_query_read=True)
if __name__=='__main__':
 import json
 print(json.dumps(checks()))
