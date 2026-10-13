"""Balanced L-only episodes; private sampler and isolated forward RNG."""
import torch
from . import design as d
from .ridge import episode_loss

class Sampler:
 def __init__(self,L,mode,seed,contract):
  if mode not in ('same_rx','cross_rx') or len(L['ids'])!=6300 or set(L['ids'])!=set(contract['role_ids']['L_s']) or len(set(L['ids']))!=6300:raise ValueError('Episode must use exact L_s physical roles')
  if set(L['ids'])&set(contract['role_ids']['V']) or set(L['ids'])&set(contract['role_ids']['U_s']):raise ValueError('Episode source role overlap')
  self.L=L;self.mode=mode;self.g=torch.Generator().manual_seed(seed+82201);self.pools={};self.episode=0
  for rx in d.SOURCE_RX:
   for cls in range(6):
    pool=((L['rx']==rx)&(L['y']==cls)).nonzero().flatten()
    if len(pool)<6:raise ValueError('Insufficient source class/RX coverage')
    self.pools[rx,cls]=pool
 def draw(self):
  held=d.SOURCE_RX[self.episode%len(d.SOURCE_RX)];a=d.SOURCE_RX[(self.episode+1)%5];b=d.SOURCE_RX[(self.episode+2)%5];support=[];query=[]
  def choose(rx,cls,n):
   pool=self.pools[rx,cls];return pool[torch.randperm(len(pool),generator=self.g)[:n]].tolist()
  for cls in range(6):
   if self.mode=='same_rx':
    chosen=choose(held,cls,6);support.extend(chosen[:2]);query.extend(chosen[2:])
   else:
    support.extend(choose(a,cls,1)+choose(b,cls,1));query.extend(choose(held,cls,4))
  s=torch.tensor(support);q=torch.tensor(query);self.episode+=1
  if set(s.tolist())&set(q.tolist()) or len(s.unique())!=12 or len(q.unique())!=24:raise ValueError('Support/query physical sample overlap')
  if not (self.L['rx'][q]==held).all() or (self.mode=='cross_rx' and (self.L['rx'][s]==held).any()):raise ValueError('Episode RX role mismatch')
  return s,q,held

class Objective:
 def __init__(self,c,L,contract):
  d.validate(c);self.c=c;self.sampler=Sampler(L,c['source_episode']['mode'],c['model_seed'],contract);self.calls=0;self.episodes=0;self.rx={str(r):0 for r in d.SOURCE_RX};self.visited=set()
 def __call__(self,model,step):
  if step!=self.calls+1:raise ValueError('Episode/optimizer clock mismatch')
  self.calls=step;device=next(model.parameters()).device
  metrics=dict(**{'train/episode_active':0.,'train/episode_loss':0.,'train/episode_weighted_loss':0.,'train/episode_support_packets':0.,'train/episode_query_packets':0.,'train/episode_query_rx':-1.})
  if step%d.INTERVAL:return torch.zeros((),device=device),metrics
  s,q,rx=self.sampler.draw();take=torch.cat((s,q));x=self.sampler.L['x'][take].to(device);y=self.sampler.L['y'][take].long().to(device)
  devices=[device.index if device.index is not None else torch.cuda.current_device()] if device.type=='cuda' else []
  with torch.random.fork_rng(devices=devices),torch.autocast(device_type=device.type,enabled=False):
   torch.random.default_generator.manual_seed(self.c['model_seed']+step+97701)
   if devices:torch.cuda.default_generators[devices[0]].manual_seed(self.c['model_seed']+step+97701)
   z=model.id_backbone.encoder.features(x.float());loss=episode_loss(z[:12],y[:12],z[12:],y[12:])
  self.episodes+=1;self.rx[str(rx)]+=1;self.visited.update(self.sampler.L['ids'][i] for i in take.tolist())
  metrics.update({'train/episode_active':1.,'train/episode_loss':loss.detach(),'train/episode_weighted_loss':(d.AUX_WEIGHT*loss).detach(),'train/episode_support_packets':12.,'train/episode_query_packets':24.,'train/episode_query_rx':float(rx)})
  return d.AUX_WEIGHT*loss,metrics
 def finish(self):
  if self.calls!=d.UPDATES or self.episodes!=d.EPISODES or any(n!=d.EPISODES//5 for n in self.rx.values()):raise ValueError('Incomplete episodic feature budget')
  return dict(status='VERIFIED',calls=self.calls,episodes=self.episodes,support_packets=self.episodes*12,query_packets=self.episodes*24,query_RX_episodes=self.rx,unique_L_visited=len(self.visited),U_opened=False,U_truth_access=False,V_training=False,target_access=False,persistent_episode_W=False,gradient_object='joint160 support and query; class weight excluded')
