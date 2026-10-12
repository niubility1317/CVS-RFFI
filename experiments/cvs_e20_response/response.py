"""Frozen response readers, detached real finite targets, live student inputs."""
from pathlib import Path
import torch
from torch import nn
from . import design as d
from .provenance import validate_response
from .model import expected_parent
from experiments.cvs_finite_action import design as fd
from experiments.cvs_finite_action.predictors import Action,inputs
from experiments.cvs_finite_action.splits import make
from experiments.cvs_finite_action.operators import quality
class Readout(nn.Module):
 def __init__(self,s,method,device):
  super().__init__();self.method=method;self.state={k:{n:v.to(device).detach() for n,v in x.items()} for k,x in s['state'].items()}
  for key,stats in self.state.items():
   for name,value in stats.items():self.register_buffer(key+'_'+name,value)
  if method=='MLP':
   self.net=Action(s['net']['layers.0.weight'].shape[1],len(self.state['y']['mean']),128,False).to(device);self.net.load_state_dict(s['net'],strict=True)
  else:self.register_buffer('bilinear',s['bilinear'].to(device).detach())
  self.eval().requires_grad_(False)
 def forward(self,h,q,a):
  x=inputs(h,q,a,self.state,bilinear=self.method=='bilinear_ridge')
  return self.net(x)[0] if self.method=='MLP' else x@self.bilinear
class Bank:
 def __init__(self,c,L,U,device):
  if not c['response']['enabled']:raise ValueError('Disabled response bank cannot load')
  root=Path(c['response']['source_root']);fc=fd.config(next(r for r in fd.rows() if r['model_seed']==c['model_seed']))
  done=fd.read(root/'completion.json');resolved=fd.read(root/'resolved_config.json');matrix=fd.read(root.parent/'completion.json')
  if done['config']!=fc or resolved['config']!=fc or resolved['commit']!=d.FINITE_EXECUTION_COMMIT or done['checkpoint_provenance']!=expected_parent(c) or resolved['checkpoint_provenance']!=expected_parent(c):raise ValueError('Finite source provenance differs')
  if matrix['status']!='SOURCE_DIAGNOSTIC_COMPLETE' or set(matrix['rows'])!={r['row_id'] for r in fd.rows()} or matrix['identity_updates'] or matrix['target_access'] or done['status']!='SOURCE_DIAGNOSTIC_COMPLETE' or done['identity_updates'] or done['head_updates'] or done['target_access'] or not done['budget']['reference_parameters_buffers_modes_gradients_unchanged'] or done['auxiliary_fits']!=117 or done['auxiliary_optimizer_steps']!=117180:raise ValueError('Incomplete/contaminated response fitting')
  observed=torch.load(root/'source_observations.pt',map_location='cpu',weights_only=False)
  source_index={pid:i for i,pid in enumerate(observed['L']['ids'])}
  if len(source_index)!=6300 or set(source_index)!=set(L['ids']):raise ValueError('Finite L role differs')
  lorder=torch.tensor([source_index[pid] for pid in L['ids']],dtype=torch.long)
  expected_split,_=make(L,observed['L']['q'][lorder],fc);fi,ev=expected_split['PID80_20'];fitids=[L['ids'][i] for i in fi.tolist()];evids=[L['ids'][i] for i in ev.tolist()]
  self.observed={};self.orders={}
  for role,data,key in [('L_s',L,'L'),('U_s',U,'U')]:
   if role=='U_s' and any(k in data for k in ('y','label','labels','tx','TX','truth')):raise ValueError('U labels forbidden')
   obs=observed[key]
   if role=='U_s' and any(k in obs for k in ('y','label','labels','tx','TX','truth')):raise ValueError('Cached U labels forbidden')
   if obs['role']!=role or not obs['source_only'] or obs['U_truth'] or len(obs['ids'])!=len(data['ids']) or len(set(obs['ids']))!=len(obs['ids']) or set(obs['ids'])!=set(data['ids']):raise ValueError('Response physical roles differ')
   if obs['q'].requires_grad or not torch.isfinite(obs['q']).all():raise ValueError('Detached finite quality required')
   index={pid:i for i,pid in enumerate(obs['ids'])};order=torch.tensor([index[pid] for pid in data['ids']],dtype=torch.long);self.orders[role]=order
   if not torch.allclose(quality(data['x'][:32]),obs['q'][order[:32]],rtol=1e-5,atol=1e-6):raise ValueError('Known IQ quality/source cache differs')
   # Only selected L/T targets survive; no R means/variance enter identity learning.
   self.observed[role]=dict(q=obs['q'],targets={k:dict(points=obs['targets'][k]['points'],mean={level:obs['targets'][k]['mean'][level] for level in ('time_down','t2.pool')}) for k in ('spectral','timing')})
  self.readers={};paths=[]
  for kind,level,method in c['response']['tasks']:
   path=root/f"PID80_20-{kind}-{level.replace('.','_')}.pt";s=torch.load(path,map_location='cpu',weights_only=False)
   if s['config']!=fc or s['split']!='PID80_20' or s['kind']!=kind or s['level']!=level or s['noise'] or set(s['fit_IDs'])!=set(fitids) or len(s['fit_IDs'])!=5040 or set(s['eval_IDs'])!=set(evids) or len(s['eval_IDs'])!=1260 or set(fitids)&set(evids):raise ValueError('Wrong finite readout/fit partition')
   reader=Readout(s,method,device)
   if any(p.requires_grad for p in reader.parameters()) or any(not torch.isfinite(t).all() for t in reader.state_dict().values()) or any((s['state'][k]['scale']<1e-4).any() for k in s['state']):raise ValueError('Finite frozen response parameters/statistics required')
   self.readers[kind,level]=reader;paths.append(str(path))
  self.proof=dict(status='VERIFIED',run_id=fd.RUN,execution_commit=d.FINITE_EXECUTION_COMMIT,seed=c['model_seed'],split='PID80_20',source_only=True,identity_updates=0,target_access=False,reference_parent=expected_parent(c),tasks=c['response']['tasks'],source_root=str(root),state_paths=paths,observations_path=str(root/'source_observations.pt'),auxiliary_trainable_parameters=0)
  validate_response(c,self.proof);self.config=c
 def loss(self,features,role,indices,device):
  if role not in self.config['response']['roles'] or set(features)!=set(('time_down','t2.pool')):raise ValueError('Unexpected response role/features')
  obs=self.observed[role];oi=self.orders[role][indices];q=obs['q'][oi].to(device);terms={};details={}
  for kind,level,method in self.config['response']['tasks']:
   reader=self.readers[kind,level];a=torch.tensor(self.config['response']['known_parameters'][kind],device=device).expand(len(indices),-1)
   column=obs['targets'][kind]['points'].index(1);target=obs['targets'][kind]['mean'][level][oi,column].to(device)
   if target.requires_grad or any(p.requires_grad for p in reader.parameters()):raise ValueError('Response target/reader must stay frozen')
   target=(target-reader.state['y']['mean'])/reader.state['y']['scale'];prediction=reader(features[level],q,a);value=(prediction-target).square().mean()
   if not value.requires_grad or not torch.isfinite(value):raise ValueError('Detached/nonfinite identity response objective')
   key=kind+':'+level;terms[key]=value;details[key]=dict(loss=float(value.detach()),method=method,count=len(indices),input_gradient=True,target_gradient=False,known_a=self.config['response']['known_parameters'][kind])
  return torch.stack(list(terms.values())).mean(),details
