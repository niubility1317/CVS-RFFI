"""Change only cross-packet carrier-selector dependence, preserving native RNG."""
import ast,copy,inspect,math,textwrap
from types import MethodType
import torch
class CarrierPolicy:
 def __init__(self,owner,mode,seed,namespace):
  if mode not in ('BATCH','PACKET'):raise ValueError('Only batch or packet carrier selectors')
  self.owner=owner;self.mode=mode;self.namespace=namespace;self.seed=int(seed);self.gates=torch.Generator(device='cpu').manual_seed(self.seed);self.parameters={};self.records=[];self.pending=None;self.phase_calls=self.cfo_calls=0
 def private_uniform(self,z,low,high):
  key=str(z.device)
  if key not in self.parameters:self.parameters[key]=torch.Generator(device=z.device).manual_seed(self.seed+1)
  return torch.empty((len(z),1),device=z.device,dtype=z.real.dtype).uniform_(low,high,generator=self.parameters[key])
 def component(self,z,name):
  phase=name=='phase';p=float(self.owner.p_phase_rot if phase else self.owner.p_cfo);maximum=2*math.pi if phase else float(self.owner.cfo_max);B=len(z)
  if not 0<=p<=1 or not math.isfinite(maximum) or maximum<0:raise ValueError('Existing bounded carrier settings required')
  if phase:
   if self.pending is not None:raise ValueError('Previous carrier call incomplete')
   self.phase_calls+=1;self.pending=dict(call=self.phase_calls,mode=self.mode,packets=B)
  else:self.cfo_calls+=1
  if p>0:
   native_gate=bool(torch.rand(())<p);parameters=self.namespace['_rand_uniform']((B,1),z.device,low=0. if phase else -maximum,high=maximum,dtype=z.real.dtype) if native_gate else None
   if self.mode=='BATCH':
    selected=B if native_gate else 0;mask=torch.full((B,1),native_gate,device=z.device,dtype=torch.bool)
   else:
    mask_cpu=torch.rand((B,1),generator=self.gates)<p;selected=int(mask_cpu.sum());mask=mask_cpu.to(z.device)
   if parameters is None and selected>0:parameters=self.private_uniform(z,0. if phase else -maximum,maximum)
   output=z if parameters is None else torch.where(mask,self.namespace['_apply_phase_rot' if phase else '_apply_cfo'](z,parameters),z)
  else:native_gate=False;selected=0;mask=torch.zeros((B,1),device=z.device,dtype=torch.bool);output=z
  if self.pending is None:raise ValueError('Ordered original phase/CFO carrier blocks required')
  self.pending.update({name+'_probability':p,name+'_native_gate':native_gate,name+'_selected':selected,name+'_pairs_selected':selected*max(selected-1,0),name+'_maximum':maximum})
  if not phase:
   if self.cfo_calls!=self.phase_calls:raise ValueError('Exactly one phase/CFO pair per normal L augmentation')
   self.records.append(self.pending);self.pending=None
  return output
 def summary(self):
  if self.pending is not None:raise ValueError('Incomplete carrier selector record')
  result=dict(mode=self.mode,calls=len(self.records),phase_calls=self.phase_calls,cfo_calls=self.cfo_calls,packets=sum(r['packets'] for r in self.records),new_IQ_operator=False,new_parameters=0,new_neural_forwards=0,labels_used=False,U_used=False,range_changed=False,probabilities_changed=False)
  for name in ('phase','cfo'):
   records=self.records;den=sum(r['packets']*(r['packets']-1) for r in records);result[name]=dict(selected=sum(r[name+'_selected'] for r in records),selected_pairs=sum(r[name+'_pairs_selected'] for r in records),pair_denominator=den,selected_count_min=min((r[name+'_selected'] for r in records),default=None),selected_count_max=max((r[name+'_selected'] for r in records),default=None),nontrivial_batches=sum(0<r[name+'_selected']<r['packets'] for r in records))
  return result
 def phase(self,z):return self.component(z,'phase')
 def cfo(self,z):return self.component(z,'cfo')
def install(owner,mode,seed):
 if hasattr(owner,'_carrier_policy'):raise ValueError('One L-only carrier organization per augmentor')
 original=owner._apply_general_channel.__func__;tree=ast.parse(textwrap.dedent(inspect.getsource(original)));function=tree.body[0];seen=[];body=[]
 for statement in function.body:
  attrs={n.attr for n in ast.walk(statement.test) if isinstance(n,ast.Attribute) and isinstance(n.value,ast.Name) and n.value.id=='self'} if isinstance(statement,ast.If) else set();kind='phase' if 'p_phase_rot' in attrs else ('cfo' if 'p_cfo' in attrs else None)
  if kind is None:body.append(statement);continue
  expected='p_phase_rot' if kind=='phase' else 'p_cfo'
  if attrs!={expected} or not isinstance(statement.test,ast.BoolOp) or len(statement.body)!=3:raise ValueError('Consumed native carrier API differs')
  seen.append(kind);replacement=ast.parse('z = self._carrier_policy.'+kind+'(z)').body[0];body.append(ast.copy_location(replacement,statement))
 if seen!=['phase','cfo']:raise ValueError('Exactly original phase then CFO blocks required')
 function.body=body;function.decorator_list=[];ast.fix_missing_locations(tree);namespace=dict(original.__globals__);exec(compile(tree,original.__code__.co_filename,'exec'),namespace);policy=CarrierPolicy(owner,mode,seed,original.__globals__);owner._carrier_policy=policy;owner._apply_general_channel=MethodType(namespace[function.name],owner);return policy
